#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>


#include <cstdio>
#include <iomanip>
#include "Options.h"
#include "Logger.h"
#include <TMath.h>
#include <TFile.h>
#include <TRandom3.h>
#include <RooFit.h>
#include <RooRealVar.h>
#include <RooGenericPdf.h>
#include <RooArgList.h>
#include <TVirtualFFT.h>
#include <rapidcsv.h>
#include <TH1.h>
#define TIME_DISCRETIZATION 8
#define TQDC_MIN_STEP         4.0  // min step in bits along voltage axis
// for scale representation only
#define TQDC_BITS_TO_MV        (1/TQDC_MIN_STEP)*2*1000.0/16384.0 // 2^14 bits/2V - 4 counts ~0.030517578125
#define TQDC_BITS_TO_PC        TQDC_BITS_TO_MV*0.02*TIME_DISCRETIZATION //  50 Ohm ~0.0048828125
#define N 2010
#define numHistograms 16
#define MINIMUM_DIFF 0.3
typedef struct {
    size_t ev;			// event number
    size_t po;			// event header position
} FILEINDEX;

double computeAverage(const std::vector<double>& data, 
                      size_t startIndex, 
                      size_t endIndex) {
    // Ensure the input is valid
    if (startIndex > endIndex || endIndex > data.size()) {
        throw std::out_of_range("Invalid start or end index");
    }

    double sum = std::accumulate(data.begin() + startIndex, data.begin() + endIndex, 0.0);
    double average = sum / static_cast<double>(endIndex - startIndex);

    return average;
}

double computeStdDev(const std::vector<double>& values) {
    double mean = computeAverage(values, 0, values.size());
    double variance = 0;
    for (double val : values) {
        variance += (val - mean) * (val - mean);
    }
    variance /= values.size();
    return std::sqrt(variance);
}

bool fileExists(const std::string& filename) {
    struct stat buf;
    return (stat(filename.c_str(), &buf) == 0);
}


std::vector<double> findNegativeMaxAfterThreshold(const std::vector<double>& data, 
                                                  size_t startIndex, 
                                                  size_t endIndex, 
                                                  double threshold) {
    std::vector<double> maxValues;

    // Ensure the input is valid
    if (startIndex > endIndex || endIndex > data.size()) {
        throw std::out_of_range("Invalid start or end index");
    }

    size_t i = startIndex;
    while (i < endIndex) {
        if (data[i] < threshold) {
            // Check next 44 points, or until the end
            size_t limit = i + 44;
            auto minIt = std::min_element(data.begin() + i, data.begin() + limit);
            double minVal = *minIt;

            maxValues.push_back(minVal);
            
            // Set the next index to start checking from
            i = i + 45;
        } else {
            ++i;
        }
    }

    return maxValues;
}
std::vector<double> findMaxAfterThreshold(const std::vector<double>& data, 
                                          size_t startIndex, 
                                          size_t endIndex, 
                                          double threshold) {
    std::vector<double> maxValues;

    // Ensure the input is valid
    if (startIndex > endIndex || endIndex > data.size()) {
        throw std::out_of_range("Invalid start or end index");
    }

    size_t i = startIndex;
    while (i < endIndex) {
        if (data[i] > threshold) {
            // Check next 44 points, or until the end
            size_t limit = i + 44;
            auto maxIt = std::max_element(data.begin() + i, data.begin() + limit);
            double maxVal = *maxIt;

            maxValues.push_back(maxVal);
            
            // Set the next index to start checking from
            i = i + 45;
        } else {
            ++i;
        }
    }

    return maxValues;
}

double butterworthLowpassFilter(double input, double cutoffFreq, int order) {
    
    double output = 1 / sqrt(1 + pow((input / cutoffFreq), 2 * order)) ;
    return output;
}


std::vector<double> avg_charge_waveform(std::vector<double> &waveform) {
    std::vector<double> charge_waveform;
    for (int i = 0; i < waveform.size() - 45; ++i) {
        double dcrQ = 0;
        for (int j = 0; j < 45; ++j) {
            dcrQ += waveform.at(i + j);
        }
        charge_waveform.push_back(dcrQ / 45);
    }
    return charge_waveform;
}

std::vector<double> alighed_waveform(std::vector<double> &waveform) {
    std::vector<double> charge_waveform;
    for (int i = 45; i < waveform.size(); ++i) {
        charge_waveform.push_back(waveform.at(i));
    }
    return charge_waveform;
}

int main(int argc, char **argv) {
    Options options(argc, argv);
    YAML::Node const config = options.GetConfig();
    std::string outputName = "output.txt";
    float cutOffFreq = Options::NodeAs<float>(config, {"Butterworth_cutoff_frequency"});
    unsigned int filterOrder = Options::NodeAs<unsigned int>(config, {"Butterworth_order"});
    if (options.Exists("output")) outputName = options.GetAs<std::string>("output");
    int runNumber = 0;
    if (options.Exists("run"))  runNumber = options.GetAs<int>("run");
    int maxEvents = 9999999;
    if (options.Exists("maxEvents"))  maxEvents = options.GetAs<int>("maxEvents");
    int skipEvents = 0;
    if (options.Exists("skipEvents"))  skipEvents = options.GetAs<int>("skipEvents");
    int ov = 0;
    if (options.Exists("voltage"))  ov = options.GetAs<int>("voltage");
    std::string suffix = "charge_fit_tile_ch16_ov1.csv";
    if (options.Exists("type"))  suffix = options.GetAs<std::string>("type");
    
    rapidcsv::Document doc(suffix);
    std::vector<int> pos_vec = doc.GetColumn<int>("pos");
    std::vector<int> run_vec = doc.GetColumn<int>("run");
    std::vector<int> ch_vec = doc.GetColumn<int>("ch");
    std::vector<int> vol_vec = doc.GetColumn<int>("vol");
    std::vector<float> gain_vec = doc.GetColumn<float>("gain");
    std::vector<float> gain_err_vec = doc.GetColumn<float>("gain_err");
    std::vector<float> lambda_vec = doc.GetColumn<float>("lambda");
    std::vector<float> lambda_err_vec = doc.GetColumn<float>("lambda_err");
    std::vector<float> bl_vec = doc.GetColumn<float>("bl");
    std::vector<float> blrms_vec = doc.GetColumn<float>("bl_rms");
    std::vector<float> sigma0_vec = doc.GetColumn<float>("sigma0");
    std::vector<float> sigma1_vec = doc.GetColumn<float>("sigma1");
    std::vector<float> sigma2_vec = doc.GetColumn<float>("sigma2");
		std::vector<float> sigma3_vec = doc.GetColumn<float>("sigma3");
    std::vector<float> sigma4_vec = doc.GetColumn<float>("sigma4");
    // Ensure pos size is 16
    if (pos_vec.size() != 16) {
        std::cerr << "The pos vector should have 16 elements!";
        return 1;
    }

    // Create a vector of indices
    std::vector<size_t> idx(pos_vec.size());
    std::iota(idx.begin(), idx.end(), 0);

    // Sort the indices based on the pos values
    std::sort(idx.begin(), idx.end(), [&pos_vec](size_t i1, size_t i2) { return pos_vec[i1] < pos_vec[i2]; });

    // Rearrange function using sorted indices
    auto rearrange = [&idx](auto& vec) {
        //decltype(vec) temp(vec.size(), 0.);
        using ValueType = typename std::remove_reference<decltype(vec)>::type::value_type;
        std::vector<ValueType> temp(vec.size(), 0.);
        for (size_t i = 0; i < idx.size(); ++i) {
            temp[i] = vec[idx[i]];
        }
        vec = std::move(temp);
    };

    // Rearrange all the vectors based on the idx
    rearrange(run_vec);
    rearrange(ch_vec);
    rearrange(gain_vec);
    rearrange(gain_err_vec);
    rearrange(vol_vec);
    rearrange(lambda_vec);
    rearrange(lambda_err_vec);
    rearrange(bl_vec);
    rearrange(blrms_vec);
    rearrange(sigma0_vec);
    rearrange(sigma1_vec);
    rearrange(sigma2_vec);
    rearrange(sigma3_vec);
    rearrange(sigma4_vec);

    
    //input file name
    std::string filename;
    if  (options.Exists("input")) {
       filename = options.GetAs<std::string>("input");
    }
    else {
       LOG_ERROR << "-i -with inputFilePath is needed!";
       std::abort();
    }
    
    TH1F* dcr[numHistograms];
    TH1F* dcr_n[numHistograms];
    TH1F* dcr_bl_up[numHistograms];
    TH1F* dcr_n_bl_up[numHistograms];
    TH1F* dcr_bl_down[numHistograms];
    TH1F* dcr_n_bl_down[numHistograms];
    for (int i = 0; i < numHistograms; ++i) {
       float gap = gain_vec[i] / 45;
       if (gap == 0) gap = 8./45.;
       dcr[i] = new TH1F(Form("dcrQ_ch%d", i), "DCR", 800 , 0, 4 * gap + 5 * sigma3_vec[i] / 45);
       dcr_bl_up[i] = new TH1F(Form("dcrQ_bl_up_ch%d", i), "DCR", 800 , 0, 4 * gap + 5 * sigma3_vec[i] / 45);
       dcr_bl_down[i] = new TH1F(Form("dcrQ_bl_down_ch%d", i), "DCR", 800 , 0, 4 * gap + 5 * sigma3_vec[i] / 45);
       dcr_n[i] = new TH1F(Form("dcrQ_neg_ch%d", i), "DCR", 800 , 0, 4 * gap + 5 * sigma3_vec[i] / 45);
       dcr_n_bl_up[i] = new TH1F(Form("dcrQ_bl_up_neg_ch%d", i), "DCR", 800 , 0, 4 * gap + 5 * sigma3_vec[i] / 45);
       dcr_n_bl_down[i] = new TH1F(Form("dcrQ_neg_bl_down_ch%d", i), "DCR", 800 , 0, 4 * gap + 5 * sigma3_vec[i] / 45);
    }
    FILE *fp;
    std::uint32_t uiEVENT_HEADER[3] = {0,0,0};
    // Determine the actual number of words in the file

    // Open the binary file for reading
    fp = fopen(filename.c_str(), "rb");
    if (fp == NULL) {
       printf("Error opening file.\n");
       return 1;
    }

    fseek(fp,0,SEEK_END);
	  size_t sSizeOfFile = ftell(fp);
    
    // Set the file pointer to the correct byte offset
    fseek(fp, 0 , SEEK_SET);
    int num_events = 0;
    std::uint32_t buffer;
    // Read the binary data into the array
    FILEINDEX FileIndex;
    FileIndex.ev = 0;
    FileIndex.po = 0;
    std::vector<FILEINDEX> vFInd;

    auto locatingStartTime = std::chrono::steady_clock::now();
    while (!feof(fp)) {
       fread(&buffer, sizeof(std::uint32_t), 1 , fp);
       if (buffer == 0x2A502A50) {
          fseek(fp,-sizeof(uint32_t),SEEK_CUR);
          fread(uiEVENT_HEADER,sizeof(uiEVENT_HEADER),1,fp);
          FileIndex.ev = num_events;   // absolute event number by data file
          FileIndex.po = ftell(fp)-3*sizeof(uint32_t);
          num_events++;
          //std::cout << "locating event:" << num_events << std::endl;
          vFInd.push_back(FileIndex);
          fseek(fp,FileIndex.po+uiEVENT_HEADER[1],SEEK_SET);		// from zero position - jump over all event length
          if (FileIndex.po+uiEVENT_HEADER[1] > sSizeOfFile) break;
       }
    }
    int events = vFInd.size();
    TH1F *wf_hist = new TH1F("wave", "wave", 2010, 0, 2010);
    double lowpass[N];
    for (int i = 0 ; i < N; i++) {
       //std::cout << i << "\t: "<< butterworthLowpassFilter(i*1.0,200, 5) << std::endl;
       lowpass[i] = butterworthLowpassFilter(i*1.0, cutOffFreq, filterOrder);
    }
    for (int nev = 0; nev < std::min(maxEvents, events); ++nev) {
        int offset = 0;
        std::uint32_t word;
        fseek(fp,vFInd.at(nev).po + sizeof(word) *  offset ,SEEK_SET);
        fread(&word,sizeof(word),1,fp);
        //std::cout << offset << "\t" << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
        int pass = 13;
        fseek(fp, sizeof(word) *  pass ,SEEK_CUR);
        offset += pass;
        for (int ch = 0; ch < 16 ; ch ++) {
           //if (ch != 0) continue;
           fread(&word,sizeof(word),1,fp);
           //std::cout << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
           int chsize =  word & 0x0000FFFF;
           fseek(fp, sizeof(word) ,SEEK_CUR);
           std::int16_t vol;
           int BS = chsize / 2 - 2;
           int count = 0;
           std::vector<double> waveform;
           std::vector<double> waveform_filtered;
           while (BS != 0) {
              fread(&vol, sizeof(vol), 1 ,fp);
              float time = count * 8.;
              float amp = 0;
              amp = vol * TQDC_BITS_TO_PC;
              waveform.push_back(amp);
              wf_hist->SetBinContent(count + 1, amp);
              //std::cout<< amp<< std::endl;
              BS --;
              count ++;
           }

           //Compute the transform and look at the magnitude of the output
           TH1 *hm =0;
           TVirtualFFT::SetTransform(0);
           hm = wf_hist->FFT(hm, "MAG R2C M"); 
           //That's the way to get the current transform object:
           TVirtualFFT *fft = TVirtualFFT::GetCurrentTransform();
           //Use the following method to get the full output:
           Double_t *re_full = new Double_t[N];
           Double_t *im_full = new Double_t[N];
      
           fft->GetPointsComplex(re_full,im_full);
           for (int i = 0 ; i < N; i++) {
              //std::cout << i << "\t: "<< butterworthLowpassFilter(i*1.0,200, 5) << std::endl;
              re_full[i] *= lowpass[i];
              im_full[i] *= lowpass[i];
           }
           int numberFreq = N;
           //Now let's make a backward transform:
           TVirtualFFT *fft_back = TVirtualFFT::FFT(1, &numberFreq, "C2R M K");
           fft_back->SetPointsComplex(re_full,im_full);
           fft_back->Transform();
           TH1 *hb = 0;
           //Let's look at the output
           hb = TH1::TransformHisto(fft_back,hb,"Re");
           for (int i = 0; i < N; i++) {
              float amp_backward = hb->GetBinContent(i + 1) / N;
              waveform_filtered.push_back(amp_backward);
              //std::cout << amp_backward << std::endl;
           }
           //auto charge_wf = avg_charge_waveform(waveform_filtered);
           //auto aligned_wf = alighed_waveform(waveform_filtered);
           auto charge_wf = avg_charge_waveform(waveform);
           auto aligned_wf = alighed_waveform(waveform);
           //for (auto wf: charge_wf) {
           //    std::cout << wf << " ";
           //}
           auto baseline = computeAverage(waveform, 300, 900);
           //auto baseline_up = baseline + blrms_vec[ch];
           //auto baseline_down = baseline - blrms_vec[ch];

           double baseline_up = computeAverage(waveform, 500, 1100);
           double baseline_down = computeAverage(waveform, 100, 700);
           //std::cout << baseline << "\t" << bl_vec[ch] << std::endl;
           auto dcr_charges = findMaxAfterThreshold(charge_wf, 0, 1155, baseline + gain_vec[ch] / 45); 
           auto dcr_charges_bl_up = findMaxAfterThreshold(charge_wf, 0, 1155, baseline_up + gain_vec[ch] / 45); 
           auto dcr_charges_bl_down = findMaxAfterThreshold(charge_wf, 0, 1155, baseline_down + gain_vec[ch] / 45); 
           auto dcr_negative_charges = findNegativeMaxAfterThreshold(charge_wf, 0, 1155, baseline - gain_vec[ch] / 45); 
           auto dcr_negative_charges_bl_up = findNegativeMaxAfterThreshold(charge_wf, 0, 1155, baseline_up - gain_vec[ch] / 45); 
           auto dcr_negative_charges_bl_down = findNegativeMaxAfterThreshold(charge_wf, 0, 1155, baseline_down - gain_vec[ch] / 45); 
           for (auto dcrQ: dcr_charges) {
              dcr[ch]->Fill(dcrQ - baseline);
           }
           for (auto dcrQ: dcr_negative_charges) {
              dcr_n[ch]->Fill(baseline - dcrQ);
           }
           for (auto dcrQ: dcr_charges_bl_up) {
              dcr_bl_up[ch]->Fill(dcrQ - baseline_up);
           }
           for (auto dcrQ: dcr_negative_charges_bl_up) {
              dcr_n_bl_up[ch]->Fill(baseline_up - dcrQ);
           }
           for (auto dcrQ: dcr_charges_bl_down) {
              dcr_bl_down[ch]->Fill(dcrQ - baseline_down);
           }
           for (auto dcrQ: dcr_negative_charges_bl_down) {
              dcr_n_bl_down[ch]->Fill(baseline_down - dcrQ);
           }
           //for (auto wf: aligned_wf) {
           //    std::cout << wf << " ";
           //}
           //std::cout << std::endl;
           delete hm;
           delete hb;
           delete [] re_full;
           delete [] im_full;
           delete fft_back;
           delete fft;
        }
    }
    //TFile *f_out = new TFile("testDCR.root", "recreate");
    TString formula = "(mu * TMath::Power((mu + 1 * lambda), 1-1) * TMath::Exp(-(mu + 1 * lambda)) / 1) * (1/sigma1 * TMath::Exp(- TMath::Power(sigQ - (ped + 1 * gain), 2)/(2 * TMath::Power(sigma1, 2))))+ (mu * TMath::Power((mu + 2 * lambda), 2-1) * TMath::Exp(-(mu + 2 * lambda)) / 2) * (1/sigma2 * TMath::Exp(- TMath::Power(sigQ - (ped + 2 * gain), 2)/(2 * TMath::Power(sigma2, 2))))+ (mu * TMath::Power((mu + 3 * lambda), 3-1) * TMath::Exp(-(mu + 3 * lambda)) / 6) * (1/sigma3 * TMath::Exp(- TMath::Power(sigQ - (ped + 3 * gain), 2)/(2 * TMath::Power(sigma3, 2))))+ (mu * TMath::Power((mu + 4 * lambda), 4-1) * TMath::Exp(-(mu + 4 * lambda)) / 24) * (1/sigma4 * TMath::Exp(- TMath::Power(sigQ - (ped + 4 * gain), 2)/(2 * TMath::Power(sigma4, 2))))"; 
    
    std::string csv_path = outputName;//+ "/run" + std::to_string(run_vec[0]) + "_ch"+ std::to_string(ch_vec[0]) + ".csv";
    //std::string csv_path = "output_test.csv";
    // Check if file exists and its size is greater than zero
    bool writeHeader = !(fileExists(csv_path) && std::ifstream(csv_path).peek() != std::ifstream::traits_type::eof());

    std::ofstream csvFile(csv_path, std::ios::app);
    if(csvFile.is_open()){
        // Write the header
        if (writeHeader) {
            csvFile << "run,ch,pos,vol,dcr,dcr_err";
            csvFile << "\n";
        }
    
        for (int i = 0; i < numHistograms; ++i) {
            //dcr[i]->Write();// = new TH1F(Form("dcrQ_ch%d", i), "DCR", 80 , 82.,  90.);
            //dcr_n[i]->Write();// = new TH1F(Form("dcrQ_ch%d", i), "DCR", 80 , 82.,  90.);
            //dcr_bl_up[i]->Write();// = new TH1F(Form("dcrQ_ch%d", i), "DCR", 80 , 82.,  90.);
            //dcr_n_bl_up[i]->Write();// = new TH1F(Form("dcrQ_ch%d", i), "DCR", 80 , 82.,  90.);
            //dcr_bl_down[i]->Write();// = new TH1F(Form("dcrQ_ch%d", i), "DCR", 80 , 82.,  90.);
            //dcr_n_bl_down[i]->Write();// = new TH1F(Form("dcrQ_ch%d", i), "DCR", 80 , 82.,  90.);
            auto sigQ = RooRealVar("sigQ", "sigQ", 0, gain_vec[i] * 4 + 5 * sigma4_vec[i]);
            sigQ.setRange("integrationRange", gain_vec[i], gain_vec[i] * 4 + 5 * sigma4_vec[i]);
            auto lambda_ = RooRealVar("lambda", "lambda", lambda_vec[i]);
            auto mu = RooRealVar("mu", "mu", 1);
            auto ped = RooRealVar("ped", "ped", 0);
            auto gain = RooRealVar("gain", "gain", gain_vec[i]);
            auto sigma1 = RooRealVar("sigma1", "sigma1", sigma1_vec[i]);
            auto sigma2 = RooRealVar("sigma2", "sigma2", sigma2_vec[i]);
            auto sigma3 = RooRealVar("sigma3", "sigma3", sigma3_vec[i]);
            auto sigma4 = RooRealVar("sigma4", "sigma4", sigma4_vec[i]);
            auto argList = RooArgList(lambda_, mu, sigQ, ped, gain, sigma1, sigma2, sigma3, sigma4);
            RooGenericPdf GP("genPdf", formula, argList);
            double integral_value_central = GP.createIntegral(sigQ, RooFit::NormSet(sigQ), RooFit::Range("integrationRange"))->getVal();

            const int n_samples = 1000;

            // Given means and standard deviations for lambda and gain
            double lambda_mean = lambda_vec[i];
            double lambda_std = lambda_err_vec[i];  // Adjust accordingly
            if abs(lambda_std * 10 > lambda_mean) lambda_std = lambda_mean * 0.1; //maximum 10% uncertainty
            double gain_mean = gain_vec[i]; // Adjust accordingly
            double gain_std = gain_err_vec[i];  // Adjust accordingly
            if abs(gain_std * 10 > gain_mean) gain_std == gain_mean * 0.1; //maximum 10% uncertainty

            TRandom3 randGen;
            std::vector<double> bootstrap_integral_values;

            for (int i = 0; i < n_samples; ++i) {
                // Resample lambda and gain
                double lambda_resampled = randGen.Gaus(lambda_mean, lambda_std);
                double gain_resampled = randGen.Gaus(gain_mean, gain_std);
                
                // Set the new values to the RooRealVars
                lambda_.setVal(lambda_resampled);
                gain.setVal(gain_resampled);

                // Compute the integral
                double integral_value = GP.createIntegral(sigQ, RooFit::NormSet(sigQ), RooFit::Range("integrationRange"))->getVal();
                bootstrap_integral_values.push_back(integral_value);
            }
            auto cumulativeProbability = integral_value_central;//computeAverage(bootstrap_integral_values, 0 , bootstrap_integral_values.size());//GP.createIntegral(sigQ, RooFit::NormSet(sigQ), RooFit::Range("integrationRange"));
            auto cumulativeProbability_std = computeStdDev(bootstrap_integral_values);
            auto norm_factor = 1 / cumulativeProbability;
            auto norm_factor_err = 1 / (cumulativeProbability - cumulativeProbability_std) - 1 /( cumulativeProbability + cumulativeProbability_std ); 
            norm_factor_err /= 2.;
            auto rel_norm_factor_err = norm_factor_err / norm_factor;
            auto dcr_events = dcr[i]->Integral() - dcr_n[i]->Integral();
            auto dcr_events_err_stat = TMath::Sqrt(dcr[i]->Integral() + dcr_n[i]->Integral());
            
            auto dcr_events_up = dcr_bl_up[i]->Integral() - dcr_n_bl_up[i]->Integral();
            auto dcr_events_down = dcr_bl_down[i]->Integral() - dcr_n_bl_down[i]->Integral();
            auto dcr_events_err_bl_up = dcr_events_up - dcr_events;
            auto dcr_events_err_bl_down = dcr_events_down - dcr_events;
            auto dcr_events_err = TMath::Sqrt(TMath::Power(dcr_events_err_stat, 2) + TMath::Power(dcr_events_err_bl_up,2) + TMath::Power(dcr_events_err_bl_down,2));
            if (vol_vec[i] > 2) {
                dcr_events =  dcr[i]->Integral();
                dcr_events_err = TMath::Sqrt(dcr[i]->Integral());
            }
            auto rate = dcr_events / 144. / (1155 * 8e-9 * events) / cumulativeProbability;
            auto rate_err = dcr_events_err / 144. / (1155 * 8e-9 * events) / cumulativeProbability;
            auto rate_err_stat = dcr_events_err_stat / 144. / (1155 * 8e-9 * events) / cumulativeProbability;
            auto rate_err_bl_up = dcr_events_err_bl_up / 144. / (1155 * 8e-9 * events) / cumulativeProbability;
            auto rate_err_bl_down = dcr_events_err_bl_down / 144. / (1155 * 8e-9 * events) / cumulativeProbability;
            auto rate_err_all = rate * TMath::Sqrt(TMath::Power(rate_err/ rate, 2) + TMath::Power(rel_norm_factor_err, 2));
            if (rate < 0) {
                rate = 0;
                rate_err_all = fabs(rate_err_all);
            }
            std::cout << "events" <<  events << std::endl;
            std::cout << "CMF" <<  cumulativeProbability << std::endl;
            std::cout << rate << "+-" << rate_err_stat << "\t" << "Hz/mm for pos" << i <<std::endl;
            std::cout << "bl up" << rate_err_bl_up << "\t" << "Hz/mm for pos" << i <<std::endl;
            std::cout << "bl down" << rate_err_bl_down << "\t" << "Hz/mm for pos" << i <<std::endl;

            std::cout << rate << "+-" << rate_err << "\t" << "Hz/mm for pos" << i <<std::endl;
            std::cout << rate << "+-" << rate_err_all << "\t" << "Hz/mm for pos" << i <<std::endl;
            // Write the values
            csvFile << run_vec[i] << ","
                    << ch_vec[i] << ","
                    << pos_vec[i] << ","
                    << vol_vec[i] << ","
                    << rate << ","
                    << rate_err_all
                    << "\n";
        }
        csvFile.close();
    }

    //f_out->Close();
    return 0;
}

