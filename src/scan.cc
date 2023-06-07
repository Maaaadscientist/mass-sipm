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

#include <cstdio>
#include <iomanip>
#include <TFile.h>

#include <TVirtualFFT.h>

#include <TH1.h>
#include <TH2F.h>
#include <TTree.h>


#include "Options.h"
#include "Logger.h"

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

double butterworthLowpassFilter(double input, double cutoffFreq, int order) {
    
    double output = 1 / sqrt(1 + pow((input / cutoffFreq), 2 * order)) ;
    return output;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>, std::vector<double>> getDCR( const std::vector<float> &SiPMWave, float baseline, int signalStart, float thres) {
    std::vector<int> po;
    std::vector<int> last;
    std::vector<double> amp;
    std::vector<double> charge;
    std::vector<double> tmp;
    for (int i = 0; i < signalStart - 200; i++) {
        if (SiPMWave[i] - thres >= baseline) {
            po.push_back(i);
            tmp.push_back(SiPMWave[i]);
            int start = i;
            for (int k = i; k >= 0; --k) {
                //std::cout<< "tracing before:" <<k << std::endl;
                if (abs(SiPMWave[k] -  baseline) <= MINIMUM_DIFF) {
                    start = k;
                    break;
                }
            }
            for (int j = i + 1; j < signalStart - 100; j++) {
                //std::cout<< "tracing after:" <<j << std::endl;
                if ( abs(SiPMWave[j] -  baseline) <= MINIMUM_DIFF) {
                    last.push_back(j - start);
                    i = j;
                    break;
                }
                else {
                    tmp.push_back(SiPMWave[j]);
                    continue;
                    
                }
            }
            double amp_tmp;
            double dcrQ_tmp;
            dcrQ_tmp = std::accumulate(tmp.begin(), tmp.end(), 0.0) - baseline * tmp.size();
            amp_tmp = *max_element(std::begin(tmp), std::end(tmp)) - baseline;
            amp.push_back(amp_tmp);
            charge.push_back(dcrQ_tmp);
            tmp.clear();
        }
    }
    return {po, last, amp, charge};
}

int getSignalTime( const std::vector<float> &SiPMWave, float baseline, int signalStart, int signalEnd, float thres) {
    int time_interval = 0.;
    for (int i = signalStart; i < signalEnd; i++) {
        if (SiPMWave[i] - thres >= baseline) {
            int start = i;
            for (int k = i; k >= signalStart; --k) {
                if (abs(SiPMWave[k] -  baseline) <= MINIMUM_DIFF) {
                    start = k;
                    break;
                }
                if (k == signalStart)
                    start = k;
            }
            int end = i + 1;
            for (int j = i + 1; j < signalEnd + 100; j++) {
                if (abs(SiPMWave[j] -  baseline) <= MINIMUM_DIFF) {
                    end = j;
                    break;
                }
            }
            time_interval = end - start;
        }
    }
    //std::cout<< "time_interval" << time_interval << std::endl;
    return time_interval; 
}
float calculateSigAmp(const std::vector<float> &SiPMWave, size_t signalStart, size_t signalEnd, float baseline, bool gateAverage) {
    float amp = 0;
    if (gateAverage) {
        float sum = 0.0;
        size_t count =0;
        for (size_t i = signalStart; i <= signalEnd; i++) {
            if (i < SiPMWave.size()) {
                sum += SiPMWave[i];
                count++;
            } 
        }
        if (count > 0)
           amp = sum / count;
    }
    else {
        auto startIter = SiPMWave.begin() + signalStart;
        auto endIter = SiPMWave.begin() + signalEnd + 1;
        auto maxElementIter = std::max_element(startIter, endIter);
        if (maxElementIter != endIter) {
           amp = *maxElementIter;
        }
    }
    return amp - baseline;  
}

std::pair<float, float> calculateBaselineAndSigQ(const std::vector<float> &SiPMWave, int signalStart, int signalEnd) {
    float baseline = 0;
    float sigQ = 0;

    // Identify the start and end indices of regions where the signal is above the threshold
    for (int i = 0; i < signalStart - 200; i++) {
        //if (i < 20)
            //baseline = (baseline * i + SiPMWave[i] ) / (i + 1);
        //else if (abs(baseline - SiPMWave[i]) < thres)
            baseline = (baseline * i + SiPMWave[i]) / (i + 1);
    }

    for (int j = signalStart; j < signalEnd; j++) {
        sigQ += SiPMWave[j] - baseline;
    }

    return {baseline, sigQ};
}


int main(int argc, char **argv) {
   Options options(argc, argv);
   YAML::Node const config = options.GetConfig();
   bool applyFilter = Options::NodeAs<bool>(config, {"apply_filter"});
   bool gateAverage = Options::NodeAs<bool>(config, {"amplitude_gate_average"});
   float cutOffFreq = Options::NodeAs<float>(config, {"Butterworth_cutoff_frequency"});
   float dcrThreshold = Options::NodeAs<float>(config, {"DCR_amplitude_threshold"});
   float thresMin = Options::NodeAs<float>(config, {"DCR_threshold_min"});
   float thresMax = Options::NodeAs<float>(config, {"DCR_threshold_max"});
   unsigned int numThresBins = Options::NodeAs<unsigned int>(config, {"DCR_threshold_bins"});
   unsigned int filterOrder = Options::NodeAs<unsigned int>(config, {"Butterworth_order"});
   unsigned int signalStart = Options::NodeAs<unsigned int>(config, {"signal_start"});
   unsigned int signalEnd = Options::NodeAs<unsigned int>(config, {"signal_end"});
   std::string outputName = "output.root";
   if (options.Exists("output")) outputName = options.GetAs<std::string>("output");
   int runNumber = 0;
   if (options.Exists("run"))  runNumber = options.GetAs<int>("run");
   int maxEvents = 9999999;
   if (options.Exists("maxEvents"))  maxEvents = options.GetAs<int>("maxEvents");
   int skipEvents = 0;
   if (options.Exists("skipEvents"))  skipEvents = options.GetAs<int>("skipEvents");
   int ov = 0;
   if (options.Exists("voltage"))  ov = options.GetAs<int>("voltage");
   std::string suffix = "suffix";
   if (options.Exists("type"))  suffix = options.GetAs<std::string>("type");
   TFile *file1 = new TFile(outputName.c_str(), "recreate");
   
   
   int n = N;
   TH2F* hists[numHistograms]; // Array of TH2F pointers
    
   
   TH2F* hists_bkgfreq_real[numHistograms];
   TH2F* hists_bkgfreq_imag[numHistograms];
   TH2F* hists_bkgfreq_amp[numHistograms];
   TH2F* hists_filtered[numHistograms];
   TH2F* hists_overthres[numHistograms];
   TH2F* hists_sigfreq_real[numHistograms];
   TH2F* hists_sigfreq_imag[numHistograms];
   TH2F* hists_sigfreq_amp[numHistograms];
   TH2F* dcr[numHistograms];
   TH1F* dcr1D[numHistograms];
   TH2F* dcr_dsp[numHistograms];
   TH1F* dcr1D_dsp[numHistograms];
   for (int i = 0; i < numHistograms; ++i) {
      // Create a new TH2F object and assign it to the array
      hists[i] = new TH2F(Form("waveform%d", i), "Title", 2010, 0, 16080, 2000, -200, 200);
      hists_bkgfreq_real[i] = new TH2F(Form("freqRealfiltered%d", i), "frequency amp real", N/2, 0, N/2, N, -N/2, N/2);
      hists_bkgfreq_imag[i] = new TH2F(Form("freqImagfiltered%d", i), "frequency amp imag", N/2, 0, N/2, N, -N/2, N/2);
      hists_bkgfreq_amp[i] = new TH2F(Form("freqAmpfiltered%d", i), "frequency amp amplitude", N/2, 0, N/2, N, -N/2, N/2);
      hists_filtered[i] = new TH2F(Form("waveformfiltered%d", i), "Title", 2010, 0, 16080, 2000, -200, 200);
      hists_sigfreq_real[i] = new TH2F(Form("freqReal%d", i), "frequency amp real", N/2, 0, N/2, N, -N/2, N/2);
      hists_sigfreq_imag[i] = new TH2F(Form("freqImag%d", i), "frequency amp imag", N/2, 0, N/2, N, -N/2, N/2);
      hists_sigfreq_amp[i] = new TH2F(Form("freqAmp%d", i), "frequency amp amplitude", N/2, 0, N/2, N, -N/2, N/2);
      hists_overthres[i] = new TH2F(Form("overthres%d", i), "wavefrom over threshold", 2010, 0, 2010, 1000, 0, 50);
      dcr[i] = new TH2F(Form("dcr2D%d", i), "DCR", numThresBins, thresMin, thresMax, 20, 0, 20);
      dcr1D[i] = new TH1F(Form("dcr1D%d",  i), "DCR", numThresBins, thresMin, thresMax);
      dcr_dsp[i] = new TH2F(Form("dcr2D_afterfilter%d", i), "DCR after filter",numThresBins, thresMin, thresMax, 20, 0, 20);
      dcr1D_dsp[i] = new TH1F(Form("dcr1D_afterfilter%d", i), "DCR after filter", numThresBins, thresMin, thresMax);
   }
   TTree* tree = new TTree(Form("run%d_ov%d_%s", runNumber, ov, suffix.c_str()),"signal Q of all ADC channels");
   const int numBranches = 16;
   double signalCharge[numBranches];
   int signalTime[numBranches];
   double signalAmplitude[numBranches];
   double pedestalBL[numBranches];
   std::vector<double> dcrAmplitude[numBranches];
   std::vector<int> dcrTime[numBranches];
   std::vector<double> dcrCharge[numBranches];

   for (int i = 0; i < numBranches; i++) {
      tree->Branch(Form("sigQ_ch%d", i), &signalCharge[i], Form("sigQ_ch%d/D", i));
      tree->Branch(Form("sigAmp_ch%d", i), &signalAmplitude[i], Form("sigAmp_ch%d/D", i));
      tree->Branch(Form("sigTime_ch%d", i), &signalTime[i], Form("sigTime_ch%d/I", i));
      tree->Branch(Form("baseline_ch%d", i), &pedestalBL[i], Form("baseline_ch%d/D", i));
      tree->Branch(Form("dcrAmp_ch%d", i), &dcrAmplitude[i]);
      tree->Branch(Form("dcrTime_ch%d", i), &dcrTime[i]);
      tree->Branch(Form("dcrQ_ch%d", i), &dcrCharge[i]);
   }
   
   //input file name
   std::string filename;
   if  (options.Exists("input")) {
      filename = options.GetAs<std::string>("input");
   }
   else {
      LOG_ERROR << "-i -with inputFilePath is needed!";
      std::abort();
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
   auto locatingEndTime = std::chrono::steady_clock::now();
   // Calculate the elapsed time
   auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(locatingEndTime - locatingStartTime);
   // Output the elapsed time
   std::cout << "Locating events time: " << elapsedTime.count() / 1000. << " seconds\n" << std::endl;
   std::cout << "Time per event: " << elapsedTime.count() / (float)vFInd.size() << " milliseconds\n" << std::endl;
   if (skipEvents > num_events){ 
      LOG_ERROR << "skipEvents larger than maxEvents";
      std::abort(); 
   }
   TH1F *waveform = new TH1F("wave", "wave", 2010, 0, 2010);
   double lowpass[N];
   for (int i = 0 ; i < N; i++) {
      //std::cout << i << "\t: "<< butterworthLowpassFilter(i*1.0,200, 5) << std::endl;
      lowpass[i] = butterworthLowpassFilter(i*1.0, cutOffFreq, filterOrder);
   }
   auto scanningStartTime = std::chrono::steady_clock::now();
   for (int nev = skipEvents ; nev < std::min(num_events, maxEvents + skipEvents)  ; ++nev) {
      if ((nev % 200 == 0 and nev > 0 ) or nev == std::min(num_events, maxEvents + skipEvents) - 1) {
         if (nev == std::min(num_events, maxEvents + skipEvents) - 1) {
            std::cout << "############### Scanning finished! ################" << std::endl;
         }
         std::cout << "scanning events:" << nev + 1 << std::endl;
         auto scanningMidTime = std::chrono::steady_clock::now();
         // Calculate the elapsed time
         auto midElapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(scanningMidTime - scanningStartTime);
         // Output the elapsed time
         std::cout << "Scanning events time passed: " << midElapsedTime.count() / 1000. << " seconds" << std::endl;
         std::cout << "Time per event: " << midElapsedTime.count() / (float) (nev + 1) << " milliseconds" << std::endl;
      }
      int offset = 0;
      std::uint32_t word;
      fseek(fp,vFInd.at(nev).po + sizeof(word) *  offset ,SEEK_SET);
      fread(&word,sizeof(word),1,fp);
      //std::cout << offset << "\t" << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
      int pass = 13;
      fseek(fp, sizeof(word) *  pass ,SEEK_CUR);
      offset += pass;
      for (int ch = 0; ch < 16 ; ch ++) {
         fread(&word,sizeof(word),1,fp);
         //std::cout << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
         int chsize =  word & 0x0000FFFF;
         fseek(fp, sizeof(word) ,SEEK_CUR);
         std::int16_t vol;
         int BS = chsize / 2 - 2;
         int count = 0;
         std::vector<float> waveforms;
         std::vector<float> waveforms_filtered;
         while (BS != 0) {
            fread(&vol, sizeof(vol), 1 ,fp);
            float time = count * 8.;
            float amp = 0;
            amp = vol * TQDC_BITS_TO_PC;
            waveforms.push_back(amp);
            waveform->SetBinContent(count + 1, amp);
            hists[ch]->Fill(time, amp);
            BS --;
            count ++;
         }
         //Compute the transform and look at the magnitude of the output
         TH1 *hm =0;
         TVirtualFFT::SetTransform(0);
         hm = waveform->FFT(hm, "MAG R2C M"); 
         //That's the way to get the current transform object:
         TVirtualFFT *fft = TVirtualFFT::GetCurrentTransform();
         //Use the following method to get the full output:
         Double_t *re_full = new Double_t[N];
         Double_t *im_full = new Double_t[N];
      
         fft->GetPointsComplex(re_full,im_full);
         for (int i = 0 ; i < N ; i++) {
            hists_sigfreq_real[ch]->Fill(i, re_full[i]);
            hists_sigfreq_imag[ch]->Fill(i, im_full[i]);
            hists_sigfreq_amp[ch]->Fill(i, sqrt(re_full[i] * re_full[i] + im_full[i] * im_full[i]));
            //re_full[i] = re_full[i] * N / (i + N);
            //im_full[i] = im_full[i] * N / (i + N);
            re_full[i] *= lowpass[i];
            im_full[i] *= lowpass[i];
            hists_bkgfreq_real[ch]->Fill(i, re_full[i]);
            hists_bkgfreq_imag[ch]->Fill(i, im_full[i]);
            hists_bkgfreq_amp[ch]->Fill(i, sqrt(re_full[i] * re_full[i] + im_full[i] * im_full[i]));
         }
         //Now let's make a backward transform:
         TVirtualFFT *fft_back = TVirtualFFT::FFT(1, &n, "C2R M K");
         fft_back->SetPointsComplex(re_full,im_full);
         fft_back->Transform();
         TH1 *hb = 0;
         //Let's look at the output
         hb = TH1::TransformHisto(fft_back,hb,"Re");
         for (int i = 0; i < N; i++) {
            float amp_backward = hb->GetBinContent(i + 1) / N;
            waveforms_filtered.push_back(amp_backward);
            //std::cout << amp_backward << std::endl;
            hists_filtered[ch]->Fill(i * 8.0, amp_backward);
         }
         auto [baseline, sigQ] = calculateBaselineAndSigQ(waveforms, signalStart, signalEnd); 
         auto [baseline2, sigQ_filter] = calculateBaselineAndSigQ(waveforms_filtered, signalStart, signalEnd); 
         auto signalAmpTmp = calculateSigAmp(waveforms_filtered, signalStart, signalEnd, baseline2, gateAverage);
         auto time_interval = getSignalTime(waveforms_filtered, baseline2, signalStart, signalEnd, 1.0);
         signalAmplitude[ch] = signalAmpTmp;
         signalTime[ch] = time_interval;
         for (unsigned int t = 0; t < numThresBins; ++t) {
             auto [dcr_po, dcr_last, dcr_amp, dcr_charge] = getDCR(waveforms, baseline, signalStart, (t+1) * (thresMax - thresMin) / numThresBins);
             //    std::cout << "thres :" <<  (t+1) * 0.05 << " dcr in one waveform:" << dcr_po.size() << std::endl;
             for (unsigned int k = 0; k < dcr_last.size(); k++) {
                 dcr1D[ch]->Fill((t+0.5) *  (thresMax - thresMin) / numThresBins);
                 dcr[ch]->Fill((t+0.5) *  (thresMax - thresMin) / numThresBins, dcr_last[k]);
             }
         }
         for (unsigned int t = 0; t < numThresBins; ++t) {
             auto [dcr_po, dcr_last, dcr_amp, dcr_charge] = getDCR(waveforms_filtered, baseline2, signalStart, (t+1) *  (thresMax - thresMin) / numThresBins);
             //    std::cout << "thres :" <<  (t+1) * 0.05 << " dcr in one waveform:" << dcr_po.size() << std::endl;
             for (unsigned int k = 0; k < dcr_last.size(); k++) {
                 dcr1D_dsp[ch]->Fill((t+0.5) *  (thresMax - thresMin) / numThresBins);
                 dcr_dsp[ch]->Fill((t+0.5) *  (thresMax - thresMin) / numThresBins, dcr_last[k]);
             }
         }
         auto [po_tmp, last_tmp, dcr_amps, dcr_charges] = getDCR(waveforms_filtered, baseline2, signalStart, dcrThreshold);
         for (size_t i = 0; i < dcr_amps.size(); ++i) {
             dcrAmplitude[ch].push_back(dcr_amps[i]);
             dcrTime[ch].push_back(last_tmp[i]);
             dcrCharge[ch].push_back(dcr_charges[i]);
         }
         if (!applyFilter)
            signalCharge[ch] = sigQ;
         else
            signalCharge[ch] = sigQ_filter;
         for (int i = 0; i < N; i++) {
            hists_overthres[ch]->Fill(i, waveforms_filtered[i] - baseline2);
         }
         if (!applyFilter)
            pedestalBL[ch] = baseline;
         else
            pedestalBL[ch] = baseline2;
         //std::cout << "channel:" << ch << " number of waveforms: " << count << std::endl;
         offset += chsize /4;
         waveforms.clear();
         waveforms_filtered.clear();
         delete hm;
         delete hb;
         delete [] re_full;
         delete [] im_full;
         delete fft_back;
         delete fft;
      }
      tree->Fill();
      for (size_t ch = 0; ch < numHistograms; ++ch) {
          dcrAmplitude[ch].clear();
          dcrTime[ch].clear();
          dcrCharge[ch].clear();
      }
   }
   tree->Write();
   // Write the histograms to the file
   TDirectory* directory = file1->mkdir(Form("dir_run%d_ov%d_%s", runNumber, ov, suffix.c_str()));
   directory->cd();
   for (int i = 0; i < numHistograms; ++i) {
      hists[i]->Write();
      hists_sigfreq_real[i]->Write();
      hists_sigfreq_imag[i]->Write();
      hists_sigfreq_amp[i]->Write();
      hists_filtered[i]->Write();
      hists_overthres[i]->Write();
      hists_bkgfreq_real[i]->Write();
      hists_bkgfreq_imag[i]->Write();
      hists_bkgfreq_amp[i]->Write();
      dcr[i]->Write();
      dcr1D[i]->Write();
      dcr_dsp[i]->Write();
      dcr1D_dsp[i]->Write();
   }
       
   // Cleanup: Delete the histograms and free memory
   for (int i = 0; i < numHistograms; ++i) {
      delete hists[i];
   }
   file1->Close();
   // Close the file
   fclose(fp);

   return 0;
}

