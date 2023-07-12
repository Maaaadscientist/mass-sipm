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
#define MINIMUM_DIFF 0.2
typedef struct {
   size_t ev;			// event number
   size_t po;			// event header position
} FILEINDEX;

double butterworthLowpassFilter(double input, double cutoffFreq, int order) {
    
    double output = 1 / sqrt(1 + pow((input / cutoffFreq), 2 * order)) ;
    return output;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> getDCR( const std::vector<float> &SiPMWave, float baseline, int dcrStart, int dcrEnd, float thres) {
    std::vector<double> amp;
    std::vector<double> charge;
    std::vector<double> chargeBkg;

    auto averageFromWaveform = [&](int k) {
        float sum = 0.0;
        for (int index = k; index < k - 10 && index >= 0; --index)
            sum += SiPMWave[index];
        return sum / 10.0;
    };

    for (int i = dcrStart; i < dcrEnd; i++) {
        //std::cout<< SiPMWave[i] << "," << baseline <<"," << thres <<std::endl;
        if (SiPMWave[i] - thres >= baseline) {
            int start = i - 10;
            for (int k = i + 10; k >= 10; --k) {
                //std::cout<< "tracing before:" <<k << std::endl;
                if (abs(averageFromWaveform(k) -  baseline) <= MINIMUM_DIFF) {
                    start = k - 5;
                    break;
                }
            }
            double amp_tmp = 0.;
            double dcrQ_tmp = 0.;
            //dcrQ_tmp = std::accumulate(tmp.begin(), tmp.end(), 0.0) - baseline * tmp.size();
            for (int index = start; index < start + 45 ; ++index) {
                dcrQ_tmp += SiPMWave[index] - baseline;
                if ( SiPMWave[index] > amp_tmp) amp_tmp = SiPMWave[index];
            }
            amp.push_back(amp_tmp - baseline);
            charge.push_back(dcrQ_tmp);
        }
    }
    for (int i = dcrStart; i < dcrEnd; i++) {
        //std::cout<< SiPMWave[i] << "," << baseline <<"," << thres <<std::endl;
        if (SiPMWave[i] - thres >= baseline and SiPMWave[i+1] - thres < baseline) {
            int start = i - 10;
            for (int k = i + 10; k >= 10; --k) {
                //std::cout<< "tracing before:" <<k << std::endl;
                if (abs(averageFromWaveform(k) -  baseline) <= MINIMUM_DIFF) {
                    start = k - 5;
                    break;
                }
            }
            double dcrQ_tmp = 0.;
            //dcrQ_tmp = std::accumulate(tmp.begin(), tmp.end(), 0.0) - baseline * tmp.size();
            for (int index = start; index < start + 45 ; ++index) {
                dcrQ_tmp += SiPMWave[index] - baseline;
            }
            chargeBkg.push_back(dcrQ_tmp);
        }
    }
    return {amp, charge, chargeBkg};
}

std::pair<float, float> calculateBaselineAndSigQ(const std::vector<float> &SiPMWave, int baselineStart, int baselineEnd) {
    float baseline = 0;
    float sigQ = 0;
    // Identify the start and end indices of regions where the signal is above the threshold
    for (int i = baselineStart; i < baselineEnd; i++) {
        //if (i < 20)
            //baseline = (baseline * i + SiPMWave[i] ) / (i + 1);
        //else if (abs(baseline - SiPMWave[i]) < thres)
            baseline += SiPMWave[i];
    }
    baseline  = baseline / (baselineEnd - baselineStart);


    return {baseline, sigQ};
}


int main(int argc, char **argv) {
   Options options(argc, argv);
   YAML::Node const config = options.GetConfig();
   float cutOffFreq = Options::NodeAs<float>(config, {"Butterworth_cutoff_frequency"});
   float dcrThreshold = Options::NodeAs<float>(config, {"DCR_amplitude_threshold"});
   unsigned int filterOrder = Options::NodeAs<unsigned int>(config, {"Butterworth_order"});
   unsigned int dcrStart = Options::NodeAs<unsigned int>(config, {"dcr_start"});
   unsigned int dcrEnd = Options::NodeAs<unsigned int>(config, {"dcr_end"});
   unsigned int baselineStart = Options::NodeAs<unsigned int>(config, {"baseline_start"});
   unsigned int baselineEnd = Options::NodeAs<unsigned int>(config, {"baseline_end"});
   std::string outputName = "output.root";
   if (options.Exists("output")) outputName = options.GetAs<std::string>("output");
   int maxEvents = 9999999;
   if (options.Exists("maxEvents"))  maxEvents = options.GetAs<int>("maxEvents");
   int skipEvents = 0;
   if (options.Exists("skipEvents"))  skipEvents = options.GetAs<int>("skipEvents");
   std::string suffix = "suffix";
   if (options.Exists("type"))  suffix = options.GetAs<std::string>("type");
   TFile *file1 = new TFile(outputName.c_str(), "recreate");
   
   
   int n = N;
   TTree* tree = new TTree("dcr","dcr Q of all ADC channels");
   const int numBranches = 16;
   double pedestalBL[numBranches];
   std::vector<double> dcrAmplitude[numBranches];
   std::vector<double> dcrCharge[numBranches];
   std::vector<double> bkgCharge[numBranches];

   for (int i = 0; i < numBranches; i++) {
      tree->Branch(Form("dcrAmp_ch%d", i), &dcrAmplitude[i]);
      tree->Branch(Form("dcrQ_ch%d", i), &dcrCharge[i]);
      tree->Branch(Form("bkgQ_ch%d", i), &bkgCharge[i]);
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
   std::cout<< "Begin to skim: "<< num_events << " events"<<std::endl;
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
            float amp = 0;
            amp = vol * TQDC_BITS_TO_PC;
            waveforms.push_back(amp);
            waveform->SetBinContent(count + 1, amp);
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
         }
         auto [baseline2, sigQ_filter] = calculateBaselineAndSigQ(waveforms_filtered, baselineStart,baselineEnd); 
         auto [dcr_amps, dcr_charges, bkg_charges] = getDCR(waveforms_filtered, baseline2, dcrStart, dcrEnd, dcrThreshold);
         for (size_t i = 0; i < dcr_amps.size(); ++i) {
             dcrAmplitude[ch].push_back(dcr_amps[i]);
             dcrCharge[ch].push_back(dcr_charges[i]);
         }
         for (size_t i = 0; i < bkg_charges.size(); ++i) {
             bkgCharge[ch].push_back(bkg_charges[i]);
         }
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
          dcrCharge[ch].clear();
          bkgCharge[ch].clear();
      }
   }
   tree->Write();
   // Write the histograms to the file
       
   file1->Close();
   // Close the file
   fclose(fp);

   return 0;
}

