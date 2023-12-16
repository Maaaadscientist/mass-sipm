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
#include <TH1.h>
#include <TH2.h>
#include <TVirtualFFT.h>
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

int getSignalTime( const std::vector<double> &SiPMWave, double baseline, int signalStart, int signalEnd, double thres) {
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
double calculateSigAmp(const std::vector<double> &SiPMWave, size_t signalStart, size_t signalEnd) {
    double amp = 0;
    auto startIter = SiPMWave.begin() + signalStart;
    auto endIter = SiPMWave.begin() + signalEnd + 1;
    auto maxElementIter = std::max_element(startIter, endIter);
    if (maxElementIter != endIter) {
       amp = *maxElementIter;
    }
    
    return amp;  
}

double chargeIntegral(const std::vector<double> &SiPMWave, int signalStart, int signalEnd) {
    double sigQ = 0;

    for (int j = signalStart; j < signalEnd; j++) {
        sigQ += SiPMWave[j];
    }

    return sigQ;
}


double butterworthLowpassFilter(double input, double cutoffFreq, int order) {
    
    double output = 1 / sqrt(1 + pow((input / cutoffFreq), 2 * order)) ;
    return output;
}
int main(int argc, char **argv) {
   Options options(argc, argv);
   YAML::Node const config = options.GetConfig();
   unsigned int signalStart = Options::NodeAs<unsigned int>(config, {"signal_start"});
   unsigned int signalEnd = Options::NodeAs<unsigned int>(config, {"signal_end"});
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
   
   
   TTree* tree = new TTree("signal","signal Q and Amp of all ADC channels");
   const int numBranches = 16;
   double signalCharge[numBranches];
   double baselineCharge[numBranches];
   double signalAmplitude[numBranches];
   double baselineAmplitude[numBranches];

   for (int i = 0; i < numBranches; i++) {
      tree->Branch(Form("sigQ_ch%d", i), &signalCharge[i], Form("sigQ_ch%d/D", i));
      tree->Branch(Form("sigAmp_ch%d", i), &signalAmplitude[i], Form("sigAmp_ch%d/D", i));
      tree->Branch(Form("baselineQ_ch%d", i), &baselineCharge[i], Form("baselineQ_ch%d/D", i));
      tree->Branch(Form("baselineAmp_ch%d", i), &baselineAmplitude[i], Form("baselineAmp_ch%d/D", i));
   }
   TH2F* hists[numHistograms]; // Array of TH2F pointers
    
   
   TH2F* hists_filtered[numHistograms];
   TH2F* hists_bkgfreq_real[numHistograms];
   TH2F* hists_bkgfreq_imag[numHistograms];
   TH2F* hists_bkgfreq_amp[numHistograms];
   TH2F* hists_sigfreq_real[numHistograms];
   TH2F* hists_sigfreq_imag[numHistograms];
   TH2F* hists_sigfreq_amp[numHistograms];
   for (int i = 0; i < numHistograms; ++i) {
      // Create a new TH2F object and assign it to the array
      hists[i] = new TH2F(Form("waveform_ch%d", i), "Title", 2010, 0, 2010, 2000, -200, 200);
      hists_bkgfreq_real[i] = new TH2F(Form("freqRealfiltered_ch%d", i), "frequency amp real", N/2, 0, N/2, N, -N/2, N/2);
      hists_bkgfreq_imag[i] = new TH2F(Form("freqImagfiltered_ch%d", i), "frequency amp imag", N/2, 0, N/2, N, -N/2, N/2);
      hists_bkgfreq_amp[i] = new TH2F(Form("freqAmpfiltered_ch%d", i), "frequency amp amplitude", N/2, 0, N/2, N, -N/2, N/2);
      hists_filtered[i] = new TH2F(Form("waveformfiltered_ch%d", i), "Title", 2010, 0, 2010, 2000, -200, 200);
      hists_sigfreq_real[i] = new TH2F(Form("freqReal_ch%d", i), "frequency amp real", N/2, 0, N/2, N, -N/2, N/2);
      hists_sigfreq_imag[i] = new TH2F(Form("freqImag_ch%d", i), "frequency amp imag", N/2, 0, N/2, N, -N/2, N/2);
      hists_sigfreq_amp[i] = new TH2F(Form("freqAmp_ch%d", i), "frequency amp amplitude", N/2, 0, N/2, N, -N/2, N/2);
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
   std::cout << "Time per event: " << elapsedTime.count() / (double)vFInd.size() << " milliseconds\n" << std::endl;
   if (skipEvents > num_events){ 
      LOG_ERROR << "skipEvents larger than maxEvents";
      std::abort(); 
   }
   auto scanningStartTime = std::chrono::steady_clock::now();
   std::cout<< "Begin to skim: "<< num_events << " events"<<std::endl;
   double lowpass[N];
   for (int i = 0 ; i < N; i++) {
      //std::cout << i << "\t: "<< butterworthLowpassFilter(i*1.0,200, 5) << std::endl;
      lowpass[i] = butterworthLowpassFilter(i*1.0, 200, 5);
   }
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
         std::cout << "Time per event: " << midElapsedTime.count() / (double) (nev + 1) << " milliseconds" << std::endl;
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
         std::vector<double> waveforms;
         std::vector<double> waveforms_filtered;
         TH1F *wf_hist = new TH1F("wave", "wave", 2010, 0, 2010);
         while (BS != 0) {
            fread(&vol, sizeof(vol), 1 ,fp);
            double amp = 0;
            amp = vol * TQDC_BITS_TO_PC;
            waveforms.push_back(amp);
            wf_hist->SetBinContent(count + 1, amp);
            hists[ch]->Fill(count, amp);
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
            hists_sigfreq_real[ch]->Fill(i, re_full[i]);
            hists_sigfreq_imag[ch]->Fill(i, im_full[i]);
            hists_sigfreq_amp[ch]->Fill(i, sqrt(re_full[i] * re_full[i] + im_full[i] * im_full[i]));
            re_full[i] *= lowpass[i];
            im_full[i] *= lowpass[i];
            hists_bkgfreq_real[ch]->Fill(i, re_full[i]);
            hists_bkgfreq_imag[ch]->Fill(i, im_full[i]);
            hists_bkgfreq_amp[ch]->Fill(i, sqrt(re_full[i] * re_full[i] + im_full[i] * im_full[i]));
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
            waveforms_filtered.push_back(amp_backward);
            hists_filtered[ch]->Fill(i, amp_backward);
         }
         auto sigQ = chargeIntegral(waveforms, signalStart, signalEnd); 
         auto baselineQ = chargeIntegral(waveforms, baselineStart, baselineEnd); 
         auto signalAmpTmp = calculateSigAmp(waveforms_filtered, signalStart, signalEnd);
         auto baselineAmpTmp = calculateSigAmp(waveforms_filtered, baselineStart, baselineEnd);
         signalAmplitude[ch] = signalAmpTmp;
         baselineAmplitude[ch] = baselineAmpTmp;
         signalCharge[ch] = sigQ;
         baselineCharge[ch] = baselineQ;
         offset += chsize /4;
         waveforms.clear();
         waveforms_filtered.clear();
         delete hm;
         delete hb;
         delete [] re_full;
         delete [] im_full;
         delete fft_back;
         delete fft;
         delete wf_hist;
      }
      tree->Fill();
   }
   tree->Write();
   // Write the histograms to the file
   for (int i = 0; i < numHistograms; ++i) {
      hists[i]->Write();
      hists_filtered[i]->Write();
      hists_bkgfreq_real[i]->Write();
      hists_bkgfreq_imag[i]->Write();
      hists_bkgfreq_amp[i]->Write();
      hists_sigfreq_real[i]->Write();
      hists_sigfreq_imag[i]->Write();
      hists_sigfreq_amp[i]->Write();
   }
       
   // Cleanup: Delete the histograms and free memory
   for (int i = 0; i < numHistograms; ++i) {
      delete hists[i];
      delete hists_filtered[i];
      delete hists_bkgfreq_real[i];
      delete hists_bkgfreq_imag[i];
      delete hists_bkgfreq_amp[i];
      delete hists_sigfreq_real[i];
      delete hists_sigfreq_imag[i];
      delete hists_sigfreq_amp[i];
   }
       
   file1->Close();
   // Close the file
   fclose(fp);

   return 0;
}

