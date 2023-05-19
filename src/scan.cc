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

typedef struct {
   size_t ev;			// event number
   size_t po;			// event header position
} FILEINDEX;

std::pair<float, float> calculateBaselineAndSigQ(const std::vector<float> &SiPMWave, int signalStart, int signalEnd, float thres) {
    float baseline = 0;
    float sigQ = 0;

    // Identify the start and end indices of regions where the signal is above the threshold
    for (int i = 0; i < signalStart - 100; i++) {
        if (i < 20)
            baseline = (baseline * i + SiPMWave[i] ) / (i + 1);
        else if (abs(baseline - SiPMWave[i]) < thres)
            baseline = (baseline * i + SiPMWave[i]) / (i + 1);
    }

    for (int j = signalStart; j < signalEnd; j++) {
        sigQ += SiPMWave[j] - baseline;
    }

    return {baseline, sigQ};
}


int main(int argc, char **argv) {
   Options options(argc, argv);
   std::string outputName = "output.root";
   if (options.Exists("output")) outputName = options.GetAs<std::string>("output");
   int runNumber = 0;
   if (options.Exists("run"))  runNumber = options.GetAs<int>("run");
   int maxEvents = 9999999;
   if (options.Exists("maxEvents"))  maxEvents = options.GetAs<int>("maxEvents");
   int skipEvents = 0;
   if (options.Exists("skipEvents"))  skipEvents = options.GetAs<int>("skipEvents");
   TFile *file1 = new TFile(outputName.c_str(), "recreate");
   
   const int numHistograms = 16; // Number of histograms in the array
   TH2F* hists[numHistograms]; // Array of TH2F pointers
    
   
   int N = 2010; // Number of samples
   TH2F* hists_bkgfreq_real[numHistograms];
   TH2F* hists_bkgfreq_imag[numHistograms];
   TH2F* hists_filtered[numHistograms];
   TH2F* hists_sigfreq_real[numHistograms];
   TH2F* hists_sigfreq_imag[numHistograms];
   for (int i = 0; i < numHistograms; ++i) {
      // Create a new TH2F object and assign it to the array
      hists[i] = new TH2F(Form("run%dwaveform%d", runNumber, i), "Title", 2010, 0, 16080, 2000, -200, 200);
      hists_bkgfreq_real[i] = new TH2F(Form("run%dbkgfreqreal%d", runNumber, i), "frequency amp real", N, 0, N, N, -N/2, N/2);
      hists_bkgfreq_imag[i] = new TH2F(Form("run%dbkgfreqimag%d", runNumber, i), "frequency amp imag", N, 0, N, N, -N/2, N/2);
      hists_filtered[i] = new TH2F(Form("run%dwaveformfiltered%d", runNumber, i), "Title", 2010, 0, 16080, 2000, -200, 200);
      hists_sigfreq_real[i] = new TH2F(Form("run%dsigfreqreal%d", runNumber, i), "frequency amp real", N, 0, N, N, -N/2, N/2);
      hists_sigfreq_imag[i] = new TH2F(Form("run%dsigfreqimag%d", runNumber, i), "frequency amp imag", N, 0, N, N, -N/2, N/2);
   }
   TTree* tree = new TTree(Form("run%d", runNumber),"signal Q of all ADC channels");
   const int numBranches = 16;
   int branchValues[numBranches];
   int filterValues[numBranches];

   for (int i = 0; i < numBranches; i++) {
      std::string branchName = "sigQ_ch_" + std::to_string(i);
      tree->Branch(branchName.c_str(), &branchValues[i], (branchName + "/I").c_str());
      std::string branchName_filter = "sigQ_filtered_ch_" + std::to_string(i);
      tree->Branch(branchName_filter.c_str(), &filterValues[i], (branchName_filter + "/I").c_str());
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

   while (!feof(fp)) {
      fread(&buffer, sizeof(std::uint32_t), 1 , fp);
      if (buffer == 0x2A502A50) {
         fseek(fp,-sizeof(uint32_t),SEEK_CUR);
         fread(uiEVENT_HEADER,sizeof(uiEVENT_HEADER),1,fp);
         FileIndex.ev = num_events;   // absolute event number by data file
         FileIndex.po = ftell(fp)-3*sizeof(uint32_t);
         num_events++;
         std::cout << num_events << std::endl;
         vFInd.push_back(FileIndex);
         fseek(fp,FileIndex.po+uiEVENT_HEADER[1],SEEK_SET);		// from zero position - jump over all event length
         if (FileIndex.po+uiEVENT_HEADER[1] > sSizeOfFile) break;
      }
   }
   if (skipEvents > num_events){ 
      LOG_ERROR << "skipEvents larger than maxEvents";
      std::abort(); 
   }
   TH1F *waveform = new TH1F("wave", "wave", 2010, 0, 2010);
   for (int nev = skipEvents ; nev < std::min(num_events, maxEvents + skipEvents)  ;nev++) {
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
         std::vector<float> amplitudes;
         std::vector<float> amplitudes_filtered;
         while (BS != 0) {
            fread(&vol, sizeof(vol), 1 ,fp);
            float time = count * 8.;
            float amp = 0;
            amp = vol * TQDC_BITS_TO_PC;
            amplitudes.push_back(amp);
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
            //re_full[i] = re_full[i] * N / (i + N);
            //im_full[i] = im_full[i] * N / (i + N);
            if (i > 100) {
              re_full[i] = 0;
              im_full[i] = 0;
            }
            hists_bkgfreq_real[ch]->Fill(i, re_full[i]);
            hists_bkgfreq_imag[ch]->Fill(i, im_full[i]);
         }
         //Now let's make a backward transform:
         TVirtualFFT *fft_back = TVirtualFFT::FFT(1, &N, "C2R M K");
         fft_back->SetPointsComplex(re_full,im_full);
         fft_back->Transform();
         TH1 *hb = 0;
         //Let's look at the output
         hb = TH1::TransformHisto(fft_back,hb,"Re");
         for (int i = 0 ; i < N ; i++) {
            float amp_backward = hb->GetBinContent(i + 1) / N;
            amplitudes_filtered.push_back(amp_backward);
            //std::cout << amp_backward << std::endl;
            hists_filtered[ch]->Fill(i * 8.0, amp_backward);
         }
         auto [baseline, sigQ] = calculateBaselineAndSigQ(amplitudes, 1255, 1300, 1.5); 
         auto [baseline2, sigQ_filter] = calculateBaselineAndSigQ(amplitudes_filtered, 1255, 1300, 1.5); 
         branchValues[ch] = sigQ;
         filterValues[ch] = sigQ_filter;
         //std::cout << "channle:" << ch << " number of waveforms: " << count << std::endl;
         offset += chsize /4;
         amplitudes.clear();
         amplitudes_filtered.clear();
         delete hm;
         delete hb;
         delete [] re_full;
         delete [] im_full;
         delete fft_back;
         delete fft;
      }
      tree->Fill();
   }
   tree->Write();
   // Write the histograms to the file
   for (int i = 0; i < numHistograms; ++i) {
      hists[i]->Write();
      hists_sigfreq_real[i]->Write();
      hists_sigfreq_imag[i]->Write();
      hists_filtered[i]->Write();
      hists_bkgfreq_real[i]->Write();
      hists_bkgfreq_imag[i]->Write();
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

