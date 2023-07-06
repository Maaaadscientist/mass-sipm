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
   std::vector<double> waveform[numBranches];
   for (int i = 0; i < numBranches; i++) {
      tree->Branch(Form("sigQ_ch%d", i), &signalCharge[i], Form("sigQ_ch%d/D", i));
      tree->Branch(Form("sigAmp_ch%d", i), &signalAmplitude[i], Form("sigAmp_ch%d/D", i));
      tree->Branch(Form("baselineQ_ch%d", i), &baselineCharge[i], Form("baselineQ_ch%d/D", i));
      tree->Branch(Form("waveform%d", i), &waveform[i]);
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
         while (BS != 0) {
            fread(&vol, sizeof(vol), 1 ,fp);
            double amp = 0;
            amp = vol * TQDC_BITS_TO_PC;
            waveforms.push_back(amp);
            BS --;
            count ++;
         }
      
         auto sigQ = chargeIntegral(waveforms, signalStart, signalEnd); 
         auto baselineQ = chargeIntegral(waveforms, baselineStart, baselineEnd); 
         auto signalAmpTmp = calculateSigAmp(waveforms, signalStart, signalEnd);
         signalAmplitude[ch] = signalAmpTmp;
         signalCharge[ch] = sigQ;
         baselineCharge[ch] = baselineQ;
         waveform[ch] = waveforms;
         offset += chsize /4;
         waveforms.clear();
      }
      tree->Fill();
   }
   tree->Write();
   // Write the histograms to the file
       
   file1->Close();
   // Close the file
   fclose(fp);

   return 0;
}

