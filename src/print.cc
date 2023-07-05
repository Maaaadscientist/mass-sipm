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

#include <cstdio>
#include <iomanip>
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

int main(int argc, char **argv) {
   Options options(argc, argv);
   YAML::Node const config = options.GetConfig();
   std::string outputName = "output.txt";
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
   if (skipEvents > num_events){ 
      LOG_ERROR << "skipEvents larger than maxEvents";
      std::abort(); 
   }
   std::ofstream outputFile;
   outputFile.open(outputName);
   for (int nev = skipEvents ; nev < std::min(num_events, maxEvents + skipEvents)  ; ++nev) {
      int offset = 0;
      std::uint32_t word;
      fseek(fp,vFInd.at(nev).po + sizeof(word) *  offset ,SEEK_SET);
      fread(&word,sizeof(word),1,fp);
      //std::cout << offset << "\t" << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
      int pass = 13;
      fseek(fp, sizeof(word) *  pass ,SEEK_CUR);
      offset += pass;
      for (int ch = 0; ch < 16 ; ch ++) {
         if (ch != 0) continue;
         fread(&word,sizeof(word),1,fp);
         //std::cout << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
         int chsize =  word & 0x0000FFFF;
         fseek(fp, sizeof(word) ,SEEK_CUR);
         std::int16_t vol;
         int BS = chsize / 2 - 2;
         int count = 0;
         while (BS != 0) {
            fread(&vol, sizeof(vol), 1 ,fp);
            float time = count * 8.;
            float amp = 0;
            amp = vol * TQDC_BITS_TO_PC;
            std::cout<< amp<< std::endl;
            if (count >= 1255 and count <= 1300) {
               if (count != 1255) outputFile << " "; 
               outputFile << amp;
            }
            BS --;
            count ++;
         }
         outputFile << std::endl;
      }
   }
   outputFile.close();
   return 0;
}

