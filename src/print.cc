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

#include <TVirtualFFT.h>

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
        for (int ch = 0; ch < 16 ; ++ch) {
           if (ch != 11) continue;
           int total_offset = 0;
           if (ch != 0) {
              for (int i = 0; i < ch; ++i) {
                  fread(&word,sizeof(word),1,fp);  // sizeof(word) == 4 bytes, offset == 4
                  total_offset += sizeof(word);
                  int chsize =  word & 0x0000FFFF;
                  fseek(fp, sizeof(word) ,SEEK_CUR); //sizeof(word) == 4 bytes, offset == 8
                  total_offset += sizeof(word);
                  std::int16_t vol;
                  int BS = chsize / 2 - 2;
                  fseek(fp, sizeof(vol) * BS ,SEEK_CUR); //sizeof(word) == 4 bytes, offset == 8
                  total_offset += sizeof(vol) * BS;
              }
           }
           fread(&word,sizeof(word),1,fp);  // sizeof(word) == 4 bytes, offset == 4
           total_offset += sizeof(word);
           //std::cout << std::hex << "0x" << std::setw(8) << std::setfill('0') << word << std::dec << std::endl;
           int chsize =  word & 0x0000FFFF;
           fseek(fp, sizeof(word) ,SEEK_CUR); //sizeof(word) == 4 bytes, offset == 8
           total_offset += sizeof(word);
           std::int16_t vol;
           int BS = chsize / 2 - 2;
           int count = 0;
           std::vector<double> waveform;
           std::vector<double> waveform_filtered;
           while (BS != 0) {
              fread(&vol, sizeof(vol), 1 ,fp); // sizeof(vol) == 2 bytes, offset += chsize - 4
              float time = count * 8.;
              float amp = 0;
              amp = vol * TQDC_BITS_TO_PC;
              waveform.push_back(amp);
              wf_hist->SetBinContent(count + 1, amp);
              //std::cout<< amp<< std::endl;
              BS --;
              count ++;
           }
           total_offset += sizeof(vol) * BS;
           fseek(fp, - total_offset ,SEEK_CUR);

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
           for (auto wf: charge_wf) {
               std::cout << wf << " ";
           }
           //for (auto wf: waveform) {
           //    std::cout << wf << " ";
           //}
            
           //for (auto wf: aligned_wf) {
           //    std::cout << wf << " ";
           //}
           std::cout << std::endl;
           delete hm;
           delete hb;
           delete [] re_full;
           delete [] im_full;
           delete fft_back;
           delete fft;
        }
    }
    return 0;
}

