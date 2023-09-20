#include <iostream>
#include <TFile.h>
#include <TH1.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <ROOT_FILE_PATH>" << std::endl;
    return 1;
  }

  TFile *file = TFile::Open(argv[1]);
  if (!file || file->IsZombie()) {
    std::cerr << "Could not open ROOT file: " << argv[1] << std::endl;
    return 1;
  }

  TH1F *histogram = (TH1F *)file->Get("reff_mu_Run227_Point0");
  if (!histogram) {
    std::cerr << "Histogram 'reff_mu_Run227_Point0' not found in file: " << argv[1] << std::endl;
    return 1;
  }

  int numBins = histogram->GetNbinsX();
  std::cout << "bin,value,error" << std::endl;  // CSV header

  for (int i = 1; i <= numBins; ++i) {
    float value = histogram->GetBinContent(i);
    float error = histogram->GetBinError(i);
    std::cout << i << "," << value << "," << error << std::endl;
  }

  file->Close();
  return 0;
}

