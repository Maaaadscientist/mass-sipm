#include <iostream>
#include <TGraph2D.h>

int main(int argc, char** argv) {
   Options options(argc, argv);
   YAML::Node const config = options.GetConfig();
   float thresMax = Options::NodeAs<float>(config, {"DCR_threshold_max"});
   unsigned int upperBound = Options::NodeAs<unsigned int>(config, {"baseline_bound"});
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



}
