#include <Options.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <ExceptionV.h>
namespace po = boost::program_options;


Options::Options(int argc, char **argv,
                 std::initializer_list<Group> const &optionGroups)
    : programName_{argv[0]} {
  
  po::options_description generalOptions{"General"};
  generalOptions.add_options()
    ("help,h", "Prints this help message")
    ("input,i",po::value<std::string>(), "input csv file with datasets")
    ("output,o", po::value<std::string>(),"output file name")
    ("run,r", po::value<int>(),"a number recording the run number")
    ("skipEvents,s", po::value<int>(),"skip first N events")
    ("maxEvents,m", po::value<int>(),"max events")
    ("config,c", po::value<std::string>()->required(),
     "Master configuration file (required)");
  allOptions_.add(generalOptions);

  for (auto const &group : optionGroups)
    allOptions_.add(group);

  po::store(po::parse_command_line(argc, argv, allOptions_), optionMap_);

  if (optionMap_.count("help") > 0) {
    PrintUsage();
    std::exit(EXIT_FAILURE);
  }

  try {
    po::notify(optionMap_);
  } catch (po::error const &e) {
    std::cout << "Error while parsing command line arguments: " << e.what() <<
      ".";
    PrintUsage();
    std::exit(EXIT_FAILURE);
  }


  if (optionMap_.count("config") > 0) {
    config_ = YAML::LoadFile(
      GetAs<std::string>("config"));
  }
}


YAML::Node const &Options::GetConfig() const {
  if (config_.IsNull())
    throw exceptionV("configuration file doesn't exist");
  return config_;
}


bool Options::Exists(std::string const &label) const {
  return (optionMap_.count(label) > 0);
}


void Options::PrintUsage() const {
  std::cerr << "Usage: " << programName_ << " [options]\n";
  std::cerr << allOptions_ << std::endl;
}


std::vector<double> Options::GetBinning(YAML::Node const &node,
                     std::initializer_list<std::string> const keys){
  YAML::Node copyNode = YAML::Clone(node);
  unsigned short int counter = 0;
  for (auto &key : keys) {
    counter++;
    copyNode = copyNode[key];
    if (not copyNode) break;
  }

  if (not copyNode) {
    std::cout << "The node ";

    for (auto key = keys.begin(); key != keys.begin() + counter; ++key)
      std::cout << "["  << *key << "]";
    std::cout << " does not exist.";
    std::cout << std::endl;
  }
  if (not copyNode.IsSequence()){
    throw "Node is not a sequence";
  //unsigned int binning_size = copyNode.size();
  } 
  std::vector<double> binning;
  for(auto it = copyNode.begin(); it != copyNode.end();++it ) {
    binning.push_back(it->as<double>());
  }
  //T* bin_array = new T[bins.size()];
  return binning;

}
 
std::vector<std::string> Options::GetStrings(YAML::Node const &node,
                     std::initializer_list<std::string> const keys){
  YAML::Node copyNode = YAML::Clone(node);
  unsigned short int counter = 0;
  for (auto &key : keys) {
    counter++;
    copyNode = copyNode[key];
    if (not copyNode) break;
  }

  if (not copyNode) {
    std::cout << "The node ";

    for (auto key = keys.begin(); key != keys.begin() + counter; ++key)
      std::cout << "["  << *key << "]";
    std::cout << " does not exist.";
    std::cout << std::endl;
  }
//  if (not copyNode.IsSequence()){
  //  throw "Node is not a sequence";
  //unsigned int binning_size = copyNode.size();
 // } 
  std::vector<std::string> binning;
  for(auto it = copyNode.begin(); it != copyNode.end();++it ) {
    binning.push_back(it->as<std::string>());
    std::cout << "push back:" <<it->as<std::string>() <<std::endl;
  }
  std::cout <<"vector size:"<<binning.size() <<std::endl; //<<"  node size:" << copyNode.size() <<std::endl;
  //T* bin_array = new T[bins.size()];
  return binning;
}
