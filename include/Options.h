#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>



/**
 * \brief Provides access to command line options and a configuration file
 *
 * The parsing of command line options is implemented with
 * Boost.Program_options. Values of options can be accessed by their labels
 * using methods \ref GetAs and \ref GetAsChecked. All supported options must be
 * registered beforehand and provided to the constructor. However, several
 * options are added automatically, see Options().
 *
 * The configuration file must be of YAML format. It is parsed with
 * <a href="https://github.com/jbeder/yaml-cpp">yaml-cpp</library> and exposed
 * via method \ref GetConfig. The content of individual nodes in the
 * configuration can be extracted using native methods of YAML::Node, but it is
 * recommended to use \ref NodeAs and \ref NodeAsChecked as they provide a
 * better error reporting.
 */
class Options {
 public:
  /**
   * \brief Exception for class Options
   */

  using Group = boost::program_options::options_description;

  /**
   * \brief Constructor from command line arguments
   *
   * \param[in] argc,argv  Number of command line arguments and an array with
   *   their values, as given to \c main.
   * \param[in] optionGroups  Zero or more instances of
   *   boost::program_options::options_description that describe possible
   *   command line options.
   *
   * If an unregistered option is encountered, terminates the program. Several
   * options are added automatically:
   *  - \c -h,--help  Prints usage information and exists the program.
   *  - \c --config   Sets location of the configuration file. The path is
   *    resolved using FileInPath.
   *  - \c --version  Prints version and exits the program.
   *  - \c -v,--verbosity  Sets the verbosity level for the log.
   */
  Options(int argc, char **argv,
          std::initializer_list<Group> const &optionGroups = {});

  /**
   * \brief Checks if an option with the given label exists
   */
  bool Exists(std::string const &label) const;

  /**
   * \brief Returns the value of the option with the given label
   *
   * The value is represented with the given type. If the requested option has
   * not been specified in the call to the program or if the option does not
   * have a value (i.e. this is a pure flag), throws an exception of type
   * Options::Error.
   */
  template<typename T>
  T GetAs(std::string const &label) const;

  /**
   * \brief Checks and returns the value of the option with the given label
   *
   * \param[in] label    Label that identifies the option.
   * \param[in] checker  Object of type Checker that defines
   *   <tt>bool operator()(T const &)</tt> and checks the validity of the
   *   returned value for the option.
   * 
   * Behaves in the same way as \ref GetAs but additionally checks the value of
   * the option using the provided functor. If the value fails the check, throws
   * an exception of type Options::Error. In a typical use case the checker will
   * be a lambda function.
   */
  template<typename T, typename Checker>
  T GetAsChecked(std::string const &label, Checker const &checker) const;

  /**
   * \brief Returns parsed YAML configuration
   *
   * A brief introduction to the parsing library is available
   * <a href="https://github.com/jbeder/yaml-cpp/wiki/Tutorial">here</a>.
   * If no configuration file has been given, this method throws an exception of
   * type Options::Error.
   */
  YAML::Node const &GetConfig() const;

  /**
   * \brief Returns the content of a node converted into requested type
   *
   * Compared to the native YAML::Node::as, this function provides a more
   * meaningful error message if the node is not defined (meaning that it does
   * not exist in the configuration).
   */
  template<typename T>
  static T NodeAs(YAML::Node const &node,
                  std::initializer_list<std::string> const keys = {});

  /**
   * \brief Checks and returns the content of a node
   *
   * \param[in] node     YAML node whose content is to be read.
   * \param[in] checker  Object of type Checker that defines
   *   <tt>bool operator()(T const &)</tt> and checks the validity of the
   *   content.
   * \tparam T  Type into which to convert the content of the node.
   * 
   * Behaves in the same way as \ref NodeAs but additionally checks the
   * extracted value using the provided functor. If the value fails the check,
   * throws an exception of type Options::Error. In a typical use case the
   * checker will be a lambda function.
   */
  template<typename T, typename Checker>
  static T NodeAsChecked(YAML::Node const &node, Checker const &checker,
                         std::initializer_list<std::string> const keys = {});
  
  static std::vector<double> GetBinning(YAML::Node const &node,
                      std::initializer_list<std::string> const keys = {});
  
  static std::vector<std::string> GetStrings(YAML::Node const &node,
                      std::initializer_list<std::string> const keys = {});
 
 private:
  /**
   * \brief Prints usage instructions
   */
  void PrintUsage() const;

  /// Name of the program extracted from the first command line argument
  std::string programName_;

  /// All registered options
  boost::program_options::options_description allOptions_;

  /// Map with parsed options
  boost::program_options::variables_map optionMap_;

  /**
   * \brief Parsed YAML configuration
   *
   * If no configuration file has been given in command line arguments, the
   * type of this node is Null.
   */
  YAML::Node config_;
};


template<typename T>
T Options::GetAs(std::string const &label) const {
  if (not Exists(label)) {
    std::cout << "Unknown option \"" << label << "\"";
    std::cout<< std::endl;
  }

  return optionMap_[label].as<T>();
}


template<typename T, typename Checker>
T Options::GetAsChecked(std::string const &label,
                        Checker const &checker) const {
  T const value = GetAs<T>(label);

  if (not checker(value)) {
  ;
  }

  return value;
}


template<typename T>
T Options::NodeAs(YAML::Node const &node,
                  std::initializer_list<std::string> const keys) {
  if (not node)
    std::cout<<"The node does not exist."<<std::endl;

  // In order to check wether proveded keys are available in the node, and
  // spot the missing key, the content of the node has to be changed repatedly.
  // For avoid destroying the origin node, checking process must be done on
  // a copy of the origin node. YAML::Clone allows creating a deep copy of
  // the origin node (see: https://github.com/jbeder/yaml-cpp/issues/162)
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

  return copyNode.as<T>();
}


template<typename T, typename Checker>
T Options::NodeAsChecked(YAML::Node const &node, Checker const &checker,
                         std::initializer_list<std::string> const keys) {
  T const value = NodeAs<T>(node, keys);

  if (not checker(value)) {
    std::cout << "Invalid value found for node: " << value;
    std::cout << std::endl;
  }

  return value;
}



  

#endif  // OPTIONS_H_
