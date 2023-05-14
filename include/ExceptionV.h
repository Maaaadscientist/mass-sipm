#ifndef EXCEPTIONV_H_
#define EXCEPTIONV_H_

#include <sstream>
#include <string>
#include <stdexcept>


/**
 * \brief Main exception class for the framework
 *
 * The exception contains an explanatory message. It can be constructed in two
 * ways: either given as a string to the constructor, or built in a stream using
 * the left shift operator. Both methods may be combined, with the initial
 * portion of the message given to the constructor and more tokens added later.
 */
class exceptionV : public std::exception {
 public:
  /// Construct from initial portion of the explanatory message
  exceptionV(std::string message = "")
     : message_{message} {}

  /// Appends given token to the explanatory message
  template<typename T>
  exceptionV &operator<<(T const &token) {
    message_ << token;
    return *this;
  }

  /// Returns explanatory message
  char const *what() const noexcept override {
    buffer_ = message_.str();  // Needed to preserve the temporary object
    return buffer_.c_str();
  }

 private:
  /// Stream with explanatory message
  std::ostringstream message_;

  /// Buffer for \ref what
  mutable std::string buffer_;
};

#endif  // EXCEPTIONV_H_
