#ifndef LOGGER_H_
#define LOGGER_H_

// This macro needs to be defined to inform Boost.Log library that it is linked
// dynamically
#ifndef BOOST_LOG_DYN_LINK
#define BOOST_LOG_DYN_LINK
#endif

#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_feature.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>


/**
 * \brief Implements simple logging
 *
 * Log messages are printed to stderr. A hierarchy of severity levels is
 * provided, and log messages can be filtered based on the assigned severity.
 * Warning and error messages are automatically prepended with textual tags that
 * allow to identify them. If the output is printed to terminal as opposed to
 * being redirected to a file, warning and error messages are additionally
 * coloured. Apart from this, no formatting is applied to messages.
 *
 * The underlying implementation logger is wrapped into a Mayer's singleton. It
 * is automatically constructed at the first usage and available globally. The
 * threshold severity level for filtering can be set with \ref SetLevel. This
 * logger is not safe to use in the deconstruction phase at the end of the
 * application since it might be destroyed before the object that uses it.
 *
 * To issue log messages the user should use macros \ref LOG_TRACE,
 * \ref LOG_DEBUG, \ref LOG_INFO, \ref LOG_WARN, \ref LOG_ERROR, depending on
 * the severity level. They support the usual stream syntax. The arguments to
 * the right of operator<< are only evaluated if the assigned severity level
 * passes the filtering, which means that rejected messages have very little
 * runtime overhead. A new line symbol is added at the end of the log message
 * automatically.
 */
class Logger {
 public:
  /// Supported severity levels for log messages
  enum class SeverityLevel : int {
    kTrace = 0,
    kDebug,
    kInfo,
    kWarning,
    kError
  };

  /// Auxiliary structure used in \ref TimeStamp manipulator
  struct _TimeStamp {};

  using logger_t = boost::log::sources::severity_logger<SeverityLevel>;

  ~Logger() noexcept;

  /// Returns a reference to the underlying implementation logger
  static logger_t &Get() {
    return GetInstance().logger_;
  }

  /**
   * \brief Sets threshold to filter log messages by severity
   *
   * Messages with lower severity level will be discarded. Before the first call
   * to this method the logger will display all messages.
   */
  static void SetLevel(SeverityLevel threshold);

  /**
   * \brief Manipulator to write current time stamp into the log
   *
   * The current time stamp can be written as in the following example:
   * \code
   * LOG_INFO << Logger::TimeStamp << " Log message";
   * \endcode
   * It includes date and local time, rounded to seconds, in ISO 8601 format.
   */
  static _TimeStamp TimeStamp() {
    return {};
  };

 private:
  Logger();
  Logger(Logger const &) = delete;
  Logger &operator=(Logger const &) = delete;

  /// Returns reference to the only instance of this class
  static Logger &GetInstance() {
    static Logger instance;
    return instance;
  }

  /// Underlying implementation of the logger
  logger_t logger_;

  /**
   * \brief Sink used with the logger
   *
   * It writes the output to stderr.
   */
  boost::shared_ptr<boost::log::sinks::synchronous_sink<
    boost::log::sinks::text_ostream_backend>> sink_;
};


/**
 * \def LOG_TRACE
 * \brief Writes to the log a message with lowest possible level of severity
 *
 * All per-event diagnostics must be reported with this severity.
 */
#define LOG_TRACE \
  BOOST_LOG_SEV(Logger::Get(), Logger::SeverityLevel::kTrace)

/**
 * \def LOG_DEBUG
 * \brief Writes a debug-level message to the log
 *
 * This should include messages that are of no interest during a regular run of
 * the application but useful for debugging. The frequency of these messages
 * should be such that the log of a complete run of the application is still
 * readable.
 */
#define LOG_DEBUG \
  BOOST_LOG_SEV(Logger::Get(), Logger::SeverityLevel::kDebug)

/**
 * \def LOG_INFO
 * \brief Writes an information-level message to the log
 *
 * This should include messages that are useful during a regular run of the
 * application.
 */
#define LOG_INFO  \
  BOOST_LOG_SEV(Logger::Get(), Logger::SeverityLevel::kInfo)

/**
 * \def LOG_WARN
 * \brief Writes a warning message to the log
 *
 * This should be used to report problems from which the affected operation can
 * recover.
 */
#define LOG_WARN  \
  BOOST_LOG_SEV(Logger::Get(), Logger::SeverityLevel::kWarning)

/**
 * \def LOG_ERROR
 * \brief Write an error message to the log
 *
 * This should be used to report problems that are fatal to an operation. An
 * error will often cause the application to terminate.
 */
#define LOG_ERROR \
  BOOST_LOG_SEV(Logger::Get(), Logger::SeverityLevel::kError)


/**
 * \brief Writes the time stamp into the stream
 *
 * \see Logger::TimeStamp.
 */
boost::log::basic_record_ostream<char> &operator<<(
    boost::log::basic_record_ostream<char> &stream, Logger::_TimeStamp (*)());

#endif  // LOGGER_H_

