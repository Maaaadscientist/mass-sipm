#include "Logger.h"

#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <iomanip>
#include <iostream>

#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <TError.h>

#define BOOST_STACKTRACE_LINK
#include <boost/stacktrace.hpp>


// This will point to the original terminate handler
std::terminate_handler original_terminate;

// Customized reporting of unhandled exceptions
void logged_terminate() noexcept {
  LOG_ERROR << "Program termination requested. Stack trace:\n"
    << boost::stacktrace::stacktrace() << "Further details:";
  original_terminate();
}


// This will point to the oirignal error handler for ROOT
ErrorHandlerFunc_t OriginalRootErrorHandler;

// Customized error handler for ROOT
void LoggedRootErrorHandler(int level, bool abort, char const *location,
                            char const *message) {
  if (level < kInfo)
    LOG_DEBUG << location << ": " << message;
  else if (level < kWarning)
    LOG_INFO << location << ": " << message;
  else if (level < kError)
    LOG_WARN << location << ": " << message;
  else
    LOG_ERROR << location << ": " << message;

  if (abort) {
    LOG_ERROR << "Program abort requested by ROOT.";
    std::abort();
  }
}


// Format function for log record
void FormatRecord(boost::log::record_view const &record,
                  boost::log::formatting_ostream &stream) {
  auto const severity = boost::log::extract<Logger::SeverityLevel>(
    "Severity", record);

  if (severity) {
    switch (severity.get()) {
      case Logger::SeverityLevel::kWarning:
        stream << "[WARN] ";
        break;

      case Logger::SeverityLevel::kError:
        stream << "[ERROR] ";
        break;

      default:
        break;
    }
  }

  stream << record[boost::log::expressions::smessage];
}


// Format function for log record, which uses coloured output
void FormatRecordColour(boost::log::record_view const &record,
                        boost::log::formatting_ostream &stream) {
  auto const severity = boost::log::extract<Logger::SeverityLevel>(
    "Severity", record);
  bool needReset = false;

  if (severity) {
    switch (severity.get()) {
      case Logger::SeverityLevel::kWarning:
        stream << "\033[33m[WARN] ";  // Yellow
        needReset = true;
        break;

      case Logger::SeverityLevel::kError:
        stream << "\033[31m[ERROR] ";  // Red
        needReset = true;
        break;

      default:
        break;
    }
  }

  stream << record[boost::log::expressions::smessage];

  if (needReset)
    stream << "\033[0m";
}


Logger::~Logger() noexcept {
  // Restore the original terminate handler so that the logger is not used if
  // std::terminate is called
  std::set_terminate(original_terminate);

  // Similarly, restore original ROOT error handler
  SetErrorHandler(OriginalRootErrorHandler);
}


Logger::Logger() {
  // Construct and register a sink that writes output to stderr
  boost::shared_ptr<std::ostream> clogStream(&std::clog, boost::null_deleter());
  sink_ = boost::make_shared<decltype(sink_)::element_type>();
  sink_->locked_backend()->add_stream(clogStream);
  boost::log::core::get()->add_sink(sink_);

  // Flush associated stream after each log record
  sink_->locked_backend()->auto_flush(true);

  // Set formatter. Use coloured output if running in an interactive terminal.
  if (isatty(fileno(stderr)))
    sink_->set_formatter(&FormatRecordColour);
  else
    sink_->set_formatter(&FormatRecord);

  // By default, use the lowest severity threshold for reporting
  auto severity = boost::log::expressions::attr<SeverityLevel>("Severity");
  sink_->set_filter(severity >= SeverityLevel::kTrace);


  // Override default terminate handler function so that it is reported in the
  // log. This should be done after the logger is fully initialized.
  original_terminate = std::set_terminate(logged_terminate);

  // Override ROOT error handler
  OriginalRootErrorHandler = SetErrorHandler(LoggedRootErrorHandler);
}


void Logger::SetLevel(SeverityLevel threshold) {
  auto severity = boost::log::expressions::attr<SeverityLevel>("Severity");
  GetInstance().sink_->set_filter(severity >= threshold);
}


boost::log::basic_record_ostream<char> &operator<<(
    boost::log::basic_record_ostream<char> &stream, Logger::_TimeStamp (*)()) {
  std::time_t const currentTime = std::time(nullptr);
  std::tm const *calendarTime = std::localtime(&currentTime);
  stream << std::put_time(calendarTime, "%F %T");
  return stream;
}

