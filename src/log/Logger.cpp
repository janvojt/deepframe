#include <log4cpp/Category.hh>
#include <log4cpp/Appender.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/Layout.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/Priority.hh>

#include "Logger.h"

bool Log::Logger::isCreated = false;

Log::Logger::Logger()
: m_cpp_logger(log4cpp::Category::getRoot()) {
    // Creates a simple log4cpp logger.
    log4cpp::Appender* p_appender = new log4cpp::OstreamAppender("console", &std::cout);
    log4cpp::PatternLayout* layout = new log4cpp::PatternLayout();
    layout->setConversionPattern("%d{%Y-%m-%d %H:%M:%S} [%p] %c: %m%n");
    p_appender->setLayout(layout);

    m_cpp_logger.setPriority(log4cpp::Priority::DEBUG);
    m_cpp_logger.addAppender(p_appender);
}

Log::Logger* Log::Logger::instance;

log4cpp::Category& Log::Logger::getLogger() {
    if (!isCreated) {
        isCreated = true;
        instance = new Log::Logger();
    }
    return instance->m_cpp_logger;
}

void Log::Logger::log(Log::LogLevel level, const std::string& msg) {
    // Translate pix logging level to cpp logging level
    log4cpp::Priority::PriorityLevel cpp_level = log4cpp::Priority::NOTSET;
    switch (level) {
        case Log::NOTSET: // allow fall through
        case Log::EMERG: cpp_level = log4cpp::Priority::EMERG;
            break;
        case Log::ALERT: cpp_level = log4cpp::Priority::ALERT;
            break;
        case Log::CRIT: cpp_level = log4cpp::Priority::CRIT;
            break;
        case Log::ERROR: cpp_level = log4cpp::Priority::ERROR;
            break;
        case Log::WARN: cpp_level = log4cpp::Priority::WARN;
            break;
        case Log::NOTICE: cpp_level = log4cpp::Priority::NOTICE;
            break;
        case Log::INFO: cpp_level = log4cpp::Priority::INFO;
            break;
        case Log::DEBUG: cpp_level = log4cpp::Priority::DEBUG;
            break;
        default: // LCOV_EXCL_LINE
            assert(false); // LCOV_EXCL_LINE
    };
    assert(cpp_level != log4cpp::Priority::NOTSET);

    // Log message
    m_cpp_logger << cpp_level << msg;
}