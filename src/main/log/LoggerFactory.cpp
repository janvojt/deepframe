#include <log4cpp/Category.hh>
#include <log4cpp/Appender.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/Layout.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/Priority.hh>

#include "LoggerFactory.h"

using namespace Log;

bool LoggerFactory::isCreated = false;
log4cpp::Category *LoggerFactory::instance;

LoggerFactory::LoggerFactory() {}

log4cpp::Category* LoggerFactory::create() {
    log4cpp::Category *logger = &log4cpp::Category::getRoot();
    log4cpp::Appender *p_appender = new log4cpp::OstreamAppender("console", &std::cout);
    log4cpp::PatternLayout *layout = new log4cpp::PatternLayout();
    layout->setConversionPattern("%d{%Y-%m-%d %H:%M:%S} [%p] %c: %m%n");
    p_appender->setLayout(layout);

    logger->setPriority(log4cpp::Priority::INFO);
    logger->addAppender(p_appender);
    return logger;
}

log4cpp::Category *LoggerFactory::getLogger() {
    if (!isCreated) {
        isCreated = true;
        instance = create();
    }
    return instance;
}

void* LoggerFactory::destroy() {
    if (isCreated) {
        delete instance;
    }
}
