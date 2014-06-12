/* 
 * File:   Logger.h
 * Author: janvojt
 *
 * Created on June 11, 2014, 9:41 PM
 */

#ifndef LOGGER_H
#define	LOGGER_H

//#define LOG(level, msg) Log::Logger::getLogger().log(level, msg)
#define LOG() Log::Logger::getLogger()

//#define LOG() log4cpp::Category::getRoot()

namespace log4cpp {
    class Category;
}

namespace Log {
    /// Logging levels used by pix. Follows the same as for syslog, taken from
    /// RFC 5424. Comments added for ease of reading.
    /// @see http://en.wikipedia.org/wiki/Syslog.

    enum LogLevel {
        EMERG, // System is unusable (e.g. multiple parts down)
        ALERT, // System is unusable (e.g. single part down)
        CRIT, // Failure in non-primary system (e.g. backup site down)
        ERROR, // Non-urgent failures; relay to developers
        WARN, // Not an error, but indicates error will occur if nothing done.
        NOTICE, // Events that are unusual, but not error conditions.
        INFO, // Normal operational messages. No action required.
        DEBUG, // Information useful during development for debugging.
        NOTSET
    };

    class Logger {
    public:
        static bool isCreated;
        static Log::Logger* instance;
        static log4cpp::Category& getLogger();
        void log(Log::LogLevel level, const std::string& msg);

    private:
        Logger();
        Logger(const Logger&);
        Logger(Logger&);
        Logger& operator=(const Logger&);
        Logger& operator=(Logger&);

    private:
        log4cpp::Category& m_cpp_logger;
    };

}

#endif	/* LOGGER_H */

