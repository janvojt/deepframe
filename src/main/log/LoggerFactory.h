/* 
 * File:   Logger.h
 * Author: janvojt
 *
 * Created on June 11, 2014, 9:41 PM
 */

#ifndef LOGGER_H
#define	LOGGER_H

#define LOG() Log::LoggerFactory::getLogger()

namespace log4cpp {
    class Category;
}

namespace Log {

    class LoggerFactory {
        
    public:
        // Get a singleton instance of logger.
        static log4cpp::Category *getLogger();
        // Factory for creating new logger instance. Caller is responsible for
        // correctly freeing the memory hold by the created logger.
        static log4cpp::Category *create();
        // Correctly destroys the logger singleton instance by freeing the
        // memory hold by the instance.
        static void *destroy();

    private:
        static bool isCreated;
        static log4cpp::Category *instance;
        LoggerFactory();
        LoggerFactory(const LoggerFactory&);
        LoggerFactory(LoggerFactory&);
        LoggerFactory& operator=(const LoggerFactory&);
        LoggerFactory& operator=(LoggerFactory&);

    };

}

#endif	/* LOGGER_H */

