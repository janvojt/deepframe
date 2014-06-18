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
        static log4cpp::Category* getLogger();
        static log4cpp::Category* create();

    private:
        static bool isCreated;
        static log4cpp::Category* instance;
        LoggerFactory();
        LoggerFactory(const LoggerFactory&);
        LoggerFactory(LoggerFactory&);
        LoggerFactory& operator=(const LoggerFactory&);
        LoggerFactory& operator=(LoggerFactory&);

    private:

    };

}

#endif	/* LOGGER_H */

