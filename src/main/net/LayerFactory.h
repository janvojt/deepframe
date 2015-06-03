/* 
 * File:   LayerFactory.h
 * Author: janvojt
 *
 * Created on May 26, 2015, 10:19 PM
 */

#ifndef LAYERFACTORY_H
#define	LAYERFACTORY_H

#include <cstdlib>
#include "map"
#include "../common.h"
#include "Layer.h"
#include <assert.h>

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template<typename T>
Layer * createT() { return new T; }

class LayerFactory {

public:
    
    typedef std::map<std::string, Layer*(*)()> map_type;
    
    static Layer *createInstance(std::string const& s);
    
protected:
    static map_type *getMap();
};

template<typename T>
class LayerRegister : LayerFactory {

public:
    LayerRegister(std::string const& s) {
        getMap()->insert(std::make_pair(s, &createT<T>));
    }
};

#endif	/* LAYERFACTORY_H */
