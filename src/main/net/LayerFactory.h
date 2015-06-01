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
    
    static map_type *map;
};

template<typename T>
class LayerRegister : LayerFactory {

public:
    LayerRegister(std::string const& s) {
        if (!map) {
            // TODO initialize private/protected static member "map" once.
            // But how??
            map = new map_type;
        }
        assert(map);
        getMap()->insert(std::make_pair(s, &createT<T>));
    }
};

#endif	/* LAYERFACTORY_H */
