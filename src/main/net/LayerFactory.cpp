/* 
 * File:   LayerFactory.cpp
 * Author: janvojt
 * 
 * Created on May 31, 2015, 8:03 PM
 */

#include "LayerFactory.h"
#include "layers/FullyConnectedLayer.h"
#include <stdio.h>

using namespace std;

LayerFactory::map_type *LayerFactory::getMap() {
    // never deleted. (exist until program termination)
    // because we can't guarantee correct destruction order
    static map_type *map = new map_type;
    return map;
}

Layer *LayerFactory::createInstance(std::string const& s) {
    typename map_type::iterator it = getMap()->find(s);
    if(it == getMap()->end()) {
        return 0;
    }
    return it->second();
}
