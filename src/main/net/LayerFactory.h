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

/** Factory for creating layers. */
class LayerFactory {

public:
    
    /** The layer registry indexed by layer names. */
    typedef std::map<std::string, Layer*(*)()> map_type;
    
    /** Creates a new instance of a layer specified by its name. */
    static Layer *createInstance(std::string const& s);
    
protected:
    /**
     * @return the layer registry
     */
    static map_type *getMap();
};

/**
 * Class used for registering the layer types. This is useful
 * when we want to be able to encapsulate all the layer logic
 * including layer registration within the framework.
 * 
 * @param s layer type
 */
template<typename T>
class LayerRegister : LayerFactory {

public:
    
    /**
     * Registers new layer type by its name.
     * 
     * @param s layer name
     */
    LayerRegister(std::string const& s) {
        getMap()->insert(std::make_pair(s, &createT<T>));
    }
};

// USAGE EXAMPLE: static LayerRegister<FullyConnectedLayer> reg("FullyConnected");

#endif	/* LAYERFACTORY_H */
