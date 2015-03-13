/* 
 * File:   FunctionCache.cpp
 * Author: janvojt
 * 
 * Created on October 29, 2014, 11:15 PM
 */

#include "FunctionCache.h"

#include "common.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
FunctionCache<dType> *FunctionCache<dType>::instance;

template <typename dType>
FunctionCache<dType>::FunctionCache(void (*activationFnc)(dType *x, dType *y, int layerSize), int samples) {
    this->samples = samples;
    cache = new dType[samples];
    slotsPerUnit = samples / 8;
    dType step = (dType)8 / samples;
    
    dType x = -4;
    int halfSamples = samples / 2;
    for (int i = 0; i<halfSamples; i++, x+=step) {
        cache[i] = x;
    }
    // Split into 2 loops, so we can round down the negative input,
    // and round up the positive input.
    x += step;
    for (int i = halfSamples; i<samples; i++, x+=step) {
        cache[i] = x;
    }
    activationFnc(cache, cache, samples);
}

template <typename dType>
FunctionCache<dType>::FunctionCache(const FunctionCache& orig) {
}

template <typename dType>
FunctionCache<dType>::~FunctionCache() {
}

template <typename dType>
void FunctionCache<dType>::compute(dType* x, dType* y, int layerSize) {
    for (int i = 0; i < layerSize; i++) {
        int j = (*x+4) * slotsPerUnit;
        
        // check bounds
        if (j<0) {
            j = 0;
        } else if (j>=samples) {
            j = samples-1;
        }
        
        *y = cache[j];
        x++;
        y++;
    }
}

template <typename dType>
void FunctionCache<dType>::init(void (*activationFnc)(dType *x, dType *y, int layerSize), int samples) {
    LOG()->info("Creating lookup table for activation function with %d samples.", samples);
    instance = new FunctionCache(activationFnc, samples);
}

template <typename dType>
FunctionCache<dType>* FunctionCache<dType>::getInstance() {
    return instance;
}

INSTANTIATE_DATA_CLASS(FunctionCache);