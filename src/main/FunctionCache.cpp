/* 
 * File:   FunctionCache.cpp
 * Author: janvojt
 * 
 * Created on October 29, 2014, 11:15 PM
 */

#include "FunctionCache.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

FunctionCache *FunctionCache::instance;

FunctionCache::FunctionCache(void (*activationFnc)(double *x, double *y, int layerSize), int samples) {
    this->samples = samples;
    cache = new double[samples];
    slotsPerUnit = samples / 8;
    double step = (double)8 / samples;
    
    double x = -4;
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

FunctionCache::FunctionCache(const FunctionCache& orig) {
}

FunctionCache::~FunctionCache() {
}

void FunctionCache::compute(double* x, double* y, int layerSize) {
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

void FunctionCache::init(void (*activationFnc)(double *x, double *y, int layerSize), int samples) {
    LOG()->info("Creating lookup table for activation function with %d samples.", samples);
    instance = new FunctionCache(activationFnc, samples);
}

FunctionCache* FunctionCache::getInstance() {
    return instance;
}
