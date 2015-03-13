/* 
 * File:   FunctionCache.h
 * Author: janvojt
 *
 * Created on October 29, 2014, 11:15 PM
 */

#ifndef FUNCTIONCACHE_H
#define	FUNCTIONCACHE_H

template <typename dType>
class FunctionCache {
public:
    FunctionCache(const FunctionCache& orig);
    virtual ~FunctionCache();
    // Looks up value in the lookup table for each input in array x,
    // and sets the value into output array y.
    void compute(dType *x, dType *y, int layerSize);
    // Singleton initializer.
    static void init(void (*activationFnc)(dType *x, dType *y, int layerSize), int samples);
    // Provides external access to the pointer to the singleton instance.
    static FunctionCache *getInstance();
private:
    // Private constructor so cache is used as a singleton only.
    FunctionCache(void (*activationFnc)(dType *x, dType *y, int layerSize), int samples);
    // Pointer to singleton instance.
    static FunctionCache *instance;
    // Lookup table.
    dType *cache;
    // Number of samples in unit length of 1 on the x-axis.
    int slotsPerUnit;
    // Total number of samples.
    int samples;
};

#endif	/* FUNCTIONCACHE_H */

