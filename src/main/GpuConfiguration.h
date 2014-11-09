/* 
 * File:   GpuConfiguration.h
 * Author: janvojt
 *
 * Created on November 8, 2014, 3:17 PM
 */

#ifndef GPUCONFIGURATION_H
#define	GPUCONFIGURATION_H

#include <cuda_runtime.h>
#include <curand.h>

class GpuConfiguration {
public:
    GpuConfiguration();
    GpuConfiguration(const GpuConfiguration& orig);
    virtual ~GpuConfiguration();
    static GpuConfiguration *create();
    cudaDeviceProp *getDeviceProp();
    void setDeviceProp(cudaDeviceProp *devideProp);
    curandGenerator_t *getRandGen();
    void setRandGen(curandGenerator_t *randGen);
    int getBlockSize();
    void setBlockSize(int blockSize);
private:
    cudaDeviceProp *deviceProp;
    curandGenerator_t *randGen;
    int blockSize;
};

#endif	/* GPUCONFIGURATION_H */

