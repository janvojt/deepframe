/* 
 * File:   GpuConfiguration.h
 * Author: janvojt
 *
 * Created on November 8, 2014, 3:17 PM
 */

#ifndef GPUCONFIGURATION_H
#define	GPUCONFIGURATION_H

#include <cuda_runtime.h>

class GpuConfiguration {
public:
    GpuConfiguration();
    GpuConfiguration(const GpuConfiguration& orig);
    virtual ~GpuConfiguration();
    static GpuConfiguration *create();
    cudaDeviceProp *getDeviceProp();
    void setDeviceProp(cudaDeviceProp *devideProp);
    int getBlockSize();
    void setBlockSize(int blockSize);
private:
    cudaDeviceProp *deviceProp;
    int blockSize;
};

#endif	/* GPUCONFIGURATION_H */

