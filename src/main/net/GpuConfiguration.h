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

/**
 * Represents the configuration for CUDA capable device.
 */
class GpuConfiguration {
public:
    GpuConfiguration();
    GpuConfiguration(const GpuConfiguration& orig);
    virtual ~GpuConfiguration();
    
    /**
     * Initializes GPU configuration by reading the properties of available CUDA capable device.
     * This is a static factory method, avoid intializing the GPU configuration object manually.
     * 
     * @return initialized GPU configuration
     */
    static GpuConfiguration *create();
    
    /**
     * @return the device properties
     */
    cudaDeviceProp *getDeviceProp();
    
    /**
     * @param devideProp the device properties
     */
    void setDeviceProp(cudaDeviceProp *devideProp);
    
    /**
     * @return properly initialized and seeded random generator
     */
    curandGenerator_t *getRandGen();
    
    /**
     * @param randGen the random generator to be used by the network
     */
    void setRandGen(curandGenerator_t *randGen);
    
    /**
     * @return the warp size supported by the CUDA streaming multiprocessor
     */
    int getBlockSize();
    
    /**
     * All threads in a warp execute concurrently. Usual warp size on contemporary HW is 32.
     * 
     * @param blockSize the warp size supported by the CUDA streaming multiprocessor
     */
    void setBlockSize(int blockSize);
    
private:
    
    /** Pointer to the CUDA capable device properties. */
    cudaDeviceProp *deviceProp;
    
    /** Pointer to the random generator. */
    curandGenerator_t *randGen;
    
    /** Block/warp size. */
    int blockSize;
};

#endif	/* GPUCONFIGURATION_H */

