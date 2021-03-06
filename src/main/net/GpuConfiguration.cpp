/* 
 * File:   GpuConfiguration.cpp
 * Author: janvojt
 * 
 * Created on November 8, 2014, 3:17 PM
 */

#include "GpuConfiguration.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

GpuConfiguration::GpuConfiguration() {
}

GpuConfiguration::GpuConfiguration(const GpuConfiguration& orig) {
}

GpuConfiguration::~GpuConfiguration() {
}

GpuConfiguration *GpuConfiguration::create() {

    // By default, we use device 0
    int devID = 0;
    cudaSetDevice(devID);

    cudaError_t error;
    cudaDeviceProp *deviceProp = new cudaDeviceProp();
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess) {
        LOG()->error("Call cudaGetDevice returned error code %d. (%s:%d)", error, __FILE__, __LINE__);
        return NULL;
    }

    error = cudaGetDeviceProperties(deviceProp, devID);

    if (error != cudaSuccess) {
        LOG()->error("Call cudaGetDeviceProperties returned error code %d. (%s:%d)\n", error, __FILE__, __LINE__);
        return NULL;
    }

    if (deviceProp->computeMode == cudaComputeModeProhibited) {
        LOG()->error("Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().");
        return NULL;
    }
    
    LOG()->info("GPU Device %d: \"%s\" with compute capability %d.%d.", devID, deviceProp->name, deviceProp->major, deviceProp->minor);
    GpuConfiguration *gpuConf = new GpuConfiguration();
    gpuConf->setDeviceProp(deviceProp);

    // Use a larger block size for Fermi and above
    gpuConf->setBlockSize((deviceProp->major < 2) ? 16 : 32);
    
    return gpuConf;
}

void GpuConfiguration::setDeviceProp(cudaDeviceProp *devideProp) {
    this->deviceProp = devideProp;
}

cudaDeviceProp* GpuConfiguration::getDeviceProp() {
    return deviceProp;
}

void GpuConfiguration::setRandGen(curandGenerator_t* randGen) {
    this->randGen = randGen;
}

curandGenerator_t* GpuConfiguration::getRandGen() {
    return randGen;
}

void GpuConfiguration::setBlockSize(int blockSize) {
    this->blockSize = blockSize;
}

int GpuConfiguration::getBlockSize() {
    return blockSize;
}
