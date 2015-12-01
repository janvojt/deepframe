/* 
 * File:   SimpleInputDataset.cpp
 * Author: janvojt
 * 
 * Created on November 23, 2015, 10:09 PM
 */

#include "gtest/gtest.h"

#include "common.h"

#include "util/cudaHelpers.h"

/**
 * Test binary data set creation.
 */
TEST(SumReduceTest, SumOnes) {
    
    const unsigned long arraySize = 10000;
    const unsigned int arrayMemSize = arraySize * sizeof(data_t);
    
    data_t *hArray = new data_t[arraySize];
    data_t *inArray = NULL;
    data_t *outArray = NULL;
    
    // allocate memory on device
    checkCudaErrors(cudaMalloc(&inArray, 2 * arrayMemSize));
    outArray = inArray + arraySize;
    
    // fill on host and copy to device
    std::fill_n(hArray, arraySize, 1.);
    checkCudaErrors(cudaMemcpy(inArray, hArray, arrayMemSize, cudaMemcpyHostToDevice));
    
    // sum on host
    data_t expected = 0;
    for (int i = 0; i<arraySize; i++) {
        expected += hArray[i];
    }
    
    // sum on device
    data_t actual = k_sumReduce(inArray, outArray, arraySize);
    
    // verify
    EXPECT_EQ(expected, actual);
}

/**
 * Test 1 + e^x computation.
 */
TEST(APlusExpReduceTest, ExpOnes) {
    
    const unsigned long arraySize = 10000;
    const unsigned int arrayMemSize = arraySize * sizeof(data_t);
    
    const data_t a = 1.;
    
    data_t *hArray = new data_t[arraySize];
    data_t *inArray = NULL;
    data_t *outArray = NULL;
    
    // allocate memory on device
    checkCudaErrors(cudaMalloc(&inArray, 2 * arrayMemSize));
    outArray = inArray + arraySize;
    
    // fill on host and copy to device
    std::fill_n(hArray, arraySize, 1.);
    checkCudaErrors(cudaMemcpy(inArray, hArray, arrayMemSize, cudaMemcpyHostToDevice));
    
    // sum on host
    data_t expected = 0;
    for (int i = 0; i<arraySize; i++) {
        expected += log(a + exp(hArray[i]));
    }
    
    // sum on device
    data_t actual = k_logPlusExpReduce(a, inArray, outArray, arraySize);
    
    // make approximate verification,
    // as float operations are not associative on devices
    data_t accuracy = 10.;
    EXPECT_GT(ceil(actual/accuracy), a);
    EXPECT_EQ(ceil(expected/accuracy), ceil(actual/accuracy));
}

/**
 * Test v*log(sig(pv)) + (1-v)*log(1-sig(pv)) computation.
 */
TEST(CrossEntropyReduceTest, Compute) {
    
    const unsigned long arraySize = 10000;
    const unsigned int arrayMemSize = arraySize * sizeof(data_t);
    
    data_t *hVisiblesArray = new data_t[arraySize];
    data_t *hPotentialsArray = new data_t[arraySize];
    
    data_t *dVisiblesArray = NULL;
    data_t *dPotentialsArray = NULL;
    data_t *dTempArray = NULL;
    
    // allocate memory on device
    checkCudaErrors(cudaMalloc(&dVisiblesArray, 3 * arrayMemSize));
    dPotentialsArray = dVisiblesArray + arraySize;
    dTempArray = dPotentialsArray + arraySize;
    
    // fill on host and copy to device
    std::fill_n(hVisiblesArray, arraySize, .6);
    checkCudaErrors(cudaMemcpy(dVisiblesArray, hVisiblesArray, arrayMemSize, cudaMemcpyHostToDevice));
    std::fill_n(hPotentialsArray, arraySize, .8);
    checkCudaErrors(cudaMemcpy(dPotentialsArray, hPotentialsArray, arrayMemSize, cudaMemcpyHostToDevice));
    
    // sum on host
    data_t expected = 0;
    for (int i = 0; i<arraySize; i++) {
        data_t sig = 1 / (1+exp(-hPotentialsArray[i]));
        expected += hVisiblesArray[i] * log(sig) + (1-hVisiblesArray[i]) * log(1-sig);
    }
    
    // sum on device
    data_t actual = k_crossEntropyReduce(dVisiblesArray, dPotentialsArray, dTempArray, arraySize);
    
    // make approximate verification,
    // as float operations are not associative on devices
    data_t accuracy = 10.;
    EXPECT_LT(ceil(actual/accuracy), 1.);
    EXPECT_EQ(ceil(expected/accuracy), ceil(actual/accuracy));
}

/**
 * Test sigmoid(x) computation.
 */
TEST(SigmoidTest, SigmoidHalfs) {
    
    const int segments = 4;
    const unsigned long segmentSize = 1000;
    const unsigned long arraySize = segmentSize * segments;
    const unsigned int arrayMemSize = arraySize * sizeof(data_t);
    
    const data_t a = 1.;
    
    data_t *hArray = new data_t[arraySize];
    data_t *verifyArray = new data_t[arraySize];
    data_t *inArray = NULL;
    data_t *outArray = NULL;
    
    // allocate memory on device
    checkCudaErrors(cudaMalloc(&inArray, 2 * arrayMemSize));
    outArray = inArray + arraySize;
    
    // fill on host and copy to device
    data_t val = .1;
    for (int i = 0; i<segments; i++) {
        std::fill_n(hArray + i*segmentSize, segmentSize, val);
        val += .2;
    }
    checkCudaErrors(cudaMemcpy(inArray, hArray, arrayMemSize, cudaMemcpyHostToDevice));
    
    // compute on device
    k_computeSigmoid(inArray, outArray, arraySize);
    
    // copy device results to host
    checkCudaErrors(cudaMemcpy(verifyArray, outArray, arrayMemSize, cudaMemcpyDeviceToHost));
    
    // compute on host and verify
    data_t accuracy = .001;
    for (int i = 0; i<arraySize; i++) {
        hArray[i] = 1/(1+exp(-hArray[i]));
        EXPECT_EQ(ceil(hArray[i]/accuracy), ceil(verifyArray[i]/accuracy));
    }
}
