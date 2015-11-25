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
    data_t accuracy = 10/arraySize;
    EXPECT_GT(a, ceil(actual*accuracy));
    EXPECT_EQ(ceil(expected*accuracy), ceil(actual*accuracy));
}
