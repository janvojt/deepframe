/* 
 * File:   SimpleInputDataset.cpp
 * Author: janvojt
 * 
 * Created on October 15, 2014, 10:09 PM
 */

#include "gtest/gtest.h"

#include "ds/SimpleInputDataset.h"

#define DATA_TYPE float

// Test binary data set creation.
TEST(SimpleInputDataset, BinaryDatasetCreation) {
    SimpleInputDataset<DATA_TYPE> *ds = new SimpleInputDataset<DATA_TYPE>(2, 4);
    ds->addInput((const DATA_TYPE[2]){0, 0});
    ds->addInput((const DATA_TYPE[2]){0, 1});
    ds->addInput((const DATA_TYPE[2]){1, 0});
    ds->addInput((const DATA_TYPE[2]){1, 1});
    
    for (int i = 0; i<2; i++) {
        for (int j = 0; j<2; j++) {
            EXPECT_TRUE(ds->hasNext());
            DATA_TYPE *input = ds->next();
            EXPECT_EQ(i, input[0]);
            EXPECT_EQ(j, input[1]);
        }
    }
    EXPECT_TRUE(!ds->hasNext());
}
