/* 
 * File:   main.cpp
 * Author: janvojt
 *
 * Created on March 15, 2015, 4:57 PM
 */

#include <cstdlib>
#include "gtest/gtest.h"

#include "log/LoggerFactory.h"
#include "log4cpp/Category.hh"

/**
 * Represents testing environment.
 */
class Environment : public ::testing::Environment {
public:

    virtual ~Environment() {
    }
    
    /**
     * Environment setup called before any test case is run.
     */
    virtual void SetUp() {
    }
    
    /**
     * Environment tear down called after all test cases are run.
     */
    virtual void TearDown() {
    }
};

/*
 * Setup the tests.
 */
int main(int argc, char **argv) {

    Environment *env = new Environment();
    
    if (argc > 1) {
        LOG()->setPriority(log4cpp::Priority::DEBUG);
    } else {
        LOG()->setPriority(log4cpp::Priority::ERROR);
    }
    
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(env);
    int result = RUN_ALL_TESTS();
    
    return result;
}