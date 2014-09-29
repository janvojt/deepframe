
#include "gtest/gtest.h"
#include "Network.h"
#include "NetworkConfiguration.h"
#include "activationFunctions.h"


NetworkConfiguration* createConf() {
    
    NetworkConfiguration *netConf = new NetworkConfiguration();
    netConf->setLayers(3);
    netConf->setNeurons(0, 2);
    netConf->setNeurons(1, 2);
    netConf->setNeurons(2, 1);
    netConf->activationFnc = sigmoidFunction;
    netConf->dActivationFnc = dSigmoidFunction;
    netConf->setBias(true);
    
    return netConf;
}

// Test network construction
TEST(Network, ConstructorFromConfig) {
    Network *net = new Network(createConf());
    
    EXPECT_EQ(5, net->getAllNeurons());
}
