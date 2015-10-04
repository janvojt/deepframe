
#include <string.h>
#include "gtest/gtest.h"
#include "common.h"
#include "net/Network.h"
#include "net/CpuNetwork.h"
#include "net/NetworkConfiguration.h"
#include "activationFunctions.h"
#include "util/cpuDebugHelpers.h"


NetworkConfiguration* createConf(bool useBias) {
    
    NetworkConfiguration *netConf = new NetworkConfiguration();
    
    netConf->setBias(useBias);
    netConf->activationFnc = sigmoidFunction;
    netConf->dActivationFnc = dSigmoidFunction;
    
    char *layerConf = new char[5];
    strcpy(layerConf, "2,2,1");
    netConf->parseLayerConf(layerConf);
    
    return netConf;
}

// Sets all the network weights to given value.
void setAllWeights(Network *net, data_t value) {
    data_t *weights = net->getWeights();
    long noWeights = net->getWeightsCount();
    
    std::fill_n(weights, noWeights, value);
}


// Test network neurons counters.
TEST(NetworkTest, NeuronCounters) {
    Network *net = new CpuNetwork(createConf(true));
    net->setup();
    
    EXPECT_EQ(5, net->getInputsCount());
    EXPECT_EQ(2, net->getInputNeurons());
    EXPECT_EQ(1, net->getOutputNeurons());
}

// Test network input setup.
TEST(NetworkTest, InputSetTest) {
    Network *net = new CpuNetwork(createConf(true));
    net->setup();
    
    data_t input[] = {.1, .1};
    net->setInput(input);
    
    EXPECT_EQ(input[0], net->getInput()[0]);
    EXPECT_EQ(input[1], net->getInput()[1]);
}

// Test run of a simple network with no bias and weights of 1.
TEST(NetworkTest, SimpleRun) {
    
    NetworkConfiguration *conf = createConf(false);
    conf->activationFnc = identityFunction;
    conf->dActivationFnc = dIdentityFunction;
    
    Network *net = new CpuNetwork(conf);
    net->setup();
    setAllWeights(net, 1);
    
    data_t input[] = {0, 0};
    
    net->setInput(input);
    net->forward();
    EXPECT_EQ(0, net->getOutput()[0]);
    
    input[0] = 0;
    input[1] = 1;
    net->setInput(input);
    net->forward();
    EXPECT_EQ(2, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 0;
    net->setInput(input);
    net->forward();
    EXPECT_EQ(2, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 1;
    net->setInput(input);
    net->forward();
    EXPECT_EQ(4, net->getOutput()[0]);
}

// Test run of a simple network with no bias and weights of 1/2.
TEST(NetworkTest, SimpleWeightTest) {
    
    NetworkConfiguration *conf = createConf(false);
    conf->activationFnc = identityFunction;
    conf->dActivationFnc = dIdentityFunction;
    
    Network *net = new CpuNetwork(conf);
    net->setup();
    setAllWeights(net, .5);
    
    data_t input[] = {0, 0};
    
    net->setInput(input);
    net->forward();
    EXPECT_EQ(0, net->getOutput()[0]);
    
    input[0] = 0;
    input[1] = 1;
    net->setInput(input);
    net->forward();
    EXPECT_EQ(.5, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 0;
    net->setInput(input);
    net->forward();
    EXPECT_EQ(.5, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 1;
    net->setInput(input);
    net->forward();
    EXPECT_EQ(1, net->getOutput()[0]);
}