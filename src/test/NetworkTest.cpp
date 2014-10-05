
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

// Sets all the network weights to given value.
void setAllWeights(Network *net, double value) {
    double *weights = net->getWeights();
    long noWeights = net->getWeightsOffset(net->getConfiguration()->getLayers());
    
    std::fill_n(weights, noWeights, value);
}

// Test network neurons counters.
TEST(Network, NeuronCounters) {
    Network *net = new Network(createConf());
    
    EXPECT_EQ(5, net->getAllNeurons());
    EXPECT_EQ(2, net->getInputNeurons());
    EXPECT_EQ(1, net->getOutputNeurons());
}

// Test network input setup.
TEST(Network, InputSetTest) {
    Network *net = new Network(createConf());
    
    double input[] = {.1, .1};
    net->setInput(input);
    
    EXPECT_EQ(input[0], net->getInput()[0]);
    EXPECT_EQ(input[1], net->getInput()[1]);
}

// Test run of a simple network with no bias and weights of 1.
TEST(Network, SimpleRun) {
    
    NetworkConfiguration *conf = createConf();
    conf->setBias(false);
    conf->activationFnc = identityFunction;
    conf->dActivationFnc = dIdentityFunction;
    
    Network *net = new Network(conf);
    setAllWeights(net, 1);
    
    double input[] = {0, 0};
    
    net->setInput(input);
    net->run();
    EXPECT_EQ(0, net->getOutput()[0]);
    
    input[0] = 0;
    input[1] = 1;
    net->setInput(input);
    net->run();
    EXPECT_EQ(2, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 0;
    net->setInput(input);
    net->run();
    EXPECT_EQ(2, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 1;
    net->setInput(input);
    net->run();
    EXPECT_EQ(4, net->getOutput()[0]);
}

// Test run of a simple network with no bias and weights of 1/2.
TEST(Network, SimpleWeightTest) {
    
    NetworkConfiguration *conf = createConf();
    conf->setBias(false);
    conf->activationFnc = identityFunction;
    conf->dActivationFnc = dIdentityFunction;
    
    Network *net = new Network(conf);
    setAllWeights(net, .5);
    
    double input[] = {0, 0};
    
    net->setInput(input);
    net->run();
    EXPECT_EQ(0, net->getOutput()[0]);
    
    input[0] = 0;
    input[1] = 1;
    net->setInput(input);
    net->run();
    EXPECT_EQ(.5, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 0;
    net->setInput(input);
    net->run();
    EXPECT_EQ(.5, net->getOutput()[0]);
    
    input[0] = 1;
    input[1] = 1;
    net->setInput(input);
    net->run();
    EXPECT_EQ(1, net->getOutput()[0]);
}