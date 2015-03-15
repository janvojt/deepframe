
#include "gtest/gtest.h"
#include "net/Network.h"
#include "net/CpuNetwork.h"
#include "net/NetworkConfiguration.h"
#include "activationFunctions.h"

#define DATA_TYPE float


template <typename dType>
NetworkConfiguration<dType>* createConf() {
    
    NetworkConfiguration<dType> *netConf = new NetworkConfiguration<dType>();
    netConf->setLayers(3);
    netConf->setNeurons(0, 2);
    netConf->setNeurons(1, 2);
    netConf->setNeurons(2, 1);
    netConf->activationFnc = sigmoidFunction;
    netConf->dActivationFnc = dSigmoidFunction;
    netConf->setBias(true);
    
    return netConf;
}
template NetworkConfiguration<DATA_TYPE>* createConf<DATA_TYPE>();

// Sets all the network weights to given value.
template <typename dType>
void setAllWeights(Network<dType> *net, dType value) {
    dType *weights = net->getWeights();
    long noWeights = net->getWeightsOffset(net->getConfiguration()->getLayers());
    
    std::fill_n(weights, noWeights, value);
}
template void setAllWeights<DATA_TYPE>(Network<DATA_TYPE>*, DATA_TYPE);


// Test network neurons counters.
TEST(NetworkTest, NeuronCounters) {
    Network<DATA_TYPE> *net = new CpuNetwork<DATA_TYPE>(createConf<DATA_TYPE>());
    
    EXPECT_EQ(5, net->getAllNeurons());
    EXPECT_EQ(2, net->getInputNeurons());
    EXPECT_EQ(1, net->getOutputNeurons());
}

// Test network input setup.
TEST(NetworkTest, InputSetTest) {
    Network<DATA_TYPE> *net = new CpuNetwork<DATA_TYPE>(createConf<DATA_TYPE>());
    
    DATA_TYPE input[] = {.1, .1};
    net->setInput(input);
    
    EXPECT_EQ(input[0], net->getInput()[0]);
    EXPECT_EQ(input[1], net->getInput()[1]);
}

// Test run of a simple network with no bias and weights of 1.
TEST(NetworkTest, SimpleRun) {
    
    NetworkConfiguration<DATA_TYPE> *conf = createConf<DATA_TYPE>();
    conf->setBias(false);
    conf->activationFnc = identityFunction;
    conf->dActivationFnc = dIdentityFunction;
    
    Network<DATA_TYPE> *net = new CpuNetwork<DATA_TYPE>(conf);
    setAllWeights<DATA_TYPE>(net, 1);
    
    DATA_TYPE input[] = {0, 0};
    
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
TEST(NetworkTest, SimpleWeightTest) {
    
    NetworkConfiguration<DATA_TYPE> *conf = createConf<DATA_TYPE>();
    conf->setBias(false);
    conf->activationFnc = identityFunction;
    conf->dActivationFnc = dIdentityFunction;
    
    Network<DATA_TYPE> *net = new CpuNetwork<DATA_TYPE>(conf);
    setAllWeights<DATA_TYPE>(net, .5);
    
    DATA_TYPE input[] = {0, 0};
    
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

// Test network weight offsets.
TEST(NetworkTest, WeightsOffsetTest) {
    Network<DATA_TYPE> *net = new CpuNetwork<DATA_TYPE>(createConf<DATA_TYPE>());
    
    EXPECT_EQ(0, net->getWeightsOffset(0));
    EXPECT_EQ(2, net->getWeightsOffset(1));
    EXPECT_EQ(6, net->getWeightsOffset(2));
    EXPECT_EQ(8, net->getWeightsOffset(3));
}

// Test neuron input offsets.
TEST(NetworkTest, NeuronInputOffsetTest) {
    Network<DATA_TYPE> *net = new CpuNetwork<DATA_TYPE>(createConf<DATA_TYPE>());
    
    EXPECT_EQ(0, net->getInputOffset(0));
    EXPECT_EQ(2, net->getInputOffset(1));
    EXPECT_EQ(4, net->getInputOffset(2));
    EXPECT_EQ(5, net->getInputOffset(3));
}