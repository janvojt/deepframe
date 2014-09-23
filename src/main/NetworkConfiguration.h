/* 
 * File:   NetworkConfiguration.h
 * Author: Jan Vojt
 *
 * Created on June 5, 2014, 8:16 PM
 */

#ifndef NETWORKCONFIGURATION_H
#define	NETWORKCONFIGURATION_H

class NetworkConfiguration {
public:
    NetworkConfiguration();
    NetworkConfiguration(const NetworkConfiguration& orig);
    virtual ~NetworkConfiguration();
    // Returns number of layers in the network
    // including input and output layer.
    int getLayers();
    void setLayers(int layers);
    // Returns number of neurons indexed from zero.
    int getNeurons(int layer);
    // Sets number of neurons in given layer, layer being indexed from zero.
    void setNeurons(int layer, int neurons);
    // Enables or disables network bias.
    void setBias(bool enabled);
    // Returns whether bias is enabled.
    bool getBias();
    // Pointer to activation function normalizing the neurons potential.
    // Input potential is preserved and the normalized value
    // is put into the target array. It is also possible to provide
    // the same pointer for input and target for in-place computation
    // saving some memory.
    void (*activationFnc)(double *inputPtr, double *targetPtr, int layerSize);
    // Derivative of activation function.
    void (*dActivationFnc)(double *inputPtr, double *targetPtr, int layerSize);
private:
    void initConf();
    // number of layers in a network
    int layers;
    // number of neurons in each network layer
    int *neuronConf;
    // flag determining whether the network uses bias, true by default
    bool bias;
};

#endif	/* NETWORKCONFIGURATION_H */

