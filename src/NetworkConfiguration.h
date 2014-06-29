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
    // Returns number of neurons indexed from 1.
    int getNeurons(int layer);
    // Sets number of neurons in given layer, layer being indexed from 1.
    void setNeurons(int layer, int neurons);
private:
    void initConf();
    // number of layers in a network
    int layers;
    // number of neurons in each network layer
    int *neuronConf;
};

#endif	/* NETWORKCONFIGURATION_H */

