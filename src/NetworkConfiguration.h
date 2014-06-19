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
    int getLayers();
    void setLayers(int layers);
    int getNeurons(int layer);
    void setNeurons(int layer, int neurons);
    void setNeuronConf(int conf, int layers);
private:
    void initConf();
    /* number of layers in a network */
    int layers;
    /* number of neurons in each network layer */
    int* neuronConf;
};

#endif	/* NETWORKCONFIGURATION_H */

