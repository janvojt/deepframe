/* 
 * File:   NetworkPretrainer.h
 * Author: janvojt
 *
 * Created on November 17, 2015, 5:14 PM
 */

#ifndef NETWORKPRETRAINER_H
#define	NETWORKPRETRAINER_H

#include "../net/Network.h"
#include "../ds/LabeledDataset.h"

/**
 * Pretrains all RBM layers in the given neural network.
 * 
 * @param network
 */
class NetworkPretrainer {
public:
    NetworkPretrainer(Network *network);
    NetworkPretrainer(const NetworkPretrainer& orig);
    virtual ~NetworkPretrainer();
    
    void setPretrainEpochs(long epochs);
    long getPretrainEpochs();
    
    void pretrain(LabeledDataset *trainingSet);
    
private:
    Network *net = NULL;
    
    NetworkConfiguration *netConf = NULL;
    
    long epochs = 0;
    
    bool useGpu = false;
};

#endif	/* NETWORKPRETRAINER_H */

