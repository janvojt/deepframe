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
    
    /**
     * Initializes the pretrainer.
     * 
     * @param network neural network to be pretrained
     */
    NetworkPretrainer(Network *network);
    NetworkPretrainer(const NetworkPretrainer& orig);
    virtual ~NetworkPretrainer();
    
    void setPretrainEpochs(long epochs);
    long getPretrainEpochs();
    
    /**
     * Pretrains the network on the given training set.
     * 
     * @param trainingSet
     */
    void pretrain(LabeledDataset *trainingSet);
    
private:
    
    /** The neural network to pretrain. */
    Network *net = NULL;
    
    NetworkConfiguration *netConf = NULL;
    
    /** Number of pretraining epochs. */
    long epochs = 0;
    
    /** Use GPU or CPU? */
    bool useGpu = false;
};

#endif	/* NETWORKPRETRAINER_H */

