/* 
 * File:   NetworkPretrainer.cpp
 * Author: janvojt
 * 
 * Created on November 17, 2015, 5:14 PM
 */

#include "NetworkPretrainer.h"

#include "../common.h"
#include "../log/LoggerFactory.h"
#include "../util/cudaHelpers.h"
#include "log4cpp/Category.hh"

NetworkPretrainer::NetworkPretrainer(Network *network) {
    net = network;
    netConf = net->getConfiguration();
    useGpu = netConf->getUseGpu();
}

NetworkPretrainer::NetworkPretrainer(const NetworkPretrainer& orig) {
}

NetworkPretrainer::~NetworkPretrainer() {
}

void NetworkPretrainer::pretrain(LabeledDataset *trainingSet) {
    
    LOG()->info("Started pretraining in %d epochs...", epochs);
    
    // input is never trainable -> start from 1
    int noLayers = netConf->getLayers();
    for (int i = 1; i<noLayers; i++) {
        
        Layer* layer = net->getLayer(i);
        if (layer->isPretrainable()) {
            
            for (int e = 0; e<epochs; e++) {

                LOG()->info("Pretraining layer %d in epoch %d.", i, e);

                // iterate over all training patterns and pretrain them
                trainingSet->reset();
                while (trainingSet->hasNext()) {
                    data_t* input = trainingSet->next();

                    // do a forward run till the layer we are pretraining
                    net->setInput(input);
                    for (int j = 1; j<i; j++) {
                        Layer* l = net->getLayer(j);
                        if (useGpu) {
                            l->forwardGpu();
                        } else {
                            l->forwardCpu();
                        }
                    }

                    // pretrain the layer
                    if (useGpu) {
                        
                        // sample binary states from input on GPU
                        Layer* prevLayer = net->getLayer(i-1);
                        k_generateUniform(*layer->curandGen, layer->randomData, prevLayer->getOutputsCount());
                        k_flattenToCoinFlip(prevLayer->getOutputs(), prevLayer->getOutputsCount());
                        
                        layer->pretrainGpu();

                    } else {

                        // TODO sample binary states from input on CPU
                        layer->pretrainCpu();
                    }
                }
            }
        }
    }
    
    LOG()->info("Finished pretraining.");
}


long NetworkPretrainer::getPretrainEpochs() {
    return epochs;
}

void NetworkPretrainer::setPretrainEpochs(long epochs) {
    this->epochs = epochs;
}
