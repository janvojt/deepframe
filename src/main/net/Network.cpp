/* 
 * File:   Network.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "Network.h"

#include <cstring>
#include <string>
#include <stdlib.h>
#include <iostream>

#include "../common.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

template <typename dType>
Network<dType>::Network(NetworkConfiguration<dType> *conf) {
    this->conf = conf;
    this->noLayers = conf->getLayers();
    
    if (conf->getLayerConf() != NULL) {
        LOG()->info("Initializing network with layer configuration of (%s).", conf->getLayerConf());
    }
}

template <typename dType>
Network<dType>::Network(const Network& orig) {
}

template <typename dType>
Network<dType>::~Network() {
}

template <typename dType>
NetworkConfiguration<dType>* Network<dType>::getConfiguration() {
    return this->conf;
}

template <typename dType>
int Network<dType>::getInputNeurons() {
    return this->conf->getNeurons(0);
}

template <typename dType>
int Network<dType>::getOutputNeurons() {
    return this->conf->getNeurons(this->noLayers-1);
}

INSTANTIATE_DATA_CLASS(Network);