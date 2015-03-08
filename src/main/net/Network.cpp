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

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

Network::Network(NetworkConfiguration *conf) {
    this->conf = conf;
    this->noLayers = conf->getLayers();
    
    LOG()->info("Initializing network with layer configuration of (%s).", conf->getLayerConf());
}

Network::Network(const Network& orig) {
}

Network::~Network() {
}

NetworkConfiguration* Network::getConfiguration() {
    return conf;
}

int Network::getInputNeurons() {
    return conf->getNeurons(0);
}

int Network::getOutputNeurons() {
    return conf->getNeurons(noLayers-1);
}
