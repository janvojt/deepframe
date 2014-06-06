/* 
 * File:   Network.cpp
 * Author: janvojt
 * 
 * Created on May 30, 2014, 12:17 AM
 */

#include "Network.h"

Network::Network(NetworkConfiguration conf) {
    this->conf = conf;
}

Network::Network(const Network& orig) {
}

Network::~Network() {
}

NetworkConfiguration Network::getConfiguration() {
    return conf;
}
