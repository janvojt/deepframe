/* 
 * File:   Network.h
 * Author: janvojt
 *
 * Created on May 30, 2014, 12:17 AM
 */

#ifndef NETWORK_H
#define	NETWORK_H

#include "NetworkConfiguration.h"


class Network {
public:
    Network(NetworkConfiguration conf);
    Network(const Network& orig);
    virtual ~Network();
    NetworkConfiguration getConfiguration();
private:
    int noLayers;
    NetworkConfiguration conf;
};

#endif	/* NETWORK_H */

