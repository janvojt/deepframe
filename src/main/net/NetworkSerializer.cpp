/*
 * File:   NetworkSerializer.cpp
 * Author: janvojt
 *
 * Created on October 14, 2016, 17:14 PM
 */

#include "NetworkSerializer.h"

#include "../log/LoggerFactory.h"
#include "log4cpp/Category.hh"

#include <iostream>
#include <fstream>

using namespace std;

NetworkSerializer::NetworkSerializer() {}

NetworkSerializer::NetworkSerializer(const NetworkSerializer& orig) {}

NetworkSerializer::~NetworkSerializer() {}

NetworkSerializer* NetworkSerializer::clone() {
    return new NetworkSerializer(*this);
}

void NetworkSerializer::save(Network *net, char *filePath) {

    // open IDX file with the dataset
    ofstream fp(filePath, ios::out|ios::binary);

    if (fp.is_open()) {

        // write weights
        data_t *weights = net->getWeights();
        int weightsCount = net->getWeightsCount();
        for (int i = 0; i<weightsCount; i++) {
            fp.write((char *) weights+i, sizeof(data_t));
        }

        fp.close();

        LOG()->info("Serialized network parameters in file '%s'.", filePath);
    } else {
        LOG()->error("Cannot open file '%s' for writing.", filePath);
    }
}
