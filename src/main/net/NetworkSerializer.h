/*
 * File:   NetworkSerializer.h
 * Author: janvojt
 *
 * Created on October 14, 2016, 17:14 PM
 */

#ifndef NETWORKSERIALIZER_H
#define	NETWORKSERIALIZER_H

#include "../net/Network.h"

class NetworkSerializer {
public:
    NetworkSerializer();
    NetworkSerializer(const NetworkSerializer& orig);
    virtual ~NetworkSerializer();

    /**
     * Clones the serializer.
     *
     * @return serializer copy
     */
    NetworkSerializer* clone();

    void save(Network *net, char* filePath);

private:
};

#endif	/* NETWORKSERIALIZER_H */

