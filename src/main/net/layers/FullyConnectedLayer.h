/* 
 * File:   FullyConnectedLayer.h
 * Author: janvojt
 *
 * Created on May 17, 2015, 12:55 AM
 */

#ifndef FULLYCONNECTEDLAYER_H
#define	FULLYCONNECTEDLAYER_H

#include "../Layer.h"

#include <string>

#include "../../common.h"

using namespace std;

struct FullyConnectedConfig {
    
    int outputSize;
    
    bool useBias;
    
};

class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer();
    FullyConnectedLayer(const FullyConnectedLayer& orig);
    virtual ~FullyConnectedLayer();

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();
    
    virtual void backwardLastCpu(data_t* expectedOutput);
    virtual void backwardLastGpu(data_t* expectedOutput);
    
protected:
    
    void setup(string confString);

private:
    
    void processConfString(string confString);
    
    void computeTotalDiffs();
    
    FullyConnectedConfig conf;
};

#endif	/* FULLYCONNECTEDLAYER_H */

