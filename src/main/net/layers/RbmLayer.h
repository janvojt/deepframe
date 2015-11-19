/* 
 * File:   RbmLayer.h
 * Author: janvojt
 *
 * Created on November 11, 2015, 8:43 PM
 */

#ifndef RBMLAYER_H
#define	RBMLAYER_H

#include "../Layer.h"
#include "../../common.h"
#include <string>

struct RbmConfig {
    // number of hidden neurons
    int outputSize;
    // use persistent CD?
    bool isPersistent;
    // use bias?
    bool useBias;
    // number of steps in Gibbs sampling
    int gibbsSteps;
};

class RbmLayer : public Layer {
public:
    RbmLayer();
    RbmLayer(const RbmLayer& orig);
    virtual ~RbmLayer();

    void forwardCpu();
    void forwardGpu();
    
    void backwardCpu();
    void backwardGpu();
    
    virtual bool isPretrainable();
    virtual void pretrainCpu();
    virtual void pretrainGpu();

    virtual void backwardLastCpu(data_t* expectedOutput);
    virtual void backwardLastGpu(data_t* expectedOutput);
    
    RbmConfig getConfig();
    
protected:
    
    void setup(string confString);
    
private:

    void propagateForwardCpu(data_t *visibles, data_t *potentials, data_t *hiddens);
    void propagateForwardGpu(data_t *visibles, data_t *potentials, data_t *hiddens);
    
    void propagateBackwardCpu(data_t *hiddens, data_t *potentials, data_t *visibles);
    void propagateBackwardGpu(data_t *hiddens, data_t *potentials, data_t *visibles);
    
    void sample_vh_gpu();
    void sample_hv_gpu();
    
    void gibbs_hvh(int steps);
    void gibbs_vhv(int steps);
    
    data_t freeEnergy();
    void costUpdates();
    
    void processConfString(string confString);
    
    RbmConfig conf;

    int inputSize;
    
    /** Sampled inputs. */
    data_t *sInputs;
    /** Sampled outputs. */
    data_t *sOutputs;
    
    /** Original potentials for visible neurons. */
    data_t *ovPotentials;
    /** Sampled potentials for visible neurons. */
    data_t *svPotentials;
    
    /** Original potentials for hidden neurons. */
    data_t *ohPotentials;
    /** Sampled potentials for hidden neurons. */
    data_t *shPotentials;
    
    data_t *randomData;
    
    bool samplesInitialized = false;
};

#endif	/* RBMLAYER_H */

