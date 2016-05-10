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

/** Holds the configuration for RBM layer. */
struct RbmConfig {
    
    /** number of hidden neurons */
    int outputSize;
    /** use persistent CD? */
    bool isPersistent;
    /** use bias? */
    bool useBias;
     /** number of steps in Gibbs sampling */
    int gibbsSteps;
};

/**
 * Represents RBM layer as a building block of Deep Belief Network.
 */
class RbmLayer : public Layer {
public:
    RbmLayer();
    RbmLayer(const RbmLayer& orig);
    virtual ~RbmLayer();

    /**
     * Performs the forward pass during training and testing on CPU.
     */
    void forwardCpu();
    
    /**
     * Performs the forward pass during training and testing on GPU.
     */
    void forwardGpu();
    
    /**
     * Performs the backward pass during training on CPU.
     */
    void backwardCpu();
    
    /**
     * Performs the backward pass during training on GPU.
     */
    void backwardGpu();
    
    /**
     * Marks the RBM layer as pretrainable.
     * 
     * @return always true for RBM layer.
     */
    virtual bool isPretrainable();
    
    /**
     * Pretrains the layer on CPU.
     */
    virtual void pretrainCpu();
    
    /**
     * Pretrains the layer on GPU.
     */
    virtual void pretrainGpu();

    /**
     * Not implemented for RBM layer, as it cannot be the last one.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastCpu(data_t* expectedOutput);

    /**
     * Not implemented for RBM layer, as it cannot be the last one.
     * 
     * @param expectedOutput
     */
    virtual void backwardLastGpu(data_t* expectedOutput);
    
    RbmConfig getConfig();
    
protected:
    
    /**
     * Layer initialization from the configuration string.
     * 
     * @param confString configuration string for this layer
     */
    void setup(string confString);
    
private:

    /**
     * Computes the potentials and hidden probabilities from the visible states
     * on CPU.
     * 
     * @param visibles
     * @param potentials
     * @param hiddens
     */
    void propagateForwardCpu(data_t *visibles, data_t *potentials, data_t *hiddens);
    
    /**
     * Computes the potentials and hidden probabilities from the visible states
     * on GPU.
     * 
     * @param visibles visible states
     * @param potentials potentials entering hidden neurons
     * @param hiddens probabilities for hidden states
     */
    void propagateForwardGpu(data_t *visibles, data_t *potentials, data_t *hiddens);
    
    /**
     * Computes the potentials and hidden probabilities from the hidden states
     * on CPU.
     * 
     * @param hiddens hidden states
     * @param potentials potentials entering visible neurons
     * @param visibles probabilities for visible states
     */
    void propagateBackwardCpu(data_t *hiddens, data_t *potentials, data_t *visibles);
    
    /**
     * Computes the potentials and hidden probabilities from the hidden states
     * on GPU.
     * 
     * @param hiddens hidden states
     * @param potentials potentials entering visible neurons
     * @param visibles probabilities for visible states
     */
    void propagateBackwardGpu(data_t *hiddens, data_t *potentials, data_t *visibles);
    
    /**
     * Samples the hidden states from the visible states.
     * 
     * @param inputs
     * @param potentials
     * @param outputs
     */
    void sample_vh_gpu(data_t *inputs, data_t *potentials, data_t *outputs);
    
    /**
     * Samples the visible states from the hidden states.
     * 
     * @param outputs
     * @param potentials
     * @param inputs
     */
    void sample_hv_gpu(data_t *outputs, data_t *potentials, data_t *inputs);
    
    /**
     * Performs Gibbs sampling.
     * 
     * @param steps number of Gibbs sampling steps
     */
    void gibbs_hvh(int steps);
    
    /**
     * Parses the configuration string for this layer.
     * 
     * @param confString configuration string
     */
    void processConfString(string confString);
    
    /** Configuration for RBM layer. */
    RbmConfig conf;

    /** Number of visible neurons for this RBM layer. */
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

    /** 
     * If using PCD we need to initialize
     * samples before starting pretraining.
     */
    bool samplesInitialized = false;
};

#endif	/* RBMLAYER_H */

