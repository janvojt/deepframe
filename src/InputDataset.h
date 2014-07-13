/* 
 * File:   InputDataset.h
 * Author: janvojt
 *
 * Created on July 13, 2014, 5:29 PM
 */

#ifndef INPUTDATASET_H
#define	INPUTDATASET_H

class InputDataset {
public:
    InputDataset();
    InputDataset(const InputDataset& orig);
    virtual ~InputDataset();
    virtual float *next() = 0;
    virtual bool hasNext() = 0;
    virtual void reset() = 0;
    virtual int getInputDimension() = 0;
private:
    
};

#endif	/* INPUTDATASET_H */

