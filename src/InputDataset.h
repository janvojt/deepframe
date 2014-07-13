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
    InputDataset(int dimension);
    InputDataset(const InputDataset& orig);
    virtual ~InputDataset();
    float *next();
    bool hasNext();
    void addInput(float *input);
private:
    void initDataset();
    int dimension;
    int cursor;
    int size;
    float *data;
};

#endif	/* INPUTDATASET_H */

