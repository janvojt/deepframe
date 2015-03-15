/* 
 * File:   TrainigResult.h
 * Author: janvojt
 *
 * Created on March 14, 2015, 6:51 PM
 */

#ifndef TRAINIGRESULT_H
#define	TRAINIGRESULT_H

#include <cstdlib>

template <typename dType>
class TrainingResult {
    
public:
    TrainingResult();
    TrainingResult(const TrainingResult& orig);
    virtual ~TrainingResult();
    
    /**
     * @return number of training epochs
     */
    long getEpochs();
    void setEpochs(long epochs);
    
    /**
     * @return mean square error on training data
     */
    dType getTrainingError();
    void setTrainingError(dType error);
    
    /**
     * @return mean square error on validation data
     */
    dType getValidationError();
    void setValidationError(dType error);
    
    /**
     * Helper method used in cases we need any available error.
     * 
     * @return validation error or training error if no validation was done
     */
    dType getError();
    
private:
    long epochs = 0;
    dType trainingError = -1;
    dType validationError = -1;
};

#endif	/* TRAINIGRESULT_H */

