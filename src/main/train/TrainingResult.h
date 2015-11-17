/* 
 * File:   TrainigResult.h
 * Author: janvojt
 *
 * Created on March 14, 2015, 6:51 PM
 */

#ifndef TRAINIGRESULT_H
#define	TRAINIGRESULT_H

#include <cstdlib>
#include "../common.h"

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
    data_t getTrainingError();
    void setTrainingError(data_t error);
    
    /**
     * @return mean square error on validation data
     */
    data_t getValidationError();
    void setValidationError(data_t error);
    
    /**
     * Helper method used in cases we need any available error.
     * 
     * @return validation error or training error if no validation was done
     */
    data_t getError();
    
private:
    long epochs = 0;
    data_t trainingError = -1;
    data_t validationError = -1;
};

#endif	/* TRAINIGRESULT_H */

