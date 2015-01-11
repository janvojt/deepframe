/* 
 * File:   IdxData.h
 * Author: janvojt
 *
 * Created on January 7, 2015, 12:17 AM
 */

#include <cstdlib>

#ifndef IDXDATA_H
#define	IDXDATA_H

class IdxData {
public:
    IdxData();
    IdxData(const IdxData& orig);
    virtual ~IdxData();
    // Returns the third byte of magic number that codes the type of the data:
    // 0x08: unsigned byte
    // 0x09: signed byte
    // 0x0B: short (2 bytes)
    // 0x0C: int (4 bytes)
    // 0x0D: float (4 bytes)
    // 0x0E: double (8 bytes)
    char getDataType();
    void setDataType(char dataType);
    // Returns the number of dimensions of the data.
    // Value is parsed from the 4th byte of the magic number.
    int getNoDimensions();
    void setNoDimensions(int noDimensions);
    // Returns an array with sizes of each respective dimension.
    // Array size must correspond with #getNoDimensions().
    int getDimensionSize(int dimension);
    void setDimensionSize(int dimension, int dimensionSize);
    // Returns data array of corresponding type and dimensions.
    void* getData();
    // Initializes data array, must be called after all dimension sizes are set.
    void initData();
    // Returns the size of the dataset in bytes.
    int getDataSize();
private:
    char dataType;
    int noDimensions;
    int dataSize = 0;
    int *dimensionSizes = NULL;
    char *data = NULL;
};

#endif	/* IDXDATA_H */

