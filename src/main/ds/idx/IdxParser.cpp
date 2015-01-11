/* 
 * File:   IdxParser.cpp
 * Author: janvojt
 * 
 * Created on January 7, 2015, 12:19 AM
 */

#include "IdxParser.h"

#include "../../log/LoggerFactory.h"
#include "log4cpp/Category.hh"


const unsigned char SUPPORTED_DATA_TYPE = 0x08;
const unsigned int SUPPORTED_DATA_SIZE = 32;
const unsigned int DATA_BYTE_SIZE = 1;


IdxParser::IdxParser() {
}

IdxParser::IdxParser(const IdxParser& orig) {
}

IdxParser::~IdxParser() {
}

IdxData* IdxParser::parse(char* filePath) {
    
    IdxData *data = new IdxData();
    
    // open IDX file with the dataset
    ifstream fp(filePath, ios::in|ios::binary);
    if (!fp.is_open()) {
        LOG()->error("Cannot open file '%s' for parsing.", filePath);
        return NULL;
    }
    
    // read the IDX magic number with information
    // about data types and dimensions
    if (!parseMagicNumber(fp, data)) {
        return NULL;
    }
    
    int dim = data->getNoDimensions();
    
    // read dimension sizes
    for (int i = 0; i<dim; i++) {
        char *strItems = new char[4];
        fp.read(strItems, 4*DATA_BYTE_SIZE);
        data->setDimensionSize(i, chars2int(strItems));
        delete strItems;
    }
    
    // read data by chunks
    data->initData();
    
    int read = 0;
    int chunk = 1024;
    char *pData = (char *) data->getData();
    int size = data->getDataSize();

    int toRead = read + chunk;
    while (toRead < size) {
        fp.read(pData, chunk*DATA_BYTE_SIZE);
        read += chunk;
        toRead += chunk;
        pData += chunk;
    }

    // read the remainder of the last data chunk
    int remainder = size - read;
    fp.read(pData, remainder*DATA_BYTE_SIZE);
    
    // release resources
    fp.close();
    
    return data;
}

bool IdxParser::parseMagicNumber(ifstream &fp, IdxData *data) {
    
    // read magic number
    char *magicNumber = new char[4];
    fp.read(magicNumber, 4*sizeof(char));
    unsigned char *uMagicNumber =  (unsigned char*) magicNumber;
    
    // only unsigned byte is the data type we support (used by MNIST) -> check
    if (uMagicNumber[2] != SUPPORTED_DATA_TYPE) {
        char fHex[2];
        char sHex[2];
        sprintf(fHex, "%02X", uMagicNumber[2]);
        sprintf(sHex, "%02X", SUPPORTED_DATA_TYPE);
        LOG()->error("IDX file claims to be using 0x%s as data type, however IDX parser only supports 0x%s.", fHex, sHex);
        return false;
    }
    
    // set data type -> always 0x08
    data->setDataType(magicNumber[2]);
    
    // get number of dimensions in IDX file
    data->setNoDimensions((int) uMagicNumber[3]);
    
    // return success
    return true;
}

bool IdxParser::isLittleEndian() {
    // int one=1 will stored as 00.01 or 10.00 depending on endianness.
    // (char*)&one will be pointing to first byte of that int.
    // now if that byte reads 1, then its little endian otherwise big endian.
    int one = 1;
    return (*(char*) &one == 1) ? true : false;
}

int IdxParser::chars2int(char *num) {
    
    // we need to cast to unsigned char,
    // as signed char uses only 7 bits
    unsigned char *unum = (unsigned char*) num;
    
    int result = 0;
    if (isLittleEndian()) {
//        result = (unum[0] << 24) + (unum[1] << 16) + (unum[2] << 8) + unum[3]; 
        result = (unum[2] << 8) + unum[3]; 
    } else {
        result = (unum[3] << 8) + unum[2]; 
    }
    
    return result;
}