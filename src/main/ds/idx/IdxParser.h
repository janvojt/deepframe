/* 
 * File:   IdxParser.h
 * Author: janvojt
 *
 * Created on January 7, 2015, 12:19 AM
 */

#ifndef IDXPARSER_H
#define	IDXPARSER_H

#include "IdxData.h"

#include <iostream>
#include <fstream>

using namespace std;

class IdxParser {
public:
    IdxParser();
    IdxParser(const IdxParser& orig);
    virtual ~IdxParser();
    // Parse the IDX file located at given filepath and return the contained
    // data. Returns null if the file cannot be read or has invalid format.
    IdxData *parse(char *filePath);
private:
    bool parseMagicNumber(ifstream &fp, IdxData *data);
    // Detects whether the architecture we are running on is little endian.
    bool isLittleEndian();
    // Converts big-endian data given as characters into integer with proper
    // endianess on given system.
    int chars2int(char *num);
};

#endif	/* IDXPARSER_H */

