<?php
/**
 * Generates IDX dataset with patterns filtered by given criteria.
 *
 * User: janvojt
 * Date: 6/13/16
 * Time: 8:53 PM
 */

// IDX header type is always 4 bytes in big endian (MSB first)
const IDX_HEADER_TYPE = "N";

$srcPath = "resources/mnist/";
$dstPath = "resources/mnist-01/";
$trainData = "train-images-idx3-ubyte";
$trainLabels = "train-labels-idx1-ubyte";
$testData = "t10k-images-idx3-ubyte";
$testLabels = "t10k-labels-idx1-ubyte";

// configure the filtered labels here
$filteredLabels = array(0, 1);

$ds = new IdxDataset($srcPath.$trainData, $srcPath.$trainLabels);
$ds->setFilteredClasses($filteredLabels);
$ds->write($dstPath.$trainData, $dstPath.$trainLabels);

$ds = new IdxDataset($srcPath.$testData, $srcPath.$testLabels);
$ds->setFilteredClasses($filteredLabels);
$ds->write($dstPath.$testData, $dstPath.$testLabels);



class IdxDataset
{
    private $dataPath;
    private $labelsPath;

    private $filteredClasses = array();

    function __construct($dataPath, $labelsPath)
    {
        $this->dataPath = $dataPath;
        $this->labelsPath = $labelsPath;
    }

    /**
     * @return array
     */
    public function getFilteredClasses()
    {
        return $this->filteredClasses;
    }

    /**
     * @param array $filteredClasses
     */
    public function setFilteredClasses(array $filteredClasses)
    {
        $this->filteredClasses = $filteredClasses;
    }

    /**
     * @param $dstDataPath
     * @param $dstLabelPath
     */
    public function write($dstDataPath, $dstLabelPath)
    {

        // process train data and labels first
        $fpSrcData = fopen($this->dataPath, "rb");
        $fpSrcLabels = fopen($this->labelsPath, "rb");

        $dataHeader = $this->parseHeader($fpSrcData);
        $labelsHeader = $this->parseHeader($fpSrcLabels);
        assert($dataHeader->getDimSize(0) == $labelsHeader->getDimSize(0));

        // open destination files for writing
        $fpDstData = fopen($dstDataPath, "wb");
        $fpDstLabels = fopen($dstLabelPath, "wb");

        // write the headers
        // write the same number of patterns as in source dataset, fix later
        $this->writeHeader($fpDstData, $dataHeader);
        $this->writeHeader($fpDstLabels, $labelsHeader);

        // compute the number of bytes taken by a single data pattern
        $patternSize = $dataHeader->getDataSize();
        for ($i = 1; $i<$dataHeader->getDims(); $i++)
        {
            // skip first dimension, as it represents dataset size
            $patternSize *= $dataHeader->getDimSize($i);
        }

        // read, filter and write data
        $newSize = 0;
        for ($i = 0; $i<$labelsHeader->getDimSize(0); $i++)
        {
            $rawLabel = fread($fpSrcLabels, $labelsHeader->getDataSize());
            $label = unpack($labelsHeader->getType(), $rawLabel)[1];

            $rawData = fread($fpSrcData, $patternSize);

            if (in_array($label, $this->filteredClasses))
            {
                fwrite($fpDstLabels, $rawLabel);
                fwrite($fpDstData, $rawData);
                $newSize++;
            }
        }

        // fix the dataset sizes
        $this->resetDatasetSize($fpDstData, $newSize);
        $this->resetDatasetSize($fpDstLabels, $newSize);

        // close all file handles
        fclose($fpDstLabels);
        fclose($fpDstData);
        fclose($fpSrcLabels);
        fclose($fpSrcData);
    }

    private function resetDatasetSize($fp, $size)
    {
        fseek($fp, 4);
        fwrite($fp, pack(IDX_HEADER_TYPE, $size));
    }

    private function writeHeader($fp, IdxDims $header)
    {
        $magicNumber = bytes2str(array(0, 0, $header->getRawType(), $header->getDims()));
        fwrite($fp, $magicNumber, 4);
        for ($i = 0; $i<$header->getDims(); $i++)
        {
            fwrite($fp, pack(IDX_HEADER_TYPE, $header->getDimSize($i)));
        }
    }

    private function parseHeader($fp)
    {
        // read the magic number
        $magicNumber = unpack(IDX_HEADER_TYPE, fread($fp, 4))[1];

        // determine data types
        $dataType = "C";
        $rawType = $magicNumber >> 8;

        // determine number of dimensions
        $dims = $magicNumber & 0xff;

        // read the dimension sizes
        $dimSizes = array();
        for ($i = 0; $i < $dims; $i++) {
            $bytes = fread($fp, 4);
            $dimSizes[] = unpack(IDX_HEADER_TYPE, $bytes)[1];
        }

        return new IdxDims($dims, $dimSizes, $rawType);
    }
}

class IdxDims {

    private $dims;
    private $dimSizes;
    private $type;
    private $rawType;
    private $dataSize = 0;

    function __construct($dims, array $dimSizes, $rawType) {
        $this->dims = $dims;
        $this->dimSizes = $dimSizes;
        $this->rawType = $rawType;

        switch ($rawType) {
            case 0x08 :
                $this->type = "C";
                $this->dataSize = 1;
                break;
            case 0x09 :
                $this->type = "c";
                $this->dataSize = 1;
                break;
            case 0x0B :
                $this->type = "S";
                $this->dataSize = 2;
                break;
            case 0x0C :
                $this->type = "N";
                $this->dataSize = 4;
                break;
            case 0x0D :
                $this->type = "N";
                $this->dataSize = 4;
                break;
            case 0x0E :
                $this->type = "J";
                $this->dataSize = 8;
                break;
        }
    }

    function getDims() {
        return $this->dims;
    }

    function getDimSize($dim) {
        return $this->dimSizes[$dim];
    }

    /**
     * @return string
     */
    public function getType()
    {
        return $this->type;
    }

    /**
     * @return string binary data representing IDX data type
     */
    public function getRawType()
    {
        return $this->rawType;
    }

    /**
     * @return int number of bytes of a single data point in a pattern
     */
    public function getDataSize()
    {
        return $this->dataSize;
    }

}

function str2bytes($str)
{
    $bytes = array();
    $len = strlen($str);
    for ($i = 0; $i < $len; $i++)
    {
        $bytes[] = ord($str[$i]);
    }
    return $bytes;
}

function bytes2str(array $bytes)
{
    $size = count($bytes);
    $str = str_repeat(" ", $size);

    for ($i = 0; $i < $size; $i++)
    {
        $str{$i} = chr($bytes[$i]);
    }

    return $str;
}
