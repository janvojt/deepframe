/* 
 * File:   MseErrorComputer.cpp
 * Author: janvojt
 * 
 * Created on November 29, 2014, 12:58 PM
 */

#include "cudaDebugHelpers.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <cfloat>

__global__
void sumVectors(data_t *dA, data_t *dB, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dA[i] += dB[i];
    }
}
void k_sumVectors(data_t *dA, data_t *dB, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    sumVectors<<<bs,ts>>>(dA, dB, elements);
}


__global__
void divideVector(data_t *dA, int divisor, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dA[i] /= divisor;
    }
}
void k_divideVector(data_t *dA, int divisor, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    divideVector<<<bs,ts>>>(dA, divisor, elements);
}


__global__
void computeOutputLocalGradient(data_t *actualOutput, data_t *expectedOutput, data_t *localGradient, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        data_t derivative = actualOutput[i] * (1.0 - actualOutput[i]);
        localGradient[i] = (actualOutput[i] - expectedOutput[i]) * derivative;
    }
}
void k_computeOutputLocalGradient(data_t *actualOutput, data_t *expectedOutput, data_t *localGradient, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeOutputLocalGradient<<<bs,ts>>>(actualOutput, expectedOutput, localGradient, elements);
}


__global__
void computeTotalDerivative(data_t learningRate, int nextNeurons,
        data_t *thisInput, data_t *nextLocalGradient,
        data_t *weightDiffs, int elements) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < elements) {
        int i = idx / nextNeurons;
        int j = idx % nextNeurons;
        weightDiffs[i*nextNeurons+j] = -learningRate * nextLocalGradient[j] * thisInput[i];
    }
}
void k_computeTotalDerivative(int thisNeurons, int nextNeurons, 
        data_t learningRate, data_t *thisInput, data_t *nextLocalGradient,
        data_t *weightDiffs) {
    int ts = 512;
    int bs = (thisNeurons * nextNeurons + ts - 1) / ts;
    computeTotalDerivative<<<bs,ts>>>(learningRate, nextNeurons,
        thisInput, nextLocalGradient,
        weightDiffs, thisNeurons * nextNeurons);
}


__global__
void computeBiasDerivative(data_t learningRate, data_t *nextLocalGradient,
        data_t *biasDiffs, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        biasDiffs[i] = -learningRate * nextLocalGradient[i];
    }
}
void k_computeBiasDerivative(
        data_t learningRate, data_t *nextLocalGradient,
        data_t *biasDiffs, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
    computeBiasDerivative<<<bs,ts>>>(learningRate, nextLocalGradient,
        biasDiffs, elements);
}


__global__
void computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        data_t *thisInput, data_t *weights,
        data_t *thisLocalGradient, data_t *nextLocalGradient) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < thisNeurons) {
        data_t derivative = thisInput[i] * (1.0 - thisInput[i]);

        data_t sumNextGradient = 0;
        for (int j = 0; j<nextNeurons; j++) {
            sumNextGradient += nextLocalGradient[j] * weights[i * nextNeurons + j];
        }
        thisLocalGradient[i] = sumNextGradient * derivative;
    }
}
void k_computeHiddenLocalGradient(
        int thisNeurons, int nextNeurons,
        data_t *thisInput, data_t *weights,
        data_t *thisLocalGradient, data_t *nextLocalGradient) {
    
    int ts = 512;
    int bs = (thisNeurons + ts - 1) / ts;
    computeHiddenLocalGradient<<<bs,ts>>>(
        thisNeurons, nextNeurons,
        thisInput, weights,
        thisLocalGradient, nextLocalGradient);
}


__global__
void computeSigmoid(data_t *inArray, data_t *outArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        outArray[i] = 1.0 / (1.0 + exp(-inArray[i]));
    }
}
void k_computeSigmoid(data_t *inArray, data_t *outArray, int elements) {
    int ts = 512;
    int bs = (elements + ts - 1) / ts;
	computeSigmoid<<<bs,ts>>>(inArray, outArray, elements);
}


__global__
void spreadInterval(data_t min, data_t max, data_t *dArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        dArray[i] = (dArray[i] * (max - min)) + min;
    }
}
void k_spreadInterval(data_t min, data_t max, data_t *dArray, int size) {
    int ts = 512;
    int bs = (size + ts - 1) / ts;
    spreadInterval<<<bs,ts>>>(min, max, dArray, size);
}

curandStatus_t k_generateUniform(curandGenerator_t generator,
        data_t *outputPtr,
        size_t num) {
    
#ifdef USE_64BIT_PRECISION
    return curandGenerateUniformDouble(generator, outputPtr, num);
#else
    return curandGenerateUniform(generator, outputPtr, num);
#endif
}


__global__
void uniformToCoinFlip(data_t *p, data_t *dArray, int elements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < elements) {
        p[i] = (dArray[i] < p[i]) ? 1 : 0;
    }
}
void k_uniformToCoinFlip(data_t *p, data_t *dArray, int size) {
    int ts = 512;
    int bs = (size + ts - 1) / ts;
    uniformToCoinFlip<<<bs,ts>>>(p, dArray, size);
}


__global__ void im2col(const int n, const data_t* data_im,
    const int height, const int width, const int kernelHeight, const int kernelWidth,
    const int padHeight, const int padWidth,
    const int strideHeight, const int strideWidth,
    const int heightCol, const int widthCol,
    data_t* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % widthCol;
    int h_index = index / widthCol;
    int h_out = h_index % heightCol;
    int channel_in = h_index / heightCol;
    int channel_out = channel_in * kernelHeight * kernelWidth;
    int h_in = h_out * strideHeight - padHeight;
    int w_in = w_out * strideWidth - padWidth;
    data_t* data_col_ptr = data_col;
    data_col_ptr += (channel_out * heightCol + h_out) * widthCol + w_out;
    const data_t* data_im_ptr = data_im;
    data_im_ptr += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < kernelHeight; ++i) {
      for (int j = 0; j < kernelWidth; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += heightCol * widthCol;
      }
    }
  }
}

void k_im2col(const data_t* data_im, const int channels,
    const int height, const int width, const int kernelHeight, const int kernelWidth,
    const int padHeight, const int padWidth,
    const int strideHeight, const int strideWidth,
    data_t* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * padHeight - kernelHeight) / strideHeight + 1;
  int width_col = (width + 2 * padWidth - kernelWidth) / strideWidth + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col<<<CUDA_GET_BLOCKS(num_kernels),
                             CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernelHeight, kernelWidth, padHeight,
      padWidth, strideHeight, strideWidth, height_col,
      width_col, data_col);
}


__global__ void col2im(const int n, const data_t* data_col,
    const int height, const int width, const int channels,
    const int patchHeight, const int patchWidth,
    const int padHeight, const int padWidth,
    const int strideHeight, const int strideWidth,
    const int heightCol, const int widthCol,
    data_t* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    data_t val = 0;
    int w = index % width + padWidth;
    int h = (index / width) % height + padHeight;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patchWidth) ? 0 : (w - patchWidth) / strideWidth + 1;
    int w_col_end = min(w / strideWidth + 1, widthCol);
    int h_col_start = (h < patchHeight) ? 0 : (h - patchHeight) / strideHeight + 1;
    int h_col_end = min(h / strideHeight + 1, heightCol);
    
    // equivalent implementation
    int offset =
        (c * patchHeight * patchWidth + h * patchWidth + w) * heightCol * widthCol;
    int coeff_h_col = (1 - strideHeight * patchWidth * heightCol) * widthCol;
    int coeff_w_col = (1 - strideWidth * heightCol * widthCol);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

void k_col2im(const data_t* data_col, const int channels,
    const int height, const int width, const int patchHeight, const int patchWidth,
    const int padHeight, const int padWidth, const int strideHeight,
    const int strideWidth, data_t* data_im) {
  int height_col = (height + 2 * padHeight - patchHeight) / strideHeight + 1;
  int width_col = (width + 2 * padWidth - patchWidth) / strideWidth + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im<<<CUDA_GET_BLOCKS(num_kernels),
                             CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patchHeight, patchWidth,
      padHeight, padWidth, strideHeight, strideWidth,
      height_col, width_col, data_im);
}

void k_gemm(cublasContext *handle, const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const data_t alpha, const data_t* A, const data_t* B, const data_t beta,
    data_t* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    
#ifdef USE_64BIT_PRECISION
  CUBLAS_CHECK(cublasDgemm(handle, cuTransB, cuTransA,
          N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
#else
  CUBLAS_CHECK(cublasSgemm(handle, cuTransB, cuTransA,
          N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
#endif
}

void k_axpy(cublasContext *handle, int n, data_t alpha, const data_t *x, int incx, data_t *y, int incy) {
        
#ifdef USE_64BIT_PRECISION
    CUBLAS_CHECK(cublasDaxpy(handle, n, &alpha, x, incx, y, incy));
#else
    CUBLAS_CHECK(cublasSaxpy(handle, n, &alpha, x, incx, y, incy));
#endif
}

void k_scal(cublasContext *handle, int n, data_t alpha, data_t *x, int incx) {
        
#ifdef USE_64BIT_PRECISION
    CUBLAS_CHECK(cublasDscal(handle, n, &alpha, x, incx));
#else
    CUBLAS_CHECK(cublasSscal(handle, n, &alpha, x, incx));
#endif
}

void k_dotProduct(cublasContext *handle, int n, const data_t *x, int incx, const data_t *y, int incy, data_t *result) {
    
#ifdef USE_64BIT_PRECISION
    CUBLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
#else
    CUBLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
#endif
}


__global__ void MaxPoolForward(const int nthreads,
    const data_t* const inputs, const int featuresCount,
    const int inputFeatureHeight, const int inputFeatureWidth, const int featureHeight,
    const int featureWidth, const int kernelHeight, const int kernelWidth,
    const int strideHeight, const int strideWidth, const int padHeight, const int padWidth,
    data_t* const outputs, int* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % featureWidth;
    const int ph = (index / featureWidth) % featureHeight;
    const int c = (index / featureWidth / featureHeight) % featuresCount;
    const int n = index / featureWidth / featureHeight / featuresCount;
    int hstart = ph * strideHeight - padHeight;
    int wstart = pw * strideWidth - padWidth;
    const int hend = min(hstart + kernelHeight, inputFeatureHeight);
    const int wend = min(wstart + kernelWidth, inputFeatureWidth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    data_t maxval = -FLT_MAX;
    int maxidx = -1;
    const data_t* const inputSlice =
        inputs + (n * featuresCount + c) * inputFeatureHeight * inputFeatureWidth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (inputSlice[h * inputFeatureWidth + w] > maxval) {
          maxidx = h * inputFeatureWidth + w;
          maxval = inputSlice[maxidx];
        }
      }
    }
    outputs[index] = maxval;
    mask[index] = maxidx;
  }
}

void k_MaxPoolForward(const int nthreads,
    const data_t* const inputs, const int featuresCount,
    const int inputFeatureHeight, const int inputFeatureWidth, const int featureHeight,
    const int featureWidth, const int kernelHeight, const int kernelWidth,
    const int strideHeight, const int strideWidth, const int padHeight, const int padWidth,
    data_t* const outputs, int* mask) {

    MaxPoolForward<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(
            nthreads, inputs, featuresCount,
            inputFeatureHeight, inputFeatureWidth, featureHeight,
            featureWidth, kernelHeight, kernelWidth,
            strideHeight, strideWidth, padHeight, padWidth,
            outputs, mask);
}


__global__ void MaxPoolBackward(const int nthreads, const data_t* const outputDiffs,
    const int* const mask,
    const int channels, const int inputFeatureHeight, const int inputFeatureWidth,
    const int featureHeight, const int featureWidth, const int kernelHeight,
    const int kernelWidth, const int strideHeight, const int strideWidth, const int padHeight,
    const int padWidth, data_t* const inputDiffs) {
    
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % inputFeatureWidth;
    const int h = (index / inputFeatureWidth) % inputFeatureHeight;
    const int c = (index / inputFeatureWidth / inputFeatureHeight) % channels;
    const int n = index / inputFeatureWidth / inputFeatureHeight / channels;
    const int phstart =
         (h + padHeight < kernelHeight) ? 0 : (h + padHeight - kernelHeight) / strideHeight + 1;
    const int phend = min((h + padHeight) / strideHeight + 1, featureHeight);
    const int pwstart =
         (w + padWidth < kernelWidth) ? 0 : (w + padWidth - kernelWidth) / strideWidth + 1;
    const int pwend = min((w + padWidth) / strideWidth + 1, featureWidth);
    data_t gradient = 0;
    const int offset = (n * channels + c) * featureHeight * featureWidth;
    const data_t* const outputDiffSlice = outputDiffs + offset;
    if (mask) {
      const int* const maskSlice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (maskSlice[ph * featureWidth + pw] == h * inputFeatureWidth + w) {
            gradient += outputDiffSlice[ph * featureWidth + pw];
          }
        }
      }
    }
    inputDiffs[index] = gradient;
  }
}

void k_MaxPoolBackward(const int nthreads, const data_t* const outputDiffs,
    const int* const mask,
    const int channels, const int inputFeatureHeight, const int inputFeatureWidth,
    const int featureHeight, const int featureWidth, const int kernelHeight,
    const int kernelWidth, const int stride_h, const int strideWidth, const int padHeight,
    const int padWidth, data_t* const inputDiffs) {
    
    MaxPoolBackward<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS>>>(nthreads, outputDiffs,
    mask, channels, inputFeatureHeight, inputFeatureWidth,
    featureHeight, featureWidth, kernelHeight,
    kernelWidth, stride_h, strideWidth, padHeight,
    padWidth, inputDiffs);
}

__global__ void reduce0(data_t *g_idata, data_t *g_odata, int size) {

    // TODO replace implementation with better performing reduce7
    // see https://docs.nvidia.com/cuda/samples/6_Advanced/reduction/doc/reduction.pdf

    extern __shared__ data_t sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if (i < size)
        sdata[tid] = g_idata[i];
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

data_t k_sumReduce(data_t *in, data_t *temp, unsigned long n) {
    
    const unsigned int ts = 256;
    int bs = (n + ts - 1) / ts;
    
    // TODO allow scaling above ts^2
    // see http://stackoverflow.com/questions/18023287/cuda-how-can-i-run-the-parallel-reduction-code-for-summation-that-is-described
    reduce0<<<bs, ts, ts*sizeof(data_t)>>>(in, temp, n);
    
    data_t *out;
    checkCudaErrors(cudaMalloc(&out, bs * sizeof(data_t)));
    reduce0<<< 1, ts, ts*sizeof(data_t)>>>(temp, out, bs);
    
    data_t res = 0;
    checkCudaErrors(cudaMemcpy(&res, out, sizeof(data_t), cudaMemcpyDeviceToHost));

    return res;
}

__global__ void logPlusExpReduce0(data_t a, data_t *g_idata, data_t *g_odata, int size){

   extern __shared__ data_t sdata[];

   unsigned int tid = threadIdx.x;
   unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
   sdata[tid] = 0;
   if(i<size)
     sdata[tid] = log(a + exp(g_idata[i]));
   __syncthreads();

  for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
         sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
     }

   if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

data_t k_logPlusExpReduce(data_t a, data_t *in, data_t *temp, unsigned long n) {
    
    const unsigned int ts = 256;
    int bs = (n + ts - 1) / ts;
    
    // TODO allow scaling above ts^2
    // see http://stackoverflow.com/questions/18023287/cuda-how-can-i-run-the-parallel-reduction-code-for-summation-that-is-described
    logPlusExpReduce0<<<bs, ts, ts*sizeof(data_t)>>>(a, in, temp, n);
    
    data_t *out;
    checkCudaErrors(cudaMalloc(&out, bs * sizeof(data_t)));
    reduce0<<< 1, ts, ts*sizeof(data_t)>>>(temp, out, bs);
    
    data_t res = 0;
    checkCudaErrors(cudaMemcpy(&res, out, sizeof(data_t), cudaMemcpyDeviceToHost));

    return res;
}

__global__ void crossEntropyReduce0(data_t *v, data_t *pv, data_t *g_odata, int size) {

    extern __shared__ data_t sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = 0;
    if (i < size) {
        data_t sig = 1/(1+exp(-pv[i]));
        sdata[tid] = v[i] * log(sig) + (1-v[i]) * log(1-sig);
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

data_t k_crossEntropyReduce(data_t *visibles, data_t *potentials, data_t *temp, unsigned long n) {

    const unsigned int ts = 256;
    int bs = (n + ts - 1) / ts;

    // TODO allow scaling above ts^2
    // see http://stackoverflow.com/questions/18023287/cuda-how-can-i-run-the-parallel-reduction-code-for-summation-that-is-described
    crossEntropyReduce0<<<bs, ts, ts * sizeof (data_t)>>>(visibles, potentials, temp, n);
    
    data_t *out;
    checkCudaErrors(cudaMalloc(&out, bs * sizeof(data_t)));
    reduce0 <<< 1, ts, ts * sizeof (data_t)>>>(temp, out, bs);

    data_t res = 0;
    checkCudaErrors(cudaMemcpy(&res, out, sizeof (data_t), cudaMemcpyDeviceToHost));

    return res;
}
