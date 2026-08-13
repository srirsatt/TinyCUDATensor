// basic tensor implementation

// tensor should have:

/*

Constructor
Destructor
1. Data Buffer (vector)
2. Shape (3 element arr int)
3. Stride (3 element arr int) - calculated from shape

getters & setters, of course!
*/

#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept> // for exceptions

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

template <typename T>
SimpleTensor<T>::SimpleTensor(std::vector<int> shape, int dimension, T* dataBuffer, bool requiresGrad) {
    // constructor
    // take in the data buffer from cpu mem, copy it to GPU mem, copy shape into the private field and dimension

    if (shape.size() != dimension) {
        throw std::invalid_argument("shape size must match dimension count!");
    }

    dimension_ = dimension;
    shape_ = shape;
    requiresGrad_ = requiresGrad;

    size_ = 1;
    for (int i = 0; i < shape.size(); i++) {
        size_ *= shape[i];
    }

    // compute Stride!
    stride_.resize(dimension_);
    stride_[dimension_ - 1] = 1; // last dim
    for (int i = dimension_ - 2; i >= 0; i--) {
        stride_[i] = stride_[i+1] * shape_[i+1];
    }

    // now i want to copy data buffer through alloc into cuda

    if (requiresGrad == true) {
        T* g_buf;
        CUDA_CHECK(cudaMalloc(&g_buf, size_*sizeof(T)));

        cudaMemset(g_buf, 0, size_*sizeof(T));

        gradBuffer_ = g_buf;
    } else {
        gradBuffer_ = nullptr;
    }



    // cudaMalloc
    T* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, size_*sizeof(T)));

    // cudaMemcpy
    CUDA_CHECK(cudaMemcpy(d_buf, dataBuffer, size_*sizeof(T), cudaMemcpyHostToDevice)); // CPU->GPU memcpy

    dataBuffer_ = d_buf;
}

template <typename T>
SimpleTensor<T>::SimpleTensor(std::vector<int> shape, int dimension, bool requiresGrad) {
    // same as before, fill with blanks

    if (shape.size() != dimension) {
        throw std::invalid_argument("shape size must match dimension count!");
    }

    dimension_ = dimension;
    shape_ = shape;
    requiresGrad_ = requiresGrad;

    size_ = 1;
    for (int i = 0; i < shape.size(); i++) {
        size_ *= shape[i];
    }

    // set your stride!
    stride_.resize(dimension_);
    stride_[dimension_ - 1] = 1;

    for (int i = dimension_ - 2; i >= 0; i--) {
        stride_[i] = stride_[i+1] * shape_[i+1];
    }

    if (requiresGrad == true) {
        T* g_buf;
        CUDA_CHECK(cudaMalloc(&g_buf, size_*sizeof(T)));

        cudaMemset(g_buf, 0, size_*sizeof(T));

        gradBuffer_ = g_buf;
    } else {
        gradBuffer_ = nullptr;
    }

    T* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, size_*sizeof(T)));

    cudaMemset(d_buf, 0, size_*sizeof(T)); // setting to all 0's, no need to have a Memcpy

    dataBuffer_ = d_buf;
}

template <typename T>
SimpleTensor<T>::~SimpleTensor() {
    // destructor

    cudaFree(dataBuffer_); // only thing, everything else takes care
    // if i have a gradBuffer_ - we should also dealloc this

    if (gradBuffer_) { // if not null
        cudaFree(gradBuffer_);
    }
}

template <typename T>
__global__ void fillKernel(T* buf, T val, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size) {
        buf[i] = val;
    }
}


template <typename T>
void SimpleTensor<T>::backward() {
    if (requiresGrad_) {
        // to check if we have an empty leaf function, and if requiresGrad_ is true - for autograd computation in the first place
            int threads = 256;
            int blocks = (size_ + threads - 1) / threads;

            fillKernel<<<blocks, threads>>>(gradBuffer_, (T)1.0f, size_);
    }

    if (backward_) {
        backward_();
    }
}

template <typename T>
T* SimpleTensor<T>::getGradBuffer() {
    return gradBuffer_;
}




template <typename T>
void SimpleTensor<T>::reshape(std::vector<int> shape, int dimension) {

    // check validity within shape

    if (shape.size() != dimension) {
        throw std::invalid_argument("shape size has to match dimension count!");
    }

    shape_ = shape;
    dimension_ = dimension;

    // calculate new size and verify it matches up, otherwise through an invalid_arg

    int newSize = 1;
    for (int i = 0; i < shape.size(); i++) {
        newSize *= shape[i];
    }

    if (newSize != size_) {
        throw std::invalid_argument("new shape has to properly conform to existing size standards.");
    }

    // recalc stride
    stride_.resize(dimension_);
    stride_[dimension_ - 1] = 1;
    for (int i = dimension_ - 2; i >= 0; i--) {
        stride_[i] = stride_[i+1] * shape_[i+1];
    }
}


template <typename T>
void SimpleTensor<T>::setBuffer(T* dataBuffer, int size) {

    // buffer - change in data
    // size check for consistency

    if (size != size_) {
        throw std::invalid_argument("new data buffer size has to match to the existing data buffer size.");
    }

    // new gpu malloc

    // free existing buffer

    cudaFree(dataBuffer_);


    T* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, size_*sizeof(T)));

    CUDA_CHECK(cudaMemcpy(d_buf, dataBuffer, size_*sizeof(T), cudaMemcpyHostToDevice));

    dataBuffer_ = d_buf;
}

template <typename T>
void SimpleTensor<T>::setRequiresGrad(bool input) {
    requiresGrad_ = input;
}


template <typename T>
void SimpleTensor<T>::print() {
    // print method - want to print in the Kernel shape

    // print per matrix, for each "layer" in the matrix

    std::vector<T> data = toHost();

    // now lets print it

    if (dimension_ == 1) {
        std::cout << "1D tensor. printing elements in the one row!" << std::endl;
        for (int i = 0; i < size_; i++) {
            std::cout << data[i] << std::endl;
        }
    } else if (dimension_ == 2) {
        std::cout << "2D Tensor. printing out matrix!" << std::endl;

        for (int i = 0; i < shape_[0]; i++) {
            for (int j = 0; j < shape_[1]; j++) {
                std::cout << data[i * stride_[0] + j * stride_[1]] << " ";
            }
            std::cout << std::endl;
        }
    } else if (dimension_ == 3) {
        std::cout << "3d tensor. printing out matrices!" << std::endl;
        for (int i = 0; i < shape_[0]; i++) {
            for (int j = 0; j < shape_[1]; j++) {
                for (int k = 0; k < shape_[2]; k++) {
                    std::cout << data[i * stride_[0] + j * stride_[1] + k * stride_[2]] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "---" << std::endl;  // separator between matrices
        } 
    } else {
        // all elements simple print
        for (int i = 0; i < data.size(); i++) {
            std::cout << "element " << i << ": " << data[i];
        }
    }
}


template <typename T>
std::vector<int> SimpleTensor<T>::getShape() {

    return shape_;

}


template <typename T>
T* SimpleTensor<T>::getBuffer() {

    return dataBuffer_; // keep in mind, this returns the Addr to the T (template) arr

}


template <typename T>
std::vector<int> SimpleTensor<T>::getStride() {

    return stride_;
}

template <typename T>
std::vector<T> SimpleTensor<T>::toHost() {

    std::vector<T> dataBuffer;

    dataBuffer.resize(size_);

    CUDA_CHECK(cudaMemcpy(dataBuffer.data(), dataBuffer_, size_*sizeof(T), cudaMemcpyDeviceToHost));


    return dataBuffer;
}

template <typename T>
std::vector<T> SimpleTensor<T>::toHostGrad() {
    if (!gradBuffer_) {
        throw std::runtime_error("no grad buffer allocated or present");
    }

    std::vector<T> result(size_);
    CUDA_CHECK(cudaMemcpy(result.data(), gradBuffer_, size_ * sizeof(T), cudaMemcpyDeviceToHost));

    return result;
}

template <typename T>
int SimpleTensor<T>::getDimension() {
    return dimension_;
}

template <typename T>
int SimpleTensor<T>::getSize() {
    return size_;
}

template <typename T>
bool SimpleTensor<T>::getRequiresGrad() {
    return requiresGrad_;
}

// copy constructor implementation
template <typename T>
SimpleTensor<T>::SimpleTensor(const SimpleTensor<T>& other) {
    dimension_ = other.dimension_;
    size_ = other.size_;
    shape_ = other.shape_;
    stride_ = other.stride_;
    requiresGrad_ = other.requiresGrad_;
    gradNode_ = other.gradNode_;

    CUDA_CHECK(cudaMalloc(&dataBuffer_, size_*sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dataBuffer_, other.dataBuffer_, size_* sizeof(T), cudaMemcpyDeviceToDevice));

    if (other.gradBuffer_) {
        CUDA_CHECK(cudaMalloc(&gradBuffer_, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(gradBuffer_, other.gradBuffer_, size_*sizeof(T), cudaMemcpyDeviceToDevice));
    } else {
        gradBuffer_ = nullptr;
    }
}

template <typename T>
SimpleTensor<T>& SimpleTensor<T>::operator=(const SimpleTensor<T>& other) {
    if (this == &other) return *this;

    cudaFree(dataBuffer_);
    if (gradBuffer_) cudaFree(gradBuffer_);

    dimension_ = other.dimension_;
    size_ = other.size_;
    shape_ = other.shape_;
    stride_ = other.stride_;
    requiresGrad_ = other.requiresGrad_;
    gradNode_ = other.gradNode_;

    CUDA_CHECK(cudaMalloc(&dataBuffer_, size_ * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(dataBuffer_, other.dataBuffer_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));

    if (other.gradBuffer_) {
        CUDA_CHECK(cudaMalloc(&gradBuffer_, size_ * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(gradBuffer_, other.gradBuffer_, size_* sizeof(T), cudaMemcpyDeviceToDevice));
    } else {
        gradBuffer_ = nullptr;
    }
    return *this;
}

// move constructor
template <typename T>
SimpleTensor<T>::SimpleTensor(SimpleTensor<T>&& other) noexcept {
    dimension_ = other.dimension_;
    size_ = other.size_;
    shape_ = std::move(other.shape_);
    stride_ = std::move(other.stride_);
    requiresGrad_ = other.requiresGrad_;
    gradNode_ = std::move(other.gradNode_);
    
    dataBuffer_ = other.dataBuffer_;
    gradBuffer_ = other.gradBuffer_;

    other.dataBuffer_ = nullptr;
    other.gradBuffer_ = nullptr;
}

// move assignment - with return to SimpleTensor pointer
template <typename T>
SimpleTensor<T>& SimpleTensor<T>::operator=(SimpleTensor<T>&& other) noexcept {
    if (this == &other) return *this;

    cudaFree(dataBuffer_);
    if (gradBuffer_) cudaFree(gradBuffer_);

    dimension_ = other.dimension_;
    size_ = other.size_;
    shape_ = std::move(other.shape_);
    stride_ = std::move(other.stride_);
    requiresGrad_ = other.requiresGrad_;

    gradNode_ = std::move(other.gradNode_);
    dataBuffer_ = other.dataBuffer_;
    gradBuffer_ = other.gradBuffer_;
    
    other.gradBuffer_ = nullptr;
    other.dataBuffer_ = nullptr;

    return *this; 
}

template class SimpleTensor<float>;
template class SimpleTensor<double>;
template class SimpleTensor<int>;
template class SimpleTensor<long>;
