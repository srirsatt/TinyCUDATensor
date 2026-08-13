// operations on tensors with GPU

#include "tensor.h"
#include "ops.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}


template <typename T>
__global__ void elementKernel(T* a, T* b, T* c, int N, ElementWiseOp operation) {
    // simple add op

    int i = blockIdx.x * blockDim.x + threadIdx.x; // calculating the exact thread ur located in

    if (i < N) {
        switch (operation) {
            case ElementWiseOp::ADD:
                c[i] = a[i] + b[i];
                break;
            case ElementWiseOp::SUBTRACT:
                c[i] = a[i] - b[i];
                break;
            case ElementWiseOp::MULTIPLY:
                c[i] = a[i] * b[i];
                break;
            case ElementWiseOp::DIVIDE:
                c[i] = a[i] / b[i];
                break;
        }
    }
}

// kernels for add & mul for autograd - to be pulled backwards by lamdbas for back prop

template <typename T>
__global__ void addBackwardKernel(T* grad, T* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < size) {
        grad[i] += output[i];
    }
}

template <typename T>
__global__ void mulBackwardKernel(T* grad, T* incomingGrad, T* other, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        grad[i] += incomingGrad[i] * other[i];
    }
}

template <typename T>
__global__ void transposeMatrix(T* input, T* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col]; // basic transpose
    }
}

template <typename T>
__global__ void accumulate(T* gradBuffer, T* incomingGrad, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        gradBuffer[i] += incomingGrad[i];
    }
}

/*
template <typename T>
__global__ void subtractKernel(T* a, T* b, T* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] - b[i];
    }
}

template <typename T>
__global__ void multiplyKernel(T* a, T* b, T* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] * b[i];
    }
}

template <typename T>
__global__ void divideKernel(T* a, T* b, T* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        c[i] = a[i] / b[i];
    }
}

*/

template <typename T>
// CPU add
SimpleTensor<T> elementOp(SimpleTensor<T>& a, SimpleTensor<T>& b, ElementWiseOp operation) {
    
    // valid input shapes
    if (a.getDimension() != b.getDimension()) {
        throw std::invalid_argument("shape sizes arent the same!");
    }

    bool needsGrad = a.getRequiresGrad() || b.getRequiresGrad();


    // check shapes themselves against each other

    for (int i = 0; i < a.getDimension(); i++) {
        if (a.getShape()[i] != b.getShape()[i]) {
            throw std::invalid_argument("shape numbers aren't the same");
        }
    }

    // create output Tensor
    SimpleTensor<T> outputTensor(a.getShape(), a.getDimension(), needsGrad); // this will Have a data buffer here

    // launch kernel, where itll cudaMalloc to the place of the output tensor
    int threads = 256; // default block thread count
    int blocks = (a.getSize() + threads - 1) / threads; // esssentially a ceil function for block counts

    // constructors manage mem!
    elementKernel<<<blocks, threads>>>(a.getBuffer(), b.getBuffer(), outputTensor.getBuffer(), a.getSize(), operation);

    // for memory concerns

    T* gradC = outputTensor.getGradBuffer();
    int size = a.getSize();

    auto node = std::make_shared<GradNode>();
    node->inputs = {a.getGradNode(), b.getGradNode()};
    node->backward_fn = [&a, &b, gradC, size, operation]() {
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        switch (operation) {
            case ElementWiseOp::ADD:
                if (a.getRequiresGrad()) {
                    addBackwardKernel<<<blocks, threads>>>(a.getGradBuffer(), gradC, size);
                }
                if (b.getRequiresGrad()) {
                    addBackwardKernel<<<blocks, threads>>>(b.getGradBuffer(), gradC, size);
                }
                break;
            case ElementWiseOp::MULTIPLY:
                if (a.getRequiresGrad()) {
                    mulBackwardKernel<<<blocks, threads>>>(a.getGradBuffer(), gradC, b.getBuffer(), size);
                }
                if (b.getRequiresGrad()) {
                    mulBackwardKernel<<<blocks, threads>>>(b.getGradBuffer(), gradC, a.getBuffer(), size);
                }
                break;
        }
    };

    outputTensor.setGradNode(node);

    return outputTensor;
}

template <typename T>
__global__ void scalarKernel(T* a, T scalar, T* c, int N, ScalarOp operation) {
    // switch on the operation

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        switch (operation) {
            case ScalarOp::ADD:
                c[i] = a[i] + scalar;
                break;
            case ScalarOp::SUBTRACT:
                c[i] = a[i] - scalar;
                break;
            case ScalarOp::MULTIPLY:
                c[i] = a[i] * scalar;
                break;
            case ScalarOp::DIVIDE:
                c[i] = a[i] / scalar;
                break;
        }
    }
}

template <typename T>
SimpleTensor<T> scalarOp(SimpleTensor<T>& a, T scalar, ScalarOp operation) {
    // outputs operand + Tensor

    SimpleTensor<T> outputTensor(a.getShape(), a.getDimension()); // creates zeroed dataBuf

    int threads = 256;
    int blocks = (a.getSize() + threads - 1) / threads;

    scalarKernel<<<blocks, threads>>>(a.getBuffer(), scalar, outputTensor.getBuffer(), a.getSize(), operation);

    return outputTensor;
}

template <typename T>
__global__ void reduceKernel(T* input, T* output, int N, ReduceOp operation) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // exact thread that im at

    __shared__ T sharedData[256];

    // syncThreads fix

    sharedData[threadIdx.x] = (i < N) ? input[i] : T(0);
    __syncthreads();

    // now we need to load all the data from input into shared mem

    if (i < N) {

        sharedData[threadIdx.x] = input[i];
    
        //sharedData[threadIdx.x] = input[i]; // per block thread index - so per block, this loads every thread with a "data point" from the dataBuf


        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            // block dim, remember, is the amount of threads in a block. move thru "half" of the array potential and add

            if (threadIdx.x < stride) {
                // runs for only half

                switch (operation) {
                    case ReduceOp::SUM:
                        sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
                        break;
                    case ReduceOp::MAX:
                        if (sharedData[threadIdx.x] < sharedData[threadIdx.x + stride]) {
                            sharedData[threadIdx.x] = sharedData[threadIdx.x + stride];
                        }
                        break;
                    case ReduceOp::MIN:
                        if (sharedData[threadIdx.x] > sharedData[threadIdx.x + stride]) {
                            sharedData[threadIdx.x] = sharedData[threadIdx.x + stride];
                        }
                        break;
                    case ReduceOp::MEAN:
                        sharedData[threadIdx.x] += sharedData[threadIdx.x + stride];
                        break;
                    case ReduceOp::PRODUCT:
                        sharedData[threadIdx.x] *= sharedData[threadIdx.x + stride];
                        break;
                    default:
                        break;
                }
                // we half the tree size at each loop, then we do the sum at each thread equals sum[thread] + sum[thread+stride]. - that way we are computing properly in parallel.
                
                __syncthreads();
            }
        }

        if (threadIdx.x == 0 && operation == ReduceOp::MEAN) {
            output[blockIdx.x] = sharedData[threadIdx.x] / N; // final check - we need to make sure after the entire tree reduction, the data that belongs in sharedData[0] is in output[0] for each block
        } else if (threadIdx.x == 0) {
            output[blockIdx.x] = sharedData[threadIdx.x];
        }

    }

}

template <typename T>
__global__ void accumulateGradScalar(T* grad, T* incomingScalar, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        grad[i] += incomingScalar[0];
    }
}


template <typename T>
SimpleTensor<T> reduceOp(SimpleTensor<T> &a, ReduceOp operation) {


    int threads = 256;
    int blocks = (a.getSize() + threads - 1) / threads; // how many blocks i really need based on thread size

    bool needsGrad = a.getRequiresGrad();

    SimpleTensor<T> partialSums({blocks}, 1, needsGrad); // tensor made from the same shape
    

    reduceKernel<<<blocks, threads>>>(a.getBuffer(), partialSums.getBuffer(), a.getSize(), operation);

    cudaDeviceSynchronize(); // making sure partialSums has proper values before reduceKernel is called again

    int blocks2 = 1;
    SimpleTensor<T> outputTensor({1}, 1, needsGrad);
    reduceKernel<<<blocks2, threads>>>(partialSums.getBuffer(), outputTensor.getBuffer(), blocks, operation);

    // backward lamdbda

    T* gradC = outputTensor.getGradBuffer();
    int inputSize = a.getSize();

    auto node = std::make_shared<GradNode>();
    node->inputs = {a.getGradNode()};
    node->backward_fn = [&a, gradC, inputSize]() {
        if (a.getRequiresGrad()) {
            int threads = 256;
            int blocks = (inputSize + threads - 1) / threads;
            accumulateGradScalar<<<blocks, threads>>>(a.getGradBuffer(), gradC, inputSize);
        }
    };

    outputTensor.setGradNode(node);

    return outputTensor;
}

template <typename T>
__global__ void naiveMatmulKernel(T* a, T* b, T* output, int N, int M, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // now we can run our computation
        T sum = 0;
        for (int i = 0; i < K; i++) {
            sum += a[row*K + i] * b[i * N + col];
        }

        output[row * N + col] = sum;
    }
}

template <typename T>
SimpleTensor<T> naiveMatmul(SimpleTensor<T> &a, SimpleTensor<T> &b) {
    // first check if its 2d - meaning dimension = 2
    if (a.getDimension() != 2 || b.getDimension() != 2) {
        throw std::invalid_argument("not 2d for matrix multiplication");
    }

    if (a.getShape()[1] != b.getShape()[0]) {
        throw std::invalid_argument("two tensor shapes dont match for proper matmul");
    }

    bool needsGrad = a.getRequiresGrad() || b.getRequiresGrad();

    int M = a.getShape()[0];
    int N = b.getShape()[1];
    int K = a.getShape()[1]; // the amount of times you have to add thru the elements in a dot prod - inner shared element

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);

    SimpleTensor<T> outputTensor = SimpleTensor<T>(std::vector<int>{M, N}, a.getDimension(), needsGrad); // locks into 2 dim
    naiveMatmulKernel<<<blocks, threads>>>(a.getBuffer(), b.getBuffer(), outputTensor.getBuffer(), N, M, K);

    return outputTensor;
}


#define TILE 16

template <typename T>
__global__ void tiledMatmulKernel(T* a, T* b, T* output, int N, int M, int K) {

    __shared__ T tileA[TILE][TILE]; // two tiles, matrix A tile
    __shared__ T tileB[TILE][TILE]; // matrix B tile -> essentially just windows over the b matrix and loads chunks into faster mem

    int row = blockIdx.y * TILE + threadIdx.y; // blockDim, the effective size of one complete block, is replaced by a tile here - as we're sliding a window across this tile for computation
    int col = blockIdx.x * TILE + threadIdx.x;

    T sum = 0; // end of current window sum

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) { // (K + TILE - 1) / tile - ceiling division of amount of tiles necessary for K elements. M & N in a and b are handleed by which blocks and threads simulataneously get those elements.
        int aCol = t * TILE + threadIdx.x; // col of choice

        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? a[row * K + aCol] : 0;

        int bRow = t * TILE + threadIdx.y; // row of choice

        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? b[bRow * N + col] : 0;
        // if your thread hands off the actual tile - tile is too big for data essentially, then we dont count it - its a 0 to the sum

        __syncthreads(); // make sure all threads store to tile before we run our compute

        for (int k = 0; k < TILE; k++) {
            // normal sum
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads(); // at the end of tile computation

    }

    if (row < M && col < N) {
        output[row * N + col] = sum;
    }

}

template <typename T>
SimpleTensor<T> tiledMatmul(SimpleTensor<T> &a, SimpleTensor<T> &b) {
    if (a.getDimension() != 2 || b.getDimension() != 2) {
        throw std::invalid_argument("not 2d for matrix multiplication");
    }

    if (a.getShape()[1] != b.getShape()[0]) {
        throw std::invalid_argument("two tensor shapes dont match for proper matmul");
    }

    bool needsGrad = a.getRequiresGrad() || b.getRequiresGrad();

    // checking tensor dimensions for proper matmul - past 2d matmul will be explored later

    int M = a.getShape()[0];
    int N = b.getShape()[1];
    int K = a.getShape()[1]; // also b.getShape()[0];

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    SimpleTensor<T> outputTensor = SimpleTensor<T>(std::vector<int>{M, N}, a.getDimension(), needsGrad); // locks into an MxN 2D
    tiledMatmulKernel<<<blocks, threads>>>(a.getBuffer(), b.getBuffer(), outputTensor.getBuffer(), N, M, K);

    // autograd backprop matmul
    T* gradC = outputTensor.getGradBuffer();
    int sizeA = a.getSize();
    int sizeB = b.getSize();
    
    
    auto node = std::make_shared<GradNode>();
    node->inputs = {a.getGradNode(), b.getGradNode()};
    node->backward_fn = [&a, &b, gradC, M, N, K, sizeA, sizeB]() {
        if (a.getRequiresGrad()) {
            T* bTranspose;
            CUDA_CHECK(cudaMalloc(&bTranspose, sizeB*sizeof(T)));
            dim3 threads(16, 16);
            dim3 blocks((b.getShape()[1] + 15) / 16, (b.getShape()[0] + 15) / 16);
            transposeMatrix<<<blocks, threads>>>(b.getBuffer(), bTranspose, b.getShape()[0], b.getShape()[1]);

            cudaDeviceSynchronize();

            T* tempA;
            CUDA_CHECK(cudaMalloc(&tempA, sizeA * sizeof(T)));

            dim3 matThreads(16, 16);
            dim3 matBlocks((K + 15) / 16, (M + 15) / 16);

            tiledMatmulKernel<<<matBlocks, matThreads>>>(gradC, bTranspose, tempA, K, M, N);
            
            int accThreads = 256;
            int accBlocks = (sizeA + accThreads - 1) / accThreads;
            accumulate<<<accBlocks, accThreads>>>(a.getGradBuffer(), tempA, sizeA);

            cudaDeviceSynchronize();

            cudaFree(tempA);
            cudaFree(bTranspose);
        }
        if (b.getRequiresGrad()) {
            T* aTranspose;
            CUDA_CHECK(cudaMalloc(&aTranspose, sizeA*sizeof(T)));
            dim3 threads(16, 16);
            dim3 blocks((a.getShape()[1] + 15) / 16, (a.getShape()[0] + 15) / 16);
            transposeMatrix<<<blocks, threads>>>(a.getBuffer(), aTranspose, a.getShape()[0], a.getShape()[1]);

            cudaDeviceSynchronize();
            T* tempB;
            CUDA_CHECK(cudaMalloc(&tempB, sizeB * sizeof(T)));

            dim3 matThreads(16, 16);
            dim3 matBlocks((N + 15) / 16, (K + 15) / 16);

            tiledMatmulKernel<<<matBlocks, matThreads>>>(aTranspose, gradC, tempB, N, K, M);

            int accThreads = 256;
            int accBlocks = (sizeB + accThreads - 1) / accThreads;
            accumulate<<<accBlocks, accThreads>>>(b.getGradBuffer(), tempB, sizeB);

            cudaDeviceSynchronize();

            cudaFree(tempB);
            cudaFree(aTranspose);
        }
    };

    outputTensor.setGradNode(node);


    return outputTensor;

}

template <typename T>
__global__ void reluForwardKernel(T* input, T* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        output[i] = max((T)0, input[i]);
    }
}

template <typename T>
__global__ void reluBackwardKernel(T* gradInput, T* gradOutput, T* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        gradInput[i] += gradOutput[i] * (input[i] > 0 ? 1 : 0);
    }
}

template <typename T>
SimpleTensor<T> reluForward(SimpleTensor<T> &a) {
    bool needsGrad = a.getRequiresGrad();

    SimpleTensor<T> outputTensor(a.getShape(), a.getDimension(), needsGrad);

    int threads = 256;
    int blocks = (a.getSize() + threads - 1) / threads;

    reluForwardKernel<<<blocks, threads>>>(a.getBuffer(), outputTensor.getBuffer(), a.getSize());

    // backward lambda
    T* gradC = outputTensor.getGradBuffer();
    T* inputData = a.getBuffer();
    int size = a.getSize();

    auto node = std::make_shared<GradNode>();
    node->inputs = {a.getGradNode()};
    node->backward_fn = [&a, gradC, inputData, size]() {
        if (a.getRequiresGrad()) {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            reluBackwardKernel<<<blocks, threads>>>(a.getGradBuffer(), gradC, inputData, size);
        }
    };

    outputTensor.setGradNode(node);


    return outputTensor;
}



template SimpleTensor<float> elementOp<float>(SimpleTensor<float>&, SimpleTensor<float>&, ElementWiseOp);
template SimpleTensor<int> elementOp<int>(SimpleTensor<int>&, SimpleTensor<int>&, ElementWiseOp);
template SimpleTensor<float> scalarOp<float>(SimpleTensor<float>&, float, ScalarOp);
template SimpleTensor<int> scalarOp<int>(SimpleTensor<int>&, int, ScalarOp);
template SimpleTensor<float> reduceOp<float>(SimpleTensor<float>&, ReduceOp);
template SimpleTensor<int> reduceOp<int>(SimpleTensor<int>&, ReduceOp);
template SimpleTensor<int> naiveMatmul<int>(SimpleTensor<int>&, SimpleTensor<int>&);
template SimpleTensor<float> naiveMatmul<float>(SimpleTensor<float>&, SimpleTensor<float>&);
template SimpleTensor<int> tiledMatmul<int>(SimpleTensor<int>&, SimpleTensor<int>&);
template SimpleTensor<float> tiledMatmul<float>(SimpleTensor<float>&, SimpleTensor<float>&);
