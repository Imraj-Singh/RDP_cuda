#include <vector>
#include "CudaKernels.h"

// Wrapper for computeRelativeDifferencePriorGradientKernel
void runGradientKernelOnCPUVectors(const std::vector<float>& tmp_grad, const std::vector<float>& image, const std::vector<float>& weights, const std::vector<float>& kappa, const std::vector<float>& penalisation_factor, const std::vector<float>& gamma, const std::vector<float>& epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // Allocate device memory
    float* d_tmp_grad;
    float* d_image;
    float* d_weights;
    float* d_kappa;
    float* d_penalisation_factor;
    float* d_gamma;
    float* d_epsilon;
    size_t size = tmp_grad.size() * sizeof(float);
    cudaMalloc((void**)&d_tmp_grad, size);
    cudaMalloc((void**)&d_image, size);
    cudaMalloc((void**)&d_weights, size);
    cudaMalloc((void**)&d_kappa, size);
    cudaMalloc((void**)&d_penalisation_factor, size);
    cudaMalloc((void**)&d_gamma, size);
    cudaMalloc((void**)&d_epsilon, size);

    // Copy vectors from host to device memory
    cudaMemcpy(d_tmp_grad, tmp_grad.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kappa, kappa.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_penalisation_factor, penalisation_factor.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_epsilon, epsilon.data(), size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (tmp_grad.size() + threadsPerBlock - 1) / threadsPerBlock;

    // Run the kernel function
    computeRelativeDifferencePriorGradientKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp_grad, d_image, d_weights, d_kappa, d_penalisation_factor, d_gamma, d_epsilon, z_dim, y_dim, x_dim);

    // Free device memory
    cudaFree(d_tmp_grad);
    cudaFree(d_image);
    cudaFree(d_weights);
    cudaFree(d_kappa);
    cudaFree(d_penalisation_factor);
    cudaFree(d_gamma);
    cudaFree(d_epsilon);
}

// Wrapper for computeRelativeDifferencePriorValueKernel
void runValueKernelOnCPUVectors(const std::vector<float>& tmp_value, const std::vector<float>& image, const std::vector<float>& weights, const std::vector<float>& kappa, const std::vector<float>& penalisation_factor, const std::vector<float>& gamma, const std::vector<float>& epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // Similar to the above function, allocate device memory, copy data, run the kernel, and free memory.
    // Allocate device memory
    float* d_tmp_grad;
    float* d_image;
    float* d_weights;
    float* d_kappa;
    float* d_penalisation_factor;
    float* d_gamma;
    float* d_epsilon;
    size_t size = tmp_value.size() * sizeof(float);
    cudaMalloc((void**)&d_tmp_grad, size);
    cudaMalloc((void**)&d_image, size);
    cudaMalloc((void**)&d_weights, size);
    cudaMalloc((void**)&d_kappa, size);
    cudaMalloc((void**)&d_penalisation_factor, size);
    cudaMalloc((void**)&d_gamma, size);
    cudaMalloc((void**)&d_epsilon, size);

    // Copy vectors from host to device memory
    cudaMemcpy(d_tmp_grad, tmp_value.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kappa, kappa.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_penalisation_factor, penalisation_factor.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_epsilon, epsilon.data(), size, cudaMemcpyHostToDevice);

    // Calculate the number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (tmp_value.size() + threadsPerBlock - 1) / threadsPerBlock;

    // Run the kernel function
    computeRelativeDifferencePriorValueKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp_grad, d_image, d_weights, d_kappa, d_penalisation_factor, d_gamma, d_epsilon, z_dim, y_dim, x_dim);

    // Free device memory
    cudaFree(d_tmp_grad);
    cudaFree(d_image);
    cudaFree(d_weights);
    cudaFree(d_kappa);
    cudaFree(d_penalisation_factor);
    cudaFree(d_gamma);
    cudaFree(d_epsilon);
}