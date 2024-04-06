#include <vector>
#include <iostream>
#include "CudaKernels.h"
#include <chrono>

// Wrapper for computeRelativeDifferencePriorGradientKernel
void runGradientKernelOnCPUVectors(std::vector<float>& tmp_grad, const std::vector<float>& image, const std::vector<float>& weights, const std::vector<float>& kappa, const std::vector<float>& penalisation_factor, const std::vector<float>& gamma, const std::vector<float>& epsilon, const int z_dim, const int y_dim, const int x_dim) {
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

    // threads per block = (9, 9, 9)
    dim3 threadsPerBlock(9, 9, 9);
    // blocks per grid = (x_dim, y_dim, z_dim)
    dim3 blocksPerGrid(x_dim/threadsPerBlock.x + 1, y_dim/threadsPerBlock.y + 1, z_dim/threadsPerBlock.z + 1);

    // Run the kernel function
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        computeRelativeDifferencePriorGradientKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp_grad, d_image, d_weights, d_kappa, d_penalisation_factor, d_gamma, d_epsilon, z_dim, y_dim, x_dim);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "CUDA time = " << elapsed_seconds.count() << "s\n";

    // copy the result back to the host
    cudaMemcpy(tmp_grad.data(), d_tmp_grad, size, cudaMemcpyDeviceToHost);

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
void runValueKernelOnCPUVectors(std::vector<float>& tmp_value, const std::vector<float>& image, const std::vector<float>& weights, const std::vector<float>& kappa, const std::vector<float>& penalisation_factor, const std::vector<float>& gamma, const std::vector<float>& epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // Similar to the above function, allocate device memory, copy data, run the kernel, and free memory.
    // Allocate device memory
    float* d_tmp_value;
    float* d_image;
    float* d_weights;
    float* d_kappa;
    float* d_penalisation_factor;
    float* d_gamma;
    float* d_epsilon;
    size_t size = tmp_value.size() * sizeof(float);
    cudaMalloc((void**)&d_tmp_value, size);
    cudaMalloc((void**)&d_image, size);
    cudaMalloc((void**)&d_weights, size);
    cudaMalloc((void**)&d_kappa, size);
    cudaMalloc((void**)&d_penalisation_factor, size);
    cudaMalloc((void**)&d_gamma, size);
    cudaMalloc((void**)&d_epsilon, size);

    // Copy vectors from host to device memory
    cudaMemcpy(d_tmp_value, tmp_value.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kappa, kappa.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_penalisation_factor, penalisation_factor.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_epsilon, epsilon.data(), size, cudaMemcpyHostToDevice);
    // threads per block = (9, 9, 9)
    dim3 threadsPerBlock(9, 9, 9);
    // blocks per grid = (x_dim, y_dim, z_dim)
    dim3 blocksPerGrid(x_dim/threadsPerBlock.x + 1, y_dim/threadsPerBlock.y + 1, z_dim/threadsPerBlock.z + 1);
    // Run the kernel function
    computeRelativeDifferencePriorValueKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp_value, d_image, d_weights, d_kappa, d_penalisation_factor, d_gamma, d_epsilon, z_dim, y_dim, x_dim);
    // print all inputs 
    std::cout << "blockPerGrid: " << blocksPerGrid.x << " " << blocksPerGrid.y << " " << blocksPerGrid.z << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock.x << " " << threadsPerBlock.y << " " << threadsPerBlock.z << std::endl;
    std::cout << "tmp_value: " << tmp_value[0] << std::endl;
    std::cout << "image: " << image[0] << std::endl;
    std::cout << "weights: " << weights[0] << std::endl;
    std::cout << "kappa: " << kappa[0] << std::endl;
    std::cout << "penalisation_factor: " << penalisation_factor[0] << std::endl;
    std::cout << "gamma: " << gamma[0] << std::endl;
    std::cout << "epsilon: " << epsilon[0] << std::endl;

    // copy the result back to the host
    cudaMemcpy(tmp_value.data(), d_tmp_value, size, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_tmp_value);
    cudaFree(d_image);
    cudaFree(d_weights);
    cudaFree(d_kappa);
    cudaFree(d_penalisation_factor);
    cudaFree(d_gamma);
    cudaFree(d_epsilon);
}
