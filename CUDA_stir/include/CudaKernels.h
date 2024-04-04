#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

extern "C" {
    __global__ void computeRelativeDifferencePriorGradientKernel(float* tmp_grad, const float* image, const float* weights, const float* kappa, const float* penalisation_factor, const float* gamma, const float* epsilon, const int z_dim, const int y_dim, const int x_dim);
    __global__ void computeRelativeDifferencePriorValueKernel(float* tmp_value, const float* image, const float* weights, const float* kappa, const float* penalisation_factor, const float* gamma, const float* epsilon, const int z_dim, const int y_dim, const int x_dim);
}

#endif // CUDA_KERNELS_H