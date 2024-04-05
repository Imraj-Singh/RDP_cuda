#ifndef CUDA_RELATIVE_DIFFERENCE_PRIOR_H
#define CUDA_RELATIVE_DIFFERENCE_PRIOR_H

#include <vector>

void runGradientKernelOnCPUVectors(std::vector<float>& tmp_grad, const std::vector<float>& image, const std::vector<float>& weights, const std::vector<float>& kappa, const std::vector<float>& penalisation_factor, const std::vector<float>& gamma, const std::vector<float>& epsilon, const int z_dim, const int y_dim, const int x_dim);

void runValueKernelOnCPUVectors(std::vector<float>& tmp_value, const std::vector<float>& image, const std::vector<float>& weights, const std::vector<float>& kappa, const std::vector<float>& penalisation_factor, const std::vector<float>& gamma, const std::vector<float>& epsilon, const int z_dim, const int y_dim, const int x_dim);
#endif