#include <torch/extension.h>

__global__ void add_kernel(const float *a, const float *b, float *c, size_t N) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    const auto N = a.numel();
    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;

    AT_ASSERTM(a.is_cuda(), "Tensor a must be a CUDA tensor");
    AT_ASSERTM(b.is_cuda(), "Tensor b must be a CUDA tensor");
    add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);

    return c;
}

__global__ void computeValueKernel(float* temp_sum, float* current_image_estimate, float* kappa, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return; // Boundary check

    int inputIndex = z * width * height + y * width + x;

    float sum = 0.0f;
    float diff = 0.0f;
    float add = 0.0f;

    // Apply convolution kernel hard coded 3x3x3 neighbourhood with unity weights
    for(int dz = -1; dz <= 1; dz++) {
        for(int dy = -1; dy <= 1; dy++) {
            for(int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                // Clamp to image boundary, i.e. replicate boundary
                nx = min(max(nx, 0), width - 1); 
                ny = min(max(ny, 0), height - 1);
                nz = min(max(nz, 0), depth - 1);

                int neighbourIndex = nz * width * height + ny * width + nx;

                diff = (current_image_estimate[inputIndex] - current_image_estimate[neighbourIndex]);
                add = (current_image_estimate[inputIndex] + current_image_estimate[neighbourIndex]);
                sum -= pow(diff, 2)/(add + 2*abs(diff) + 1e-9);
                
            }
        }
    }
    // Use atomicAdd to safely accumulate the sum into a global variable
    temp_sum[inputIndex] = sum; // * kappa[inputIndex];
}

torch::Tensor computeValueCuda(torch::Tensor current_image_estimate, torch::Tensor kappa) {
    AT_ASSERTM(current_image_estimate.is_cuda(), "Tensor current_image_estimate must be a CUDA tensor");
    AT_ASSERTM(kappa.is_cuda(), "Tensor kappa must be a CUDA tensor");
    int width = current_image_estimate.size(2);
    int height = current_image_estimate.size(1);
    int depth = current_image_estimate.size(0);

    // Create a temporary array to store the sum of the values
    auto options = current_image_estimate.options();
    auto temp_sum = torch::zeros({depth, height, width}, options);
    
    // Launch the kernel
    dim3 threads = dim3(9, 9, 9);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, (depth + threads.z - 1) / threads.z);
    computeValueKernel<<<blocks, threads>>>(temp_sum.data_ptr<float>(), current_image_estimate.data_ptr<float>(), kappa.data_ptr<float>(), width, height, depth);
    
    // Ensure the kernel execution completes
    cudaDeviceSynchronize();

    // Sum the temporary array values
    auto value = torch::sum(temp_sum);

    return value;
}

__global__ void computeGradientKernel(float* gradient, float* current_image_estimate, float* kappa, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return; // Boundary check

    int inputIndex = z * width * height + y * width + x;

    float voxel_gradient = 0.0f;
    float diff = 0.0f;
    float diff_abs = 0.0f;
    float add_3 = 0.0f;
    float add = 0.0f;

    // Apply convolution kernel hard coded 3x3x3 neighbourhood with unity weights
    for(int dz = -1; dz <= 1; dz++) {
        for(int dy = -1; dy <= 1; dy++) {
            for(int dx = -1; dx <= 1; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;
                // Clamp to image boundary, i.e. replicate boundary
                nx = min(max(nx, 0), width - 1); 
                ny = min(max(ny, 0), height - 1);
                nz = min(max(nz, 0), depth - 1);

                int neighbourIndex = nz * width * height + ny * width + nx;

                diff = (current_image_estimate[inputIndex] - current_image_estimate[neighbourIndex]);
                diff_abs = abs(diff);
                add = (current_image_estimate[inputIndex] + current_image_estimate[neighbourIndex]);
                add_3 = (current_image_estimate[inputIndex] + 3*current_image_estimate[neighbourIndex]);
                voxel_gradient -= (diff*(2*diff_abs + add_3))/pow(add + 2*diff_abs + 1e-9, 2);
                
            }
        }
    }
    // Use atomicAdd to safely accumulate the sum into a global variable
    gradient[inputIndex] = voxel_gradient; // * kappa[inputIndex];
}

torch::Tensor computeGradientCuda(torch::Tensor current_image_estimate, torch::Tensor kappa) {
    AT_ASSERTM(current_image_estimate.is_cuda(), "Tensor current_image_estimate must be a CUDA tensor");
    AT_ASSERTM(kappa.is_cuda(), "Tensor kappa must be a CUDA tensor");
    int width = current_image_estimate.size(2);
    int height = current_image_estimate.size(1);
    int depth = current_image_estimate.size(0);

    // Create a temporary array to store the sum of the values
    auto options = current_image_estimate.options();
    auto gradient = torch::zeros({depth, height, width}, options);
    
    // Launch the kernel
    dim3 threads = dim3(9, 9, 9);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, (depth + threads.z - 1) / threads.z);
    computeGradientKernel<<<blocks, threads>>>(gradient.data_ptr<float>(), current_image_estimate.data_ptr<float>(), kappa.data_ptr<float>(), width, height, depth);
    
    // Ensure the kernel execution completes
    cudaDeviceSynchronize();

    return gradient;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Element-wise addition of two tensors");
    m.def("compute_value", &computeValueCuda, "Compute RDP value");
    m.def("compute_gradient", &computeGradientCuda, "Compute RDP gradient");
}