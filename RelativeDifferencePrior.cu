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

__global__ void computeValueKerneltest(int min_z, int max_z, int min_y, int max_y, int min_x, int max_x) {
    int z = blockIdx.z * blockDim.z + threadIdx.z + min_z;
    int y = blockIdx.y * blockDim.y + threadIdx.y + min_y;
    int x = blockIdx.x * blockDim.x + threadIdx.x + min_x;
    printf("Hello from the kernel!\n");
    printf("z: %d, y: %d, x: %d\n", z, y, x);
    if (z > max_z || y > max_y || x > max_x) return; // Check bounds
}

float rdp_compute_val_cudatest(torch::Tensor img) {
    AT_ASSERTM(img.is_cuda(), "Tensor a must be a CUDA tensor");
    int max_z = img.size(0) - 1; // Depth
    int max_y = img.size(1) - 1; // Height
    int max_x = img.size(2) - 1; // Width
    double val = 0.0;
    dim3 threads = dim3(max_z, max_y, max_x);
    dim3 blocks = dim3(1, 1, 1);
    computeValueKerneltest<<<blocks, threads>>>(0, max_z, 0, max_y, 0, max_x);
    return val;
}

__global__ void convolveAndSumKernel(float* input, float* globalSum, int width, int height, int depth, const float* convKernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return; // Boundary check

    float sum = 0.0f;
    int kernelRadius = 1; // For a 3x3x3 kernel

    // Apply convolution kernel
    for(int dz = -kernelRadius; dz <= kernelRadius; dz++) {
        for(int dy = -kernelRadius; dy <= kernelRadius; dy++) {
            for(int dx = -kernelRadius; dx <= kernelRadius; dx++) {
                int nx = x + dx;
                int ny = y + dy;
                int nz = z + dz;

                // Boundary check for the volume
                if(nx > 0 && nx < width-1 && ny > 0 && ny < height-1 && nz > 0 && nz < depth-1) {
                    int kernelIndex = (dz + kernelRadius) * 9 + (dy + kernelRadius) * 3 + (dx + kernelRadius);
                    int inputIndex = nz * width * height + ny * width + nx;
                    sum += input[inputIndex] * convKernel[kernelIndex];
                }
            }
        }
    }
    // Use atomicAdd to safely accumulate the sum into a global variable
    atomicAdd(globalSum, sum);
}

torch::Tensor convolveAndSumCuda(torch::Tensor input, torch::Tensor convKernel) {
    AT_ASSERTM(input.is_cuda(), "Tensor a must be a CUDA tensor");
    AT_ASSERTM(convKernel.is_cuda(), "Tensor b must be a CUDA tensor");

    int width = input.size(2);
    int height = input.size(1);
    int depth = input.size(0);
    // print input options
    std::cout << "Input options: " << input.options() << std::endl;
    auto globalSum = torch::zeros({1}, input.options());
    dim3 threads = dim3(10, 10, 10);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, (depth + threads.z - 1) / threads.z);
    convolveAndSumKernel<<<blocks, threads>>>(input.data_ptr<float>(), globalSum.data_ptr<float>(), width, height, depth, convKernel.data_ptr<float>());

    return globalSum;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Element-wise addition of two tensors");
    m.def("rdp_compute_val", &rdp_compute_val_cudatest, "Compute values of RDP");
    m.def("convolveAndSum", &convolveAndSumCuda, "Convolve and sum");
}