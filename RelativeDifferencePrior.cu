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

__global__ void computeValueKernel(float* value, float* current_image_estimate, float* kappa, int width, int height, int depth) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= width || y >= height || z >= depth) return; // Boundary check

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

                // Boundary check for the volume
                if(nx > 0 && nx < width && ny > 0 && ny < height && nz > 0 && nz < depth) {
                    int inputIndex = nz * width * height + ny * width + nx;
                    int neighbourIndex = z * width * height +  y * width + x;
                    diff = (current_image_estimate[inputIndex] - current_image_estimate[neighbourIndex]);
                    add = (current_image_estimate[inputIndex] + current_image_estimate[neighbourIndex]);
                    sum -= pow(diff, 2)/(add + 2*abs(diff) + 1e-9);
                }
            }
        }
    }
    // Use atomicAdd to safely accumulate the sum into a global variable
    atomicAdd(value, sum);
}

torch::Tensor computeValueCuda(torch::Tensor current_image_estimate, torch::Tensor kappa) {
    AT_ASSERTM(current_image_estimate.is_cuda(), "Tensor current_image_estimate must be a CUDA tensor");
    AT_ASSERTM(kappa.is_cuda(), "Tensor kappa must be a CUDA tensor");

    int width = current_image_estimate.size(2);
    int height = current_image_estimate.size(1);
    int depth = current_image_estimate.size(0);
    // print input options
    std::cout << "Input options: " << current_image_estimate.options() << std::endl;
    auto value = torch::zeros({1}, current_image_estimate.options());
    dim3 threads = dim3(9, 9, 9);
    dim3 blocks = dim3((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y, (depth + threads.z - 1) / threads.z);
    computeValueKernel<<<blocks, threads>>>(value.data_ptr<float>(), current_image_estimate.data_ptr<float>(), kappa.data_ptr<float>(), width, height, depth);
    return value;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Element-wise addition of two tensors");
    m.def("compute_value", &computeValueCuda, "Compute RDP value");
}