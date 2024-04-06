#include "CudaRelativeDifferencePriorClass.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"

extern "C" __global__
void computeCudaRelativeDifferencePriorGradientKernel(float* tmp_grad, const float* image, const float* weights, const float gamma, const float epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // tmp_grad, cp_image, cp_weights, cp_kappa, cp_penalisation_factor, cp_gamma, cp_epsilon, z_dim, y_dim, x_dim
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (z >= z_dim || y >= y_dim || x >= x_dim) return; // Boundary check

    int inputIndex = z * y_dim * x_dim + y * x_dim + x;

    double voxel_gradient = 0.0f;
    float diff = 0.0f;
    float diff_abs = 0.0f;
    float add_3 = 0.0f;
    float add = 0.0f;

    // Define the neighbourhood
    int min_dz = -1;
    int max_dz = 1;
    int min_dy = -1;
    int max_dy = 1;
    int min_dx = -1;
    int max_dx = 1;

    if (z == 0) min_dz = 0;
    if (z == z_dim - 1) max_dz = 0;
    if (y == 0) min_dy = 0;
    if (y == y_dim - 1) max_dy = 0;
    if (x == 0) min_dx = 0;
    if (x == x_dim - 1) max_dx = 0;

    // Apply convolution kernel hard coded 3x3x3 neighbourhood with unity weights
    for(int dz = min_dz; dz <= max_dz; dz++) {
        for(int dy = min_dy; dy <= max_dy; dy++) {
            for(int dx = min_dx; dx <= max_dx; dx++) {
                int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
                int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                diff = (image[inputIndex] - image[neighbourIndex]);
                diff_abs = abs(diff);
                add = (image[inputIndex] + image[neighbourIndex]);
                add_3 = (image[inputIndex] + 3*image[neighbourIndex]);
                voxel_gradient += weights[weightsIndex]*(diff*(gamma*diff_abs + add_3))/((add + gamma*diff_abs + epsilon)*(add + gamma*diff_abs + epsilon));
            }
        }
    }
    tmp_grad[inputIndex] = voxel_gradient;
}

extern "C" __global__
void computeCudaRelativeDifferencePriorValueKernel(double* tmp_value, const float* image, const float* weights, const float gamma, const float epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // tmp_value, cp_image, cp_weights, cp_penalisation_factor, cp_gamma, cp_epsilon, cp_kappa, z_dim, y_dim, x_dim
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (z >= z_dim || y >= y_dim || x >= x_dim) return; // Boundary check

    int inputIndex = z * y_dim * x_dim + y * x_dim + x;

    double sum = 0.0f;
    float diff = 0.0f;
    float add = 0.0f;

    // Define the neighbourhood
    int min_dz = -1;
    int max_dz = 1;
    int min_dy = -1;
    int max_dy = 1;
    int min_dx = -1;
    int max_dx = 1;

    if (z == 0) min_dz = 0;
    if (z == z_dim - 1) max_dz = 0;
    if (y == 0) min_dy = 0;
    if (y == y_dim - 1) max_dy = 0;
    if (x == 0) min_dx = 0;
    if (x == x_dim - 1) max_dx = 0;
    // Apply convolution kernel hard coded 3x3x3 neighbourhood
    for(int dz = min_dz; dz <= max_dz; dz++) {
        for(int dy = min_dy; dy <= max_dy; dy++) {
            for(int dx = min_dx; dx <= max_dx; dx++) {
                int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
                int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);

                diff = (image[inputIndex] - image[neighbourIndex]);
                add = (image[inputIndex] + image[neighbourIndex]);
                sum += (weights[weightsIndex]*0.5*diff*diff)/(add + gamma*abs(diff) + epsilon);
            }
        }
    }
    tmp_value[inputIndex] = sum;
}

START_NAMESPACE_STIR


static void
compute_weights(Array<3, float>& weights, const CartesianCoordinate3D<float>& grid_spacing, const bool only_2D)
{
  int min_dz, max_dz;
  if (only_2D)
    {
      min_dz = max_dz = 0;
    }
  else
    {
      min_dz = -1;
      max_dz = 1;
    }
  weights = Array<3, float>(IndexRange3D(min_dz, max_dz, -1, 1, -1, 1));
  for (int z = min_dz; z <= max_dz; ++z)
    for (int y = -1; y <= 1; ++y)
      for (int x = -1; x <= 1; ++x)
        {
          if (z == 0 && y == 0 && x == 0)
            weights[0][0][0] = 0;
          else
            {
              weights[z][y][x]
                  = grid_spacing.x()
                    / sqrt(square(x * grid_spacing.x()) + square(y * grid_spacing.y()) + square(z * grid_spacing.z()));
            }
        }
}


template <typename elemT>
double CudaRelativeDifferencePriorClass<elemT>::compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) {
    // Assuming z_dim, y_dim, and x_dim are correctly set
    const int z_dim = this->z_dim;
    const int y_dim = this->y_dim;
    const int x_dim = this->x_dim;

    std::vector<float> image_data(current_image_estimate.size_all());
    std::vector<float> weights_data(this->weights.size_all());
    std::copy(current_image_estimate.begin_all(), current_image_estimate.end_all(), image_data.begin());
    std::copy(this->weights.begin_all(), this->weights.end_all(), weights_data.begin());

    // GPU memory pointers
    float *d_image_data, *d_weights_data;
    double *d_tmp_value;

    // Allocate memory on the GPU
    cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(float));
    cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float)); // Assuming weights is also a flat vector
    cudaMalloc(&d_tmp_value, current_image_estimate.size_all() * sizeof(double));
    
    // Copy data from host to device
    cudaMemcpy(d_image_data, image_data.data(), current_image_estimate.size_all() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_data, weights_data.data(), this->weights.size_all() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    computeCudaRelativeDifferencePriorValueKernel<<<this->grid_dim, this->block_dim>>>(
        d_tmp_value, d_image_data, d_weights_data, this->gamma, this->epsilon, z_dim, y_dim, x_dim
    );

    // Check for any errors during kernel execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in compute_value kernel execution: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_image_data); cudaFree(d_weights_data); cudaFree(d_tmp_value);
        return 0.0; // Handle error appropriately
    }

    // Allocate host memory for the result and copy from device to host
    std::vector<double> tmp_value(current_image_estimate.size_all());
    cudaMemcpy(tmp_value.data(), d_tmp_value, current_image_estimate.size_all() * sizeof(double), cudaMemcpyDeviceToHost);

    // Compute the total value from tmp_value if necessary
    double totalValue = std::accumulate(tmp_value.begin(), tmp_value.end(), 0.0);

    // Cleanup
    cudaFree(d_image_data);
    cudaFree(d_weights_data);
    cudaFree(d_tmp_value);

    return totalValue;
}


template <typename elemT>
void CudaRelativeDifferencePriorClass<elemT>::compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient, const DiscretisedDensity<3, elemT>& current_image_estimate) {
    
    assert(prior_gradient.has_same_characteristics(current_image_estimate));
    
    const int z_dim = this->z_dim;
    const int y_dim = this->y_dim;
    const int x_dim = this->x_dim;

    std::vector<float> image_data(current_image_estimate.size_all());
    std::vector<float> weights_data(this->weights.size_all());
    std::copy(current_image_estimate.begin_all(), current_image_estimate.end_all(), image_data.begin());
    std::copy(this->weights.begin_all(), this->weights.end_all(), weights_data.begin());

    float *d_image_data, *d_weights_data, *d_gradient_data;

    // Allocate memory on the GPU
    cudaMalloc(&d_image_data, current_image_estimate.size_all() * sizeof(float));
    cudaMalloc(&d_weights_data, this->weights.size_all() * sizeof(float));
    cudaMalloc(&d_gradient_data, prior_gradient.size_all() * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_image_data, image_data.data(), current_image_estimate.size_all() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_data, weights_data.data(), this->weights.size_all() * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    computeCudaRelativeDifferencePriorGradientKernel<<<this->grid_dim, this->block_dim>>>(
        d_gradient_data, d_image_data, d_weights_data, this->gamma, this->epsilon, z_dim, y_dim, x_dim
    );

    // Check for any errors during kernel execution
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in compute_value kernel execution: " << cudaGetErrorString(error) << std::endl;
        cudaFree(d_image_data); cudaFree(d_weights_data); cudaFree(d_gradient_data);
    }

    // Allocate host memory for the result and copy from device to host
    std::vector<float> gradient_data(current_image_estimate.size_all());
    cudaMemcpy(gradient_data.data(), d_gradient_data, current_image_estimate.size_all() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Copy the gradient data to the prior_gradient
    std::copy(gradient_data.begin(), gradient_data.end(), prior_gradient.begin_all());

    // Cleanup
    cudaFree(d_image_data);
    cudaFree(d_weights_data);
    cudaFree(d_gradient_data);
}

template <typename elemT>
Succeeded CudaRelativeDifferencePriorClass<elemT>::set_up(shared_ptr<DiscretisedDensity<3, elemT>> const& target_sptr) {
    // Optionally call RelativeDifferencePrior's set_up if it adds value
    if (RelativeDifferencePrior<elemT>::set_up(target_sptr) == Succeeded::no) {
        return Succeeded::no;
    }
    // Get the number of voxels in each dimension
    const DiscretisedDensityOnCartesianGrid<3, float>& target_cast 
        = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>&>(*target_sptr);

    this->z_dim = target_cast.get_max_index() - target_cast.get_min_index() + 1;
    this->y_dim = target_cast[0].get_max_index() - target_cast[0].get_min_index() + 1;
    this->x_dim = target_cast[0][0].get_max_index() - target_cast[0][0].get_min_index() + 1;

    // set the thread block and grid dimensions
    this->block_dim = dim3(8, 8, 8);
    this->grid_dim = dim3((this->x_dim + this->block_dim.x - 1) / this->block_dim.x, (this->y_dim + this->block_dim.y - 1) / this->block_dim.y, (this->z_dim + this->block_dim.z - 1) / this->block_dim.z);
    // Check if z_dim is 1 or only 2D is trueand return an error if it is
    if (this->z_dim == 1 || this->only_2D) {
        std::cerr << "CudaRelativeDifferencePriorClass: This prior requires a 3D image" << std::endl;
        return Succeeded::no;
    }    
    compute_weights(this->weights, target_cast.get_grid_spacing(), this->only_2D);
    std::cout << "z_dim: " << z_dim << " y_dim: " << y_dim << " x_dim: " << x_dim << std::endl;
    return Succeeded::yes;
}


// Explicit template instantiations
template class stir::CudaRelativeDifferencePriorClass<float>;
template <typename elemT>
const char* const CudaRelativeDifferencePriorClass<elemT>::registered_name = "Cuda Relative Difference Prior";
template <typename elemT>
CudaRelativeDifferencePriorClass<elemT>::CudaRelativeDifferencePriorClass() : RelativeDifferencePrior<elemT>() {}
template <typename elemT>
CudaRelativeDifferencePriorClass<elemT>::CudaRelativeDifferencePriorClass(const bool only_2D, float penalization_factor, float gamma, float epsilon) : RelativeDifferencePrior<elemT>(only_2D, penalization_factor, gamma, epsilon) {}

END_NAMESPACE_STIR