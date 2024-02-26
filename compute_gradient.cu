extern "C" __global__
void computeGradientKernel(float* tmp_grad, const float* image, const float* weights, const float* kappa, const float* penalisation_factor, const float* gamma, const float* epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // tmp_grad, cp_image, cp_weights, cp_kappa, cp_penalisation_factor, cp_gamma, cp_epsilon, z_dim, y_dim, x_dim
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (z >= z_dim || y >= y_dim || x >= x_dim) return; // Boundary check

    int inputIndex = z * y_dim * x_dim + y * x_dim + x;

    float voxel_gradient = 0.0f;
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
                voxel_gradient += (kappa[inputIndex]*kappa[neighbourIndex])*weights[weightsIndex]*(diff*(gamma[0]*diff_abs + add_3))/((add + gamma[0]*diff_abs + epsilon[0])*(add + gamma[0]*diff_abs + epsilon[0]));
            }
        }
    }
    tmp_grad[inputIndex] = penalisation_factor[0] * voxel_gradient;
}