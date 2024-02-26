extern "C" __global__
void computeValueKernel(float* tmp_value, const float* image, const float* weights, const float* kappa, const float* penalisation_factor, const float* gamma, const float* epsilon, const int z_dim, const int y_dim, const int x_dim) {
    // tmp_value, cp_image, cp_weights, cp_penalisation_factor, cp_gamma, cp_epsilon, cp_kappa, z_dim, y_dim, x_dim
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= z_dim || y >= y_dim || x >= x_dim) return; // Boundary check

    int inputIndex = z * y_dim * x_dim + y * x_dim + x;

    float sum = 0.0f;
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
    // Apply convolution kernel hard coded 3x3x3 neighbourhood with unity weights
    for(int dz = min_dz; dz <= max_dz; dz++) {
        for(int dy = min_dy; dy <= max_dy; dy++) {
            for(int dx = min_dx; dx <= max_dx; dx++) {
                int neighbourIndex = (z + dz) * y_dim * x_dim + (y + dy) * x_dim + (x + dx);
                int weightsIndex = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);

                diff = (image[inputIndex] - image[neighbourIndex]);
                add = (image[inputIndex] + image[neighbourIndex]);
                sum += (kappa[inputIndex]*kappa[neighbourIndex])*(weights[weightsIndex]*0.5*diff*diff)/(add + gamma[0]*abs(diff) + epsilon[0]);
            }
        }
    }
    tmp_value[inputIndex] = penalisation_factor[0] * sum; 
}