#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/error.h"
#include "stir/Succeeded.h"
#include "CudaRelativeDifferencePrior.h"
#include <vector>
#include <cmath>

int main() {
    // Use namespace stir
    using namespace stir;
    // Define the target type to keep things pretty
    typedef DiscretisedDensity<3, float> target_type;

    std::string image_filename  = "/home/user/sirf/brain/emission.hv";
    shared_ptr<OutputFileFormat<target_type>> output_file_format_sptr = OutputFileFormat<target_type>::default_sptr();
    shared_ptr<target_type> density_sptr(read_from_file<target_type>(image_filename));

    // Get the number of voxels in each dimension
    
    int z_dim = (*density_sptr).get_max_index() - (*density_sptr).get_min_index();
    int y_dim = (*density_sptr)[0].get_max_index() - (*density_sptr)[0].get_min_index();
    int x_dim = (*density_sptr)[0][0].get_max_index() - (*density_sptr)[0][0].get_min_index();

    int total_dim = (z_dim+1) * (y_dim+1) * (x_dim+1);

    // print voxel dims 
    std::cout << "z_dim: " << z_dim << std::endl;
    std::cout << "y_dim: " << y_dim << std::endl;
    std::cout << "x_dim: " << x_dim << std::endl;
    std::cout << "total_dim: " << total_dim << std::endl;

    // Assume voxel_sizes is a std::vector<float> of size 3
    std::vector<float> voxel_sizes = image.voxel_sizes();

    // Create a 3D vector for cp_weights
    std::vector<std::vector<std::vector<float>>> cp_weights(3, std::vector<std::vector<float>>(3, std::vector<float>(3, 0.0f)));

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                cp_weights[i][j][k] = voxel_sizes[2] / std::sqrt(std::pow((i - 1) * voxel_sizes[0], 2) + std::pow((j - 1) * voxel_sizes[1], 2) + std::pow((k - 1) * voxel_sizes[2], 2));
            }
        }
    }

    // Create some dummy data
    std::vector<float> tmp_grad(total_dim, 1.0f);
    std::vector<float> image(total_dim, 2.0f);

    std::vector<float> weights(1000, 3.0f);
    std::vector<float> kappa(1000, 4.0f);
    std::vector<float> penalisation_factor(1000, 5.0f);
    std::vector<float> gamma(1000, 6.0f);
    std::vector<float> epsilon(1000, 1e-9f);

    // Call the gradient kernel wrapper
    runGradientKernelOnCPUVectors(tmp_grad, image, weights, kappa, penalisation_factor, gamma, epsilon, z_dim, y_dim, x_dim);

    // Create some more dummy data for the value kernel
    std::vector<float> tmp_value(1000, 8.0f);

    // Call the value kernel wrapper
    runValueKernelOnCPUVectors(tmp_value, image, weights, kappa, penalisation_factor, gamma, epsilon, z_dim, y_dim, x_dim);
    std::cout << "Success!" << std::endl;
    // print all the values in the vector
    for (int i = 0; i < 1; i++) {
        std::cout << tmp_value[i] << std::endl;
    }

    return 0;
}