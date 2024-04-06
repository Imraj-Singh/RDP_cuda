#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/error.h"
#include "stir/Succeeded.h"
#include "CudaRelativeDifferencePrior.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

int main() {
    // Use namespace stir
    using namespace stir;
    // Define the target type to keep things pretty
    typedef DiscretisedDensity<3, float> target_type;

    std::string image_filename  = "/home/user/sirf/brain/emission.hv";
    shared_ptr<OutputFileFormat<target_type>> output_file_format_sptr = OutputFileFormat<target_type>::default_sptr();
    shared_ptr<target_type> density_sptr(read_from_file<target_type>(image_filename));
    const DiscretisedDensityOnCartesianGrid<3, float>& density_cast 
        = dynamic_cast<const DiscretisedDensityOnCartesianGrid<3, float>&>(*density_sptr);

    // Get the number of voxels in each dimension
    
    int z_dim = density_cast.get_max_index() - density_cast.get_min_index() + 1;
    int y_dim = density_cast[0].get_max_index() - density_cast[0].get_min_index() + 1;
    int x_dim = density_cast[0][0].get_max_index() - density_cast[0][0].get_min_index() + 1;

    // print voxel dims 
    std::cout << "z_dim: " << z_dim << std::endl;
    std::cout << "y_dim: " << y_dim << std::endl;
    std::cout << "x_dim: " << x_dim << std::endl;
    std::cout << "total_dim: " << density_cast.size_all() << std::endl;

    // Assume voxel_sizes is a std::vector<float> of size 3
    const CartesianCoordinate3D<float>& grid_spacing = density_cast.get_grid_spacing();
    float grid_spacing_x = grid_spacing.x();
    float grid_spacing_y = grid_spacing.y();
    float grid_spacing_z = grid_spacing.z();
    std::cout << "grid_spacing_z: " << grid_spacing_z << std::endl;
    std::cout << "grid_spacing_y: " << grid_spacing_y << std::endl;
    std::cout << "grid_spacing_x: " << grid_spacing_x << std::endl;

    // Create a 3D vector for cp_weights
    std::vector<float> weights(27, 0.0f);
    int index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                index = i * 3 * 3 + j * 3 + k;
                weights[index] = grid_spacing_x / std::sqrt(std::pow((i - 1) * grid_spacing_z, 2) + std::pow((j - 1) * grid_spacing_y, 2) + std::pow((k - 1) * grid_spacing_x, 2));
                if (i == 1 && j == 1 && k == 1) {
                    weights[index] = 0.0f;
                }
                //std::cout << "weights[" << index << "]: " << weights[index] << std::endl;
            }
        }
    }

    // Create some dummy data
    std::vector<float> tmp_grad(density_cast.size_all(), 0.0f);
    std::vector<float> image(density_cast.size_all(), 0.0f);
    // interate over all dimensions putting image values into the vector
    int index_image = 0;
    for (int z = density_cast.get_min_index(); z <= density_cast.get_max_index(); ++z) {
        for (int y = density_cast[z].get_min_index(); y <= density_cast[z].get_max_index(); ++y) {
            for (int x = density_cast[z][y].get_min_index(); x <= density_cast[z][y].get_max_index(); ++x) {
                image[index_image] = density_cast[z][y][x];
                index_image++;
            }
        }
    }
    
    std::vector<float> kappa(density_cast.size_all(), 1.0f);
    std::vector<float> penalisation_factor(1, 1.0f);
    std::vector<float> gamma(1, 2.f);
    std::vector<float> epsilon(1, 1e-9f);

    // Call the gradient kernel wrapper
    runGradientKernelOnCPUVectors(tmp_grad, image, weights, kappa, penalisation_factor, gamma, epsilon, z_dim, y_dim, x_dim);
    
    // clone density_sptr and fill it with the gradient values
    shared_ptr<target_type> gradient_sptr(density_sptr->get_empty_copy());
    DiscretisedDensityOnCartesianGrid<3, float>& gradient_cast 
        = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, float>&>(*gradient_sptr);
    index_image = 0;
    for (int z = gradient_cast.get_min_index(); z <= gradient_cast.get_max_index(); ++z) {
        for (int y = gradient_cast[z].get_min_index(); y <= gradient_cast[z].get_max_index(); ++y) {
            for (int x = gradient_cast[z][y].get_min_index(); x <= gradient_cast[z][y].get_max_index(); ++x) {
                gradient_cast[z][y][x] = tmp_grad[index_image];
                index_image++;
            }
        }
    }
    
    //std::string gradient_filename = "rdp_gradient_cuda";
    //output_file_format_sptr->write_to_file(gradient_filename, *gradient_sptr);

    std::vector<float> tmp_value(density_cast.size_all(), 0.0f);

    // Call the value kernel wrapper
    runValueKernelOnCPUVectors(tmp_value, image, weights, kappa, penalisation_factor, gamma, epsilon, z_dim, y_dim, x_dim);

    float sum = 0.0f;
    for (int i = 0; i < tmp_value.size(); ++i) {
        sum += tmp_value[i];
    }
    std::cout << "The Prior Value = " << sum << "\n";
    return 0;
}