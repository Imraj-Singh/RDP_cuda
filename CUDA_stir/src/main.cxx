#include "stir/IO/OutputFileFormat.h"
#include "CudaRelativeDifferencePriorClass.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/error.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <chrono>

int main()
{
    // Use namespace stir
    using namespace stir;
    // Define the target type to keep things pretty
    typedef DiscretisedDensity<3, float> target_type;

    std::string image_filename  = "/home/user/sirf/brain/emission.hv";
    shared_ptr<OutputFileFormat<target_type>> output_file_format_sptr = OutputFileFormat<target_type>::default_sptr();
    bool only_2D = false;
    float kappa = 1.0f;
    float gamma = 2.0f;
    float epsilon = 1e-9f;

    /////// load initial density from file
    shared_ptr<target_type> density_sptr(read_from_file<target_type>(image_filename));

    //////// gradient it copied Density filled with 0's
    shared_ptr<target_type> gradient_cuda_sptr(density_sptr->get_empty_copy());
    shared_ptr<target_type> gradient_cpu_sptr(density_sptr->get_empty_copy());

    /////// setup prior objects
    CudaRelativeDifferencePriorClass<float> prior_cuda(only_2D, kappa, gamma, epsilon);
    prior_cuda.set_up(density_sptr);

    RelativeDifferencePrior<float> prior_cpu(only_2D, kappa, gamma, epsilon);
    prior_cpu.set_up(density_sptr);

    /////// Compute and add prior gradients to density_sptr
    prior_cuda.compute_gradient(*gradient_cuda_sptr, *density_sptr);
    double cuda_prior_value = prior_cuda.compute_value(*density_sptr);

    prior_cpu.compute_gradient(*gradient_cpu_sptr, *density_sptr);
    double cpu_prior_value = prior_cpu.compute_value(*density_sptr);


    std::string gradient_cuda_filename = "rdp_gradient_cuda";
    output_file_format_sptr->write_to_file(gradient_cuda_filename, *gradient_cuda_sptr);
    std::string gradient_cpu_filename = "rdp_gradient_cpu";
    output_file_format_sptr->write_to_file(gradient_cpu_filename, *gradient_cpu_sptr);

    // Print two prior values and difference
    std::cout << "The Prior Value (CUDA) = " << cuda_prior_value << "\n";
    std::cout << "The Prior Value (CPU) = " << cpu_prior_value << "\n";
    std::cout << "Difference = " << cuda_prior_value - cpu_prior_value << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        prior_cuda.compute_gradient(*gradient_cuda_sptr, *density_sptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "CUDA time = " << elapsed_seconds.count() << "s\n";

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; i++)
    {
        prior_cpu.compute_gradient(*gradient_cpu_sptr, *density_sptr);
    }
    end = std::chrono::high_resolution_clock::now();
    elapsed_seconds = end - start;
    std::cout << "CPU time = " << elapsed_seconds.count() << "s\n";

    return EXIT_SUCCESS;
}