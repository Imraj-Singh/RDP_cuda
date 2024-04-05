#include "stir/IO/OutputFileFormat.h"
#include "CudaRelativeDifferencePriorClass.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/error.h"
#include "stir/Succeeded.h"

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
    shared_ptr<target_type> gradient_sptr(density_sptr->get_empty_copy());

    /////// setup prior object
    CudaRelativeDifferencePriorClass<float> prior(only_2D, kappa, gamma, epsilon);
    prior.set_up(density_sptr);


    /////// Compute and add prior gradients to density_sptr
    prior.compute_gradient(*gradient_sptr, *density_sptr);
    const double my_prior_value = prior.compute_value(*density_sptr);

    std::string gradient_filename = "rdp_gradient_cpp";
    output_file_format_sptr->write_to_file(gradient_filename, *gradient_sptr);

    /////// Return the prior value
    std::cout << "The Prior Value = " << my_prior_value << "\n";
    // print kappa, gamma, epsilon
    std::cout << "kappa = " << kappa << "\n";
    std::cout << "gamma = " << gamma << "\n";
    std::cout << "epsilon = " << epsilon << "\n";
    return EXIT_SUCCESS;
}