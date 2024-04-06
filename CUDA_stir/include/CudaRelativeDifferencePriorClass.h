#ifndef CUDA_RELATIVE_DIFFERENCE_PRIOR_CLASS_H
#define CUDA_RELATIVE_DIFFERENCE_PRIOR_CLASS_H

#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/DiscretisedDensity.h"
#include <cuda_runtime.h>

START_NAMESPACE_STIR

template <typename elemT>
class CudaRelativeDifferencePriorClass : public RelativeDifferencePrior<elemT> {
    public:
        // Name which will be used when parsing a GeneralisedPrior object
        static const char* const registered_name;

        // Constructors
        CudaRelativeDifferencePriorClass();
        CudaRelativeDifferencePriorClass(const bool only_2D, float penalization_factor, float gamma, float epsilon);

        // Overridden methods
        virtual Succeeded set_up(shared_ptr<DiscretisedDensity<3, elemT>> const& target_sptr) override;
        double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;
        void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient, const DiscretisedDensity<3, elemT>& current_image_estimate) override;
    protected:
        int z_dim, y_dim, x_dim;
        dim3 grid_dim, block_dim;

};

END_NAMESPACE_STIR

#endif // CUDA_RELATIVE_DIFFERENCE_PRIOR_CLASS_H