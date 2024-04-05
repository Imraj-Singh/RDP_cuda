#ifndef CUDA_RELATIVE_DIFFERENCE_PRIOR_CLASS_H
#define CUDA_RELATIVE_DIFFERENCE_PRIOR_CLASS_H

#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

template <typename elemT>
class CudaRelativeDifferencePriorClass : public RelativeDifferencePrior<elemT> {
public:
    static const char* const registered_name;

    CudaRelativeDifferencePriorClass();

    CudaRelativeDifferencePriorClass(const bool only_2D, float penalization_factor, float gamma, float epsilon);

    double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;
    void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient, const DiscretisedDensity<3, elemT>& current_image_estimate) override;
};

END_NAMESPACE_STIR

#endif // CUDA_RELATIVE_DIFFERENCE_PRIOR_CLASS_H
