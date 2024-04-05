#include "CudaRelativeDifferencePriorClass.h"

/* // Define any necessary CUDA kernels here
__global__ void valueKernel(...) {
    // Kernel code...
}

__global__ void gradientKernel(...) {
    // Kernel code...
} */

START_NAMESPACE_STIR

template <typename elemT>
double CudaRelativeDifferencePriorClass<elemT>::compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) {
    return 1.0f;
}

template <typename elemT>
void CudaRelativeDifferencePriorClass<elemT>::compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient, const DiscretisedDensity<3, elemT>& current_image_estimate) {
}

// Explicit template instantiations
template class stir::CudaRelativeDifferencePriorClass<float>;
template <typename elemT>
const char* const CudaRelativeDifferencePriorClass<elemT>::registered_name = "CudaRDP";
template <typename elemT>
CudaRelativeDifferencePriorClass<elemT>::CudaRelativeDifferencePriorClass() : RelativeDifferencePrior<elemT>() {}
template <typename elemT>
CudaRelativeDifferencePriorClass<elemT>::CudaRelativeDifferencePriorClass(const bool only_2D, float penalization_factor, float gamma, float epsilon) : RelativeDifferencePrior<elemT>(only_2D, penalization_factor, gamma, epsilon) {}

END_NAMESPACE_STIR
