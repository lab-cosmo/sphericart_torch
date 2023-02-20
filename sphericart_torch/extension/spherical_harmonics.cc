#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <vector>
#include "sphericart.c"
#include "gradient_dot.cc"


class SphericalHarmonics : public torch::autograd::Function<SphericalHarmonics> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx, int l_max, torch::Tensor prefactors, torch::Tensor xyz) {
            int n_samples = xyz.sizes()[0];
            torch::Tensor spherical_harmonics = torch::empty({n_samples, (l_max+1)*(l_max+1)}, xyz.options());                
            if (xyz.requires_grad()) {
                torch::Tensor spherical_harmonics_gradients = torch::empty({n_samples, 3, (l_max+1)*(l_max+1)}, xyz.options());
                cartesian_spherical_harmonics(n_samples, l_max, prefactors.data_ptr<double>(), 
                    xyz.data_ptr<double>(), spherical_harmonics.data_ptr<double>(), spherical_harmonics_gradients.data_ptr<double>());
                ctx->saved_data["gradients"] = spherical_harmonics_gradients;
            } else {
                double* dsph = nullptr;
                cartesian_spherical_harmonics(n_samples, l_max, prefactors.data_ptr<double>(), xyz.data_ptr<double>(), spherical_harmonics.data_ptr<double>(), dsph);                
            }
            return spherical_harmonics;
        }

        static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> d_loss_d_outputs) {
            torch::Tensor d_loss_d_outputs_contiguous = d_loss_d_outputs[0].contiguous();
            torch::Tensor spherical_harmonics_gradients = ctx->saved_data["gradients"].toTensor();
            torch::Tensor d_loss_d_inputs = gradient_dot(d_loss_d_outputs_contiguous, spherical_harmonics_gradients);
            return {torch::Tensor(), torch::Tensor(), d_loss_d_inputs};
        }
};


torch::Tensor spherical_harmonics(int l_max, torch::Tensor prefactors, torch::Tensor xyz) {
    return SphericalHarmonics::apply(l_max, prefactors, xyz);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spherical_harmonics", &spherical_harmonics, "spherical harmonics");
}
