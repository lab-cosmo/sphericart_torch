#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include <vector>
#include "sphericart.c"


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
                ctx->save_for_backward({xyz});
            } else {
                double* dsph = nullptr;
                cartesian_spherical_harmonics(n_samples, l_max, prefactors.data_ptr<double>(), xyz.data_ptr<double>(), spherical_harmonics.data_ptr<double>(), dsph);                
                ctx->save_for_backward({xyz});
            }
            return spherical_harmonics;
        }

        static std::vector<torch::Tensor> backward(torch::autograd::AutogradContext *ctx, std::vector<torch::Tensor> d_loss_d_outputs) {
            auto saved = ctx->get_saved_variables();
            torch::Tensor xyz = saved[0];
            // if (! xyz.requires_grad() ) {
            //     throw "Cannot compute backwards, no derivatives have been computed";
            // }
            torch::Tensor spherical_harmonics_gradients = ctx->saved_data["gradients"].toTensor();

            torch::Tensor d_loss_d_inputs = torch::empty_like(xyz);
            for (int alpha=0; alpha<3; alpha++) {
                d_loss_d_inputs.index_put_(
                    {torch::indexing::Slice(), alpha}, 
                    torch::sum(
                        d_loss_d_outputs[0]*spherical_harmonics_gradients.index({torch::indexing::Slice(), alpha, torch::indexing::Slice()}),
                        1
                    )
                );
            }

            return {torch::Tensor(), torch::Tensor(), d_loss_d_inputs};
        }
};


torch::Tensor spherical_harmonics(int l_max, torch::Tensor prefactors, torch::Tensor xyz) {
    return SphericalHarmonics::apply(l_max, prefactors, xyz);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spherical_harmonics", &spherical_harmonics, "spherical harmonics");
}
