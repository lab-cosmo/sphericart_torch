#include <iostream>
#include <cmath>
#include <torch/extension.h>
#include "sphericart.c"


class SphericalHarmonics : public torch::autograd::Function<SphericalHarmonics> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx, int l_max, torch::Tensor prefactors, torch::Tensor xyz) {
            ctx->saved_data["l_max"] = l_max;
            int n_samples = xyz.sizes()[0];
            torch::Tensor spherical_harmonics = torch::empty({n_samples, (l_max+1)*(l_max+1)}, xyz.options());                
            if (xyz.requires_grad()) {
                torch::Tensor spherical_harmonics_gradients = torch::empty({n_samples, 3, (l_max+1)*(l_max+1)}, xyz.options());
                cartesian_spherical_harmonics(n_samples, l_max, prefactors.data_ptr<double>(), 
                    xyz.data_ptr<double>(), spherical_harmonics.data_ptr<double>(), spherical_harmonics_gradients.data_ptr<double>());
                ctx->save_for_backward({xyz, spherical_harmonics_gradients});
            } else {
                double* dsph = nullptr;
                cartesian_spherical_harmonics(n_samples, l_max, prefactors.data_ptr<double>(), xyz.data_ptr<double>(), spherical_harmonics.data_ptr<double>(), dsph);                
                ctx->save_for_backward({xyz});
            }
            return spherical_harmonics;
        }

        static torch::autograd::tensor_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::tensor_list d_loss_d_outputs) {
            int l_max = ctx->saved_data["l_max"].toInt();
            auto saved = ctx->get_saved_variables();
            torch::Tensor xyz = saved[0];
            if (! xyz.requires_grad() ) {
                throw "Cannot compute backwards, no derivatives have been computed";
            }
            torch::Tensor spherical_harmonics_gradients = saved[1];
            int n_samples = xyz.sizes()[0];

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
