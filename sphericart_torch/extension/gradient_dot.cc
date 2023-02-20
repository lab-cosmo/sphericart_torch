#include <iostream>
#include <torch/extension.h>


torch::Tensor gradient_dot(torch::Tensor& d_loss_d_outputs, torch::Tensor& sh_gradients) {

    int n_samples = sh_gradients.sizes()[0];
    int n_sh = sh_gradients.sizes()[2];
    torch::Tensor d_loss_d_inputs = torch::zeros({n_samples, 3}, d_loss_d_outputs.options());
    double* d_loss_d_outputs_ptr = d_loss_d_outputs.data_ptr<double>();
    double* sh_gradients_ptr = sh_gradients.data_ptr<double>();
    double* d_loss_d_inputs_ptr = d_loss_d_inputs.data_ptr<double>();

    // If the tensors here are not contiguous, not only the results are wrong...
    // They can also segfault.
    // All should be guaranteed to be contiguous on entry to this function.
    /*
    std::cout << sh_gradients.is_contiguous() << std::endl;
    std::cout << d_loss_d_inputs.is_contiguous() << std::endl;
    std::cout << d_loss_d_outputs.is_contiguous() << std::endl;
    */
            
    // #pragma omp parallel for
    for (int i_sample=0; i_sample<n_samples; i_sample++) {
        for (int alpha=0; alpha<3; alpha++) {
            for (int i_sh=0; i_sh<n_sh; i_sh++) {
                // std::cout << d_loss_d_inputs_ptr[3*i_sample+alpha] << "\n";
                // std::cout << d_loss_d_outputs_ptr[n_sh*i_sample+i_sh] << "\n";
                // std::cout << sh_gradients_ptr[n_sh*3*i_sample+n_sh*alpha+i_sh] << "\n";
                // std::cout << std::endl;
                //std::cout << 3*i_sample+alpha << "/" << size_input << std::endl;
                //std::cout << n_sh*i_sample+i_sh << "/" << size_output << std::endl;
                //std::cout << n_sh*3*i_sample+n_sh*alpha+i_sh << "/" << size_sh_gradients << std::endl;
                d_loss_d_inputs_ptr[3*i_sample+alpha] += d_loss_d_outputs_ptr[n_sh*i_sample+i_sh]*sh_gradients_ptr[n_sh*3*i_sample+n_sh*alpha+i_sh];
                // d_loss_d_inputs_ptr[3*i_sample+alpha] = 1.0;
            }
        }
    }

    return d_loss_d_inputs;
}
