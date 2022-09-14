from functions import *
import examples as ex


print('do x')
# x = ex.smooth_path_singularity(N=3)
# x = ex.unit_circle(N=3)
x = ex.smooth_fractional_path(H=0.4, n=1000000, save_level=3)
# x = ex.brownian_path(n=1000, dim=2, T=1., final_value=((np.log(1.02) - 0.005*0.1) / 0.2, 0.1))
print('did x')
print('do f')
f = ex.simple_smooth_2x2_vector_field(N=3)
# f = ex.smooth_2x2_vector_field(N=4)
# f = ex.smooth_2x2_vector_field_singularity(N=3)
# f = ex.bergomi_vector_field(N=3)
# f = ex.wrong_black_scholes_vector_field(N=3)
print('did f')
# g, g_grad = ex.smoothed_digital_call_option_payoff(eps=0.1)
g, g_grad = None, None
discuss_example(x=x, f=f, y_0=np.array([0., 0.]), N_min=2, N_max=3, atol=2e-03, rtol=2e-03, g=g, g_grad=g_grad, n=16,
                verbose=1)

print('Finished')
