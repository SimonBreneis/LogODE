from functions import *
import examples as ex
import sympy as sp
import timeit


print('do x')
# x = ex.smooth_path_singularity(N=3)
# x = ex.unit_circle(N=4)
# x = ex.smooth_fractional_path(H=0.4, n=100000, save_level=3)
# x = ex.brownian_path(n=1000, dim=2, T=1., final_value=((np.log(1.02) - 0.005*0.1) / 0.2, 0.1))
x = ex.brownian_path(n=1048576, dim=2, T=1., save_level=4)
# x = ex.smooth_3d_path(N=4)
print('did x')
print('do f')
# f = ex.simple_smooth_2x2_vector_field(N=3)
f = ex.smooth_2x2_vector_field(N=4)
# f = ex.smooth_2x2_vector_field_singularity(N=3)
# f = ex.bergomi_vector_field(N=3)
# f = ex.wrong_black_scholes_vector_field(N=3)
# f = ex.linear_2x3_vector_field(N=4)
print('did f')
# g, g_grad = ex.smoothed_digital_call_option_payoff(eps=0.1)

g, g_grad = None, None

print(computational_time_increases_linearly_with_number_of_intervals(x=x, f=f, y_0=np.zeros(2), verbose=2,
                                                               plot_title='Computational time of the Log-ODE method\nfor a Brownian path'))

'''
y_0 = np.array([0., 0.])
N_min = 2
N_max = 3
atol = 2e-04
rtol = 2e-04
n = 16
verbose = 1
T = 1.
lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=y_0, N_min=N_min, N_max=N_max, atol=10000 * atol, rtol=10000 * rtol, g=g, g_grad=g_grad, n=n, T=T, verbose=verbose)
profile("lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=y_0, N_min=N_min, N_max=N_max, atol=atol, rtol=rtol, g=g, g_grad=g_grad, n=n, T=T, verbose=verbose)")

time.sleep(360000)
'''
discuss_example(x=x, f=f, y_0=np.array([0., 0.]), N_min=1, N_max=3, atol=1e-04, rtol=1e-04, g=g, g_grad=g_grad, n=16,
                verbose=1)

print('Finished')


print(lo.solve_fixed(x=x, f=f, y_0=np.array([0., 0.]), N=3, partition=np.linspace(0, 1, 201), verbose=1)[0][-1, :])
