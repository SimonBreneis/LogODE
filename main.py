from functions import *
import examples as ex
import sympy as sp
import timeit


q = sp.symbols('q')
r = sp.symbols('r')
s = sp.symbols('s')
expr = q*q*sp.exp(q) - 2*q*sp.exp(q)
print(expr)
print(sp.diff(expr, q))

'''
sp_arr = sp.Array([sp.exp(3 * q) * sp.sin(2 * r),
                   sp.exp(2 * q) * sp.sin(9 * r),
                   sp.exp(4 * q) * sp.sin(7 * r),
                   sp.exp(5 * q) * sp.sin(23 * r),
                   sp.exp(1 * q) * sp.sin(22 * r),
                   sp.exp(3 * q) * sp.sin(22 * r),
                   sp.exp(4 * q) * sp.sin(21 * r),
                   sp.exp(9 * q) * sp.sin(26 * r),
                   sp.exp(5 * q) * sp.sin(28 * r),
                   sp.exp(7 * q) * sp.sin(20 * r),
                   sp.exp(3 * q) * sp.sin(22 * r),
                   sp.exp(1 * q) * sp.sin(21 * r),
                   sp.exp(2 * q) * sp.sin(52 * r),
                   sp.exp(4 * q) * sp.sin(26 * r),
                   sp.exp(9 * q) * sp.sin(27 * r),
                   sp.exp(8 * q) * sp.sin(29 * r),
                   sp.exp(6 * q) * sp.sin(20 * r),
                   sp.exp(3 * q) * sp.sin(27 * r),
                   sp.exp(1 * q) * sp.sin(24 * r),
                   sp.exp(5 * q) * sp.sin(22 * r),
                   sp.exp(3 * s) * sp.sin(2 * r * q),
                   sp.exp(2 * s) * sp.sin(9 * r * q),
                   sp.exp(4 * s) * sp.sin(7 * r * q),
                   sp.exp(5 * s) * sp.sin(23 * r * q),
                   sp.exp(1 * s) * sp.sin(22 * r * q),
                   sp.exp(3 * s) * sp.sin(22 * r * q),
                   sp.exp(4 * s) * sp.sin(21 * r * q),
                   sp.exp(9 * s) * sp.sin(26 * r * q),
                   sp.exp(5 * s) * sp.sin(28 * r * q),
                   sp.exp(7 * s) * sp.sin(20 * r * q),
                   sp.exp(3 * s) * sp.sin(22 * r * q),
                   sp.exp(1 * s) * sp.sin(21 * r * q),
                   sp.exp(2 * s) * sp.sin(52 * r * q),
                   sp.exp(4 * s) * sp.sin(26 * r * q),
                   sp.exp(9 * s) * sp.sin(27 * r * q),
                   sp.exp(8 * s) * sp.sin(29 * r * q),
                   sp.exp(6 * s) * sp.sin(20 * r * q),
                   sp.exp(3 * s) * sp.sin(27 * r * q),
                   sp.exp(1 * s) * sp.sin(24 * r * q),
                   sp.exp(5 * s) * sp.sin(22 * r * q)])

sp_fun = sp.lambdify([q, r, s], sp_arr, 'numpy')

np_fun = lambda q, r, s: np.array([np.exp(3 * q) * np.sin(2 * r),
                   np.exp(2 * q) * np.sin(9 * r),
                   np.exp(4 * q) * np.sin(7 * r),
                   np.exp(5 * q) * np.sin(23 * r),
                   np.exp(1 * q) * np.sin(22 * r),
                   np.exp(3 * q) * np.sin(22 * r),
                   np.exp(4 * q) * np.sin(21 * r),
                   np.exp(9 * q) * np.sin(26 * r),
                   np.exp(5 * q) * np.sin(28 * r),
                   np.exp(7 * q) * np.sin(20 * r),
                   np.exp(3 * q) * np.sin(22 * r),
                   np.exp(1 * q) * np.sin(21 * r),
                   np.exp(2 * q) * np.sin(52 * r),
                   np.exp(4 * q) * np.sin(26 * r),
                   np.exp(9 * q) * np.sin(27 * r),
                   np.exp(8 * q) * np.sin(29 * r),
                   np.exp(6 * q) * np.sin(20 * r),
                   np.exp(3 * q) * np.sin(27 * r),
                   np.exp(1 * q) * np.sin(24 * r),
                   np.exp(5 * q) * np.sin(22 * r),
                   np.exp(3 * s) * np.sin(2 * r * q),
                   np.exp(2 * s) * np.sin(9 * r * q),
                   np.exp(4 * s) * np.sin(7 * r * q),
                   np.exp(5 * s) * np.sin(23 * r * q),
                   np.exp(1 * s) * np.sin(22 * r * q),
                   np.exp(3 * s) * np.sin(22 * r * q),
                   np.exp(4 * s) * np.sin(21 * r * q),
                   np.exp(9 * s) * np.sin(26 * r * q),
                   np.exp(5 * s) * np.sin(28 * r * q),
                   np.exp(7 * s) * np.sin(20 * r * q),
                   np.exp(3 * s) * np.sin(22 * r * q),
                   np.exp(1 * s) * np.sin(21 * r * q),
                   np.exp(2 * s) * np.sin(52 * r * q),
                   np.exp(4 * s) * np.sin(26 * r * q),
                   np.exp(9 * s) * np.sin(27 * r * q),
                   np.exp(8 * s) * np.sin(29 * r * q),
                   np.exp(6 * s) * np.sin(20 * r * q),
                   np.exp(3 * s) * np.sin(27 * r * q),
                   np.exp(1 * s) * np.sin(24 * r * q),
                   np.exp(5 * s) * np.sin(22 * r * q)])

input = np.array([1., 2., 3.])
sp_fun(1., 2., 3.)
print(timeit.timeit('sp_fun(1., 2., 3.)', "from __main__ import sp_fun", number=10000))
print(timeit.timeit('np.array(sp_fun(1., 2., 3.))', "from __main__ import sp_fun; import numpy as np", number=10000))
print(timeit.timeit('np_fun(1., 2., 3.)', "from __main__ import np_fun", number=10000))
print('Finished')
time.sleep(36000)
'''

print('do x')
# x = ex.smooth_path_singularity(N=3)
x = ex.unit_circle(N=2)
# x = ex.smooth_fractional_path(H=0.4, n=100000, save_level=3)
# x = ex.brownian_path(n=1000, dim=2, T=1., final_value=((np.log(1.02) - 0.005*0.1) / 0.2, 0.1))
print('did x')
print('do f')
# f = ex.simple_smooth_2x2_vector_field(N=3)
f = ex.smooth_2x2_vector_field(N=2)
# f = ex.smooth_2x2_vector_field_singularity(N=3)
# f = ex.bergomi_vector_field(N=3)
# f = ex.wrong_black_scholes_vector_field(N=3)
print('did f')
# g, g_grad = ex.smoothed_digital_call_option_payoff(eps=0.1)

g, g_grad = None, None
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
discuss_example(x=x, f=f, y_0=np.array([0., 0.]), N_min=1, N_max=2, atol=2e-03, rtol=2e-03, g=g, g_grad=g_grad, n=16,
                verbose=1)

print('Finished')


print(lo.solve_fixed(x=x, f=f, y_0=np.array([0., 0.]), N=3, partition=np.linspace(0, 1, 201), verbose=1)[0][-1, :])
