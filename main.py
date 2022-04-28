import numpy as np
import scipy
from scipy import integrate, special
from esig import tosig as ts
import matplotlib.pyplot as plt
import p_var
import time
import logode as lo
import examples as ex
import roughpath as rp
import vectorfield as vf
import cProfile
import sympy as sp
import tensoralgebra as ta
from fbm import FBM
import euler


def stupid_flow(x, f, y_0, s, T=1, N=2, n=200, h=3e-04):
    dim_y = len(y_0)
    partition = np.linspace(s, T, n)
    Psi = np.zeros((dim_y, dim_y))
    y = lo.solve_fixed(x, f, y_0, N, partition)[0][:, -1]
    for i in range(dim_y):
        perturbation = np.zeros(dim_y)
        perturbation[i] = 1
        y_pert = lo.solve_fixed(x, f, y_0 + h*perturbation, N, partition)[0][:, -1]
        Psi[:, i] = (y_pert - y)/h
    return Psi




'''
f_sym = ex.smooth_2x2_vector_field()
f_num = ex.smooth_2x2_vector_field(symbolic=False)
ls = ex.unit_circle().logsig(0, 1, 2)
print(f_sym.apply(ls)(np.array([0.323, 0.124])))
print(f_num.apply(ls)(np.array([0.323, 0.124])))
time.sleep(3600)
'''
'''
if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run("ex.ex_smooth_path(solver='fssr', n=100)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("cumtime").print_stats(100)
time.sleep(3600)
'''

'''
y, _ = lo.solve_fixed_adj_full(x=ex.time_rough_path(), f=ex.linear_1x1_vector_field(), y_0=np.array([1]), N=1, partition=np.linspace(0, 1, 6))
y, _ = lo.solve_fixed_adj_full(x=ex.time_rough_path(), f=ex.linear_1x1_vector_field(), y_0=ta.sig_first_level_num(np.array([1]), 1), N=1, partition=np.linspace(0, 1, 6))
print(lo.solve_fixed_error_representation(x=ex.time_rough_path(), f=ex.linear_1x1_vector_field(), y_0=np.array([1]), N=1, partition=np.linspace(0, 1, 6)))
'''
'''
x = ex.unit_circle(param=1)
x = ex.unit_circle(param=1, symbolic=False)
f = ex.smooth_2x2_vector_field()
f.new_derivative()
print(f.f[1])
sigma = lambda x: 1/(1+np.exp(-x))
a = lambda y: y[0] - 2*y[1]
second_level = lambda y: -0.6142 * np.array([y[1] + sigma(a(y)) + sigma(y[1]),
                                            sigma(a(y))*sigma(y[1])*(1-sigma(y[1])) + (y[0]-y[1]+2*sigma(y[1]))*sigma(a(y))*(1-sigma(a(y)))])
first_level_0 = lambda y: np.array([-1.5*(y[1]-y[0]) - 0.8660*y[1], -1.5*sigma(y[1]) + 0.8660*sigma(a(y))])
first_level_1 = lambda y: np.array([1.7321*y[1], -1.7321*sigma(a(y))])
first_level_2 = lambda y: np.array([1.5*(y[1]-y[0]) - 0.8660*y[1], 1.5*sigma(y[1]) + 0.8660*sigma(a(y))])
true_vf = lambda t, y: 2*np.pi*np.array([-np.sin(2*np.pi*t)*(y[1]-y[0]) - np.cos(2*np.pi*t)*y[1], -np.sin(2*np.pi*t)*sigma(y[1]) + np.cos(2*np.pi*t)*sigma(a(y))])
print(integrate.solve_ivp(fun=true_vf, t_span=(0, 1/3), y0=np.array([0, 0])).y[:, -1])
print(integrate.solve_ivp(fun=lambda t, y: first_level_0(y) + second_level(y), t_span=(0, 1), y0=np.array([0, 0])).y[:, -1])
# print(integrate.solve_ivp(fun=lambda t, y: np.array([2*y[0] - 2*y[1], -2*sigma(y[1])]), t_span=(0, 1), y0=np.array([0, 0])).y[:, -1])
# print(integrate.solve_ivp(fun=lambda t, y: -np.array([2*y[0] - 2*y[1], -2*sigma(y[1])]), t_span=(0, 1), y0=np.array([1.92961986, -0.7920599])).y[:, -1])

print(lo.solve_fixed_error_representation(x=ex.unit_circle(param=1), f=ex.smooth_2x2_vector_field(), y_0=np.array([0, 0]), N=2, partition=np.linspace(0, 1, 4)))
time.sleep(3600)
'''
x = ex.unit_circle()
# print(np.argwhere(np.array([0, 1, 2, 3]) > 1).flatten())

f = ex.smooth_2x2_vector_field(symbolic=False)
# x = ex.rough_fractional_Brownian_path(H=0.5, n=1000*2000, dim=2, T=1., p=2.2, var_steps=15, norm=ta.l1,
#                                             save_level=3)
print('constructed')

'''
ex.ex_fBm_path(solver='assr', N=2, x=x, f=f, speed=0.1)
print('finished')
time.sleep(3600)
'''


y_on_partition, _, _ = ex.ex_smooth_path(solver='fssr', N=2, n=200, x=x, f=f, speed=0)
y_1 = y_on_partition[1, :]
partition = np.linspace(1/200, 2/200, 101)
y, _ = lo.solve_fixed(x, f, y_1, N=2, partition=partition)
z = np.zeros((101, 4))
z[:, 2:] = y.transpose()
z[:, :2] = np.array([x.at(partition[i], 1)[1] for i in range(101)])
z_rp = rp.RoughPathDiscrete(partition, z)
print(f'stupidly computed z: {z_rp.sig(1/200, 2/200, 2)}')
sig = lambda x: 1/(1+np.exp(-x))
matrix_1 = lambda y: np.array([[-1, 1], [0, sig(y[1])*(1-sig(y[1]))]])
matrix_2 = lambda y: np.array([[0, -1], [sig(y[0]-2*y[1])*(1-sig(y[0]-2*y[1])), -2*sig(y[0]-2*y[1])*(1-sig(y[0]-2*y[1]))]])
h_vec = np.zeros((2, 2, 100))
for i in range(100):
    h_vec[:, :, i] = matrix_1(y[:, i]) * (x.sig(partition[i], partition[i+1], 1)[1][0]) \
                     + matrix_2(y[:, i]) * (x.sig(partition[i], partition[i+1], 1)[1][1])
print('Stupidly computed h, twice:')
print(np.sum(h_vec, axis=-1))
print(np.sum(h_vec, axis=-1).flatten())



print(f'stupidly computed flow: {stupid_flow(x, f, np.zeros(2), 0)}')
print(f'stupidly computed flow, different h: {stupid_flow(x, f, np.zeros(2), 0, h=1e-04)}')
tic = time.perf_counter()
true_solution, _, true_time = ex.ex_smooth_path(N=3, n=1000, x=x, f=f, speed=0)
toc = time.perf_counter()
print(f'solving for true solution: {toc-tic}')
true_solution = true_solution[:, -1]
print(true_solution, true_time)
N_vec = np.array([2])
# N_vec = np.array([1, 2, 3])
n_vec = np.array([200])
# n_vec = np.array([16, 23, 32, 45, 64, 91, 128, 181, 256, 362])
abs_global_error = np.zeros(len(n_vec))
abs_true_error = np.zeros(len(n_vec))
abs_difference = np.zeros(len(n_vec))
abs_relative = np.zeros(len(n_vec))
for i in range(len(N_vec)):
    print(f'N={N_vec[i]}')
    for j in range(len(n_vec)):
        print(f'n={n_vec[j]}')
        y, local_errors, tictoc = ex.ex_smooth_path(solver='fssr', N=N_vec[i], n=n_vec[j], x=x, f=f, speed=0)

        print(f'penultimate stupidly computed flow: {stupid_flow(x, f, y[-2, :], 199 / 200)}')
        global_error = np.sum(local_errors, axis=0)
        true_error = true_solution - y[-1, :]
        abs_global_error[j] = ta.l1(global_error)
        abs_true_error[j] = ta.l1(true_error)
        abs_difference[j] = ta.l1(true_error - global_error)
        abs_relative[j] = abs_global_error[j]/abs_true_error[j]
        print(abs_global_error[j], abs_true_error[j], abs_difference[j], abs_relative[j], tictoc)
    plt.loglog(n_vec, abs_global_error, label='estimated global error')
    plt.loglog(n_vec, abs_true_error, label='true global error')
    plt.loglog(n_vec, abs_difference, label='true - global error')
    plt.title('Smooth path, smooth vector field\nComparing true and estimated global errors')
    plt.legend(loc='best')
    plt.xlabel('Number of intervals')
    plt.ylabel('Error')
    plt.show()




y, propagated_local_errors, tictoc = ex.ex_smooth_path(solver='fssr', N=3)
print(y)
print(propagated_local_errors)
global_error = np.sum(propagated_local_errors, axis=0)
print(global_error)
true_sol = np.array([-0.95972823, -0.97375321])
print(true_sol - y[-1, :])
print(tictoc)
time.sleep(360000)

tens = [0, np.array([1, 2, 3]), np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        np.array([[[13, 14, 15], [16, 17, 18], [19, 20, 21]],
                  [[22, 23, 24], [25, 26, 27], [28, 29, 30]],
                  [[31, 32, 33], [34, 35, 36], [37, 38, 39]]])]

tens = ta.NumericTensor(tens)
print(tens)
print(tens.project_space([2, 0]))

tens = [0, sp.Array([1, 2, 3]), sp.Array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        sp.Array([[[13, 14, 15], [16, 17, 18], [19, 20, 21]],
                  [[22, 23, 24], [25, 26, 27], [28, 29, 30]],
                  [[31, 32, 33], [34, 35, 36], [37, 38, 39]]])]

tens = ta.SymbolicTensor(tens)
print(tens)
print(tens.project_space([2, 0]))

time.sleep(360000)


ex.discussion(example=1.24, show=True, save=False, sym_path=True, sym_vf=False, full=1)
print('Finished')
time.sleep(360000)

vals, n = ex.asymmetric_Brownian_path(500, 1)
plt.plot(np.linspace(0, 1, n+1), vals)
plt.show()

ex.discussion(example=5.5, show=False, save=True, sym_path=True, sym_vf=True)
time.sleep(360000)

# Sanity check
t = sp.symbols('t')
path = sp.Array([sp.cos(2 * 4 * sp.pi * t) / sp.sqrt(4), sp.sin(2 * 4 * sp.pi * t) / sp.sqrt(4)])
x = rp.RoughPathSymbolic(path=path, t=t, p=1, var_steps=15, norm=ta.l1)

path_ = lambda t: np.array([np.cos(2 * np.pi * 4 * t), np.sin(2 * np.pi * 4 * t)]) / np.sqrt(4)
x_ = rp.RoughPathContinuous(path=path_, sig_steps=1000, p=1, var_steps=15, norm=ta.l1)

y, z = sp.symbols('y z')
f = sp.Array([[z - y, -z], [1/(1+sp.exp(-z)), 1/(1+sp.exp(-(y - 2 * z)))]])
vec_field = vf.VectorFieldSymbolic(f=[f], norm=ta.l1, variables=[y, z])

f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                           1 / (1 + np.exp(-y[1])) * x[0] + 1 / (1 + np.exp(-(y[0] - 2 * y[1]))) * x[1]])
vec_field_ = vf.VectorFieldNumeric(f=[f], h=1e-07, norm=ta.l1)

print(vec_field.apply(x.logsig(0.1, 0.143, 2), True)(np.array([-0.3, -0.52])))
print(vec_field_.apply(x_.logsig(0.1, 0.143, 2), True)(np.array([-0.3, -0.52])))

ex.discussion(example=0, show=False, save=True, sym_path=True, sym_vf=True)
time.sleep(360000)
ex.discussion(example=1.02785, show=False, save=True, sym_path=True, sym_vf=True)
time.sleep(360000)
ex.ex_third_level(N=0, plot=False, sig_steps=10000, atol=1e-06, rtol=1e-05, sym_path=False, sym_vf=True, param=1000)
time.sleep(360000)
ex.discussion(example=2.01562, show=False, save=True, sym_path=True, sym_vf=True)
time.sleep(360000)
ex.discussion(example=0, show=False, save=True, sym_path=True, sym_vf=True)
time.sleep(360000)
ex.ex_third_level(N=0, plot=False, sig_steps=2000, sym_path=True, sym_vf=True, param=64, atol=1e-06, rtol=1e-05)
time.sleep(360000)
ex.discussion(example=0.3, show=False, save=True, sym_path=False, sym_vf=True)
time.sleep(360000)
ex.discussion(example=0.4, show=False, save=True, sym_path=False, sym_vf=True)
time.sleep(360000)
ex.discussion(example=0.5, show=False, save=True, sym_path=False, sym_vf=True)
time.sleep(360000)
ex.discussion(example=0.75, show=False, save=True, sym_path=False, sym_vf=True)
time.sleep(3600)
ex.discussion(example=1, save=True, show=False, sym_path=False, sym_vf=True)
time.sleep(3600)
ex.discussion(example=1.02785, save=True, show=False, sym_path=True, sym_vf=True)
time.sleep(3600)
ex.discussion(example=1.23, save=True, show=False, sym_path=False, sym_vf=True)
time.sleep(3600)
ex.discussion(example=1.23, save=False, show=False, sym_path=True, sym_vf=True)
time.sleep(3600)


def con(N, p, d):
    """
    Returns the constant in the error bound for a single step of the Log-ODE method.
    :param N: Degree of the method
    :return: The constant
    """
    if p == -1:
        return 0.34 * (7 / 3.) ** (N + 1)
    if p == -2:
        return 25 * d / scipy.special.gamma((N + 1) / -p + 1) + 0.081 * (7 / 3) ** (N + 1)
    if p == -3:
        return 1000 * d ** 3 / scipy.special.gamma((N + 1) / -p + 1) + 0.038 * (7 / 3) ** (N + 1)
    if 1 <= p < 2:
        C = 1
    elif 2 <= p < 3:
        C = 3 * np.sqrt(d)
    elif 3 <= p < 4:
        C = 7 * d
    else:
        C = 21 * d ** (9 / 4)
    beta = rp.beta(p)
    return (1.13 / beta) ** (1 / int(p)) * (int(p) * C) ** int(p) / scipy.special.factorial(int(p)) / scipy.special.gamma(
        (N + 1) / p + 1) \
           + 0.83 * (7 / (3 * beta ** (1 / N))) ** (N + 1)


print(con(3, -1, 2), con(3, 1, 2))
print(con(3, -2, 2), con(3, 2, 2))
print(con(3, -3, 2), con(3, 3, 2))
var_steps = 200
x = ex.rough_fractional_Brownian_path(H=0.5, n=10*var_steps, dim=2, T=1., var_steps=var_steps, norm=ta.l1, save_level=2)
ps = np.linspace(2., 4., 100)[:-1]
constants = np.zeros(99)
vars = np.zeros(99)
omegas = np.zeros(99)

for i in range(len(ps)):
    print(i)
    constants[i] = con(3, ps[i], 2)
    if i == 0:
        vars[i] = np.amax(x.p_variation(0., 1., ps[i], var_steps=var_steps, norm=ta.l1))
    else:
        vars[i] = np.amax(x.p_variation(0., 1., ps[i], var_steps=var_steps, norm=ta.l1))
    omegas[i] = x.omega(0., 1., ps[i], var_steps=var_steps)
plt.loglog(ps, constants, label='error constant')
plt.loglog(ps, vars, label='p-variation')
plt.loglog(ps, omegas/10000, label='control function')
plt.loglog(ps, constants*omegas/100000, 'k-', label='error')
plt.title("Varying p for Brownian motion")
# plt.ylim([0, 10])
plt.legend(loc='best')
plt.show()



time.sleep(3600)


'''
if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run("ex.smooth_path(plot=True, N=2, n=300, sym_path=True, sym_vf=True)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("time").print_stats(100)
time.sleep(3600)
'''
