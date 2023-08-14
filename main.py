import time
import numpy as np
import scipy.stats
import logode
from functions import *
import examples as ex
import sympy as sp
import timeit
import esig


x = ex.unit_circle()
print(x.logsig(0, 0.3, 3))
print(x.sig(0, 0.3, 3))
param = 4


def path(s):
    return np.array([np.cos(2 * np.pi * param * s), np.sin(2 * np.pi * param * s)]) / np.sqrt(param)


print(path(np.linspace(0, 0.3, 2001)).T.shape)
print(esig.stream2sig(path(np.linspace(0, 0.3, 2001)).T, 3))
print(esig.stream2logsig(path(np.linspace(0, 0.3, 2001)).T, 3))
print(esig.logsigkeys(2, 3))
time.sleep(360000)

dim = 3
levels = 4
print(int(np.around((dim ** (levels + 1) - 1) / (dim - 1))))
print(esig.sigdim(dim, levels))
time.sleep(360000)
T = 1.
x = ex.unit_circle(N=3)
f = ex.smooth_2x2_vector_field(N=3)
y_0 = np.array([0, 0])
tic = time.perf_counter()
_, y, error_bounds, time_vec = logode.solve_fixed(x, f, y_0, N=3, partition=np.linspace(0, 1, 101))
print((y[-1],))
print(time.perf_counter() - tic)

tic = time.perf_counter()
f = ex.smooth_2x2_vector_field(N=3)
_, y, prop_loc_err, _ = logode.solve_fully_adaptive_error_representation_fast(x, f, y_0, N_min=1, N_max=3, T=1., n=16, atol=1e-00, rtol=1e-00)
print(time.perf_counter() - tic)

tic = time.perf_counter()
_, y, prop_loc_err, _ = logode.solve_fully_adaptive_error_representation_fast(x, f, y_0, N_min=1, N_max=3, T=1., n=16, atol=1e-05, rtol=1e-05, verbose=3)
global_err = np.sum(prop_loc_err, axis=0)
abs_err = ta.l1(global_err)
print((y[-1],))
print(global_err)
print(time.perf_counter() - tic)

tic = time.perf_counter()
_, y, prop_loc_err, _ = logode.solve_fully_adaptive_error_representation_slow(x, f, y_0, N_min=1, N_max=3, T=1., n=16, atol=1e-05, rtol=1e-05, verbose=3)
global_err = np.sum(prop_loc_err, axis=0)
abs_err = ta.l1(global_err)
print((y[-1],))
print(global_err)
print(time.perf_counter() - tic)

tic = time.perf_counter()
_, y, prop_loc_err, _ = logode.solve_fully_adaptive_error_representation(x, f, y_0, N_min=1, N_max=3, T=1., n=16, atol=1e-05, rtol=1e-05, predict=True, verbose=3)
global_err = np.sum(prop_loc_err, axis=0)
abs_err = ta.l1(global_err)
print((y[-1],))
print(global_err)
print(time.perf_counter() - tic)

tic = time.perf_counter()
_, y, prop_loc_err, _ = logode.solve_fully_adaptive_error_representation(x, f, y_0, N_min=1, N_max=3, T=1., n=16, atol=1e-05, rtol=1e-05, predict=False, verbose=3)
global_err = np.sum(prop_loc_err, axis=0)
abs_err = ta.l1(global_err)
print((y[-1],))
print(global_err)
print(time.perf_counter() - tic)
print('Finished')
time.sleep(3600000)

'''
g = lambda y: np.array([y[0]])
g_grad = lambda y: np.array([1., 0.])
f = ex.bergomi_vector_field(eta=2., rho=-0.9, symbolic=True, N=3)

tic = time.perf_counter()
v_values = np.array([])
dt_values = np.array([])
for i in range(100):
    x = ex.brownian_path_time(n=10000, dim=2, T=1., save_level=1)
    partition, y, prop_loc_err, N = lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=np.array([0., 0.2]),
                                                                                 N_min=2, N_max=2, atol=3e-03, rtol=100.,
                                                                                 g=g, g_grad=g_grad, n=16, T=1., verbose=3)
    v_values = np.concatenate((v_values, y[:-1, 1]))
    dt_values = np.concatenate((dt_values, partition[1:] - partition[:-1]))
toc = time.perf_counter()
print(f'This all took {toc - tic} seconds. We have {len(v_values)} samples.')
plt.plot(v_values, dt_values, 'o')
res = scipy.stats.linregress(np.log(v_values), np.log(dt_values))
v_lims = np.array([np.amin(v_values), np.amax(v_values)])
print(v_lims, res.intercept, res.slope)
plt.plot(v_lims, np.exp(res.intercept) * v_lims ** res.slope)
plt.xscale('log')
plt.yscale('log')
plt.show()

f = ex.bergomi_vector_field(eta=2., rho=-0.9, symbolic=True, N=2)
partition = lambda t, y: np.fmin(0.0003 * y[1] ** (-1.72), 1 - t)
samples = np.empty(100000)
n_steps = np.empty(100000)
tic = time.perf_counter()
for i in range(100000):
    print(f'Path {i} of 100000')
    x = ex.brownian_path_time(n=10000, dim=2, T=1., save_level=1)
    part, y, _, _ = lo.solve_fixed(x=x, f=f, y_0=np.array([0., 0.2]), N=2, partition=partition)
    samples[i] = y[-1, 0]
    n_steps[i] = len(part) - 1
print(samples)
print(n_steps)
print('Time: ', time.perf_counter() - tic)
np.save('adaptive samples.npy', samples)
print(np.sum(n_steps))
print('Totally finished!')
time.sleep(36000000)
'''

T = 1.
print('do x')
# x = ex.smooth_path_singularity(N=2)
x = ex.unit_circle(N=2)
# x = ex.smooth_fractional_path(H=0.4, n=2 ** 20, save_level=3)
# x = ex.brownian_path(n=1000, dim=2, T=1., final_value=((np.log(1.02) - 0.005*0.1) / 0.2, 0.1))
# x = ex.brownian_path(n=1048576, dim=2, T=1., save_level=4)
# x = ex.smooth_3d_path(N=4)
# x = ex.brownian_path_time(n=1024 * int(T), dim=1, T=T, save_level=3)

'''
def brown_fun(n, dim, T, save_level):
    p = 2.05
    times = T - np.exp(np.linspace(np.log(T), -50, n + 1))
    times[-1] = T
    brownian = np.concatenate((np.zeros((1, dim)), np.cumsum(np.sqrt(times[1:] - times[:-1])[:, None] * np.random.normal(0, 1, (n, dim)), axis=0)),
                              axis=0)
    values = np.empty((n + 1, dim + 1))
    values[:, 0] = times
    values[:, 1:] = brownian
    if save_level <= 5:
        return rp.RoughPathDiscrete(times=times, values=values, p=p, var_steps=15, norm=ta.l1,
                                    save_level=save_level)
    return rp.RoughPathContinuous(path=scipy.interpolate.interp1d(times, values, axis=0),
                                  sig_steps=int(max(15, n / 1000)), p=p, var_steps=15, norm=ta.l1)

'''
# x_fun = lambda i: ex.brownian_path_time(n=100000, dim=1, T=T, save_level=3)
# x = ex.brownian_path_time(n=2 ** 20, dim=1, T=T, save_level=3)
print('did x')
print('do f')
# f = ex.simple_smooth_2x2_vector_field(N=3)
# f = ex.smooth_2x2_vector_field(N=4)
f = ex.smooth_2x2_vector_field_singularity(N=2)
# f = ex.bergomi_vector_field(N=3)
# f = ex.wrong_black_scholes_vector_field(N=3)
# f = ex.linear_2x3_vector_field(N=4)
# f = ex.heston_vector_field(rho=-0.7, nu=0.5, lambda_=0.3, theta=0.02, symbolic=True, N=2)
# f = ex.bergomi_vector_field(eta=2., rho=-0.9, symbolic=True, N=3)
# f = ex.langevin_vector_field()
# f = ex.OU_vector_field()
print('did f')
# g, g_grad = ex.smoothed_digital_call_option_payoff(eps=0.1)
# g, g_grad = lambda y: np.array([y[0]]), lambda y: np.array([1., 0.])
g, g_grad = None, None


y_0 = np.array([0., 0.])
N_min = 1
N_max = 2
atol = 3*2e-06
rtol = 100000000 * 3*5e-04
n = 16
verbose = 1
# T = 1.
# discuss_MC_example(x_fun=x_fun, f=f, y_0=y_0, N_min=N_min, N_max=N_max, atol=atol, rtol=rtol, g=g, g_grad=g_grad, n=n, T=T, verbose=verbose, m=20)
# print('Finished')
# time.sleep(36000)
discuss_example(x=x, f=f, y_0=y_0, N_min=N_min, N_max=N_max, atol=atol, rtol=rtol, g=g, g_grad=g_grad, n=n, T=T, verbose=verbose)
# lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=y_0, N_min=N_min, N_max=N_max, atol=10000 * atol, rtol=10000 * rtol, g=g, g_grad=g_grad, n=n, T=T, verbose=verbose)
# profile("lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=y_0, N_min=N_min, N_max=N_max, atol=atol, rtol=rtol, g=g, g_grad=g_grad, n=n, T=T, verbose=verbose)")
print('Finished')
time.sleep(360000)

tic = time.perf_counter()
v_values = np.array([])
dt_values = np.array([])
for i in range(100):
    x = ex.brownian_path_time(n=100000, dim=2, T=1., save_level=1)
    partition, y, prop_loc_err, N = lo.solve_fully_adaptive_error_representation_fast(x=x, f=f, y_0=np.array([0., 0.2]),
                                                                                      N_min=2, N_max=2, atol=1e-03, rtol=100.,
                                                                                      g=g, g_grad=g_grad, n=16, T=1., verbose=3)
    v_values = np.concatenate((v_values, y[:-1, 1]))
    dt_values = np.concatenate((dt_values, partition[1:] - partition[:-1]))
toc = time.perf_counter()
print(f'This all took {toc - tic} seconds. We have {len(v_values)} samples.')
plt.plot(v_values, dt_values, 'o')
res = scipy.stats.linregress(np.log(v_values), np.log(dt_values))
v_lims = np.array([np.amin(v_values), np.amax(v_values)])
print(v_lims, res.intercept, res.slope)
plt.plot(v_lims, np.exp(res.intercept) * v_lims ** res.slope)
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(partition, y[:, 0])
plt.plot(partition, y[:, 1])
for i in range(len(partition) - 1):
    plt.plot(partition[i:i + 2], 10 * (partition[i + 1] - partition[i]) * np.ones(2), 'k-')
plt.show()

discuss_example(x=x, f=f, y_0=np.array([0., 0.2]), N_min=2, N_max=2, atol=1e-03, rtol=100., g=g, g_grad=g_grad, n=16,
                verbose=3)

print('Finished')
time.sleep(360000)

print(lo.solve_fixed(x=x, f=f, y_0=np.array([0., 0.]), N=3, partition=np.linspace(0, 1, 201), verbose=1)[0][-1, :])
