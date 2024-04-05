import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import examples as ex
import sympy as sp
import timeit
from functions import *
import logode
import esig
import roughpy as rp


def a(p_, al_, gam_, eps_):
    return al_ ** (3 / p_) * (1 - al_) ** (1 / p_) * eps_ - (1 - al_ ** (3 / p_) - gam_ ** (3 / p_))


def b(p_, al_, gam_):
    return (2 * al_ ** (1 / p_) + 3 / 2 * (1 - al_) ** (1 / p_)) * al_ ** (1 / p_) * (1 - al_) ** (1 / p_) \
           + 1 / 2 * (1 - al_ - gam_) ** (2 / p_) * gam_ ** (1 / p_) + 3 / 2 * (1 - al_ - gam_) ** (1 / p_) * gam_ ** (2 / p_)


def C_001(p_, al_, gam_, eps_):
    a_ = a(p_, al_, gam_, eps_)
    b_ = b(p_, al_, gam_)
    if a_ >= 0:
        return np.inf
    else:
        return - b_ / a_


def C_002(p_, eps_, eta_):
    res = -scipy.optimize.minimize(lambda x_: -C_001(p_, x_[0], x_[1], eps_),
                                    x0=np.array([eta_/2, (1-eta_)/2]), bounds=((0, eta_), (0, 1-eta_)),
                                   method='Nelder-Mead').fun
    print(eta_, res)
    return res


def C_003(p_, eps_):
    return scipy.optimize.minimize(lambda x_: C_002(p_, eps_, x_[0]), x0=np.array([0.5]), bounds=((0, 1),),
                                   method='Nelder-Mead').fun


p_vals = np.linspace(2., 3, 51)[:-1]
sols = np.empty(len(p_vals))
for i in range(len(p_vals)):
    sols[i] = C_003(p_vals[i], 2 * (1 - 2 ** (1 - 3 / p_vals[i])))

plt.plot(p_vals, sols * (1 - 2 ** (1 - 3 / p_vals)))
plt.plot(p_vals, 1.8 + 2 * (p_vals-2))
plt.show()
print(C_001(2.5, 0, 0, 0.1))
print(C_002(2.5, 0.1, 0.8))
print(C_003(2.5, 1))

time.sleep(360000)

T = 1.
x = ex.unit_circle(N=3)
f = ex.smooth_2x2_vector_field(N=3)
y_0 = np.array([0, 0])
tic = time.perf_counter()
_, y, error_bounds, time_vec = logode.solve_fixed(x, f, y_0, N=3, partition=np.linspace(0, 1, 101))
plt.plot(y[:, 0], y[:, 1])
plt.show()

'''
t = np.linspace(0, 1, 1001)
x_1_1 = lambda t_: t_
x_1_2 = lambda t_: np.zeros_like(t_)
x_2_1 = lambda t_: 2 - np.cos(2 * np.pi * t)
x_2_2 = lambda t_: -np.sin(2 * np.pi * t)
plt.plot(x_1_1(t), x_1_2(t), 'b-')
plt.plot(x_2_1(t), x_2_2(t), 'b-')
plt.tight_layout()
plt.show()
t = np.linspace(0, 5.32597, 1001)
r = 1.01409
phi = 2.11622388
x_3_1 = lambda t_: r * (np.cos((t_ + phi) / r) - np.cos(phi / r))
x_3_2 = lambda t_: r * (np.sin((t_ + phi) / r) - np.sin(phi / r))
plt.plot(x_3_1(t), x_3_2(t))
plt.tight_layout()
plt.show()
t = np.linspace(0, 1, 78967)
n = 10
x_4_1 = lambda t_: t_ + 1 / n * np.cos(2 * np.pi * n ** 2 * t_)
x_4_2 = lambda t_: 1 / n * np.sin(2 * np.pi * n ** 2 * t_)
plt.plot(x_4_1(t), x_4_2(t))
plt.tight_layout()
plt.show()
'''
'''
p = np.linspace(2.001, 3, 10001)
expr = (100* 2**((4*p-2)/(3*p)) *48*(1 + 2**(-1/4))**2 * p*(15*p-18)**2/(81*np.exp(1)*(1 - 2**(-(p-2)/(3*p)))**2 *(p-2)))**((p-1)/(2*p))*(4*(1+2**(-1/4))*np.exp(1)*(p/(p+1))**0.5/(1 - 2**(-(1/3-2/(3*p)))))
expr = (100* (p+1)* 2**((4*p-2)/(3*p)) *48*(1 + 2**(-1/4))**2 * p*(15*p-18)**2/(27*np.exp(1)*(1 - 2**(-(p-2)/(3*p)))**2 *(p-2)**2))**((p-1)/(2*p))*(4*np.sqrt(3)*(1+2**(-1/4))*(p/(p+1))**0.5/(1 - 2**(-(1/3-2/(3*p)))))
plt.plot(p, expr)
plt.show()

p = np.linspace(2, 3, 10001)
C_3_11 = 4 * scipy.special.zeta(2)
C_3_1p = 2 ** (1 + 1 / p) * scipy.special.zeta(1 + 1 / p)
gamma = 1.18
beta = 5
left = 1 / (gamma ** (1 / p) - 1)
right = C_3_11 / (-C_3_1p * gamma ** (1 / p) + np.sqrt(C_3_1p ** 2 * gamma ** (2 / p) + 0.5 * C_3_11 * ((beta - gamma) ** (2 / p) - 1)))
plt.plot(p, left)
plt.plot(p, right)
plt.plot(p, 15 * p - 18)
plt.show()
'''
'''
p = np.linspace(2.0001, 2.9999, 1000)
C_1 = 0
C_2 = 0
C_3 = 2 ** (1 + 1 / p) * scipy.special.zeta(1 + 1 / p)
C_3_1_1 = 4 * scipy.special.zeta(4.)
C_4 = (C_3 + C_3_1_1) / 2 ** (2 / p - 1 / 2) + 1
C_5 = 0
C_6 = 1 / (2 * C_4 ** 2)
plt.plot(p, C_6)
plt.show()


p = np.linspace(2.0001, 2.9999, 1000)
C_6 = 10 / (1 - 2 ** (1 - 3 / p))
C_pN = 2 ** (3 / p) * scipy.special.zeta(3 / p)
A_0 = (5 / 4 - 2 ** (-1 - 3 / p))
leading = C_pN / C_6  # 2 ** (4 / p) * 9 / 160
A_1 = leading * 2 ** (1 - 2 / p)
A_2 = leading * (2 ** (-1 / p) * 3 + 2 ** (2 - 3 / p) * A_0 ** 2 / C_6)
A_3 = leading * (5 * 2 ** (-2 / p) + 2 ** (1 - 2 / p) * A_0 ** 2 + 2 ** (-1 / p) * 7 * A_0)
A_4 = leading * 2 ** (1 - 2 / p) * ((5 / 2) ** (p / 2) + 1 / 2 * A_0 ** p) ** (2 / p)
A_5 = leading * 3 * A_0
B_1 = 3 ** (1 - 1 / p) * A_2 / (5 ** (2 / p - 1) - A_1) / C_6
B_2 = 3 ** (1 - 1 / p) * ((A_3 + 3 / 2) / (5 ** (2 / p - 1) - A_1) / C_6 + 1)
B_3 = 3 ** (1 - 1 / p) * (A_4 / (5 ** (2 / p - 1) - A_1) / C_6 + 1)
B_4 = 3 ** (1 - 1 / p) * (A_5 + 1) / (5 ** (2 / p - 1) - A_1)
C_1 = B_2 / (1 - B_1) + 1
C_2 = 2 ** (1 - 1 / p) * (B_3 / (1 - B_1) + 1)
C_3 = 2 ** (1 - 1 / p) * B_4 / (1 - B_1)

C_6_2 = 10 / (1 - 2 ** (1 - 3 / 2))
C_pN_2 = 2 ** (3 / 2) * scipy.special.zeta(3 / 2)
A_0_2 = (5 / 4 - 2 ** (-1 - 3 / 2))
leading_2 = C_pN_2 / C_6_2
A_1_2 = leading_2
A_2_2 = leading_2 * (2 ** (-1 / 2) * 3 + 2 ** (1 / 2) * A_0_2 ** 2 / C_6_2)
A_3_2 = leading_2 * (5 / 2 + A_0_2 ** 2 + 2 ** (-1 / 2) * 7 * A_0_2)
B_1_2 = 3 ** (1 - 1 / 2) * A_2_2 / (5 ** (2 / 2 - 1) - A_1_2) / C_6_2
B_2_2 = 3 ** (1 - 1 / 2) * ((A_3_2 + 3 / 2) / (5 ** (2 / 2 - 1) - A_1_2) / C_6_2 + 1)
C_1_2 = B_2_2 / (1 - B_1_2) + 1
print(6.97 * np.sqrt(2))
print(6.97 * (10 / (1 - 2 ** (1 - 3 / 2))) ** (1 / 4) * 2 ** (3 / 4))
better_const = C_1_2 / (1 - C_1_2 * C_6_2 ** (- 1 / 2))
print(better_const * np.sqrt(2))
print(better_const * (10 / (1 - 2 ** (1 - 3 / 2))) ** (1 / 4) * 2 ** (3 / 4))
L_p_2 = 2 * np.sqrt(7) * C_pN_2
K_p_2 = 9 * 8 * np.sqrt(6) * np.sqrt(C_pN_2)
K_p_2_dash = 3 * (1 + K_p_2 ** 2)
c_pN_2 = 4 * np.exp(4) * (L_p_2 + K_p_2 ** 2 + K_p_2_dash) ** 2
c_pn_2_dash = np.sqrt(c_pN_2)
print(2 * c_pn_2_dash)
print(2 * c_pN_2)


plt.loglog(p, A_1, label='A1')
plt.loglog(p, A_2, label='A2')
plt.loglog(p, A_3, label='A3')
plt.loglog(p, A_4, label='A4')
plt.loglog(p, A_5, label='A5')
plt.legend(loc='best')
plt.show()
plt.loglog(p, B_1, label='B1')
plt.loglog(p, B_2, label='B2')
plt.loglog(p, B_3, label='B3')
plt.loglog(p, B_4, label='B4')
plt.legend(loc='best')
plt.show()
plt.loglog(p, C_1, label='C1')
plt.loglog(p, C_2, label='C2')
plt.loglog(p, C_3, label='C3')
plt.legend(loc='best')
plt.show()
plt.loglog(p, C_6 ** (-1), label='C6-1')
plt.loglog(p, 3.27 ** (-p), label='C-p')
plt.legend(loc='best')
plt.show()
plt.loglog(p, C_1 / (1 - C_1 * C_6 ** (-1 / p)))
plt.show()
'''

p = np.linspace(2.0001, 3, 1001)
e = np.exp(1)
r = e / (1 + (8 * e**4 / (p-2))**(1/3))
const = 2 ** ((p+1)/2) * (2 ** (-p) / ((r * (1-r))**2) + (p/2)**(2/(p-2)) / (1 - r * (2 / p) ** (1 / (p-2))) ** 2 * r ** (p-2) * p / (p-2)) ** (1/2)
plt.plot(p, const * (p-2) ** (1/2))
plt.plot(p, 2*p+4)
plt.show()

C_3 = lambda x: 2 ** (1 + 1 / x) * scipy.special.zeta(1 + 1 / x)
C_13 = lambda x: 1 + np.sqrt(1 + 2 * C_3(x))
C_14 = lambda x: 1 + np.sqrt(2) * np.sqrt(C_3(1) + C_3(x))
print(1 / (4 * C_14(2.5)**2))
print(2 * C_13(2.5) ** 2 / (4 * C_14(2.5)**2))
print(2 * C_13(2.5) ** 2 / (4 * C_14(2.5)**2) *(4 * C_14(2.5)**2))

print(2**(3/4) * C_3(2.5)**(1/4), np.sqrt(2) * C_3(1)**(1/4) + 2 * np.sqrt(C_3(2.5)), np.sqrt(2 * C_3(1)))
fun = lambda x: 1 + 2 * (2 * C_3(2.5) + C_3(1) * x) ** (1 / 2) * x ** (1 / 2) + 4 * C_3(2.5) * x - 2 * C_3(1) * x ** 2
fun = lambda x: 1 + 2 * (2 + x) ** (1 / 2) * x ** (1 / 2) + 4 * x - 2 * x ** 2
fun = lambda x: 1 + np.sqrt(2) * (4 + x) ** (1 / 2) * x ** (1 / 2) + 4 * x - x ** 2
fun = lambda x, p, eta: min((1 + 2 ** (p/4) * (8 + x) ** (p/4) * x ** (p/4) + (8 + x) ** (p/2) * x ** (p/2)) ** (2/p),
                            2 ** (1 - 2/p) * 5 * (1 + x ** p) ** (2/p)) - x ** 2 / (2 * eta)
fun_1 = lambda x, p, eta: (2 ** (1 - 2/p) * 5 * (1 + x ** p) ** (2/p)) - x ** 2 / (2 * eta)
fun_2 = lambda x, p, eta: ((1 + 2 ** (p/4) * (8 + x) ** (p/4) * x ** (p/4) + (8 + x) ** (p/2) * x ** (p/2)) ** (2/p)) - x ** 2 / (2 * eta)
opt = scipy.optimize.minimize_scalar(fun=lambda x: -fun(x, 2.5, 0.25), bounds=(0, 100))
print(opt.x, fun(opt.x, 2.5, 0.25) / 4, fun(opt.x, 2.5, 0.25), 1/4)
eta_arr = np.linspace(0, 0.49, 10000)[1:]
exponents_1 = np.array([fun_2(scipy.optimize.minimize_scalar(fun=lambda x: -fun_2(x, 2.5, eta_arr[i]), bounds=(0, 30 * eta_arr[i] / (1 - 2 * eta_arr[i])), method='bounded').x, 2.5, eta_arr[i]) for i in range(len(eta_arr))])
plt.plot(eta_arr, exponents_1)
exponents_2 = np.array([fun_1(scipy.optimize.minimize_scalar(fun=lambda x: -fun_1(x, 2.5, eta_arr[i]), bounds=(0, 80 * eta_arr[i] / (1 - 2 * eta_arr[i])), method='bounded').x, 2.5, eta_arr[i]) for i in range(len(eta_arr))])
plt.plot(eta_arr, exponents_2)
plt.show()
plt.plot(eta_arr, (6 * eta_arr ** (5/11) + 36 * eta_arr / (1 - 2 * eta_arr)) / (exponents_1-1))
plt.yscale('log')
plt.title('Factor we lose in the exponent')
plt.xlabel('eta')
plt.tight_layout()
plt.show()

p = np.linspace(2, 3, 1000)
plt.plot(p, 2 ** ((6*p+8)/(8-p)) / p - 2 ** ((9*p-16) / (8-p)))
plt.show()
plt.plot(p, (2 ** (1/4) + 9 ** (p/2)) / p * 2 ** ((2 * p ** 2 - 3 * p + 8) / (8 - p)))
plt.show()
time.sleep(2500000)

'''
stream = rp.LieIncrementStream.from_increments([[0., 1., 2.], [3., 4., 5.]], depth=2)

#stream = rp.LieIncrementStream.from_increments([[0., 1., 2.], [3., 4., 5.]], width=3, depth=2, coeffs=rp.DPReal)
interval = rp.RealInterval(0., 2.)
#lsig = stream.log_signature(interval)
#help(lsig)
#sig = stream.signature(interval)
#print(lsig)
#print(sig)
#print(np.array(lsig))
#print(np.array(sig))
context = rp.get_context(width=3, depth=2, coeffs=rp.DPReal)
#tensor_form = context.lie_to_tensor(lsig)
#print(tensor_form)
#print(tensor_form.exp())
#help(tensor_form)
help(context)
#help(context.lie_basis())
basis = context.tensor_basis
for key in basis:
    print(key)
print(basis)
time.sleep(360000)
'''
'''
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
'''
'''
dim = 3
levels = 4
print(int(np.around((dim ** (levels + 1) - 1) / (dim - 1))))
print(esig.sigdim(dim, levels))
time.sleep(360000)
'''

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
