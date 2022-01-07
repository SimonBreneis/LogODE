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


'''
a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.sin(y) + y
tic = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
for _ in range(100000):
    function(0.53)
toc = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
print(function(0.53))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.sin(y) + y
tic = time.perf_counter()
function = lambda x: f.evalf(subs={y: x})
for _ in range(100000):
    function(0.53)
toc = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
print(function(0.53))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.Array([sp.sin(y) + y])
tic = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
for _ in range(100000):
    function(0.53)
toc = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
print(function(0.53))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.Array([[sp.sin(y) + y, sp.cos(y)], [sp.exp(y), 1]])
tic = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
for _ in range(100000):
    function(0.53)
toc = time.perf_counter()
function = sp.lambdify(y, f, 'numpy')
print(function(0.53))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.Array([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y-2*z)))]])
tic = time.perf_counter()
function = sp.lambdify([y, z], f, 'numpy')
for _ in range(100000):
    function(0.53, 0.47)
toc = time.perf_counter()
function = sp.lambdify([y, z], f, 'numpy')
print(function(0.53, 0.47))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.Array([[z - y, -z], [logistic.subs(a, z), logistic.subs(a, y - 2 * z)]])
tic = time.perf_counter()
for _ in range(1000):
    function = sp.lambdify(sp.Array([y, z]), f, modules='numpy')
    function(0.53, 0.47)
toc = time.perf_counter()
function = sp.lambdify(sp.Array([y, z]), f, modules='numpy')
print(function(0.53, 0.47))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.SparseMatrix([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y-2*z)))]])
tic = time.perf_counter()
for _ in range(1000):
    function = sp.lambdify(sp.Array([y, z]), f, modules='numpy')
    function(0.53, 0.47)
toc = time.perf_counter()
function = sp.lambdify(sp.Array([y, z]), f, modules='numpy')
print(function(0.53, 0.47))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.Array([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y-2*z)))]])
tic = time.perf_counter()
function = lambda a, b: np.array([[sp.lambdify([y, z], f[i, j], 'numpy') for j in range(2)] for i in range(2)])
for _ in range(1000):
    function(0.53, 0.47)
toc = time.perf_counter()
function = sp.lambdify(sp.Array([y, z]), f, modules='numpy')
print(function(0.53, 0.47))
print(toc-tic)

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = lambda y, z: np.array([[z - y, -z], [1 / (1 + np.exp(-z)), 1 / (1 + np.exp(-(y-2*z)))]])
tic = time.perf_counter()
for _ in range(100000):
    f(0.53, 0.47)
toc = time.perf_counter()
print(f(0.53, 0.47))
print(toc-tic)

time.sleep(36000)
'''



t = sp.symbols('t')
path = sp.Array(
    [sp.cos(2 * 4 * sp.pi * t) / sp.sqrt(4), sp.sin(2 * 4 * sp.pi * t) / sp.sqrt(4)])
x = rp.RoughPathSymbolic(path=path, t=t, p=1, var_steps=15, norm=ta.l1)

path_ = lambda t: np.array([np.cos(2 * np.pi * 4 * t), np.sin(2 * np.pi * 4 * t)]) / np.sqrt(4)
x_ = rp.RoughPathContinuous(path=path_, n_steps=1000, p=1, var_steps=15, norm=ta.l1)

print(x.log_incr(0.1, 0.143, 3))
print(x_.log_incr(0.1, 0.143, 3))

a, y, z = sp.symbols('a y z')
logistic = 1 / (1 + sp.exp(-a))
f = sp.Array([[z - y, -z], [logistic.subs(a, z), logistic.subs(a, y - 2 * z)]])
variables = sp.Array([y, z])
vec_field = vf.VectorFieldSymbolic(f=[f], norm=ta.l1, variables=variables)

logistic_ = lambda z: 1 / (1 + np.exp(-z))


def f(y, dx):
    return np.array(
        [(y[1] - y[0]) * dx[0] - y[1] * dx[1], logistic_(y[1]) * dx[0] + logistic_(y[0] - 2 * y[1]) * dx[1]])


vec_field_ = vf.VectorFieldNumeric(f=[f], h=1e-07, norm=ta.l1)

print(vec_field.vector_field(x.log_incr(0.1, 0.143, 2))(np.array([-0.3, -0.52])))
print(vec_field_.vector_field(x_.log_incr(0.1, 0.143, 2))(np.array([-0.3, -0.52])))

if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run("ex.smooth_path(plot=True, N=2, n=300, sym_path=True, sym_vf=True)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("time").print_stats(10000)
time.sleep(3600)

ex.discussion(example=1.23, save=False, show=True, sym_path=True, sym_vf=True, test=True)
time.sleep(360000)
ex.discussion(example=0.4, show=False, save=True, sym_path=False, sym_vf=False, test=False)
time.sleep(3600)
ex.discussion(example=0.5, show=False, save=True, sym_path=False, sym_vf=False, test=False)
time.sleep(3600)
ex.discussion(example=0.75, show=False, save=True, sym_path=False, sym_vf=False, test=False)
time.sleep(3600)
ex.discussion(example=1, save=True, show=False, sym_path=False, sym_vf=False, test=False)
time.sleep(3600)
ex.discussion(example=1.039, save=True, show=False, sym_path=True, sym_vf=True, test=False)
time.sleep(360000)
ex.discussion(example=1.23, save=True, show=False, sym_path=False, sym_vf=False, test=True)
time.sleep(3600)
ex.discussion(example=1.23, save=False, show=False, sym_path=True, sym_vf=True, test=True)
print('Finished!')
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

ex.four_dim(plot=True)
ex.fBm_path(plot=True)
ex.pure_area(plot=True)
ex.smooth_path(plot=True, param=50, n=273)
time.sleep(3600)
ex.discussion(example=1.23, show=True, sym_path=True, sym_vf=True)
time.sleep(3600)
ex.smooth_path(plot=True, N=2, n=300, sym_path=False, sym_vf=True)
time.sleep(3600)

'''
def count():
    from math import sqrt
    for x in range(10**5):
        sqrt(x)

if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run("count()", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("time").print_stats(10)
'''

cProfile.run('ex.smooth_vf_smooth_path(n=102, N=1, k=4, plot=False, exact=True, n_steps=128, norm=rp.l1, var_steps=15)')
print("Hello")

'''
M = 1000
N = 1
path = np.cumsum(np.random.normal(0., np.sqrt(1./M), (M, 2)), axis=0)
print(f'True signature: {ts.stream2sig(path, N)}')
print(f'Signature: {rp.sig(path, N)}')
print(f'SigLogSig: {rp.logsig_to_sig(rp.sig_to_logsig(rp.sig(path, N)))}')
print(f'True LogSig: {ts.stream2logsig(path, N)}')
print(f'LogSig: {rp.sig_to_logsig(rp.sig(path, N))}')
time.sleep(3600)
'''

ex.smooth_vf_smooth_path_discussion(show=True, save=False, rounds=1, second_der=True, N_vec=np.array([1, 2, 3]))
