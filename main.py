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

print(vec_field.vector_field(x.logsig(0.1, 0.143, 2))(np.array([-0.3, -0.52])))
print(vec_field_.vector_field(x_.logsig(0.1, 0.143, 2))(np.array([-0.3, -0.52])))

'''
ex.discussion(example=0.4, show=False, save=True, sym_path=False, sym_vf=True)
time.sleep(3600)
ex.discussion(example=0.5, show=False, save=True, sym_path=False, sym_vf=True)
time.sleep(3600)
'''
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
