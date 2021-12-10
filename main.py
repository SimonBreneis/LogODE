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
import cProfile

import sympy as sp
x = sp.symbols('x')
a = sp.Integral(sp.cos(x)*sp.exp(x), x)
print(sp.Eq(a, a.doit()))

y, z = sp.symbols('y z')
logistic = 1/(1 + sp.exp(-x))
print(logistic.evalf(subs={x: 1}))
f = sp.Array([[z-y, -z], [logistic.subs(x, z), logistic.subs(x, y-2*z)]])
print(f.subs([(y, 1), (z, sp.Rational(3, 10))]))
highest_der = f
base_func = f
vars = sp.Array([y, z])
der_highest_der = sp.Array([sp.diff(highest_der, vars[i]) for i in range(len(vars))])
permutations = [*range(3)]
permutations[0] = 1
permutations[1] = 0
next_order = sp.permutedims(sp.tensorcontraction(sp.tensorproduct(base_func, der_highest_der), (0, 2)), permutations)
print(next_order.rank())
a, b, c, d = sp.symbols('a b c d')
dx = sp.Array([[a, b], [c, d]])
print(sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(next_order, dx), (1, 3)), (1, 2)))
time.sleep(3600)

ex.smooth_vf_smooth_path(exact=False)
time.sleep(3600)

cProfile.run('ex.smooth_vf_smooth_path(n=102, N=1, k=4, plot=False, exact=True, n_steps=128, norm=rp.l1, var_steps=15)')
print("Hello")
'''
k = 4
x = lambda t: np.array([np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]) / np.sqrt(k)
s = 0.2005
t = 0.3005
s = 0.
t = 1.
print(rp.beta(1)*lo.var(x(np.linspace(s, t, 16)).T, 1., lambda a, b: rp.l1(b-a)))
N = 3
n_steps = 15
rough_path = rp.RoughPathDiscrete(np.linspace(0, 1, 1501), x(np.linspace(0, 1, 1501)).T, save_level=4)
print(rough_path.p_variation(s, t, 2., 15, rp.l1))
print(rough_path.omega(s, t, 2.))
print(rough_path.incr(s, t, N))
print(ts.stream2sig(x(np.linspace(s, t, 10*n_steps+1)).T, N))
time.sleep(3600)
'''

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

ex.smooth_vf_smooth_path_discussion(show=True, save=False, rounds=1, exact=True, N_vec=np.array([1, 2, 3]))
