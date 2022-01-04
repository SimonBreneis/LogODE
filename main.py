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


def con(N, p, d):
    """
    Returns the constant in the error bound for a single step of the Log-ODE method.
    :param N: Degree of the method
    :return: The constant
    """
    '''
    if p == 1:
        return 0.34 * (7 / 3.) ** (N + 1)
    if p == 2:
        return 25 * d / scipy.special.gamma((N + 1) / p + 1) + 0.081 * (7 / 3) ** (N + 1)
    if p == 3:
        return 1000 * d ** 3 / scipy.special.gamma((N + 1) / p + 1) + 0.038 * (7 / 3) ** (N + 1)
    '''
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


var_steps = 200
x = ex.rough_fractional_Brownian_path(H=0.5, n=10*var_steps, dim=2, T=1., var_steps=var_steps, norm=ta.l1, save_level=2)
ps = np.linspace(2., 3., 50)[:-1]
constants = np.zeros(49)
vars = np.zeros(49)
omegas = np.zeros(49)

for i in range(len(ps)):
    print(i)
    constants[i] = con(3, ps[i], 2)
    if i == 0:
        vars[i] = np.amax(x.p_variation(0., 1., ps[i], var_steps=var_steps, norm=ta.l1))
    else:
        vars[i] = min(np.amax(x.p_variation(0., 1., ps[i], var_steps=var_steps, norm=ta.l1)), vars[i-1])
    omegas[i] = x.omega(0., 1., ps[i], var_steps=var_steps)
plt.plot(ps, constants, label='error constant')
plt.plot(ps, vars, label='p-variation')
plt.plot(ps, omegas/10000, label='control function')
plt.plot(ps, constants*omegas/100000, 'k-', label='error')
plt.title("Varying p for Brownian motion")
plt.ylim([0, 10])
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
