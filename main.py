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
