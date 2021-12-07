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


k = 4
x = lambda t: np.array([np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]) / np.sqrt(k)
s = 0.2005
t = 0.3005
N = 3
n_steps = 15
rough_path = rp.RoughPathDiscrete(np.linspace(0, 1, 1501), x(np.linspace(0, 1, 1501)).T, max_degree=4, store_signatures=True)
print(rough_path.incr(s, t, N))
print(ts.stream2sig(x(np.linspace(s, t, 10*n_steps+1)).T, N))
time.sleep(3600)

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

ex.smooth_vf_smooth_path_discussion(show=False, save=True, rounds=1, exact=True, N_vec=np.array([1, 2, 3]))
ex.smooth_vf_smooth_path(plot=True, exact=True)
