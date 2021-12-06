import numpy as np
import scipy
from scipy import integrate, special
from esig import tosig as ts
import matplotlib.pyplot as plt
import p_var
import time
import logode as lo
import examples as ex


'''
M = 1000
N = 3
path = np.cumsum(np.random.normal(0., np.sqrt(1./M), (M, 2)), axis=0)
print(ts.stream2logsig(path, N))
print(lo.log_sig(path, N))
time.sleep(3600)
'''

ex.smooth_vf_smooth_path_discussion(show=False, save=True, rounds=1, exact=True, N_vec=np.array([1, 2, 3]))
ex.smooth_vf_smooth_path(plot=True, exact=True)
