import numpy as np
import scipy
from scipy import integrate, special
from esig import tosig as ts
import matplotlib.pyplot as plt
import p_var
import time
import logode as lo
import examples as ex


ex.smooth_vf_smooth_path_discussion(show=False, save=True, rounds=1, exact=True, N_vec=np.array([1, 2, 3]))
ex.smooth_vf_smooth_path(plot=True, exact=True)
