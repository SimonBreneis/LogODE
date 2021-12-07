import numpy as np
import math
from esig import tosig as ts
import scipy
import p_var


def l1(x):
    return np.sum(np.abs(x))


def var(x, p, dist):
    return p_var.p_var_backbone(np.shape(x)[0], p, lambda a, b: dist(x[a, :], x[b, :])).value ** (1 / p)


def beta(p):
    """
    Computes beta = p * (1 + sum_{r=2}^infinity (2/r)^((p+1)/p)).
    :param p: The roughness
    :return: beta
    """
    if p == 1:
        return 2 * np.pi ** 2 / 3 - 3
    if p == 2:
        return 11.12097
    if p == 3:
        return 22.66186
    else:
        return beta(min(3, int(p)))


def trivial_sig(dim, N):
    """
    Returns the signature up to degree N of a trivial dim-dimensional path.
    :param dim:
    :param N:
    :return:
    """
    result = [np.zeros([dim] * i) for i in range(N + 1)]
    result[0] = 1.
    return result


def sig_tensor_product(x, y, N=math.inf):
    """
    Computes the tensor product of the two signatures x and y up to level N.
    :param N:
    :param x:
    :param y:
    :return:
    """
    n = min(len(x), len(y), N+1)
    res = [np.sum(np.array([np.tensordot(x[j], y[i - j], axes=0) for j in range(i + 1)]), axis=0) for i in range(n)]
    res[0] = np.array(res[0])
    return res


def sig(x, N):
    """
    Computes the signature of the path x (given as a vector) up to degree N.
    :param x:
    :param N:
    :return:
    """
    if N == 1:
        return [1., x[-1, :] - x[0, :]]
    dim = x.shape[1]
    sig_vec = ts.stream2sig(x, N)
    indices = [int((dim ** (k + 1) - 1) / (dim - 1) + 0.1) for k in range(N + 1)]
    indices.insert(0, 0)
    res = [sig_vec[indices[i]:indices[i + 1]].reshape([dim] * i) for i in range(N + 1)]
    res[0] = float(res[0])
    return res


def sig_to_logsig(s):
    N = len(s)-1
    s_ = s.copy()
    s_[0] -= 1.
    ls = s_.copy()
    curr_tensor_prod = s_.copy()
    for k in range(2, N+1):
        curr_tensor_prod = sig_tensor_product(curr_tensor_prod, s_)
        ls = [ls[i] + curr_tensor_prod[i]/k*(-1)**(k+1) for i in range(len(ls))]
    return ls


def logsig_to_sig(ls):
    N = len(ls) - 1
    s = ls.copy()
    s[0] += 1.
    curr_tensor_prod = ls
    curr_factorial = 1.
    for k in range(2, N + 1):
        curr_tensor_prod = sig_tensor_product(curr_tensor_prod, ls)
        curr_factorial *= k
        s = [s[i] + curr_tensor_prod[i]/curr_factorial for i in range(len(s))]
    return s


def logsig(x, N):
    return sig_to_logsig(sig(x, N))


def extend_logsig(ls, N):
    """
    Extends the given level n log-signature ls to a level N log-signature
    :param ls:
    :param N:
    :return:
    """
    n = len(ls) - 1
    if N <= n:
        return ls[:(N + 1)]
    new_ls = trivial_sig(len(ls[1]), N)
    for i in range(n + 1):
        new_ls[i] = ls[i]
    return new_ls


def extend_sig(s, N):
    """
    Extends the given level n signature s to a level N signature.
    :param s:
    :param N:
    :return:
    """
    n = len(s) - 1
    if N <= n:
        return s[:(N + 1)]
    return logsig_to_sig(extend_logsig(sig_to_logsig(s), N))


class RoughPath:
    def __init__(self, max_degree, p=1, var_steps=15, norm=l1):
        self.max_degree = max_degree
        self.p = p
        self.var_steps = var_steps
        self.norm = norm

    def incr(self, s, t, N):
        """
        Returns the n-th level of the rough path on the interval [s,t].
        :param s:
        :param t:
        :param N:
        :return:
        """
        return trivial_sig(1, N)

    def p_variation(self, s, t, p, var_steps, norm):
        levels = int(p)
        times = np.linspace(s, t, var_steps+1)
        increments = [self.incr(times[i], times[i+1], levels) for i in range(var_steps)]
        variations = np.zeros(levels)
        for level in range(1, levels+1):
            def distance(i, j):
                if j < i:
                    temp = i
                    i = j
                    j = temp
                elif i == j:
                    return 0.
                total_increment = increments[i][:(level+1)]
                for k in range(i+1, j):
                    total_increment = sig_tensor_product(total_increment, increments[k][:(level+1)])
                return norm(total_increment[level])
            variations[level-1] = p_var.p_var_backbone(var_steps+1, p/level, distance).value**(level/p)
        return variations

    def omega(self, s, t, p=0., var_steps=0, norm=None):
        if p == 0.:
            p = self.p
        if var_steps == 0:
            var_steps = self.var_steps
        if norm is None:
            norm = self.norm
        variations = self.p_variation(s, t, p, var_steps, norm)
        omegas = beta(p) * np.array([scipy.special.gamma(i/p + 1) for i in range(1, int(p)+1)]) * variations
        omegas = np.array([omegas[i]**(p/(i+1)) for i in range(int(p))])
        return np.amax(omegas)


class RoughPathDiscrete(RoughPath):
    def __init__(self, times, values, max_degree=math.inf, p=1., var_steps=15, norm=l1, store_signatures=False):
        super().__init__(max_degree, p, var_steps, norm)
        self.times = times
        self.val = values
        self.store_signatures = store_signatures

        if store_signatures:
            length = len(times) - 1
            n_discr_levels = int(np.log2(length)) + 1
            n_intervals = [0] * n_discr_levels
            n_intervals[0] = length
            for i in range(n_discr_levels - 1):
                n_intervals[i + 1] = int(n_intervals[i] / 2.)
            self.sig = [[trivial_sig(self.val.shape[1], max_degree) for _ in range(n_intervals[i])] for i in
                        range(n_discr_levels)]
            for j in range(n_intervals[0]):
                self.sig[0][j] = sig(np.array([self.val[j, :], self.val[j+1, :]]), max_degree)
            for i in range(1, len(self.sig)):
                for j in range(n_intervals[i]):
                    self.sig[i][j] = sig_tensor_product(self.sig[i - 1][2 * j], self.sig[i - 1][2 * j + 1], max_degree)

    def incr_canonical(self, s_ind, t_ind, N):
        if s_ind == t_ind:
            return trivial_sig(self.val.shape[1], N)
        diff = t_ind - s_ind
        approx_level = int(np.log2(diff))
        # at this stage, we can be sure that there is a dyadic interval between s_ind and t_ind of length
        # 2**(approx_level-1), but we are not sure if there is a dyadic interval of length 2**(approx_level)
        approx_diff = 2 ** approx_level
        s_ind_ = s_ind % approx_diff
        # now that we reduced the setting to s_ind_ being in [0, approx_diff), we see that if there is an interval of
        # length approx_diff, it must be either [0, approx_diff] (if s_ind_==0), or [approx_diff, 2*approx_diff].
        if s_ind_ == 0:
            inner = self.sig[approx_level][int(s_ind / approx_diff)][:(N + 1)]
            left = trivial_sig(self.val.shape[1], N)
            right = self.incr_canonical(s_ind + approx_diff, t_ind, N)
        else:
            # the interval can then only be [approx_diff, 2*approx_diff] if s_ind_ + diff >= 2*approx_diff.
            if s_ind_ + diff >= 2 * approx_diff:
                inner = self.sig[approx_level][int(s_ind / approx_diff) + 1][:(N + 1)]
                left = self.incr_canonical(s_ind, int(s_ind / approx_diff + 1) * approx_diff, N)
                right = self.incr_canonical(int(s_ind / approx_diff + 2) * approx_diff, t_ind, N)
            else:
                # we conclude that there is no dyadic interval of length approx_diff in [s_ind, t_ind]. We thus can find
                # at least one interval of length approx_diff/2.
                inner = self.sig[approx_level - 1][int(s_ind / approx_diff * 2) + 1][:(N + 1)]
                left = self.incr_canonical(s_ind, int(int(s_ind / approx_diff * 2 + 1) * (approx_diff / 2)), N)
                right = self.incr_canonical(int(int(s_ind / approx_diff * 2 + 2) * (approx_diff / 2)), t_ind, N)
        li = sig_tensor_product(left, inner, N)
        return sig_tensor_product(li, right, N)

    def incr(self, s, t, N):
        s_ind = sum(map(lambda x: x < s, self.times))
        t_ind = sum(map(lambda x: x <= t, self.times)) - 1
        x_s = self.val[0, :]
        if s_ind > 0:
            x_s = self.val[s_ind - 1, :] + (self.val[s_ind, :] - self.val[s_ind - 1, :]) * (
                    s - self.times[s_ind - 1]) / (self.times[s_ind] - self.times[s_ind - 1])
        x_t = self.val[-1, :]
        if t_ind < len(self.times) - 1:
            x_t = self.val[t_ind, :] + (self.val[t_ind + 1, :] - self.val[t_ind, :]) * (t - self.times[t_ind]) / (
                    self.times[t_ind + 1] - self.times[t_ind])
        if s_ind > t_ind:
            result = sig(np.array([x_s, x_t]), N)
        elif s_ind == t_ind:
            result = sig(np.array([x_s, self.val[s_ind, :], x_t]), N)
        else:  # (s_ind < t_ind)
            left = sig(np.array([x_s, self.val[s_ind, :]]), N)
            right = sig(np.array([self.val[t_ind, :], x_t]), N)
            if self.store_signatures:
                inner = self.incr_canonical(s_ind, t_ind, N)
            else:
                inner = sig(self.val[s_ind:(t_ind + 1), :], N)
            li = sig_tensor_product(left, inner, N)
            result = sig_tensor_product(li, right, N)
        return result


class RoughPathContinuous(RoughPath):
    def __init__(self, path, n_steps=15, p=1., var_steps=15, norm=l1):
        super().__init__(math.inf, p, var_steps, norm)
        self.path = path
        self.n_steps = n_steps

    def incr(self, s, t, N):
        return sig(self.path(np.linspace(s, t, self.n_steps + 1)).T, N)


class RoughPathExact(RoughPath):
    def __init__(self, path, n_steps=15, p=1., var_steps=15, norm=l1):
        super().__init__(math.inf, p, var_steps, norm)
        self.path = path
        self.n_steps = n_steps

    def incr(self, s, t, N):
        result = self.path(s, t)
        exact_degree = len(result) - 1
        if N <= exact_degree:
            return result
        if self.n_steps == 1:
            return extend_sig(result, N)
        times = np.linspace(s, t, self.n_steps + 1)
        result = extend_sig(self.path(s, times[1]), N)
        for i in range(1, self.n_steps):
            result = sig_tensor_product(result, extend_sig(self.path(times[i], times[i+1]), N))
        return result
