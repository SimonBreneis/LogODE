import numpy as np
import math
from esig import tosig as ts


def trivial_sig(dim, N):
    """
    Returns the signature up to degree N of a trivial dim-dimensional path.
    :param dim:
    :param N:
    :return:
    """
    result = [np.zeros([dim] * i) for i in range(N + 1)]
    result[0][0] = 1.
    return result


def sig_tensor_product(x, y, N=math.inf):
    """
    Computes the tensor product of the two signatures x and y up to level N.
    :param N:
    :param x:
    :param y:
    :return:
    """
    n = min(len(x), len(y), N)
    z = [0] * n
    for i in range(n):
        z[i] = np.sum(np.array([np.tensordot(x[j], y[i - j], axes=0) for j in range(i + 1)]), axis=0)
    return z


def sig(x, N):
    """
    Computes the signature of the path x (given as a vector) up to degree N.
    :param x:
    :param N:
    :return:
    """
    dim = x.shape[1]
    sig_vec = ts.stream2sig(x, N)
    indices = [int((dim ** (k + 1) - 1) / (dim - 1) + 0.1) for k in range(N + 1)]
    indices.insert(0, 0)
    return [sig_vec[indices[i]:indices[i + 1]].reshape([dim] * (i + 1)) for i in range(N + 1)]


def get_partitions(k, i):
    """
    Returns a list of all partitions of k into i summands (no commutativity of the addition is assumed).
    :param k:
    :param i:
    :return:
    """
    partitions = []
    current_partition = np.zeros(i, dtype=int)
    current_length = 0  # we have not yet added a number to current_partition

    def get_next_partition():
        nonlocal current_length

        if current_length == i:  # we have a complete current_partition
            if int(np.sum(current_partition)) == k:  # current_partition is an actual partition
                partitions.append(current_partition.copy())
            return

        next_element = 1  # next element of current_partition
        previous_sum = int(np.sum(current_partition[:current_length]))
        current_length += 1  # increase current length as we will now add a next element to current_partition

        while previous_sum + next_element + (i - current_length) <= k:
            current_partition[current_length - 1] = next_element
            get_next_partition()
            next_element += 1

        current_length -= 1

    get_next_partition()
    return partitions


def sig_to_logsig(s):
    N = len(s) - 1
    log_sig = trivial_sig(len(s[1]), N)
    log_sig[0][0] = 0.
    for k in range(1, N + 1):
        # computing the k-th level of the log-signature
        ls = s[k - 1].copy()
        for i in range(2, k + 1):
            # here are the terms of the k-th level we get from (-1)**(i+1) * 1/i * X**i
            ls_i = 0
            partitions = get_partitions(k, i)
            for partition in partitions:
                # we have a specific partition x^l_1 * x^l_2 * ... * x^l_i with l_1 + l_2 + ... + l_i = k
                partition = partition - 1  # indices start at 0
                partition_tensor = s[partition[0]].copy()
                for j in range(2, i + 1):
                    # compute the tensor product x^l_1 * ... * x^l_j
                    partition_tensor = np.tensordot(partition_tensor, s[partition[j - 1]].copy(), axes=0)
                ls_i += partition_tensor
            ls += (-1) ** (i + 1) / i * ls_i
        log_sig[k] = ls
    return log_sig


def logsig_to_sig(ls):
    N = len(ls) - 1
    s = trivial_sig(len(ls[1]), N)
    s += ls
    curr_tensor_prod = ls
    curr_factorial = 1.
    for k in range(2, N + 1):
        curr_tensor_prod = sig_tensor_product(curr_tensor_prod, ls)
        curr_factorial *= k
        s += curr_tensor_prod / curr_factorial
    return s


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
    def __init__(self, max_degree):
        self.max_degree = max_degree

    def incr(self, s, t, N):
        """
        Returns the n-th level of the rough path on the interval [s,t].
        :param s:
        :param t:
        :param N:
        :return:
        """
        pass


class RoughPathDiscrete(RoughPath):
    def __init__(self, times, values, max_degree=math.inf, store_signatures=False):
        super().__init__(max_degree)
        self.times = times
        self.val = values
        self.store_signatures = store_signatures

        if store_signatures:
            length = len(times) - 1
            n_discr_levels = int(np.log2(length)) + 1
            n_intervals = [0] * n_discr_levels
            n_intervals[0] = length
            for i in range(n_discr_levels - 1):
                n_intervals[i + 1] = np.ceil(n_intervals[i] / 2.)
            self.sig = [[trivial_sig(self.val.shape[1], max_degree) for _ in range(n_intervals[i])] for i in
                        range(n_discr_levels)]
            for j in range(n_intervals[0]):
                self.sig[0][j] = sig(np.array([self.val[times[j], :], self.val[times[j + 1], :]]), max_degree)
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
                left = self.incr_canonical(s_ind, int(s_ind / approx_diff) * (approx_diff + 1), N)
                right = self.incr_canonical(int(s_ind / approx_diff) * (approx_diff + 2), N)
            else:
                # we conclude that there is no dyadic interval of length approx_diff in [s_ind, t_ind]. We thus can find
                # at least one interval of length approx_diff/2.
                inner = self.sig[approx_level - 1][int(s_ind / approx_diff * 2) + 1][:(N + 1)]
                left = self.incr_canonical(s_ind, int(s_ind / approx_diff * 2) * (approx_diff / 2 + 1), N)
                right = self.incr_canonical(int(s_ind / approx_diff * 2) * (approx_diff / 2 + 2), N)
        li = sig_tensor_product(left, inner, N)
        return sig_tensor_product(li, right, N)

    def incr(self, s, t, N):
        s_ind = np.sum(map(lambda x: x < s, self.times))
        t_ind = np.sum(map(lambda x: x <= t, self.times)) - 1
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
                inner = self.incr_canonical(s_ind, t_ind)
            else:
                inner = sig(self.val[s_ind:(t_ind + 1), :], N)
            li = sig_tensor_product(left, inner, N)
            result = sig_tensor_product(li, right, N)
        return result


class RoughPathContinuous(RoughPath):
    def __init__(self, path, n_steps=15):
        super().__init__(math.inf)
        self.path = path
        self.n_steps = n_steps

    def incr(self, s, t, N):
        return sig(self.path(np.linspace(s, t, self.n_steps + 1)), N)


class RoughPathExact(RoughPath):
    def __init__(self, path, n_steps=15):
        super().__init__(max_degree=math.inf)
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
