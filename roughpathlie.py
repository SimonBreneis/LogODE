import numpy as np
import tensoralgebra as ta
import esig
import scipy


def beta(p):
    """
    Computes (or upper bounds) beta = p * (1 + sum_{r=2}^infinity (2/r)^(([p]+1)/p)).
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
        return 2 * p * (int(p) + 1) / (int(p) + 1 - p)  # upper bound


def local_log_ode_error_constant(N, p, dim):
    """
    Returns the constant in the error bound for a single step of the Log-ODE method.
    :param N: Degree of the method
    :param p: Roughness of the path
    :param dim: Dimension of the path
    :return: The constant
    """
    '''
    if p == 1:
        return 0.34 * (7 / 3.) ** (N + 1)
    if p == 2:
        return 25 * self.dim / scipy.special.gamma((N + 1) / p + 1) + 0.081 * (7 / 3) ** (N + 1)
    if p == 3:
        return 1000 * self.dim ** 3 / scipy.special.gamma((N + 1) / p + 1) + 0.038 * (7 / 3) ** (N + 1)
    '''
    if 1 <= p < 2:
        C = 1
    elif 2 <= p < 3:
        C = 3 * np.sqrt(dim)
    elif 3 <= p < 4:
        C = 7 * dim
    else:
        C = 21 * dim ** (9 / 4)
    b = beta(p)
    return (1.13 / b) ** (1 / int(p)) * (int(p) * C) ** int(p) / scipy.special.factorial(
        int(p)) / scipy.special.gamma((N + 1) / p + 1) + 0.83 * (7 / (3 * b ** (1 / N))) ** (N + 1)


class RoughPath:
    def __init__(self, p=1, norm=ta.l1, x_0=None):
        """
        Initialization.
        :param p: Measures the roughness
        :param norm: Norm used in computing the p-variation of the path
        :param x_0: Initial value of the rough path
        """
        self.p = p
        self.norm = norm
        self.at_0 = x_0

    def at(self, t, N):
        """
        Returns the value of the rough path up to level N at time t.
        :param t: Time point
        :param N: Level of the signature
        :return: The value of the path at time t up to level N
        """
        pass

    def sig(self, s, t, N):
        """
        Returns the signature up to level N of the rough path on the interval [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the signature
        :return: The signature up to level N on [s,t]
        """
        pass

    def logsig(self, s, t, N):
        """
        Returns the log-signature up to level N of the rough path on the interval [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the log-signature
        :return: The log-signature up to level N on [s,t]
        """
        pass

    def project_space(self, indices):
        """
        If the RoughPath is over the vector space R^d with basis (e_0, ..., e_{d-1}), then this method projects the
        RoughPath onto R^k with basis (e_{i_1}, ..., e_{i_k}), where indices = [i_1, ..., i_k].
        :param indices: Indices onto which the RoughPath should be projected
        :return: The projected RoughPath
        """
        pass

    def dim(self):
        """
        Returns the dimension of the underlying path.
        :return: The dimension d of x
        """
        pass


class RoughPathDiscrete(RoughPath):
    def __init__(self, times, values, p=1., norm=ta.l1, x_0=None):
        """
        Representation of the rough path that is useful and efficient if the rough path is given only by its first
        level, and only as a sequence of points.
        :param times: Time points where the rough path is given
        :param values: Values (only first level) of the rough path on the time points times
        :param p: Roughness of the path
        :param norm: Norm used in computing the p-variation of the path
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            x_0 = values[0, :]
        super().__init__(p, norm, x_0)
        self.times = times
        self.val = values

    def __values_in_interval(self, s, t):
        """
        Returns the array of path values in the interval [s, t].
        :param s: Initial time point
        :param t: Final time point
        :return: The array of values
        """
        s_ind = np.sum(self.times < s)
        t_ind = np.sum(self.times <= t) - 1
        x_s = self.val[0, :]
        if s_ind > 0:
            x_s = self.val[s_ind - 1, :] + (self.val[s_ind, :] - self.val[s_ind - 1, :]) * (
                    s - self.times[s_ind - 1]) / (self.times[s_ind] - self.times[s_ind - 1])
        x_t = self.val[-1, :]
        if t_ind < len(self.times) - 1:
            x_t = self.val[t_ind, :] + (self.val[t_ind + 1, :] - self.val[t_ind, :]) * (t - self.times[t_ind]) / (
                    self.times[t_ind + 1] - self.times[t_ind])
        val_array = np.empty((t_ind - s_ind + 3, self.dim()))
        val_array[0, :] = x_s
        val_array[-1, :] = x_t
        val_array[1:-1, :] = self.val[s_ind:t_ind + 1, :]
        return val_array

    def sig(self, s, t, N):
        return esig.stream2sig(self.__values_in_interval(s, t), N)

    def logsig(self, s, t, N):
        return esig.stream2logsig(self.__values_in_interval(s, t), N)

    def at(self, t, N):
        val_array = self.__values_in_interval(0, t)
        val_array = np.concatenate((np.zeros(self.dim()), val_array), axis=0)
        return esig.stream2sig(val_array, N)

    def dim(self):
        return self.val.shape[1]

    def project_space(self, indices):
        new_values = self.val[:, indices]
        new_starting_value = self.at_0.project_space(indices)
        return RoughPathDiscrete(times=self.times, values=new_values, p=self.p, norm=self.norm, x_0=new_starting_value)


class RoughPathContinuous(RoughPath):
    def __init__(self, path, sig_steps=2000, p=1., norm=ta.l1, x_0=None):
        """
        This representation of a rough path is useful if only the first level of the rough path is given, and this level
        is given as a vectorized function of one variable.
        :param path: The (first level of the) path given as a vectorized function
        :param sig_steps: Number of steps used in computing the signature
        :param p: Roughness of the path
        :param norm: Norm used in computing the p-variation
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            x_0 = path(0)
        super().__init__(p, norm, x_0)
        self.path = path
        self.sig_steps = sig_steps

    def sig(self, s, t, N):
        return esig.stream2sig(self.path(np.linspace(s, t, self.sig_steps + 1)).T, N)

    def logsig(self, s, t, N):
        return esig.stream2logsig(self.path(np.linspace(s, t, self.sig_steps + 1)).T, N)

    def at(self, t, N):
        val_array = self.path(np.linspace(0, t, self.sig_steps + 1)).T
        val_array = np.concatenate((np.zeros(self.dim()), val_array), axis=0)
        return esig.stream2sig(val_array, N)

    def dim(self):
        return len(self.at_0)

    def project_space(self, indices):
        return RoughPathContinuous(path=lambda t: self.path(t)[indices], sig_steps=self.sig_steps, p=self.p,
                                   norm=self.norm, x_0=self.at_0.project_space(indices))


class RoughPathExact(RoughPath):
    def __init__(self, path, sig_steps=2000, p=1., norm=ta.l1, x_0=None):
        """
        This representation of a rough path is useful if the rough path is given for multiple levels as a function of
        two time variables. It need not be vectorized.
        :param path: The path given as a function of two variables returning an instance of Tensor
        :param sig_steps: Number of steps used in approximating the signature (if the level needed is higher than the
            level given)
        :param p: Roughness of the path
        :param norm: Norm used in computing the p-variation
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            tens = path(0, 0)
            x_0 = ta.trivial_sig_num(dim=tens.dim(), N=tens.n_levels())
        elif isinstance(x_0, np.ndarray):
            x_0 = ta.sig_first_level_num(x_0, 1)
        super().__init__(p=p, norm=norm, x_0=x_0)
        self.path = path
        self.sig_steps = sig_steps

    def at(self, t, N):
        return (self.at_0 * ta.array_to_tensor(self.sig(0, t, N), self.dim())).to_array()

    def sig(self, s, t, N):
        result = self.path(s, t)
        exact_degree = len(result) - 1
        if N <= exact_degree:
            return result.project_level(N)
        if self.sig_steps == 1:
            return result.extend_sig(N)
        times = np.linspace(s, t, self.sig_steps + 1)
        result = self.path(s, times[1]).extend_sig(N)
        for i in range(1, self.sig_steps):
            result = result * self.path(times[i], times[i + 1]).extend_sig(N)
        return result.project_lie().to_array()

    def logsig(self, s, t, N):
        return ta.array_to_tensor(self.sig(s, t, N), self.dim()).log().to_array()

    def exact_degree(self):
        return len(self.path(0., 0.)) - 1

    def project_space(self, indices):
        return RoughPathExact(path=lambda s, t: self.path(s, t).project_space(indices), sig_steps=self.sig_steps,
                              p=self.p, norm=self.norm, x_0=self.at_0.project_space(indices))

    def dim(self):
        return self.at_0.dim()


class RoughPathList(RoughPath):
    def __init__(self, path, p=1., var_steps=15, norm=ta.l1):
        """
        This is a representation of the rough path as a list. It is indexed not by time but by the list index. For
        example, instead of calling x.sig(partition[i], partition[i + 1]), you would call x.sig(i, i + 1), where
        partition is the underlying partition of this rough path. Note that the rough path does not know or save the
        partition.
        :param path: A list of tensors, where path[i] is the path at time partition[i]
        :param p: Roughness of the path
        :param var_steps: Number of steps used in computing the p-variation
        :param norm: Norm used in computing the p-variation
        """
        x_0 = path[0]
        super().__init__(p, norm, x_0)
        self.path = path

    def at(self, t, N):
        return self.path[t].extend_sig(N).to_array()

    def sig(self, s, t, N):
        computation_level = int(np.fmin(np.fmax(self.path[s].n_levels(), self.path[t].n_levels()), N))
        return (self.path[s].inverse().extend_sig(computation_level)
                * self.path[t].extend_sig(computation_level)).extend_sig(N).to_array()

    def project_space(self, indices):
        return RoughPathList(path=[tens.project_space(indices) for tens in self.path], p=self.p, norm=self.norm)

    def dim(self):
        return self.path[0].dim()





def rough_path_exact_from_exact_path(times, path, sig_steps=2000, p=1, var_steps=15, norm=ta.l1, x_0=None):
    """
    Given the sequence X_{0, t_i} of (truncated) signatures of X for t_i in times, returns the corresponding RoughPath
    object (an instance of RoughPathExact).
    :param times: Time grid
    :param path: The rough path, instance of list, path[i] = X_{0, times[i]}
    :param sig_steps: Number of steps used in approximating the signature (if the level needed is higher than the
            level given)
    :param p: Roughness of the path
    :param var_steps: Number of steps used in approximating the p-variation
    :param norm: Norm used in computing the p-variation
    :param x_0: Initial value of the rough path
    :return: The path as an instance of RoughPathExact
    """
    def x_rp(s, t):
        """
        Computes X_{s, t}, with the same truncation level as in the input path.
        :param s: Initial time point
        :param t: Final time point
        :return: The signature on [s,t]
        """
        if s > t:
            return x_rp(t, s).inverse()
        s_ind = np.sum(times < s)
        t_ind = np.sum(times <= t) - 1
        x_s = path[0]
        if s_ind > 0:
            ds = (s - times[s_ind - 1]) / (times[s_ind] - times[s_ind - 1])
            x_s = path[s_ind - 1] * (path[s_ind - 1].inverse() * path[s_ind]) ** ds
        x_t = path[-1]
        if t_ind < len(times) - 1:
            dt = (t - times[t_ind]) / (times[t_ind + 1] - times[t_ind])
            x_t = path[t_ind] * (path[t_ind].inverse() * path[t_ind + 1]) ** dt
        return x_s.inverse() * x_t

    return RoughPathExact(path=x_rp, sig_steps=sig_steps, p=p, norm=norm, x_0=x_0)
