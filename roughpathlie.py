import numpy as np
import tensoralgebra as ta
from esig import tosig as ts


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
        return ts.stream2sig(self.__values_in_interval(s, t), N)

    def logsig(self, s, t, N):
        return ts.stream2logsig(self.__values_in_interval(s, t), N)

    def at(self, t, N):
        val_array = self.__values_in_interval(0, t)
        val_array = np.concatenate((np.zeros(self.dim()), val_array), axis=0)
        return ts.stream2sig(val_array, N)

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
        return ts.stream2sig(self.path(np.linspace(s, t, self.sig_steps + 1)).T, N)

    def logsig(self, s, t, N):
        return ts.stream2logsig(self.path(np.linspace(s, t, self.sig_steps + 1)).T, N)

    def at(self, t, N):
        val_array = self.path(np.linspace(0, t, self.sig_steps + 1)).T
        val_array = np.concatenate((np.zeros(self.dim()), val_array), axis=0)
        return ts.stream2sig(val_array, N)

    def dim(self):
        return len(self.at_0)

    def project_space(self, indices):
        return RoughPathContinuous(path=lambda t: self.path(t)[indices], sig_steps=self.sig_steps, p=self.p,
                                   norm=self.norm, x_0=self.at_0.project_space(indices))
