import numpy as np
import scipy
import sympy as sp
from scipy import special
import p_var
import tensoralgebra as ta


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
        return 2*p*(int(p)+1)/(int(p)+1-p)  # upper bound


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
    def __init__(self, p=1, var_steps=15, norm=ta.l1, x_0=None):
        """
        Initialization.
        :param p: Measures the roughness
        :param var_steps: Number of steps used in estimating the p-variation of the path
        :param norm: Norm used in computing the p-variation of the path
        :param x_0: Initial value of the rough path
        """
        self.p = p
        self.var_steps = var_steps
        self.norm = norm
        self.at_0 = x_0

    def at(self, t, N):
        """
        Returns the value of the rough path up to level N at time t.
        :param t: Time point
        :param N: Level of the signature
        :return: The value of the path at time t up to level N
        """
        if self.at_0 is None:
            return self.sig(0, t, N)
        return self.at_0.extend_sig(N) * self.sig(0, t, N)

    def sig(self, s, t, N):
        """
        Returns the signature up to level N of the rough path on the interval [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the signature
        :return: The signature up to level N on [s,t]
        """
        return ta.trivial_tens_num(1, N)

    def logsig(self, s, t, N):
        """
        Returns the log-signature up to level N of the rough path on the interval [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the log-signature
        :return: The log-signature up to level N on [s,t]
        """
        return self.sig(s, t, N).log()

    def project_space(self, indices):
        """
        If the RoughPath is over the vector space R^d with basis (e_0, ..., e_{d-1}), then this method projects the
        RoughPath onto R^k with basis (e_{i_1}, ..., e_{i_k}), where indices = [i_1, ..., i_k].
        :param indices: Indices onto which the RoughPath should be projected
        :return: The projected RoughPath
        """
        pass

    def p_variation(self, s, t, p, var_steps, norm):
        """
        Estimates (lower bounds) the p-variation of the rough path on the interval [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param p: Roughness of the path
        :param var_steps: Number of steps in estimating the p-variation of the path
        :param norm: Norm used in computing the p-variation of the path
        :return: The p-variation of the path on [s,t]. An array, with the p/i-variation of the i-th component in the
            i-th place
        """
        if norm is None:
            return 0.
        levels = int(p)
        times = np.linspace(s, t, var_steps+1)
        increments = [self.sig(times[i], times[i + 1], levels) for i in range(var_steps)]
        variations = np.zeros(levels)
        for level in range(1, levels+1):
            if level == 1:
                if p == 1 or p == 1.:
                    return np.sum(np.array([increments[i].norm(1, norm) for i in range(var_steps)]))
                # This code also works for p == 1, but is considerably slower
                values = np.zeros((var_steps+1, increments[0].dim()))
                for i in range(var_steps):
                    values[i+1, :] = values[i, :] + increments[i][1]

                def distance(i, j):
                    return norm(values[j, :] - values[i, :])
            else:
                # This code also works for level == 1, but is considerably slower
                def distance(i, j):
                    if j < i:
                        i, j = j, i
                    elif i == j:
                        return 0.
                    total_increment = increments[i].project(level)
                    for k in range(i+1, j):
                        total_increment = total_increment * increments[k].project_level(level)
                    return total_increment.norm(level, norm)

            variations[level-1] = p_var.p_var_backbone(var_steps+1, p/level, distance).value**(level/p)
        return variations

    def omega(self, s, t, p=0., var_steps=0, norm=None, by_level=False):
        """
        Computes (lower bounds) the control function of the rough path on [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param p: Roughness parameter of the control function/the rough path
        :param var_steps: Number of steps used in estimating the p-variation
        :param norm: Norm used in computing the p-variation
        :param by_level: If True, returns an array with a dominating control function for each level
        :return: The control function of the rough path on [s,t]
        """
        if p == 0.:
            p = self.p
        if var_steps == 0:
            var_steps = self.var_steps
        if norm is None:
            norm = self.norm
        variations = self.p_variation(s, t, p, var_steps, norm)
        omegas = beta(p) * np.array([scipy.special.gamma(i/p + 1) for i in range(1, int(p)+1)]) * variations
        omegas = np.array([omegas[i]**(p/(i+1)) for i in range(int(p))])
        if by_level:
            return omegas
        return np.amax(omegas)

    def dim(self):
        return self.sig(0., 0., 1).dim()

    def rough_total_omega_estimate(self, T=1., n_intervals=100, p=0., var_steps=0, norm=None):
        times = np.linspace(0., T, n_intervals+1)
        return np.sum(np.array([self.omega(s=times[i], t=times[i+1], p=p, var_steps=var_steps, norm=norm)
                                for i in range(n_intervals)]))

    def estimate_p(self, T=1., p_lower=1., p_upper=5., reset_p=True):
        """
        Estimates a good choice for the roughness parameter p of the rough path on the interval [0,T].
        :param T: Final time, terminal time of the rough path
        :param p_lower: Lower point of range where an appropriate p is searched
        :param p_upper: Upper point of range where an appropriate p is searched
        :param reset_p: If True, resets the roughness p of the rough path to p
        :return: Estimate for p
        """
        var_steps = max(self.var_steps, 100)
        current_p = p_upper
        found_critical_p = False
        while not found_critical_p:
            omegas = self.omega(0., T, current_p, var_steps=var_steps, by_level=True)
            index = np.argmax(omegas)
            if index == len(omegas)-1:
                found_critical_p = True
                p_lower = int(current_p)
            else:
                current_p -= 0.5
                if current_p < p_lower:
                    found_critical_p = True
        print(p_lower)

        ps = np.linspace(p_lower, p_upper, 100)
        error_bounds = np.zeros(len(ps))
        for j in range(len(ps)):
            print(j)
            error_bounds[j] += self.omega(s=0., t=T, p=ps[j], var_steps=var_steps)
            error_bounds[j] *= local_log_ode_error_constant(int(np.ceil(ps[j])+1), ps[j], self.dim())
        index = np.argmin(error_bounds)
        p = ps[index]
        if reset_p:
            self.p = p
        return p


class RoughPathDiscrete(RoughPath):
    def __init__(self, times, values, p=1., var_steps=15, norm=ta.l1, save_level=0, x_0=None):
        """
        Representation of the rough path that is useful and efficient if the rough path is given only by its first
        level, and only as a sequence of points.
        :param times: Time points where the rough path is given
        :param values: Values (only first level) of the rough path on the time points times
        :param p: Roughness of the path
        :param var_steps: Number of steps used in computing the p-variation of the path
        :param norm: Norm used in computing the p-variation of the path
        :param save_level: The signature up to save_level is precomputed and saved on a dyadic grid to enhance
            performance
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            x_0 = ta.sig_first_level_num(values[0, :], max(1, save_level))
        elif isinstance(x_0, np.ndarray):
            x_0 = ta.sig_first_level_num(x_0, 1)
        super().__init__(p, var_steps, norm, x_0)
        self.times = times
        self.val = values
        self.save_level = save_level

        if save_level < 1:
            save_level = 1

        length = len(times) - 1
        n_discretization_levels = int(np.log2(length)) + 1
        n_intervals = [0] * n_discretization_levels
        n_intervals[0] = length
        for i in range(n_discretization_levels - 1):
            n_intervals[i + 1] = int(n_intervals[i] / 2.)
        self.signature = [[ta.trivial_sig_num(self.val.shape[1], save_level) for _ in range(n_intervals[i])] for i in
                          range(n_discretization_levels)]
        for j in range(n_intervals[0]):
            self.signature[0][j] = ta.sig(np.array([self.val[j, :], self.val[j + 1, :]]), save_level)
        for i in range(1, len(self.signature)):
            for j in range(n_intervals[i]):
                self.signature[i][j] = self.signature[i - 1][2 * j] * self.signature[i - 1][2 * j + 1]

    def get_sig(self, i, j, N):
        """
        Private function, do not call from outside!
        :param i: Level in the dyadic grid
        :param j: Index in the i-th level of the dyadic grid
        :param N: Degree of the signature
        :return: The signature
        """
        if N <= self.save_level:
            return self.signature[i][j].project_level(N)
        return self.signature[i][j].extend_sig(N)

    def canonical_increment(self, s_ind, t_ind, N):
        """
        Private function, do not call from outside! Computes the signature on [times[s_ind], times[t_ind]].
        :param s_ind: Index for initial time point
        :param t_ind: Index for final time point
        :param N: Level of the signature
        :return: The signature
        """
        if s_ind == t_ind:
            return ta.trivial_sig_num(self.val.shape[1], N)
        diff = t_ind - s_ind
        approx_level = int(np.log2(diff))
        # at this stage, we can be sure that there is a dyadic interval between s_ind and t_ind of length
        # 2**(approx_level-1), but we are not sure if there is a dyadic interval of length 2**(approx_level)
        approx_diff = 2 ** approx_level
        s_ind_ = s_ind % approx_diff
        # now that we reduced the setting to s_ind_ being in [0, approx_diff), we see that if there is an interval of
        # length approx_diff, it must be either [0, approx_diff] (if s_ind_==0), or [approx_diff, 2*approx_diff].
        if s_ind_ == 0:
            inner = self.get_sig(approx_level, int(s_ind / approx_diff), N)
            left = ta.trivial_sig_num(self.val.shape[1], N)
            right = self.canonical_increment(s_ind + approx_diff, t_ind, N)
        else:
            # the interval can then only be [approx_diff, 2*approx_diff] if s_ind_ + diff >= 2*approx_diff.
            if s_ind_ + diff >= 2 * approx_diff:
                inner = self.get_sig(approx_level, int(s_ind / approx_diff) + 1, N)
                left = self.canonical_increment(s_ind, int(s_ind / approx_diff + 1) * approx_diff, N)
                right = self.canonical_increment(int(s_ind / approx_diff + 2) * approx_diff, t_ind, N)
            else:
                # we conclude that there is no dyadic interval of length approx_diff in [s_ind, t_ind]. We thus can find
                # at least one interval of length approx_diff/2.
                inner = self.get_sig(approx_level - 1, int(s_ind / approx_diff * 2) + 1, N)
                left = self.canonical_increment(s_ind, int(int(s_ind / approx_diff * 2 + 1) * (approx_diff / 2)), N)
                right = self.canonical_increment(int(int(s_ind / approx_diff * 2 + 2) * (approx_diff / 2)), t_ind, N)
        return left * inner * right

    def sig(self, s, t, N):
        """
        Computes the signature up to level N on [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the signature
        :return: The signature
        """
        s_ind = np.sum(self.times < s)
        t_ind = np.sum(self.times <= t)-1
        x_s = self.val[0, :]
        if s_ind > 0:
            x_s = self.val[s_ind - 1, :] + (self.val[s_ind, :] - self.val[s_ind - 1, :]) * (
                    s - self.times[s_ind - 1]) / (self.times[s_ind] - self.times[s_ind - 1])
        x_t = self.val[-1, :]
        if t_ind < len(self.times) - 1:
            x_t = self.val[t_ind, :] + (self.val[t_ind + 1, :] - self.val[t_ind, :]) * (t - self.times[t_ind]) / (
                    self.times[t_ind + 1] - self.times[t_ind])
        if s_ind > t_ind:
            result = ta.sig(np.array([x_s, x_t]), N)
        elif s_ind == t_ind:
            result = ta.sig(np.array([x_s, self.val[s_ind, :], x_t]), N)
        else:  # (s_ind < t_ind)
            left = ta.sig(np.array([x_s, self.val[s_ind, :]]), N)
            right = ta.sig(np.array([self.val[t_ind, :], x_t]), N)
            if self.save_level > 0:
                inner = self.canonical_increment(s_ind, t_ind, N)
            else:
                inner = ta.sig(self.val[s_ind:(t_ind + 1), :], N)
            result = left * inner * right
        return result.project_lie()

    def project_space(self, indices):
        new_values = self.val[:, indices]
        new_starting_value = self.at_0.project_space(indices)
        return RoughPathDiscrete(times=self.times, values=new_values, p=self.p, var_steps=self.var_steps,
                                 norm=self.norm, save_level=self.save_level, x_0=new_starting_value)


class RoughPathContinuous(RoughPath):
    def __init__(self, path, sig_steps=2000, p=1., var_steps=15, norm=ta.l1, x_0=None):
        """
        This representation of a rough path is useful if only the first level of the rough path is given, and this level
        is given as a vectorized function of one variable.
        :param path: The (first level of the) path given as a vectorized function
        :param sig_steps: Number of steps used in computing the signature
        :param p: Roughness of the path
        :param var_steps: Number of steps used in computing the p-variation
        :param norm: Norm used in computing the p-variation
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            x_0 = ta.sig_first_level_num(path(0), 1)
        elif isinstance(x_0, np.ndarray):
            x_0 = ta.sig_first_level_num(x_0, 1)
        super().__init__(p, var_steps, norm, x_0)
        self.path = path
        self.sig_steps = sig_steps

    def sig(self, s, t, N):
        """
        Computes the signature up to level N on [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the signature
        :return: The signature
        """
        return ta.sig(self.path(np.linspace(s, t, self.sig_steps + 1)).T, N)

    def project_space(self, indices):
        return RoughPathContinuous(path=lambda t: self.path(t)[indices], sig_steps=self.sig_steps, p=self.p,
                                   var_steps=self.var_steps, norm=self.norm, x_0=self.at_0.project_space(indices))


class RoughPathExact(RoughPath):
    def __init__(self, path, sig_steps=2000, p=1., var_steps=15, norm=ta.l1, x_0=None):
        """
        This representation of a rough path is useful if the rough path is given for multiple levels as a function of
        two time variables. It need not be vectorized.
        :param path: The path given as a function of two variables returning an instance of Tensor
        :param sig_steps: Number of steps used in approximating the signature (if the level needed is higher than the
            level given)
        :param p: Roughness of the path
        :param var_steps: Number of steps used in approximating the p-variation
        :param norm: Norm used in computing the p-variation
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            tens = path(0, 0)
            x_0 = ta.trivial_sig_num(dim=tens.dim(), N=tens.n_levels())
        elif isinstance(x_0, np.ndarray):
            x_0 = ta.sig_first_level_num(x_0, 1)
        super().__init__(p, var_steps, norm, x_0)
        self.path = path
        self.sig_steps = sig_steps

    def sig(self, s, t, N):
        """
        Computes the signature up to level N on [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the signature
        :return: The signature
        """
        result = self.path(s, t)
        exact_degree = len(result) - 1
        if N <= exact_degree:
            return result.project_level(N)
        if self.sig_steps == 1:
            return result.extend_sig(N)
        times = np.linspace(s, t, self.sig_steps + 1)
        result = self.path(s, times[1]).extend_sig(N)
        for i in range(1, self.sig_steps):
            result = result * self.path(times[i], times[i+1]).extend_sig(N)
        return result.project_lie()

    def exact_degree(self):
        return len(self.path(0., 0.))-1

    def project_space(self, indices):
        return RoughPathExact(path=lambda s, t: self.path(s, t).project_space(indices), sig_steps=self.sig_steps,
                              p=self.p, var_steps=self.var_steps, norm=self.norm, x_0=self.at_0.project_space(indices))


class RoughPathSymbolic(RoughPath):
    def __init__(self, path, t, p=1, var_steps=15, norm=ta.l1, x_0=None):
        """
        This representation of a rough path is useful if the path is given as the first level in a symbolic (sympy)
        form, and is such that there is hope that symbolic integration for computing the signature may prove
        fruitful.
        :param path: The path, given as a function of one variable returning a sympy array that is the first level
        :param t: Sympy variable of the path path
        :param p: Roughness of the path
        :param var_steps: Number of steps used in approximating the p-variation
        :param norm: Norm used in computing the p-variation
        :param x_0: Initial value of the rough path
        """
        if x_0 is None:
            x_0 = ta.SymbolicTensor([1, path.subs(t, sp.Integer(0))])
        elif isinstance(x_0, np.ndarray):
            x_0 = ta.sig_first_level_num(x_0, 1)
        elif isinstance(x_0, sp.Array):
            x_0 = ta.SymbolicTensor([1, x_0])
        super().__init__(p, var_steps, norm, x_0)
        self.path = ta.SymbolicTensor([sp.Integer(1), path - path.subs(t, sp.Integer(0))])
        self.t = t
        self.path_num = [lambda s: 1, sp.lambdify(self.t, self.path[1], 'numpy')]
        self.derivatives = sp.Array([sp.diff(path[i], t) for i in range(len(path))])

    def new_level(self):
        """
        Private function, do not call from outside! If more signature levels are needed than so far have been computed,
        call this function to compute (and append) the next-highest level.
        :return: Nothing
        """
        nl = sp.tensorproduct(self.path[-1], self.derivatives).applyfunc(lambda x: sp.integrate(x, self.t))
        self.path.append(nl - nl.subs(self.t, sp.Integer(0)))
        self.path_num.append(sp.lambdify(self.t, self.path[-1], 'numpy'))

    def eval_path(self, t, N):
        """
        Private function, do not call from outside! Returns the signature of the path on [0,t] up to level N.
        :param t: Final time
        :param N: Level of the signature
        :return: The signature
        """
        res = ta.NumericTensor([np.array(self.path_num[i](t)) for i in range(N + 1)])
        res[0] = 1.
        return res

    def sig(self, s, t, N):
        """
        Computes the signature up to level N on [s,t].
        :param s: Initial time point
        :param t: Final time point
        :param N: Level of the signature
        :return: The signature
        """
        while N > len(self.path) - 1:
            self.new_level()
        x_s = self.eval_path(s, N)
        x_t = self.eval_path(t, N)
        return (x_s.inverse() * x_t).project_lie()

    def project_space(self, indices):
        return RoughPathSymbolic(path=self.path[1][indices], t=self.t, p=self.p, var_steps=self.var_steps,
                                 norm=self.norm, x_0=self.at_0.project_space(indices))


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
            x_s = path[s_ind - 1] * (path[s_ind-1].inverse()*path[s_ind])**ds
        x_t = path[-1]
        if t_ind < len(times) - 1:
            dt = (t - times[t_ind]) / (times[t_ind + 1] - times[t_ind])
            x_t = path[t_ind] * (path[t_ind].inverse() * path[t_ind+1])**dt
        return x_s.inverse() * x_t

    return RoughPathExact(path=x_rp, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm, x_0=x_0)
