import math
import time
import numpy as np
import scipy
from scipy import integrate, special
import roughpath as rp
import vectorfield as vf


def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def insert_list(master, insertion, index):
    """
    Inserts the list insertion to the list master starting at index.
    :param master: List that is being modified
    :param insertion: List that is being inserted
    :param index: Index where the list insertion is being inserted
    :return: Nothing
    """
    for i in range(len(insertion)):
        master.insert(index + i, insertion[i])


class LogODESolver:
    def __init__(self, x, f, y_0, method='RK45'):
        """
        Initialization.
        :param x: Driving rough path, instance of RoughPath
        :param f: Vector field, instance of VectorField
        :param y_0: Initial condition, instance of numpy.ndarray
        :param method: Method used in the ODE solver (see scipy.integrate.ivp)
        """
        self.x = x
        self.f = f
        self.y_0 = y_0
        self.method = method
        self.dim = self.x.sig(0., 0., 1).dim()  # Dimension of the underlying rough path x

    def solve_step(self, y_s, s, t, N, atol, rtol, compute_bound=False):
        """
        Implementation of the Log-ODE method.
        :param N: The degree of the Log-ODE method (f needs to be Lip(N))
        :param y_s: Current solution value
        :param s: Initial interval point
        :param t: Final interval point
        :param atol: Absolute error tolerance of the ODE solver
        :param rtol: Relative error tolerance of the ODE solver
        :param compute_bound: If True, also returns a theoretical error bound
        :return: Solution on partition points
        """
        if compute_bound:
            self.f.reset_local_norm()
        ls = self.x.logsig(s, t, N)
        y = integrate.solve_ivp(lambda t, z: self.f.vector_field(ls, compute_bound)(z), (0, 1), y_s, method=self.method,
                                atol=atol, rtol=rtol).y[:, -1]
        if compute_bound:
            return y, self.f.local_norm, self.x.omega(s, t)
        return y

    def solve_fixed(self, N, partition, atol, rtol, compute_bound=False):
        """
        Implementation of the Log-ODE method.
        :param N: The degree of the Log-ODE method (f needs to be Lip(N))
        :param partition: Partition of the interval on which we apply the Log-ODE method
        :param atol: Absolute error tolerance of the ODE solver
        :param rtol: Relative error tolerance of the ODE solver
        :param compute_bound: If True, also returns a theoretical error bound
        :return: Solution on partition points, error bound (-1 if no norm was specified)
        """
        y = np.zeros(shape=(len(self.y_0), len(partition)))
        y[:, 0] = self.y_0
        p = self.x.p

        error_estimate = 0.
        tic = time.perf_counter()
        last_time = tic
        for i in range(1, len(partition)):
            toc = time.perf_counter()
            if toc - last_time > 10:
                print(
                    f'{100 * (i - 1) / (len(partition) - 1):.2f}% complete, estimated {int((toc - tic) / (i - 1) * (len(partition) - i))}s remaining.')
                last_time = toc
            if compute_bound:
                y[:, i], vf_norm, omega = self.solve_step(y[:, i - 1], partition[i - 1], partition[i], N, atol, rtol, compute_bound)
                vf_norm = np.amax(np.array(vf_norm)[:N])
                error_estimate += vf_norm ** (N + 1) * omega ** ((N + 1) / p)
            else:
                y[:, i] = self.solve_step(y[:, i - 1], partition[i - 1], partition[i], N, atol, rtol, compute_bound)
        if compute_bound:
            return y, self.local_log_ode_error_constant(N) * error_estimate
        return y

    def local_log_ode_error_constant(self, N):
        """
        Returns the constant in the error bound for a single step of the Log-ODE method.
        :param N: Degree of the method
        :return: The constant
        """
        p = self.x.p
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
            C = 3 * np.sqrt(self.dim)
        elif 3 <= p < 4:
            C = 7 * self.dim
        else:
            C = 21 * self.dim ** (9 / 4)
        beta = rp.beta(p)
        return (1.13 / beta) ** (1 / int(p)) * (int(p) * C) ** int(p) / scipy.special.factorial(
            int(p)) / scipy.special.gamma((N + 1) / p + 1) + 0.83 * (7 / (3 * beta ** (1 / N))) ** (N + 1)

    def solve_adaptive_faster(self, T, atol=1e-03, rtol=1e-02, N_min=1, N_max=5, n_min=30, n_max=100):
        atol = atol/2
        rtol = rtol/2
        if isinstance(self.x, rp.RoughPathExact):
            exact_degree = self.x.exact_degree()
            N_min = max(N_min, exact_degree)
            N_max = max(N_max, exact_degree)
        parameter_search_start = time.perf_counter()
        ns = np.exp(np.linspace(np.log(n_min), np.log(n_max), 5))
        ns = np.array([int(n) for n in ns])
        n_ref = 2*ns[-1]
        Ns = np.array([i for i in range(N_min, N_max+1)])
        sol_dim = len(self.solve_fixed(N=1, partition=np.array([0., T]), atol=1e+10*atol, rtol=1e+10*atol)[:, -1])
        true_sols = np.zeros((len(Ns), sol_dim))
        true_sol_paths = [0]*len(Ns)
        approx_sols = np.zeros((len(Ns), len(ns), sol_dim))
        errors = np.zeros((len(Ns), len(ns)))
        times = np.zeros((len(Ns), len(ns)))
        error_constants = np.zeros(len(Ns))
        error_exponents = np.zeros(len(Ns))
        time_constants = np.zeros(len(Ns))
        time_exponents = np.zeros(len(Ns))
        intervals_needed = np.zeros(len(Ns))
        times_needed = np.zeros(len(Ns))

        i = 0
        index = 0
        found_parameters = False
        increase_n = False
        current_time_estimate = math.inf  # estimate for how long it takes to solve the Log-ODE with the desired accuracy
        while not found_parameters:
            i = 0
            index = 0
            found_parameters = False
            while not found_parameters and not increase_n and i < len(Ns) and time.perf_counter() - parameter_search_start < current_time_estimate/10:
                if time_constants[i] == 0:
                    print('Computing new derivatives or integrals...')
                    self.solve_fixed(N=Ns[i], partition=np.array([0, T]), atol=atol * 1e+10, rtol=rtol * 1e+10)
                    print('Compute time estimates')
                    time_estimate_start = time.perf_counter()
                    self.solve_fixed(N=Ns[i], partition=np.linspace(0, T, 11), atol=atol/20, rtol=rtol/20)
                    time_estimate = time.perf_counter() - time_estimate_start
                    time_constants[i] = time_estimate/10
                    time_exponents[i] = 1.
                approx_time_est_N = time_constants[i] * (np.sum(ns ** time_exponents[i]) + n_ref**time_exponents[i])
                print(f'Time estimate: {time.perf_counter() - parameter_search_start + approx_time_est_N}, {current_time_estimate/2}')
                if time.perf_counter() - parameter_search_start + approx_time_est_N > current_time_estimate/2:
                    found_parameters = True
                else:
                    print('Onto the true solution!')
                    true_sol_paths[i] = self.solve_fixed(N=Ns[i], partition=np.linspace(0, T, n_ref+1), atol=atol/(2*n_ref), rtol=rtol/(2*n_ref))
                    true_sols[i, :] = true_sol_paths[i][:, -1]
                    for j in range(len(ns)):
                        print(f'N={Ns[i]}, n={ns[j]}')
                        tic = time.perf_counter()
                        approx_sols[i, j, :] = self.solve_fixed(N=Ns[i], partition=np.linspace(0, T, ns[j]+1),
                                                                atol=atol / (2 * ns[j]),
                                                                rtol=rtol / (2 * ns[j]))[:, -1]
                        times[i, j] = time.perf_counter() - tic
                        errors[i, j] = self.f.norm(true_sols[i, :] - approx_sols[i, j, :])
                        print(times[i, j], errors[i, j])
                    error_exponents[i], error_constants[i], _, _, _ = log_linear_regression(ns, errors[i])
                    print(error_exponents[i], error_constants[i])
                    if error_exponents[i] > 0:
                        error_exponents[i] = 0.
                        if i >= 2:
                            if error_exponents[i-2] == 0 or error_exponents[i-1] == 0:
                                increase_n = True
                    time_exponents[i], time_constants[i], _, _, _ = log_linear_regression(ns, times[i])
                    if time_exponents[i] < 0.01:
                        time_exponents[i] = 0.01
                    print(time_exponents[i], time_constants[i])
                    if error_exponents[i] >= 0:
                        intervals_needed[i] = math.inf
                    else:
                        intervals_needed[i] = (atol / error_constants[i]) ** (1 / error_exponents[i])
                    print(intervals_needed)
                    times_needed[i] = time_constants[i] * intervals_needed[i] ** time_exponents[i]
                    print(times_needed)
                    index = np.argmin(times_needed[:(i+1)])
                    current_time_estimate = times_needed[index]
                    print(index)
                    if error_exponents[i] < 0 and current_time_estimate < 10*(time.perf_counter() - parameter_search_start):
                        found_parameters = True
                    print(found_parameters)
                    i += 1

            if found_parameters and 10*(time.perf_counter()-parameter_search_start) < current_time_estimate:
                found_parameters = False
                increase_n = True

            if not increase_n:
                found_parameters = True
            else:
                ns = 2*ns
                n_ref = 3*n_ref
                increase_n = False

        N = Ns[index]
        n = int(max(intervals_needed[index], 1))
        print(f'Chosen N={N}, n={n}')
        print(f'Finding suitable parameters took {time.perf_counter()-parameter_search_start:.3g} seconds.')
        tic = time.perf_counter()
        if n < n_ref:
            print('Already had good enough approximation.')
            return true_sol_paths[index]
        sol = self.solve_fixed(N=N, partition=np.linspace(0, T, n), atol=atol / (3 * n), rtol=rtol / (3 * n))
        print(f'Solving the Log-ODE took {time.perf_counter() - tic:.3g} seconds.')
        return sol

    def solve_adaptive(self, T, atol=1e-03, rtol=1e-02):
        """
        Implementation of the Log-ODE method.
        :param T: Final time
        :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
        :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
        :return: Solution on partition points
        """
        var_steps = self.x.var_steps
        p = self.x.p
        p = int(p)
        norm_estimates = np.zeros(10)
        self.x.var_steps = max(var_steps, 100)
        _, norm_estimates[0], omega_estimate = self.solve_step(self.y_0, s=0, t=T, N=p, atol=10 * atol, rtol=10 * rtol)
        _, norm_estimates[1], _ = self.solve_step(self.y_0, s=0, t=T, N=p + 1, atol=10 * atol, rtol=10 * rtol)
        _, norm_estimates[2], _ = self.solve_step(self.y_0, s=0, t=T, N=p + 2, atol=10 * atol, rtol=10 * rtol)
        _, norm_estimates[3], _ = self.solve_step(self.y_0, s=0, t=T, N=p + 3, atol=10 * atol, rtol=10 * rtol)
        self.x.var_steps = var_steps
        if norm_estimates[3] == 0: norm_estimates[3] = max(norm_estimates[0], norm_estimates[1], norm_estimates[2])
        if norm_estimates[3] == 0: norm_estimates[3] = 1.
        if norm_estimates[2] == 0: norm_estimates[2] = norm_estimates[3]
        if norm_estimates[1] == 0: norm_estimates[1] = norm_estimates[2]
        if norm_estimates[0] == 0: norm_estimates[0] = norm_estimates[1]
        norm_incr = max(norm_estimates[3] - norm_estimates[2], norm_estimates[2] - norm_estimates[1],
                        norm_estimates[1] - norm_estimates[0])
        norm_estimates[4:] = norm_estimates[3] + norm_incr * np.arange(1, 7)
        print(f"Norm estimates: {norm_estimates}")
        print(f"Error constants: {np.array([self.local_log_ode_error_constant(N) for N in range(p, p + 10)])}")
        print(f"Omega: {omega_estimate}")
        number_intervals = np.array([(self.local_log_ode_error_constant(N) * norm_estimates[N - p] ** (
                N + 1) * omega_estimate ** ((N + 1.) / p) / atol) ** (p / (N - p + 1)) for N in range(p, p + 10)])
        print(f"Number of intervals: {number_intervals}")
        complexity = np.array([self.dim ** N for N in range(p, p + 10)]) * number_intervals
        N = np.argmin(complexity) + p
        print(f"N = {N}")
        number_intervals = number_intervals[N - p]
        print("Let us find a partition!")
        partition = self.find_partition(s=0, t=T, total_omega=omega_estimate, n=number_intervals)
        print("We found a partition!")
        atol = atol
        rtol = rtol
        local_Ns = [N] * (len(partition) - 1)
        max_local_error = [atol / len(partition)] * (len(partition) - 1)
        y = [self.y_0]

        i = 0
        while i < len(partition) - 1:
            print(f"At index {i + 1} of {len(partition) - 1}")
            local_N = local_Ns[i]
            y_next, norm_est, omega_est = self.solve_step(y_s=y[i], s=partition[i], t=partition[i + 1], N=local_N,
                                                          atol=max_local_error[i] / 2,
                                                          rtol=rtol / atol * max_local_error[i] / 2)
            print(f"Norm estimate: {norm_est}, Omega estimate: {omega_est}")
            error_est = self.local_log_ode_error_constant(local_N) * norm_est ** (local_N + 1) * omega_est ** (
                    (local_N + 1.) / p)
            print(f"Error estimate: {error_est}, Maximal error: {max_local_error[i]}")
            if error_est < max_local_error[i]:
                y.append(y_next)
                i += 1
            else:
                new_error_est = error_est
                new_local_N = local_N
                subpartition = 1
                while new_error_est >= max_local_error[i]:
                    error_est_N = self.local_log_ode_error_constant(new_local_N + 1) * norm_est ** (
                            new_local_N + 2) * omega_est ** ((new_local_N + 2.) / p)
                    error_est_part = error_est / self.dim ** ((new_local_N + 1) / p) * self.dim
                    if error_est_N < error_est_part:
                        new_local_N += 1
                    else:
                        subpartition *= 4
                    new_error_est = subpartition * self.local_log_ode_error_constant(new_local_N) * norm_est ** (
                            new_local_N + 1) * (omega_est / subpartition) ** ((new_local_N + 1.) / p)
                if subpartition > 1:
                    better_var_est = self.x.omega(partition[i], partition[i + 1], var_steps=var_steps * 3)
                    new_subpartition = self.find_partition(partition[i], partition[i + 1], better_var_est, subpartition)
                    insert_list(partition, new_subpartition[1:-1], i + 1)
                    insert_list(local_Ns, [new_local_N] * (len(new_subpartition) - 2), i + 1)
                    insert_list(max_local_error,
                                [max_local_error[i] / (len(new_subpartition) - 1)] * (len(new_subpartition) - 2), i + 1)
                    max_local_error[i] = max_local_error[i] / (len(new_subpartition) - 1)
                local_Ns[i] = new_local_N
        return np.array(partition), np.array(y)

    def find_next_interval(self, s, t, lower_omega, upper_omega):
        """
        Finds an interval of the form [s,u] with s <= u <= t such that lower_omega <= omega(s, u) <= upper_omega.
        :param s: Initial point of total interval
        :param t: Final point of total interval
        :param lower_omega: Lower bound of the control function on the new interval
        :param upper_omega: Upper bound of the control function on the new interval
        :return: The partition point u.
        """
        total_omega = self.x.omega(s, t)
        if total_omega <= upper_omega:
            return t
        u_current = s + (t - s) * (lower_omega + upper_omega) / (2 * total_omega)
        u_left = s
        u_right = t

        current_omega = self.x.omega(s, u_current)

        while not lower_omega <= current_omega <= upper_omega and u_right - u_left > (t - s) * 10 ** (-10):
            if current_omega > lower_omega:
                u_right = u_current
            else:
                u_left = u_current

            u_current = (u_left + u_right) / 2
            current_omega = self.x.omega(s, u_current)

        return u_current

    def find_partition(self, s, t, total_omega, n, q=2.):
        """
        Finds a partition of the interval [0,T] such that omega(0,T)/(qn) <= omega(s,t) <= q*omega(0,T)/n.
        :param s: Initial time
        :param t: Final time
        :param total_omega: Estimate for the total control function of x on [0,T]
        :param n: Approximate number of intervals into which [0,T] should be split
        :param q: Tolerance in finding the "perfect" partition.
        :return: The partition
        """
        q = max(q, 1.1)
        p = self.x.p
        partition = [s]
        lower_omega = total_omega / (q * n) ** (1 / p)
        upper_omega = total_omega * (q / n) ** (1 / p)
        print(f"Total omega: {total_omega}")
        print(f"Lower omega: {lower_omega}")
        print(f"Upper omega: {upper_omega}")
        while not partition[-1] == t:
            next_point = self.find_next_interval(partition[-1], t, lower_omega, upper_omega)
            partition.append(next_point)
        return partition
