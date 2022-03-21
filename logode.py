import math
import time
import numpy as np
import scipy
from scipy import integrate, special, stats
import roughpath as rp
import vectorfield as vf
import tensoralgebra as ta


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


def solve_step(x, f, y_s, s, t, N, atol, rtol, method='RK45', compute_bound=False):
    """
    Implementation of the Log-ODE method.
    :param x: Rough path
    :param f: Vector field
    :param N: The degree of the Log-ODE method (f needs to be Lip(N))
    :param y_s: Current solution value
    :param s: Initial interval point
    :param t: Final interval point
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :return: Solution on partition points
    """
    ls = x.logsig(s, t, N)
    if compute_bound:
        f.reset_local_norm()
    y = integrate.solve_ivp(lambda t, z: f.vector_field(ls, compute_bound)(z), (0, 1), y_s, method=method, atol=atol,
                            rtol=rtol).y[:, -1]
    if compute_bound:
        return y, f.local_norm, x.omega(s, t)
    return y


def solve_fixed(x, f, y_0, N, partition, atol, rtol, method='RK45', compute_bound=False):
    """
    Implementation of the Log-ODE method.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial value
    :param N: The degree of the Log-ODE method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    y = np.zeros(shape=(len(y_0), len(partition)))
    y[:, 0] = y_0
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
            y[:, i], vf_norm, omega = solve_step(x, f, y[:, i - 1], partition[i - 1], partition[i], N, atol, rtol,
                                                 method, compute_bound)
            vf_norm = np.amax(np.array(vf_norm)[:N])
            error_estimate += vf_norm ** (N + 1) * omega ** ((N + 1) / x.p)
        else:
            y[:, i] = solve_step(x, f, y[:, i - 1], partition[i - 1], partition[i], N, atol, rtol, method,
                                 compute_bound)
    if compute_bound:
        return y, local_log_ode_error_constant(N, x.sig(0., 0., 1).dim(), x.p) * error_estimate
    return y, -1


def solve_fixed_full(x, f, y_0, N, partition, atol, rtol, method='RK45', compute_bound=False, N_sol=None):
    """
    Implementation of the Log-ODE method. Returns the full solution, i.e. the solution as a rough path.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Log-ODE method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            N_sol = N
    if isinstance(y_0, np.ndarray):
        y_0 = ta.sig_first_level_num(y_0, N_sol)
    f_full = f.extend(N_sol)
    y, error = solve_fixed(x, f_full, y_0.to_array(), N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                           compute_bound=compute_bound)
    sig_steps = 2000
    if isinstance(x, rp.RoughPathContinuous) or isinstance(x, rp.RoughPathExact):
        sig_steps = x.sig_steps
    y_list = [ta.array_to_tensor(y[:, i], len(y_0[1])) for i in range(y.shape[1])]
    y = rp.rough_path_exact_from_exact_path(times=partition, path=y_list, sig_steps=sig_steps, p=x.p,
                                            var_steps=x.var_steps, norm=x.norm)
    # y = rp.RoughPathPrefactor(y, y_0)
    return y, error


def solve_fixed_full_alt(x, f, y_0, N, partition, atol, rtol, method='RK45', compute_bound=False, N_sol=None):
    """
    Half-assed implementation of the Log-ODE method. Returns the full solution, i.e. the solution as a rough path.
    Really only solves the first level, and afterwards computes the signature. Faster, but in general (for p large)
    incorrect.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Log-ODE method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            N_sol = N
    if isinstance(y_0, ta.Tensor):
        y_0 = y_0[1]
    if compute_bound:
        y, error = solve_fixed(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                               compute_bound=compute_bound)
    else:
        y = solve_fixed(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                        compute_bound=compute_bound)
        error = -1

    y = rp.RoughPathDiscrete(times=partition, values=y, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol)
    # y = rp.RoughPathPrefactor(y, y_0)
    return y, error


def local_log_ode_error_constant(N, dim, p):
    """
    Returns the constant in the error bound for a single step of the Log-ODE method.
    :param N: Degree of the method
    :param dim: Dimension of the driving signal
    :param p: Roughness of the driving signal
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
    beta = rp.beta(p)
    return (1.13 / beta) ** (1 / int(p)) * (int(p) * C) ** int(p) / scipy.special.factorial(
        int(p)) / scipy.special.gamma((N + 1) / p + 1) + 0.83 * (7 / (3 * beta ** (1 / N))) ** (N + 1)


def solve_adaptive_faster(x, f, y_0, T, atol=1e-03, rtol=1e-02, N_min=1, N_max=5, n_min=30, n_max=100, method='RK45'):
    """
    Implementation of the Log-ODE method. Computes error estimates for small number of intervals by comparing these
    approximate solutions to a solution computed with a higher number of intervals. Does this for various values of N
    (i.e. various levels of the Log-ODE method). For each N, estimates a convergence rate (and the corresponding
    constant) from these error estimates. Then uses the N and the number of intervals which is most efficient at
    achieving the desired error accuracy given these estimates.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param T: Final time
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param N_min: Minimum level of the Log-ODE method that is used
    :param N_max: Maximum level of the Log-ODE method that is used
    :param n_min: Minimal number of subintervals used in estimating the errors
    :param n_max: Maximal number of subintervals used in estimating the errors
    :param method: Method for solving the ODEs
    :return: Solution on partition points
    """
    atol = atol/2
    rtol = rtol/2
    if isinstance(x, rp.RoughPathExact):
        exact_degree = x.exact_degree()
        N_min = max(N_min, exact_degree)
        N_max = max(N_max, exact_degree)
    parameter_search_start = time.perf_counter()
    ns = np.exp(np.linspace(np.log(n_min), np.log(n_max), 5))
    ns = np.array([int(n) for n in ns])
    n_ref = 2*ns[-1]
    Ns = np.array([i for i in range(N_min, N_max+1)])
    sol_dim = len(solve_fixed(x, f, y_0, N=1, partition=np.array([0., T]), atol=1e+10*atol, rtol=1e+10*atol,
                              method=method, compute_bound=False)[:, -1])
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

    index = 0
    found_parameters = False
    increase_n = False
    current_time_estimate = math.inf  # estimate for how long it takes to solve the Log-ODE with the desired accuracy
    while not found_parameters:
        i = 0
        index = 0
        found_parameters = False
        while not found_parameters and not increase_n and i < len(Ns) \
                and time.perf_counter() - parameter_search_start < current_time_estimate/10:
            if time_constants[i] == 0:
                print('Computing new derivatives or integrals...')
                solve_fixed(x, f, y_0, N=Ns[i], partition=np.array([0, T]), atol=atol * 1e+10, rtol=rtol * 1e+10,
                            method=method)
                print('Compute time estimates')
                time_estimate_start = time.perf_counter()
                solve_fixed(x, f, y_0, N=Ns[i], partition=np.linspace(0, T, 11), atol=atol/20, rtol=rtol/20,
                            method=method)
                time_estimate = time.perf_counter() - time_estimate_start
                time_constants[i] = time_estimate/10
                time_exponents[i] = 1.
            approx_time_est_N = time_constants[i] * (np.sum(ns ** time_exponents[i]) + n_ref**time_exponents[i])
            print(f'Time estimate: {time.perf_counter() - parameter_search_start + approx_time_est_N}, {current_time_estimate/2}')
            if time.perf_counter() - parameter_search_start + approx_time_est_N > current_time_estimate/2:
                found_parameters = True
            else:
                print('Onto the true solution!')
                true_sol_paths[i] = solve_fixed(x, f, y_0, N=Ns[i], partition=np.linspace(0, T, n_ref+1),
                                                atol=atol/(2*n_ref), rtol=rtol/(2*n_ref), method=method)
                true_sols[i, :] = true_sol_paths[i][:, -1]
                for j in range(len(ns)):
                    print(f'N={Ns[i]}, n={ns[j]}')
                    tic = time.perf_counter()
                    approx_sols[i, j, :] = solve_fixed(x, f, y_0, N=Ns[i], partition=np.linspace(0, T, ns[j]+1),
                                                       atol=atol / (2 * ns[j]), rtol=rtol / (2 * ns[j]),
                                                       method=method)[:, -1]
                    times[i, j] = time.perf_counter() - tic
                    errors[i, j] = f.norm(true_sols[i, :] - approx_sols[i, j, :])
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
                i = i + 1

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
    sol = solve_fixed(x, f, y_0, N=N, partition=np.linspace(0, T, n), atol=atol / (3 * n), rtol=rtol / (3 * n),
                      method=method)
    print(f'Solving the Log-ODE took {time.perf_counter() - tic:.3g} seconds.')
    return sol


def solve_adaptive(x, f, y_0, T, atol=1e-03, rtol=1e-02, method='RK45'):
    """
    Implementation of the Log-ODE method. Using a-priori estimates, tries to find an efficient and sufficiently fine
    partition of [0, T] in the beginning. If the partition is not fine enough (this is cheched with the a-priori
    bounds), then it is refined.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param T: Final time
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :return: Solution on partition points
    """
    var_steps = x.var_steps
    x_dim = x.sig(0, 0, 1).dim()
    p = x.p
    p = int(p)
    norm_estimates = np.zeros(10)
    x.var_steps = max(var_steps, 100)
    _, norm_estimates[0], omega_estimate = solve_step(x, f, y_0, s=0, t=T, N=p, atol=10 * atol, rtol=10 * rtol,
                                                      method=method, compute_bound=True)
    _, norm_estimates[1], _ = solve_step(x, f, y_0, s=0, t=T, N=p + 1, atol=10 * atol, rtol=10 * rtol, method=method,
                                         compute_bound=True)
    _, norm_estimates[2], _ = solve_step(x, f, y_0, s=0, t=T, N=p + 2, atol=10 * atol, rtol=10 * rtol, method=method,
                                         compute_bound=True)
    _, norm_estimates[3], _ = solve_step(x, f, y_0, s=0, t=T, N=p + 3, atol=10 * atol, rtol=10 * rtol, method=method,
                                         compute_bound=True)
    x.var_steps = var_steps
    if norm_estimates[3] == 0:
        norm_estimates[3] = max(norm_estimates[0], norm_estimates[1], norm_estimates[2])
    if norm_estimates[3] == 0:
        norm_estimates[3] = 1.
    if norm_estimates[2] == 0:
        norm_estimates[2] = norm_estimates[3]
    if norm_estimates[1] == 0:
        norm_estimates[1] = norm_estimates[2]
    if norm_estimates[0] == 0:
        norm_estimates[0] = norm_estimates[1]
    norm_increment = max(norm_estimates[3] - norm_estimates[2], norm_estimates[2] - norm_estimates[1],
                         norm_estimates[1] - norm_estimates[0])
    norm_estimates[4:] = norm_estimates[3] + norm_increment * np.arange(1, 7)
    print(f"Norm estimates: {norm_estimates}")
    print(f"Error constants: {np.array([local_log_ode_error_constant(N, x_dim, p) for N in range(p, p + 10)])}")
    print(f"Omega: {omega_estimate}")
    number_intervals = np.array([(local_log_ode_error_constant(N, x_dim, p) * norm_estimates[N - p] ** (
            N + 1) * omega_estimate ** ((N + 1.) / p) / atol) ** (p / (N - p + 1)) for N in range(p, p + 10)])
    print(f"Number of intervals: {number_intervals}")
    complexity = np.array([x_dim ** N for N in range(p, p + 10)]) * number_intervals
    N = np.argmin(complexity) + p
    print(f"N = {N}")
    number_intervals = number_intervals[N - p]
    print("Let us find a partition!")
    partition = find_partition(omega=x.omega, p=x.p, s=0, t=T, n=number_intervals)
    print("We found a partition!")
    atol = atol
    rtol = rtol
    local_Ns = [N] * (len(partition) - 1)
    max_local_error = [atol / len(partition)] * (len(partition) - 1)
    y = [y_0]

    i = 0
    while i < len(partition) - 1:
        print(f"At index {i + 1} of {len(partition) - 1}")
        local_N = local_Ns[i]
        y_next, norm_est, omega_est = solve_step(x, f, y_s=y[i], s=partition[i], t=partition[i + 1], N=local_N,
                                                 atol=max_local_error[i] / 2,
                                                 rtol=rtol / atol * max_local_error[i] / 2, method=method,
                                                 compute_bound=True)
        print(f"Norm estimate: {norm_est}, Omega estimate: {omega_est}")
        error_est = local_log_ode_error_constant(local_N, x_dim, p) * norm_est ** (local_N + 1) * omega_est ** (
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
                error_est_N = local_log_ode_error_constant(new_local_N + 1, x_dim, p) * norm_est ** (
                        new_local_N + 2) * omega_est ** ((new_local_N + 2.) / p)
                error_est_part = error_est / x_dim ** ((new_local_N + 1) / p) * x_dim
                if error_est_N < error_est_part:
                    new_local_N += 1
                else:
                    subpartition *= 4
                new_error_est = subpartition * local_log_ode_error_constant(new_local_N, x_dim, p) * norm_est ** (
                        new_local_N + 1) * (omega_est / subpartition) ** ((new_local_N + 1.) / p)
            if subpartition > 1:
                x.var_steps = x.var_steps*3
                new_subpartition = find_partition(omega=x.omega, p=x.p, s=partition[i], t=partition[i + 1],
                                                  n=subpartition)
                x.var_steps = int(x.var_steps)/3
                insert_list(partition, new_subpartition[1:-1], i + 1)
                insert_list(local_Ns, [new_local_N] * (len(new_subpartition) - 2), i + 1)
                insert_list(max_local_error,
                            [max_local_error[i] / (len(new_subpartition) - 1)] * (len(new_subpartition) - 2), i + 1)
                max_local_error[i] = max_local_error[i] / (len(new_subpartition) - 1)
            local_Ns[i] = new_local_N
    return np.array(partition), np.array(y)


def find_next_interval(omega, s, t, lower_omega, upper_omega):
    """
    Finds an interval of the form [s,u] with s <= u <= t such that lower_omega <= omega(s, u) <= upper_omega.
    :param omega: The control function
    :param s: Initial point of total interval
    :param t: Final point of total interval
    :param lower_omega: Lower bound of the control function on the new interval
    :param upper_omega: Upper bound of the control function on the new interval
    :return: The partition point u.
    """
    total_omega = omega(s, t)
    if total_omega <= upper_omega:
        return t
    u_current = s + (t - s) * (lower_omega + upper_omega) / (2 * total_omega)
    u_left = s
    u_right = t

    current_omega = omega(s, u_current)

    while not lower_omega <= current_omega <= upper_omega and u_right - u_left > (t - s) * 10 ** (-10):
        if current_omega > lower_omega:
            u_right = u_current
        else:
            u_left = u_current

        u_current = (u_left + u_right) / 2
        current_omega = omega(s, u_current)

    return u_current


def find_partition(omega, p, s, t, n, q=2.):
    """
    Finds a partition of the interval [0,T] such that omega(0,T)/(qn) <= omega(s,t) <= q*omega(0,T)/n.
    :param omega: The control function with respect to which a partition should be found
    :param p: The roughness parameter of omega (omega is the control of a p-rough path)
    :param s: Initial time
    :param t: Final time
    :param n: Approximate number of intervals into which [s, t] should be split
    :param q: Tolerance in finding the "perfect" partition.
    :return: The partition
    """
    total_omega = omega(s, t)
    q = max(q, 1.1)
    partition = [s]
    lower_omega = total_omega / (q * n) ** (1 / p)
    upper_omega = total_omega * (q / n) ** (1 / p)
    print(f"Total omega: {total_omega}")
    print(f"Lower omega: {lower_omega}")
    print(f"Upper omega: {upper_omega}")
    while not partition[-1] == t:
        next_point = find_next_interval(omega, partition[-1], t, lower_omega, upper_omega)
        partition.append(next_point)
    return partition
