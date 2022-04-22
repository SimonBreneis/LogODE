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


def solve_step_sig(g, f, y_0, atol=1e-07, rtol=1e-04, method='RK45'):
    """
    Implementation of the Log-ODE method.
    :param g: Signature/group-like element
    :param f: Vector field
    :param y_0: Current solution value
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :return: Solution of the Log-ODE method
    """
    return integrate.solve_ivp(lambda t, z: f.apply(g.log(), False)(z), (0, 1), y_0, method=method, atol=atol,
                               rtol=rtol).y[:, -1]


def solve_step_sig_full(g, f, y_0, atol=1e-07, rtol=1e-04, method='RK45', N_sol=None, n_intervals=0):
    """
    Implementation of the Log-ODE method. Returns the full solution, i.e. the solution as a rough path.
    :param g: Signature/group-like element
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :param n_intervals: Natural number, if it is greater than 0, uses an alternative, incorrect but faster method of
        computing the full solution. Larger value of partition is more accurate (as partition tends to infinity,
        the result is theoretically correct again)
    :return: Solution of the Log-ODE method
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            N_sol = g.n_levels()
    if isinstance(y_0, np.ndarray):
        y_0 = ta.sig_first_level_num(y_0, N_sol)
    if n_intervals == 0:
        f_full = f.extend(N_sol)
        solution = solve_step_sig(g, f_full, y_0.to_array(), atol=atol, rtol=rtol, method=method)
        solution = ta.array_to_tensor(solution, dim=y_0.dim())
    else:
        g_frac = g**(1. / n_intervals)
        y = np.zeros((n_intervals + 1, y_0.dim()))
        y[0, :] = y_0[1]
        for i in range(n_intervals):
            y[i+1, :] = solve_step_sig(g_frac, f, y[i, :], atol=atol / n_intervals, rtol=rtol / n_intervals,
                                       method=method)
        solution = ta.sig(y, N_sol)
    return solution


def solve_step(x, f, y_s, s, t, N, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False):
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
    y = integrate.solve_ivp(lambda r, z: f.apply(ls, compute_bound)(z), (0, 1), y_s, method=method, atol=atol,
                            rtol=rtol).y[:, -1]
    if compute_bound:
        return y, f.local_norm, x.omega(s, t)
    return y


def solve_fixed(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False):
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


def solve_fixed_full(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False, N_sol=None):
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
    if isinstance(x, rp.RoughPathContinuous) or isinstance(x, rp.RoughPathExact):
        sig_steps = x.sig_steps
    else:
        sig_steps = 2000
    y_list = [ta.array_to_tensor(y[:, i], len(y_0[1])).project_lie() for i in range(y.shape[1])]
    y = rp.rough_path_exact_from_exact_path(times=partition, path=y_list, sig_steps=sig_steps, p=x.p,
                                            var_steps=x.var_steps, norm=x.norm, x_0=y_0)
    return y, error


def solve_fixed_full_alt(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
                         N_sol=None):
    """
    Lazy implementation of the Log-ODE method. Returns the full solution, i.e. the solution as a rough path.
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

    y = rp.RoughPathDiscrete(times=partition, values=y, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol,
                             x_0=y_0)
    return y, error


def solve_fixed_adj_full(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
                         N_sol=None):
    """
    Implementation of the Log-ODE method. Returns the full solution z = (x, y), i.e. the solution as a rough path.
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
    f_ext = f.adjoin()
    if isinstance(y_0, np.ndarray):
        z_0 = np.zeros(x.dim() + len(y_0))
        z_0[x.dim():] = y_0
    else:
        z_0 = y_0.add_dimensions(front=x.dim(), back=0)
    return solve_fixed_full(x, f_ext, z_0, N, partition, atol, rtol, method, compute_bound, N_sol)


def solve_fixed_adj_full_alt(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
                             N_sol=None):
    """
    Lazy implementation of the Log-ODE method. Returns the full solution z = (x, y), i.e. the solution as a rough
    path. Really only solves the first level, and afterwards computes the signature. Faster, but in general
    (for p large) incorrect.
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

    x_dim = x.dim()
    z = np.empty((x_dim + y.shape[0], y.shape[1]))
    for i in range(y.shape[1]):
        z[:x_dim, i] = x.sig(partition[0], partition[i], 1)[1]
    z[x_dim:, :] = y
    z_0 = np.zeros(x_dim + y.shape[0])
    z_0[x_dim:] = y_0
    z = rp.RoughPathDiscrete(times=partition, values=z, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol,
                             x_0=z_0)
    return z, error


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


def solve_fixed_error_representation(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', speed=0):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N: The degree of the Log-ODE method (f needs to be Lip(N+1))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :return: Solution on partition points
    """
    if isinstance(y_0, np.ndarray):
        y_0 = ta.sig_first_level_num(y_0, N)
    x_dim = x.dim()
    y_dim = y_0.dim()
    z_dim = x_dim + y_dim
    h_dim = y_dim * y_dim
    linear_vf = vf.matrix_multiplication_vector_field(y_dim, norm=f.norm)
    tic = time.perf_counter()
    z, _ = solve_fixed_adj_full(x, f, y_0, N, partition, atol=atol, rtol=rtol, method=method, compute_bound=False,
                                N_sol=N)
    toc = time.perf_counter()
    reference_time = toc-tic
    y = z.project_space([i + x_dim for i in range(y_dim)])
    h_sig = [0]*(len(partition)-2)
    Psi = [0]*(len(partition)-1)
    Psi[-1] = np.eye(y_dim).flatten()
    propagated_local_errors = np.zeros((len(partition)-1, y_dim))
    y_on_partition = np.zeros((len(partition), y_dim))
    y_on_partition[0, :] = y_0[1]

    for i in range(1, len(partition)):
        y_on_partition[i, :] = y.at(partition[i], 1)[1]

    if speed < 0:
        n_intervals = 20
        if len(partition) > 3:
            tic = time.perf_counter()
            v_init = z.at(partition[1], N).add_dimensions(front=0, back=h_dim)
            v_sig = v_init.inverse() * solve_step_sig_full(z.sig(partition[1], partition[2], N), f.flow(), y_0=v_init,
                                                           atol=atol, rtol=rtol, method=method, N_sol=N,
                                                           n_intervals=n_intervals)
            v_sig.project_space([i + z_dim for i in range(h_dim)])
            toc = time.perf_counter()
            time_h = (toc-tic)*(len(partition)-2)/n_intervals
            n_intervals = int(np.ceil(np.fmax(1, reference_time/time_h*10)))
    elif speed > 0:
        n_intervals = int(np.ceil(1/speed))
    else:
        n_intervals = 0
    if N == 1:
        n_intervals = 0

    for i in range(len(partition)-2):
        v_init = z.at(partition[i], N).add_dimensions(front=0, back=h_dim)
        v_sig = v_init.inverse() * solve_step_sig_full(z.sig(partition[i], partition[i+1], N), f.flow(), y_0=v_init,
                                                       atol=atol, rtol=rtol, method=method, N_sol=N,
                                                       n_intervals=n_intervals)
        h_sig[i] = v_sig.project_space([i + z_dim for i in range(h_dim)])

    for i in range(len(partition)-2):
        Psi[-2-i] = solve_step_sig(h_sig[-1-i].inverse(), linear_vf, Psi[-1-i], atol=atol, rtol=rtol, method=method)

    if speed <= 0.1:
        n_intervals = 10
    elif speed >= 0.5:
        n_intervals = 2
    else:
        n_intervals = int(np.ceil(1/speed))

    for i in range(len(partition)-1):
        subpartition = np.linspace(partition[i], partition[i+1], n_intervals+1)
        y_local, _ = solve_fixed(x, f, y_0=y_on_partition[i, :], N=N, partition=subpartition, atol=atol/n_intervals,
                                 rtol=rtol/n_intervals, method=method, compute_bound=False)
        propagated_local_errors[i, :] = Psi[i].reshape((y_dim, y_dim)) @ (y_local[:, -1] - y_on_partition[i+1, :])

    # global_error = np.sum(propagated_local_errors, axis=0)
    return y_on_partition, propagated_local_errors


def solve_adaptive_error_representation_fixed_N(x, f, y_0, N, T=1., n=20, atol=1e-04, rtol=1e-02, method='RK45',
                                                speed=-1):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N: The degree of the Log-ODE method (f needs to be Lip(N+1))
    :param T: Final time
    :param n: Initial number of time intervals
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :return: Solution on partition points
    """
    atol = atol/3
    rtol = rtol/3
    partition = np.linspace(0, T, n+1)
    y, loc_err = solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, atol=atol/n, rtol=rtol/n,
                                                  method=method, speed=speed)
    global_err = np.sum(loc_err, axis=0)
    abs_err = ta.l1(global_err)
    rel_err = abs_err/x.norm(y[-1, :])
    print(len(partition)-1, abs_err, rel_err)
    while abs_err > atol and rel_err > rtol:
        abs_loc_err = np.array([ta.l1(loc_err[i, :]) for i in range(len(partition)-1)])
        med_abs_loc_err = np.median(abs_loc_err)
        print(np.median(abs_loc_err), np.average(abs_loc_err), np.std(abs_loc_err), np.amax(abs_loc_err))
        i = np.argmax(abs_loc_err)
        print(partition[i-2], partition[i-1], partition[i], partition[i+1], partition[i+2])
        relevant_ind = np.argwhere(abs_loc_err >= med_abs_loc_err).flatten()
        additional_times = np.zeros(len(relevant_ind))
        for i in range(len(relevant_ind)):
            additional_times[i] = (partition[relevant_ind[i]] + partition[relevant_ind[i]+1])/2
        joint_partition = np.zeros(len(partition) + len(additional_times))
        joint_partition[:len(partition)] = partition
        joint_partition[len(partition):] = additional_times
        partition = np.sort(joint_partition)
        y, loc_err = solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, atol=atol / n, rtol=rtol / n,
                                                      method=method, speed=speed)
        global_err = np.sum(loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err / x.norm(y[-1, :])
        print(len(partition)-1, abs_err, rel_err)
    return partition, y


def solve_adaptive_error_representation(x, f, y_0, N_min=1, N_max=3, T=1., n=20, atol=1e-04, rtol=1e-02, method='RK45',
                                        speed=-1):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N_min: Minimal degree of the Log-ODE method
    :param N_max: Maximal degree of the Log-ODE method
    :param T: Final time
    :param n: Initial number of time intervals
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :return: Solution on partition points
    """
    p = x.p
    if N_min + 1 <= p:
        N_min = int(np.ceil(p-0.999))
    if N_max < N_min:
        N_max = N_min
    atol = atol/3
    rtol = rtol/3
    partition = np.linspace(0, T, 3)
    for N in range(N_min, N_max+1):
        print(f'N={N}')
        solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, atol=1000*atol, rtol=1000*rtol,
                                         method=method, speed=speed)

    partition = np.linspace(0, T, 21)
    N_vec = np.array([i for i in range(N_min, N_max + 1)])
    time_vec = np.zeros(N_max + 1 - N_min)
    for i in range(len(N_vec)):
        print(f'N={N_vec[i]}')
        tic = time.perf_counter()
        y, loc_err = solve_fixed_error_representation(x, f, y_0, N=N_vec[i], partition=partition, atol=atol/n,
                                                      rtol=rtol/n, method=method, speed=speed)
        total_time = time.perf_counter() - tic
        global_err = np.sum(loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err/x.norm(y[-1, :])
        factor = np.fmin(abs_err/atol, rel_err/rtol)
        needed_n = 20 * factor**(p/(N_vec[i]+1-p))
        time_vec[i] = needed_n / 20 * total_time
    N = np.argmin(time_vec) + N_min
    print(f'found N={N}')
    return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N, T=T, n=n, atol=3*atol, rtol=3*rtol,
                                                       method=method, speed=speed)


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
    partition = np.linspace(0, T, n)
    sol = solve_fixed(x, f, y_0, N=N, partition=partition, atol=atol / (3 * n), rtol=rtol / (3 * n),
                      method=method)
    print(f'Solving the Log-ODE took {time.perf_counter() - tic:.3g} seconds.')
    return partition, sol


def solve_adaptive(x, f, y_0, T, atol=1e-03, rtol=1e-02, method='RK45'):
    """
    Implementation of the Log-ODE method. Using a-priori estimates, tries to find an efficient and sufficiently fine
    partition of [0, T] in the beginning. If the partition is not fine enough (this is checked with the a-priori
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


def solve(x, f, y_0, solver, N=1, T=1., partition=None, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
          N_sol=1, N_min=1, N_max=5, n_min=30, n_max=100, speed=-1):
    """
    Various Log-ODE implementations.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial value
    :param solver: f/a (fixed or adaptive), s/f (simple or full), s/a (simple or adjoined), last letter indicating the
        kind of algorithm used (if only one implementation exists, s for standard)
    :param N: Level of the Log-ODE method
    :param T: Final time
    :param partition: Time partition on which the Log-ODE method is applied
    :param atol: Absolute error tolerance
    :param rtol: Relative error tolerance
    :param method: Method for solving the ODEs
    :param compute_bound: Whether the theoretical a priori error bound should be computed
    :param N_sol: Level of the solution
    :param N_min: Minimal degree of the Log-ODE method
    :param N_max: Maximal degree of the Log-ODE method
    :param n_min: Minimal number of time intervals
    :param n_max: Maximal number of time intervals
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :return: Depends on the solver
    """
    if solver == 'fsss':
        return solve_fixed(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                           compute_bound=compute_bound)
    elif solver == 'ffss':
        return solve_fixed_full(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                                compute_bound=compute_bound, N_sol=N_sol)
    elif solver == 'ffsa':
        return solve_fixed_full_alt(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                                    compute_bound=compute_bound, N_sol=N_sol)
    elif solver == 'ffas':
        return solve_fixed_adj_full(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                                    compute_bound=compute_bound, N_sol=N_sol)
    elif solver == 'ffaa':
        return solve_fixed_adj_full_alt(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                                        compute_bound=compute_bound, N_sol=N_sol)
    elif solver == 'fssr':
        return solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol,
                                                method=method, speed=speed)
    elif solver == 'asss':
        return solve_adaptive(x, f, y_0, T=T, atol=atol, rtol=rtol, method=method)
    elif solver == 'assf':
        return solve_adaptive_faster(x, f, y_0, T=T, atol=atol, rtol=rtol, N_min=N_min, N_max=N_max, n_min=n_min,
                                     n_max=n_max, method=method)
    elif solver == 'assN':
        return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N, T=T, atol=atol, rtol=rtol, method=method,
                                                           speed=speed, n=n_min)
    elif solver == 'assr':
        return solve_adaptive_error_representation(x, f, y_0, N_min=N_min, N_max=N_max, T=T, n=n_min, atol=atol,
                                                   rtol=rtol, method=method, speed=speed)
