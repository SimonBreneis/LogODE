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
        v_init = z.at(partition[i+1], N).add_dimensions(front=0, back=h_dim)
        v_sig = v_init.inverse() * solve_step_sig_full(z.sig(partition[i+1], partition[i+2], N), f.flow(), y_0=v_init,
                                                       atol=atol, rtol=rtol, method=method, N_sol=N,
                                                       n_intervals=n_intervals)
        h_sig[i] = v_sig.project_space([j + z_dim for j in range(h_dim)])

    for i in range(len(partition)-2):
        Psi[-2-i] = solve_step_sig(h_sig[-1-i], linear_vf, Psi[-1-i], atol=atol, rtol=rtol, method=method)

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


def solve(x, f, y_0, solver, N=1, T=1., partition=None, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
          N_sol=1, N_min=1, N_max=5, n=20, speed=-1):
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
    :param n: Number of time intervals
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
    elif solver == 'assN':
        return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N, T=T, atol=atol, rtol=rtol, method=method,
                                                           speed=speed, n=n)
    elif solver == 'assr':
        return solve_adaptive_error_representation(x, f, y_0, N_min=N_min, N_max=N_max, T=T, n=n, atol=atol,
                                                   rtol=rtol, method=method, speed=speed)
