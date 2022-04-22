import math
import time
import numpy as np
import scipy
from scipy import integrate, special, stats
import roughpath as rp
import vectorfield as vf
import tensoralgebra as ta
import sympy as sp
import examples as ex


def step_sig(g, f, y_0):
    """
    Implementation of the Euler method.
    :param g: Signature/group-like element
    :param f: Vector field or one-form
    :param y_0: Current solution value
    :return: Solution of the Euler method
    """
    return y_0 + f.apply(g, False)(y_0)


def step_sig_full(g, f, y_0, lie=True, N_sol=None, n_intervals=0):
    """
    Implementation of the Euler method. Returns the full solution, i.e. the solution as a rough path.
    :param g: Signature/group-like element
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param lie: If True, projects onto the Lie group
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :param n_intervals: Natural number, if it is greater than 0, uses an alternative, incorrect but faster method of
        computing the full solution. Larger value of partition is more accurate (as partition tends to infinity,
        the result is theoretically correct again)
    :return: Solution of the Euler method
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
        solution = step_sig(g, f_full, y_0.to_array())
        solution = ta.array_to_tensor(solution, dim=y_0.dim())
        if lie:
            solution = solution.project_lie()
    else:
        g_fraction = g**(1. / n_intervals)
        y = np.zeros((n_intervals + 1, y_0.dim()))
        y[0, :] = y_0[1]
        for i in range(n_intervals):
            y[i+1, :] = step_sig(g_fraction, f, y[i, :])
        solution = ta.sig(y, N_sol)
    return solution


def step(x, f, y_s, s, t, N):
    """
    Implementation of the Euler method.
    :param x: Rough path
    :param f: Vector field
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param y_s: Current solution value
    :param s: Initial interval point
    :param t: Final interval point
    :return: Solution on partition points
    """
    return step_sig(x.sig(s, t, N), f, y_s)


def fixed(x, f, y_0, N, partition):
    """
    Implementation of the Euler method.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial value
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    y = np.zeros(shape=(len(y_0), len(partition)))
    y[:, 0] = y_0
    tic = time.perf_counter()
    last_time = tic
    for i in range(1, len(partition)):
        toc = time.perf_counter()
        if toc - last_time > 10:
            print(
                f'{100 * (i - 1) / (len(partition) - 1):.2f}% complete, estimated {int((toc - tic) / (i - 1) * (len(partition) - i))}s remaining.')
            last_time = toc
        y[:, i] = step(x, f, y[:, i - 1], partition[i - 1], partition[i], N)
    return y


def fixed_full(x, f, y_0, N, partition, lie=True, N_sol=None, n_intervals=0):
    """
    Implementation of the Euler method. Returns the full solution, i.e. the solution as a rough path.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
    :param lie: If True, projects onto the Lie group
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :param n_intervals: Natural number, if it is greater than 0, uses an alternative, incorrect but faster method of
        computing the full solution. Larger value of partition is more accurate (as partition tends to infinity,
        the result is theoretically correct again)
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            N_sol = N
    if isinstance(y_0, np.ndarray):
        y_0 = ta.sig_first_level_num(y_0, N_sol)
    f.extend(N_sol)  # pre-computation
    y = [0]*len(partition)
    y[0] = y_0
    tic = time.perf_counter()
    last_time = tic
    for i in range(1, len(partition)):
        toc = time.perf_counter()
        if toc - last_time > 10:
            print(
                f'{100 * (i - 1) / (len(partition) - 1):.2f}% complete, estimated {int((toc - tic) / (i - 1) * (len(partition) - i))}s remaining.')
            last_time = toc
        y[i] = step_sig_full(x.sig(partition[i - 1], partition[i], N), f, y[i - 1], lie=lie, N_sol=N_sol,
                             n_intervals=n_intervals)
    if isinstance(x, rp.RoughPathContinuous) or isinstance(x, rp.RoughPathExact):
        sig_steps = x.sig_steps
    else:
        sig_steps = 2000
    y = rp.rough_path_exact_from_exact_path(times=partition, path=y, sig_steps=sig_steps, p=x.p,
                                            var_steps=x.var_steps, norm=x.norm)
    # y = rp.RoughPathPrefactor(y, y_0)
    return y


def solve_fixed_full_alt(x, f, y_0, N, partition, N_sol=None):
    """
    Lazy implementation of the Euler method. Returns the full solution, i.e. the solution as a rough path.
    Really only solves the first level, and afterwards computes the signature. Faster, but in general (for p large)
    incorrect.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
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
    y = fixed(x, f, y_0, N=N, partition=partition)
    y = rp.RoughPathDiscrete(times=partition, values=y, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol)
    return y


def solve_fixed_adj_full(x, f, y_0, N, partition, lie=True, N_sol=None):
    """
    Implementation of the Euler method. Returns the full solution z = (x, y), i.e. the solution as a rough path.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
    :param lie: If True, projects onto the Lie group
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    f_ext = f.adjoin()
    if isinstance(y_0, np.ndarray):
        z_0 = np.zeros(x.dim() + len(y_0))
        z_0[x.dim():] = y_0
    else:
        z_0 = y_0.add_dimensions(front=x.dim(), back=0)
    return fixed_full(x, f_ext, z_0, N=N, partition=partition, lie=lie, N_sol=N_sol)


def solve_fixed_adj_full_alt(x, f, y_0, N, partition, N_sol=None):
    """
    Lazy implementation of the Euler method. Returns the full solution z = (x, y), i.e. the solution as a rough
    path. Really only solves the first level, and afterwards computes the signature. Faster, but in general
    (for p large) incorrect.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
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
    y = fixed(x, f, y_0, N=N, partition=partition)
    x_dim = x.dim()
    z = np.empty((x_dim + y.shape[0], y.shape[1]))
    for i in range(y.shape[1]):
        z[:x_dim, i] = x.sig(partition[0], partition[i], 1)[1]
    z[x_dim:, :] = y
    z = rp.RoughPathDiscrete(times=partition, values=z, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol)
    return z


def integral(x, f, N, partition):
    """
    Computes the rough integral int_0^T f(x_t) dx_t using the Euler method.
    :param x: Rough path
    :param f: One-form
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
    :return: The integral. Precise output determined by the variables full_output, adjoined_output, and path_output
    """
    y = np.empty((len(partition), f.dim_y))
    y[0, :] = np.zeros(f.dim_y)

    for i in range(len(partition)-1):
        g = x.sig(partition[i], partition[i+1], N)
        x_s = x.at(partition[i], 1)[1]
        y[i+1, :] = y[i, :] + f.apply(g)(x_s)

    return y


def integral_full(x, f, N, partition, N_sol=None, lie=True):
    """
    Computes the rough integral int_0^T f(x_t) dx_t using the Euler method.
    :param x: Rough path
    :param f: One-form
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
    :param N_sol: Level of the solution if full_output is 1 or 2. If None, uses N as the level
    :param lie: If True, projects onto the Lie group
    :return: The integral. Precise output determined by the variables full_output, adjoined_output, and path_output
    """
    if N_sol is None:
        N_sol = N

    y = [0]*len(partition)
    y[0] = ta.trivial_sig_num(dim=f.dim_y, N=N_sol)

    for i in range(len(partition) - 1):
        g = x.sig(partition[i], partition[i + 1], N)
        x_s = x.at(partition[i], 1)[1]
        y[i + 1, :] = y[i]*f.full_one_form(N_sol, x_s, g, lie)

    sig_steps = 2000
    if isinstance(x, rp.RoughPathContinuous) or isinstance(x, rp.RoughPathExact):
        sig_steps = x.sig_steps

    return rp.rough_path_exact_from_exact_path(times=partition, path=y, sig_steps=sig_steps, p=x.p,
                                               var_steps=x.var_steps, norm=x.norm,
                                               x_0=ta.trivial_sig_num(dim=f.dim_y, N=N_sol))


def integral_full_alt(x, f, N, partition, N_sol=None):
    """
    Computes the rough integral int_0^T f(x_t) dx_t using the Euler method. Lazy implementation of the Euler method.
    Really only solves the first level, and afterwards computes the signature. Faster, but in general
    (for p large) incorrect.
    :param x: Rough path
    :param f: One-form
    :param N: The degree of the Euler method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Euler method
    :param N_sol: Level of the solution if full_output is 1 or 2. If None, uses N as the level
    :return: The integral. Precise output determined by the variables full_output, adjoined_output, and path_output
    """
    if N_sol is None:
        N_sol = N

    y = integral(x=x, f=f, N=N, partition=partition)

    return rp.RoughPathDiscrete(times=partition, values=y, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol,
                                x_0=ta.trivial_sig_num(dim=f.dim_y, N=N_sol))


def solve_fixed_error_representation(x, f, y_0, N, partition, speed=0, lie=True):
    """
    Implementation of the Euler method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N: The degree of the Euler method (f needs to be Lip(N+1))
    :param partition: Partition of the interval on which we apply the Euler method
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param lie: If True, projects onto the Lie group
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
    z, _ = solve_fixed_adj_full(x, f, y_0, N, partition, lie=lie, N_sol=N)
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
            v_sig = v_init.inverse() * step_sig_full(z.sig(partition[1], partition[2], N), f.flow(), y_0=v_init,
                                                     lie=lie, N_sol=N, n_intervals=n_intervals)
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
        v_sig = v_init.inverse() * step_sig_full(z.sig(partition[i], partition[i + 1], N), f.flow(), y_0=v_init,
                                                 lie=lie, N_sol=N, n_intervals=n_intervals)
        h_sig[i] = v_sig.project_space([i + z_dim for i in range(h_dim)])

    for i in range(len(partition)-2):
        Psi[-2-i] = step_sig(h_sig[-1 - i].inverse(), linear_vf, Psi[-1 - i])

    if speed <= 0.1:
        speedy = 10
    elif speed >= 0.5:
        speedy = 2
    else:
        speedy = int(np.ceil(1/speed))

    for i in range(len(partition)-1):
        subpartition = np.linspace(partition[i], partition[i+1], speedy+1)
        y_local, _ = fixed(x, f, y_0=y_on_partition[i, :], N=N, partition=subpartition)
        propagated_local_errors[i, :] = Psi[i].reshape((y_dim, y_dim)) @ (y_local[:, -1] - y_on_partition[i+1, :])

    # global_error = np.sum(propagated_local_errors, axis=0)
    return y_on_partition, propagated_local_errors


def solve_adaptive_error_representation_fixed_N(x, f, y_0, N, T=1., n=20, atol=1e-04, rtol=1e-02, speed=-1, lie=True):
    """
    Implementation of the Euler method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N: The degree of the Euler method (f needs to be Lip(N+1))
    :param T: Final time
    :param n: Initial number of time intervals
    :param atol: Absolute error tolerance
    :param rtol: Relative error tolerance
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param lie: If True, projects onto the Lie group
    :return: Solution on partition points
    """
    atol = atol/3
    rtol = rtol/3
    partition = np.linspace(0, T, n+1)
    y, loc_err = solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, speed=speed, lie=lie)
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
        y, loc_err = solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, speed=speed, lie=lie)
        global_err = np.sum(loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err / x.norm(y[-1, :])
        print(len(partition)-1, abs_err, rel_err)
    return partition, y


def solve_adaptive_error_representation(x, f, y_0, N_min=1, N_max=3, T=1., n=20, atol=1e-04, rtol=1e-02, speed=-1,
                                        lie=True):
    """
    Implementation of the Euler method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N_min: Minimal degree of the Euler method
    :param N_max: Maximal degree of the Euler method
    :param T: Final time
    :param n: Initial number of time intervals
    :param atol: Absolute error tolerance
    :param rtol: Relative error tolerance
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param lie: If True, projects onto the Lie group
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
        solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, speed=speed, lie=lie)

    partition = np.linspace(0, T, 21)
    N_vec = np.array([i for i in range(N_min, N_max + 1)])
    time_vec = np.zeros(N_max + 1 - N_min)
    for i in range(len(N_vec)):
        print(f'N={N_vec[i]}')
        tic = time.perf_counter()
        y, loc_err = solve_fixed_error_representation(x, f, y_0, N=N_vec[i], partition=partition, speed=speed, lie=lie)
        total_time = time.perf_counter() - tic
        global_err = np.sum(loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err/x.norm(y[-1, :])
        factor = np.fmin(abs_err/atol, rel_err/rtol)
        needed_n = 20 * factor**(p/(N_vec[i]+1-p))
        time_vec[i] = needed_n / 20 * total_time
    N = np.argmin(time_vec) + N_min
    print(f'found N={N}')
    return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N, T=T, n=n, atol=3*atol, rtol=3*rtol, speed=speed,
                                                       lie=lie)


def solve(x, f, y_0, solver, N=1, T=1., partition=None, atol=1e-07, rtol=1e-04, N_sol=1, N_min=1, N_max=5, n=20,
          speed=-1, lie=True):
    """
    Various implementations of the Euler method.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial value
    :param solver: f/a (fixed or adaptive), s/f (simple or full), s/a (simple or adjoined), last letter indicating the
        kind of algorithm used (if only one implementation exists, s for standard)
    :param N: Level of the Euler method
    :param T: Final time
    :param partition: Time partition on which the Euler method is applied
    :param atol: Absolute error tolerance
    :param rtol: Relative error tolerance
    :param N_sol: Level of the solution
    :param N_min: Minimal degree of the Euler method
    :param N_max: Maximal degree of the Euler method
    :param n: Number of time intervals
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param lie: If True, projects onto the Lie group
    :return: Depends on the solver
    """
    if solver == 'fsss':
        return fixed(x, f, y_0, N=N, partition=partition)
    elif solver == 'ffss':
        return fixed_full(x, f, y_0, N=N, partition=partition, lie=lie, N_sol=N_sol)
    elif solver == 'ffsa':
        return solve_fixed_full_alt(x, f, y_0, N=N, partition=partition, N_sol=N_sol)
    elif solver == 'ffas':
        return solve_fixed_adj_full(x, f, y_0, N=N, partition=partition, lie=lie, N_sol=N_sol)
    elif solver == 'ffaa':
        return solve_fixed_adj_full_alt(x, f, y_0, N=N, partition=partition, N_sol=N_sol)
    elif solver == 'fssr':
        return solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, speed=speed, lie=lie)
    elif solver == 'assN':
        return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N, T=T, atol=atol, rtol=rtol, speed=speed,
                                                           n=n, lie=lie)
    elif solver == 'assr':
        return solve_adaptive_error_representation(x, f, y_0, N_min=N_min, N_max=N_max, T=T, n=n, atol=atol, rtol=rtol,
                                                   speed=speed, lie=lie)
