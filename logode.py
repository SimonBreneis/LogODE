import time
import numpy as np
import scipy
from scipy import integrate, special, stats
import roughpath as rp
import vectorfield as vf
import tensoralgebra as ta


def invert_permutation(p):
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


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


def init_g(g=None, g_grad=None, eps=1e-06, second_level=False):
    """
    Initializes the payoff function. If g is None, sets g to be the identity function. Else, if g_grad is None,
    implements g_grad to be the gradient of g using numerical differentiation.
    :param g: The payoff function
    :param g_grad: The gradient of the payoff function
    :param eps: Step size for numerical differentiation
    :param second_level: If True, uses the second-level two-sided difference quotient. If False, uses the first-level
        forward difference quotient
    :return: The functions g and g_grad
    """
    if g is None:
        def g(y_):
            return y_

        def g_grad(y_):
            return np.eye(len(y_))

    if g_grad is None:
        def g_grad(y_):
            gy = g(y_)
            result = np.empty((len(gy), len(y_)))
            for j in range(len(y_)):
                vec = np.zeros(len(y_))
                vec[j] = eps
                if second_level:
                    result[:, j] = (g(y_ + vec) - g(y_ - vec)) / (2 * eps)
                else:
                    result[:, j] = (g(y_ + vec) - gy) / eps
            return result

    return g, g_grad


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
        g_fraction = g**(1. / n_intervals)
        y = np.zeros((n_intervals + 1, y_0.dim()))
        y[0, :] = y_0[1]
        for i in range(n_intervals):
            y[i+1, :] = solve_step_sig(g_fraction, f, y[i, :], atol=atol / n_intervals, rtol=rtol / n_intervals,
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


def solve_fixed(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False, verbose=0):
    """
    Implementation of the Log-ODE method.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial value
    :param N: The degree of the Log-ODE method (f needs to be Lip(N)). May be an integer or an array of length equal
        to the number of intervals
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if isinstance(N, np.ndarray):
        varying_degree = True
    else:
        varying_degree = False
    time_vec = np.empty(len(partition)-1)
    N_vec = N
    y = np.zeros(shape=(len(partition), len(y_0)))
    y[0, :] = y_0
    error_estimate = 0.
    tic = time.perf_counter()
    last_time = tic
    for i in range(1, len(partition)):
        if varying_degree:
            N = N_vec[i-1]
        toc = time.perf_counter()
        if toc - last_time > 10:
            if verbose >= 1:
                print(f'{100 * (i - 1) / (len(partition) - 1):.2f}% complete, estimated '
                      f'{int((toc - tic) / (i - 1) * (len(partition) - i))}s remaining.')
            last_time = toc
        tic_2 = time.perf_counter()
        if compute_bound:
            y[i, :], vf_norm, omega = solve_step(x, f, y[i-1, :], partition[i - 1], partition[i], N, atol, rtol,
                                                 method, compute_bound)
            vf_norm = np.amax(np.array(vf_norm)[:N])
            error_estimate += vf_norm ** (N + 1) * omega ** ((N + 1) / x.p)
        else:
            y[i, :] = solve_step(x, f, y[i-1, :], partition[i - 1], partition[i], N, atol, rtol, method,
                                 compute_bound)
        time_vec[i-1] = time.perf_counter()-tic_2
    if compute_bound:
        if varying_degree:
            N = int(np.amax(N_vec))
        return y, local_log_ode_error_constant(N, x.sig(0., 0., 1).dim(), x.p) * error_estimate, time_vec
    return y, -1, time_vec


def solve_fixed_full(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False, N_sol=None,
                     verbose=0):
    """
    Implementation of the Log-ODE method. Returns the full solution, i.e. the solution as a rough path.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Log-ODE method (f needs to be Lip(N)). May be an integer or an array with length
        equal to the number of intervals
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            if isinstance(N, np.ndarray):
                N_sol = int(np.amin(N))
            else:
                N_sol = N
    if isinstance(y_0, np.ndarray):
        y_0 = ta.sig_first_level_num(y_0, N_sol)
    f_full = f.extend(N_sol)
    y, error, time_vec = solve_fixed(x, f_full, y_0.to_array(), N=N, partition=partition, atol=atol, rtol=rtol,
                                     method=method, compute_bound=compute_bound, verbose=verbose)
    if isinstance(x, rp.RoughPathContinuous) or isinstance(x, rp.RoughPathExact):
        sig_steps = x.sig_steps
    else:
        sig_steps = 2000
    y_list = [ta.array_to_tensor(y[i, :], len(y_0[1])).project_lie() for i in range(y.shape[0])]
    y = rp.rough_path_exact_from_exact_path(times=partition, path=y_list, sig_steps=sig_steps, p=x.p,
                                            var_steps=x.var_steps, norm=x.norm, x_0=y_0)
    return y, error, time_vec


def solve_fixed_full_alt(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
                         N_sol=None, verbose=0):
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
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            N_sol = N
    if isinstance(y_0, ta.Tensor):
        y_0 = y_0[1]
    y, error, time_vec = solve_fixed(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                                     compute_bound=compute_bound, verbose=verbose)

    y = rp.RoughPathDiscrete(times=partition, values=y, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol,
                             x_0=y_0)
    return y, error, time_vec


def solve_fixed_adj_full(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
                         N_sol=None, verbose=0):
    """
    Implementation of the Log-ODE method. Returns the full solution z = (x, y), i.e. the solution as a rough path.
    :param x: Rough path
    :param f: Vector field (non-extended!)
    :param y_0: Initial condition (tensor or vector)
    :param N: The degree of the Log-ODE method (f needs to be Lip(N)). May be a number or a vector of the same
        length as the number of intervals
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param compute_bound: If True, also returns a theoretical error bound
    :param N_sol: Level of the solution. If None, the level of y_0 (if y_0 is a Tensor), or N as the level
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    f_ext = f.adjoin()
    if isinstance(y_0, np.ndarray):
        z_0 = np.zeros(x.dim() + len(y_0))
        z_0[x.dim():] = y_0
    else:
        z_0 = y_0.add_dimensions(front=x.dim(), back=0)
    return solve_fixed_full(x=x, f=f_ext, y_0=z_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                            compute_bound=compute_bound, N_sol=N_sol, verbose=verbose)


def solve_fixed_adj_full_alt(x, f, y_0, N, partition, atol=1e-07, rtol=1e-04, method='RK45', compute_bound=False,
                             N_sol=None, verbose=0):
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
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    if N_sol is None:
        if isinstance(y_0, ta.Tensor):
            N_sol = y_0.n_levels()
        else:
            N_sol = N
    if isinstance(y_0, ta.Tensor):
        y_0 = y_0[1]
    y, error, time_vec = solve_fixed(x, f, y_0, N=N, partition=partition, atol=atol, rtol=rtol, method=method,
                                     compute_bound=compute_bound, verbose=verbose)

    x_dim = x.dim()
    z = np.empty((y.shape[0], x_dim + y.shape[1]))
    for i in range(y.shape[0]):
        z[i, :x_dim] = x.sig(partition[0], partition[i], 1)[1]
    z[:, x_dim:] = y
    z_0 = np.zeros(x_dim + y.shape[1])
    z_0[x_dim:] = y_0
    z = rp.RoughPathDiscrete(times=partition, values=z, p=x.p, var_steps=x.var_steps, norm=x.norm, save_level=N_sol,
                             x_0=z_0)
    return z, error, time_vec


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


def solve_error_tolerance(solver=None, n=16, T=1., atol=1e-04, rtol=1e-02, full_tensor_tolerance=False, verbose=0):
    """
    Solves the RDE given by solver with the desired accuracy. Does NOT use the error representation formula. Instead,
    uniformly refines the grid and compares the solutions until the accuracy target is met.
    :param n: Initial number of intervals
    :param T: Final time
    :param atol: Absolute error tolerance
    :param rtol: Relative error tolerance
    :param solver: Solver which returns the solution of the RDE. Is a function which takes as input the partition on
        which the RDE should be solved
    :param full_tensor_tolerance: Only relevant if solver returns a full (Tensor-valued) solution. If True, compares the
        entire Tensors when determining whether the accuracy target has been met. If False, only compares the first
        level
    :param verbose: Determines the number of intermediary results printed to the console
    :return: The solution on the final partition (which is uniform on [0, T])
    """

    atol, rtol = atol / 2, rtol / 2

    def compare():
        """
        Compares the approximate solution with the solution. If the error criteria are met, returns False, if not,
        returns True (think of it as asking continue?).
        :return: True if we should continue, False if we should stop
        """
        approx_final = approx_solution[-1]
        final = solution[-1]
        difference = approx_final - final
        if isinstance(final, ta.Tensor):
            if full_tensor_tolerance:
                abs_err = np.sum(np.array([difference.norm(N=N) for N in range(1, difference.n_levels() + 1)]))
                final_norm = np.sum(np.array([final.norm(N=N) for N in range(1, final.n_levels() + 1)]))
            else:
                abs_err = ta.l1(difference)
                final_norm = ta.l1(final)
        else:
            abs_err = ta.l1(difference)
            final_norm = ta.l1(final)
        if abs_err < atol and (final_norm < atol or abs_err / final_norm < rtol):
            return False
        return True

    if verbose >= 1:
        print(f'Now computes the solution using {n} intervals.')
    approx_solution = solver(np.linspace(0, T, n + 1))
    n = 2 * n
    if verbose >= 1:
        print(f'Now computes the solution using {n} intervals.')
    solution = solver(np.linspace(0, T, n + 1))

    while compare():
        approx_solution = solution
        n = 2 * n
        if verbose >= 1:
            print(f'Now computes the solution using {n} intervals.')
        solution = solver(np.linspace(0, T, n + 1))

    if verbose >= 1:
        print(f'Computed a sufficiently accurate solution using {n} intervals.')
    return solution


def solve_fixed_error_representation(x, f, y_0, N, partition, g=None, g_grad=None, atol=1e-07, rtol=1e-04,
                                     method='RK45', speed=-1, verbose=0):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N: The degree of the Log-ODE method (f needs to be Lip(N+1))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param g: Payoff function. If None, uses the identity function
    :param g_grad: Gradient of the payoff function. If None, uses numerical differentiation
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points, propagated local errors, absolute values of non-propagated local errors,
        vector of computational times for solving each interval
    """
    g, g_grad = init_g(g=g, g_grad=g_grad, eps=1e-06, second_level=True)
    if not isinstance(N, np.ndarray):
        N = np.array([N]*(len(partition)-1), dtype=int)
    if isinstance(y_0, np.ndarray):
        y_0 = ta.sig_first_level_num(y_0, int(np.amax(N)))
    N_max = int(np.amax(N))
    n = len(partition)
    x_dim = x.dim()
    y_dim = y_0.dim()
    z_dim = x_dim + y_dim
    h_dim = y_dim * y_dim
    result_dim = len(g(y_0[1]))
    linear_vf = vf.matrix_multiplication_vector_field(y_dim, result_dim, norm=f.norm)
    if verbose >= 1:
        print('Now solves for the adjoined full solution z.')
    tic = time.perf_counter()
    z, _, time_vec = solve_fixed_adj_full(x, f, y_0, N, partition, atol=atol, rtol=rtol, method=method,
                                          compute_bound=False, N_sol=N_max, verbose=verbose - 1)
    toc = time.perf_counter()
    reference_time = toc-tic
    y = z.project_space([i + x_dim for i in range(y_dim)])
    h_sig = [0]*(n-2)
    Psi = [0]*(n-1)
    Psi[-1] = g_grad(y.at(partition[-1], 1)[1]).flatten()
    local_errors = np.zeros(n-1)
    propagated_local_errors = np.zeros((n-1, result_dim))
    y_on_partition = np.zeros((n, y_dim))
    y_on_partition[0, :] = y_0[1]

    for i in range(1, n):
        y_on_partition[i, :] = y.at(partition[i], 1)[1]

    if verbose >= 1:
        print('Now solves for the rough path h driving the RDE for the derivative of the flow.')
    if speed < 0:
        n_intervals = 20
        if n > 3:
            tic = time.perf_counter()
            v_init = z.at(partition[1], N_max).add_dimensions(front=0, back=h_dim)
            v_sig = v_init.inverse() * solve_step_sig_full(z.sig(partition[1], partition[2], N_max), f.flow(),
                                                           y_0=v_init, atol=atol, rtol=rtol, method=method, N_sol=N_max,
                                                           n_intervals=n_intervals)
            v_sig.project_space([i + z_dim for i in range(h_dim)])
            toc = time.perf_counter()
            time_h = (toc-tic)*(n-2)/n_intervals
            n_intervals = int(2**np.ceil(np.log2(np.fmax(1, reference_time/time_h*1))))
            print(n_intervals)
    elif speed > 0:
        n_intervals = int(2**np.around(np.log2(1/speed)))
    else:
        n_intervals = 0
    if N_max == 1:
        n_intervals = 0

    tic = time.perf_counter()
    last_time = tic
    for i in range(n - 2):
        toc = time.perf_counter()
        if toc - last_time > 10:
            if verbose >= 2:
                print(f'{100 * i / (n - 2):.2f}% complete, '
                      f'estimated {int((toc - tic) / i * (n - i - 2))}s remaining.')
            last_time = toc
        v_init = z.at(partition[i+1], N[i+1]).add_dimensions(front=0, back=h_dim)
        v_sig = v_init.inverse() * solve_step_sig_full(z.sig(partition[i+1], partition[i+2], N[i+1]), f.flow(),
                                                       y_0=v_init, atol=atol, rtol=rtol, method=method, N_sol=N[i+1],
                                                       n_intervals=n_intervals)
        h_sig[i] = v_sig.project_space([j + z_dim for j in range(h_dim)])

    if verbose >= 1:
        print('Now solves for the derivative of the flow Psi.')
    tic = time.perf_counter()
    last_time = tic
    for i in range(n - 2):
        toc = time.perf_counter()
        if toc - last_time > 10:
            if verbose >= 2:
                print(f'{100 * i / (n - 2):.2f}% complete, estimated {int((toc - tic) / i * (n - i - 2))}s remaining.')
            last_time = toc
        Psi[-2-i] = solve_step_sig(h_sig[-1-i], linear_vf, Psi[-1-i], atol=atol, rtol=rtol, method=method)

    if verbose >= 1:
        print('Now computes the local errors.')
    n_intervals = 8
    tic = time.perf_counter()
    last_time = tic
    for i in range(n - 1):
        toc = time.perf_counter()
        if toc - last_time > 10:
            if verbose >= 2:
                print(f'{100 * i / (n - 1):.2f}% complete, estimated {int((toc - tic) / i * (n - i - 1))}s remaining.')
            last_time = toc
        subpartition = np.linspace(partition[i], partition[i+1], n_intervals+1)
        y_local, _, _ = solve_fixed(x, f, y_0=y_on_partition[i, :], N=N[i], partition=subpartition,
                                    atol=atol/n_intervals, rtol=rtol/n_intervals, method=method, compute_bound=False,
                                    verbose=verbose - 1)
        local_error = y_local[-1, :] - y_on_partition[i+1, :]
        local_errors[i] = ta.l1(local_error)
        propagated_local_errors[i, :] = Psi[i].reshape((result_dim, y_dim)) @ local_error
    # global_error = np.sum(propagated_local_errors, axis=0)
    return y_on_partition, propagated_local_errors, local_errors, time_vec


def solve_adaptive_error_representation_fixed_N(x, f, y_0, N, T=1., n=16, g=None, g_grad=None, atol=1e-04, rtol=1e-02,
                                                method='RK45', speed=-1, verbose=0):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N: The degree of the Log-ODE method (f needs to be Lip(N+1))
    :param T: Final time
    :param n: Initial number of time intervals
    :param g: Payoff function. If None, uses the identity function
    :param g_grad: Gradient of the payoff function. If None, uses numerical differentiation
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points
    """
    atol = atol / 2
    rtol = rtol / 2
    partition = np.linspace(0, T, n + 1)
    if verbose >= 1:
        print(f'Computes the solution using {n} intervals.')
    y, loc_err, _, _ = solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, g=g, g_grad=g_grad,
                                                        atol=atol / n, rtol=rtol / n, method=method, speed=speed,
                                                        verbose=verbose - 1)
    global_err = np.sum(loc_err, axis=0)
    abs_err = ta.l1(global_err)
    rel_err = abs_err/x.norm(y[-1, :])
    if verbose >= 1:
        print(f'Using {len(partition) - 1} intervals, we achieve an absolute error of {abs_err} and a relative error '
              f'of {rel_err}.')
    while abs_err > atol and rel_err > rtol:
        abs_loc_err = np.array([ta.l1(loc_err[i, :]) for i in range(len(partition)-1)])
        med_abs_loc_err = np.median(abs_loc_err)
        relevant_ind = np.argwhere(abs_loc_err >= med_abs_loc_err).flatten()
        additional_times = np.zeros(len(relevant_ind))
        for i in range(len(relevant_ind)):
            additional_times[i] = (partition[relevant_ind[i]] + partition[relevant_ind[i]+1])/2
        joint_partition = np.zeros(len(partition) + len(additional_times))
        joint_partition[:len(partition)] = partition
        joint_partition[len(partition):] = additional_times
        partition = np.sort(joint_partition)
        if verbose >= 1:
            print(f'Computes the solution using {len(partition) - 1} intervals.')
        y, loc_err, _, _ = solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, g=g, g_grad=g_grad,
                                                            atol=atol / len(partition), rtol=rtol / len(partition),
                                                            method=method, speed=speed, verbose=verbose - 1)
        global_err = np.sum(loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err / x.norm(y[-1, :])
        if verbose >= 1:
            print(f'Using {len(partition) - 1} intervals, we achieve an absolute error of {abs_err} and a relative '
                  f'error of {rel_err}.')
    return partition, y, loc_err


def update_grid(N, part, incr_deg_ind, div_int_ind):
    """
    Internal function for updating the partitions and degrees when using the adaptive algorithm with the error
    representation formula.
    :param N: Vector of degrees corresponding to the intervals
    :param part: Partition
    :param incr_deg_ind: Indices of the partition where the degree should be increased
    :param div_int_ind: Indices of the partition where the interval should be subdivided
    :return: The new vector of degrees, the new partition, the indices of the new partition where the degree was
        increased, and the indices of the new partition where the interval was subdivided
    """
    n = len(part) - 1
    additional_times = (part[div_int_ind] + part[div_int_ind + 1]) / 2
    new_partition = np.zeros(n + 1 + len(additional_times))
    new_N = np.zeros(n + len(additional_times), dtype=int)
    new_partition[:n + 1] = part
    new_N[:n] = N
    new_N[incr_deg_ind] += 1
    new_partition[n + 1:] = additional_times
    new_N[n:] = N[div_int_ind]
    permutation = np.argsort(new_partition)
    new_partition = new_partition[permutation]
    last_ind = permutation[-1]
    reduced_permutation = permutation[:-1]
    overshoot_ind = np.where(reduced_permutation > last_ind)
    reduced_permutation[overshoot_ind] = reduced_permutation[overshoot_ind] - 1
    new_N = new_N[reduced_permutation]
    inverted_permutation = invert_permutation(reduced_permutation)
    new_increase_degree_ind = inverted_permutation[incr_deg_ind]
    new_divide_interval_ind = inverted_permutation[div_int_ind]
    return new_N, new_partition, new_increase_degree_ind, new_divide_interval_ind


def solve_fully_adaptive_error_representation(x, f, y_0, N_min=1, N_max=3, T=1., n=16, g=None, g_grad=None, atol=1e-04,
                                              rtol=1e-02, method='RK45', speed=-1, verbose=0):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N_min: Minimal degree of the Log-ODE method
    :param N_max: Maximal degree of the Log-ODE method
    :param T: Final time
    :param n: Initial number of time intervals
    :param g: Payoff function. If None, uses the identity function
    :param g_grad: Gradient of the payoff function. If None, uses numerical differentiation
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points
    """
    p = x.p
    if N_min + 1 <= p:
        N_min = int(np.ceil(p - 0.999))
    if N_max < N_min:
        print(f'The parameter N_min={N_min} is larger than the parameter N_max={N_max}. Carries out the computation '
              f'with fixed degree N={N_min}.')
        return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N_min, T=T, n=n, g=g, g_grad=g_grad, atol=atol,
                                                           rtol=rtol, method=method, speed=speed, verbose=verbose - 1)
    atol = atol / 3
    rtol = rtol / 3
    part = np.linspace(0, T, 4)
    if verbose >= 1:
        print('Carries out some precomputations.')
    for N in range(N_min, N_max+1):
        # just running the solver once for each possible N because there might be initial computational costs
        # associated with running some degree N for the first time
        if verbose >= 2:
            print(f'Carries out precomputations for degree N={N}.')
        solve_fixed_error_representation(x, f, y_0, N=N, partition=part, g=g, g_grad=g_grad, atol=1000*atol,
                                         rtol=1000*rtol, method=method, speed=speed, verbose=verbose - 2)

    tic = time.perf_counter()
    part = np.linspace(0, T, n + 1)  # starting partition
    N = np.ones(n, dtype=int) * N_min  # for each interval, gives the current degree

    int_err_est = np.zeros((N_max-N_min, 1024))  # for each degree N (up to the last), is a vector of the
    # previously observed relative changes of the local error when going from one interval to two
    n_int_err_est = np.zeros(N_max-N_min, dtype=int)  # for each degree N (up to the last), the number of
    # previously observed relative changes of the local error when going from one interval to two
    deg_time_est = np.zeros((N_max-N_min, 1024))  # for each degree N (up to the last), is a vector
    # of the previously observed relative changes of the computational time when increasing the degree by one
    n_deg_time_est = np.zeros(N_max-N_min, dtype=int)  # for each degree N (up to the last), the number
    # of previously observed relative changes of the computational time when increasing the degree by one
    deg_err_est = np.zeros((N_max-N_min, 1024))  # for each degree N (up to the last), is a vector
    # of the previously observed relative changes of the local error when increasing the degree by one. more precisely,
    # always computes the fraction e(N+1) / e(N)**((N+2)/N+1))
    n_deg_err_est = np.zeros(N_max-N_min, dtype=int)  # for each degree N (up to the last), the
    # number of previously observed relative changes of the local error when increasing the degree by one
    int_err_estrs = 2. ** (1 - (np.arange(N_min, N_max)+1)/p)  # for each degree N (up to the last), is an
    # estimator for the relative change of the local error when going from one interval to two
    deg_time_estrs = 1. * np.arange(N_min, N_max) * x.dim()**np.arange(N_min, N_max)  # for each degree N (up to
    # the last), is an estimator for the relative change of the computational time when increasing the degree by one
    deg_err_estrs = -np.ones(N_max-N_min)  # for each degree N (up to the last), is an estimator
    # for the relative change of the local error when increasing the degree by one

    def solve_(N_, part_, y_0_=y_0):
        if verbose >= 1:
            message = f'Solves the RDE using {len(part_) - 1} intervals, of which'
            for i in range(N_min, N_max + 1):
                if i == N_max:
                    message += ' and'
                message += f' {np.sum(N_ == i)} intervals have degree {i}'
                if i == N_max:
                    message += '.'
                else:
                    message += ','
            print(message)
        return solve_fixed_error_representation(x=x, f=f, y_0=y_0_, N=N_, partition=part_, g=g, g_grad=g_grad,
                                                atol=atol / (len(part_) - 1), rtol=rtol / (len(part_) - 1),
                                                method=method, speed=speed)

    def add_deg_est(N_ind, part_ind):
        s = part[part_ind]
        t = part[part_ind + 1]
        y_s = y[part_ind, :]
        y_, _, loc_err_, times_ = solve_(y_0_=y_s, N_=N_min+N_ind+1, part_=np.array([s, t]))
        deg_time_est[N_ind, n_deg_time_est[N_ind]] = times_[0] / times[part_ind]
        n_deg_time_est[N_ind] += 1
        deg_err_est[N_ind, n_deg_err_est[N_ind]] = \
            loc_err_[0] / loc_err[part_ind] ** ((N_min + N_ind + 2) / (N_min + N_ind + 1))
        n_deg_err_est[N_ind] += 1

    def update_estrs():
        for k in range(N_max-N_min):
            if n_deg_err_est[k] > 0:
                deg_err_estrs[k] = np.median(deg_err_est[k, :n_deg_err_est[k]])
            if n_deg_time_est[k] > 0:
                deg_time_estrs[k] = np.median(deg_time_est[k, :n_deg_time_est[k]])
        for k in range(N_max-N_min):
            n_est = n_int_err_est[k]
            if n_est >= 2:
                possible_estimator = np.median(int_err_est[k, :n_est])
                log_estimator = np.log(possible_estimator)
                std_log_estimator = np.std(int_err_est[k, :n_est])
                lhs = log_estimator + 1.65 * std_log_estimator / np.sqrt(n_est)
                rhs = -((N_min+k+1)/p - 1) * np.log(2) / np.log(n_est)
                if lhs < rhs:
                    int_err_estrs[k] = possible_estimator

    def enlarge_array(a):
        temp = np.zeros((a.shape[0], 2 * a.shape[1]))
        temp[:, :a.shape[1]] = a
        return temp

    y, prop_loc_err, loc_err, times = solve_(N_=N, part_=part)
    global_err = np.sum(prop_loc_err, axis=0)
    abs_err = ta.l1(global_err)
    rel_err = abs_err/x.norm(y[-1, :]) if x.norm(y[-1, :]) > atol else rtol/2
    if verbose >= 1:
        print(f'The absolute error is {abs_err} and the relative error is {rel_err}.')
    if abs_err < atol and rel_err < rtol:
        if verbose >= 1:
            print(f'The algorithm terminates. The runtime was {time.perf_counter() - tic} seconds.')
        return part, y, prop_loc_err, N

    while abs_err > atol or rel_err > rtol:
        loc_N_max = np.amax(N)
        if not loc_N_max == N_max and n_deg_err_est[loc_N_max-N_min] < 100:
            # we may wish to increase the degree but do not have a good error estimator yet
            critical_indices = np.nonzero(N == loc_N_max)[0]  # indices of the partition for which we do not have
            # an estimator
            if len(critical_indices) == 1:
                add_deg_est(N_ind=loc_N_max-N_min, part_ind=critical_indices[0])
            else:
                i = np.random.randint(0, len(critical_indices))
                j = np.random.randint(0, len(critical_indices))
                if j == i:
                    if j+1 == len(critical_indices):
                        j = j-1
                    else:
                        j = j+1
                add_deg_est(N_ind=loc_N_max-N_min, part_ind=critical_indices[i])
                add_deg_est(N_ind=loc_N_max-N_min, part_ind=critical_indices[j])
        update_estrs()
        '''
        print(f'degree error estimators: {degree_error_estimators}')
        print(degree_error_estimates[:, :20])
        print(f'degree time estimators: {degree_time_estimators}')
        print(degree_time_estimates[:, :20])
        print(f'interval error estimators: {interval_error_estimators}')
        print(interval_error_estimates[:, :20])
        '''

        n = len(part)-1
        abs_prop_loc_err = ta.l1(prop_loc_err, axis=1)
        relevant_ind = np.argwhere(np.logical_or(abs_prop_loc_err >= atol/n,
                                                 (abs_prop_loc_err / ta.l1(y[-1, :]) if ta.l1(y[-1, :]) > atol
                                                  else rtol / (2*n)) >= rtol/n)).flatten()
        interesting_ind = relevant_ind[N[relevant_ind] != N_max]
        e_N = deg_err_estrs[N[interesting_ind]-N_min] * loc_err[interesting_ind]**(1/N[interesting_ind])
        e_I = int_err_estrs[N[interesting_ind]-N_min]
        t_N = deg_time_estrs[N[interesting_ind]-N_min]
        incr_deg = 2**(np.log(e_N) / np.log(e_I)) > t_N
        incr_deg_ind = interesting_ind[incr_deg]
        div_int_ind = interesting_ind[np.invert(incr_deg)]
        div_int_ind = np.concatenate((div_int_ind, relevant_ind[N[relevant_ind] == N_max]))
        new_N, new_part, new_incr_deg_ind, new_div_int_ind = update_grid(N=N, part=part, incr_deg_ind=incr_deg_ind,
                                                                         div_int_ind=div_int_ind)

        new_y, new_prop_loc_err, new_loc_err, new_times = solve_(N_=new_N, part_=new_part)

        new_int_err_est = (new_loc_err[new_div_int_ind] + new_loc_err[new_div_int_ind+1]) / loc_err[div_int_ind]
        for i in range(N_max-N_min):
            loc_new_int_err_est = new_int_err_est[N[div_int_ind] == N_min+i]
            if n_int_err_est[i] + len(loc_new_int_err_est) > len(int_err_est[i, :]):
                int_err_est = enlarge_array(int_err_est)
            int_err_est[i, n_int_err_est[i]:n_int_err_est[i] + len(loc_new_int_err_est)] = loc_new_int_err_est
            n_int_err_est[i] = n_int_err_est[i] + len(loc_new_int_err_est)

        new_deg_time_est = new_times[new_incr_deg_ind] / times[incr_deg_ind]
        for i in range(N_max-N_min):
            loc_new_deg_time_est = new_deg_time_est[N[incr_deg_ind] == N_min+i]
            if n_deg_time_est[i] + len(loc_new_deg_time_est) > len(deg_time_est[i, :]):
                deg_time_est = enlarge_array(deg_time_est)
            deg_time_est[i, n_deg_err_est[i]:n_deg_time_est[i] + len(loc_new_deg_time_est)] = loc_new_deg_time_est
            n_deg_time_est[i] = n_deg_time_est[i] + len(loc_new_deg_time_est)

        new_deg_err_est = new_loc_err[new_incr_deg_ind] / \
            loc_err[incr_deg_ind] ** ((N[incr_deg_ind] + 2) / (N[incr_deg_ind] + 1))
        for i in range(N_max-N_min):
            loc_new_deg_err_est = new_deg_err_est[N[incr_deg_ind] == N_min+i]
            if n_deg_err_est[i] + len(loc_new_deg_err_est) > len(deg_err_est[i, :]):
                deg_err_est = enlarge_array(deg_err_est)
            deg_err_est[i, n_deg_err_est[i]:n_deg_err_est[i] + len(loc_new_deg_err_est)] = loc_new_deg_err_est
            n_deg_err_est[i] = n_deg_err_est[i] + len(loc_new_deg_err_est)

        update_estrs()
        part = new_part
        N = new_N
        y = new_y
        loc_err = new_loc_err
        prop_loc_err = new_prop_loc_err
        times = new_times

        global_err = np.sum(prop_loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err / x.norm(y[-1, :]) if x.norm(y[-1, :]) > atol else rtol / 2
        if verbose >= 1:
            print(f'The absolute error is {abs_err}, where we have a tolerance of {atol}.')
            print(f'The relative error is {rel_err}, where we have a tolerance of {rtol}.')

    if verbose >= 1:
        print(f'The algorithm terminates. The total time was {time.perf_counter() - tic} seconds.')
    return part, y, prop_loc_err, N


def solve_fully_adaptive_error_representation_slow(x, f, y_0, N_min=1, N_max=3, T=1., n=16, g=None, g_grad=None,
                                                   atol=1e-04, rtol=1e-02, method='RK45', speed=-1, verbose=0):
    """
    Implementation of the Log-ODE method. Uses the a-posteriori error representation.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N_min: Minimal degree of the Log-ODE method
    :param N_max: Maximal degree of the Log-ODE method
    :param T: Final time
    :param n: Initial number of time intervals
    :param g: Payoff function. If None, uses the identity function
    :param g_grad: Gradient of the payoff function. If None, uses numerical differentiation
    :param atol: Total (over the entire time interval) absolute error tolerance of the ODE solver
    :param rtol: Total (over the entire time interval) relative error tolerance of the ODE solver
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Solution on partition points
    """
    p = x.p
    if N_min + 1 <= p:
        N_min = int(np.ceil(p - 0.999))
    if N_max < N_min:
        print(f'The parameter N_min={N_min} is larger than the parameter N_max={N_max}. Carries out the computation '
              f'with fixed degree N={N_min}.')
        return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N_min, T=T, n=n, g=g, g_grad=g_grad, atol=atol,
                                                           rtol=rtol, method=method, speed=speed)
    atol = atol / 3
    rtol = rtol / 3
    part = np.linspace(0, T, 4)
    if verbose >= 1:
        print('Carries out some precomputations.')
    for N in range(N_min, N_max+1):
        # just running the solver once for each possible N because there might be initial computational costs
        # associated with running some degree N for the first time
        if verbose >= 2:
            print(f'Carries out precomputations for degree N={N}.')
        solve_fixed_error_representation(x, f, y_0, N=N, partition=part, g=g, g_grad=g_grad, atol=1000*atol,
                                         rtol=1000*rtol, method=method, speed=speed)
    part = np.linspace(0, T, n + 1)  # starting partition
    N = np.ones(n, dtype=int) * N_min  # for each interval, gives the current degree

    def solve_(N_, part_):
        if verbose >= 1:
            message = f'Solves the RDE using {len(part_) - 1} intervals, of which'
            for i in range(N_min, N_max + 1):
                if i == N_max:
                    message += ' and'
                message += f' {np.sum(N_ == i)} intervals have degree {i}'
                if i == N_max:
                    message += '.'
                else:
                    message += ','
            print(message)
        return solve_fixed_error_representation(x=x, f=f, y_0=y_0, N=N_, partition=part_, g=g, g_grad=g_grad,
                                                atol=atol / (len(part) - 1), rtol=rtol / (len(part) - 1),
                                                method=method, speed=speed, verbose=verbose - 1)

    tic = time.perf_counter()
    y, prop_loc_err, loc_err, times = solve_(N_=N, part_=part)
    global_err = np.sum(prop_loc_err, axis=0)
    abs_err = ta.l1(global_err)
    rel_err = abs_err/x.norm(y[-1, :]) if x.norm(y[-1, :]) > atol else rtol/2
    print(f'The absolute error is {abs_err} and the relative error is {rel_err}.')
    if abs_err < atol and rel_err < rtol:
        print(f'The algorithm terminates. The total runtime was {time.perf_counter() - tic} seconds.')
        return part, y, prop_loc_err, N

    while abs_err > atol or rel_err > rtol:
        n = len(part) - 1
        abs_prop_loc_err = ta.l1(prop_loc_err, axis=1)
        relevant_ind = np.argwhere(np.logical_or(abs_prop_loc_err >= atol / n,
                                                 (abs_prop_loc_err / ta.l1(y[-1, :]) if ta.l1(y[-1, :]) > atol
                                                  else rtol / (2 * n)) >= rtol / n)).flatten()
        interesting_ind = relevant_ind[N[relevant_ind] != N_max]
        N_1, part_1, _, interesting_ind_1 = update_grid(N=N, part=part, incr_deg_ind=np.argwhere(np.array([False])),
                                                        div_int_ind=interesting_ind)

        y_1, prop_loc_err_1, loc_err_1, times_1 = solve_(N_=N_1, part_=part_1)

        global_err_1 = np.sum(prop_loc_err_1, axis=0)
        abs_err_1 = ta.l1(global_err_1)
        rel_err_1 = abs_err_1 / x.norm(y_1[-1, :]) if x.norm(y_1[-1, :]) > atol else rtol / 2

        if abs_err_1 < atol and rel_err_1 < rtol:
            if verbose >= 1:
                print('Finished prematurely when subdividing intervals.')
                print(f'The absolute error is {abs_err_1}, where we have a tolerance of {atol}.')
                print(f'The relative error is {rel_err_1}, where we have a tolerance of {rtol}.')
                print(f'The total runtime is {time.perf_counter() - tic} seconds.')
            return part_1, y_1, prop_loc_err_1, N_1

        N_2 = np.copy(N)
        N_2[interesting_ind] = N[interesting_ind] + 1
        y_2, prop_loc_err_2, loc_err_2, times_2 = solve_(N_=N_2, part_=part)

        global_err_2 = np.sum(prop_loc_err_2, axis=0)
        abs_err_2 = ta.l1(global_err_2)
        rel_err_2 = abs_err_2 / x.norm(y_2[-1, :]) if x.norm(y_2[-1, :]) > atol else rtol / 2

        if abs_err_2 < atol and rel_err_2 < rtol:
            if verbose >= 1:
                print('Finished prematurely when increasing the degree.')
                print(f'The absolute error is {abs_err_2}, where we have a tolerance of {atol}.')
                print(f'The relative error is {rel_err_2}, where we have a tolerance of {rtol}.')
                print(f'The total runtime is {time.perf_counter() - tic} seconds.')
            return part, y_2, prop_loc_err_2, N_2

        int_err_est = (loc_err_1[interesting_ind_1] + loc_err_1[interesting_ind_1+1]) / loc_err[interesting_ind]
        deg_time_est = times_2[interesting_ind] / times[interesting_ind]
        deg_err_est = loc_err_2[interesting_ind] / loc_err[interesting_ind]

        incr_deg = 2 ** (np.log(deg_err_est) / np.log(int_err_est)) > deg_time_est
        incr_deg_ind = interesting_ind[incr_deg]
        div_int_ind = interesting_ind[np.invert(incr_deg)]
        div_int_ind = np.concatenate((div_int_ind, relevant_ind[N[relevant_ind] == N_max]))
        new_N, new_part, _, _ = update_grid(N=N, part=part, incr_deg_ind=incr_deg_ind, div_int_ind=div_int_ind)

        new_y, new_prop_loc_err, new_loc_err, new_times = solve_(N_=new_N, part_=new_part)

        part = new_part
        N = new_N
        y = new_y
        loc_err = new_loc_err
        prop_loc_err = new_prop_loc_err
        times = new_times

        global_err = np.sum(prop_loc_err, axis=0)
        abs_err = ta.l1(global_err)
        rel_err = abs_err / x.norm(y[-1, :]) if x.norm(y[-1, :]) > atol else rtol / 2
        if verbose >= 1:
            print(f'The absolute error is {abs_err}, where we have a tolerance of {atol}.')
            print(f'The relative error is {rel_err}, where we have a tolerance of {rtol}.')

    if verbose >= 1:
        print(f'The algorithm terminates. The total runtime was {time.perf_counter() - tic} seconds.')
    return part, y, prop_loc_err, N


def solve(x, f, y_0, solver, N=1, T=1., partition=None, g=None, g_grad=None, atol=1e-07, rtol=1e-04, method='RK45',
          compute_bound=False, N_sol=1, N_min=1, N_max=5, n=32, speed=-1):
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
    :param g: Payoff function. If None, uses the identity function
    :param g_grad: Gradient of the payoff function. If None, uses numerical differentiation
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
        return solve_fixed_error_representation(x, f, y_0, N=N, partition=partition, g=g, g_grad=g_grad, atol=atol,
                                                rtol=rtol, method=method, speed=speed)
    elif solver == 'assN':
        return solve_adaptive_error_representation_fixed_N(x, f, y_0, N=N, T=T, g=g, g_grad=g_grad, atol=atol,
                                                           rtol=rtol, method=method, speed=speed, n=n)
