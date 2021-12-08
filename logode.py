import numpy as np
import scipy
from scipy import integrate, special
import roughpath as rp


def l1(x):
    return np.sum(np.abs(x))


def vfd(f_vec, y, dx, h=1e-05):
    """
    Computes the nth derivative of a vector field.
    :param f_vec: The vector field
    :param y: Point at which the derivative is calculated
    :param dx: Direction to which the vector field is applied, an n-tensor
    :param h: Step size for the numerical differentiation
    :return: An approximation of the n-th derivative
    """
    N = len(dx.shape)
    if N <= len(f_vec):
        return f_vec[N - 1](y, dx)
    x_dim = np.shape(dx)[0]
    result = 0
    for i in range(x_dim):
        vec = np.zeros(x_dim)
        vec[i] = 1.
        direction = f_vec[0](y, vec)
        result += (vfd(f_vec, y + h / 2 * direction, dx[..., i], h)
                   - vfd(f_vec, y - h / 2 * direction, dx[..., i], h)) / h
    return result


def vector_field(f_vec, ls, h=1e-05, norm=None, norm_estimate=None):
    """
    Computes the vector field used in the Log-ODE method.
    :param f_vec: List, first element is the vector field. Further elements may be the derivatives of the vector field,
        if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives are
        computed numerically
        If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes as
        input the position vector y and the (i+1)-tensor dx^(i+1)
        If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function that
        takes as input only the position vector y
    :param ls: The log-signature of the driving path up to level deg
    :param h: Step size for numerical differentiation (if the derivatives of the vector field are not given)
    :param norm: If a norm is specified, computes a local (directional) estimate of the Lipschitz norm of f
    :param norm_estimate: This needs to be specified if norm is not None. In that case, it needs to be a list with one
        floating point number (e.g. [3.14], this is to make the number mutable). This should contain the previous norm
        estimate, and is updated accordingly
    :return: Solution on partition points
    """
    deg = len(ls)

    if norm is None:
        return lambda y: np.sum(np.array([vfd(f_vec, y, ls[i], h) for i in range(deg)]), axis=0)

    def vf_norm(y):
        summands = np.array([vfd(f_vec, y, ls[i], h) for i in range(deg)])
        vf = np.sum(summands, axis=0)
        for i in range(deg):
            norm_ls_i = norm(ls[i])
            if norm_ls_i > 0:
                norm_estimate[0] = np.fmax(norm_estimate[0], (norm(summands[i]) / norm_ls_i) ** (1. / (i + 1)))
        return vf

    return vf_norm


def log_ode_step(x, f_vec, y_0, N, s, t, method='RK45', atol=1e-09, rtol=1e-05, h=1e-05, norm=None):
    """
    Implementation of the Log-ODE method.
    :param x: Driving rough path
    :param f_vec: Vector field, containing also the derivatives
    :param y_0: Initial value
    :param N: The degree of the Log-ODE method (f needs to be Lip(N))
    :param s: Initial interval point
    :param t: Final interval point
    :param method: A method for solving initial value problems implemented in the scipy.integrate library
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param h: Step size for numerical differentiation (if the derivatives of the vector field are not given)
    :param norm: If a norm is specified, computes a local (directional) estimate of the Lipschitz norm of f. Also
        computes an estimate of the p-variation of the underlying
        path x. Note that this is an estimate of the p-variation only for x itself, not for its higher-order
        signature terms. Also, it is not an actual estimate of the p-variation, but rather an estimate of the sum
        sum_{i in partition} |x|_{p, [t_i, t_{i+1}]}^deg,
        as this is more relevant to the problem.
    :return: Solution on partition points
    """
    ls = x.log_incr(s, t, N)[1:]
    for i in range(N):
        ls[i] = np.transpose(ls[i], [i + 1 - j for j in range(1, i + 2)])

    if norm is None:
        vf = lambda t, z: vector_field(f_vec, ls, h)(z)
        return integrate.solve_ivp(vf, (0, 1), y_0, method=method, atol=atol, rtol=rtol).y[:, -1], [0.], 0.

    norm_estimate = [0.]
    vf = lambda t, z: vector_field(f_vec, ls, h, norm, norm_estimate)(z)
    y = integrate.solve_ivp(vf, (0, 1), y_0, method=method, atol=atol, rtol=rtol).y[:, -1]
    omega = x.omega(s, t)
    return y, norm_estimate[0], omega


def log_ode(x, f_vec, y_0, N, partition, method='RK45', atol=1e-09, rtol=1e-05, h=1e-05, norm=None, p=1):
    """
    Implementation of the Log-ODE method.
    :param x: Driving rough path
    :param f_vec: Vector field, containing also the derivatives
    :param y_0: Initial value
    :param N: The degree of the Log-ODE method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param method: A method for solving initial value problems implemented in the scipy.integrate library
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param h: Step size for numerical differentiation (if the derivatives of the vector field are not given)
    :param norm: If a norm is specified, computes a local (directional) estimate of the Lipschitz norm of f. Also
        computes an estimate of the p-variation of the underlying
        path x. Note that this is an estimate of the p-variation only for x itself, not for its higher-order
        signature terms. Also, it is not an actual estimate of the p-variation, but rather an estimate of the sum
        sum_{i in partition} |x|_{p, [t_i, t_{i+1}]}^deg,
        as this is more relevant to the problem.
    :param p: The exponent for which the p-variation of x should be computed.
        This method is only reasonable if p < N
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    y = np.zeros(shape=(len(y_0), len(partition)))
    y[:, 0] = y_0

    error_estimate = 0.
    for i in range(1, len(partition)):
        y[:, i], vf_norm, omega = log_ode_step(x, f_vec, y[:, i - 1], N, partition[i - 1], partition[i], method,
                                               atol, rtol, h, norm)
        error_estimate += vf_norm ** (N + 1) * omega ** ((N + 1) / p)
    return y, local_log_ode_error_constant(p, N, len(x.incr(partition[0], partition[1], 1)[1])) * error_estimate


def local_log_ode_error_constant(p, N, d):
    """
    Returns the constant in the error bound for a single step of the Log-ODE method.
    :param p: Parameter measuring the roughness of the signal. Has to be in {1, 2, 3}.
    :param N: Degree of the method
    :param d: Dimension of the signal
    :return: The constant
    """
    if p == 1:
        return 0.34 * (7 / 3.) ** (N + 1)
    if p == 2:
        return 25 * d / scipy.special.gamma((N + 3) / 2.) + 0.081 * (7 / 3) ** (N + 1)
    return 1000 * d ** 3 / scipy.special.gamma((N + 4) / 3.) + 0.038 * (7 / 3) ** (N + 1)


def find_next_interval(x, s, t, lower_omega, upper_omega):
    """
    Finds an interval of the form [s,u] with s <= u <= t such that lower_omega <= omega(s, u) <= upper_omega.
    :param x: The path
    :param s: Initial point of total interval
    :param t: Final point of total interval
    :param lower_omega: Lower bound of the control function on the new interval
    :param upper_omega: Upper bound of the control function on the new interval
    :return: The partition point u.
    """
    total_omega = x.omega(s, t)
    if total_omega <= upper_omega:
        return t
    u_current = s + (t - s) * (lower_omega + upper_omega) / (2 * total_omega)
    u_left = s
    u_right = t

    current_omega = x.omega(s, u_current)

    while not lower_omega <= current_omega <= upper_omega and u_right - u_left > (t - s) * 10 ** (-10):
        if current_omega > lower_omega:
            u_right = u_current
        else:
            u_left = u_current

        u_current = (u_left + u_right) / 2
        current_omega = x.omega(s, u_current)

    return u_current


def find_partition(x, s, t, total_omega, n, q=2.):
    """
    Finds a partition of the interval [0,T] such that omega(0,T)/(qn) <= omega(s,t) <= q*omega(0,T)/n.
    :param x: The path
    :param s: Initial time
    :param t: Final time
    :param total_omega: Estimate for the total control function of x on [0,T]
    :param n: Approximate number of intervals into which [0,T] should be split
    :param q: Tolerance in finding the "perfect" partition.
    :return: The partition
    """
    q = max(q, 1.1)
    p = x.p
    partition = [s]
    lower_omega = total_omega / (q * n) ** (1 / p)
    upper_omega = total_omega * (q / n) ** (1 / p)
    print(f"Total omega: {total_omega}")
    print(f"Lower omega: {lower_omega}")
    print(f"Upper omega: {upper_omega}")
    while not partition[-1] == t:
        next_point = find_next_interval(x, partition[-1], t, lower_omega, upper_omega)
        partition.append(next_point)
    return partition


def insert_list(master, insertion, index):
    """
    Inserts the list insertion to the list master starting at index.
    :param master:
    :param insertion:
    :param index:
    :return:
    """
    for i in range(len(insertion)):
        master.insert(index + i, insertion[i])


def log_ode_user(x, f_vec, y_0, T, n_steps=100, method='RK45', eps=1e-03, atol=1e-09, rtol=1e-05, h=1e-05, norm=None,
                 p=1, var_steps=15):
    """
    Implementation of the Log-ODE method.
    :param x: Driving path given as a function
    :param f_vec: Vector field, containing also the derivatives
    :param y_0: Initial value
    :param T: Final time
    :param n_steps: Number of (equidistant) steps used in the approximation of the signature of x
    :param method: A method for solving initial value problems implemented in the scipy.integrate library
    :param eps: Total error tolerance
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param h: Step size for numerical differentiation (if the derivatives of the vector field are not given)
    :param norm: If a norm is specified, computes a local (directional) estimate of the Lipschitz norm of f. Also
        computes an estimate of the p-variation of the underlying
        path x. Note that this is an estimate of the p-variation only for x itself, not for its higher-order
        signature terms. Also, it is not an actual estimate of the p-variation, but rather an estimate of the sum
        sum_{i in partition} |x|_{p, [t_i, t_{i+1}]}^deg,
        as this is more relevant to the problem.
    :param p: The exponent for which the p-variation of x should be computed. Must be in {1,2,3}.
    :param var_steps: Number of steps used when computing the p-variation of x. Lower var_steps leads to a speed-up, but
        may be more inaccurate.
    :return: Solution on partition points
    """
    p = int(p)
    d = len(x(0.))
    norm_estimates = np.zeros(10)
    x = rp.RoughPathContinuous(x, n_steps=max(n_steps, 100), p=p, var_steps=var_steps, norm=norm)
    _, norm_estimates[0], omega_estimate = log_ode_step(x=x, f_vec=f_vec, y_0=y_0, N=p, s=0, t=T, method=method,
                                                        atol=100 * atol, rtol=100 * rtol, h=h, norm=norm)
    _, norm_estimates[1], _ = log_ode_step(x=x, f_vec=f_vec, y_0=y_0, N=p + 1, s=0, t=T, method=method,
                                           atol=100 * atol, rtol=100 * rtol, h=h, norm=norm)
    _, norm_estimates[2], _ = log_ode_step(x=x, f_vec=f_vec, y_0=y_0, N=p + 2, s=0, t=T, method=method,
                                           atol=100 * atol, rtol=100 * rtol, h=h, norm=norm)
    x.n_steps = n_steps
    norm_incr = max(norm_estimates[2] - norm_estimates[1], norm_estimates[1] - norm_estimates[0])
    norm_estimates[3:] = norm_estimates[2] + norm_incr * np.arange(1, 8)
    print(f"Norm estimates: {norm_estimates}")
    print(f"Error constants: {np.array([local_log_ode_error_constant(p, N, d) for N in range(p, p + 10)])}")
    print(f"Omega: {omega_estimate}")
    number_intervals = np.array([(local_log_ode_error_constant(p, N, d) * norm_estimates[N - p] ** (
            N + 1) * omega_estimate ** ((N + 1.) / p) / eps) ** (p / (N - p + 1)) for N in range(p, p + 10)])
    print(f"Number of intervals: {number_intervals}")
    complexity = np.array([d ** N for N in range(p, p + 10)]) * number_intervals
    N = np.argmin(complexity) + p
    print(f"N = {N}")
    number_intervals = (number_intervals[N - p] / 10) ** (2. / (1 + p))
    print("Let us find a partition!")
    partition = find_partition(x=x, s=0, t=T, total_omega=omega_estimate, n=number_intervals)
    print("We found a partition!")
    local_Ns = [N] * (len(partition) - 1)
    max_local_error = [eps / number_intervals] * (len(partition) - 1)
    y = [y_0]

    i = 0
    while i < len(partition) - 1:
        print(f"At index {i + 1} of {len(partition) - 1}")
        local_N = local_Ns[i]
        y_next, norm_est, omega_est = log_ode_step(x=x, f_vec=f_vec, y_0=y[i], N=local_N, s=partition[i],
                                                   t=partition[i + 1], method=method, atol=atol, rtol=rtol, h=h,
                                                   norm=norm)
        print(f"Norm estimate: {norm_est}, Omega estimate: {omega_est}")
        error_est = local_log_ode_error_constant(p, local_N, d) * norm_est ** (local_N + 1) * omega_est ** (
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
                error_est_N = local_log_ode_error_constant(p, new_local_N + 1, d) * norm_est ** (
                        new_local_N + 2) * omega_est ** ((new_local_N + 2.) / p)
                error_est_part = error_est / d ** ((new_local_N + 1) / p) * d
                if error_est_N < error_est_part:
                    new_local_N += 1
                else:
                    subpartition *= 4
                new_error_est = subpartition * local_log_ode_error_constant(p, new_local_N, d) * norm_est ** (
                        new_local_N + 1) * (omega_est / subpartition) ** ((new_local_N + 1.) / p)
            if subpartition > 1:
                better_var_est = x.omega(partition[i], partition[i + 1], var_steps=var_steps * 3)
                new_subpartition = find_partition(x, partition[i], partition[i + 1], better_var_est, subpartition)
                insert_list(partition, new_subpartition[1:-1], i + 1)
                insert_list(local_Ns, [new_local_N] * (len(new_subpartition) - 2), i + 1)
                insert_list(max_local_error,
                            [max_local_error[i] / (len(new_subpartition) - 1)] * (len(new_subpartition) - 2), i + 1)
                max_local_error[i] = max_local_error[i] / (len(new_subpartition) - 1)
            local_Ns[i] = new_local_N
    return np.array(partition), np.array(y)
