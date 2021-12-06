import numpy as np
import scipy
from scipy import integrate, special
from esig import tosig as ts
import p_var


def l1(x):
    return np.sum(np.abs(x))


def l2(x):
    """
    Computes the l2-norm of a vector x.
    :param x: The vector
    :return: The l2-norm
    """
    return np.sqrt(np.sum(x ** 2))


def var(x, p, dist):
    return p_var.p_var_backbone(np.shape(x)[0], p, lambda a, b: dist(x[a, :], x[b, :])).value ** (1 / p)


def var_sparse(x, p, dist, n=15):
    n = min(n, len(x))
    indices = np.around(np.linspace(0, len(x) - 1, n)).astype(int)
    return var(x[indices], p, dist)


def var_path(x, s, t, p, dist, n=15):
    """
    Computes an estimate for the p-variation of x on [s,t] using n equidistant points.
    :param x: The path
    :param s: Initial interval point
    :param t: Final interval point
    :param p: Roughness of the path
    :param dist: Metric
    :param n: Number of points used for the estimate of the p-variation
    :return: Estimate (lower bound) for the p-variation
    """
    return var(x(np.linspace(s, t, n)).T, p, dist)


def variation_greedy(x, p, dist):
    """
    Computes the p-variation of the function x.
    :param x: The path, given as an array of values
    :param p: The exponent
    :param dist: The distance function/norm
    :return: An estimate (lower bound) for the p-variation of x
    """
    length = np.shape(x)[0]
    if p == 1:
        return np.sum(np.array([dist(x[i], x[i + 1]) for i in range(length - 1)]))
    if length == 2:
        return dist(x[0], x[1])

    var_total_interval = dist(x[0], x[-1]) ** p
    var_initial = np.array([dist(x[0], x[i]) for i in range(1, length - 1)])
    var_final = np.array([dist(x[i], x[-1]) for i in range(1, length - 1)])
    var_sum = var_initial ** p + var_final ** p
    cut = np.argmax(var_sum) + 1
    var_cut = var_sum[cut - 1]

    if var_total_interval > var_cut:
        return dist(x[0], x[-1])

    return (variation_greedy(x[:cut + 1], p, dist) ** p + variation_greedy(x[cut:], p, dist) ** p) ** (1 / p)


def get_partitions(k, i):
    """
    Returns a list of all partitions of k into i summands (no commutativity of the addition is assumed).
    :param k:
    :param i:
    :return:
    """
    partitions = []
    current_partition = np.zeros(i, dtype=int)
    current_length = 0  # we have not yet added a number to current_partition

    def get_next_partition():
        nonlocal current_length

        if current_length == i:  # we have a complete current_partition
            if int(np.sum(current_partition)) == k:  # current_partition is an actual partition
                partitions.append(current_partition.copy())
            return

        next_element = 1  # next element of current_partition
        previous_sum = int(np.sum(current_partition[:current_length]))
        current_length += 1  # increase current length as we will now add a next element to current_partition

        while previous_sum + next_element + (i - current_length) <= k:
            current_partition[current_length - 1] = next_element
            get_next_partition()
            next_element += 1

        current_length -= 1

    get_next_partition()
    return partitions


def log_sig(x, n):
    """
    Computes the log-signature of a path x up to level n.
    :param x: The path, given as a vector
    :param n: The degree of the log-signature
    :return: A list of length n, the k-th element is a k-tensor which is the k-th level of the log-signature
    """
    if n == 1:
        return [x[-1, :] - x[0, :]]

    dim = np.shape(x)[1]
    sig_vec = ts.stream2sig(x, n)
    indices = np.array([int((dim ** (k + 1) - 1) / (dim - 1) + 0.1) for k in range(n + 1)])
    sig = [sig_vec[indices[i]:indices[i + 1]].reshape([dim] * (i + 1)) for i in range(n)]
    log_sig = []
    for k in range(1, n + 1):
        # computing the k-th level of the log-signature
        ls = sig[k - 1].copy()
        for i in range(2, k + 1):
            # here are the terms of the k-th level we get from (-1)**(i+1) * 1/i * X**i
            ls_i = 0
            partitions = get_partitions(k, i)
            for partition in partitions:
                # we have a specific partition x^l_1 * x^l_2 * ... * x^l_i with l_1 + l_2 + ... + l_i = k
                partition = partition - 1  # indices start at 0
                partition_tensor = sig[partition[0]].copy()
                for j in range(2, i + 1):
                    # compute the tensor product x^l_1 * ... * x^l_j
                    partition_tensor = np.tensordot(partition_tensor, sig[partition[j - 1]].copy(), axes=0)
                ls_i += partition_tensor
            ls += (-1) ** (i + 1) / i * ls_i
        log_sig.append(ls)
    return log_sig


def vfd(f, y, dx, n, h=1e-05):
    """
    Computes the nth derivative of a vector field.
    :param f: The vector field
    :param y: Point at which the derivative is calculated
    :param dx: Direction to which the vector field is applied, an n-tensor
    :param n: Order of the derivative
    :param h: Step size for the numerical differentiation
    :return: An approximation of the n-th derivative
    """
    x_dim = np.shape(dx)[0]

    if n == 0:
        return f(y, dx)
    '''
    if n == 1:
        result = 0
        for i in range(np.shape(dx)[0]):
            direction = np.einsum('ij,j->i', f(y), x[i, :])
            result += (f(y + h/2 * direction)[:, i] - f(y - h/2 * direction)[:, i])/h
        return result.T
    '''
    result = 0
    for i in range(x_dim):
        vec = np.zeros(x_dim)
        vec[i] = 1.
        direction = f(y, vec)
        result += (vfd(f, y + h / 2 * direction, dx[..., i], n - 1, h) - vfd(f, y - h / 2 * direction, dx[..., i],
                                                                             n - 1, h)) / h
        # result += (f(y + h / 2 * direction) @ x[:, i] - f(y - h / 2 * direction) @ x[:, i]) / h
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
        if len(f_vec) >= deg:
            return lambda y: np.sum(np.array([f_vec[i](y, ls[i]) for i in range(deg)]), axis=0)
        else:
            return lambda y: np.sum(np.array([vfd(f_vec[0], y, ls[i], i, h) for i in range(deg)]), axis=0)

    if len(f_vec) >= deg:
        def vf_norm(y):
            summands = np.array([f_vec[i](y, ls[i]) for i in range(deg)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                norm_ls_i = norm(ls[i])
                if norm_ls_i > 0:
                    norm_estimate[0] = np.fmax(norm_estimate[0], (norm(summands[i]) / norm_ls_i) ** (1. / (i + 1)))
            return vf

        return vf_norm
    else:
        def vf_norm(y):
            summands = np.array([vfd(f_vec[0], y, ls[i], i, h) for i in range(deg)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                norm_ls_i = norm(ls[i])
                if norm_ls_i > 0:
                    norm_estimate[0] = np.fmax(norm_estimate[0], (norm(summands[i]) / norm_ls_i) ** (1. / (i + 1)))
            return vf

        return vf_norm


def log_ode_step(x, f_vec, y_0, deg, s, t, n_steps=100, method='RK45', atol=1e-09, rtol=1e-05, h=1e-05, norm=None, p=1,
                 var_steps=15):
    """
    Implementation of the Log-ODE method.
    :param x: Driving path given as a function
    :param f_vec: Vector field, containing also the derivatives
    :param y_0: Initial value
    :param deg: The degree of the Log-ODE method (f needs to be Lip(N))
    :param s: Initial interval point
    :param t: Final interval point
    :param n_steps: Number of (equidistant) steps used in the approximation of the signature of x
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
    :param var_steps: Number of steps used when computing the p-variation of x. Lower var_steps leads to a speed-up, but
        may be more inaccurate.
    :return: Solution on partition points
    """
    x_vector = x(np.linspace(s, t, n_steps + 1)).T
    log_signature = log_sig(x_vector, deg)

    if norm is None:
        vf = lambda t, z: vector_field(f_vec, log_signature, h)(z)
        return integrate.solve_ivp(vf, (0, 1), y_0, method=method, atol=atol, rtol=rtol).y[:, -1]

    norm_estimate = [0.]
    vf = lambda t, z: vector_field(f_vec, log_signature, h, norm, norm_estimate)(z)
    y = integrate.solve_ivp(vf, (0, 1), y_0, method=method, atol=atol, rtol=rtol).y[:, -1]
    variation_estimate = var_sparse(x_vector, p, lambda a, b: norm(b - a), n=var_steps)
    return y, norm_estimate[0], variation_estimate


def log_ode(x, f_vec, y_0, deg, partition, n_steps=100, method='RK45', atol=1e-09, rtol=1e-05, h=1e-05, norm=None, p=1,
            var_steps=15):
    """
    Implementation of the Log-ODE method.
    :param x: Driving path given as a function
    :param f_vec: Vector field, containing also the derivatives
    :param y_0: Initial value
    :param deg: The degree of the Log-ODE method (f needs to be Lip(N))
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param n_steps: Number of (equidistant) steps used in the approximation of the signature of x
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
    :param var_steps: Number of steps used when computing the p-variation of x. Lower var_steps leads to a speed-up, but
        may be more inaccurate.
    :return: Solution on partition points, error bound (-1 if no norm was specified)
    """
    y = np.zeros(shape=(len(y_0), len(partition)))
    y[:, 0] = y_0

    if norm is None:
        for i in range(1, len(partition)):
            x_vector = x(np.linspace(partition[i - 1], partition[i], n_steps + 1)).T
            log_signature = log_sig(x_vector, deg)
            vf = lambda t, z: vector_field(f_vec, log_signature, h)(z)
            y[:, i] = integrate.solve_ivp(vf, (0, 1), y[:, i - 1], method=method, atol=atol, rtol=rtol).y[:, -1]
        return y, -1

    norm_estimate = [0.]
    error_estimate = 0
    for i in range(1, len(partition)):
        x_vector = x(np.linspace(partition[i - 1], partition[i], n_steps + 1)).T
        log_signature = log_sig(x_vector, deg)
        vf = lambda t, z: vector_field(f_vec, log_signature, h, norm, norm_estimate)(z)
        y[:, i] = integrate.solve_ivp(vf, (0, 1), y[:, i - 1], method=method, atol=atol, rtol=rtol).y[:, -1]
        error_estimate += var_to_omega(var_sparse(x_vector, p, lambda a, b: norm(b - a), n=var_steps), p)**((deg+1.)/p) * norm_estimate[0]**(deg+1.)
        norm_estimate[0] = 0.
    return y, local_log_ode_error_constant(p, deg, len(x(0.)))*error_estimate


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


def find_next_interval(x, s, t, p, dist, lower_var, upper_var, var_steps):
    """
    Finds an interval of the form [s,u] with s <= u <= t such that lower_var <= var(x;[s,u]) <= upper_var.
    :param x: The path
    :param s: Initial point of total interval
    :param t: Final point of total interval
    :param p: Roughness of the path
    :param dist: The metric
    :param lower_var: Lower bound of the variation on the new interval
    :param upper_var: Upper bound of the variation on the new interval
    :param var_steps: Number of steps used when computing the p-variation of x. Lower var_steps leads to a speed-up, but
        may be more inaccurate.
    :return: The partition point u.
    """
    total_var = var_path(x, s, t, p, dist, var_steps)
    if total_var <= upper_var:
        return t
    u_current = s + (t - s) * (lower_var + upper_var) / (2 * total_var)
    u_left = s
    u_right = t

    current_var = var_path(x, s, u_current, p, dist, var_steps)

    while not lower_var <= current_var <= upper_var and u_right - u_left > (t - s) * 10 ** (-10):
        if current_var > lower_var:
            u_right = u_current
        else:
            u_left = u_current

        u_current = (u_left + u_right) / 2
        current_var = var_path(x, s, u_current, p, dist, var_steps)

    return u_current


def find_partition(x, s, t, p, dist, total_var, n, var_steps, q=2.):
    """
    Finds a partition of the interval [0,T] such that omega(0,T)/(qn) <= omega(s,t) <= q*omega(0,T)/n.
    :param x: The path
    :param s: Initial time
    :param t: Final time
    :param p: Roughness of the path
    :param dist: The metric
    :param total_var: Estimate for the total variation of x on [0,T]
    :param n: Approximate number of intervals into which [0,T] should be split
    :param var_steps: Number of steps used when computing the p-variation of x. Lower var_steps leads to a speed-up, but
        may be more inaccurate.
    :param q: Tolerance in finding the "perfect" partition.
    :return: The partition
    """
    q = max(q, 1.1)
    partition = [s]
    lower_var = total_var / (q * n) ** (1 / p)
    upper_var = total_var * (q / n) ** (1 / p)
    print(f"Total variation: {total_var}")
    print(f"Lower variation: {lower_var}")
    print(f"Upper variation: {upper_var}")
    while not partition[-1] == t:
        next_point = find_next_interval(x, partition[-1], t, p, dist, lower_var, upper_var, var_steps)
        partition.append(next_point)
    return partition


def beta(p):
    """
    Computes beta = p * (1 + sum_{r=2}^infinity (2/r)^((p+1)/p)).
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
        return beta(3)


def var_to_omega(var, p):
    """
    Converts the p-variation of x to the corresponding control function.
    :param var: The p-variation of x
    :param p: The roughness of x
    :return: Control function on that interval
    """
    return beta(p) * scipy.special.gamma(1. / p + 1) * var ** p


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
    dist = lambda a, b: norm(a - b)
    norm_estimates = np.zeros(10)
    _, norm_estimates[0], var_estimate = log_ode_step(x=x, f_vec=f_vec, y_0=y_0, deg=p, s=0, t=T,
                                                      n_steps=max(n_steps, 100), method=method, atol=100 * atol,
                                                      rtol=100 * rtol, h=h, norm=norm, p=p, var_steps=100)
    _, norm_estimates[1], _ = log_ode_step(x=x, f_vec=f_vec, y_0=y_0, deg=p + 1, s=0, t=T,
                                           n_steps=max(n_steps, 100), method=method, atol=100 * atol,
                                           rtol=100 * rtol, h=h, norm=norm, p=p, var_steps=100)
    _, norm_estimates[2], _ = log_ode_step(x=x, f_vec=f_vec, y_0=y_0, deg=p + 2, s=0, t=T,
                                           n_steps=max(n_steps, 100), method=method, atol=100 * atol,
                                           rtol=100 * rtol, h=h, norm=norm, p=p, var_steps=100)
    norm_incr = max(norm_estimates[2] - norm_estimates[1], norm_estimates[1] - norm_estimates[0])
    norm_estimates[3:] = norm_estimates[2] + norm_incr * np.arange(1, 8)
    print(f"Norm estimates: {norm_estimates}")
    print(f"Error constants: {np.array([local_log_ode_error_constant(p, N, d) for N in range(p, p + 10)])}")
    print(f"Omega: {var_to_omega(var_estimate, p)}")
    number_intervals = np.array([(local_log_ode_error_constant(p, N, d) * norm_estimates[N - p] ** (
                N + 1) * var_to_omega(var_estimate, p) ** ((N + 1.) / p) / eps) ** (p / (N - p + 1)) for N in
                                 range(p, p + 10)])
    print(f"Number of intervals: {number_intervals}")
    complexity = np.array([d ** N for N in range(p, p + 10)]) * number_intervals
    N = np.argmin(complexity) + p
    print(f"N = {N}")
    number_intervals = (number_intervals[N - p] / 10) ** (2. / (1 + p))
    print("Let us find a partition!")
    partition = find_partition(x=x, s=0, t=T, p=p, dist=dist, total_var=var_estimate, n=number_intervals,
                               var_steps=var_steps)
    print("We found a partition!")
    local_Ns = [N] * (len(partition) - 1)
    max_local_error = [eps / number_intervals] * (len(partition) - 1)
    y = [y_0]

    i = 0
    while i < len(partition) - 1:
        print(f"At index {i + 1} of {len(partition) - 1}")
        local_N = local_Ns[i]
        y_next, norm_est, var_est = log_ode_step(x=x, f_vec=f_vec, y_0=y[i], deg=local_N, s=partition[i],
                                                 t=partition[i + 1],
                                                 n_steps=n_steps, method=method, atol=atol, rtol=rtol, h=h, norm=norm,
                                                 p=p, var_steps=var_steps)
        omega_est = var_to_omega(var_est, p)
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
                better_var_est = var_path(x, partition[i], partition[i + 1], p, dist, var_steps * 3)
                new_subpartition = find_partition(x, partition[i], partition[i + 1], p, dist, better_var_est,
                                                  subpartition, var_steps)
                insert_list(partition, new_subpartition[1:-1], i + 1)
                insert_list(local_Ns, [new_local_N] * (len(new_subpartition) - 2), i + 1)
                insert_list(max_local_error,
                            [max_local_error[i] / (len(new_subpartition) - 1)] * (len(new_subpartition) - 2), i + 1)
                max_local_error[i] = max_local_error[i] / (len(new_subpartition) - 1)
            local_Ns[i] = new_local_N
    return np.array(partition), np.array(y)

