import numpy as np
from scipy import integrate
from esig import tosig as ts
import matplotlib.pyplot as plt
import p_var
import time


circle = lambda t: np.array([np.cos(t), np.sin(t)])
n = 1000
print(ts.stream2logsig(np.array(circle(n*n*np.linspace(0, 2*np.pi, 100*n*n))).T/n, 5))
time.sleep(3600)
solution = integrate.solve_ivp(lambda t, x: x, (0., 1.), np.array([1.]))
print(solution.t)
print(solution.y)
print(np.e)


def l2(x):
    """
    Computes the l2-norm of a vector x.
    :param x: The vector
    :return: The l2-norm
    """
    return np.sqrt(np.sum(x**2))


def var(x, p, dist):
    return p_var.p_var_backbone(np.shape(x)[0], p, lambda a, b: dist(x[a, :], x[b, :])).value ** (1 / p)


def get_deg(gamma):
    """
    Computes the degree of a log-ODE method given the regularity of the vector field.
    :param gamma: The vector field is Lip(gamma-1)
    :return: The corresponding degree of the log-ODE method
    """
    deg = int(gamma)
    if deg == gamma:
        deg -= 1
    return deg


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
        return np.sum(np.array([dist(x[i], x[i+1]) for i in range(length-1)]))
    if length == 2:
        return dist(x[0], x[1])

    var_total_interval = dist(x[0], x[-1])**p
    var_initial = np.array([dist(x[0], x[i]) for i in range(1, length-1)])
    var_final = np.array([dist(x[i], x[-1]) for i in range(1, length-1)])
    var_sum = var_initial**p + var_final**p
    cut = np.argmax(var_sum) + 1
    var_cut = var_sum[cut-1]

    if var_total_interval > var_cut:
        return dist(x[0], x[-1])

    return (variation_greedy(x[:cut+1], p, dist)**p + variation_greedy(x[cut:], p, dist)**p)**(1/p)


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

        while previous_sum + next_element + (i-current_length) <= k:
            current_partition[current_length-1] = next_element
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
    indices = np.array([int((dim**(k+1)-1)/(dim-1) + 0.1) for k in range(n+1)])
    sig = [sig_vec[indices[i]:indices[i+1]].reshape([dim]*(i+1)) for i in range(n)]
    log_sig = []
    for k in range(1, n+1):
        # computing the k-th level of the log-signature
        ls = sig[k-1]
        for i in range(2, k+1):
            # here are the terms of the k-th level we get from (-1)**(i+1) * 1/i * X**i
            ls_i = 0
            partitions = get_partitions(k, i)
            for partition in partitions:
                # we have a specific partition x^l_1 * x^l_2 * ... * x^l_i with l_1 + l_2 + ... + l_i = k
                partition = partition - 1  # indices start at 0
                partition_tensor = sig[partition[0]]
                for j in range(2, i+1):
                    # compute the tensor product x^l_1 * ... * x^l_j
                    partition_tensor = np.tensordot(partition_tensor, sig[partition[j-1]], axes=0)
                ls_i += partition_tensor
            ls += (-1)**(i+1) / i * ls_i
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
        result += (vfd(f, y + h/2*direction, dx[..., i], n-1, h) - vfd(f, y - h/2*direction, dx[..., i], n-1, h))/h
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
                    norm_estimate[0] = np.fmax(norm_estimate[0], norm(summands[i])/norm_ls_i)
            return vf
        return vf_norm
    else:
        def vf_norm(y):
            summands = np.array([vfd(f_vec[0], y, ls[i], i, h) for i in range(deg)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                norm_ls_i = norm(ls[i])
                if norm_ls_i > 0:
                    norm_estimate[0] = np.fmax(norm_estimate[0], norm(summands[i])/norm_ls_i)
            return vf
        return vf_norm


def log_ode(x, f_vec, y_0, gamma, partition, n_steps=100, method='RK45', atol=1e-09, rtol=1e-05, h=1e-05, norm=None, p=1):
    """
    Implementation of the Log-ODE method.
    :param x: Driving path given as a function
    :param f_vec: Vector field, containing also the derivatives
    :param y_0: Initial value
    :param gamma: f is interpreted as a Lip(gamma-1) vector field. Order of the Log-ODE method is floor(gamma)
        (if gamma is an integer, the order is gamma-1)
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
        sum_{i in partition} \|x\|_{p, [t_i, t_{i+1}]}^deg,
        as this is more relevant to the problem.
    :param p: The exponent for which the p-variation of x should be computed.
        This method is only reasonable if p < gamma
    :return: Solution on partition points
    """
    deg = get_deg(gamma)
    y = np.zeros(shape=(len(y_0), len(partition)))
    y[:, 0] = y_0

    if norm is None:
        for i in range(1, len(partition)):
            x_vector = x(np.linspace(partition[i-1], partition[i], n_steps+1)).T
            log_signature = log_sig(x_vector, deg)
            vf = lambda t, z: vector_field(f_vec, log_signature, h)(z)
            y[:, i] = integrate.solve_ivp(vf, (0, 1), y[:, i - 1], method=method, atol=atol, rtol=rtol).y[:, -1]
        return y

    norm_estimate = [0.]
    variation_estimate = 0
    for i in range(1, len(partition)):
        x_vector = x(np.linspace(partition[i-1], partition[i], n_steps+1)).T
        log_signature = log_sig(x_vector, deg)
        vf = lambda t, z: vector_field(f_vec, log_signature, h, norm, norm_estimate)(z)
        y[:, i] = integrate.solve_ivp(vf, (0, 1), y[:, i - 1], method=method, atol=atol, rtol=rtol).y[:, -1]
        variation_estimate += var(x_vector, p, lambda a, b: norm(b-a))**gamma
    return y, norm_estimate[0], variation_estimate


'''
def log_ode_1(x, f, y_0, partition, method='RK45', atol=1e-09, rtol=1e-05):
    """
    First order Log-ODE method.
    :param x: Driving path given as a vector on the partition
    :param f: Vector field
    :param y_0: Initial value
    :param partition: Partition of the interval on which we apply the Log-ODE method
    :param method: A method for solving initial value problems implemented in the scipy.integrate library
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :return: Solution on partition points
    """
    y = np.zeros(shape=(len(y_0), len(partition)))
    y[:, 0] = y_0

    for i in range(1, len(partition)):
        y[:, i] = integrate.solve_ivp(lambda t, z: f(z)@(x[:, i] - x[:, i-1]), (0, 1), y[:, i-1], method=method,
                                      atol=atol, rtol=rtol).y[:, -1]

    return y
'''


x = lambda t: np.array([np.cos(2*np.pi*t), np.sin(2*np.pi*t)])
partition = np.linspace(0, 1, 100)
print(f"variation = {variation_greedy(x(partition).T, 10, lambda a, b: l2(b-a))}")
print(f"variation = {var(x(partition).T, 10, lambda a, b: l2(b-a))}")

x = lambda t: np.array([np.cos(t), np.sin(t)])
partition = np.linspace(0, 1, 100)
sig = ts.stream2sig(x(partition).T, 2)
sig_1 = sig[1:(2+1)]
sig_2 = sig[(2+1):].reshape(2, 2)
log_1 = sig_1
log_2 = sig_2 - 1/2 * np.einsum('i,j', sig_1, sig_1)
print(log_1)
print(log_2)
print(log_sig(x(partition).T, 2))

'''
# Solving dx_t = x_t dt, x_0=1, on [0,2] with level 1 log-ODE method and 50 equisized subintervals
partition = np.linspace(0, 2, 50)
solution = log_ode(lambda t: np.array([t]), [lambda y, dx: np.array([y*dx])], np.array([1.]), 1, partition)
plt.plot(partition, np.exp(partition))
plt.plot(partition, solution[0, :])
plt.show()
'''

# Solving dx_t = x_t dt, x_0=1, on [0,2] with level 2 log-ODE method and 50 equisized subintervals
logistic = lambda x: 1/(1+np.exp(-x))
k = 4
partition = np.linspace(0, 1, 100*k)
x = lambda t: np.array([np.cos(2*np.pi*k*t), np.sin(2*np.pi*k*t)])/np.sqrt(k)
f_vec = [lambda y, dx: np.einsum('ij,j', np.array([[y[1]-y[0], -y[1]], [logistic(y[1]), logistic(y[0] - 2*y[1])]]), dx)]
y_0 = np.array([0., 0.])
gamma = 1.1
solution = log_ode(x, f_vec, y_0, gamma, partition)
plt.plot(solution[0, :], solution[1, :])

# Solving dx_t = x_t dt, x_0=1, on [0,2] with level 2 log-ODE method and 50 equisized subintervals
logistic = lambda x: 1/(1+np.exp(-x))
k = 4
partition = np.linspace(0, 1, 100*k)
x = lambda t: np.array([np.cos(2*np.pi*k*t), np.sin(2*np.pi*k*t)])/np.sqrt(k)


def f(y, dx):
    return np.einsum('ij,j', np.array([[y[1]-y[0], -y[1]], [logistic(y[1]), logistic(y[0] - 2*y[1])]]), dx)


def df(y, dx):
    return (dx[0, 0] * np.array([logistic(y[1]) + y[0] - y[1], logistic(y[1])**2 * (1-logistic(y[1]))])
            + dx[0, 1] * np.array([y[1] + logistic(y[0]-2*y[1]), logistic(y[1])*(1-logistic(y[1]))*logistic(y[0]-2*y[1])])
            + dx[1, 0] * np.array([-logistic(y[1]), logistic(y[0]-2*y[1])*(1-logistic(y[0]-2*y[1]))*(y[1]-y[0]-2*logistic(y[1]))])
            + dx[1, 1] * np.array([-logistic(y[0]-2*y[1]), logistic(y[0]+2*y[1])*(1-logistic(y[0]-2*y[1]))*(-y[1]-2*logistic(y[0]-2*y[1]))]))


f_vec = [f, df]
y_0 = np.array([0., 0.])
gamma = 2.1
solution = log_ode(x, f_vec, y_0, gamma, partition)
plt.plot(solution[0, :], solution[1, :])

# Solving dx_t = x_t dt, x_0=1, on [0,2] with level 2 log-ODE method and 50 equisized subintervals
logistic = lambda x: 1/(1+np.exp(-x))
k = 4
partition = np.linspace(0, 1, 100*k)
x = lambda t: np.array([np.cos(2*np.pi*k*t), np.sin(2*np.pi*k*t)])/np.sqrt(k)


def f(y, dx):
    return np.array([[y[1]-y[0], -y[1]], [logistic(y[1]), logistic(y[0] - 2*y[1])]])@dx


f_vec = [f]
y_0 = np.array([0., 0.])
gamma = 2.1
solution = log_ode(x, f_vec, y_0, gamma, partition)
plt.plot(solution[0, :], solution[1, :])

# Solving dx_t = x_t dt, x_0=1, on [0,2] with level 2 log-ODE method and 50 equisized subintervals
logistic = lambda x: 1/(1+np.exp(-x))
k = 4
partition = np.linspace(0, 1, 100*k)
x = lambda t: np.array([np.cos(2*np.pi*k*t), np.sin(2*np.pi*k*t)])/np.sqrt(k)


def f(y, dx):
    return np.array([[y[1]-y[0], -y[1]], [logistic(y[1]), logistic(y[0] - 2*y[1])]])@dx


f_vec = [f]
y_0 = np.array([0., 0.])
gamma = 3.1
tic = time.perf_counter()
solution = log_ode(x, f_vec, y_0, gamma, partition, n_steps=20, norm=l2, p=1.5)
toc = time.perf_counter()
print(f"Norm estimate: {solution[1]}")
print(f"Variation estimate: {solution[2]}")
print(f"Time: {np.round(toc-tic, 2)}")
plt.plot(solution[0][0, :], solution[0][1, :])
plt.show()
