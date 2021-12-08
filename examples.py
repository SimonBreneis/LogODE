import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import logode as lo
import roughpath as rp
import vectorfield as vf


def log_linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def comparison_plot(N, n, n_steps, true_errors, error_bounds, times, kind, regression=True, show=False, save=False,
                    dir=None, adaptive_tol=False, atol=0., rtol=0., h=0.):
    kind = kind[0].upper() + kind[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance'
    else:
        title += 'without adaptive error tolerance'
    if h == 0.:
        title += ', with exact derivatives'
    else:
        title += ', with numerical derivatives'

    if isinstance(N, np.ndarray):
        case = 1
        variable = N
    elif isinstance(n, np.ndarray):
        case = 2
        variable = n
    else:
        case = 3
        variable = n_steps

    fig, ax1 = plt.subplots()

    if case == 1:
        xlabel = "Degree"
    elif case == 2:
        xlabel = "Number of intervals"
    else:
        xlabel = "Signature steps"
    ax1.set_ylabel('Error')
    ax1.loglog(variable, true_errors, color='r', label='True error')
    ax1.loglog(variable, error_bounds, color='g', label='Error bound')
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(variable, true_errors)
        ax1.loglog(variable, constant * variable ** power, 'r--')
        xlabel += '\n\nError ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

        power, constant, r_value, p_value, std_err = log_linear_regression(variable, error_bounds)
        ax1.loglog(variable, constant * variable ** power, 'g--')
        xlabel += ',  Bound ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

    ax1.legend(loc='center left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Time (s)')  # we already handled the x-label with ax1
    ax2.loglog(variable, times, color='b', label='Time')
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(variable, times)
        ax2.loglog(variable, constant * variable ** power, 'b--')
        xlabel += ',\nTime ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

    ax2.legend(loc='center right')

    if case == 1:
        title += f'\n{n} intervals, {n_steps} signature steps'
    elif case == 2:
        title += f'\ndegree {N}, {n_steps} signature steps'
    else:
        title += f'\ndegree {N}, {n} intervals'
    filename = title
    filename = filename.replace('\n', ', ')
    if adaptive_tol:
        title += f', a_tol={atol:.2g}' + r'$n^{-N/p}$' + f', r_tol={rtol:.2g}' + r'$n^{-N/p}$'
    else:
        title += f', a_tol={atol:.2g}, r_tol={rtol:.2g}'
    if h != 0.:
        title += f', h={h:.2g}'
    plt.title(kind + '\n' + title)
    ax1.set_xlabel(xlabel)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:
        plt.show()
    if save:
        plt.savefig(dir + '/' + kind + ', ' + filename, format='pdf')


def smooth_vf_smooth_path(n=100, N=2, k=4, plot=False, exact=False, n_steps=100, method='RK45', atol=1e-09, rtol=1e-06,
                          h=1e-07, norm=None, p=1, var_steps=15):
    """
    Uses a smooth vector field that consists of a linear and a C^infinity part. The path is smooth, and essentially
    k times the (rescaled) unit circle. The driving path is 2-dimensional, the solution is 2-dimensional.
    :param n: Number of intervals
    :param N: Degree of the Log-ODE method
    :param k: Number of times the path revolves around the origin
    :param plot: If true, plots the entire path. If false, does not plot anything
    :param exact: If true, uses as input both the vector field and its first derivative. If false, uses as input only
        the vector field.
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
    :return: The entire path, an error bound, and the time
    """
    logistic = lambda z: 1 / (1 + np.exp(-z))
    partition = np.linspace(0, 1, n + 1)
    path = lambda t: np.array([np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]) / np.sqrt(k)
    x = rp.RoughPathContinuous(path, n_steps=n_steps, var_steps=var_steps)

    def f(y, dx):
        return np.einsum('ij,j', np.array([[y[1] - y[0], -y[1]], [logistic(y[1]), logistic(y[0] - 2 * y[1])]]), dx)

    def df(y, dx):
        return (dx[0, 0] * np.array([logistic(y[1]) + y[0] - y[1], logistic(y[1]) ** 2 * (1 - logistic(y[1]))])
                + dx[0, 1] * np.array(
                    [y[1] + logistic(y[0] - 2 * y[1]),
                     logistic(y[1]) * (1 - logistic(y[1])) * logistic(y[0] - 2 * y[1])])
                + dx[1, 0] * np.array([-logistic(y[1]), logistic(y[0] - 2 * y[1]) * (1 - logistic(y[0] - 2 * y[1])) * (
                        y[1] - y[0] - 2 * logistic(y[1]))])
                + dx[1, 1] * np.array([-logistic(y[0] - 2 * y[1]),
                                       logistic(y[0] + 2 * y[1]) * (1 - logistic(y[0] - 2 * y[1])) * (
                                               -y[1] - 2 * logistic(y[0] - 2 * y[1]))]))

    if exact:
        vec_field = vf.VectorField([f, df], h=h, norm=norm)
    else:
        vec_field = vf.VectorField([f], h=h, norm=norm)

    y_0 = np.array([0., 0.])
    solver = lo.LogODESolver(x, vec_field, y_0, method=method)
    tic = time.perf_counter()
    solution, error_bound = solver.solve_fixed(N=N, partition=partition, atol=atol, rtol=rtol)
    toc = time.perf_counter()
    if plot:
        plt.plot(solution[0, :], solution[1, :])
        plt.show()
    return solution, error_bound, toc - tic


def smooth_vf_smooth_path_discussion(n_vec=np.array([100, 215, 464, 1000, 2150]),
                                     N_vec=np.array([1, 2, 3]), k=4, exact=False,
                                     n_steps_vec=np.array([1, 10, 100, 1000]), method='RK45',
                                     atol=1e-09, rtol=1e-06, h=1e-07, norm=rp.l1, p=1, var_steps=15, show=False,
                                     save=False,
                                     dir='C:/Users/breneis/Desktop/Backup 09112021/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots',
                                     rounds=10, adaptive_tol=False):
    kind = 'smooth vector field, smooth pathqq'
    solutions = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec), 2))
    error_bounds = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec)))
    times = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec)))
    true_errors = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec)))

    true_sol, _, _ = smooth_vf_smooth_path(n=4096, N=2, k=k, plot=False, exact=True, n_steps=3000, method=method,
                                           atol=atol,
                                           rtol=rtol, h=h, norm=norm, p=p, var_steps=1)
    plt.plot(true_sol[0, :], true_sol[1, :])
    plt.title('Solution for ' + kind)
    plt.show()
    true_sol = true_sol[:, -1]
    print(f'The final value for the problem ' + kind + f' is {true_sol}.')

    for i in range(len(N_vec)):
        for j in range(len(n_vec)):
            if adaptive_tol:
                atol_ = atol * n_vec[j]**(-N_vec[i]/float(p))
                rtol_ = rtol * n_vec[j]**(-N_vec[i]/float(p))
            else:
                atol_ = atol
                rtol_ = rtol
            for l in range(len(n_steps_vec)):
                print(f"N = {N_vec[i]}, n = {n_vec[j]}, n_steps = {n_steps_vec[l]}")
                for _ in range(rounds):
                    sol, err, tim = smooth_vf_smooth_path(n=n_vec[j], N=N_vec[i], k=k, plot=False, exact=exact,
                                                          n_steps=n_steps_vec[l], method=method, atol=atol_, rtol=rtol_,
                                                          h=h,
                                                          norm=norm, p=p, var_steps=var_steps)
                    solutions[i, j, l, :] = sol[:, -1]
                    error_bounds[i, j, l] = err
                    times[i, j, l] += tim
                    true_errors[i, j, l] = norm(sol[:, -1] - true_sol)
                times[i, j, l] /= rounds

            comparison_plot(N_vec[i], n_vec[j], n_steps_vec, true_errors[i, j, :], error_bounds[i, j, :],
                            times[i, j, :], kind, True, show, save, dir, adaptive_tol, atol, rtol, h)

        for l in range(len(n_steps_vec)):
            comparison_plot(N_vec[i], n_vec, n_steps_vec[l], true_errors[i, :, l], error_bounds[i, :, l],
                            times[i, :, l], kind, True, show, save, dir, adaptive_tol, atol, rtol, h)
    for j in range(len(n_vec)):
        for l in range(len(n_steps_vec)):
            comparison_plot(N_vec, n_vec[j], n_steps_vec[l], true_errors[:, j, l], error_bounds[:, j, l],
                            times[:, j, l], kind, True, show, save, dir, adaptive_tol, atol, rtol, h)

    times_flat = times.flatten()
    permutation = times_flat.argsort()
    current_error = np.amax(true_errors) + 1.
    opt_times = []
    opt_N = []
    opt_n = []
    opt_n_steps = []
    opt_err = []
    opt_bounds = []
    for i in range(len(permutation)):
        index = np.unravel_index(permutation[i], times.shape)
        err = true_errors[index]
        if err < current_error:
            print(f'In {times[index]:.3g} s, one can achieve an error of {err:.3g} by choosing N={N_vec[index[0]]}, n={n_vec[index[1]]}, n_steps={n_steps_vec[2]}. The corresponding error bound is {error_bounds[index]:.3g}.')
            current_error = err
            opt_times.append(times[index])
            opt_N.append(N_vec[index[0]])
            opt_n.append(n_vec[index[1]])
            opt_n_steps.append(n_steps_vec[2])
            opt_err.append(err)
            opt_bounds.append(error_bounds[index])
    plt.loglog(opt_times, opt_err, 'r', label='True error')
    plt.loglog(opt_times, opt_bounds, 'g', label='Error bound')
    kind = kind[0].upper() + kind[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance'
    else:
        title += 'without adaptive error tolerance'
    if h == 0.:
        title += ', with exact derivatives'
    else:
        title += ', with numerical derivatives'
    plt.title(kind + '\n' + title)
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.show()


'''
# Solving dx_t = x_t dt, x_0=1, on [0,2] with level 2 log-ODE method and 50 equisized subintervals
k = 4
x = lambda t: np.array([np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)]) / np.sqrt(k)
f_vec = [
    lambda y, dx: np.einsum('ij,j', np.array([[y[1] - y[0], -y[1]], [logistic(y[1]), logistic(y[0] - 2 * y[1])]]), dx)]
y_0 = np.array([0., 0.])
partition, solution = lo.log_ode_user(x, f_vec, y_0, 1., norm=lo.l1, p=1)
plt.plot(solution[:, 0], solution[:, 1])
plt.show()
'''
