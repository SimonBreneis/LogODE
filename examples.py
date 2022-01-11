import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
import logode as lo
import roughpath as rp
import vectorfield as vf
import sympy as sp
import tensoralgebra as ta
import fbm


def rough_fractional_Brownian_path(H, n, dim, T=1., p=0., var_steps=15, norm=ta.l1, save_level=0):
    """
    Returns a sample path of fractional Brownian motion as an instance of a RoughPath.
    :param H: Hurst parameter
    :param n: Number of equidistant evaluation points
    :param dim: Dimension of the path
    :param T: Final time
    :param p: Parameter in which the roughness should be measured (default choice 1/H + 0.05)
    :param var_steps: Number of steps used in approximating p-variation
    :param norm: Norm for computing p-variation
    :param save_level: Level up to which the signatures on the time grid are stored
    :return: An instance of a RoughPathDiscrete
    """
    if p == 0.:
        p = 1 / H + 0.05
        if p - int(p) > 0.95:
            p = np.ceil(p)
    if save_level == 0:
        save_level = np.ceil(p)
    f = fbm.FBM(n=n, hurst=H, length=T)
    times = fbm.times(n=n, length=T)
    values = np.array([f.fbm() for _ in range(dim)])
    if save_level <= 5:
        return rp.RoughPathDiscrete(times=times, values=values.T, p=p, var_steps=var_steps, norm=norm,
                                    save_level=save_level)
    return rp.RoughPathContinuous(path=scipy.interpolate.interp1d(times, values, axis=-1),
                                  sig_steps=int(max(15, n / 1000)), p=p, var_steps=var_steps, norm=norm)


def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def compare_N(N, n, sig_steps, errors, error_bounds, times, description, x_time=False, show=False,
              save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=False, sym_path=False):
    """
    Constructs many plots for the discussion of the examples.
    :param N: The level(s) of the log-ODE method
    :param n: The number(s) of intervals of the log-ODE method
    :param sig_steps: The number(s) of steps used in the approximation of the signatures
    :param errors: The errors of the log-ODE approximations
    :param error_bounds: The error bounds of the log-ODE approximations
    :param times: The run-times of applying the log-ODE approximations
    :param description: Description of the example that is being discussed
    :param x_time: If True, the argument (x-axis) is time
    :param show: If True, shows all the plots when they are constructed
    :param save: If True, saves all the plots (as PDF files) in directory
    :param directory: Folder where the plots are saved (if save is True)
    :param adaptive_tol: Were the error tolerances of the ODE solvers chosen adaptively?
    :param atol: (Base) absolute error tolerance of the ODE solver
    :param rtol: (Base) relative error tolerance of the ODE solver
    :param sym_vf: Were symbolic vector fields used?
    :param sym_path: Were symbolic rough paths used?
    :return: Nothing
    """
    description = description[0].upper() + description[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance, '
    if sym_vf:
        title += 'symbolic derivatives'
    else:
        title += 'numeric derivatives'
    if sym_path:
        title += ', symbolic signatures'
    else:
        title += ', numeric signatures'
    if adaptive_tol:
        title += f'\na_tol={atol:.2g}' + r'$n^{-N/p}$' + f', r_tol={rtol:.2g}' + r'$n^{-N/p}$'
    else:
        title += f'\na_tol={atol:.2g}, r_tol={rtol:.2g}'

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Error')
    color = ['b', 'g', 'r', 'y', 'm', 'k', 'c']

    if x_time:
        ax1.set_xlabel('Time (s)')
        for i in range(len(N)):
            c = color[int(i % 7)]
            t = times[i, :]
            e = errors[i, :]
            j = 1
            while j < len(t):
                if t[j] < t[j - 1]:
                    t = np.delete(t, [j])
                    e = np.delete(e, [j])
                else:
                    j += 1
            ax1.loglog(t, e, c, label=f'N={N[i]}')
    else:
        ax1.set_xlabel('Number of intervals')
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Time (s)')  # we already handled the x-label with ax1
        for i in range(len(N)):
            c = color[int(i % 7)]
            ax1.loglog(n, errors[i, :], c + '-', label=f'N={N[i]}')
            ax1.loglog(n, error_bounds[i, :], c + '--')
            ax2.loglog(n, times[i, :], c + '+-')

    ax1.legend(loc='best')

    specifics = 'compare N'
    if x_time:
        specifics += ' w.r.t. time'
    if not sym_path:
        specifics += f', {sig_steps} signature steps'
    filename = specifics
    plt.title(description + '\n' + title + '\n' + specifics)
    if not x_time:
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:
        plt.show()
    if save:
        plt.savefig(directory + '/' + description + ', ' + filename, format='pdf')


def compare_n(N, n, sig_steps, errors, error_bounds, times, description, regression=True, show=False,
              save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=False, sym_path=False):
    """
    Constructs many plots for the discussion of the examples.
    :param N: The level(s) of the log-ODE method
    :param n: The number(s) of intervals of the log-ODE method
    :param sig_steps: The number(s) of steps used in the approximation of the signatures
    :param errors: The errors of the log-ODE approximations
    :param error_bounds: The error bounds of the log-ODE approximations
    :param times: The run-times of applying the log-ODE approximations
    :param description: Description of the example that is being discussed
    :param regression: If True, apply log-log-regression and also plot the result
    :param show: If True, shows all the plots when they are constructed
    :param save: If True, saves all the plots (as PDF files) in directory
    :param directory: Folder where the plots are saved (if save is True)
    :param adaptive_tol: Were the error tolerances of the ODE solvers chosen adaptively?
    :param atol: (Base) absolute error tolerance of the ODE solver
    :param rtol: (Base) relative error tolerance of the ODE solver
    :param sym_vf: Were symbolic vector fields used?
    :param sym_path: Were symbolic rough paths used?
    :return: Nothing
    """
    description = description[0].upper() + description[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance, '
    if sym_vf:
        title += 'symbolic derivatives'
    else:
        title += 'numeric derivatives'
    if sym_path:
        title += ', symbolic signatures'
    else:
        title += ', numeric signatures'

    if adaptive_tol:
        title += f'\na_tol={atol:.2g}' + r'$n^{-N/p}$' + f', r_tol={rtol:.2g}' + r'$n^{-N/p}$'
    else:
        title += f'\na_tol={atol:.2g}, r_tol={rtol:.2g}'

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    x_label = 'Number of intervals'
    ax1.set_ylabel('Error')
    ax2.set_ylabel('Time (s)')  # we already handled the x-label with ax1

    ax1.loglog(n, errors, color='r', label='True error')
    ax1.loglog(n, error_bounds, color='g', label='Error bound')
    reg_index = np.sum(n < np.ones(len(n)) * 35)
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(n[reg_index:], errors[reg_index:])
        ax1.loglog(n, constant * n ** power, 'r--')
        x_label += '\n\nError ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

        power, constant, r_value, p_value, std_err = log_linear_regression(n[reg_index:], error_bounds[reg_index:])
        ax1.loglog(n, constant * n ** power, 'g--')
        x_label += ',  Bound ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

    ax2.loglog(n, times, color='b', label='Time')
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(n[reg_index:], times[reg_index:])
        ax2.loglog(n, constant * n ** power, 'b--')
        x_label += ',\nTime ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

    ax1.set_xlabel(x_label)
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')

    specifics = f'degree {N}'
    if not sym_path:
        specifics += f', {sig_steps} signature steps'
    filename = specifics
    plt.title(description + '\n' + title + '\n' + specifics)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:
        plt.show()
    if save:
        plt.savefig(directory + '/' + description + ', ' + filename, format='pdf')


def compare_sig_steps(N, n, sig_steps, errors, error_bounds, times, description, regression=True, show=False,
                      save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=False, sym_path=False):
    """
    Constructs many plots for the discussion of the examples.
    :param N: The level(s) of the log-ODE method
    :param n: The number(s) of intervals of the log-ODE method
    :param sig_steps: The number(s) of steps used in the approximation of the signatures
    :param errors: The errors of the log-ODE approximations
    :param error_bounds: The error bounds of the log-ODE approximations
    :param times: The run-times of applying the log-ODE approximations
    :param description: Description of the example that is being discussed
    :param regression: If True, apply log-log-regression and also plot the result
    :param show: If True, shows all the plots when they are constructed
    :param save: If True, saves all the plots (as PDF files) in directory
    :param directory: Folder where the plots are saved (if save is True)
    :param adaptive_tol: Were the error tolerances of the ODE solvers chosen adaptively?
    :param atol: (Base) absolute error tolerance of the ODE solver
    :param rtol: (Base) relative error tolerance of the ODE solver
    :param sym_vf: Were symbolic vector fields used?
    :param sym_path: Were symbolic rough paths used?
    :return: Nothing
    """
    description = description[0].upper() + description[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance, '
    if sym_vf:
        title += 'symbolic derivatives'
    else:
        title += 'numeric derivatives'
    if sym_path:
        title += ', symbolic signatures'
    else:
        title += ', numeric signatures'
    if adaptive_tol:
        title += f'\na_tol={atol:.2g}' + r'$n^{-N/p}$' + f', r_tol={rtol:.2g}' + r'$n^{-N/p}$'
    else:
        title += f'\na_tol={atol:.2g}, r_tol={rtol:.2g}'

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    x_label = "Signature steps"
    ax1.set_ylabel('Error')
    ax2.set_ylabel('Time (s)')  # we already handled the x-label with ax1

    ax1.loglog(sig_steps, errors, color='r', label='True error')
    ax1.loglog(sig_steps, error_bounds, color='g', label='Error bound')
    reg_index = np.sum(sig_steps < np.ones(len(sig_steps)) * 35)
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(sig_steps[reg_index:], errors[reg_index:])
        ax1.loglog(sig_steps, constant * sig_steps ** power, 'r--')
        x_label += '\n\nError ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

        power, constant, r_value, p_value, std_err = log_linear_regression(sig_steps[reg_index:],
                                                                           error_bounds[reg_index:])
        ax1.loglog(sig_steps, constant * sig_steps ** power, 'g--')
        x_label += ',  Bound ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

    ax2.loglog(sig_steps, times, color='b', label='Time')
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(sig_steps[reg_index:], times[reg_index:])
        ax2.loglog(sig_steps, constant * sig_steps ** power, 'b--')
        x_label += ',\nTime ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'

    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    ax1.set_xlabel(x_label)

    specifics = f'degree {N}, {n} intervals'
    filename = specifics
    plt.title(description + '\n' + title + '\n' + specifics)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:
        plt.show()
    if save:
        plt.savefig(directory + '/' + description + ', ' + filename, format='pdf')


def smooth_path(n=100, N=2, plot=False, sig_steps=100, atol=1e-09, rtol=1e-06, sym_path=False, sym_vf=False, param=4):
    """
    Uses a smooth vector field that consists of a linear and a C^infinity part. The path is smooth, and essentially
    k times the (rescaled) unit circle. The driving path is 2-dimensional, the solution is 2-dimensional.
    :param n: Number of intervals
    :param N: Degree of the Log-ODE method
    :param param: Number of times the path revolves around the origin
    :param plot: If true, plots the entire path. If false, does not plot anything
    :param sig_steps: Number of (equidistant) steps used in the approximation of the signature of x
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param sym_path: If true, uses symbolic computation for the signature increments. Else, approximates them
        numerically
    :param sym_vf: If true, uses symbolic computation for the vector field derivatives. Else, approximates them
        numerically
    :return: The entire path, an error bound, and the time
    """
    p = 1
    var_steps = 15
    norm = ta.l1
    method = 'RK45'
    partition = np.linspace(0, 1, n + 1)
    h = 1e-07

    if sym_path:
        param = sp.Integer(param)
        t = sp.symbols('t')
        path = sp.Array(
            [sp.cos(2 * param * sp.pi * t) / sp.sqrt(param), sp.sin(2 * param * sp.pi * t) / sp.sqrt(param)])
        x = rp.RoughPathSymbolic(path=path, t=t, p=p, var_steps=var_steps, norm=norm)
    else:
        path = lambda t: np.array([np.cos(2 * np.pi * param * t), np.sin(2 * np.pi * param * t)]) / np.sqrt(param)
        x = rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    if sym_vf:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z)))]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
    else:
        f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                                   1 / (1 + np.exp(-y[1])) * x[0] + 1 / (1 + np.exp(y[0] - 2 * y[1])) * x[1]])
        vec_field = vf.VectorFieldNumeric(f=[f], h=h, norm=norm)

    y_0 = np.array([0., 0.])
    solver = lo.LogODESolver(x, vec_field, y_0, method=method)
    tic = time.perf_counter()
    solution, error_bound = solver.solve_fixed(N=N, partition=partition, atol=atol, rtol=rtol)
    toc = time.perf_counter()
    if plot:
        plt.plot(solution[0, :], solution[1, :])
        plt.show()
    return solution, error_bound, toc - tic


def pure_area(n=100, N=2, plot=False, sig_steps=100, atol=1e-09, rtol=1e-06, sym_path=False, sym_vf=False, param=None):
    """
    Uses a smooth vector field that consists of a linear and a C^infinity part. The path is pure area. The driving path
    is 2-dimensional, the solution is 2-dimensional.
    :param n: Number of intervals
    :param N: Degree of the Log-ODE method
    :param param: Deprecated parameter
    :param plot: If true, plots the entire path. If false, does not plot anything
    :param sig_steps: Number of (equidistant) steps used in the approximation of the signature of x
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param sym_path: If true, uses symbolic computation for the signature increments. Else, approximates them
        numerically
    :param sym_vf: If true, uses symbolic computation for the vector field derivatives. Else, approximates them
        numerically
    :return: The entire path, an error bound, and the time
    """
    p = 2
    var_steps = 15
    norm = ta.l1
    method = 'RK45'
    partition = np.linspace(0, 1, n + 1)
    h = 1e-07

    path = lambda s, t: ta.NumericTensor(
        [1., np.array([0., 0.]), np.array([[0., np.pi * (t - s)], [np.pi * (s - t), 0.]])])
    x = rp.RoughPathExact(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    if sym_vf:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z)))]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
    else:
        f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                                   1 / (1 + np.exp(-y[1])) * x[0] + 1 / (1 + np.exp(y[0] - 2 * y[1])) * x[1]])
        vec_field = vf.VectorFieldNumeric(f=[f], h=h, norm=norm)

    y_0 = np.array([0., 0.])
    solver = lo.LogODESolver(x, vec_field, y_0, method=method)
    tic = time.perf_counter()
    solution, error_bound = solver.solve_fixed(N=N, partition=partition, atol=atol, rtol=rtol)
    toc = time.perf_counter()
    if plot:
        plt.plot(solution[0, :], solution[1, :])
        plt.show()
    return solution, error_bound, toc - tic


def third_level(n=100, N=2, plot=False, sig_steps=100, atol=1e-09, rtol=1e-06, sym_path=False, sym_vf=False, param=4):
    """
    Uses a smooth vector field that consists of a linear and a C^infinity part. The path is smooth, and essentially
    k times the (rescaled) unit circle. The driving path is 2-dimensional, the solution is 2-dimensional.
    :param n: Number of intervals
    :param N: Degree of the Log-ODE method
    :param param: Number of times the path revolves around the origin
    :param plot: If true, plots the entire path. If false, does not plot anything
    :param sig_steps: Number of (equidistant) steps used in the approximation of the signature of x
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param sym_path: If true, uses symbolic computation for the signature increments. Else, approximates them
        numerically
    :param sym_vf: If true, uses symbolic computation for the vector field derivatives. Else, approximates them
        numerically
    :return: The entire path, an error bound, and the time
    """
    p = 1
    var_steps = 15
    norm = ta.l1
    method = 'RK45'
    partition = np.linspace(0, 1, n + 1)
    h = 1e-07

    def path(t):
        t = ((t * param) % 1.0) * 4
        if isinstance(t, np.ndarray):
            a = (t < 1) * np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), np.zeros(len(t))])
            b = ((1 <= t) & (t < 2)) * np.array([np.ones(len(t)), np.zeros(len(t)), t - 1])
            c = ((2 <= t) & (t < 3)) * np.array([np.cos(2 * np.pi * (t - 2)), -np.sin(2 * np.pi * (t - 2)), np.ones(len(t))])
            d = (3 <= t) * np.array([np.ones(len(t)), np.zeros(len(t)), 1 - (t - 3)])
        else:
            a = (t < 1) * np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), 0])
            b = ((1 <= t) & (t < 2)) * np.array([1, 0, t - 1])
            c = ((2 <= t) & (t < 3)) * np.array([np.cos(2 * np.pi * (t - 2)), -np.sin(2 * np.pi * (t - 2)), 1])
            d = (3 <= t) * np.array([1, 0, 1 - (t - 3)])
        return (a + b + c + d) / param**(1/3)

    x = rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    if sym_vf:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z, sp.sin(y * y)],
                      [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z))), sp.tanh(y * z) * sp.cos(y)]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
    else:
        f = lambda y, x: np.array(
            [(y[1] - y[0]) * x[0] - y[1] * x[1] + np.sin(y[0] * y[0]) * x[2],
             1 / (1 - np.exp(y[1])) * x[0] + 1 / (1 - np.exp(y[0] - 2 * y[1])) * x[1] + np.tanh(y[1] * y[0]) * np.cos(
                 y[0]) * x[2]])
        vec_field = vf.VectorFieldNumeric(f=[f], h=h, norm=norm)

    y_0 = np.array([0., 0.])
    solver = lo.LogODESolver(x, vec_field, y_0, method=method)
    tic = time.perf_counter()
    solution, error_bound = solver.solve_fixed(N=N, partition=partition, atol=atol, rtol=rtol)
    toc = time.perf_counter()
    if plot:
        plt.plot(solution[0, :], solution[1, :])
        plt.show()
    return solution, error_bound, toc - tic


def fBm_path(n=100, N=2, plot=False, sig_steps=100, atol=1e-09, rtol=1e-06, sym_path=False, sym_vf=False, param=0.5):
    """
    Uses a smooth vector field that consists of a linear and a C^infinity part. The path is a fractional Brownian
    motion. The driving path is 2-dimensional, the solution is 2-dimensional.
    :param n: Number of intervals
    :param N: Degree of the Log-ODE method
    :param param: Either Hurst parameter of the driving path, or a fBm given as a rough path
    :param plot: If true, plots the entire path. If false, does not plot anything
    :param sig_steps: Number of (equidistant) steps used in the approximation of the signature of x
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param sym_path: If true, uses symbolic computation for the signature increments. Else, approximates them
        numerically
    :param sym_vf: If true, uses symbolic computation for the vector field derivatives. Else, approximates them
        numerically
    :return: The entire path, an error bound, and the time
    """

    var_steps = 15
    norm = ta.l1
    method = 'RK45'
    partition = np.linspace(0, 1, n + 1)
    h = 1e-07

    if isinstance(param, rp.RoughPath):
        x = param
        p = x.p
    else:
        p = 1 / param + 0.1
        x = rough_fractional_Brownian_path(H=param, n=n * sig_steps, dim=2, T=1., p=p, var_steps=var_steps, norm=norm,
                                           save_level=N)

    if sym_vf:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z)))]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
    else:
        f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                                   1 / (1 + np.exp(-y[1])) * x[0] + 1 / (1 + np.exp(y[0] - 2 * y[1])) * x[1]])
        vec_field = vf.VectorFieldNumeric(f=[f], h=h, norm=norm)

    y_0 = np.array([0., 0.])
    solver = lo.LogODESolver(x, vec_field, y_0, method=method)
    tic = time.perf_counter()
    solution, error_bound = solver.solve_fixed(N=N, partition=partition, atol=atol, rtol=rtol)
    toc = time.perf_counter()
    if plot:
        plt.plot(solution[0, :], solution[1, :])
        plt.show()
    return solution, error_bound, toc - tic


def four_dim(n=100, N=2, plot=False, sig_steps=100, atol=1e-09, rtol=1e-06, sym_path=False, sym_vf=False, param=0.5):
    """
    Uses a smooth vector field that consists of a linear and a C^infinity part. The path is a fractional Brownian
    motion. The driving path is 2-dimensional, the solution is 2-dimensional.
    :param n: Number of intervals
    :param N: Degree of the Log-ODE method
    :param param: Either Hurst parameter of the driving path, or a fBm given as a rough path
    :param plot: If true, plots the entire path. If false, does not plot anything
    :param sig_steps: Number of (equidistant) steps used in the approximation of the signature of x
    :param atol: Absolute error tolerance of the ODE solver
    :param rtol: Relative error tolerance of the ODE solver
    :param sym_path: If true, uses symbolic computation for the signature increments. Else, approximates them
        numerically
    :param sym_vf: If true, uses symbolic computation for the vector field derivatives. Else, approximates them
        numerically
    :return: The entire path, an error bound, and the time
    """

    var_steps = 15
    norm = ta.l1
    method = 'RK45'
    partition = np.linspace(0, 1, n + 1)
    h = 1e-07

    if isinstance(param, rp.RoughPath):
        x = param
        p = x.p
    else:
        p = 1 / param + 0.1
        x = rough_fractional_Brownian_path(H=param, n=n * sig_steps, dim=4, T=1., p=p, var_steps=var_steps, norm=norm,
                                           save_level=N)

    if sym_vf:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z, sp.sin(y), sp.cos(z - y / 3)],
                      [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z))), sp.sin(y * z) ** 2,
                       sp.tanh(y * y * z - y)]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
    else:
        f = lambda y, x: np.array(
            [(y[1] - y[0]) * x[0] - y[1] * x[1] + np.sin(y[0]) * x[2] + np.cos(y[1] - y[0] / 3) * x[3],
             1 / (1 + np.exp(-y[1])) * x[0] + 1 / (1 + np.exp(y[0] - 2 * y[1])) * x[1]
             + np.sin(y[0] * y[1]) ** 2 * x[2] + np.tanh(y[0] ** 2 * y[1] - y[0]) * x[3]])
        vec_field = vf.VectorFieldNumeric(f=[f], h=h, norm=norm)

    y_0 = np.array([0., 0.])
    solver = lo.LogODESolver(x, vec_field, y_0, method=method)
    tic = time.perf_counter()
    solution, error_bound = solver.solve_fixed(N=N, partition=partition, atol=atol, rtol=rtol)
    toc = time.perf_counter()
    if plot:
        plt.plot(solution[0, :], solution[1, :])
        plt.show()
    return solution, error_bound, toc - tic


def discussion(example, show=False, save=False, rounds=1, adaptive_tol=False, sym_path=False, sym_vf=False):
    """
    Discusses the problem in smooth_vf_smooth_path.
    :param example: Float that chooses the example that is discussed.
        0 < example < 1: Path is two-dimensional fractional Brownian motion with Hurst parameter example
        1: Path is two-dimensional pure area path
        1 < example <= 2: Path is two-dimensional rescaled unit circle, with 1/(example-1) rotations
        4: Path is four-dimensional
    :param show: Shows all the plots that are made
    :param save: Saves all the plots that are made
    :param rounds: Number of times each problem is solved (to get a better estimate of the run time)
    :param adaptive_tol: If true, takes into account that the error tolerance of the ODE solver must scale with the
        number of intervals used. Uses this to try to eliminate the error from the ODE solver
    :param sym_path: If true, uses symbolic computation for the signature increments. Else, approximates them
        numerically
    :param sym_vf: If true, uses symbolic computation for the vector field derivatives. Else, approximates them
        numerically
    :return: The entire path, an error bound, and the time
    """
    # 10000, 15849, 25119, 39811, 63096, 100000
    if adaptive_tol:
        atol = 1e-05
        rtol = 1e-02
    else:
        atol = 1e-09
        rtol = 1e-06

    if 0 < example < 1:
        kind = fBm_path
        description = f'fractional Brownian motion, H={example}'
        var_steps = 15
        norm = ta.l1
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512, 3981])
        sig_steps = np.array([50000])
        if sym_vf:
            if example >= 0.5:
                N = np.array([1, 2, 3, 4])
            else:
                N = np.array([i for i in range(1, int(1 / example + 3))])
        else:
            N = np.array([1, 2, 3])
        sol_dim = 2
        ref_n = int(n[-1] * 2 ** (1 / example))
        ref_N = N[-1]
        ref_sym_path = False
        ref_sym_vf = True
        param = rough_fractional_Brownian_path(H=example, n=ref_n * int(sig_steps[-1]), dim=2, T=1.,
                                               var_steps=var_steps, norm=norm, save_level=ref_N)
        print('Constructed')
        p = param.p
    elif example == 1:
        param = None
        kind = pure_area
        description = 'pure area'
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
        sig_steps = np.array([10000])
        N = np.array([1, 2, 3])
        norm = ta.l1
        p = 2
        sol_dim = 2
        ref_n = 1000
        ref_N = N[-1]
        ref_sym_path = False
        ref_sym_vf = True
    elif 1 < example <= 2:
        param = int(1 / (example - 1))
        kind = smooth_path
        if param < 15:
            description = 'smooth path'
        else:
            description = 'oscillatory path'
        if sym_path:
            if sym_vf:
                N = np.array([1, 2, 3, 4])
            else:
                N = np.array([1, 2, 3])
            if description == 'oscillatory path':
                n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
            else:
                n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398])
            sig_steps = np.array([1])
            ref_n = n[-1] * 4
            ref_N = N[-1]
        else:
            N = np.array([3])
            n = np.array([300])
            sig_steps = np.array([1, 2, 4, 8, 16, 31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
            ref_n = n[0]
            ref_N = N[0]
        norm = ta.l1
        p = 1
        sol_dim = 2
        ref_sym_path = True
        ref_sym_vf = True
    elif 2 < example <= 3:
        param = int(1 / (example - 2))
        kind = third_level
        description = 'path with significant third level'
        if sym_vf:
            N = np.array([1, 2, 3, 4])
        else:
            N = np.array([1, 2, 3])
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
        sig_steps = np.array([2000])
        norm = ta.l1
        p = 1
        sol_dim = 2
        ref_n = n[-1] * 4
        ref_N = N[-1]
        ref_sym_path = False
        ref_sym_vf = True
    elif example == 4:
        kind = four_dim
        description = 'four-dim fractional Brownian motion'
        var_steps = 15
        norm = ta.l1
        n = np.array([10, 25, 63, 158, 398, 1000])
        sig_steps = np.array([2000])
        if sym_vf:
            N = np.array([1, 2, 3, 4])
        else:
            N = np.array([1, 2, 3])
        sol_dim = 2
        ref_n = int(n[-1] * 2 ** (1 / example))
        ref_N = N[-1]
        ref_sym_path = False
        ref_sym_vf = True
        param = rough_fractional_Brownian_path(H=example, n=ref_n * int(sig_steps[-1]), dim=4, T=1.,
                                               var_steps=var_steps,
                                               norm=norm, save_level=ref_N)
        p = param.p
    elif 4 < example <= 5:
        param = int(1 / (example - 4))
        kind = third_level
        description = 'third level'
        if sym_vf:
            N = np.array([1, 2, 3, 4, 5])
        else:
            N = np.array([1, 2, 3])
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000])
        sig_steps = np.array([1])
        ref_n = n[-1] * 4
        ref_N = N[-1]
        norm = ta.l1
        p = 1
        sol_dim = 2
        ref_sym_path = True
        ref_sym_vf = True
    else:
        return

    directory = 'C:/Users/breneis/Desktop/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots/'
    directory = directory + description

    solutions = np.zeros((len(N), len(n), len(sig_steps), sol_dim))
    error_bounds = np.zeros((len(N), len(n), len(sig_steps)))
    times = np.zeros((len(N), len(n), len(sig_steps)))
    true_errors = np.zeros((len(N), len(n), len(sig_steps)))

    if adaptive_tol:
        atol_ = atol * ref_n ** (-ref_N / float(p))
        rtol_ = rtol * ref_n ** (-ref_N / float(p))
    else:
        atol_ = atol
        rtol_ = rtol
    true_sol, _, _ = kind(n=ref_n, N=ref_N, param=param, plot=False, atol=atol_, rtol=rtol_,
                          sym_path=ref_sym_path, sym_vf=ref_sym_vf)
    plt.plot(true_sol[0, :], true_sol[1, :])
    plt.title('Solution for ' + description)
    if show and sol_dim == 2:
        plt.show()
    if save and sol_dim == 2:
        plt.savefig(directory + '/' + description[0].upper() + description[1:] + ', solution', format='pdf')
    true_sol = true_sol[:, -1]

    # true_sol = np.array([-0.95972823, -0.97375321]) (smooth, k=4)
    # true_sol = np.array([-1.18830677, -0.74708976]) (oscillatory, k=25)
    # true_sol = np.array([-1.23253942, -0.70746147]) (oscillatory, k=36)
    # true_sol = np.array([-1.60472409, -0.46299073]) (pure area)
    print(f'The final value for the problem ' + description + f' is {true_sol}.')

    for i in range(len(N)):
        for j in range(len(n)):
            if adaptive_tol:
                atol_ = atol * n[j] ** (-N[i] / float(p))
                rtol_ = rtol * n[j] ** (-N[i] / float(p))
            else:
                atol_ = atol
                rtol_ = rtol
            for l in range(len(sig_steps)):
                print(f"N = {N[i]}, n = {n[j]}, sig_steps = {sig_steps[l]}")
                for _ in range(rounds):
                    sol, err, tim = kind(n=n[j], N=N[i], param=param, plot=False, sig_steps=sig_steps[l],
                                         atol=atol_, rtol=rtol_, sym_path=sym_path, sym_vf=sym_vf)
                    solutions[i, j, l, :] = sol[:, -1]
                    error_bounds[i, j, l] = err
                    times[i, j, l] += tim
                    true_errors[i, j, l] = norm(sol[:, -1] - true_sol)
                times[i, j, l] /= rounds

            if len(sig_steps) > 1:
                compare_sig_steps(N[i], n[j], sig_steps, true_errors[i, j, :], error_bounds[i, j, :], times[i, j, :],
                                  description, True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
        if len(n) > 1:
            for l in range(len(sig_steps)):
                compare_n(N[i], n, sig_steps[l], true_errors[i, :, l], error_bounds[i, :, l], times[i, :, l],
                          description, True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
    if len(N) > 1:
        compare_N(N, n, sig_steps[-1], true_errors[:, :, -1], error_bounds[:, :, -1], times[:, :, -1], description,
                  False, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
        compare_N(N, n, sig_steps[-1], true_errors[:, :, -1], error_bounds[:, :, -1], times[:, :, -1], description,
                  True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
        if len(sig_steps) > 4:
            compare_N(N, n, sig_steps[-3], true_errors[:, :, -3], error_bounds[:, :, -3], times[:, :, -3], description,
                      False, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
            compare_N(N, n, sig_steps[-3], true_errors[:, :, -3], error_bounds[:, :, -3], times[:, :, -3], description,
                      True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
