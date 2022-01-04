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
        p = 1/H + 0.05
    if save_level == 0:
        save_level = np.ceil(p)
    f = fbm.FBM(n=n, hurst=H, length=T)
    times = fbm.times(n=n, length=T)
    values = np.array([f.fbm() for _ in range(dim)]).T
    return rp.RoughPathDiscrete(times=times, values=values, p=p, var_steps=var_steps, norm=norm, save_level=save_level)


def log_linear_regression(x, y):
    """
    Applies log-linear regression of y against x.
    :param x: The argument
    :param y: The function value
    :return: Exponent, constant, R-value, p-value, empirical standard deviation
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def comparison_plot(N, n, n_steps, true_errors, error_bounds, times, description, regression=True, show=False,
                    save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=False, sym_path=False):
    """
    Constructs many plots for the discussion of the examples.
    :param N: The level(s) of the log-ODE method
    :param n: The number(s) of intervals of the log-ODE method
    :param n_steps: The number(s) of steps used in the approximation of the signatures
    :param true_errors: The errors of the log-ODE approximations
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
        title += 'with adaptive error tolerance'
    else:
        title += 'without adaptive error tolerance'
    if sym_vf:
        title += '\nsymbolic derivatives'
    else:
        title += '\nnumeric derivatives'
    if sym_path:
        title += ', symbolic signatures'
    else:
        title += ', numeric signatures'

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
        if sym_path:
            title += f'\n{n} intervals'
        else:
            title += f'\n{n} intervals, {n_steps} signature steps'
    elif case == 2:
        if sym_path:
            title += f'\ndegree {N}'
        else:
            title += f'\ndegree {N}, {n_steps} signature steps'
    else:
        title += f'\ndegree {N}, {n} intervals'
    filename = title
    filename = filename.replace('\n', ', ')
    if adaptive_tol:
        title += f', a_tol={atol:.2g}' + r'$n^{-N/p}$' + f', r_tol={rtol:.2g}' + r'$n^{-N/p}$'
    else:
        title += f', a_tol={atol:.2g}, r_tol={rtol:.2g}'
    plt.title(description + '\n' + title)
    ax1.set_xlabel(xlabel)
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
        x = rp.RoughPathContinuous(path=path, n_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    if sym_vf:
        a, y, z = sp.symbols('a y z')
        logistic = 1 / (1 + sp.exp(-a))
        f = sp.Array([[z - y, -z], [logistic.subs(a, z), logistic.subs(a, y - 2 * z)]])
        variables = sp.Array([y, z])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=variables)
    else:
        logistic = lambda z: 1 / (1 + np.exp(-z))

        def f(y, dx):
            return np.array([(y[1]-y[0])*dx[0] - y[1]*dx[1], logistic(y[1])*dx[0] + logistic(y[0]-2*y[1])*dx[1]])

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

    path = lambda s, t: ta.NumericTensor([1., np.array([0., 0.]), np.array([[0., t - s], [s - t, 0.]])])
    x = rp.RoughPathExact(path=path, n_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    if sym_vf:
        a, y, z = sp.symbols('a y z')
        logistic = 1 / (1 + sp.exp(-a))
        f = sp.Array([[z - y, -z], [logistic.subs(a, z), logistic.subs(a, y - 2 * z)]])
        variables = sp.Array([y, z])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=variables)
    else:
        logistic = lambda z: 1 / (1 + np.exp(-z))

        def f(y, dx):
            return np.array([(y[1]-y[0])*dx[0] - y[1]*dx[1], logistic(y[1])*dx[0] + logistic(y[0]-2*y[1])*dx[1]])

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
        x = rough_fractional_Brownian_path(H=param, n=n*sig_steps, dim=2, T=1., p=p, var_steps=var_steps, norm=norm,
                                           save_level=N)

    if sym_vf:
        a, y, z = sp.symbols('a y z')
        logistic = 1 / (1 + sp.exp(-a))
        f = sp.Array([[z - y, -z], [logistic.subs(a, z), logistic.subs(a, y - 2 * z)]])
        variables = sp.Array([y, z])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=variables)
    else:
        logistic = lambda z: 1 / (1 + np.exp(-z))

        def f(y, dx):
            return np.array([(y[1]-y[0])*dx[0] - y[1]*dx[1], logistic(y[1])*dx[0] + logistic(y[0]-2*y[1])*dx[1]])

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
        x = rough_fractional_Brownian_path(H=param, n=n*sig_steps, dim=4, T=1., p=p, var_steps=var_steps, norm=norm,
                                           save_level=N)

    if sym_vf:
        a, y, z = sp.symbols('a y z')
        logistic = 1 / (1 + sp.exp(-a))
        f = sp.Array([[z - y, -z, sp.sin(y), sp.cos(z-y/3)],
                      [logistic.subs(a, z), logistic.subs(a, y - 2 * z), sp.sin(y*z)**2, sp.tanh(y*y*z-y)]])
        variables = sp.Array([y, z])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=variables)
    else:
        logistic = lambda z: 1 / (1 + np.exp(-z))

        def f(y, dx):
            return np.array([(y[1]-y[0])*dx[0] - y[1]*dx[1] + np.sin(y[0])*dx[2] + np.cos(y[1]-y[0]/3)*dx[3],
                             logistic(y[1])*dx[0] + logistic(y[0]-2*y[1])*dx[1] + np.sin(y[0]*y[1])**2
                             + np.tanh(y[0]**2 * y[1] - y[0])])

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


def discussion(example, show=False, save=False, rounds=1, adaptive_tol=False, sym_path=False, sym_vf=False, test=False):
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
    :param test: If true, uses a significantly smaller number of parameter values
    :return: The entire path, an error bound, and the time
    """
    # 10000, 15849, 25119, 39811, 63096, 100000
    if 0 < example < 1:
        kind = fBm_path
        description = 'fractional Brownian motion'
        var_steps = 15
        norm = ta.l1
        if test:
            n_vec = np.array([10, 25, 63, 158, 398, 1000])
            n_steps_vec = np.array([1, 6, 40, 251, 1585])
        else:
            n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512])
            n_steps_vec = np.array([1, 3, 6, 16, 40, 100, 251, 631, 1585, 3981, 10000])
        if sym_vf:
            if test:
                N_vec = np.array([1, 2, 3])
            else:
                N_vec = np.array([1, 2, 3, 4])
        else:
            N_vec = np.array([1, 2, 3])
        if adaptive_tol:
            atol = 1e-05
            rtol = 1e-02
        else:
            atol = 1e-09
            rtol = 1e-06
        sol_dim = 2
        if test:
            reference_n = 1585
            reference_N = 3
        else:
            reference_n = 3981
            reference_N = 4
        ref_sym_path = True
        ref_sym_vf = True
        param = rough_fractional_Brownian_path(H=example, n=reference_n*reference_N, dim=2, T=1., var_steps=var_steps,
                                               norm=norm, save_level=reference_N)
        p = param.p
    elif example == 1:
        param = None
        kind = pure_area
        description = 'pure area'
        if test:
            n_vec = np.array([10, 25, 63, 158, 398, 1000])
            n_steps_vec = np.array([1, 6, 40, 251, 1585])
        else:
            n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512])
            n_steps_vec = np.array([1, 3, 6, 16, 40, 100, 251, 631, 1585, 3981, 10000])
        if sym_vf:
            if test:
                N_vec = np.array([1, 2, 3])
            else:
                N_vec = np.array([1, 2, 3, 4])
        else:
            N_vec = np.array([1, 2, 3])
        if adaptive_tol:
            atol = 1e-05
            rtol = 1e-02
        else:
            atol = 1e-09
            rtol = 1e-06
        norm = ta.l1
        p = 2
        sol_dim = 2
        if test:
            reference_n = 1585
            reference_N = 3
        else:
            reference_n = 3981
            reference_N = 4
        ref_sym_path = False
        ref_sym_vf = True
    elif 1 < example <= 2:
        param = int(1 / (example - 1))
        kind = smooth_path
        description = 'smooth path'
        if sym_path:
            if test:
                n_vec = np.array([6, 16, 40, 100, 251])
            else:
                n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
            n_steps_vec = np.array([1])
        else:
            if test:
                n_vec = np.array([10, 25, 63, 158, 398, 1000])
                n_steps_vec = np.array([1, 6, 40, 251, 1585])
            else:
                n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512])
                n_steps_vec = np.array([1, 3, 6, 16, 40, 100, 251, 631, 1585, 3981, 10000])
        if sym_vf:
            if test:
                N_vec = np.array([1, 2, 3])
            else:
                N_vec = np.array([1, 2, 3, 4])
        else:
            N_vec = np.array([1, 2, 3])
        if adaptive_tol:
            atol = 1e-05
            rtol = 1e-02
        else:
            atol = 1e-09
            rtol = 1e-06
        norm = ta.l1
        p = 1
        sol_dim = 2
        if test:
            reference_n = 1585
            reference_N = 3
        else:
            reference_n = 3981
            reference_N = 4
        ref_sym_path = True
        ref_sym_vf = True
    elif example == 4:
        kind = four_dim
        description = 'four-dim fractional Brownian motion'
        var_steps = 15
        norm = ta.l1
        if test:
            n_vec = np.array([10, 25, 63, 158, 398, 1000])
            n_steps_vec = np.array([1, 6, 40, 251, 1585])
        else:
            n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512])
            n_steps_vec = np.array([1, 3, 6, 16, 40, 100, 251, 631, 1585, 3981, 10000])
        if sym_vf:
            if test:
                N_vec = np.array([1, 2, 3])
            else:
                N_vec = np.array([1, 2, 3, 4])
        else:
            N_vec = np.array([1, 2, 3])
        if adaptive_tol:
            atol = 1e-05
            rtol = 1e-02
        else:
            atol = 1e-09
            rtol = 1e-06
        sol_dim = 2
        if test:
            reference_n = 1585
            reference_N = 3
        else:
            reference_n = 3981
            reference_N = 4
        ref_sym_path = True
        ref_sym_vf = True
        param = rough_fractional_Brownian_path(H=example, n=reference_n * reference_N, dim=4, T=1., var_steps=var_steps,
                                               norm=norm, save_level=reference_N)
        p = param.p
    else:
        return

    directory = 'C:/Users/breneis/Desktop/Backup 09112021/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots/'
    directory = directory + description

    solutions = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec), sol_dim))
    error_bounds = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec)))
    times = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec)))
    true_errors = np.zeros((len(N_vec), len(n_vec), len(n_steps_vec)))

    if adaptive_tol:
        atol_ = atol * reference_n ** (-reference_N / float(p))
        rtol_ = rtol * reference_n ** (-reference_N / float(p))
    else:
        atol_ = atol
        rtol_ = rtol
    true_sol, _, _ = kind(n=reference_n, N=reference_N, param=param, plot=False, atol=atol_, rtol=rtol_,
                          sym_path=ref_sym_path, sym_vf=ref_sym_vf)
    plt.plot(true_sol[0, :], true_sol[1, :])
    plt.title('Solution for ' + description)
    if show and sol_dim == 2:
        plt.show()
    if save and sol_dim == 2:
        plt.savefig(directory + '/' + description[0].upper() + description[1:] + ', solution', format='pdf')
    true_sol = true_sol[:, -1]
    print(f'The final value for the problem ' + description + f' is {true_sol}.')

    for i in range(len(N_vec)):
        for j in range(len(n_vec)):
            if adaptive_tol:
                atol_ = atol * n_vec[j] ** (-N_vec[i] / float(p))
                rtol_ = rtol * n_vec[j] ** (-N_vec[i] / float(p))
            else:
                atol_ = atol
                rtol_ = rtol
            for l in range(len(n_steps_vec)):
                print(f"N = {N_vec[i]}, n = {n_vec[j]}, n_steps = {n_steps_vec[l]}")
                for _ in range(rounds):
                    sol, err, tim = kind(n=n_vec[j], N=N_vec[i], param=param, plot=False, sig_steps=n_steps_vec[l],
                                         atol=atol_, rtol=rtol_)
                    solutions[i, j, l, :] = sol[:, -1]
                    error_bounds[i, j, l] = err
                    times[i, j, l] += tim
                    true_errors[i, j, l] = norm(sol[:, -1] - true_sol)
                times[i, j, l] /= rounds

            comparison_plot(N_vec[i], n_vec[j], n_steps_vec, true_errors[i, j, :], error_bounds[i, j, :],
                            times[i, j, :], description, True, show, save, directory, adaptive_tol, atol, rtol, sym_vf,
                            sym_path)
        if not sym_path:
            for l in range(len(n_steps_vec)):
                comparison_plot(N_vec[i], n_vec, n_steps_vec[l], true_errors[i, :, l], error_bounds[i, :, l],
                                times[i, :, l], description, True, show, save, directory, adaptive_tol, atol, rtol,
                                sym_vf, sym_path)
    if not sym_path:
        for j in range(len(n_vec)):
            for l in range(len(n_steps_vec)):
                comparison_plot(N_vec, n_vec[j], n_steps_vec[l], true_errors[:, j, l], error_bounds[:, j, l],
                                times[:, j, l], description, True, show, save, directory, adaptive_tol, atol, rtol,
                                sym_vf, sym_path)

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
            print(
                f'In {times[index]:.3g} s, one can achieve an error of {err:.3g} by choosing N={N_vec[index[0]]}, n={n_vec[index[1]]}, n_steps={n_steps_vec[2]}. The corresponding error bound is {error_bounds[index]:.3g}.')
            current_error = err
            opt_times.append(times[index])
            opt_N.append(N_vec[index[0]])
            opt_n.append(n_vec[index[1]])
            opt_n_steps.append(n_steps_vec[index[2]])
            opt_err.append(err)
            opt_bounds.append(error_bounds[index])
    plt.loglog(opt_times, opt_err, 'r', label='True error')
    plt.loglog(opt_times, opt_bounds, 'g', label='Error bound')
    description = description[0].upper() + description[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance'
    else:
        title += 'without adaptive error tolerance'
    if sym_vf:
        title += ', with symbolic derivatives'
    else:
        title += ', with numeric derivatives'
    if sym_path:
        title += ', symbolic signatures'
    else:
        title += ', numeric signagures'
    plt.title(description + '\n' + title)
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend(loc='best')
    plt.show()
