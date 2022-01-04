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


def log_linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.log(x), np.log(y))
    return slope, np.exp(intercept), r_value, p_value, std_err


def comparison_plot(N, n, n_steps, true_errors, error_bounds, times, description, regression=True, show=False,
                    save=False, dir=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=False, sym_path=False):
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
        plt.savefig(dir + '/' + description + ', ' + filename, format='pdf')


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
    if 0 < example < 1:
        return
    elif example == 1:
        return
    elif 1 < example <= 2:
        param = int(1 / (example - 1))
        kind = smooth_path
        description = 'smooth path'
        if sym_path:
            n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
            n_steps_vec = np.array([1])
        else:
            n_vec = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512])
            n_steps_vec = np.array([1, 3, 6, 16, 40, 100, 251, 631, 1585, 3981, 10000])
        if sym_vf:
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
        reference_n = 3981
        reference_N = 4
        ref_sym_path = True
        ref_sym_vf = True
    elif example == 4:
        return
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
