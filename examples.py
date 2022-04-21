import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats, interpolate
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
    :return: An instance of a RoughPath
    """
    if p == 0.:
        p = 1 / H + 0.05
        if p - int(p) > 0.95:
            p = np.ceil(p)
    if save_level == 0:
        save_level = int(np.ceil(p))
    f = fbm.FBM(n=n, hurst=H, length=T)
    times = fbm.times(n=n, length=T)
    values = np.array([f.fbm() for _ in range(dim)])
    if save_level <= 5:
        return rp.RoughPathDiscrete(times=times, values=values.T, p=p, var_steps=var_steps, norm=norm,
                                    save_level=save_level)
    return rp.RoughPathContinuous(path=scipy.interpolate.interp1d(times, values, axis=-1),
                                  sig_steps=int(max(15, n / 1000)), p=p, var_steps=var_steps, norm=norm)


def rough_fractional_Brownian_path_time(H, n, dim, T=1., p=0., var_steps=15, norm=ta.l1, save_level=0):
    """
    Returns a sample path of fractional Brownian motion together with a time component as an instance of a RoughPath.
    Time is the first component.
    :param H: Hurst parameter
    :param n: Number of equidistant evaluation points
    :param dim: Dimension of the path
    :param T: Final time
    :param p: Parameter in which the roughness should be measured (default choice 1/H + 0.05)
    :param var_steps: Number of steps used in approximating p-variation
    :param norm: Norm for computing p-variation
    :param save_level: Level up to which the signatures on the time grid are stored
    :return: An instance of a RoughPath
    """
    if p == 0.:
        p = 1 / H + 0.05
        if p - int(p) > 0.95:
            p = np.ceil(p)
    if save_level == 0:
        save_level = int(np.ceil(p))
    f = fbm.FBM(n=n, hurst=H, length=T)
    times = fbm.times(n=n, length=T)
    values = np.array([times] + [f.fbm() for _ in range(dim)])
    if save_level <= 5:
        return rp.RoughPathDiscrete(times=times, values=values.T, p=p, var_steps=var_steps, norm=norm,
                                    save_level=save_level)
    return rp.RoughPathContinuous(path=scipy.interpolate.interp1d(times, values, axis=-1),
                                  sig_steps=int(max(15, n / 1000)), p=p, var_steps=var_steps, norm=norm)


def asymmetric_Brownian_path(n, T=1., q=0.5):
    """
    Returns a sample path of an asymmetric Brownian motion.
    :param n: Number of equidistant evaluation points
    :param T: Final time
    :param q: Probability of reflecting a negative area
    :return: An instance of a RoughPath
    """
    n = int(2**(int(np.ceil(np.log2(n)))))
    log_n = int(np.round(np.log2(n)))
    values = np.empty(n+1)
    values[0] = 0
    values[1] = np.random.normal(0, np.sqrt(T))
    n_step = n
    for i in range(1, log_n+1):
        n_step = int(n_step/2)
        for j in range(int(2**(i-1))):
            bridge = np.random.normal(0, np.sqrt(T)/2**i)
            if bridge < 0:
                if np.random.random(1) > q:
                    bridge = -bridge
            values[n_step*(2*j+1)] = (values[n_step*2*j] + values[n_step*2*(j+1)])/2 + bridge
    return values, n


def rough_asymmetric_Brownian_path_time(n, dim, T=1., q=0.5, p=0., var_steps=15, norm=ta.l1, save_level=0):
    """
    Returns a sample path of an asymmetric Brownian motion as an instance of a RoughPath.
    :param n: Number of equidistant evaluation points
    :param dim: Dimension of the path
    :param T: Final time
    :param q: Probability of reflecting a negative area
    :param p: Parameter in which the roughness should be measured (default choice 1/H + 0.05)
    :param var_steps: Number of steps used in approximating p-variation
    :param norm: Norm for computing p-variation
    :param save_level: Level up to which the signatures on the time grid are stored
    :return: An instance of a RoughPath
    """
    if p == 0.:
        p = 2.05
    if save_level == 0:
        save_level = int(np.ceil(p))
    val, n = asymmetric_Brownian_path(n, T, q)
    times = np.linspace(0, T, n+1)
    values = np.empty((dim+1, n+1))
    values[1, :] = val
    for i in range(2, dim+1):
        values[i, :] = asymmetric_Brownian_path(n, T, q)[0]
    values[0, :] = times
    if save_level <= 5:
        return rp.RoughPathDiscrete(times=times, values=values.T, p=p, var_steps=var_steps, norm=norm,
                                    save_level=save_level)
    return rp.RoughPathContinuous(path=scipy.interpolate.interp1d(times, values, axis=-1),
                                  sig_steps=int(np.fmax(15, n / 1000)), p=p, var_steps=var_steps, norm=norm)


def time_rough_path(symbolic=True, norm=ta.l1, p=1., var_steps=15, N=1, sig_steps=2000):
    if symbolic:
        t = sp.symbols('t')
        path = sp.Array([t])
        x = rp.RoughPathSymbolic(path=path, t=t, p=p, var_steps=var_steps, norm=norm)
        if N > 1:
            for _ in range(1, N):
                x.new_level()
    else:
        path = lambda t: np.array([t])
        x = rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)
    return x


def unit_circle(symbolic=True, param=4, norm=ta.l1, p=1., var_steps=15, N=1, sig_steps=2000):
    if symbolic:
        t = sp.symbols('t')
        path = sp.Array(
            [sp.cos(2 * param * sp.pi * t) / sp.sqrt(param), sp.sin(2 * param * sp.pi * t) / sp.sqrt(param)])
        x = rp.RoughPathSymbolic(path=path, t=t, p=p, var_steps=var_steps, norm=norm)
        if N > 1:
            for _ in range(1, N):
                x.new_level()
    else:
        path = lambda t: np.array([np.cos(2 * np.pi * param * t), np.sin(2 * np.pi * param * t)]) / np.sqrt(param)
        x = rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    return x


def wiggly_sinh(symbolic=True, param=4, norm=ta.l1, p=1., var_steps=15, N=1, sig_steps=2000):
    if symbolic:
        param = sp.Integer(param)
        t = sp.symbols('t')
        path = sp.Array([sp.cos(2 * sp.pi * param * t), sp.sinh(t-sp.Rational(1, 2))])
        x = rp.RoughPathSymbolic(path=path, t=t, p=p, var_steps=var_steps, norm=norm)
        for _ in range(2, N+1):
            x.new_level()
    else:
        path = lambda t: np.array([np.cos(2 * np.pi * param * t), np.sinh(t-1/2)])
        x = rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)
    return x


def pure_area(norm=ta.l1, p=2., var_steps=15, sig_steps=2000):
    path = lambda s, t: ta.NumericTensor(
        [1., np.array([0., 0.]), np.array([[0., np.pi * (t - s)], [np.pi * (s - t), 0.]])])
    return rp.RoughPathExact(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)


def third_level_path(param=4, norm=ta.l1, p=1., var_steps=15, sig_steps=2000):
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

    return rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)


def linear_1x1_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y = sp.symbols('y')
        f = sp.Array([[y]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([y[0]*x[0]])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=1, dim_y=1, h=h, norm=norm)

    return vec_field


def smooth_2x2_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z], [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z)))]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                                   1 / (1 + np.exp(-y[1])) * x[0] + 1 / (1 + np.exp(-(y[0] - 2 * y[1]))) * x[1]])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=2, dim_y=2, h=h, norm=norm)

    return vec_field


def linear_nilpotent_3x2_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        a, b, c = sp.symbols('a b c')
        f = sp.Array([[a+b, a], [b, b+c], [c, c]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[a, b, c])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([(y[0] + y[1]) * x[0] + y[0] * x[1], y[1] * x[0] + (y[1] + y[2]) * x[1],
                                   y[2] * (x[0] + x[1])])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=2, dim_y=3, h=h, norm=norm)
    return vec_field


def smooth_2x3_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z, sp.sin(y * y)],
                      [1 / (1 + sp.exp(-z)), 1 / (1 + sp.exp(-(y - 2 * z))), sp.tanh(y * z) * sp.cos(y)]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array(
            [(y[1] - y[0]) * x[0] - y[1] * x[1] + np.sin(y[0] * y[0]) * x[2],
             1 / (1 - np.exp(y[1])) * x[0] + 1 / (1 - np.exp(y[0] - 2 * y[1])) * x[1] + np.tanh(y[1] * y[0]) * np.cos(
                 y[0]) * x[2]])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=3, dim_y=2, h=h, norm=norm)
    return vec_field


def langevin_banana_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y, z = sp.symbols('y z')
        f = sp.Array([[-4*y*(y**2-z), 1, 0], [2*(y**2-z), 0, 1]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([(-4*y[0]*(y[0]**2-y[1])) * x[0] + x[1],
                                   2*(y[0]**2-y[1]) * x[0] + x[2]])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=3, dim_y=2, h=h, norm=norm)
    return vec_field


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
            if error_bounds is not None:
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
    if error_bounds is not None:
        ax1.loglog(n, error_bounds, color='g', label='Error bound')
    reg_index = np.sum(n < np.ones(len(n)) * 35)
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(n[reg_index:], errors[reg_index:])
        ax1.loglog(n, constant * n ** power, 'r--')
        x_label += '\n\nError ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
            '{:.3g}'.format(power)) + '  ' + r'$(R=$' + f'{r_value:.3g}' + r'$)$'
        if error_bounds is not None:
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


def compare_sig_steps(N, n, sig_steps, errors, times, description, regression=True, show=False,
                      save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=False, sym_path=False):
    """
    Constructs many plots for the discussion of the examples.
    :param N: The level(s) of the log-ODE method
    :param n: The number(s) of intervals of the log-ODE method
    :param sig_steps: The number(s) of steps used in the approximation of the signatures
    :param errors: The errors of the log-ODE approximations
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
    reg_index = np.sum(sig_steps < np.ones(len(sig_steps)) * 35)
    if regression:
        power, constant, r_value, p_value, std_err = log_linear_regression(sig_steps[reg_index:], errors[reg_index:])
        ax1.loglog(sig_steps, constant * sig_steps ** power, 'r--')
        x_label += '\n\nError ' + r'$\approx$' + f' {constant:.3g}' + r'$x^{{{}}}$'.format(
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


def apply_example(x, f, y_0, solver='fsss', N=2, partition=None, plot=False, atol=1e-09, rtol=1e-06, N_sol=2,
                  compute_bound=False, method='RK45', T=1., N_min=1, N_max=4, n_min=10, n_max=40, speed=-1):
    tic = time.perf_counter()
    output_1, output_2 = lo.solve(x, f, y_0, solver, N=N, T=T, partition=partition, atol=atol, rtol=rtol, method=method,
                                  compute_bound=compute_bound, N_sol=N_sol, N_min=N_min, N_max=N_max, n_min=n_min,
                                  n_max=n_max, speed=speed)
    toc = time.perf_counter()

    if plot:
        if solver[0] == 'f':
            solution = output_1
        else:
            solution = output_2
        plt.plot(solution[0, :], solution[1, :])
        plt.show()

    return output_1, output_2, toc - tic


def ex_nilpotent_vf(solver='fsss', n=100, N=2, param=4, plot=False, atol=1e-09, rtol=1e-06, N_sol=2,
                    compute_bound=False, sig_steps=100, p=1, var_steps=15, norm=ta.l1, method='RK45', T=1.,
                    sym_path=True, sym_vf=True, h=1e-07, N_min=1, N_max=4, n_min=10, n_max=40, x=None, f=None,
                    speed=-1):
    partition = np.linspace(0, T, n + 1)
    if x is None:
        x = wiggly_sinh(symbolic=sym_path, param=param, norm=norm, p=p, var_steps=var_steps, N=N, sig_steps=sig_steps)
    if f is None:
        f = linear_nilpotent_3x2_vector_field(symbolic=sym_vf, norm=norm, N=N, h=h)
    y_0 = np.array([1., 1., 1.])
    return apply_example(x=x, f=f, y_0=y_0, solver=solver, N=N, partition=partition, plot=plot, atol=atol, rtol=rtol,
                         N_sol=N_sol, compute_bound=compute_bound, method=method, T=T, N_min=N_min, N_max=N_max,
                         n_min=n_min, n_max=n_max, speed=speed)


def ex_smooth_path(solver='fsss', n=100, N=2, param=4, plot=False, atol=1e-09, rtol=1e-06, N_sol=2,
                   compute_bound=False, sig_steps=100, p=1, var_steps=15, norm=ta.l1, method='RK45', T=1.,
                   sym_path=True, sym_vf=True, h=1e-07, N_min=1, N_max=5, n_min=10, n_max=40, x=None, f=None, speed=-1):
    partition = np.linspace(0, 1, n + 1)
    if x is None:
        x = unit_circle(symbolic=sym_path, param=param, norm=norm, p=p, var_steps=var_steps, N=N, sig_steps=sig_steps)
    if f is None:
        f = smooth_2x2_vector_field(symbolic=sym_vf, norm=norm, N=N, h=h)
    y_0 = np.array([0., 0.])
    return apply_example(x=x, f=f, y_0=y_0, solver=solver, N=N, partition=partition, plot=plot, atol=atol, rtol=rtol,
                         N_sol=N_sol, compute_bound=compute_bound, method=method, T=T, N_min=N_min, N_max=N_max,
                         n_min=n_min, n_max=n_max, speed=speed)


def ex_pure_area(solver='fsss', n=100, N=2, plot=False, atol=1e-09, rtol=1e-06, N_sol=2, compute_bound=False,
                 sig_steps=100, p=2, var_steps=15, norm=ta.l1, method='RK45', T=1., sym_vf=True, h=1e-07, N_min=1,
                 N_max=5, n_min=10, n_max=40, x=None, f=None, speed=-1):
    partition = np.linspace(0, 1, n + 1)
    if x is None:
        x = pure_area(norm=norm, p=p, var_steps=var_steps, sig_steps=sig_steps)
    if f is None:
        f = smooth_2x2_vector_field(symbolic=sym_vf, norm=norm, N=N, h=h)
    y_0 = np.array([0., 0.])
    return apply_example(x=x, f=f, y_0=y_0, solver=solver, N=N, partition=partition, plot=plot, atol=atol, rtol=rtol,
                         N_sol=N_sol, compute_bound=compute_bound, method=method, T=T, N_min=N_min, N_max=N_max,
                         n_min=n_min, n_max=n_max, speed=speed)


def ex_third_level(solver='fsss', n=100, N=2, param=4, plot=False, atol=1e-09, rtol=1e-06, N_sol=2,
                   compute_bound=False, sig_steps=100, p=1, var_steps=15, norm=ta.l1, method='RK45', T=1., sym_vf=True,
                   h=1e-07, N_min=1, N_max=5, n_min=10, n_max=40, x=None, f=None, speed=-1):
    partition = np.linspace(0, 1, n + 1)
    if x is None:
        x = third_level_path(param=param, norm=norm, p=p, var_steps=var_steps, sig_steps=sig_steps)
    if f is None:
        f = smooth_2x3_vector_field(symbolic=sym_vf, norm=norm, N=N, h=h)
    y_0 = np.array([0., 0.])
    return apply_example(x=x, f=f, y_0=y_0, solver=solver, N=N, partition=partition, plot=plot, atol=atol, rtol=rtol,
                         N_sol=N_sol, compute_bound=compute_bound, method=method, T=T, N_min=N_min, N_max=N_max,
                         n_min=n_min, n_max=n_max, speed=speed)


def ex_fBm_path(solver='fsss', n=100, N=2, param=4, plot=False, atol=1e-09, rtol=1e-06, N_sol=2, compute_bound=False,
                sig_steps=100, var_steps=15, norm=ta.l1, method='RK45', T=1., sym_vf=True, h=1e-07, N_min=1, N_max=5,
                n_min=10, n_max=40, x=None, f=None, speed=-1):
    partition = np.linspace(0, 1, n + 1)
    if x is None:
        x = rough_fractional_Brownian_path(H=param, n=n * sig_steps, dim=2, T=1., p=1 / param + 0.1, var_steps=var_steps,
                                           norm=norm, save_level=N)
    if f is None:
        f = smooth_2x2_vector_field(symbolic=sym_vf, norm=norm, N=N, h=h)
    y_0 = np.array([0., 0.])
    return apply_example(x=x, f=f, y_0=y_0, solver=solver, N=N, partition=partition, plot=plot, atol=atol, rtol=rtol,
                         N_sol=N_sol, compute_bound=compute_bound, method=method, T=T, N_min=N_min, N_max=N_max,
                         n_min=n_min, n_max=n_max, speed=speed)


def ex_langevin_banana(solver='fsss', n=100, N=2, param=4, plot=False, atol=1e-09, rtol=1e-06, N_sol=2,
                       compute_bound=False, sig_steps=100, var_steps=15, norm=ta.l1, method='RK45', T=1., sym_vf=True,
                       h=1e-07, N_min=1, N_max=5, n_min=10, n_max=40, x=None, f=None, speed=-1):
    partition = np.linspace(0, 1, n + 1)
    if x is None:
        x = rough_fractional_Brownian_path_time(H=param, n=n * sig_steps, dim=2, T=1., p=1 / param + 0.1,
                                                var_steps=var_steps, norm=norm, save_level=N)
    if f is None:
        f = langevin_banana_vector_field(symbolic=sym_vf, norm=norm, N=N, h=h)
    y_0 = np.array([0., 0.])
    return apply_example(x=x, f=f, y_0=y_0, solver=solver, N=N, partition=partition, plot=plot, atol=atol, rtol=rtol,
                         N_sol=N_sol, compute_bound=compute_bound, method=method, T=T, N_min=N_min, N_max=N_max,
                         n_min=n_min, n_max=n_max, speed=speed)


def discussion(example, show=False, save=False, rounds=1, solver='fsss', N_sol=2):
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
    :param full: If 0, solves the ordinary (not full) RDE. If 1, solves the full RDE. If 2, solves the full RDE with
        the inaccurate but faster algorithm
    :param N_sol: Only relevant if full == 1 or full == 2. Is the level of the full solution that is computed
    :return: None
    """
    # 10000, 15849, 25119, 39811, 63096, 100000
    if solver[0] == 'f':
        adaptive = False
    else:
        adaptive = True
    if solver[1] == 'f':
        full = True
    else:
        full = False
    if solver[2] == 'a':
        adjoined = True
    else:
        adjoined = False

    if adaptive:
        atol = 1e-04
        rtol = 1e-02
    else:
        atol = 1e-09
        rtol = 1e-06
    adaptive_tol = False
    T = 1

    if example == 0:
        param = 4
        kind = ex_nilpotent_vf
        description = 'nilpotent vector field'
        N = np.array([1, 2, 3])
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398])
        sig_steps = np.array([1])
        ref_n = n[-1] * 4
        ref_N = N[-1]
        sol_dim = 3
        norm = ta.l1
        sym_vf = True
        sym_path = True
    elif 0 < example < 1:
        kind = ex_fBm_path
        description = f'fractional Brownian motion, H={example}'
        var_steps = 15
        norm = ta.l1
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585, 2512, 3981])
        sig_steps = np.array([50000])
        if example >= 0.5:
            N = np.array([1, 2, 3, 4])
        else:
            N = np.array([i for i in range(1, int(1 / example + 3))])
        sol_dim = 2
        ref_n = int(n[-1] * 2 ** (1 / example))
        ref_N = N[-1]
        param = rough_fractional_Brownian_path(H=example, n=ref_n * int(sig_steps[-1]), dim=2, T=1.,
                                               var_steps=var_steps, norm=norm, save_level=ref_N)
        print('Constructed')
        p = param.p
        sym_vf = True
        sym_path = False
    elif example == 1:
        param = None
        kind = ex_pure_area
        description = 'pure area'
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
        sig_steps = np.array([10000])
        N = np.array([1, 2, 3])
        norm = ta.l1
        p = 2
        sol_dim = 2
        ref_n = 1000
        ref_N = N[-1]
        sym_vf = True
        sym_path = False
    elif 1 < example <= 2:
        param = int(1 / (example - 1))
        kind = ex_smooth_path
        if param < 15:
            description = 'smooth path'
        else:
            description = 'oscillatory path'
        N = np.array([1, 2, 3, 4])
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
        sig_steps = np.array([1])
        norm = ta.l1
        p = 1
        sol_dim = 2
        ref_n = n[-1] * 4
        ref_N = N[-1]
        sym_vf = True
        sym_path = True
    elif 2 < example <= 3:
        param = int(1 / (example - 2))
        kind = ex_third_level
        description = 'third level'
        N = np.array([1, 2, 3, 4])
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
        sig_steps = np.array([5000])
        norm = ta.l1
        p = 1
        sol_dim = 2
        ref_n = n[-1] * 4
        ref_N = N[-1]
        sym_vf = True
        sym_path = False
    elif 5 <= example <= 6:
        kind = ex_langevin_banana
        description = f'Langevin with banana potential, q={example-5}'
        var_steps = 15
        norm = ta.l1
        n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000])
        sig_steps = np.array([2000])
        N = np.array([1, 2, 3, 4])
        sol_dim = 2
        ref_n = int(n[-1] * 4)
        ref_N = N[-1]
        param = rough_asymmetric_Brownian_path_time(q=example-5, n=ref_n * int(sig_steps[-1]), dim=2, T=10.,
                                               var_steps=var_steps, norm=norm, save_level=ref_N)
        print('Constructed')
        p = param.p
        sym_vf = True
        sym_path = False
    else:
        return

    directory = 'C:/Users/breneis/Desktop/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots/'
    directory = directory + description

    solutions = np.zeros((len(N), len(n), len(sig_steps), sol_dim))
    error_bounds = np.zeros((len(N), len(n), len(sig_steps)))
    times = np.zeros((len(N), len(n), len(sig_steps)))
    true_errors = np.zeros((len(N), len(n), len(sig_steps)))

    output_1, output_2, _ = kind(n=ref_n, N=ref_N, plot=False, atol=atol, rtol=rtol, solver=solver, N_sol=N_sol)
    if adaptive:
        true_sol = output_2
    else:
        true_sol = output_1
    if not full:
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
                for l in range(len(sig_steps)):
                    print(f"N = {N[i]}, n = {n[j]}, sig_steps = {sig_steps[l]}")
                    for _ in range(rounds):
                        sol, err, tim = kind(n=n[j], N=N[i], param=param, plot=False, sig_steps=sig_steps[l],
                                             atol=atol, rtol=rtol, solver=solver, N_sol=N_sol)
                        solutions[i, j, l, :] = sol[:, -1]
                        error_bounds[i, j, l] = err
                        times[i, j, l] += tim
                        true_errors[i, j, l] = norm(sol[:, -1] - true_sol)
                    times[i, j, l] /= rounds

                if len(sig_steps) > 1:
                    compare_sig_steps(N[i], n[j], sig_steps, true_errors[i, j, :], times[i, j, :],
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

    elif full:
        true_sol_path = np.array([true_sol.sig(0, i/200., 1)[1] for i in range(201)])
        plt.plot(true_sol_path[:, 0], true_sol_path[:, 1])
        plt.show()
        true_sol = true_sol.sig(0, T, N_sol).to_array()
        solutions = np.zeros((len(N), len(n), len(sig_steps), len(true_sol)))

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
                        sol, _, tim = kind(n=n[j], N=N[i], param=param, plot=False, sig_steps=sig_steps[l], atol=atol_,
                                           rtol=rtol_, sym_path=sym_path, sym_vf=sym_vf, solver=solver, N_sol=N_sol)
                        solutions[i, j, l, :] = sol.sig(0, T, N_sol).to_array()
                        times[i, j, l] += tim
                        true_errors[i, j, l] = norm(solutions[i, j, l, :] - true_sol)
                    times[i, j, l] /= rounds

                if len(sig_steps) > 1:
                    compare_sig_steps(N[i], n[j], sig_steps, true_errors[i, j, :], times[i, j, :], description,
                                      True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
            if len(n) > 1:
                for l in range(len(sig_steps)):
                    compare_n(N[i], n, sig_steps[l], true_errors[i, :, l], None, times[i, :, l],
                              description, True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
        if len(N) > 1:
            compare_N(N, n, sig_steps[-1], true_errors[:, :, -1], None, times[:, :, -1], description,
                      False, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
            compare_N(N, n, sig_steps[-1], true_errors[:, :, -1], None, times[:, :, -1], description,
                      True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
            if len(sig_steps) > 4:
                compare_N(N, n, sig_steps[-3], true_errors[:, :, -3], None, times[:, :, -3], description,
                          False, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
                compare_N(N, n, sig_steps[-3], true_errors[:, :, -3], None, times[:, :, -3], description,
                          True, show, save, directory, adaptive_tol, atol, rtol, sym_vf, sym_path)
