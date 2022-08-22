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
import brownianroughtree as brt


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


def smooth_3d_path(symbolic=True, param=1296, norm=ta.l1, p=1, var_steps=15, N=1, sig_steps=2000):
    if symbolic:
        t = sp.symbols('t')
        path = sp.Array([sp.cos(2*param*sp.pi*t)/sp.sqrt(param) + t, sp.sin(2*param*sp.pi*t)/sp.sqrt(param) + 2*t,
                         sp.sin(2*param*sp.pi*t + sp.pi/4)/sp.sqrt(param) - t])
        x = rp.RoughPathSymbolic(path=path, t=t, p=p, var_steps=var_steps, norm=norm)
        if N > 1:
            for _ in range(1, N):
                x.new_level()
    else:
        path = lambda t: np.array([np.cos(2*param*np.pi*t), np.sin(2*param*np.pi*t), np.sin(2*param*np.pi*t+np.pi/4)]) / np.sqrt(param) + np.array([1, 2, -1])*t
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


def smooth_fractional_path(H, n, p=0., var_steps=15, norm=ta.l1, save_level=0):
    if p == 0.:
        p = 1 / H + 0.05
        if p - int(p) > 0.95:
            p = np.ceil(p)
    if save_level == 0:
        save_level = int(np.ceil(p))

    times = np.linspace(0, 1, n+1)
    cutoff_1 = int(np.around(n/4))
    cutoff_2 = int(np.around(3*n/4))
    values = np.zeros((2, n+1))

    values[0, :cutoff_1+1] = np.cos(8*np.pi*times[:cutoff_1+1])/2
    values[1, :cutoff_1+1] = np.sin(8*np.pi*times[:cutoff_1+1])/2

    f = fbm.FBM(n=cutoff_2-cutoff_1, hurst=H, length=0.5)
    values[0, cutoff_1:cutoff_2+1] = f.fbm() + values[0, cutoff_1]
    values[1, cutoff_1:cutoff_2+1] = f.fbm() + values[1, cutoff_1]

    values[0, cutoff_2:] = np.cos(8*np.pi*times[cutoff_2:])/2 + values[0, cutoff_2]
    values[1, cutoff_2:] = np.sin(8*np.pi*times[cutoff_2:])/2 + values[1, cutoff_2]

    if save_level <= 5:
        return rp.RoughPathDiscrete(times=times, values=values.T, p=p, var_steps=var_steps, norm=norm,
                                    save_level=save_level)
    return rp.RoughPathContinuous(path=scipy.interpolate.interp1d(times, values, axis=-1),
                                  sig_steps=int(max(15, n / 1000)), p=p, var_steps=var_steps, norm=norm)


def smooth_path_singularity(symbolic=True, norm=ta.l1, p=1., var_steps=15, N=1, sig_steps=2000):
    if symbolic:
        t = sp.symbols('t')
        path = sp.Array([sp.Integer(1) / (sp.Integer(5000)*(t-sp.Rational(1, 2))*(t-sp.Rational(1, 2)) + sp.Integer(1)),
                         t])
        x = rp.RoughPathSymbolic(path=path, t=t, p=p, var_steps=var_steps, norm=norm)
        if N > 1:
            for _ in range(1, N):
                print('here')
                x.new_level()
    else:
        path = lambda t: np.array([1 / (5000*(t-0.5)**2 + 1), t])
        x = rp.RoughPathContinuous(path=path, sig_steps=sig_steps, p=p, var_steps=var_steps, norm=norm)

    return x


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


def linear_2x3_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y, z = sp.symbols('y z')
        f = sp.Array([[y-z, y, z-y], [2*z, y+z, 2*y-z]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([(y[0]-y[1])*x[0] + y[0]*x[1] + (y[1]-y[0])*x[2],
                                   2*y[1]*x[0] + (y[0] + y[1])*x[1] + (2*y[0] - y[1])*x[2]])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=3, dim_y=2, h=h, norm=norm)
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


def simple_smooth_2x2_vector_field(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z], [sp.tanh(-z), sp.cos(-(y - 2 * z))]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                                   np.tanh(-y[1]) * x[0] + np.cos(-(y[0] - 2 * y[1])) * x[1]])
        vec_field = vf.VectorFieldNumeric(f=[f], dim_x=2, dim_y=2, h=h, norm=norm)

    return vec_field


def smooth_2x2_vector_field_singularity(symbolic=True, norm=ta.l1, N=1, h=1e-07):
    if symbolic:
        y, z = sp.symbols('y z')
        f = sp.Array([[z - y, -z], [1 + 20/(1000*(y+1)*(y+1)+1), 20/(1000*(z+1)*(z+1)+1)]])
        vec_field = vf.VectorFieldSymbolic(f=[f], norm=norm, variables=[y, z])
        if N > 1:
            for _ in range(1, N):
                vec_field.new_derivative()
    else:
        f = lambda y, x: np.array([(y[1] - y[0]) * x[0] - y[1] * x[1],
                                   np.tanh(-y[1]) * x[0] + np.cos(-(y[0] - 2 * y[1])) * x[1]])
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
              save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=None, sym_path=None):
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
    if np.amin(error_bounds) <= 0:
        error_bounds = None
    description = description[0].upper() + description[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance, '
    if sym_vf is None:
        pass
    elif sym_vf:
        title += 'symbolic derivatives'
    else:
        title += 'numeric derivatives'
    if sym_path is None:
        pass
    elif sym_path:
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
              save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=None, sym_path=None):
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
    if np.amin(error_bounds) <= 0:
        error_bounds = None
    description = description[0].upper() + description[1:]
    title = ''
    if adaptive_tol:
        title += 'with adaptive error tolerance, '
    if sym_vf is None:
        pass
    elif sym_vf:
        title += 'symbolic derivatives'
    else:
        title += 'numeric derivatives'
    if sym_path is None:
        pass
    elif sym_path:
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
                      save=False, directory=None, adaptive_tol=False, atol=0., rtol=0., sym_vf=None, sym_path=None):
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
    if sym_vf is None:
        pass
    elif sym_vf:
        title += 'symbolic derivatives'
    else:
        title += 'numeric derivatives'
    if sym_path is None:
        pass
    elif sym_path:
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


def compare_n_error_representation(N, n, true_errors, estimated_errors, error_differences, description, show=False,
                                   save=False, directory=None):
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

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    x_label = 'Number of intervals'
    ax1.set_ylabel('Error')
    ax2.set_ylabel('Estimated error / error')  # we already handled the x-label with ax1

    ax1.loglog(n, true_errors, color='r', label='Error')
    ax1.loglog(n, estimated_errors, color='g', label='Estimated error')
    ax1.loglog(n, error_differences, color='c', label='Error after correction')

    fraction = estimated_errors / true_errors
    ax2.loglog(n, fraction, color='b', label='Estimated error / error')
    ax2.set_ylim(min(1/2, np.amin(fraction)), max(2, np.amax(fraction)))

    ax1.set_xlabel(x_label)
    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    specifics = f'Degree {N}'
    filename = 'error representation, ' + specifics
    plt.title(description + '\n' + specifics)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if show:
        plt.show()
    if save:
        plt.savefig(directory + '/' + description + ', ' + filename, format='pdf')


def apply_example(x, f, y_0, solver='fsss', N=2, partition=None, plot=False, atol=1e-09, rtol=1e-06, N_sol=2,
                  compute_bound=False, method='RK45', T=1., N_min=1, N_max=4, n=20, speed=-1):
    tic = time.perf_counter()
    if solver[0] == 'a':
        partition, y, loc_err = lo.solve(x, f, y_0, solver, N=N, T=T, partition=partition, atol=atol, rtol=rtol,
                                         method=method, compute_bound=compute_bound, N_sol=N_sol, N_min=N_min,
                                         N_max=N_max, n=n, speed=speed)
        toc = time.perf_counter()
        if plot:
            plt.plot(y[0, :], y[1, :])
            plt.show()
        return partition, y, loc_err, toc-tic
    elif solver[0] == 'f':
        y, err = lo.solve(x, f, y_0, solver, N=N, T=T, partition=partition, atol=atol, rtol=rtol, method=method,
                          compute_bound=compute_bound, N_sol=N_sol, N_min=N_min, N_max=N_max, n=n, speed=speed)
        toc = time.perf_counter()
        if plot:
            plt.plot(y[0, :], y[1, :])
            plt.show()
        return y, err, toc-tic


def ex_nilpotent_vf(param=4):
    N = np.array([1, 2, 3])
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
    sig_steps = np.array([1])
    x = wiggly_sinh(symbolic=True, param=param, N=N[-1])
    f = linear_nilpotent_3x2_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([1., 1., 1.])
    description = 'nilpotent vector field'
    return x, f, y_0, N, n, description, True, True, sig_steps


def ex_smooth_path(param=4):
    N = np.array([1, 2, 3])
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
    sig_steps = np.array([1])
    x = unit_circle(symbolic=True, param=param, N=N[-1])
    f = smooth_2x2_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    if param < 15:
        description = 'smooth path'
    else:
        description = 'oscillatory path'
    return x, f, y_0, N, n, description, True, True, sig_steps


def ex_smooth_3d_path(param=1296):
    N = np.array([1, 2, 3, 4])
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
    sig_steps = np.array([1])
    x = smooth_3d_path(symbolic=True, param=param, N=N[-1])
    f = linear_2x3_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([1., 1.])
    if param < 15:
        description = 'smooth 3d path'
    else:
        description = 'oscillatory 3d path'
    return x, f, y_0, N, n, description, True, True, sig_steps


def ex_pure_area():
    description = 'pure area'
    sig_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
    N = np.array([1, 2, 3])
    x = pure_area(sig_steps=sig_steps[-1])
    f = smooth_2x2_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    return x, f, y_0, N, n, description, False, True, sig_steps


def ex_third_level(param=4):
    description = 'third level'
    N = np.array([1, 2, 3])
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631])
    sig_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    x = third_level_path(param=param, sig_steps=sig_steps[-1])
    f = smooth_2x3_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    return x, f, y_0, N, n, description, False, True, sig_steps


def ex_fBm_path(H=0.5):
    description = f'fractional Brownian motion, H={H}'
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585])
    N = np.array([1, 2, 3])
    sig_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    x = rough_fractional_Brownian_path(H=H, n=int(n[-1] * sig_steps[-1]), dim=2, T=1., p=1 / H + 0.1, save_level=N[-1])
    f = smooth_2x2_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    return x, f, y_0, N, n, description, False, True, sig_steps


def ex_Bm_path():
    description = f'Brownian motion'
    n = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    N = np.array([1, 2, 3])
    sig_steps = np.array([1])
    x = brt.BrownianRoughTree(dim=2, T=1., has_time=False)
    f = smooth_2x2_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    return x, f, y_0, N, n, description, False, True, sig_steps


def ex_langevin_banana(H=0.5):
    description = f'Langevin with banana potential, q={H-5}'
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000])
    N = np.array([1, 2, 3])
    sig_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    x = rough_fractional_Brownian_path_time(H=H, n=n[-1] * sig_steps[-1], dim=2, T=1., p=1 / H + 0.1, save_level=N[-1])
    f = langevin_banana_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    return x, f, y_0, N, n, description, False, True, sig_steps


def ex_smooth_fractional(H=0.5):
    description = f'smooth and fBm path, H={H:.1f}'
    n = np.array([1, 2, 3, 4, 6, 10, 16, 25, 40, 63, 100, 158, 251, 398, 631, 1000, 1585])
    N = np.array([3])
    sig_steps = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    x = smooth_fractional_path(H=H, n=int(n[-1] * sig_steps[-1]), p=1 / H + 0.1, save_level=N[-1])
    f = smooth_2x2_vector_field(symbolic=True, N=N[-1])
    y_0 = np.array([0., 0.])
    return x, f, y_0, N, n, description, False, True, sig_steps


def example(param):
    if param == 0:
        return ex_nilpotent_vf(param=4)
    elif 0 < param < 1:
        return ex_fBm_path(H=param)
    elif param == 1:
        return ex_pure_area()
    elif 1 < param <= 2:
        param = int(1 / (param - 1))
        return ex_smooth_path(param=param)
    elif 2 < param <= 3:
        param = int(1 / (param - 2))
        return ex_third_level(param=param)
    elif param == 4:
        return ex_Bm_path()
    elif 5 <= param <= 6:
        return ex_langevin_banana(H=param - 5)
    elif 6 < param <= 7:
        return ex_smooth_fractional(H=param - 6)
    else:
        return


def discussion(ex, show=False, save=False, solver='fsss', N_sol=2):
    """
    Discusses the problem in smooth_vf_smooth_path.
    :param ex: Float that chooses the example that is discussed.
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
    if solver[1] == 'f':
        full = True
    else:
        full = False

    atol = 1e-09
    rtol = 1e-06
    adaptive_tol = False
    T = 1

    x, f, y_0, N, n, description, sym_path, sym_vf, sig_steps = example(ex)

    sol_dim = f.dim_y
    ref_N = N[-1]
    ref_n = n[-1] * 4
    p = x.p
    directory = 'C:/Users/breneis/Desktop/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots/'
    directory = directory + description

    solutions = np.zeros((len(N), len(n), len(sig_steps), sol_dim))
    error_bounds = np.zeros((len(N), len(n), len(sig_steps)))
    times = np.zeros((len(N), len(n), len(sig_steps)))
    true_errors = np.zeros((len(N), len(n), len(sig_steps)))

    true_sol, _, _ = apply_example(x=x, f=f, y_0=y_0, partition=np.linspace(0, T, ref_n+1), N=ref_N, atol=atol,
                                   rtol=rtol, solver=solver, N_sol=N_sol)
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

        # The following line is for the Brownian rough tree, to precompute the signatures on the larger intervals
        apply_example(x=x, f=f, y_0=y_0, partition=np.linspace(0, T, n[0]+1), N=N[-1], atol=atol, rtol=rtol,
                      solver=solver, N_sol=N_sol)

        for i in range(len(N)):
            for j in range(len(n)):
                for l in range(len(sig_steps)):
                    print(f"N = {N[i]}, n = {n[j]}, sig_steps = {sig_steps[l]}")
                    if len(sig_steps) > 1:
                        x.sig_steps = sig_steps[l]
                    sol, err, tim = apply_example(x=x, f=f, y_0=y_0, partition=np.linspace(0, T, n[j]+1), N=N[i],
                                                  atol=atol, rtol=rtol, solver=solver, N_sol=N_sol)
                    solutions[i, j, l, :] = sol[:, -1]
                    error_bounds[i, j, l] = err
                    times[i, j, l] = tim
                    true_errors[i, j, l] = ta.l1(sol[:, -1] - true_sol)

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
                    if len(sig_steps) > 1:
                        x.sig_steps = sig_steps[l]
                    sol, _, tim = apply_example(x=x, f=f, y_0=y_0, partition=np.linspace(0, T, n[j]+1), N=N[i],
                                                atol=atol_, rtol=rtol_, solver=solver, N_sol=N_sol)
                    solutions[i, j, l, :] = sol.sig(0, T, N_sol).to_array()
                    times[i, j, l] = tim
                    true_errors[i, j, l] = ta.l1(solutions[i, j, l, :] - true_sol)

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


def error_representation_discussion(ex, show=False, save=False, speed=-1):
    """
    Discusses the problem in smooth_vf_smooth_path.
    :param ex: Float that chooses the example that is discussed.
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
    atol = 1e-09
    rtol = 1e-06
    T = 1

    x, f, y_0, N, n, description, _, _, _ = example(ex)

    sol_dim = f.dim_y
    ref_N = N[-1]
    ref_n = n[-1] * 4
    directory = 'C:/Users/breneis/Desktop/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots/'
    directory = directory + description

    solutions = np.zeros((len(N), len(n), sol_dim))
    estimated_errors = np.zeros((len(N), len(n), sol_dim))
    times = np.zeros((len(N), len(n)))
    true_errors = np.zeros((len(N), len(n), sol_dim))
    abs_estimated_errors = np.zeros((len(N), len(n)))
    abs_true_errors = np.zeros((len(N), len(n)))
    abs_differences = np.zeros((len(N), len(n)))

    true_sol, _, _ = apply_example(x=x, f=f, y_0=y_0, solver='fssr', N=ref_N, partition=np.linspace(0, T, ref_n+1),
                                   plot=False, atol=atol, rtol=rtol, T=T, speed=speed)
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
            print(f"N = {N[i]}, n = {n[j]}")
            sol, loc_errors, tim = apply_example(x=x, f=f, y_0=y_0, solver='fssr', N=N[i],
                                                 partition=np.linspace(0, T, n[j]+1), atol=atol, rtol=rtol, T=T,
                                                 speed=speed)
            solutions[i, j, :] = sol[-1, :]
            estimated_errors[i, j, :] = np.sum(loc_errors, axis=0)
            times[i, j] = tim
            true_errors[i, j, :] = true_sol - sol[-1, :]
            abs_estimated_errors[i, j] = ta.l1(estimated_errors[i, j])
            abs_true_errors[i, j] = ta.l1(true_errors[i, j])
            abs_differences[i, j] = ta.l1(true_errors[i, j, :] - estimated_errors[i, j, :])

        if len(n) > 1:
            compare_n_error_representation(N[i], n, abs_true_errors[i, :], abs_estimated_errors[i, :],
                                           abs_differences[i, :], description, show, save, directory)


def adaptive_error_fixed_N_discussion(ex, show=False, save=False, speed=-1):
    """
    Discusses the problem in smooth_vf_smooth_path.
    :param ex: Float that chooses the example that is discussed.
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
    atol = 1e-05
    rtol = 1e-07
    T = 1

    x, f, y_0, N, _, description, _, _, _ = example(ex)

    sol_dim = f.dim_y
    directory = 'C:/Users/breneis/Desktop/Studium/Mathematik WIAS/T/9999-99 Main file/LogODE plots/'
    directory = directory + description

    solutions = np.zeros((len(N), sol_dim))
    estimated_errors = np.zeros((len(N), sol_dim))
    times = np.zeros(len(N))
    true_errors = np.zeros((len(N), sol_dim))
    abs_estimated_errors = np.zeros(len(N))
    abs_true_errors = np.zeros(len(N))
    abs_differences = np.zeros(len(N))

    for i in range(len(N)):
        print(f"N = {N[i]}")
        partition, sol, loc_errors, tim = apply_example(x=x, f=f, y_0=y_0, solver='assN', N=N[i], atol=atol, rtol=rtol,
                                                        T=T, speed=speed, n=20)
        solutions[i, :] = sol[-1, :]
        estimated_errors[i, :] = np.sum(loc_errors, axis=0)
        times[i] = tim

        vec_1 = np.zeros(2*(len(partition)-1))
        vec_2 = np.zeros(2*(len(partition)-1))
        for j in range(len(partition)-1):
            vec_1[2*j] = partition[j]
            vec_1[2*j+1] = partition[j+1] - 1e-10
            vec_2[2*j] = partition[j+1] - partition[j]
            vec_2[2*j+1] = vec_2[2*j]
        plt.yscale('log')
        plt.xlabel('t')
        plt.ylabel('Time interval length')
        plt.title(description[0].upper() + description[1:] + f'\nDegree {N[i]}')
        plt.plot(vec_1, vec_2)
        if show:
            plt.show()
        if save:
            plt.savefig(directory + '/' + description[0].upper() + description[1:] + f', degree {N[i]}, time intervals',
                        format='pdf')

    fine_partition = np.zeros(20*len(partition)-19)
    for i in range(len(partition)-1):
        fine_partition[20*i:20*(i+1)+1] = np.linspace(partition[i], partition[i+1], 21)

    true_sol, _, _ = apply_example(x=x, f=f, y_0=y_0, solver='fsss', N=N[-1], partition=fine_partition,
                                   atol=1e-09, rtol=1e-06, T=T, speed=speed)

    for i in range(len(N)):
        true_errors[i, :] = true_sol[:, -1] - solutions[i, :]
        abs_estimated_errors[i] = ta.l1(estimated_errors[i])
        abs_true_errors[i] = ta.l1(true_errors[i])
        abs_differences[i] = ta.l1(true_errors[i, :] - estimated_errors[i, :])

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

    return abs_true_errors, abs_estimated_errors, abs_differences


def accuracy_of_signature(m=2):
    T = 0.1
    f = smooth_2x2_vector_field(N=3)
    y_0 = np.array([0, 0])
    # n_vec = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    # n_sig_vec = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144])
    n_vec = np.array([1, 2, 4, 8, 16])
    # n_sig_vec = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
    # n_sig_vec = np.array([1, 3, 9, 27, 81, 243, 729, 2187, 6561])
    # n_sig_vec = np.array([1, 4, 16, 64, 256, 1024, 4096, 16384])
    n_sig_vec = np.array([1, 3, 7, 18, 50, 150, 300, 700, 1575, 3150, 6300])
    n_total = n_vec[-1] * n_sig_vec[-1]
    errors = np.empty((2, len(n_vec), len(n_sig_vec), m))

    for i in range(m):
        print(f'{i} of {m}')
        dW = np.random.normal(0, np.sqrt(T/n_total), (n_total, 2))
        W = np.zeros((n_total+1, 2))
        W[1:, :] = np.cumsum(dW, axis=0)

        brownian_path = rp.RoughPathDiscrete(times=np.linspace(0, 1, n_total+1), values=W, p=1, save_level=1)
        true_solution, _ = lo.solve_fixed(x=brownian_path, f=f, y_0=y_0, N=1,
                                          partition=np.linspace(0, 1, n_total+1), atol=1e-08, rtol=1e-06,
                                          compute_bound=False)
        true_solution = true_solution[:, -1]
        approx_solutions = np.empty((2, len(n_vec), len(n_sig_vec), 2))

        for j in range(len(n_vec)):
            for k in range(len(n_sig_vec)):
                print(i, j, k)
                n_loc = n_vec[j] * n_sig_vec[k]
                W_loc = W[::n_total//n_loc]
                brownian_path = rp.RoughPathDiscrete(times=np.linspace(0, 1, n_loc+1), values=W_loc, p=2.1,
                                                     save_level=3)
                for a in range(2):
                    print(i, j, k, a)
                    approx_solution, _ = lo.solve_fixed(x=brownian_path, f=f, y_0=y_0, N=a+2,
                                                        partition=np.linspace(0, 1, n_vec[j]+1))
                    approx_solutions[a, j, k, :] = approx_solution[:, -1]
                    errors[a, j, k, i] = ta.l1(approx_solutions[a, j, k, :] - true_solution)

    avg_errors = np.average(errors, axis=-1)
    std_errors = 1.96*np.std(errors, axis=-1)/np.sqrt(m)

    for a in range(2):
        for j in range(len(n_vec)):
            plt.loglog(n_sig_vec, avg_errors[a, j, :], 'k-', label='average error')
            plt.loglog(n_sig_vec, avg_errors[a, j, :] + std_errors[a, j, :], 'k--', label=r'$95%$ confidence interval')
            plt.loglog(n_sig_vec, avg_errors[a, j, :] - std_errors[a, j, :], 'k--')
            plt.title(f'Degree {a+2}, {n_vec[j]} intervals')
            plt.xlabel('Number of intervals for computing the signature')
            plt.ylabel('Error')
            plt.legend(loc='upper right')
            plt.show()

    return avg_errors, std_errors

'''

N_vec = np.array([2, 3])
n_vec = np.array([1, 2, 4, 8, 16])
n_sig_vec = np.array([1, 3, 7, 18, 50, 150, 300, 700, 1575, 3150, 6300])

avg_errors = np.array([[[4.00170308e-02, 3.83928261e-02, 1.37198092e-02, 1.02783530e-02,
         8.12912037e-03, 5.58490083e-03, 5.33321787e-03, 5.80815465e-03,
         5.96183779e-03, 6.03094321e-03, 5.84881509e-03],
        [3.01619198e-02, 1.74132385e-02, 9.59243739e-03, 7.82346998e-03,
         4.81571340e-03, 2.18475837e-03, 3.20767527e-03, 3.19115547e-03,
         2.97873134e-03, 2.74145485e-03, 2.56124137e-03],
        [1.85394109e-02, 1.11286396e-02, 1.14751998e-02, 5.67240806e-03,
         4.20135237e-03, 3.14959721e-03, 3.04143992e-03, 2.34851246e-03,
         2.26885478e-03, 2.11913578e-03, 1.88006640e-03],
        [1.20162025e-02, 1.18322569e-02, 5.66390802e-03, 4.20704437e-03,
         2.90022488e-03, 2.10459939e-03, 1.73189865e-03, 1.18981668e-03,
         1.06307760e-03, 8.42194068e-04, 7.74461752e-04],
        [1.01706887e-02, 4.94961219e-03, 4.53687569e-03, 4.34033964e-03,
         2.46090911e-03, 1.45847636e-03, 1.24737282e-03, 9.39809011e-04,
         5.53756247e-04, 4.74464110e-04, 3.98900586e-04]],

       [[4.00170308e-02, 3.85194282e-02, 1.29814534e-02, 9.10506585e-03,
         5.99736961e-03, 4.40283077e-03, 1.65488875e-03, 2.80445413e-03,
         2.25265764e-03, 1.63800597e-03, 1.13430673e-03],
        [3.01619198e-02, 1.69752388e-02, 9.54899904e-03, 7.91149774e-03,
         4.69922989e-03, 1.63941439e-03, 1.94544043e-03, 1.85671727e-03,
         1.63172650e-03, 1.02369530e-03, 5.10786956e-04],
        [1.85394109e-02, 1.09256556e-02, 1.20136582e-02, 5.81986431e-03,
         3.10934246e-03, 1.95031133e-03, 1.56715220e-03, 1.11621326e-03,
         1.05900788e-03, 5.69787258e-04, 2.90031761e-04],
        [1.20162025e-02, 1.19708761e-02, 5.83301980e-03, 4.35263340e-03,
         2.72521717e-03, 1.48683145e-03, 1.30532845e-03, 9.37398883e-04,
         5.31672836e-04, 2.30912813e-04, 1.36393187e-04],
        [1.01706887e-02, 4.92838090e-03, 4.54418425e-03, 4.29670829e-03,
         2.36310817e-03, 1.32894340e-03, 1.19739262e-03, 7.69926393e-04,
         2.29142972e-04, 1.53665206e-04, 2.67806669e-05]]])

std_errors = np.array([[[2.34489020e-02, 2.05674940e-02, 3.96247055e-03, 5.40000098e-03,
         2.80621001e-03, 1.81567726e-03, 2.41767801e-03, 2.67198473e-03,
         2.41378907e-03, 2.29804886e-03, 2.32705722e-03],
        [2.54432805e-02, 8.82780036e-03, 3.23520000e-03, 2.98936120e-03,
         1.68695033e-03, 4.82578112e-04, 1.07317074e-03, 1.07537823e-03,
         5.02795448e-04, 4.98547769e-04, 6.08042607e-04],
        [1.17384020e-02, 4.58443177e-03, 4.60841816e-03, 2.27811983e-03,
         1.72801661e-03, 1.37963854e-03, 1.03755895e-03, 9.51646225e-04,
         9.19747466e-04, 9.30358991e-04, 1.05665007e-03],
        [5.97604530e-03, 6.03180328e-03, 2.69734932e-03, 1.82726665e-03,
         1.32353875e-03, 5.47265461e-04, 5.24597704e-04, 2.65037694e-04,
         3.53067236e-04, 4.02977072e-04, 4.19799315e-04],
        [3.44365769e-03, 3.31251234e-03, 1.55884859e-03, 1.44807063e-03,
         9.55177075e-04, 4.86161489e-04, 4.14050861e-04, 2.17079280e-04,
         1.71271828e-04, 1.52536410e-04, 1.82896169e-04]],

       [[2.34489020e-02, 2.05387624e-02, 4.41472625e-03, 5.05851427e-03,
         3.80010952e-03, 1.46280436e-03, 5.88312809e-04, 9.12539641e-04,
         7.34965672e-04, 6.06594907e-04, 5.60747035e-04],
        [2.54432805e-02, 8.93082242e-03, 3.30561875e-03, 3.21412699e-03,
         1.62464376e-03, 6.53446218e-04, 1.09242807e-03, 8.88568290e-04,
         6.36279056e-04, 2.29893627e-04, 2.69205357e-04],
        [1.17384020e-02, 4.92572883e-03, 4.92036137e-03, 2.28687481e-03,
         1.83846043e-03, 9.86025041e-04, 7.31134809e-04, 5.05357646e-04,
         2.70784548e-04, 2.65396396e-04, 1.33811774e-04],
        [5.97604530e-03, 5.95360639e-03, 2.76994880e-03, 1.80928011e-03,
         1.31394656e-03, 7.22615640e-04, 5.46115192e-04, 1.20006612e-04,
         2.61683977e-04, 8.91266156e-05, 5.43330720e-05],
        [3.44365769e-03, 3.29377610e-03, 1.48645262e-03, 1.45967130e-03,
         1.04419797e-03, 5.35279829e-04, 3.85428887e-04, 2.15977391e-04,
         9.52974758e-05, 6.00496108e-05, 7.29130279e-06]]])

colors = ['r', 'C1', 'y', 'g', 'b']

plt.loglog(n_sig_vec, avg_errors[0, 0, :], colors[0] + '-', label=r'$n=$' + f'{n_vec[0]}')
plt.loglog(n_sig_vec, avg_errors[0, 1, :], colors[1] + '-', label=r'$n=$' + f'{n_vec[1]}')
plt.loglog(n_sig_vec, avg_errors[0, 2, :], colors[2] + '-', label=r'$n=$' + f'{n_vec[2]}')
plt.loglog(n_sig_vec, avg_errors[0, 3, :], colors[3] + '-', label=r'$n=$' + f'{n_vec[3]}')
plt.loglog(n_sig_vec, avg_errors[0, 4, :], colors[4] + '-', label=r'$n=$' + f'{n_vec[4]}')
plt.loglog(n_sig_vec, avg_errors[1, 0, :], colors[0] + '--')
plt.loglog(n_sig_vec, avg_errors[1, 1, :], colors[1] + '--')
plt.loglog(n_sig_vec, avg_errors[1, 2, :], colors[2] + '--')
plt.loglog(n_sig_vec, avg_errors[1, 3, :], colors[3] + '--')
plt.loglog(n_sig_vec, avg_errors[1, 4, :], colors[4] + '--')
plt.legend(loc='upper right')
plt.xlabel('Number of intervals for computing the signature')
plt.ylabel('Error')
plt.show()
'''


def accuracy_of_level_N_signature(N, log_n, m):
    n = int(2**log_n)
    n_vec = np.array([int(2**i) for i in range(log_n-3+1)])
    errors = np.empty((len(n_vec), m))

    for i in range(m):
        print(i)
        brownian_path = np.zeros((n+1, 2))
        brownian_path[1:, :] = np.cumsum(np.random.normal(0, 1/np.sqrt(n), (n, 2)), axis=0)
        true_sig = ta.sig(brownian_path, N)
        for j in range(len(n_vec)):
            print(i, j)
            approx_sig = ta.sig(brownian_path[::n//n_vec[j]], N)
            errors[j, i] = (approx_sig - true_sig).norm(N)

    avg_errors = np.average(errors, axis=1)
    std_errors = 1.96*np.std(errors, axis=1) / np.sqrt(m)

    plt.loglog(n_vec, avg_errors, 'k-', label='average errors')
    plt.loglog(n_vec, avg_errors - std_errors, 'k--', label=r'$95\%$ confidence interval')
    plt.loglog(n_vec, avg_errors + std_errors, 'k--')
    plt.legend(loc='upper right')
    plt.xlabel('Number of intervals for computing the signature')
    plt.ylabel('Error')
    plt.title(f'Error in the level {N} signature for Brownian motion')
    plt.show()


