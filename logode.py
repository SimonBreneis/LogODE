import numpy as np
import scipy
from scipy import integrate, special
import roughpath as rp
import vectorfield as vf


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


class LogODESolver:
    def __init__(self, x, f, y_0, method='RK45'):
        self.x = x
        self.f = f
        self.y_0 = y_0
        self.method = method
        self.dim = len(self.x.incr(0., 0., 1)[1])

    def solve_step(self, y_s, s, t, N, atol, rtol):
        """
        Implementation of the Log-ODE method.
        :param N: The degree of the Log-ODE method (f needs to be Lip(N))
        :param y_s: Current solution value
        :param s: Initial interval point
        :param t: Final interval point
        :return: Solution on partition points
        """
        self.f.reset_local_norm()
        ls = self.x.log_incr(s, t, N)[1:]
        for i in range(N):
            ls[i] = np.transpose(ls[i], [i + 1 - j for j in range(1, i + 2)])

        y = integrate.solve_ivp(lambda t, z: self.f.vector_field(ls)(z), (0, 1), y_s, method=self.method,
                                atol=atol, rtol=rtol).y[:, -1]
        return y, self.f.local_norm, self.x.omega(s, t)

    def solve_fixed(self, N, partition, atol, rtol):
        """
        Implementation of the Log-ODE method.
        :param N: The degree of the Log-ODE method (f needs to be Lip(N))
        :param partition: Partition of the interval on which we apply the Log-ODE method
        :return: Solution on partition points, error bound (-1 if no norm was specified)
        """
        y = np.zeros(shape=(len(self.y_0), len(partition)))
        y[:, 0] = self.y_0
        p = self.x.p

        error_estimate = 0.
        for i in range(1, len(partition)):
            y[:, i], vf_norm, omega = self.solve_step(y[:, i - 1], partition[i - 1], partition[i], N, atol, rtol)
            error_estimate += vf_norm ** (N + 1) * omega ** ((N + 1) / p)
        return y, self.local_log_ode_error_constant(N) * error_estimate

    def local_log_ode_error_constant(self, N):
        """
        Returns the constant in the error bound for a single step of the Log-ODE method.
        :param N: Degree of the method
        :return: The constant
        """
        p = self.x.p
        if p == 1:
            return 0.34 * (7 / 3.) ** (N + 1)
        if p == 2:
            return 25 * self.dim / scipy.special.gamma((N + 3) / 2.) + 0.081 * (7 / 3) ** (N + 1)
        return 1000 * self.dim ** 3 / scipy.special.gamma((N + 4) / 3.) + 0.038 * (7 / 3) ** (N + 1)

    def solve_adaptive(self, T, atol=1., rtol=1.):
        """
        Implementation of the Log-ODE method.
        :return: Solution on partition points
        """
        eps = atol
        n_steps = self.x.n_steps
        var_steps = self.x.var_steps
        p = self.x.p
        p = int(p)
        norm_estimates = np.zeros(10)
        self.x.n_steps = max(n_steps, 100)
        _, norm_estimates[0], omega_estimate = self.solve_step(self.y_0, s=0, t=T, N=p, atol=10*atol, rtol=10*rtol)
        _, norm_estimates[1], _ = self.solve_step(self.y_0, s=0, t=T, N=p+1, atol=10*atol, rtol=10*rtol)
        _, norm_estimates[2], _ = self.solve_step(self.y_0, s=0, t=T, N=p+2, atol=10*atol, rtol=10*rtol)
        self.x.n_steps = n_steps
        norm_incr = max(norm_estimates[2] - norm_estimates[1], norm_estimates[1] - norm_estimates[0])
        norm_estimates[3:] = norm_estimates[2] + norm_incr * np.arange(1, 8)
        print(f"Norm estimates: {norm_estimates}")
        print(f"Error constants: {np.array([self.local_log_ode_error_constant(N) for N in range(p, p + 10)])}")
        print(f"Omega: {omega_estimate}")
        number_intervals = np.array([(self.local_log_ode_error_constant(N) * norm_estimates[N - p] ** (
                N + 1) * omega_estimate ** ((N + 1.) / p) / eps) ** (p / (N - p + 1)) for N in range(p, p + 10)])
        print(f"Number of intervals: {number_intervals}")
        complexity = np.array([self.dim ** N for N in range(p, p + 10)]) * number_intervals
        N = np.argmin(complexity) + p
        print(f"N = {N}")
        number_intervals = (number_intervals[N - p] / 10) ** (2. / (1 + p))
        print("Let us find a partition!")
        partition = self.find_partition(s=0, t=T, total_omega=omega_estimate, n=number_intervals)
        print("We found a partition!")
        atol = atol/len(partition)
        rtol = rtol/len(partition)
        local_Ns = [N] * (len(partition) - 1)
        max_local_error = [eps / number_intervals] * (len(partition) - 1)
        y = [self.y_0]

        i = 0
        while i < len(partition) - 1:
            print(f"At index {i + 1} of {len(partition) - 1}")
            local_N = local_Ns[i]
            y_next, norm_est, omega_est = self.solve_step(y_s=y[i], s=partition[i], t=partition[i + 1], N=local_N,
                                                          atol=atol, rtol=rtol)
            print(f"Norm estimate: {norm_est}, Omega estimate: {omega_est}")
            error_est = self.local_log_ode_error_constant(local_N) * norm_est ** (local_N + 1) * omega_est ** (
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
                    error_est_N = self.local_log_ode_error_constant(new_local_N + 1) * norm_est ** (
                            new_local_N + 2) * omega_est ** ((new_local_N + 2.) / p)
                    error_est_part = error_est / self.dim ** ((new_local_N + 1) / p) * self.dim
                    if error_est_N < error_est_part:
                        new_local_N += 1
                    else:
                        subpartition *= 4
                    new_error_est = subpartition * self.local_log_ode_error_constant(new_local_N) * norm_est ** (
                            new_local_N + 1) * (omega_est / subpartition) ** ((new_local_N + 1.) / p)
                if subpartition > 1:
                    better_var_est = self.x.omega(partition[i], partition[i + 1], var_steps=var_steps * 3)
                    new_subpartition = self.find_partition(partition[i], partition[i + 1], better_var_est, subpartition)
                    insert_list(partition, new_subpartition[1:-1], i + 1)
                    insert_list(local_Ns, [new_local_N] * (len(new_subpartition) - 2), i + 1)
                    insert_list(max_local_error,
                                [max_local_error[i] / (len(new_subpartition) - 1)] * (len(new_subpartition) - 2), i + 1)
                    max_local_error[i] = max_local_error[i] / (len(new_subpartition) - 1)
                local_Ns[i] = new_local_N
        return np.array(partition), np.array(y)

    def find_next_interval(self, s, t, lower_omega, upper_omega):
        """
        Finds an interval of the form [s,u] with s <= u <= t such that lower_omega <= omega(s, u) <= upper_omega.
        :param s: Initial point of total interval
        :param t: Final point of total interval
        :param lower_omega: Lower bound of the control function on the new interval
        :param upper_omega: Upper bound of the control function on the new interval
        :return: The partition point u.
        """
        total_omega = self.x.omega(s, t)
        if total_omega <= upper_omega:
            return t
        u_current = s + (t - s) * (lower_omega + upper_omega) / (2 * total_omega)
        u_left = s
        u_right = t

        current_omega = self.x.omega(s, u_current)

        while not lower_omega <= current_omega <= upper_omega and u_right - u_left > (t - s) * 10 ** (-10):
            if current_omega > lower_omega:
                u_right = u_current
            else:
                u_left = u_current

            u_current = (u_left + u_right) / 2
            current_omega = self.x.omega(s, u_current)

        return u_current

    def find_partition(self, s, t, total_omega, n, q=2.):
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
        p = self.x.p
        partition = [s]
        lower_omega = total_omega / (q * n) ** (1 / p)
        upper_omega = total_omega * (q / n) ** (1 / p)
        print(f"Total omega: {total_omega}")
        print(f"Lower omega: {lower_omega}")
        print(f"Upper omega: {upper_omega}")
        while not partition[-1] == t:
            next_point = self.find_next_interval(partition[-1], t, lower_omega, upper_omega)
            partition.append(next_point)
        return partition
