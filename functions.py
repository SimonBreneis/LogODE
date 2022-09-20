import time
import matplotlib.pyplot as plt
import numpy as np
import logode as lo
import roughpath as rp
import tensoralgebra as ta

"""
Brownian rough trees were precomputed and saved as
for i in range(10):
    tree = brt.initialize_brownian_rough_tree(dim=2, T=1, has_time=False, depth=15, accuracy=20, N=4, delete=True)
    tree.save(directory=f'brownian rough trees/tree {i}')
"""


def color(i, N):
    """
    Returns a color for plotting.
    :param i: Index of the line. This is the color of the ith line that is being plotted
    :param N: Total number of lines to plot. Depending on how many colors are needed, returns a different color to
        ensure that the colors are well distinguishable.
    :return: Some color for plotting (a string)
    """
    c = ['r', 'C1', 'y', 'g', 'b', 'purple']
    c_ = ['darkred', 'r', 'C1', 'y', 'lime', 'g', 'deepskyblue', 'b', 'purple', 'deeppink']
    c_short = ['r', 'g', 'b']
    c_intermediate = ['r', 'C1', 'g', 'b']
    if N <= 3:
        return c_short[i % 3]
    if N <= 4:
        return c_intermediate[i % 4]
    if N <= 6 or N == 11 or N == 12:
        return c[i % 6]
    return c_[i % 10]


def profile(statement):
    """
    Profiles statement. Only call in main file under if __name__ == '__main__': clause.
    :param statement: The statement (function call) that should be profiled. Is a string
    :return: Nothing, but prints the results
    """
    import cProfile
    import pstats
    cProfile.run(statement, "{}.profile".format(__file__))
    stats = pstats.Stats("{}.profile".format(__file__))
    stats.strip_dirs()
    stats.sort_stats("cumtime").print_stats(100)
    stats.sort_stats("tottime").print_stats(100)


def discuss_example(x, f, y_0, N_min=1, N_max=3, T=1., n=16, g=None, g_grad=None, atol=1e-04, rtol=1e-02, method='RK45',
                    speed=-1, verbose=0):
    """
    Discusses the solution algorithms for RDEs by comparing the predicting adaptive algorithm with the error
    representation formula, the adaptive algorithm with the error representation formula which tests out everything,
    the algorithm without the error representation formula which only computes the first level, and the algorithm
    without the error representation formula which computes the full solution.
    :param x: Rough path
    :param f: Vector field
    :param y_0: Initial condition
    :param N_min: Minimal degree that should be used
    :param N_max: Maximal degree that should be used
    :param T: Final time
    :param n: Initial number of intervals
    :param g: Payoff function
    :param g_grad: Gradient of payoff function
    :param atol: Absolute error tolerance
    :param rtol: Relative error tolerance
    :param method: Method for solving the ODEs
    :param speed: Non-negative number. The larger speed, the faster the algorithm, but the more inaccurate the
        estimated global error. If speed is -1, automatically finds a good value for speed
    :param verbose: Determines the number of intermediary results printed to the console
    :return: Nothing
    """

    def small_discussion():
        print(f'The number of intervals used is {len(partition) - 1}.')
        print('Computing a reference solution on a finer grid')
        fine_partition = np.linspace(0, T, 8 * len(partition) - 7)
        good_y, _, _, _ = lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_min, partition=fine_partition,
                                      atol=atol / len(fine_partition),
                                      rtol=rtol / len(fine_partition), method=method)
        if isinstance(y, rp.RoughPath):
            final_value = y.at(t=T, N=1)[1]
        else:
            final_value = y[-1, :]
        print('Computed a reference solution on a finer grid')
        print(f'The final point of the solution given by the algorithm is {final_value}.')
        print(f'The reference final point of the solution is {good_y[-1, :]}.')
        print(f'The global error estimated by comparing the two solution is {ta.l1(final_value - good_y[-1, :])}.')

    def discuss():
        print('Computing a reference solution on a finer grid')
        fine_partition = np.empty(8 * len(partition) - 7)
        fine_N = np.zeros(8 * len(N), dtype=int)
        for i in range(len(partition) - 1):
            fine_partition[8 * i:(8 * i + 9)] = np.linspace(partition[i], partition[i + 1], 9)
            fine_N[8 * i:8 * (i + 1)] = np.ones(8, dtype=int) * N[i]
        good_y, _, _ = lo.solve_fixed(x=x, f=f, y_0=y_0, N=fine_N, partition=fine_partition,
                                      atol=atol / len(fine_partition),
                                      rtol=rtol / len(fine_partition), method=method)
        print('Computed a reference solution on a finer grid')
        print(f'The final point of the solution given by the algorithm with the fully adaptive error representation is '
              f'{y[-1, :]}.')
        print(f'The reference final point of the solution is {good_y[-1, :]}.')
        global_error = np.sum(prop_loc_err, axis=0)
        print(f'The global error estimated by the error representation is {global_error}, with norm '
              f'{ta.l1(global_error)}.')
        print(f'The global error estimated by comparing the two solution is {ta.l1(y[-1, :] - good_y[-1, :])}.')
        corrected_y = y[-1, :] + np.sum(prop_loc_err, axis=0)
        print(f'The final point of the solution given by the algorithm with the fully adaptive error representation '
              f'after correcting using the error estimate is {corrected_y}.')
        print(f'The global error estimated by comparing the corrected solution with the refined solution is '
              f'{ta.l1(good_y[-1, :] - corrected_y)}.')
        plt.plot(y[:, 0], y[:, 1])
        plt.title('Solution path')
        plt.show()
        prev_N = np.array([True for _ in range(N_min, N_max + 1)])
        for i in range(len(partition) - 1):
            if prev_N[N[i] - N_min]:
                prev_N[N[i] - N_min] = False
                plt.plot(np.array([partition[i], partition[i + 1]]),
                         np.array([partition[i + 1] - partition[i], partition[i + 1] - partition[i]]),
                         color=color(N[i] - N_min, N_max + 1 - N_min), label=f'N={N[i]}')
            else:
                plt.plot(np.array([partition[i], partition[i + 1]]),
                         np.array([partition[i + 1] - partition[i], partition[i + 1] - partition[i]]),
                         color=color(N[i] - N_min, N_max + 1 - N_min))
        plt.title('Length and degree of partition intervals')
        plt.xlabel('Time')
        plt.ylabel('Length of interval')
        plt.yscale('log')
        plt.legend(loc='best')
        plt.show()

        plt.plot(partition, y[:, 0], label='first component')
        plt.plot(partition, y[:, 1], label='second components')
        plt.title('Solution against time')
        plt.xlabel('Time')
        plt.legend(loc='best')
        plt.show()

    print('Starting to solve the RDE using the fully adaptive error representation')
    partition, y, prop_loc_err, N = lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=y_0, N_min=N_min,
                                                                                 N_max=N_max, atol=atol, rtol=rtol, g=g,
                                                                                 g_grad=g_grad, n=n, T=T, method=method,
                                                                                 speed=speed, verbose=verbose)
    print(f'Finished solving the RDE using the fully adaptive error representation')
    discuss()

    print('Starting to solve the RDE using the fully adaptive error representation and the algorithm which tries '
          'everything')
    partition, y, prop_loc_err, N = lo.solve_fully_adaptive_error_representation_slow(x=x, f=f, y_0=y_0, N_min=N_min,
                                                                                      N_max=N_max, atol=atol, rtol=rtol,
                                                                                      g=g, g_grad=g_grad, n=n, T=T,
                                                                                      method=method, speed=speed,
                                                                                      verbose=verbose)
    print(f'Finished solving the RDE using the fully adaptive error representation and the algorithm which tries '
          f'everything')
    discuss()

    print('Starting to solve the RDE without the error representation formula, where we solve only for the first level')
    tic = time.perf_counter()

    def solver(partition_):
        return lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_min, partition=partition_, atol=atol, rtol=rtol, method=method,
                              compute_bound=False, verbose=verbose)[0]

    partition, y = lo.solve_error_tolerance(solver=solver, n=n, T=T, atol=atol, rtol=rtol)
    print(f'Finished solving the RDE without the error representation formula, where we solve only for the first level.'
          f' It took {time.perf_counter() - tic} seconds.')
    small_discussion()

    print('Starting to solve the RDE without the error representation formula, where we solve for all levels')

    def solver(partition_):
        return lo.solve_fixed_full(x=x, f=f, y_0=y_0, N=N_min, partition=partition_, atol=atol, rtol=rtol,
                                   method=method, compute_bound=False, N_sol=N_max, verbose=verbose)[0]

    partition, y = lo.solve_error_tolerance(solver=solver, n=n, T=T, atol=atol, rtol=rtol)
    print(f'Finished solving the RDE without the error representation formula, where we solve only for the first level.'
          f' It took {time.perf_counter() - tic} seconds.')
    small_discussion()


def stupid_flow(x, f, y_0, s, T=1, N=2, n=200, h=3e-04):
    """
    Stupid reference computation of the flow to check the correctness of the true computation.
    :param x:
    :param f:
    :param y_0:
    :param s:
    :param T:
    :param N:
    :param n:
    :param h:
    :return:
    """
    dim_y = len(y_0)
    partition = np.linspace(s, T, n)
    Psi = np.zeros((dim_y, dim_y))
    y = lo.solve_fixed(x, f, y_0, N, partition)[0][:, -1]
    for i in range(dim_y):
        perturbation = np.zeros(dim_y)
        perturbation[i] = 1
        y_pert = lo.solve_fixed(x, f, y_0 + h*perturbation, N, partition)[0][:, -1]
        Psi[:, i] = (y_pert - y)/h
    return Psi
