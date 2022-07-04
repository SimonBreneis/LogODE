import numpy as np
import scipy
from scipy import integrate, special
from esig import tosig as ts
import matplotlib.pyplot as plt
import p_var
import time
import logode as lo
import examples as ex
import roughpath as rp
import vectorfield as vf
import cProfile
import sympy as sp
import tensoralgebra as ta
from fbm import FBM
import euler
import brownianroughtree as brt

'''
if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run("lo.solve_fully_adaptive_error_representation(x=ex.smooth_fractional_path(H=0.4, n=10000, save_level=3), f=ex.smooth_2x2_vector_field(N=3), y_0=np.array([0, 0]), N_min=2, N_max=3, atol=1e-03, rtol=1e-03)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("cumtime").print_stats(100)
    s.sort_stats("tottime").print_stats(100)
time.sleep(360000)
'''
x = ex.smooth_fractional_path(H=0.4, n=10000, save_level=3)
f = ex.smooth_2x2_vector_field(N=3)
partition, y, prop_loc_err, N = lo.solve_fully_adaptive_error_representation(x=x, f=f, y_0=np.array([0, 0]), N_min=2, N_max=3, atol=1e-04, rtol=1e-04)
fine_partition = np.empty(8*len(partition)-7)
fine_N = np.zeros(8*len(N), dtype=int)
for i in range(len(partition)-1):
    fine_partition[8*i:(8*i+9)] = np.linspace(partition[i], partition[i+1], 9)
    fine_N[8*i:8*(i+1)] = np.ones(8, dtype=int) * N[i]
good_y, _, _ = lo.solve_fixed(x=x, f=f, y_0=np.array([0, 0]), N=fine_N, partition=fine_partition, atol=1e-04/len(fine_partition), rtol=1e-04/len(fine_partition))
print(y[-1, :])
print(good_y[-1, :])
print(np.sum(prop_loc_err, axis=0))
plt.plot(y[:, 0], y[:, 1])
plt.title('Solution path')
plt.show()
c = ['red', 'orange', 'red', 'green']
prev_N = np.array([True, True, True, True])
for i in range(len(partition)-1):
    if prev_N[N[i]-1]:
        prev_N[N[i]-1] = False
        plt.plot(np.array([partition[i], partition[i+1]]), np.array([partition[i+1]-partition[i], partition[i+1]-partition[i]]), color=c[N[i]-1], label=f'N={N[i]}')
    else:
        plt.plot(np.array([partition[i], partition[i + 1]]),
                   np.array([partition[i + 1] - partition[i], partition[i + 1] - partition[i]]), color=c[N[i] - 1])
plt.title('Length and degree of partition intervals')
plt.xlabel('Time')
plt.ylabel('Length of interval')
plt.yscale('log')
plt.legend(loc='best')
plt.show()
print('Finished!!')
time.sleep(360000)


# x, f, y_0, _, _, _, _, _, _ = ex.ex_smooth_path()
x = ex.unit_circle(param=4, N=4)
# x = brt.load_brownian_rough_tree('brownian rough trees/tree 0')
f = ex.smooth_2x2_vector_field(N=4)
# x = ex.smooth_3d_path(N=4)
# f = ex.linear_2x3_vector_field(N=4)

'''
interval_vec = np.random.uniform(0, 1, (1000, 2))
interval_vec = np.sort(interval_vec, axis=1)
'''
depth_vec = np.random.randint(1, 10, 1000)
k_vec = np.array([np.random.randint(0, 2**depth_vec[i]) for i in range(1000)])
interval_vec = np.empty((1000, 2))
for i in range(1000):
    interval_vec[i, 0] = k_vec[i] * 2.**(-depth_vec[i])
    interval_vec[i, 1] = (k_vec[i]+1) * 2.**(-depth_vec[i])

N_vec = np.array([1, 2, 3, 4])

errors = np.empty((4, 1000))
better_errors = np.empty((4, 1000))
predictions_0 = np.empty((4, 1000))
predictions_1 = np.empty((4, 1000))
predictions_2 = np.empty((4, 991, 1000))
predictions_3 = np.empty((4, 991, 1000))
predictions_4 = np.empty((4, 1000, 1000))
relative_errors_0 = np.empty((4, 1000))
relative_errors_1 = np.empty((4, 1000))
relative_errors_2 = np.empty((4, 991, 1000))
relative_errors_3 = np.empty((4, 991, 1000))
relative_errors_4 = np.empty((4, 1000, 1000))
times = np.empty((4, 1000))
pred_times = np.empty((4, 1000))

# lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_vec[-1], partition=np.linspace(0, 1, n_vec[0]+1), atol=1e-04/n_vec[0], rtol=1e-04/n_vec[0])


for j in range(1000):
    y_0 = np.random.uniform(-5, 5, 2)
    for i in range(len(N_vec)):
        print(j, i)
        s = interval_vec[j, 0]
        t = interval_vec[j, 1]

        approx, _ = lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_vec[i], partition=np.array([s, t]), atol=1e-03*(t-s), rtol=1e-03*(t-s))
        better, _ = lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_vec[i], partition=np.linspace(s, t, 3), atol=1e-03*(t-s)/2, rtol=1e-03*(t-s)/2)
        true, _ = lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_vec[i], partition=np.linspace(s, t, 33), atol=1e-03*(t-s)/32, rtol=1e-03*(t-s)/32)
        errors[i, j] = ta.l1(true[:, -1]-approx[:, -1])
        better_errors[i, j] = ta.l1(true[:, -1] - better[:, -1])
        '''
        tic = time.perf_counter()
        lo.solve_fixed(x=x, f=f, y_0=y_0, N=N_vec[i], partition=np.array([s, t]), atol=1e-03 * (t - s),
                       rtol=1e-03 * (t - s))
        times[i, j] = time.perf_counter() - tic
        tic = time.perf_counter()
        f.apply(x.logsig(s, t, N_vec[i]))(y_0)
        pred_times[i, j] = time.perf_counter() - tic
        '''
for k in range(1000):
    for i in range(len(N_vec)):
        print(k, i)
        predictions_0[i, k] = errors[i, k] * 2**(-(N_vec[i-1]+1)/1-1)
        relative_errors_0[i, k] = better_errors[i, k] / predictions_0[i, k]
        '''
        predictions_0[i - 1, k] = N_vec[i - 1] * 2 * times[i - 1, k]
        predictions_1[i - 1, k] = pred_times[i, k] / pred_times[i - 1, k] * times[i - 1, k]
        relative_errors_0[i-1, k] = times[i, k] / predictions_0[i-1, k]
        relative_errors_1[i-1, k] = times[i, k] / predictions_1[i-1, k]
        '''
for j in range(1000):
    for k in range(1000):
        for i in range(len(N_vec)):
            print(j, k, i)
            '''
            predictions_4[i-1, j, k] = times[i, j] / times[i-1, j] * times[i-1, k]
            relative_errors_4[i-1, j, k] = times[i, k] / predictions_4[i-1, j, k]
            '''
            predictions_4[i, j, k] = better_errors[i, j] / errors[i, j] * errors[i, k]
            relative_errors_4[i, j, k] = better_errors[i, k] / predictions_4[i, j, k]
            if j <= 990:
                p = np.empty(10)
                '''
                predictions_2[i-1, j, k] = 1
                for l in range(10):
                    predictions_2[i-1, j, k] *= times[i, j+l] / times[i-1, j+l]
                    p[l] = times[i, j+l] / times[i-1, j+l]
                predictions_2[i-1, j, k] = predictions_2[i-1, j, k]**0.1 * times[i-1, k]
                predictions_3[i - 1, j, k] = np.median(p) * times[i-1, k]
                relative_errors_2[i-1, j, k] = times[i, k] / predictions_2[i-1, j, k]
                relative_errors_3[i - 1, j, k] = times[i, k] / predictions_3[i - 1, j, k]
                '''
                predictions_2[i, j, k] = 1
                for l in range(10):
                    predictions_2[i, j, k] *= better_errors[i, j+l] / errors[i, j+l]
                    p[l] = better_errors[i, j+l] / errors[i, j+l]
                predictions_2[i, j, k] = predictions_2[i, j, k]**0.1 * errors[i, k]
                predictions_3[i, j, k] = np.median(p) * errors[i, k]
                relative_errors_2[i, j, k] = better_errors[i, k] / predictions_2[i, j, k]
                relative_errors_3[i, j, k] = better_errors[i, k] / predictions_3[i, j, k]
'''
print('Actual average relative log-increase, 1->2:', np.average(np.log(times[1, :]/times[0, :])))
print('Actual standard deviation, 1->2:', np.std(np.log(times[1, :]/times[0, :])))
print('Actual average relative log-increase, 2->3:', np.average(np.log(times[2, :]/times[1, :])))
print('Actual standard deviation, 2->3:', np.std(np.log(times[2, :]/times[1, :])))
print('Actual average relative log-increase, 3->4:', np.average(np.log(times[3, :]/times[2, :])))
print('Actual standard deviation, 3->4:', np.std(np.log(times[3, :]/times[2, :])))
'''
print('Actual average relative log-increase, 1:', np.average(np.log(better_errors[0, :]/errors[0, :])))
print('Actual standard deviation, 1:', np.std(np.log(better_errors[0, :]/errors[0, :])))
print('Actual average relative log-increase, 2:', np.average(np.log(better_errors[1, :]/errors[1, :])))
print('Actual standard deviation, 2:', np.std(np.log(better_errors[1, :]/errors[1, :])))
print('Actual average relative log-increase, 3:', np.average(np.log(better_errors[2, :]/errors[2, :])))
print('Actual standard deviation, 3:', np.std(np.log(better_errors[2, :]/errors[2, :])))
print('Actual average relative log-increase, 4:', np.average(np.log(better_errors[3, :]/errors[3, :])))
print('Actual standard deviation, 4:', np.std(np.log(better_errors[3, :]/errors[3, :])))
log_fraction_0 = np.log(relative_errors_0)
l1_error_0 = np.abs(log_fraction_0)
l1_average_0 = np.average(l1_error_0, axis=-1)
l1_std_0 = np.std(l1_error_0, axis=-1)
l1_med_0 = np.median(l1_error_0, axis=-1)
print('l1 average:', l1_average_0)
print('l1 standard deviation:', l1_std_0)
print('l1 median:', l1_med_0)
'''
log_fraction_1 = np.log(relative_errors_1)
l1_error_1 = np.abs(log_fraction_1)
l1_average_1 = np.average(l1_error_1, axis=-1)
l1_std_1 = np.std(l1_error_1, axis=-1)
l1_med_1 = np.median(l1_error_1, axis=-1)
print('l1 average:', l1_average_1)
print('l1 standard deviation:', l1_std_1)
print('l1 median:', l1_med_1)
'''
log_fraction_2 = np.log(relative_errors_2)
l1_error_2 = np.sum(np.abs(log_fraction_2), axis=-1) / 990
l1_average_2 = np.average(l1_error_2, axis=-1)
l1_std_2 = np.std(l1_error_2, axis=-1)
l1_med_2 = np.median(l1_error_2, axis=-1)
print('l1 average:', l1_average_2)
print('l1 standard deviation:', l1_std_2)
print('l1 median:', l1_med_2)
log_fraction_3 = np.log(relative_errors_3)
l1_error_3 = np.sum(np.abs(log_fraction_3), axis=-1) / 990
l1_average_3 = np.average(l1_error_3, axis=-1)
l1_std_3 = np.std(l1_error_3, axis=-1)
l1_med_3 = np.median(l1_error_3, axis=-1)
print('l1 average:', l1_average_3)
print('l1 standard deviation:', l1_std_3)
print('l1 median:', l1_med_3)
log_fraction_4 = np.log(relative_errors_4)
l1_error_4 = np.sum(np.abs(log_fraction_4), axis=-1) / 999
l1_average_4 = np.average(l1_error_4, axis=-1)
l1_std_4 = np.std(l1_error_4, axis=-1)
l1_med_4 = np.median(l1_error_4, axis=-1)
print('l1 average:', l1_average_4)
print('l1 standard deviation:', l1_std_4)
print('l1 median:', l1_med_4)
time.sleep(360000)
print((time.perf_counter()-tic)/10000)
print(times)
time.sleep(360000)
plt.loglog(n_vec, times[0, :], 'b-', label=r'$N=1$')
plt.loglog(n_vec, times[1, :], 'g-', label=r'$N=2$')
plt.loglog(n_vec, times[2, :], 'y-', label=r'$N=3$')
plt.loglog(n_vec, times[3, :], 'r-', label=r'$N=4$')

exp, const, _, _, _ = ex.log_linear_regression(x=n_vec, y=times[0, :])
print(N_vec[0], exp, const)
plt.loglog(n_vec, const * n_vec**exp, 'b--')

exp, const, _, _, _ = ex.log_linear_regression(x=n_vec, y=times[1, :])
print(N_vec[1], exp, const)
plt.loglog(n_vec, const * n_vec**exp, 'g--')
print(np.average(times[1, :] / times[0, :]))

exp, const, _, _, _ = ex.log_linear_regression(x=n_vec, y=times[2, :])
print(N_vec[2], exp, const)
plt.loglog(n_vec, const * n_vec**exp, 'y--')
print(np.average(times[2, :] / times[1, :]))

exp, const, _, _, _ = ex.log_linear_regression(x=n_vec, y=times[3, :])
print(N_vec[3], exp, const)
plt.loglog(n_vec, const * n_vec**exp, 'r--')
print(np.average(times[3, :] / times[2, :]))

plt.xlabel('Number of intervals')
plt.ylabel('Time in seconds')
plt.legend(loc='best')
plt.title('Computational time of the Log-ODE method\nfor an oscillatory path and a linear vector field')
plt.show()



for i in range(10):
    tree = brt.initialize_brownian_rough_tree(dim=2, T=1, has_time=False, depth=15, accuracy=20, N=4, delete=True)
    tree.save(directory=f'brownian rough trees/tree {i}')

time.sleep(36000000)

tree = brt.initialize_brownian_rough_tree()
tree.save(directory='brownian rough trees/gay_dir')
new_tree = brt.load_brownian_rough_tree('brownian rough trees/gay_dir')
print(tree.sig(0.235, 0.634, 4))
print(new_tree.sig(0.235, 0.634, 4))
time.sleep(3600000)

a = np.array([1])
b = np.array([2])
with open('gay.npy', 'wb') as f:
    np.save(f, a)
    np.save(f, b)
with open('gay.npy', 'rb') as f:
    c = np.load(f)
    d = np.load(f)
print(a, b, c, d)

tens = ta.NumericTensor([1, np.array([2, 3]), np.array([[4, 5], [6, 7]])])
tens.save('here.npy')
new_tens = ta.load_tensor('here.npy', N=2)
print(tens)
print(new_tens)
time.sleep(3600000)

tree = brt.initialize_brownian_rough_tree(depth=15, accuracy=15)
print('finished!')
ex.accuracy_of_level_N_signature(N=4, log_n=25, m=10)
#print(ex.accuracy_of_signature(m=10))
time.sleep(360000)

ex.discussion(ex=4, show=True)

print(ex.adaptive_error_fixed_N_discussion(ex=6.4, show=True, save=True, speed=-1))
print('Finished!')
time.sleep(360000)


ex.error_representation_discussion(ex=0.4, show=True, save=True, speed=-1)


def stupid_flow(x, f, y_0, s, T=1, N=2, n=200, h=3e-04):
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


x = ex.unit_circle()
f = ex.smooth_2x2_vector_field()
# x = ex.rough_fractional_Brownian_path(H=0.5, n=1000*2000, dim=2, T=1., p=2.2, var_steps=15, norm=ta.l1, save_level=3)
print('constructed')

'''
ex.ex_fBm_path(solver='assr', N=2, x=x, f=f, speed=0.1)
print('finished')
time.sleep(3600)
'''

tic = time.perf_counter()
true_solution, _, true_time = ex.ex_smooth_path(N=3, n=1000, x=x, f=f, speed=0)
toc = time.perf_counter()
print(f'solving for true solution: {toc-tic}')
true_solution = true_solution[:, -1]
print(true_solution, true_time)
N_vec = np.array([1, 2, 3])
n_vec = np.array([16, 23, 32, 45, 64, 91, 128, 181, 256, 362])
abs_global_error = np.zeros(len(n_vec))
abs_true_error = np.zeros(len(n_vec))
abs_difference = np.zeros(len(n_vec))
abs_relative = np.zeros(len(n_vec))
for i in range(len(N_vec)):
    print(f'N={N_vec[i]}')
    for j in range(len(n_vec)):
        print(f'n={n_vec[j]}')
        y, local_errors, tictoc = ex.ex_smooth_path(solver='fssr', N=N_vec[i], n=n_vec[j], x=x, f=f, speed=0)
        global_error = np.sum(local_errors, axis=0)
        true_error = true_solution - y[-1, :]
        abs_global_error[j] = ta.l1(global_error)
        abs_true_error[j] = ta.l1(true_error)
        abs_difference[j] = ta.l1(true_error - global_error)
        abs_relative[j] = abs_global_error[j]/abs_true_error[j]
        print(abs_global_error[j], abs_true_error[j], abs_difference[j], abs_relative[j], tictoc)
    plt.loglog(n_vec, abs_global_error, label='estimated global error')
    plt.loglog(n_vec, abs_true_error, label='true global error')
    plt.loglog(n_vec, abs_difference, label='true - global error')
    plt.title('Smooth path, smooth vector field\nComparing true and estimated global errors')
    plt.legend(loc='best')
    plt.xlabel('Number of intervals')
    plt.ylabel('Error')
    plt.show()


ex.discussion(ex=0, show=False, save=True)
time.sleep(360000)
ex.discussion(ex=1.02785, show=False, save=True)
time.sleep(360000)
ex.ex_third_level(N=0, plot=False, sig_steps=10000, atol=1e-06, rtol=1e-05, sym_vf=True, param=1000)
time.sleep(360000)
ex.discussion(ex=2.01562, show=False, save=True)
time.sleep(360000)
ex.discussion(ex=0, show=False, save=True)
time.sleep(360000)
ex.ex_third_level(N=0, plot=False, sig_steps=2000, sym_vf=True, param=64, atol=1e-06, rtol=1e-05)
time.sleep(360000)
ex.discussion(ex=0.3, show=False, save=True)
time.sleep(360000)
ex.discussion(ex=0.4, show=False, save=True)
time.sleep(360000)
ex.discussion(ex=0.5, show=False, save=True)
time.sleep(360000)
ex.discussion(ex=0.75, show=False, save=True)
time.sleep(3600)
ex.discussion(ex=1, save=True, show=False)
time.sleep(3600)
ex.discussion(ex=1.02785, save=True, show=False)
time.sleep(3600)
ex.discussion(ex=1.23, save=True, show=False)
time.sleep(3600)
ex.discussion(ex=1.23, save=False, show=False)
time.sleep(3600)


'''
if __name__ == '__main__':
    import cProfile, pstats
    cProfile.run("ex.smooth_path(plot=True, N=2, n=300, sym_path=True, sym_vf=True)", "{}.profile".format(__file__))
    s = pstats.Stats("{}.profile".format(__file__))
    s.strip_dirs()
    s.sort_stats("time").print_stats(100)
time.sleep(3600)
'''
