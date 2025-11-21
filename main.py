import numpy as np
import time
from gaussian_mixture import gaussian_mixture

from kde import kernel_density

from tensor_estimate import vrs_prediction

from utils import plot_3d_samples, plot_2d_samples, energy_distance,compare_mean_var

dim = 7
N_train = 10000
N_samples = 1000

##############tuning parameter selection
lr_rec = 0
kde_rec = 0
MM = 10

if N_train < 2 ** dim * MM:
    print('insufficient data')
    LL = 1
else:
    LL = 2

print(MM, LL)
tensor_shape = [LL for _ in range(dim)]

tensor_shape[0] = MM

#########################################

distribution = gaussian_mixture(dim, [1, -1], [1, 0.5])
for rr in range(1):
    X_train = distribution.generate(N_train)
    N_test = 1000
    X_test = distribution.generate(N_test)

    #############density transform
    vrs_model = vrs_prediction(tensor_shape, dim, MM, X_train)
    y_lr = vrs_model.predict(X_test)
    y_true = np.array([distribution.density_value(xx) for xx in X_test])
    lr_rec += np.linalg.norm(y_lr - y_true, 2) ** 2 / np.linalg.norm(y_true, 2) ** 2
    print('lr transform error = ', lr_rec / (rr + 1))

    y_kde = kernel_density().compute(dim, X_train, X_test)
    err_kde = np.linalg.norm(y_kde - y_true, 2) ** 2 / np.linalg.norm(y_true, 2) ** 2

    kde_rec += err_kde

    print('kde error = ', kde_rec / (rr + 1))




#Sampling:
start = time.perf_counter()
samples = vrs_model.sampling_N_ori_domain(N_samples)
elapsed = time.perf_counter() - start

#accuracy evaluation
ED2 = energy_distance(X_train, samples)
print("Energy distance^2:", ED2)

(mean_train, mean_samples, var_train, var_samples,
 ss_mean_diff, ss_var_diff, ss_total,) = compare_mean_var(X_train, samples)
print("SS(mean diff) / dim =", ss_mean_diff / dim)
print("SS(var  diff) / dim =", ss_var_diff / dim)
print("SS(total) / dim     =", ss_total / dim)

#timing
print(f"Sampling {N_samples} points in dim={dim} took {elapsed:.4f} seconds")
print(f"time per point ≈ {elapsed / N_samples:.6e} seconds")

#plot the samples when d=2 or 3
if dim == 2:
    plot_2d_samples(X_train, samples)
if dim == 3:
    plot_3d_samples(X_train, samples)