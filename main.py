from functools import partial
from qiskit.quantum_info import SparsePauliOp, random_statevector

import scipy.sparse.linalg as ssla
from scipy import sparse
import pandas as pd
import scipy

import matplotlib.pyplot as plt
import numpy as np

from utils import *
# from lightcone import *
from spin_ham import *
from trotter import *
from bounds import *
# from noise import *

figs_dir, data_dir = './figs', './data'

prefix, suffix = 'TFI', 'n'
cmm_data = pd.read_csv(f'{data_dir}/{prefix}_nested_commutator_norm_max=300_{suffix}.csv')


prefix, suffix = 'TFI', 'n'
# prefix, suffix = 'NN', 'n'
n_max = 11
# t = 0.2
t = 0.5
J, h = 1, 0.2
eps = 0.005 # 1e-3
n_list_bnd = [4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20,  26,  33,  42,  54,  69,  88, 112, 142, 181, 231, 295]
# n_list_bnd = [4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 21, 25, 30, 40, 54, 70, 94, 124, 164, 216, 286, 378, 499]
n_list_emp = np.arange(4, n_max+1)
print('n_list_emp: ', n_list_emp)
r_start, r_end = 1, 100

data_keys = ['worst', 'singl', 'multi', 'n']
bnd_data = dict([(k, []) for k in data_keys])
emp_data = dict([(k, []) for k in data_keys])
bnd_data['n'], emp_data['n'] = n_list_bnd, n_list_emp

for n in n_list_bnd:
    # hnn = Nearest_Neighbour_1d(n, Jx=J, Jy=J, Jz=J, hx=h, hy=0, hz=0, pbc=False, verbose=False)
    print(f'-------------- n = {n} --------------')
    bnd_worst_err_r = partial(measure_error, h_list=None, t=t, exact_U=None, type='worst_loose_bound', coeffs=['singl', n, cmm_data])
    bnd_singl_err_r = partial(measure_error, h_list=None, t=t, exact_U=None, type='lightcone_bound', coeffs=['singl', n, cmm_data])
    bnd_multi_err_r = partial(measure_error, h_list=None, t=t, exact_U=None, type='lightcone_bound', coeffs=['multi', n, cmm_data])

    bnd_data['worst'].append(binary_search_r(r_start, r_end, eps, bnd_worst_err_r, verbose=True))
    bnd_data['singl'].append(binary_search_r(r_start, r_end, eps, bnd_singl_err_r, verbose=True))
    bnd_data['multi'].append(binary_search_r(r_start, r_end, eps, bnd_multi_err_r, verbose=True))

for n in n_list_emp:
    print(f'-------------- n_emp = {n} --------------')
    hnn = Nearest_Neighbour_1d(n, Jx=0, Jy=0, Jz=J, hx=h, hy=0, hz=0, pbc=False, verbose=False)
    # hnn = Nearest_Neighbour_1d(n, Jx=J, Jy=J, Jz=J, hx=h, hy=0, hz=0, pbc=False, verbose=False)
    singl_ob = SparsePauliOp.from_sparse_list([('Z', [0], 1)], n).to_matrix()
    multi_ob = SparsePauliOp.from_sparse_list([('Z', [i], 1/n) for i in range(0, n)], n).to_matrix()
    # multi_ob = SparsePauliOp.from_sparse_list([('Y', [i], 1) for i in range(0, n)], n).to_matrix()
    # multi_ob = SparsePauliOp.from_sparse_list([('ZZ', [i,i+1], 1) for i in range(0, n-1)], n).to_matrix()
    # multi_ob = SparsePauliOp.from_sparse_list([(random.choice(['X','Y','Z']), [i], 1) for i in range(0, n)], n).to_matrix()
    # multi_ob = multi_ob / np.linalg.norm(multi_ob, ord=2)
    par_group = [h.to_matrix(True) for h in hnn.ham_par]
    xyz_group = [h.to_matrix(True) for h in hnn.ham_xyz]

    exact_U = scipy.linalg.expm(-1j * t * sum([h.to_matrix() for h in hnn.ham_par]))
    # verfiy the exact_U
    assert np.allclose(exact_U, scipy.linalg.expm(-1j * t * sum([h.to_matrix() for h in hnn.ham_xyz])))

    emp_worst_err_r = partial(measure_error, h_list=par_group, t=t, exact_U=exact_U, type='worst_empirical')
    emp_singl_err_r = partial(measure_error, h_list=par_group, t=t, exact_U=exact_U, ob=singl_ob, type='worst_ob_empirical')
    emp_multi_err_r = partial(measure_error, h_list=par_group, t=t, exact_U=exact_U, ob=multi_ob, type='worst_ob_empirical')

    emp_data['worst'].append(binary_search_r(r_start, r_end, eps, emp_worst_err_r, verbose=True))
    emp_data['singl'].append(binary_search_r(r_start, r_end, eps, emp_singl_err_r, verbose=True))
    emp_data['multi'].append(binary_search_r(r_start, r_end, eps, emp_multi_err_r, verbose=True))
print('emp_data: ', emp_data)
# print(pd.DataFrame(emp_data))
# save to csv
pd.DataFrame(emp_data).to_csv(f'{data_dir}/{prefix}_emp_lightcone_max={n_max}_{suffix}.csv', index=False)
pd.DataFrame(bnd_data).to_csv(f'{data_dir}/{prefix}_bnd_lightcone_max={n_max}_{suffix}.csv', index=False)


factor = 2
def exp_count_LC(r, n_qubits, n_terms):
    exp_count = 0
    for i in range(1, r+1):
        # print('i: ', i)
        if i < int(n_qubits/2):
            exp_count += (4 * i - 1) * 2    
        elif i == int(n_qubits/2):
            exp_count += (4 * i - 1) * 2  - 1
        else:
            exp_count += n_terms * 2
    return exp_count

cost_st_list = [2 * factor*bnd_data['n'][i] * r for i, r in enumerate(bnd_data['worst'])]
cost_lc_list_s = [exp_count_LC(r, bnd_data['n'][i], factor*bnd_data['n'][i]) for i, r in enumerate(bnd_data['singl'])]
cost_lc_list_m = [2 * factor*bnd_data['n'][i] * r for i, r in enumerate(bnd_data['multi'])]
cost_em_list_s = [exp_count_LC(r, emp_data['n'][i], factor*emp_data['n'][i]) for i, r in enumerate(emp_data['singl'])]
cost_em_list_m =  [2 * factor*emp_data['n'][i] * r for i, r in enumerate(emp_data['multi'])]

fig, ax2 = plt.subplots(figsize=(9, 6), layout='tight')
# ob_string = 'XIII'
ax2.plot(n_list_bnd, cost_st_list, 's', color='#E64B35FF', label=r'Worst-case', markersize=10)
plot_fit(ax2, n_list_bnd[10:], cost_st_list[10:], var=suffix, offset=1.12)
# # ax2.plot(n_list, cost_st_found_list, 's', label=r'Worst-case (Ref \textcolor{blue}{[13]})', markersize=10)
ax2.plot(n_list_bnd, cost_lc_list_s, '^', color='#0A75C7', label='Single-ob (Theoretical)', markersize=12)
plot_fit(ax2, n_list_bnd[15:], cost_lc_list_s[15:], var=suffix, offset=1.12)
ax2.plot(n_list_bnd, cost_lc_list_m, '^', color='#F39B7FFF', label='Multip-ob (Theoretical)', markersize=12)
plot_fit(ax2, n_list_bnd[15:], cost_lc_list_m[15:], var=suffix, offset=1.12)
ax2.plot(n_list_emp, cost_em_list_s, 'o', color='#0A75C7', label='Single-ob (Empirical)')
plot_fit(ax2, n_list_emp[-2:], cost_em_list_s[-2:], var=suffix, offset=1.12)
ax2.plot(n_list_emp, cost_em_list_m, 'o', color='#F39B7FFF', label='Multip-ob (Empirical)')
plot_fit(ax2, n_list_emp[3:], cost_em_list_m[3:], var=suffix, offset=1.12)
# ax2.plot(n_list, r_lc_found_list, '-*', label='Lightcone (bound)', markeredgecolor='k')
# Add labels and a legend
ax2.set_xlabel('Number of qubits')
ax2.set_ylabel('Number of exponentials')
ax2.set_title(fr'NN Heisenberg (fix t={t}, $\epsilon$={eps}) PF2')  
ax2.loglog(); ax2.grid(); ax2.legend()
fig.savefig(f'{figs_dir}/lightcone_NN1d_n={n}_eps={eps}_t={t}.pdf', bbox_inches='tight')