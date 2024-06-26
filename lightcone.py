# Various utility functions used by the scripts
import itertools, random, sys

from cmath import exp
from math import ceil, floor, log
import numpy as np
from numpy.linalg import matrix_power
from jax.scipy.linalg import expm
# from scipy.linalg import expm
np.set_printoptions(precision=6)

from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.quantum_info import SparsePauliOp

import scipy.sparse.linalg as ssla
from scipy import sparse
from scipy.optimize import curve_fit
# import scipy
import multiprocessing

import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp

def analy_st_bound(r, n, J, h, t, ob_type='single'):
    if ob_type == 'single':
        return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r
    elif ob_type == 'multi':
        # return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r  * n
        return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r 
    else:
        raise ValueError('ob_type should be either single or multi')

def analy_lc_bound(r, n, J, h, t, ob_type='single', verbose=False):
    err_bound = 0
    for i in range(1, r+1):
        if ob_type == 'single':
            # print('single')
            n_lc = min(i*2, n)
            err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose=verbose) 
        elif ob_type == 'multi':   
            for j in range(0, n):
                n_lc = min(min(n-j, i*2) + min(j, 2*i), n)
                # err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
                err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose=verbose) / n
        else:
            raise ValueError('ob_type should be either single or multi')

    return err_bound

def analytic_loose_commutator_bound(n, J, h, dt, pbc=False, verbose=False):
    if pbc:
        c1 = 16*J**2*h*(n) + 4*J**2*h*(n)
        c2 = 8*(n)*J**2*h
    else:
        # c1 = 16*J**2*h*(n-1) + 4*J**2*h*(n-2)
        # c2 = 8*(n-1)*J**2*h
        if n % 2 == 1:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
        else:
            c1 = 4*J*h**2*(n-1) + 4*J**2*h*(n-2)
            c2 = 4*J*h**2*(n-1) + 4*J**2*h*(n-1)

    if verbose: print(f'c1 (analy)={c1}, c2={c2}')
    analytic_error_bound = c1 * dt**3 / 12 + c2 * dt**3 / 24
    return analytic_error_bound

def bin_search_r(n, J, h, t, epsilon, search_precision, pf_type='standard', ob_type='single', verbose=False):
    r_start = 1
    r_end = 5 * n
    error_list, r_list, exp_count_list = [], [], []
    error = 2
    print(f'=========pf_type={pf_type}, ob_type={ob_type}=========')

    if pf_type == 'empirical':
        r = r_start
        tfI = transverse_field_ising_1d(n, J, h, [0], t=t, initialize=False)
        exact_U = jax.scipy.linalg.expm(-1j * t * tfI.H_mat.toarray())
        appro_U = standard_trotter(tfI.H_parity, t, r)
        # appro_U = standard_trotter(tfI.H_parity[::-1], t, r_start)
        if ob_type == 'single':
            magn_op = SparsePauliOp.from_sparse_list([('Z', [0], 1)], n)
            # # print('single observable error (jax): ', ob_error(magn_op, exact_U, appro_U))
            # exact_ob = exact_U.conj().T @ magn_op.to_matrix() @ exact_U 
            # appro_ob = appro_U.conj().T @ magn_op.to_matrix() @ appro_U
            error = ob_error(magn_op, exact_U, appro_U)
            print(f'r={r_start}, single observable error (jax): ', error)
            exp_count = exp_count_LC(r, n, 2*n)
        elif ob_type == 'multi':
            magn_op = SparsePauliOp.from_sparse_list([('Z', [i], 1) for i in range(0, n)], n)/n
            error = ob_error(magn_op, exact_U, appro_U)
            exp_count = 2 * (2*n) * r
    if error < epsilon: 
        r_list.append(r_start)
        error_list.append(error)
        exp_count_list.append(exp_count)
        print(f'r={r_start}; error={error:.6f}; exp_count={exp_count}')
        r_found = r
    else:
        while r_start < r_end - 1: 
        # while abs(error - epsilon) > epsilon * search_precision and r_end - r_start != 1:
            r = floor((r_start + r_end) / 2)
            r_list.append(r)
            if pf_type == 'standard':
                error = analy_st_bound(r, n, J, h, t, ob_type=ob_type)
                exp_count = 2 * (2*n) * r
            elif pf_type == 'lightcone':
                error = analy_lc_bound(r, n, J, h, t, ob_type=ob_type, verbose=False)
                if ob_type == 'single':
                    exp_count = exp_count_LC(r, n, 2*n)
                elif ob_type == 'multi':
                    exp_count = 2 * (2*n) * r
            elif pf_type == 'empirical':
                tfI = transverse_field_ising_1d(n, J, h, [0], t=t, initialize=False)
                exact_U = expm(-1j * t * tfI.H_mat.toarray())
                # magn_op = SparsePauliOp.from_sparse_list([('Z', [i], 1) for i in range(0, n)], n)
                appro_U = standard_trotter(tfI.H_parity, t, r)
                if ob_type == 'single':
                    magn_op = SparsePauliOp.from_sparse_list([('Z', [0], 1)], n)
                    error = ob_error(magn_op, exact_U, appro_U)
                    exp_count = exp_count_LC(r, n, 2*n)
                elif ob_type == 'multi':
                    magn_op = SparsePauliOp.from_sparse_list([('Z', [i], 1) for i in range(0, n)], n)/n
                    error = ob_error(magn_op, exact_U, appro_U)
                    exp_count = 2 * (2*n) * r

            exp_count_list.append(exp_count)

            if verbose: print(f'r={r}; error={error:.6f}; exp_count={exp_count}')
            error_list.append(error)

            if error > epsilon: r_start = r
            else: r_end = r

            if r_end - r_start == 0: 
                print('precision warning!!!')
                # raise ValueError('Binary search failed. Please increase the search range.')
        print(f'-------- binary search end --------')
    
        # r_abs_err_dict = dict(zip(r_list, [abs(error-epsilon) for error in error_list]))
        # r_found = min(r_abs_err_dict, key=r_abs_err_dict.get)
        r_found = r_end 
    r_err_dict = dict(zip(r_list, error_list))

    return r_list, error_list, exp_count_list, r_found, r_err_dict

# def linear_loglog_fit(x, y, verbose=False):
#     # Define the linear function
#     def linear_func(x, a, b):
#         return a * x + b

#     log_x = np.array([log(n) for n in x])
#     log_y = np.array([log(cost) for cost in y])
#     # Fit the linear function to the data
#     params, covariance = curve_fit(linear_func, log_x, log_y)
#     # Extract the parameters
#     a, b = params
#     # Predict y values
#     y_pred = linear_func(log_x, a, b)
#     # Print the parameters
#     if verbose: print('Slope (a):', a, 'Intercept (b):', b)
#     exp_y_pred = [exp(cost) for cost in y_pred]

#     return exp_y_pred, a, b

# def plot_fit(ax, x, y, var='n', offset=1.07, verbose=True):
#     y_pred_em, a_em, b_em = linear_loglog_fit(x, y)
#     if verbose: print(f'a_em: {a_em}; b_em: {b_em}')
#     text_a_em = "{:.2f}".format(round(abs(a_em), 4))
#     y_pred_em = [exp(cost) for cost in a_em*np.array([log(n) for n in x]) + b_em]
#     ax.plot(x, y_pred_em, 'k--', linewidth=1)
#     ax.annotate(r'$O(%s^{%s})$' % (var, text_a_em), xy=(x[-1], np.real(y_pred_em)[-1]), xytext=(x[-1]*offset, np.real(y_pred_em)[-1]))
# =====================================================
# =====================================================
def partial_trotter(approx_U, tau, a, pauli_strs, h, J, parity=1, verbose=True):
    if verbose: print(pauli_strs)
    if len(pauli_strs) == 1:
        approx_U = jax.scipy.linalg.expm(-1j * tau * a * SparsePauliOp(pauli_strs, [h]).to_matrix(False)) @ approx_U
    elif len(pauli_strs) == 2:
        approx_U = jax.scipy.linalg.expm(-1j * tau * a * SparsePauliOp(pauli_strs, [h, J]).to_matrix(False)) @ approx_U
    elif len(pauli_strs) == 4:
        if parity == 1:
            approx_U = jax.scipy.linalg.expm(-1j * tau * a * SparsePauliOp(pauli_strs[:2], [J, h]).to_matrix(False)) @ approx_U
            approx_U = jax.scipy.linalg.expm(-1j * tau * a * SparsePauliOp(pauli_strs[-2:], [h, J]).to_matrix(False)) @ approx_U
        elif parity == 2:
            approx_U = jax.scipy.linalg.expm(-1j * tau * a * SparsePauliOp(pauli_strs[-2:], [h, J]).to_matrix(False)) @ approx_U
            approx_U = jax.scipy.linalg.expm(-1j * tau * a * SparsePauliOp(pauli_strs[:2], [J, h]).to_matrix(False)) @ approx_U
    else:
        raise ValueError('Not implemented yet.')
    # Ham = sum([SparsePauliOp(p_str, [h]).to_matrix(False) if set(list(p_str))-{'I'}=={'X'} else SparsePauliOp(p_str, [J]).to_matrix(False) for p_str in pauli_strs])
    # for p_str in list(reversed(pauli_strs)):
    # # for p_str in pauli_strs:
    #     print(p_str)
    #     if set(list(p_str))-{'I'} == {'X'}:
    #         gate = SparsePauliOp(p_str, [h]).to_matrix(False)
    #         approx_U = jax.scipy.linalg.expm(-1j * tau * a * gate) @ approx_U
    #     else:
    #         gate = SparsePauliOp(p_str, [J]).to_matrix(False)
    #         approx_U = jax.scipy.linalg.expm(-1j * tau * a * gate) @ approx_U
    return approx_U

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

def local_ob(ob_index, n):
    ## define the local observable
    single_ob = dict({'X': ob_index})
    print(f'single local observabel: {single_ob}')
    ob_string = 'X' * len(ob_index) + 'I' * (n - len(ob_index))
    # ob = SparsePauliOp(['I'*(ob_index - 1) + 'X' + 'I'*(ising1d.n_qubits - ob_index - 1)], [1])
    ob = SparsePauliOp([ob_string], [1])
    print('observable: ', ob)

    return ob, single_ob

def lightcone_bound_simplified(model, r, verbose=False):
    bounds = (model.r_saturate/r)**3 * np.array(model.lightcone_segment_error_bounds)
    if verbose: print('bound list: ', bounds)
    result = sum(bounds) + bounds[-1] * (r - len(bounds))
    print(f'r={r}, bound={result:.6f}')

    return result

def lightcone_bound(model, exp_list, t, r, ord=2, loose=True, verbose=False):
    # ising1d.H_dict
    lc_bound = 0
    lc_bound_r_list = []
    ob_norm = 1
    previous_pstrs = ['dummy']
    for pstrs in exp_list:
        # even/odd?
        h_list = [sum([model.H_dict[pstr] for pstr in pstrs[0]]), sum([model.H_dict[pstr] for pstr in pstrs[1]])]
        if verbose: print('h_list: ', h_list)
        # print('h_list: ', h_list)
        ## t/2r ???
        if pstrs == previous_pstrs:
            print(pstrs)
            lc_bound_r = lc_bound_r
        else:
            # print(h_list)
            if loose: 
                lc_bound_r = 2 * ob_norm * analytic_loose_commutator_bound(len(pstrs[0]), model.J, model.h, t/r)
            else:
                lc_bound_r = 2 * ob_norm * tight_bound([ham.to_matrix() for ham in h_list], ord, t/r, 1)
            lc_bound_r_list.append(lc_bound_r)

        if verbose: print(f'light cone error bound (one step) = {lc_bound_r}')
        lc_bound += lc_bound_r
        previous_pstrs = pstrs

    if verbose:
        print(f'Lightcone Trotter error bound: {lc_bound:.6f}')
        model.partition('parity')   
        if loose:
            print('Standard  Trotter bound: ', 2 * ob_norm * analytic_loose_commutator_bound(model.n_qubits, model.J, model.h, t/r))
        else:
            print('Standard  Trotter bound: ', 2 * ob_norm * tight_bound(model.H_parity, ord, t, r))

    return lc_bound, lc_bound_r_list

def lightcone_trotter(model, ob, r, t, empirical=True, verbose=False):
    Gamma = 2
    a = 1/2
    b = 0
    tau = t/r
    n = model.n_qubits
    approx_U_LC = np.eye(2**model.n_qubits)
    exp_count = 0
    exp_list = []
    max_LC = len(model.h_LC_decomp)
    if verbose: print(max_LC, 'max_LC')
    for j in range(1, r+1):
        if verbose: print(f'============== j={j} (r={r}); exp_count={exp_count} ==============')
        temp_list = []
        for v in range(1, Gamma+1):
            if v + b < n:
                temp_index = (v + b) / 2
            else:
                temp_index = int(n/2)
            if verbose: print(f'----v={v}, b={b}, (v+b)/2={temp_index}----')
            inner_temp_list = []
            if (v + b) % 2 != 0:
                # print(f'v + b = {v} + {b} is odd')
                # for gamma in range(1, ceil(temp_index)+1):
                for gamma in range(1, floor(temp_index)+1):
                    if 2*gamma < max_LC:
                        temp = model.h_LC_decomp[2*gamma]
                        # even_list = even_list + temp
                        if empirical:
                            approx_U_LC = partial_trotter(approx_U_LC, tau, a, temp, model.h, model.J, parity=v, verbose=verbose)
                        exp_count += len(temp)
                    else:
                        break
                if verbose: print('............')
                for gamma in range(1, ceil(temp_index)+1):
                    if 2*gamma-1 < max_LC:
                        temp = model.h_LC_decomp[2*gamma-1]
                        inner_temp_list = inner_temp_list + temp
                        if empirical:
                            approx_U_LC = partial_trotter(approx_U_LC, tau, a, temp, model.h, model.J, parity=v, verbose=verbose)
                        exp_count += len(temp)
                    else:
                        break
            else:
                for gamma in range(1, ceil(temp_index)+1):
                    if 2*gamma-1 < max_LC:
                        temp = model.h_LC_decomp[2*gamma-1]
                        if empirical:
                            approx_U_LC = partial_trotter(approx_U_LC, tau, a, temp, model.h, model.J, parity=v, verbose=verbose)
                        exp_count += len(temp)
                    else:
                        break
                if verbose: print('............')
                for gamma in range(1, floor(temp_index)+1):
                    if 2*gamma < max_LC:
                        temp = model.h_LC_decomp[2*gamma]
                        inner_temp_list = inner_temp_list + temp
                        if empirical:
                            approx_U_LC = partial_trotter(approx_U_LC, tau, a, temp, model.h, model.J, parity=v, verbose=verbose)
                        exp_count += len(temp)
                    else:
                        break
            temp_list.append(inner_temp_list)
        exp_list.append(temp_list)
        b = b + Gamma

    if verbose: print('exponential list: \n', exp_list)

    
    if empirical:
        error = ob_error(ob, model.exact_U, approx_U_LC)
        print(f'r={r}; empirical ob_error = {error:.6f}; exp_count(LC)={exp_count}')
    else:
        error = 2 
        if verbose: print(f'r={r}; exp_count(LC)={exp_count}; no Trotter error evaluated')

    return error, exp_count, exp_list

def standard_trotter(h_list, t, r, ord=2, verbose=False):
    """
    [todo higher order]
    Args:
        h_list (list): A list of Hamiltonian terms (even/odd parity partition).
        t (float): The time step.

    Returns:

    """
    # list_U = [ssla.expm(-1j * (t / r) * herm) for herm in ising_1d.h_list]
    if ord == 2:
        list_U = [jax.scipy.linalg.expm(-1j * (t / (2*r)) * herm.toarray()) for herm in h_list]
        if verbose: print('----expm Herm finished----')
        appro_U_dt = list_U[0] @ list_U[1]
        if verbose: print('----matrix product finished----')
        appro_U = jnp.linalg.matrix_power(appro_U_dt @ list_U[1] @ list_U[0], r)
        # appro_U = jnp.linalg.matrix_power(appro_U_dt, r)
        # appro_U = matrix_power(appro_U_dt, r)
        # appro_U = matrix_power(appro_U_dt.toarray(), r)
        if verbose: print('----matrix power finished----')
    return appro_U

def ob_error(ob, exact_U, appro_U, norm='spectral'):
    """
    Args:
        ob (Operator): The observable operator.
        exact_U
        approx_U: 

    Returns:

    """
    # exact_ob = exact_U @ ob.to_matrix() @ jax.numpy.linalg.inv(exact_U)
    # # exact_ob = exact_U @ ob.to_matrix() @ exact_U.conj().T
    # appro_ob = appro_U @ ob.to_matrix() @ jax.numpy.linalg.inv(appro_U)
    exact_ob =jax.numpy.linalg.inv(exact_U) @ ob.to_matrix() @ exact_U 
    # exact_ob = exact_U @ ob.to_matrix() @ exact_U.conj().T
    appro_ob = jax.numpy.linalg.inv(appro_U) @ ob.to_matrix() @ appro_U
    if norm == 'spectral':
        error = jnp.linalg.norm(exact_ob - appro_ob, ord=2)

    return error
    
def binary_search_r(model, ob, t, epsilon, search_precision, decompose='standard', type='empirical', r_max=20, verbose=False):

    if decompose == 'standard':   
        h_list = model.H_parity
    elif decompose == 'lightcone':
        h_list = model.h_LC_decomp
    # elif decompose == 'all':
    #     raise ValueError('Not implemented yet.')
    else:
        raise ValueError('Not implemented yet.')

    exact_U = model.exact_U
    n = model.n_qubits

    r_start = 1
    # r_start = int(n/3)
    if type == 'empirical':
        r_end = max(1 * n, r_max)
    elif type == 'bound':
        r_end = max(2 * n, r_max)
        ob_norm = norm(ob)
    # r_end = 20000
    r_list = []

    print(f'========== {type.upper()} {decompose.upper()} ==========')
    print(f'--------binary search parameters--------')
    print(f'binary search range: r_start={r_start}, r_end={r_end} ')
    print(f'Trotter error epsilon={epsilon}, binary search precision={100*search_precision}%')
    print(f'-------- binary search start --------')
    error_list = []
    exp_count_list = []
    error = 2
    while abs(error - epsilon) > epsilon * search_precision and r_end - r_start != 1:
        r = int((r_start + r_end) / 2)
        r_list.append(r)
        if type == 'empirical':
            if decompose == 'standard':
                appro_U = standard_trotter(h_list, t, r)
                # error = jnp.linalg.norm(jnp.array(appro_U - exact_U), ord=2)
                error = ob_error(ob, exact_U, appro_U)
                exp_count = 2 * model.n_terms * r
            elif decompose == 'lightcone':
                error, exp_count, exp_list = lightcone_trotter(model, ob, r, t, verbose=verbose)
        elif type == 'bound':
            if decompose == 'standard':
                error = model.standard_error / r**2
                # error = 2 * ob_norm * tight_bound(h_list, 2, t, r)
                exp_count = 2 * model.n_terms * r
            elif decompose == 'lightcone':
                # error, exp_count, exp_list = lightcone_trotter(model, ob, r, t, empirical=False, verbose=verbose)
                # error = lightcone_bound(model, exp_list, t, r)
                error = lightcone_bound_simplified(model, r, verbose=verbose)
                exp_count = exp_count_LC(r, n, model.n_terms)
            # print('observable error (L terms)', 2 * ob_norm * tight_bound([term.to_matrix() for term in ising1d.all_terms], 2, t, r_bound))
        elif type == 'lc_emp':
            print(type)
        exp_count_list.append(exp_count)

        # print('----spectral norm finished----')
        if verbose: print(f'r={r}; error={error:.6f}')
        error_list.append(error)
        if error > epsilon: r_start = r
        else: r_end = r

        if r_end - r_start == 1: 
            print('precision warning!!!')
            # raise ValueError('Binary search failed. Please increase the search range.')
    print(f'-------- binary search end --------')
    
    r_err_dict = dict(zip(r_list, error_list))
    r_abs_err_dict = dict(zip(r_list, [abs(error-epsilon) for error in error_list]))
    r_found = min(r_abs_err_dict, key=r_abs_err_dict.get)

    return r_list, error_list, exp_count_list, r_found 
    # return r_list, error_list


def plot_binary_search(ax, r_list, trotter_error_list, epsilon, search_precision, n, t, annotation=''):
    r_err_dict = dict(zip(r_list, trotter_error_list))
    r_abs_err_dict = dict(zip(r_list, [abs(error-epsilon) for error in trotter_error_list]))
    r_found = min(r_abs_err_dict, key=r_abs_err_dict.get)

    ax.plot(r_list, trotter_error_list, 'o', markersize=6, markeredgecolor='k', markeredgewidth=0.5, label='Binary search - '+annotation)
    ax.plot(r_found, r_err_dict[r_found], 'ro', markersize=7, markeredgecolor='k', markeredgewidth=1.5, label='Found - '+annotation)
    ax.axhline(y=epsilon, c='k', linestyle='--', linewidth=1.5)
    ax.axhline(y=epsilon * (1 + search_precision), c='k', linestyle='--', linewidth=0.5)
    ax.axhline(y=epsilon * (1 - search_precision), c='k', linestyle='--', linewidth=0.5)
    ax.axvline(x=r_found, c='k', linestyle='--', linewidth=0.5)
    # ax.xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel('Repetition r')
    # ax.xticks([r_list[-1]])
    # ax.gca().xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=False))
    ax.set_ylabel(r'Error $||e^{iHt}Oe^{-iHt}-UOU^\dagger||$')
    # ax.set_yticks([epsilon])
    ax.legend()
    # ax.grid()
    ax.set_title(f'1D TF Ising model, n={n}, t={t:.2f}, eps={epsilon}')

class transverse_field_ising_1d:
    def __init__(self, n: int, J, h, ob_index, t=1, pbc=False, initialize=True, verbose=False):
        """
        Constructs the Hamiltonian for the 1D transverse-field Ising model using Qiskit.

        Args:
            n (int): Number of spins in the chain.
            J (float): Coupling constant determining the interaction strength between neighboring spins.
            h (float): Strength of the transverse magnetic field.

        Returns:
            H (Operator): The Hamiltonian operator.

        """
        def rotate_str(string :str):
            return [string[-shift:] + string[:-shift] for shift in range(len(string)) ]

        # shift the string by one to the right
        ZZ = 'ZZ' + (n-2) * 'I'
        XI = 'X' + (n-1) * 'I'
        self.n_qubits = n
        self.zz_pstr = rotate_str(ZZ)
        self.x_pstr = rotate_str(XI)
        if not pbc:
            self.zz_pstr = self.zz_pstr[:-1]

        self.J = J; self.h = h
        self.zz_op = SparsePauliOp(self.zz_pstr, [self.J] * len(self.zz_pstr))
        self.x_op = SparsePauliOp(self.x_pstr, [self.h] * len(self.x_pstr))
        # self.x = SparsePauliOp(self.transverse, [self.h] * len(self.transverse)).to_matrix()
        self.H_op = self.zz_op + self.x_op

        self.all_terms = [SparsePauliOp(op, self.J) for op in self.zz_pstr] + [SparsePauliOp(op, self.h) for op in self.x_pstr]
        self.n_terms = len(self.all_terms)
        self.H_dict = dict(zip(self.zz_pstr+self.x_pstr, self.all_terms))

        self.t = t
        self.H_mat = self.H_op.to_matrix(True)
        self.partition('parity', verbose=verbose)

        if verbose: 
            print('---------Transverse-field Ising Hamiltonian---------')
            print(f'n={n}, J={J}, h={h}')
            print('Interaction: ', self.zz_pstr)
            print('Transverse: ', self.x_pstr)
            # print(SparsePauliOp(self.interaction, [self.J] * len(self.interaction)))
            # print('Hamiltonian matrix: \n', self.H_matrix)

        if initialize:
            ## evaluate exact evolution U
            if not np.array_equal(sum(self.all_terms).to_matrix(), self.H_mat.toarray()):
                raise ValueError('Hamiltonian is not constructed correctly.')
            print('---------------------initialization start@---------------------')
            self.exact_evolution(t)
            ## even/odd partition (grouping)
            ## lightcone decompositiona of the Hamiltonian
            self.ob, self.ob_dict = local_ob(ob_index, n)
            print('evaluation: standard error bound')
            self.standard_error_bound(verbose=verbose)
            print('evaluation: lightcone decompose')
            self.lightcone_decompose(self.ob_dict, verbose=verbose)
            print('evalution: lightcone error bound')
            self.lightcone_error_bound(verbose=verbose)
            print('---------------------initialization done---------------------')

    def exact_evolution(self, t):
        ## evaluate exact U = exp(-iHt)
        # exact_U = ssla.expm(-1j * t * sum(ising1d.h_list))
        self.exact_U = jax.scipy.linalg.expm(-1j * t * self.H_mat.toarray())
        # exact_U = scipy.linalg.expm(-1j * t * ising1d.H_matrix.toarray())
        print(f'----expm: exact U evaluated (t={t})----')
        
    def partition(self, method, verbose=False):
        """
        Partitions the Hamiltonian into two parts, the interaction and the external field part.

        Args:
            method 
        Returns:
            H (Operator): The Hamiltonian operator.

        """
        if method == 'parity':
            self.zz_even = self.zz_pstr[::-1][::2]
            self.zz_odd = self.zz_pstr[::-1][1::2]
            self.x_even = self.x_pstr[::-1][::2]
            self.x_odd = self.x_pstr[::-1][1::2]
            # self.x_even = self.x_pstr[::2]
            # self.x_odd = self.x_pstr[1::2]

            self.even_op = SparsePauliOp(self.zz_even, [self.J]*len(self.zz_even)) + SparsePauliOp(self.x_even, [self.h]*len(self.x_even)) 

            self.odd_op = SparsePauliOp(self.zz_odd, [self.J]*len(self.zz_odd))  + SparsePauliOp(self.x_odd, [self.h]*len(self.x_odd)) 

            # self.H_parity = [self.odd_op.to_matrix(True), self.even_op.to_matrix(True)]
            self.H_parity = [self.even_op.to_matrix(True), self.odd_op.to_matrix(True)]

            if verbose:
                print(f'---------({method}) Partitioned Hamiltonian---------')
                print('inter_xx_even:', self.zz_even)
                print('inter_xx_odd:', self.zz_odd)
                print('external_even:', self.x_even)
                print('external_odd:', self.x_odd)

    def lightcone_decompose(self, ob_dict, verbose=False):
        # self.ob_support = [0]
        self.ob_support = ob_dict['X']
        self.h_LC_decomp = []
        self.edge_set = []
        def pstr_support(pstr: str):
            support = []
            for i, c in enumerate(pstr):
                if c != 'I':
                    support.append(i)
            return support

        def pstr_list_support(pstr_list: list):
            support = []
            for pstr in pstr_list:
                support += pstr_support(pstr)
            return set(support)

        # int_supp = {p_str: pstr_support(p_str) for p_str in ising_1d.interaction[:self.n_qubits-1]}
        self.int_supp = {p_str: pstr_support(p_str) for p_str in self.zz_pstr}
        self.ext_supp = {p_str: pstr_support(p_str) for p_str in self.x_pstr}
        self.all_terms_supp = {**self.int_supp, **self.ext_supp}

        # print(ising_1d.interaction)
        if verbose:
            print('=============light cone decomposition============')
            # print('All terms support dict: ', self.all_terms_supp)
            print('Observable support: ', self.ob_support)
            print('Interaction term support dict: \n', self.int_supp)
            print('Transverse term support dict: \n', self.ext_supp)

        for i in range(self.n_qubits + 1):
            if verbose: print(f'----------step ({i})---------')
            temp = []
            if i == 0: 
                for item in self.all_terms_supp:
                    if set(self.all_terms_supp[item]).issubset(set(self.ob_support)):
                        # print(f'$H_S^{(0)}$ = {item}, {all_terms_supp[item]}')
                        temp.append(item)
                # print(temp_int, temp_ext)
                self.h_LC_decomp.append(temp)
                self.edge_set.append(set(self.ob_support))
            # elif i > 0 and i < n:
            else:
                for item in self.all_terms_supp:
                    if set(self.all_terms_supp[item]).intersection(set(self.edge_set[i-1])):
                        # print(item, all_terms_supp[item])
                        temp.append(item)
                if verbose: print(f'Intesect = {temp}')        
                self.h_LC_decomp.append(sorted(list(set(temp) - set(self.h_LC_decomp[i-1]))))
                self.edge_set.append(pstr_list_support(self.h_LC_decomp[i]) - self.edge_set[i-1])
            # else: 
            #     print('-----------stop-------------')
                # raise ValueError('Not implemented yet.')
            if verbose:
                print(f'$H_S^{(i)}$ = {self.h_LC_decomp[i]}; $E_S^{(i)}$ = {self.edge_set[i]}')

            if len(self.edge_set[i]) == 0:
                break
            # print('H light-cone decompose: ', h_LC_decomp)

        # II. SINGLE LOCAL OBSERVABLE partition
        self.h_LC_decomp[1] = self.h_LC_decomp[0] + self.h_LC_decomp[1]
    
    def lightcone_error_bound(self, verbose=False):
        self.r_saturate = int(self.n_qubits/2) + 1
        _, _, self.LC_gates = lightcone_trotter(self, self.ob, self.r_saturate, self.t, empirical=False, verbose=verbose)
        self.lightcone_segment_error_bounds = lightcone_bound(self, self.LC_gates, self.t, self.r_saturate, verbose=verbose)[1]

    def standard_error_bound(self, loose=True, verbose=False):
        if loose:
            self.standard_error = 2 * norm(self.ob) * analytic_loose_commutator_bound(self.n_qubits, self.J, self.h, self.t)
        else:
            self.standard_error = 2 * norm(self.ob) * tight_bound(self.H_parity, 2, self.t, 1)
        if verbose: print(f'Standard Trotter error bound (one step): {self.standard_error:.6f}')
        # return lightcone_segment_error_bounds

def purge_pauli(pauli_list):
    new_pauli_list = [] 
    for pauli in pauli_list:
        pauli = pauli.simplify()
        # print(pauli)
        if len(pauli.coeffs)>1 or abs(pauli.coeffs[0])>1e-8:
            new_pauli_list.append(pauli)
    return new_pauli_list

def commutator_bound(h_list, ord, t, r, type='tight'):
    # print(h_list)
    err = 0
    dt = t/r
    if ord == 1:
        for index, h1 in enumerate(h_list[:-1]):
            # print(index, h1)
            if type == 'tight':
                # print(sum([h2.to_matrix() for h2 in h_list[index+1:]]), h1.to_matrix())
                err += norm(commutator(sum([h2.to_matrix() for h2 in h_list[index+1:]]), h1.to_matrix()))
            elif type == 'loose':
                for h2 in h_list[index+1:]:
                    err += norm(commutator(h1, h2))
            else:
                raise ValueError(f'Unknown type: {type}')
        print(f'{type} bound: {err * dt**2 / 2}')
        return err * dt**2 / 2
    elif ord == 2:
        # raise NotImplementedError
        c1, c2 = 0, 0
        if type == 'tight':
            # for index, h1 in enumerate(h_list[:-1]):
            #     h2sum = sum(h_list[index+1:])
            #     c1 += norm(commutator(h2sum, commutator(h2sum, h1)))
            #     c2 += norm(commutator(h1, commutator(h1, h2sum)))
            c1 = sum([norm(sum([commutator(h3, sum([commutator(h2, h1) for h2 in h_list[index+1:]])) for h3 in h_list[index+1:]])) for index, h1 in enumerate(h_list[:-1])])    
            # c2 = sum([norm(commutator(h1, commutator(h1, sum(h_list[index+1:])))) for index, h1 in enumerate(h_list[:-1])])    
            c2 = sum([norm(commutator(h1, sum([commutator(h1, h2) for h2 in h_list[index+1:]]))) for index, h1 in enumerate(h_list[:-1])])    
            print(f'c1 (tight)={c1}, c2={c2}')
            err = c1 * dt**3 / 12 + c2 * dt**3 / 24
        elif type == 'loose':   
            c1 = sum([sum([norm(commutator(h3, sum([commutator(h2, h1) for h2 in h_list[index+1:]]))) for h3 in h_list[index+1:]]) for index, h1 in enumerate(h_list[:-1])])    
            c2 = sum([norm(commutator(h1, sum([commutator(h1, h2) for h2 in h_list[index+1:]]))) for index, h1 in enumerate(h_list[:-1])])    
            print(f'c1 (loose)={c1}, c2={c2}')
            err = c1 * dt**3 / 12 + c2 * dt**3 / 24
        print(f'{type} bound: {err}')
        return err

def commutator(A, B):
    return A @ B - B @ A

def norm(A, ord='spectral'):
    if ord == 'fro':
        return np.linalg.norm(A)
    elif ord == 'spectral':
        return np.linalg.norm(A, ord=2)
    elif ord == 'nuc':
        return np.linalg.norm(A, ord='nuc') # nuclear (trace) norm
    else:
        return np.linalg.norm(A, ord=ord)
        # raise ValueError('norm is not defined')

def operator_err(exact, approx, norm='spectral'):
    ''' 
    Frobenius norm of the difference between the exact and approximated operator
    input:
        exact: exact operator
        approx: approximated operator
    return: error of the operator
    '''
    if norm == 'fro':
        return np.linalg.norm(exact - approx)
    elif norm == 'spectral':
        return np.linalg.norm(exact - approx, ord=2)
    else:
        raise ValueError('norm is not defined')
    # return np.linalg.norm(exact - approx)/len(exact)

# def commutator_bound(H, t, eps):
#     return 

def triangle_bound(h, k, t, r):
    L = len(h)
    if k == 1:
        if L == 2:
            raise ValueError('k=1 is not defined for L=2')
        elif L == 3:
            c = norm(commutator(h[0], h[1])) + norm(commutator(h[1], h[2])) + norm(commutator(h[2], h[0]))
            error = c * t**2 / (2*r) 
    return error

def tight_bound(h, order: int, t: float, r: int):
    L = len(h)
    if order == 1:
        a_comm = 0
        for i in range(0, L):
            # temp = np.zeros(2**n_qubits, dtype=complex)
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += commutator(h[i], h[j])
            a_comm += norm(temp)
        error = a_comm * t**2 / (2*r)
    elif order == 2:
        c1 = 0
        c2 = 0
        for i in range(0, L):
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += h[j]
            # h_sum3 = sum(h[k] for k in range(i+1, L))
            # print(h_sum3.shape)
            # h_sum2 = sum(h[k] for k in range(i+1, L))
            c1 += norm(commutator(temp, commutator(temp, h[i]))) 
            # c1 = norm(commutator(h[0]+h[1], commutator(h[1]+h[2], h[0]))) + norm(commutator(h[2], commutator(h[2], h[1])))
            # c2 = norm(commutator(h[0], commutator(h[0],h[1]+h[2]))) + norm(commutator(h[1], commutator(h[1], h[2])))
            c2 += norm(commutator(h[i], commutator(h[i], temp)))
        print(f'c1 (tight bound by matrix)={c1}, c2={c2}')
        error = c1 * t**3 / r**2 / 12 + c2 *  t**3 / r**2 / 24 
    else: 
        raise ValueError(f'higer order (order={order}) is not defined')

    return error

def interference_bound(H, t, r):
    # Layden_2022_First-Order Trotter Error from a Second-Order Perspective
    try:
        assert len(H) == 2
    except:
        raise ValueError('The Hamiltonian contains not exactly 2 terms')

    h1 = H[0]
    h2 = H[1]
    C1 = min(norm(h1), norm(h2))
    C2 = 0.5 * norm(commutator(h1, h2))
    S = [norm(commutator(h1, commutator(h1, h2))), norm(commutator(h2, commutator(h2, h1)))]
    C3 = 1 / 12 * (min(S) + 0.5 * max(S))
    e1 = C1 * t / r
    e2 = C2 * t**2 / r
    e3 = C3 * t**3 / r**2
    bound = min(e2, e1 + e3, 2 * len(h1))

    return bound, e1, e2, e3


