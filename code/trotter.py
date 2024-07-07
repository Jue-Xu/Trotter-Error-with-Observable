import random, sys
import multiprocessing

from cmath import cos, exp, pi, sin, sqrt
from jax.scipy.linalg import expm
# from scipy.linalg import expm
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as ssla
import scipy

import numpy as np
from numpy import log
from numpy.linalg import matrix_power
np.set_printoptions(precision=6)
FLOATING_POINT_PRECISION = 1e-10

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

import matplotlib.pyplot as plt

from bounds import *

def jax_matrix_exponential(matrix):
    # return jsl.expm( matrix)
    return ssla.expm(matrix)
jax_matrix_exponential = jax.jit(jax.vmap(jax_matrix_exponential))

def sparse_multi_dot(sparse_matrices):
    '''
    计算一个列表中所有矩阵的乘积
    '''
    product = sparse_matrices[0]
    for matrix in sparse_matrices[1:]:
        product = product.dot(matrix)
    return product
    # return product.toarray()

vectorized_sparse_expm = jax.vmap(ssla.expm)

def mpi_sparse_expm(list_herms, t, r):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    list_unitaries = pool.map(ssla.expm, -1j * t / r * np.array(list_herms))
    # Close the pool of workers
    pool.close()
    pool.join()

    return list_unitaries

def sparse_trotter_error(list_herm: list, r: int, t: int) -> float:
    print('-------sparse_trotter_error--------')
    exact_U = ssla.expm(-1j * t * sum(list_herm))
    # list_U = jax_matrix_exponential(jnp.array(-1j * t / (2*r) * np.array(list_herm)))
    # list_U = vectorized_sparse_expm(-1j * t / (2*r) * np.array(list_herm))
    # list_herm_scaled = np.array([-1j * t / (2*r) * herm for herm in list_herm])
    # list_U = ssla.expm(list_herm_scaled) 
    # list_U = [ssla.expm(-1j * t / (2*r) * herm) for herm in list_herm]
    list_U = mpi_sparse_expm(list_herm, t, 2*r)
    # list_U = jax_matrix_exponential(jnp.array([-1j * t / (2*r) * herm.toarray() for herm in np.array(list_herm)]))
    list_U2 = [U**2 for U in list_U]
    # trotter_error_list = op_error(exact_U, matrix_power(sparse_multi_dot(list_U2), r))
    trotter_error_list = op_error(exact_U, sparse_multi_dot(list_U2)**r)
    # trotter_error_list = op_error(exact_U, np.linalg.matrix_power(np.linalg.multi_dot(np.array(list_U2)), r))
    # second-order trotter
    trotter_error_list_2nd = op_error(exact_U, (sparse_multi_dot(list_U) @ sparse_multi_dot(list_U[::-1]))**r)
    # trotter_error_list_2nd = op_error(exact_U, np.linalg.matrix_power(np.linalg.multi_dot(np.array(list_U)) @ np.linalg.multi_dot(np.array(list_U[::-1])), r))
    
    return [trotter_error_list, trotter_error_list_2nd]

# matrix product of a list of matrices
def unitary_matrix_product(list_herm_matrices, t=1):
    ''' 
    matrix product of a list of unitary matrices exp(itH)
    input: 
        list_herm_matrices: a list of Hermitian matrices
        t: time
    return: the product of the corresponding matrices
    '''
    product = expm(-1j * t * list_herm_matrices[0])
    for i in range(1, len(list_herm_matrices)):
        product = product @ expm(-1j * t * list_herm_matrices[i])

    return product

def matrix_product(list_U, t=1):
    # product = matrix_power(list_U[0], t)
    # for i in range(1, len(list_U)):
    #     product = matrix_power(list_U[i], t) @ product
    #     # product = product @ matrix_power(list_U[i], t)
    product = np.linalg.multi_dot([matrix_power(U, t) for U in list_U])
    return product

def pf_r(h_list, t, r, order=2, verbose=False, use_jax=True):
    if order == 2:
        if use_jax:
            list_U = [jax.scipy.linalg.expm(-1j * (t / (2*r)) * herm) for herm in h_list]
        else:
            if isinstance(h_list[0], csr_matrix):
                list_U = [scipy.linalg.expm(-1j * (t / (2*r)) * herm.toarray()) for herm in h_list]
            elif isinstance(h_list[0], np.ndarray):
                list_U = [scipy.linalg.expm(-1j * (t / (2*r)) * herm) for herm in h_list]
            else:
                raise ValueError('h_list is not defined')
        if verbose: print('----expm Herm finished----')
        appro_U_dt_forward = np.linalg.multi_dot(list_U)
        appro_U_dt_reverse = np.linalg.multi_dot(list_U[::-1])
        # appro_U_dt = list_U[0] @ list_U[1]
        if verbose: print('----matrix product finished----')
        appro_U = np.linalg.matrix_power(appro_U_dt_reverse @ appro_U_dt_forward, r)
        appro_U = np.linalg.matrix_power(appro_U_dt_forward @ appro_U_dt_reverse, r)
        if verbose: print('----matrix power finished----')
    elif order == 1:
        list_U = [jax.scipy.linalg.expm(-1j * (t / (r)) * herm.toarray()) for herm in h_list]
        appro_U_dt = np.linalg.multi_dot(list_U)
        appro_U = np.linalg.matrix_power(appro_U_dt, r)

    return appro_U

def measure_error(r, h_list, t, exact_U, type, rand_states=[], ob=None, pf_ord=2, coeffs=[], use_jax=False, verbose=False, return_error_list=False): 
    # print(type)
    if type == 'worst_empirical':
        return 2 * np.linalg.norm(exact_U - pf_r(h_list, t, r, order=pf_ord, use_jax=use_jax), ord=2)
    elif type == 'worst_bound':
        if coeffs != []:
            return 2 * tight_bound(h_list, 2, t, r) * coeffs[0]
        else:
            return 2 * tight_bound(h_list, 2, t, r)
    elif type == 'worst_ob_empirical':
        appro_U = pf_r(h_list, t, r, order=pf_ord, use_jax=use_jax)
        # appro_U = pf_r(h_list, t, r, order=pf_ord)
        exact_ob = exact_U.conj().T @ ob @ exact_U 
        appro_ob = appro_U.conj().T @ ob @ appro_U
        # ob_error = np.linalg.norm(exact_ob - appro_ob, ord=2)
        ob_error = np.sort(abs(np.linalg.eigvalsh(exact_ob - appro_ob)))[-1]
        print('ob error (operator norm, largest eigen): ', ob_error, '; r:', r, '; t:', t)
        return ob_error
    elif type == 'worst_loose_bound':
        return relaxed_st_bound(r, coeffs[1], coeffs[2], t, ob_type=coeffs[0])
    elif type == 'lightcone_bound':
        return lc_tail_bound(r, coeffs[1], coeffs[2], t, ob_type=coeffs[0], verbose=False)
        # return relaxed_lc_bound(r, coeffs[1], coeffs[2], t, ob_type=coeffs[0], verbose=False)
    elif type == 'average_bound':
        # return tight_bound(h_list, 2, t, r, type='4')
        if coeffs != []:
            return 2 * tight_bound(h_list, 2, t, r, type='fro') * coeffs[0]
        else:
            return 2 * tight_bound(h_list, 2, t, r, type='fro')
    elif type == 'average_empirical':
        appro_U = pf_r(h_list, t, r, order=pf_ord, use_jax=use_jax)
        err_list = [np.linalg.norm(np.outer(exact_U @ state.data.conj().T , (exact_U @ state.data.conj().T).conj().T) - np.outer(appro_U @ state.data.conj().T, (appro_U @ state.data.conj().T).conj().T), ord='nuc') for state in rand_states]
        # err_list = [np.linalg.norm((exact_U - pf_r(h_list, t, r, order=pf_ord, use_jax=use_jax)) @ state.data) for state in rand_states]
        if return_error_list:
            return np.array(err_list) * np.linalg.norm(ob, ord=2)
        else:
            return np.mean(err_list) * np.linalg.norm(ob, ord=2)
    elif type == 'average_ob_bound_legacy':
    # elif type == 'average_ob_bound':
        if isinstance(h_list[0], csr_matrix):
            onestep_exactU = scipy.linalg.expm(-1j * t/r * sum([herm.toarray() for herm in h_list]))
            d = len(h_list[0].toarray())
        elif isinstance(h_list[0], np.ndarray):
            onestep_exactU = scipy.linalg.expm(-1j * t/r * sum([herm for herm in h_list]))
            d = len(h_list[0])
        E_op = onestep_exactU - pf_r(h_list, t/r, 1, order=pf_ord, use_jax=use_jax)
        # print((np.trace(E_op @ E_op.conj().T @ E_op @ E_op.conj().T)/d)**(1/4))
        bound = 2 * r * (np.trace(E_op @ E_op.conj().T @ E_op @ E_op.conj().T)/d)**(1/4) * (np.trace(ob @ ob @ ob @ ob)/d)**(1/4)
        # print(f'bound_e={bound_e}, bound={bound}')
        return bound
    elif type == 'average_ob_bound':
    # elif type == 'average_ob_bound_nc':
        if isinstance(h_list[0], csr_matrix):
            d = len(h_list[0].toarray())
        elif isinstance(h_list[0], np.ndarray):
            d = len(h_list[0])
        # if coeffs == []:
        #     bound = 2 * tight_bound(h_list, 2, t, r, type='4') * (np.trace(ob @ ob @ ob @ ob)/d)**(1/4)
        # else:
        #     bound = np.sqrt(2) * tight_bound(h_list, 2, t, r, type='fro') * (sum([np.linalg.norm(ob, ord='fro') for ob in coeffs[0]])/(d+1)**(1/2)) 
        bound = np.sqrt(2) * tight_bound(h_list, 2, t, r, type='fro') * (sum([np.linalg.norm(o, ord='fro') for o in ob])/(d+1)**(1/2)) 
        return bound
    # elif type == 'observable_empirical':
    elif type == 'average_ob_empirical':
        approx_U = pf_r(h_list, t, r, order=pf_ord, use_jax=use_jax)
        exact_final_states = [exact_U @ state.data.T for state in rand_states]
        appro_final_states = [approx_U @ state.data.T for state in rand_states]
        err_list = [abs(appro_final_states[i].conj().T @ ob @ appro_final_states[i] - exact_final_states[i].conj().T @ ob @ exact_final_states[i]) for i in range(len(rand_states))]
        if return_error_list:
            return np.array(err_list)
        else:
            return np.mean(err_list)
    # elif type == 'observable_bound':
    #     return None
    else: 
        raise ValueError(f'type={type} is not defined!')

def binary_search_r(r_start, r_end, epsilon, error_measure, step=1, comment='', verbose=False):
    # print(f'----binary search r ({error_measure.__name__})----')
    print(f'----[{comment}] binary search r (r_start={r_start}, r_end={r_end})----')
    while error_measure(r_end) > epsilon:
        print("the initial r_end is too small, increase it by 10 times.")
        r_end *= 10

    if error_measure(r_start) <= epsilon:
        r = r_start
    else: 
        while r_start < r_end - step: 
            r_mid = int((r_start + r_end) / 2)
            if error_measure(r_mid) > epsilon:
                r_start = r_mid
            else:
                r_end = r_mid
            if verbose: print('r_start:', r_start, '; r_end:', r_end)
        r = r_end
    if verbose: print('r:', r, '; err: ', error_measure(r))
    return r

def commutator(A, B):
    return A @ B - B @ A

# def anticommutator(A, B, to_sparse=False):
def anticommutator(A, B):
    return A @ B + B @ A

def second_order_trotter(list_herm_matrices, t=1):
    forward_order_product = unitary_matrix_product(list_herm_matrices, t/2) 
    reverse_order_product = unitary_matrix_product(list_herm_matrices[::-1], t/2)

    return forward_order_product @ reverse_order_product

def pf_U(list_U, order, t=1):
    # print('order: ', order)
    if order == 1:
        return matrix_product(list_U, t)
    elif order == 2:
        forward_order_product = matrix_product(list_U, t/2) 
        reverse_order_product = matrix_product(list_U[::-1], t/2)
        return forward_order_product @ reverse_order_product
    elif order > 0 and order != 1 and order != 2 and order % 2 == 0:
        p = 1 / (4 - 4**(1/(order-1)))
        # print('p: ', p)
        return matrix_power(pf_U(list_U, order-2, p*t), 2) @ pf_U(list_U, order-2, (1-4*p)*t) @ matrix_power(pf_U(list_U, order-2, p*t), 2)
    else:
        raise ValueError('k is not defined')

def pf(list_herm, order, t):
    # print('order: ', order)
    if order == 1:
        return unitary_matrix_product(list_herm, t)
    elif order == 2:
        forward_order_product = unitary_matrix_product(list_herm, t/2) 
        reverse_order_product = unitary_matrix_product(list_herm[::-1], t/2)
        return forward_order_product @ reverse_order_product
        # return second_order_trotter(list_herm, t)
    elif order > 0 and order!= 1 and order != 2 and order % 2 == 0:
        p = 1 / (4 - 4**(1/(order-1)))
        # print('p: ', p)
        return matrix_power(pf(list_herm, order-2, p*t), 2) @ pf(list_herm, order-2, (1-4*p)*t) @ matrix_power(pf(list_herm, order-2, p*t), 2)
    else:
        raise ValueError('k is not defined')

def op_error(exact, approx, norm='spectral'):
    ''' 
    Frobenius norm of the difference between the exact and approximated operator
    input:
        exact: exact operator
        approx: approximated operator
    return: error of the operator
    '''
    if norm == 'fro':
        return jnp.linalg.norm(exact - approx)
    elif norm == 'spectral':
        # if the input is in csr_matrix format
        if isinstance(exact, csc_matrix) and isinstance(approx, csc_matrix):
            return jnp.linalg.norm(jnp.array(exact.toarray() - approx.toarray()), ord=2)
        else:
            return jnp.linalg.norm(exact - approx, ord=2)
    else:
        raise ValueError('norm is not defined')
    # return np.linalg.norm(exact - approx)/len(exact)

# evaluate trotter error for different number of trotter steps
def trotter_error(list_herm, r_list, t, norm='spectral', n_perm=50, verbose=False):
    ''' 
    evaluate trotter error for different number of trotter steps
    input: 
        list_herm: a list of Hermitian matrices
        r_list: number of trotter steps
    return: trotter error
    '''
    exact_U = expm(-1j * t * sum(list_herm))
    list_U = [expm(-1j * t / (2*r_list[-1]) * herm) for herm in list_herm]
    if len(list_U) >= 5:
        print('number of terms: ', len(list_U))
        perm_list = [list_U] 
        seed_value = random.randrange(sys.maxsize)
        random.seed(seed_value)  
        # randomly select 5 permutations from perm_list
        for _ in range(n_perm-1):
            # random.shuffle(list_U) 
            # perm_list.append(list_U[:])
            perm_list.append(random.sample(list_U, len(list_U)))
        # perm_list = random.sample(perm_list, 50) 
        print('# randomly selected perm: ', len(perm_list))
    else:
        # generate a list of permutation of the order of the matrices
        perm_list = list(itertools.permutations(list_U))
        # print('perm_list', perm_list)
        print('# all perm: ', len(perm_list))
    # perm_list = list(itertools.permutations(list_herm))[:5]
    # for r in r_list:
    # first-order trotter
    trotter_error_list = [op_error(matrix_power(matrix_product(perm, int(2*r_list[-1]/r)), r), exact_U, norm) for r in r_list for perm in perm_list]
    # trotter_error_list = [op_error(matrix_power(unitary_matrix_product(perm, t=t/r), r), exact_U, norm) for r in r_list for perm in perm_list]
    # second-order trotter
    trotter_error_list_2nd = [op_error(matrix_power(matrix_product(perm, int(r_list[-1]/r)) @ matrix_product(perm[::-1], int(r_list[-1]/r)), r), exact_U, norm) for r in r_list for perm in perm_list]
    err_1st_reshaped = np.array(trotter_error_list).reshape(len(r_list), len(perm_list))
    err_2nd_reshaped = np.array(trotter_error_list_2nd).reshape(len(r_list), len(perm_list))

    return err_1st_reshaped , err_2nd_reshaped

def search_r_for_error(r_start, r_end, epsilon, t, list_herm, k, norm='spectral', verbose=False):
    tol = r_end - r_start
    exact_U = expm(-1j * t * sum(list_herm))
    # binary search from r_start to r_end
    while tol > 2:
        r = int((r_start + r_end) / 2)
        err = op_error(matrix_power(pf(list_herm, k, t=t/r), r), exact_U, norm)
        # if k == 1:
        #     err = op_error(matrix_power(unitary_matrix_product(list_herm, t=t/r), r), exact_U, norm)
        # elif k == 2:
        #     err = op_error(matrix_power(second_order_trotter(list_herm, t=t/r), r), exact_U, norm)
        # elif k != 2 and k > 1 and k % 2 == 0:
        #     err = op_error(matrix_power(high_order_trotter(list_herm, k, t=t/r), r), exact_U, norm)
        # else:
        #     raise ValueError('k is not defined')

        if err > epsilon:
            r_start = r
        else:
            r_end = r
        tol = abs(r_end - r_start)
    if verbose: print('err: ', err)
    return r



def plot_trotter_error_vs_r(epsilon, t, ham_group, r_list, perm_label, markers, plot=True, locate=True):
    trotter_error_list, trotter_error_list_2nd = trotter_error(ham_group, r_list, t)
    # print('trotter_error_list: \n', trotter_error_list)
    # for index, trotter_error in enumerate(trotter_error_list):
    #     plt.plot(r_list, trotter_error, '^-', label='ordering: '+perm_label[index])
    if plot:
        for i in range(len(trotter_error_list[0])):
            plt.plot(r_list, trotter_error_list[:,i], markers[i], markeredgecolor='black', label= perm_label[i] + ' (1st)')

        for i in range(len(trotter_error_list_2nd[0])):
            plt.plot(r_list, trotter_error_list_2nd[:,i], markers[i], markeredgecolor='black', label=perm_label[i] + ' (2nd)')

    if locate:
        epsilon_list = [epsilon] * len(trotter_error_list[:, 0])
        idx_1st_0 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list[:,0])))
        idx_1st_1 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list[:,1])))
        idx_2nd_0 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list_2nd[:,0])))
        idx_2nd_1 = np.argwhere(np.diff(np.sign(epsilon_list - trotter_error_list_2nd[:,1])))
        intersect_indices = [ r_list[index] for index in np.array([idx_1st_0, idx_1st_1, idx_2nd_0, idx_2nd_1]).flatten() ]
        print('intersect_indices: ',intersect_indices)

        return intersect_indices