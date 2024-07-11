from cmath import exp
from scipy import sparse
from scipy.linalg import expm
from numpy.linalg import matrix_power
import scipy.sparse.linalg as ssla

import numpy as np

def commutator(A, B):
    return A @ B - B @ A

def norm(A, ord='spectral'):
    if ord == 'fro':
        return np.linalg.norm(A)
    elif ord == 'spectral':
        return np.linalg.norm(A, ord=2)
    elif ord == '4':
        return np.trace(A @ A.conj().T @ A @ A.conj().T)**(1/4)
    else:
        # raise ValueError('norm is not defined')
        return np.linalg.norm(A, ord=ord)

def analytic_bound(H, k, t, r):
    L = len(H)
    Lambda = max([norm(h) for h in H])

    return (2 * L * 5**(k-1) * Lambda * t)**(2*k+1)/(3*r**(2*k)) * exp((2*L*5**(k-1)*Lambda*t)/r)

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
    bound = min(e2, e1 + e3, 2)
    # bound = min(e2, e1 + e3, 2 * len(h1))

    return bound, e1, e2, e3

def triangle_bound(h, k, t, r):
    L = len(h)
    if k == 1:
        if L == 2:
            raise ValueError('k=1 is not defined for L=2')
        elif L == 3:
            c = norm(commutator(h[0], h[1])) + norm(commutator(h[1], h[2])) + norm(commutator(h[2], h[0]))
            error = c * t**2 / (2*r) 
    return error

def tight_bound(h, order: int, t: float, r: int, type='spectral', verbose=False):
    L = len(h)
    d = h[0].shape[0]
    if order == 1:
        a_comm = 0
        for i in range(0, L):
            temp = np.zeros(h[0].shape, dtype=complex)
            for j in range(i + 1, L):
                temp += commutator(h[i], h[j])
            a_comm += norm(temp, ord=type)

        if type == 'spectral':
            error = a_comm * t**2 / (2*r)
        elif type == 'fro':
            error = a_comm * t**2 / (2*r*np.sqrt(d))
        else:
            raise ValueError(f'type={type} is not defined')
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
            c1 += norm(commutator(temp, commutator(temp, h[i])), ord=type) 
            # c1 = norm(commutator(h[0]+h[1], commutator(h[1]+h[2], h[0]))) + norm(commutator(h[2], commutator(h[2], h[1])))
            # c2 = norm(commutator(h[0], commutator(h[0],h[1]+h[2]))) + norm(commutator(h[1], commutator(h[1], h[2])))
            c2 += norm(commutator(h[i], commutator(h[i], temp)), ord=type)
        if type == 'spectral':
            error = c1 * t**3 / r**2 / 12 + c2 *  t**3 / r**2 / 24 
        elif type == 'fro':
            # print(c1, c2)
            error = c1 * t**3 / r**2 / 12 / np.sqrt(d) + c2 *  t**3 / r**2 / 24 / np.sqrt(d)
            # print('random input:', error)
        elif type == '4':
            error = c1 * t**3 / r**2 / 12 / d**(1/4) + c2 *  t**3 / r**2 / 24 / d**(1/4)
        else:
            raise ValueError(f'type={type} is not defined')
    else: 
        raise ValueError(f'higer order (order={order}) is not defined')

    return error

def analytic_loose_commutator_bound_parity(n, J, h, dt, pbc=False, verbose=False):
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
    return analytic_error_bound, c1, c2

def analytic_loose_commutator_bound_xyz(n, J, h, dt, pbc=False, verbose=False):
    if pbc:
        c1 = 16*J**2*h*(n) + 4*J**2*h*(n)
        c2 = 8*(n)*J**2*h
    else:
        c1 = 8*J*h**2*2*(n-1) 
        c2 =  8*J**2*h*(2*(n-1)-2) + 4*J**2*h*(2)

    if verbose: print(f'c1 (analy)={c1}, c2={c2}')
    analytic_error_bound = c1 * dt**3 / 12 + c2 * dt**3 / 24
    return analytic_error_bound, c1, c2

def analy_st_loose_bound(r, n, J, h, t, ob_type='single', group='parity'):
    if group == 'parity':
        return 2 * analytic_loose_commutator_bound_parity(n, J, h, t/r)[0] * r
    elif group == 'xyz':
        return 2 * analytic_loose_commutator_bound_xyz(n, J, h, t/r)[0] * r
    else:
        raise ValueError(f'group={group} not recognized')

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
    # for i in range(1, r+1):
    for i in range(1, r+2):
        if ob_type == 'single':
            n_lc = min(i*2, n)
            err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
        elif ob_type == 'multi':   
            for j in range(0, n):
                n_lc = min(min(n-j, i*2) + min(j, 2*i), n)
                # err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
                err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) / n
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

from spin_ham import *

# def relaxed_commutator_bound(n, J, h, dt, verbose=False):
#     hnn = Nearest_Neighbour_1d(n, Jx=J, Jy=J, Jz=J, hx=h, hy=0, hz=0, pbc=False, verbose=False)
#     h_list = hnn.ham_par
def relaxed_commutator_bound(n, cmm_data, dt, verbose=False):
    if verbose: print(f'c1 (relax)={cmm_data["c1"][n-1]}, c2={cmm_data["c2"][n-1]}')
    relaxed_error_bound = cmm_data['c1'][n-1] * dt**3 / 12 + cmm_data['c2'][n-1] * dt**3 / 24
    return relaxed_error_bound

# def relaxed_st_bound(r, n, cmm_data, t, ob_type='singl'):
def relaxed_st_bound(r, n, h, t, h_group=[], ob_type='singl'):
    if h_group == []:
        h_list = h.ham_par
        # h_list = h.ham_xyz[:2]
    else:
        h_list = h_group
    # h_list = lc_group(h, 0, 0, 2*n+2)[:2]
    dt = t/r
    if ob_type == 'singl':
        c1_cmm = commutator(h_list[1], commutator(h_list[1], h_list[0]).simplify()).simplify()
        c2_cmm = commutator(h_list[0], commutator(h_list[0], h_list[1]).simplify()).simplify()
        c1 = np.linalg.norm(c1_cmm.coeffs, ord=1)
        c2 = np.linalg.norm(c2_cmm.coeffs, ord=1)
        if c1 >= c2:
            return 2 * (c1 * dt**3 / 12 + c2 * dt**3 / 24) * r
        else:
            return 2 * (c2 * dt**3 / 12 + c1 * dt**3 / 24) * r
    #     return 2 * relaxed_commutator_bound(n, cmm_data, t/r) * r
    # elif ob_type == 'multi':
    #     # return 2 * analytic_loose_commutator_bound(n, J, h, t/r) * r  * n
    #     return 2 * relaxed_commutator_bound(n, cmm_data, t/r) * r 
    else:
        raise ValueError('ob_type should be either single or multi')

def relaxed_lc_bound(r, n, cmm_data, t, ob_type='singl', verbose=False):
    err_bound = 0
    for i in range(1, r+2):
    # for i in range(1, r+1):
        if ob_type == 'singl':
            n_lc = min(i*2, n)
            err_bound += 2 * relaxed_commutator_bound(n_lc, cmm_data, t/r, verbose=verbose) 
        elif ob_type == 'multi':   
            for j in range(0, n):
                n_lc = min(min(n-j, i*2) + min(j, 2*i), n)
                # err_bound += 2 * analytic_loose_commutator_bound(n_lc, J, h, t/r, verbose) 
                err_bound += 2 * relaxed_commutator_bound(n_lc, cmm_data, t/r, verbose=verbose) / n
        else:
            raise ValueError('ob_type should be either single or multi')

    return err_bound

# # def lc_tail_bound(r, n, tail_cmm_data, t, ob_type='singl', verbose=False):
# def lc_tail_bound(r, n, tail_cmm_data, t, ob_type='singl', verbose=True):
#     err_bound = 0
#     dt = t/r
#     if verbose: print(f'index={n}, r={r}')
#     if verbose: print(tail_cmm_data['n'][n-1])
#     if ob_type == 'singl':
#         err_bound = 2 * (tail_cmm_data['c1_z'][n-1][r-1][0] * dt**3 / 12 + tail_cmm_data['c2_z'][n-1][r-1][0] * dt**3 / 24)
#     elif ob_type == 'multi_z':
#         err_bound = 2 * (sum(tail_cmm_data['c1_z'][n-1][r-1]) * dt**3 / 12 + sum(tail_cmm_data['c2_z'][n-1][r-1]) * dt**3 / 24 ) / tail_cmm_data['n'][n-1]
#     elif ob_type == 'multi_zz':
#         err_bound = 2 * (sum(tail_cmm_data['c1_zz'][n-1][r-1]) * dt**3 / 12 + sum(tail_cmm_data['c2_zz'][n-1][r-1]) * dt**3 / 24) / (tail_cmm_data['n'][n-1]-1)
#     else:
#         raise ValueError('ob_type should be either single or multi')

#     return err_bound

def lc_group(h, right, left, step, verbose=False, legacy=False):
    if verbose: print(f'n={h.n}, right={right}, left={left}, step={step}')
    tail_tuples = []
    right_range = list(range(0, right-step))
    left_range = list(range(left+step, h.n-1))
    if verbose: print(right_range, left_range)
    all_tuples = [h.x_tuples, h.y_tuples, h.z_tuples, h.xx_tuples, h.yy_tuples, h.zz_tuples]
    if verbose: print(all_tuples)
    for i in right_range:
        for tuple in all_tuples:
            tail_tuples.append(tuple[i])
            # if verbose: print(tuple[i])

    for i in left_range:
        # for tuple in all_tuples:
        tail_tuples.append(h.xx_tuples[i])
        tail_tuples.append(h.yy_tuples[i])
        tail_tuples.append(h.zz_tuples[i])
        tail_tuples.append(h.x_tuples[i+1])
        tail_tuples.append(h.y_tuples[i+1])
        tail_tuples.append(h.z_tuples[i+1])
            # tail_tuples.append(tuple[i])
            # if verbose: print(tuple[i])

    # if left_range != []:
    #     tail_tuples.append(h.x_tuples[-1])
    #     tail_tuples.append(h.y_tuples[-1])
    #     tail_tuples.append(h.z_tuples[-1])
    # print([*h.x_tuples, *h.y_tuples, *h.z_tuples, *h.xx_tuples, *h.yy_tuples, *h.zz_tuples]-tail_terms)
    odd_lc_tuples, even_lc_tuples = [], []
    for i in list(range(max(right-step, 0), right))[::-1][::2]:
        odd_lc_tuples.append(h.xx_tuples[i])
        odd_lc_tuples.append(h.yy_tuples[i])
        odd_lc_tuples.append(h.zz_tuples[i])
        odd_lc_tuples.append(h.x_tuples[i])
        odd_lc_tuples.append(h.y_tuples[i])
        odd_lc_tuples.append(h.z_tuples[i])

    if verbose: print(list(range(left, min(left+step, h.n-1)))[::2])
    for i in list(range(left, min(left+step, h.n-1)))[::2]:
        odd_lc_tuples.append(h.xx_tuples[i])
        odd_lc_tuples.append(h.yy_tuples[i])
        odd_lc_tuples.append(h.zz_tuples[i])
        odd_lc_tuples.append(h.x_tuples[i+1])
        odd_lc_tuples.append(h.y_tuples[i+1])
        odd_lc_tuples.append(h.z_tuples[i+1])

    for i in range(right, left):
        even_lc_tuples.append(h.xx_tuples[i])
        even_lc_tuples.append(h.yy_tuples[i])
        even_lc_tuples.append(h.zz_tuples[i])
        even_lc_tuples.append(h.x_tuples[i])
        even_lc_tuples.append(h.y_tuples[i])
        even_lc_tuples.append(h.z_tuples[i])
    even_lc_tuples.append(h.x_tuples[left])
    even_lc_tuples.append(h.y_tuples[left])
    even_lc_tuples.append(h.z_tuples[left])

    for i in list(range(max(right-step, 0), right-1))[::-1][::2]:
        even_lc_tuples.append(h.xx_tuples[i])
        even_lc_tuples.append(h.yy_tuples[i])
        even_lc_tuples.append(h.zz_tuples[i])
        even_lc_tuples.append(h.x_tuples[i])
        even_lc_tuples.append(h.y_tuples[i])
        even_lc_tuples.append(h.z_tuples[i])

    for i in list(range(left+1, min(left+step, h.n-1)))[::2]:
        even_lc_tuples.append(h.xx_tuples[i])
        even_lc_tuples.append(h.yy_tuples[i])
        even_lc_tuples.append(h.zz_tuples[i])
        even_lc_tuples.append(h.x_tuples[i+1])
        even_lc_tuples.append(h.y_tuples[i+1])
        even_lc_tuples.append(h.z_tuples[i+1])
    # for i in
    all_pauli = SparsePauliOp.from_sparse_list([*h.xx_tuples, *h.yy_tuples, *h.zz_tuples, *h.x_tuples, *h.y_tuples, *h.z_tuples], h.n).simplify() 
    lc_pauli = SparsePauliOp.from_sparse_list([*odd_lc_tuples, *even_lc_tuples, *tail_tuples], h.n).simplify()
    difference = (all_pauli - lc_pauli).simplify()

    if len(difference.coeffs) == 1 and difference.coeffs[0] == 0:
        if verbose: print('Success!')
    else:
        print('Lightcone decomposition error!: ', difference)
        # print(set([*odd_lc_tuples, *even_lc_tuples, *tail_tuples])-set([*h.x_tuples, *h.y_tuples, *h.z_tuples, *h.xx_tuples, *h.yy_tuples, *h.zz_tuples]))
        # print('', [*odd_lc_tuples, *even_lc_tuples, *tail_tuples])
        # print('', [*h.x_tuples, *h.y_tuples, *h.z_tuples, *h.xx_tuples, *h.yy_tuples, *h.zz_tuples])
        # raise Exception( e.args )

    even_lc_terms = SparsePauliOp.from_sparse_list(even_lc_tuples, h.n).simplify()
    odd_lc_terms  = SparsePauliOp.from_sparse_list(odd_lc_tuples, h.n).simplify()
    tail_lc_terms = SparsePauliOp.from_sparse_list(tail_tuples, h.n).simplify()
    # print(h.even_terms)
    # print(h.x_tuples)
    # return [even_lc_tuples, odd_lc_tuples, tail_tuples]
    
    new_even_lc_tuples, new_odd_lc_tuples = [], []
    for item in even_lc_tuples + odd_lc_tuples:
        if verbose: print(item)
        if item[1][0] % 2 == 0 and item[2] != 0:
            if verbose: print('even: ', item)
            new_even_lc_tuples.append(item)
        elif item[1][0] % 2 == 1 and item[2] != 0:
            if verbose: print('odd: ', item)
            new_odd_lc_tuples.append(item)

    temp = []
    if len(new_even_lc_tuples) % 2 == 1:
        for item in new_even_lc_tuples:
            if len(item[0]) == 1:
                temp.append(item)
        if verbose: print('single even: ', sorted(temp, key=lambda x: x[1][0]))
        boundary_element = sorted(temp, key=lambda x: x[1][0])[-1]
        new_even_lc_tuples.remove(boundary_element)
        new_odd_lc_tuples.append(boundary_element)
    else:
        for item in new_odd_lc_tuples:
            if len(item[0]) == 1:
                temp.append(item)
        if verbose: print('single odd: ', sorted(temp, key=lambda x: x[1][0]))
        boundary_element = sorted(temp, key=lambda x: x[1][0])[-1]
        new_odd_lc_tuples.remove(boundary_element)
        new_even_lc_tuples.append(boundary_element)

    if verbose:
        print("even: ", SparsePauliOp.from_sparse_list(new_even_lc_tuples, h.n))
        print("odd:  ", SparsePauliOp.from_sparse_list(new_odd_lc_tuples, h.n))
        print("tail: ", SparsePauliOp.from_sparse_list(tail_tuples, h.n))

    all_pauli = SparsePauliOp.from_sparse_list([*new_odd_lc_tuples, *new_even_lc_tuples, *tail_tuples], h.n).simplify() 
    lc_pauli = SparsePauliOp.from_sparse_list([*odd_lc_tuples, *even_lc_tuples, *tail_tuples], h.n).simplify()
    difference = (all_pauli - lc_pauli).simplify()
    if len(difference.coeffs) == 1 and difference.coeffs[0] == 0:
        # print('Success!')
        if verbose: print('Success!')
    else:
        print('Lightcone decomposition error!: ', difference)

    if legacy:
        return [even_lc_terms, odd_lc_terms, tail_lc_terms]
    else:
        even_lc_terms = SparsePauliOp.from_sparse_list(new_even_lc_tuples, h.n).simplify()
        odd_lc_terms  = SparsePauliOp.from_sparse_list(new_odd_lc_tuples, h.n).simplify()

        return [even_lc_terms, odd_lc_terms, tail_lc_terms]


def nested_commutator_norm(h_list):
    if len(h_list) == 2:
        c1_cmm = commutator(h_list[1], commutator(h_list[1], h_list[0]).simplify()).simplify()
        c2_cmm = commutator(h_list[0], commutator(h_list[0], h_list[1]).simplify()).simplify()
        c1_cmm_norm = np.linalg.norm(c1_cmm.coeffs, ord=1)
        c2_cmm_norm = np.linalg.norm(c2_cmm.coeffs, ord=1)
    elif len(h_list) == 3:
        c1_cmm_0 = commutator(h_list[1]+h_list[2], commutator(h_list[1]+h_list[2], h_list[0]).simplify()).simplify() 
        c1_cmm_1 = commutator(h_list[2], commutator(h_list[2], h_list[1]).simplify()).simplify()
        c2_cmm_0 = commutator(h_list[0], commutator(h_list[0], h_list[1]+h_list[2]).simplify()).simplify() 
        c2_cmm_1 = commutator(h_list[1], commutator(h_list[1], h_list[2]).simplify()).simplify()
        c1_cmm_norm = np.linalg.norm(c1_cmm_0.coeffs, ord=1) + np.linalg.norm(c1_cmm_1.coeffs, ord=1)
        c2_cmm_norm = np.linalg.norm(c2_cmm_0.coeffs, ord=1) + np.linalg.norm(c2_cmm_1.coeffs, ord=1)
    else:
        raise Exception('Invalid number of terms in the Hamiltonian list')

    return c1_cmm_norm, c2_cmm_norm

import jax
import jax.numpy as jnp
from jax import jit, vmap

import multiprocess as mp 
# import multiprocessing as mp

def lc_tail_bound(r, n, h, t, ob_type='singl', right=0, left=0, verbose=True):
    err_bound = 0
    dt = t/r
    if verbose: print(f'index={n}, r={r}')

    PROCESSES = 9
    # j_values = jnp.arange(1, r+1)
    if ob_type == 'singl':
        # @jit
        def process_item(j):
            h_list_z = lc_group(h, right, left, 2*j, verbose=False, legacy=True)
            c1_cmm_z, c2_cmm_z = nested_commutator_norm(h_list_z)
            return 2 * (c1_cmm_z * dt**3 / 12 + c2_cmm_z * dt**3 / 24)

        with mp.Pool(PROCESSES) as pool:
            results = pool.map(process_item, range(1, r+1))

        err_bound = sum(results)

        # process_item_vmap = vmap(process_item)
        # results = jax.device_get(process_item_vmap(j_values))
        # err_bound = jnp.sum(results)
        # err_bound = 0
        # for j in range(1, r+1):
        #     h_list_z  = lc_group(h, right, left, 2*j, verbose=False, legacy=True)
        #     c1_cmm_z, c2_cmm_z = nested_commutator_norm(h_list_z) 
        #     err_bound += 2 * (c1_cmm_z * dt**3 / 12 + c2_cmm_z * dt**3 / 24)
        # # err_bound = 2 * r * (c1_cmm_z * dt**3 / 12 + c2_cmm_z * dt**3 / 24)
    elif ob_type == 'multi_z':
        # err_bound = 0
        # for j in range(1, r+1):
        #     h_list_z_list = [lc_group(h, i, i, 2*j+2, verbose=False) for i in range(0, n)]
        #     # h_list_z_list = [lc_group(h, i, i, 4*r+1, False) for i in range(0, n)]
        #     # h_list_z_list = [lc_group(h, i, i, 2*r, False) for i in range(0, n)]
        #     tail_cmm_data = np.array([nested_commutator_norm(h_list_z) for h_list_z in h_list_z_list])
        #     err_bound += 2 * (sum(tail_cmm_data[:, 0])*dt**3/12 + sum(tail_cmm_data[:, 1])*dt**3/24) / n
        def process_item(j):
            h_list_z_list = [lc_group(h, i, i, 2*j+2, verbose=False) for i in range(0, n)]
            tail_cmm_data = np.array([nested_commutator_norm(h_list_z) for h_list_z in h_list_z_list])
            return 2 * (sum(tail_cmm_data[:, 0])*dt**3/12 + sum(tail_cmm_data[:, 1])*dt**3/24) / n

        with mp.Pool(PROCESSES) as pool:
            results = pool.map(process_item, range(1, r+1))

        err_bound = sum(results)
    elif ob_type == 'multi_zz':
        def process_item(j):
            h_list_zz_list = [lc_group(h, i, i+1, 2*j+2, verbose=False) for i in range(0, n-1)]
            tail_cmm_data = np.array([nested_commutator_norm(h_list_zz) for h_list_zz in h_list_zz_list])
            return 2 * (sum(tail_cmm_data[:, 0])*dt**3/12 + sum(tail_cmm_data[:, 1])*dt**3/24) / (n-1)

        with mp.Pool(PROCESSES) as pool:
            results = pool.map(process_item, range(1, r+1))

        err_bound = sum(results)
        # err_bound = 0
        # for j in range(1, r+1):
        #     h_list_zz_list = [lc_group(h, i, i+1, 2*j+2, verbose=False) for i in range(0, n-1)]
        #     # h_list_zz_list = [lc_group(h, i, i+1, 2*r, False) for i in range(0, n-1)]
        #     tail_cmm_data = np.array([nested_commutator_norm(h_list_zz) for h_list_zz in h_list_zz_list])
        #     err_bound += 2 * (sum(tail_cmm_data[:, 0])*dt**3/12 + sum(tail_cmm_data[:, 1])*dt**3/24) / (n-1)
        # # err_bound = 2 * (sum(tail_cmm_data['c1_zz'][n-1][r-1]) * dt**3 / 12 + sum(tail_cmm_data['c2_zz'][n-1][r-1]) * dt**3 / 24) / (n-1)
    else:
        raise ValueError('ob_type should be either single or multi')

    return err_bound


# def lc_nested_commutator_norm(i, h, r):
#     h_list_z  = lc_group(h, i, i, 2*r, False)
#     c1_cmm_z, c2_cmm_z = nested_commutator_norm(h_list_z) 

#     h_list_zz = lc_group(h, i, i+1, 2*r, False)
#     c1_cmm_zz, c2_cmm_zz = nested_commutator_norm(h_list_zz) 

#     return c1_cmm_z, c2_cmm_z, c1_cmm_zz, c2_cmm_zz

# def process_data(i):
#     # print(np.log(exp(i)))
#     # random matrix of dim i
#     m = np.random.rand(i, i)
#     m_norm = np.linalg.norm(m, ord=1)
#     print(i, m_norm)
#     return m_norm
