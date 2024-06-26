import numpy as np
from scipy.linalg import expm
import os
# from openfermion.hamiltonians import jellium_model
from openfermion.utils import Grid, count_qubits
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.linalg import get_sparse_operator

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.chem import geometry_from_pubchem

import openfermionpyscf as ofpyscf
import openfermion as of

import multiprocessing
from qiskit.quantum_info import SparsePauliOp

import matplotlib.pyplot as plt

FLOATING_POINT_PRECISION = 1e-10

def openfermion_matrix_list(qubit_operator, verbose=True):
    total_qubits = count_qubits(qubit_operator)
    matrix_list = []
    op_list = list(qubit_operator)
    if verbose: print('len(op_list): ', len(op_list))
    # print('op_list: ', op_list)
    for index, i in enumerate(op_list):
        # print(f'{index}: {i}')
        matrix_list.append(get_sparse_operator(i, total_qubits).toarray()) #changed from qubit operator and made no differnce
    # if verbose: print('len(op_list): ', len(op_list))
    return np.array(matrix_list)

    #Test -- shows ops are equivalent
def test_list_generator(openfermion_output):
    max_val = []
    of_generator = get_sparse_operator(openfermion_output).toarray()
    list_generator = sum(openfermion_matrix_list(openfermion_output))
    the_zero_op = of_generator - list_generator
    for i in range(the_zero_op.shape[0]):
        for j in range(the_zero_op.shape[0]):
            max_val.append((the_zero_op)[i][j])
    print(max(max_val))
    norm = np.linalg.norm(the_zero_op, ord=2)
    if norm < FLOATING_POINT_PRECISION:
        print("success!")
    else:
        print("failed!")
    return 0

# def jellium_hamiltonian(dimensions, length, spinless=True):
#     grid = Grid(dimensions=dimensions, length=length, scale=1.0)
#     # Get the momentum Hamiltonian.
#     momentum_hamiltonian = jellium_model(grid, spinless)
#     momentum_qubit_operator = jordan_wigner(momentum_hamiltonian)
#     momentum_qubit_operator.compress()

#     #Generate the matrix list
#     jellium_hamiltonian_list = openfermion_matrix_list(momentum_qubit_operator) #load this into simulator
#     print("Hamiltonian has dimensions: " + str(jellium_hamiltonian_list.shape))
#     #print(momentum_qubit_operator)
#     #test_list_generator(momentum_qubit_operator)
#     return jellium_hamiltonian_list

def ham_spec(hamiltonian_list):
    norms = []
    index = []
    zero_norms = 0
    for i in range(len(hamiltonian_list)):
        h = hamiltonian_list[i]
        spec = np.linalg.norm(h, ord=2)
        norms.append(spec)
        index.append(i)
        if spec == 0:
            zero_norms += 1
    print('norms: ', norms)
    norms.sort()
    print('sorted norms: ', norms)
    plt.figure(0)
    plt.plot(index, norms, 'o-', markeredgecolor='k')
    plt.xlabel("Index")
    plt.ylabel("Spectral Norm")
    plt.show()
    print("There are " + str(zero_norms) + " terms with 0 spectral norm")
    return norms

from functools import partial
def op2mat(ops, n_qubits, to_array=False):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    partial_get_sparse_operator = partial(get_sparse_operator, n_qubits=n_qubits)
    if not to_array:
        list_sparse_mat = pool.map(partial_get_sparse_operator, ops)
    
    pool.close()
    pool.join()

    return list_sparse_mat

class h2o_hamiltonian:
    def __init__(self, verbose=False):
        # Set molecule parameters
        self.use_ActiveSpace = True
        self.d = 1.0
        self.angle = 104.5
        self.x1 = self.d*np.sin(np.pi * self.angle / 360)
        self.x2 = -self.d*np.sin(np.pi * self.angle / 360)
        self.z = self.d * np.cos(np.pi * self.angle / 360)
        self.geometry = [("H", (self.x1, 0.0, self.z)), ("O", (0.0, 0.0, 0.0)), ("H", (self.x2, 0.0, self.z))]
        self.occupied_indices = [0,1]
        self.active_indices = [2,3,4,5,6]
        self.basis = "sto-3g"
        self.multiplicity = 1
        self.charge = 0
        # Perform electronic structure calculations and
        # obtain Hamiltonian as an InteractionOperator
        self.molecule = MolecularData(self.geometry, self.basis, self.multiplicity, self.charge)
        self.molecule.load()
        if self.use_ActiveSpace:
            print('=====Using active space=====')
            self.molecular_hamiltonian = self.molecule.get_molecular_hamiltonian(
                occupied_indices = self.occupied_indices,
                active_indices = self.active_indices)
        else:
            self.molecular_hamiltonian = ofpyscf.generate_molecular_hamiltonian(
                self.geometry, self.basis, self.multiplicity, self.charge
            )
        # print('molecular_hamiltonian: ', self.molecular_hamiltonian)

        # Convert to a FermionOperator
        self.fermion = of.get_fermion_operator(self.molecular_hamiltonian)
        self.n_qubits = count_qubits(self.fermion)
        # self.fermi_terms = [get_sparse_operator(op, n_qubits=self.n_qubits) for op in self.fermion]
        self.fermi_terms = op2mat(self.fermion, self.n_qubits)
        # self.hamiltonian_matrix = [get_sparse_operator(self.fermion_hamiltonian).toarray()]
        # print(new_h2o_hamiltonian)

        # Convert to a QubitOperator
        self.jw = jordan_wigner(self.fermion)
        self.jw.compress()
        # self.qubit_terms = [get_sparse_operator(op, n_qubits=self.n_qubits) for op in self.jw]
        self.qubit_terms = op2mat(self.jw, self.n_qubits)
        self.hamiltonian_list = openfermion_matrix_list(self.jw, verbose=verbose)
        # self.qubit_ham_matrix = openfermion_matrix_list(self.qubit_ham)
        # assert np.allclose(expm(-1j * sum(self.qubit_terms)), expm(-1j * sum(self.fermi_terms)))

        if verbose:
            print(f'# fermion_ham_terms: {len(self.fermi_terms)}')
            print(f'# qubit_ham_terms: {len(self.qubit_terms)}')

def tuple2pstr(term, size):
    # print(term)
    # size = max([item[0] for item in term]+[1]) + 1
    # print('size; ', size)
    temp = ['I' for _ in range(size)]
    for item in term:
        # print(item)
        temp[item[0]] = item[1]

    return ''.join(temp)

def pauli_commutator(pauli_str1, pauli_str2):
    assert len(pauli_str1) == len(pauli_str2)
    product = []
    factor = 0
    phase = 1
    for index, pauli in enumerate(list(pauli_str1)):
        # print(index, pauli, pauli_str2[index])
        if pauli == 'I':
            product.append(pauli_str2[index])
        elif pauli == 'X':
            if pauli_str2[index] == 'X':
                product.append('I')
            elif pauli_str2[index] == 'Y':
                product.append('Z')
                factor += 1
                phase *= 1j
            elif pauli_str2[index] == 'Z':
                product.append('Y')
                factor += 1
                phase *= -1j
            else:
                product.append(pauli)
        elif pauli == 'Y':  
            if pauli_str2[index] == 'X':
                product.append('Z')
                factor += 1
                phase *= -1j
            elif pauli_str2[index] == 'Y':
                product.append('I')
            elif pauli_str2[index] == 'Z':
                product.append('X')
                factor += 1
                phase *= 1j
            else:
                product.append(pauli)
        elif pauli == 'Z':
            if pauli_str2[index] == 'X':
                product.append('Y')
                factor += 1
                phase *= 1j
            elif pauli_str2[index] == 'Y':
                product.append('X')
                factor += 1
                phase *= -1j
            elif pauli_str2[index] == 'Z':
                product.append('I')
            else:
                product.append(pauli)
        else:
            raise ValueError('Invalid Pauli string')

    return product, 2*(factor%2) * phase   

class hydrogen_chain_hamiltonian:
    def __init__(self, chain_length, bond_length, verbose=False):

# def hydrogen_chain_hamiltonian(chain_length, bond_length): 
        self.chain_length = chain_length
        self.bond_length = bond_length
        self.hydrogen_geometry = []
        for i in range(self.chain_length):
            self.hydrogen_geometry.append(('H', (self.bond_length * i, 0, 0)))

        #print("Geometry in use:")
        #print(self.hydrogen_geometry)
        self.basis = 'sto-3g'
        if self.chain_length % 2 == 0:
            self.multiplicity = 1 #2ns+1
        else:
            self.multiplicity = 2

        # Set Hamiltonian parameters.
        self.active_space_start = 0
        self.active_space_stop = chain_length 

        # Set calculation parameters (to populate the molecular data class)
        self.run_scf = False #Hartree-Fock
        self.run_mp2 = False #2nd order Moller-Plesset (special case of R-S PT)
        self.run_cisd = False # Configuration interaction with single and double excitations
        self.run_ccsd = False #Coupled Cluster
        self.run_fci = True #Full configuration interaction
        self.verbose = False

        # Generate and populate instance of MolecularData.
        self.hydrogen = MolecularData(self.hydrogen_geometry, self.basis, self.multiplicity, description="hydrogen_chain_" + str(chain_length) +"_"+str(bond_length), filename="./data/hydrogen_" + str(chain_length) +"_"+str(bond_length))
        if os.path.exists(self.hydrogen.filename + '.hdf5'):
            self.hydrogen.load()
        else:
            self.hydrogen = run_pyscf(self.hydrogen, run_scf=self.run_scf, run_mp2=self.run_mp2, run_cisd=self.run_cisd, run_ccsd=self.run_ccsd, run_fci=self.run_fci, verbose=self.verbose)
            #two_body_integrals = hydrogen.two_body_integrals
            self.hydrogen.save()

        # Get the Hamiltonian in an active space.
        self.molecular_hamiltonian = self.hydrogen.get_molecular_hamiltonian(occupied_indices=range(self.active_space_start),
            active_indices=range(self.active_space_start, self.active_space_stop))

        # Map operator to fermions and qubits.
        self.fermion_hamiltonian = get_fermion_operator(self.molecular_hamiltonian)
        self.jw = jordan_wigner(self.fermion_hamiltonian)
        self.n_qubits = count_qubits(self.jw)
        self.qubit_terms = op2mat(self.jw, self.n_qubits)
        self.hamiltonian_list = openfermion_matrix_list(self.jw, verbose=self.verbose)
        self.l_terms = self.hamiltonian_list.shape[0]
        # self.ground_energy, self.ground_state = 

        self.pstrs, self.pstrs_coeff = [], []
        for term in self.jw.terms:
            # print(SparsePauliOp(tuple2pstr(term, self.n_qubits), self.jw.terms[term]))
            self.pstrs.append(tuple2pstr(term, self.n_qubits))
            self.pstrs_coeff.append(self.jw.terms[term])
        print(f'# pstr: {len(self.pstrs)}, {self.pstrs}')

        self.h_group = [self.qubit_terms[0]]
        self.pstr_group = [[{self.pstrs[0]: self.pstrs_coeff[0]}]]
        # print(self.pstrs[1:])
        # print(self.qubit_terms[0])
        for index, pstr in enumerate(self.pstrs[1:]):
            if pauli_commutator(pstr, self.pstrs[index-1])[1] == 0:
                self.h_group[-1] += self.qubit_terms[index+1]
                self.pstr_group[-1].append({pstr: self.pstrs_coeff[index+1]})
            else:
                self.h_group.append(self.qubit_terms[index+1])
                self.pstr_group.append([{pstr: self.pstrs_coeff[index+1]}])

        print(f'# groups: {len(self.h_group)}')
        print(self.pstr_group)

        if verbose:
            print(f'fermion_ham:\n {self.fermion_hamiltonian}')
            print(f'qubit_ham:\n {self.jw}')
            # print(f'hamiltonian_list: {self.hamiltonian_list}')
            print(f'L: {self.hamiltonian_list.shape}')
            print(f'grouped Hamiltonian: {self.h_group}')

    # return hydrogen_hamiltonian_list

def LiH_hamiltonian():
    basis = 'sto-3g'
    multiplicity = 1
    # Set Hamiltonian parameters.
    active_space_start = 1
    active_space_stop = 3
    LiH_geometry = geometry_from_pubchem('LiH')
    #print(LiH_geometry)
    #diatomic_bond_length = 1.45
    #LiH_geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., diatomic_bond_length))]
    #print(LiH_geometry)

    # Generate and populate instance of MolecularData.
    LiH_molecule = MolecularData(LiH_geometry, basis, multiplicity, description="1.45")
    LiH_molecule.load()

    # Get the Hamiltonian in an active space.
    LiH_molecular_hamiltonian = LiH_molecule.get_molecular_hamiltonian(
    occupied_indices=range(active_space_start),
    active_indices=range(active_space_start, active_space_stop))

    # Map operator to fermions and qubits.
    LiH_fermion_hamiltonian = get_fermion_operator(LiH_molecular_hamiltonian)
    LiH_qubit_hamiltonian = jordan_wigner(LiH_fermion_hamiltonian)
    LiH_hamiltonian_list = openfermion_matrix_list(LiH_qubit_hamiltonian)
    return LiH_hamiltonian_list