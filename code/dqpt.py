from qiskit.quantum_info import Statevector, SparsePauliOp, Operator, partial_trace, entropy, shannon_entropy, DensityMatrix
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit_algorithms import TimeEvolutionProblem, TrotterQRTE
from qiskit.synthesis import ProductFormula, LieTrotter, SuzukiTrotter
from scipy.linalg import expm
import numpy as np
import itertools as it

# from utils import lighten_color

def normalize(data):
    s = sum(a**2 for a in data)
    return [a**2/s for a in data]

def generate_proj_pstrs(n, k, verbose=False):
    local_pstrs = [''.join(i) for i in list(it.product(['I', 'Z'], repeat=k))]
    # local_pstrs = ['II', 'IZ', 'ZI', 'ZZ']
    result = np.array([['I'*i+pstr+'I'*(n-i-k) for pstr in local_pstrs] for i in range(n-1)]).flatten()   
    if verbose: print(f'local_pstrs: {local_pstrs}\nresult: {result}')

    return result

# def plot_evo(ax, t_list, y_list, marker, color='', title='', xlabel='', ylabel='', label='', markersize=5, markeredgewidth=1, inset=False):
#     if color == '':
#         ax.plot(t_list, y_list, marker, label=label, markersize=markersize, markeredgewidth=markeredgewidth)
#         # ax.plot(t_list, y_list, '-', markersize=5)
#         # ax.plot(t_list, y_list, 'o', label=label, markersize=5)
#         # ax.plot(t_list, y_list, marker, label=label, markeredgecolor='k', markeredgewidth=0.4, markersize=5)
#     else:
#         ax.plot(t_list, y_list, marker, color=color, label=label, markeredgecolor=color, markeredgewidth=markeredgewidth, markersize=markersize, mfc=lighten_color(color, 0.3))
#         # ax.plot(t_list, y_list, marker, color=color, label=label, markeredgecolor=color, markeredgewidth=0.4, markersize=markersize, mfc=color[:-2]+"80")
#     if not inset: 
#         ax.set_title(title)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#     if title  != '': ax.set_title(title)
#     if xlabel != '': 
#         ax.set_xlabel(xlabel)
#     # else:
#     #     ax.set_xticks([]) 
#     if ylabel != '': ax.set_ylabel(ylabel)
#     # else:
#     #     ax.set_xticks([])

# def letter_annotation(axes, xoffset, yoffset, letters):
#     # https://towardsdatascience.com/a-guide-to-matplotlib-subfigures-for-creating-complex-multi-panel-figures-70fa8f6c38a4
#     for letter in letters:
#         axes[letter].text(xoffset, yoffset, f'({letter})', transform=axes[letter].transAxes, size=14, weight='bold')

def ob_dt(ob_list, t_list, ord=1):
    """time derivative of observable expectation 

    Args:
        ob_list (_type_): _description_
        t_list (_type_): _description_

    Returns:
        ob_dt_list: _description_
    """
    if ord == 1:
        ob_dt_list = [(ob_list[i + 1] - ob_list[i]) / (t_list[-1]/len(t_list))  for i in range(len(ob_list) - 1)]
    elif ord == 2:
        ob_dt_list = [(ob_list[i + 2] - 2*ob_list[i + 1] + ob_list[i]) / (0.5*t_list[-1]/len(t_list))  for i in range(len(ob_list) - 2)]
    return ob_dt_list
