# Various utility functions used by the scripts
import itertools

from cmath import cos, exp, pi, sin, sqrt
from numpy.linalg import matrix_power
from scipy.linalg import expm
from numpy import log
import numpy as np
np.set_printoptions(precision=6)
FLOATING_POINT_PRECISION = 1e-10
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse.linalg as ssla

from scipy.optimize import curve_fit
from math import ceil, floor, exp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.colors import ListedColormap

color_cycle = ["#E64B35FF", "#47B167FF", "#0A75C7FF", "#F39B7FFF", "#70699eFF", "#4DBBD5FF", "#FFAA00FF"]
colors = mpl.cycler(color=color_cycle, alpha=[.9] * len(color_cycle)) 

mpl.rc('axes', prop_cycle=colors)

plt.rcParams['font.family'] = 'Helvetica' # 'sans-serif'
mpl.rcParams['lines.markersize'] = 10
plt.rcParams['lines.markeredgecolor'] = 'k'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"

SMALL_SIZE = 14
MEDIUM_SIZE = 15  #default 10
LARGE_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE+4)  # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=LARGE_SIZE+4)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=LARGE_SIZE+2)  # fontsize of the tick labels
plt.rc('ytick', labelsize=LARGE_SIZE+2)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title


def data_plot(x, y, marker, label, alpha=1, linewidth=1, loglog=True, markeredgecolor='black'):
    if loglog:
        plt.loglog(x, y, marker, label=label, linewidth=linewidth, markeredgecolor=markeredgecolor, markeredgewidth=0.5, alpha=alpha)
    else:
        plt.plot(x, y, marker, label=label, linewidth=linewidth, markeredgecolor=markeredgecolor, markeredgewidth=0.5, alpha=alpha)

def linear_loglog_fit(x, y, verbose=False):
    # Define the linear function
    def linear_func(x, a, b):
        return a * x + b

    log_x = np.array([log(n) for n in x])
    log_y = np.array([log(cost) for cost in y])
    # Fit the linear function to the data
    params, covariance = curve_fit(linear_func, log_x, log_y)
    # Extract the parameters
    a, b = params
    # Predict y values
    y_pred = linear_func(log_x, a, b)
    # Print the parameters
    if verbose: print('Slope (a):', a, 'Intercept (b):', b)
    exp_y_pred = [exp(cost) for cost in y_pred]

    return exp_y_pred, a, b

def plot_fit(ax, x, y, var='t', x_offset=1.07, y_offset=1.0, label='', ext_x=[], verbose=True):
    y_pred_em, a_em, b_em = linear_loglog_fit(x, y)
    if verbose: print(f'a_em: {a_em}; b_em: {b_em}')
    text_a_em = "{:.2f}".format(round(abs(a_em), 4))

    if ext_x != []: x = ext_x
    y_pred_em = [exp(cost) for cost in a_em*np.array([log(n) for n in x]) + b_em]
    if label =='':
        ax.plot(x, y_pred_em, 'k--', linewidth=1)
    else:
        ax.plot(x, y_pred_em, 'k--', linewidth=1, label=label)
    ax.annotate(r'$O(%s^{%s})$' % (var, text_a_em), xy=(x[-1], np.real(y_pred_em)[-1]), xytext=(x[-1]*x_offset, np.real(y_pred_em)[-1]*y_offset))