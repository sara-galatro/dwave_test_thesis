'''
Library of functions used in the notebooks for my Master's Thesis.

Inspired by the work of ADD AUTHORS, ADD ARTICLE.
'''

#################################### Imports

import numpy as np
from sympy import symbols, Subs
from pyqubo import Binary, Array

import dimod, minorminer
from dimod import BQM
from dwave.system import EmbeddingComposite, DWaveSampler
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.inspector import show

from examples_nnf import *

#################################### Hamiltonian construction

def building_H_direct_pyqubo(bits: int, chosen_model: str):

    '''
        Build the Hamiltonian in direct embedding, either 
        in QUBO model or BQM model, using PyQubo.

        Inputs:
            bits (int): number of variables/classical bits
            chosen_model (str): variable that determines the model to build. 
                                The possible values are QUBO or BQM.

        Outputs:
            H (bqm/qubo pyqubo obj): Hamiltonian in the chosen model
            sol (list(list)): knowns solution(s) of the given MQ system
    '''

    ## setup
    #x_bin = Array([Binary(f'x{i}') for i in range(bits)])  # variables 0-8
    x_bin = Array([Binary(f'x{i}') for i in range(1,bits+1)])       # variables 1-9
    #display(x_bin)

    if bits == 5:
        p_bin, sol = example_5(*x_bin)      # system def + known solution
    elif bits == 9:
        p_bin, sol = example_9(*x_bin)      # system def + known solution
    print(f'Actual solution(s) for the MQ problem:\n{sol}')

    print('\n-----------------------------------------------------\n')

    ## Hamiltonian
    H_bin = sum(p_bin)
    model_H = H_bin.compile()
    if chosen_model == 'QUBO':
        print('You chose the QUBO model.')
        H = model_H.to_qubo()
    elif chosen_model == 'BQM':
        print('You chose the BQM model.')
        H = model_H.to_bqm() 
    else:
        print('You can only choose QUBO or BQM')

    print('\n-----------------------------------------------------')

    return(H, sol)

def example_9_from_sym(x0, x1, x2, x3, x4, x5, x6, x7, x8):

    '''
        Build the 9-bits Hamiltonian from symbolic calculus. 
        Using sympy first ensures that the variables are simplified (i.e. summed).

        Inputs:
            x0, x1, x2, x3, x4, x5, x6, x7, x8 (symbols || Binary variables):
                variables to substitute, either with symbols or Binary variables from pyqubo.

        Outputs:
            H (expression || pyqubo.obj): Hamiltonian, either symbolic or as pyqubo Binary object.
    '''

    # broken in multiple lines for parse error
    H = 16*x0*x1*x2*x3*x4*x5*x6*x7*x8 - 8*x0*x1*x2*x3*x4*x5*x6*x8 + 8*x0*x1*x2*x3*x4*x5*x6 - 8*x0*x1*x2*x3*x4*x5*x7*x8 - 16*x0*x1*x2*x3*x4*x5*x7 - 8*x0*x1*x2*x3*x4*x5*x8 + 12*x0*x1*x2*x3*x4*x5 - 32*x0*x1*x2*x3*x4*x6*x7*x8 + 16*x0*x1*x2*x3*x4*x6*x7 + 24*x0*x1*x2*x3*x4*x6*x8 - 12*x0*x1*x2*x3*x4*x6 + 24*x0*x1*x2*x3*x4*x7*x8 
    H += - 8*x0*x1*x2*x3*x4*x8 + 8*x0*x1*x2*x3*x5*x6*x7*x8 - 16*x0*x1*x2*x3*x5*x6*x7 - 4*x0*x1*x2*x3*x5*x6 + 16*x0*x1*x2*x3*x5*x7 + 8*x0*x1*x2*x3*x5*x8 - 8*x0*x1*x2*x3*x5 + 16*x0*x1*x2*x3*x6*x7*x8 - 4*x0*x1*x2*x3*x6*x7 - 12*x0*x1*x2*x3*x6*x8 + 8*x0*x1*x2*x3*x6 - 12*x0*x1*x2*x3*x7*x8 - 8*x0*x1*x2*x4*x5*x6*x7*x8 - 8*x0*x1*x2*x4*x5*x6*x8 
    H += - 4*x0*x1*x2*x4*x5*x6 + 16*x0*x1*x2*x4*x5*x7*x8 + 4*x0*x1*x2*x4*x5*x7 + 8*x0*x1*x2*x4*x5*x8 - 4*x0*x1*x2*x4*x5 + 24*x0*x1*x2*x4*x6*x7*x8 - 8*x0*x1*x2*x4*x6*x7 - 4*x0*x1*x2*x4*x6*x8 + 4*x0*x1*x2*x4*x6 - 28*x0*x1*x2*x4*x7*x8 + 4*x0*x1*x2*x4*x7 + 4*x0*x1*x2*x4*x8 - 2*x0*x1*x2*x4 + 8*x0*x1*x2*x5*x6*x7 - 4*x0*x1*x2*x5*x6*x8 + 8*x0*x1*x2*x5*x6 
    H += - 8*x0*x1*x2*x5*x7*x8 - 8*x0*x1*x2*x5*x7 - 12*x0*x1*x2*x6*x7*x8 + 8*x0*x1*x2*x6*x8 - 4*x0*x1*x2*x6 + 12*x0*x1*x2*x7*x8 - 4*x0*x1*x2*x8 + 2*x0*x1*x2 + 8*x0*x1*x3*x4*x5*x6*x8 - 8*x0*x1*x3*x4*x5*x6 - 8*x0*x1*x3*x4*x5*x7*x8 + 12*x0*x1*x3*x4*x5*x7 + 4*x0*x1*x3*x4*x5*x8 - 4*x0*x1*x3*x4*x5 + 16*x0*x1*x3*x4*x6*x7*x8 - 12*x0*x1*x3*x4*x6*x7 
    H += - 16*x0*x1*x3*x4*x6*x8 + 8*x0*x1*x3*x4*x6 - 12*x0*x1*x3*x4*x7*x8 + 8*x0*x1*x3*x4*x8 - 2*x0*x1*x3*x4 - 12*x0*x1*x3*x5*x6*x8 + 12*x0*x1*x3*x5*x6 - 4*x0*x1*x3*x5*x7 + 4*x0*x1*x3*x5*x8 - 2*x0*x1*x3*x5 - 12*x0*x1*x3*x6*x7*x8 + 8*x0*x1*x3*x6*x7 + 8*x0*x1*x3*x6*x8 - 6*x0*x1*x3*x6 + 8*x0*x1*x3*x7*x8 - 2*x0*x1*x3*x7 - 4*x0*x1*x3*x8 + 2*x0*x1*x3 
    H += + 8*x0*x1*x4*x5*x6*x7*x8 - 4*x0*x1*x4*x5*x6*x7 + 4*x0*x1*x4*x5*x6 - 8*x0*x1*x4*x5*x7*x8 - 16*x0*x1*x4*x6*x7*x8 + 8*x0*x1*x4*x6*x7 + 4*x0*x1*x4*x6*x8 - 2*x0*x1*x4*x6 + 16*x0*x1*x4*x7*x8 - 2*x0*x1*x4*x7 - 4*x0*x1*x4*x8 - 4*x0*x1*x5*x6*x7*x8 + 4*x0*x1*x5*x6*x8 - 6*x0*x1*x5*x6 + 4*x0*x1*x5*x7*x8 + 2*x0*x1*x5*x7 - 4*x0*x1*x5*x8 + 2*x0*x1*x5 
    H += + 8*x0*x1*x6*x7*x8 - 4*x0*x1*x6*x7 - 4*x0*x1*x6*x8 + 2*x0*x1*x6 - 6*x0*x1*x7*x8 + 4*x0*x1*x8 - x0*x1 - 32*x0*x2*x3*x4*x5*x6*x7*x8 + 16*x0*x2*x3*x4*x5*x6*x7 + 16*x0*x2*x3*x4*x5*x6*x8 - 8*x0*x2*x3*x4*x5*x6 + 8*x0*x2*x3*x4*x5*x7*x8 - 4*x0*x2*x3*x4*x5*x7 + 8*x0*x2*x3*x4*x5*x8 - 8*x0*x2*x3*x4*x5 + 24*x0*x2*x3*x4*x6*x7*x8 - 12*x0*x2*x3*x4*x6*x7 
    H += - 12*x0*x2*x3*x4*x6*x8 + 4*x0*x2*x3*x4*x6 - 16*x0*x2*x3*x4*x7*x8 + 8*x0*x2*x3*x4*x7 + 4*x0*x2*x3*x4*x8 - 2*x0*x2*x3*x4 + 8*x0*x2*x3*x5*x6*x7*x8 - 8*x0*x2*x3*x5*x6*x7 - 8*x0*x2*x3*x5*x6*x8 + 12*x0*x2*x3*x5*x6 - 4*x0*x2*x3*x5*x7*x8 + 4*x0*x2*x3*x5*x7 - 4*x0*x2*x3*x5*x8 - 12*x0*x2*x3*x6*x7*x8 + 4*x0*x2*x3*x6*x7 + 8*x0*x2*x3*x6*x8 - 4*x0*x2*x3*x6 
    H += + 4*x0*x2*x3*x7*x8 - 4*x0*x2*x3*x7 + 4*x0*x2*x3*x8 + 16*x0*x2*x4*x5*x6*x7*x8 - 8*x0*x2*x4*x5*x6*x7 - 4*x0*x2*x4*x5*x6*x8 + 4*x0*x2*x4*x5*x6 - 16*x0*x2*x4*x5*x7*x8 + 8*x0*x2*x4*x5*x7 - 16*x0*x2*x4*x6*x7*x8 + 4*x0*x2*x4*x6*x7 + 2*x0*x2*x4*x6 + 20*x0*x2*x4*x7*x8 - 6*x0*x2*x4*x7 - 4*x0*x2*x4*x8 + 2*x0*x2*x4 - 8*x0*x2*x5*x6*x7*x8 + 8*x0*x2*x5*x6*x7 
    H += + 4*x0*x2*x5*x6*x8 - 8*x0*x2*x5*x6 + 12*x0*x2*x5*x7*x8 - 6*x0*x2*x5*x7 - 4*x0*x2*x5*x8 + 4*x0*x2*x5 + 12*x0*x2*x6*x7*x8 - 4*x0*x2*x6*x7 - 4*x0*x2*x6*x8 - 10*x0*x2*x7*x8 + 4*x0*x2*x7 + 2*x0*x2*x8 - x0*x2 + 8*x0*x3*x4*x5*x6*x7*x8 - 4*x0*x3*x4*x5*x6*x7 - 12*x0*x3*x4*x5*x6*x8 + 4*x0*x3*x4*x5*x6 + 4*x0*x3*x4*x5*x8 - 4*x0*x3*x4*x6*x7*x8 + 4*x0*x3*x4*x6*x7 
    H += + 4*x0*x3*x4*x6*x8 + 8*x0*x3*x4*x7*x8 - 4*x0*x3*x4*x7 - 6*x0*x3*x4*x8 + 2*x0*x3*x4 - 8*x0*x3*x5*x6*x7*x8 + 4*x0*x3*x5*x6*x7 + 16*x0*x3*x5*x6*x8 - 10*x0*x3*x5*x6 + 8*x0*x3*x5*x7*x8 - 6*x0*x3*x5*x7 - 10*x0*x3*x5*x8 + 6*x0*x3*x5 + 4*x0*x3*x6*x7*x8 - 2*x0*x3*x6*x7 - 4*x0*x3*x6*x8 + 2*x0*x3*x6 - 4*x0*x3*x7*x8 + 4*x0*x3*x7 + 2*x0*x3*x8 - 2*x0*x3 
    H += - 12*x0*x4*x5*x6*x7*x8 + 8*x0*x4*x5*x6*x7 + 8*x0*x4*x5*x6*x8 - 4*x0*x4*x5*x6 + 8*x0*x4*x5*x7*x8 - 6*x0*x4*x5*x7 - 6*x0*x4*x5*x8 + 4*x0*x4*x5 + 4*x0*x4*x6*x7*x8 - 2*x0*x4*x6*x7 - 2*x0*x4*x6 - 8*x0*x4*x7*x8 + 2*x0*x4*x7 + 4*x0*x4*x8 - x0*x4 + 8*x0*x5*x6*x7*x8 - 6*x0*x5*x6*x7 - 8*x0*x5*x6*x8 + 6*x0*x5*x6 - 8*x0*x5*x7*x8 + 6*x0*x5*x7 + 8*x0*x5*x8 - 6*x0*x5 
    H += - 2*x0*x6*x7*x8 + 2*x0*x6*x7 + 2*x0*x6*x8 + 4*x0*x7*x8 - 2*x0*x7 - 4*x0*x8 + 2*x0 - 8*x1*x2*x3*x4*x5*x6*x7*x8 - 8*x1*x2*x3*x4*x5*x6*x7 - 8*x1*x2*x3*x4*x5*x6*x8 + 8*x1*x2*x3*x4*x5*x6 + 24*x1*x2*x3*x4*x5*x7*x8 + 4*x1*x2*x3*x4*x5*x7 - 8*x1*x2*x3*x4*x5 + 24*x1*x2*x3*x4*x6*x7*x8 - 12*x1*x2*x3*x4*x6*x7 - 4*x1*x2*x3*x4*x6*x8 - 24*x1*x2*x3*x4*x7*x8 + 8*x1*x2*x3*x4*x7 
    H += + 4*x1*x2*x3*x4*x8 + 8*x1*x2*x3*x5*x6*x7*x8 + 12*x1*x2*x3*x5*x6*x7 - 4*x1*x2*x3*x5*x6*x8 - 4*x1*x2*x3*x5*x6 - 20*x1*x2*x3*x5*x7*x8 - 4*x1*x2*x3*x5*x7 + 4*x1*x2*x3*x5*x8 + 4*x1*x2*x3*x5 - 16*x1*x2*x3*x6*x7*x8 + 8*x1*x2*x3*x6*x8 + 16*x1*x2*x3*x7*x8 - 4*x1*x2*x3*x7 - 4*x1*x2*x3*x8 - 16*x1*x2*x4*x5*x6*x7*x8 + 16*x1*x2*x4*x5*x6*x7 + 20*x1*x2*x4*x5*x6*x8 - 8*x1*x2*x4*x5*x6 
    H += - 8*x1*x2*x4*x5*x7*x8 - 4*x1*x2*x4*x5*x7 - 8*x1*x2*x4*x5*x8 + 6*x1*x2*x4*x5 - 8*x1*x2*x4*x6*x7*x8 - 8*x1*x2*x4*x6*x8 + 4*x1*x2*x4*x6 + 20*x1*x2*x4*x7*x8 - 6*x1*x2*x4*x7 + 4*x1*x2*x5*x6*x7*x8 - 12*x1*x2*x5*x6*x7 - 4*x1*x2*x5*x6*x8 + 4*x1*x2*x5*x6 + 8*x1*x2*x5*x7*x8 + 4*x1*x2*x5*x7 - 2*x1*x2*x5 + 8*x1*x2*x6*x7*x8 + 4*x1*x2*x6*x7 - 2*x1*x2*x6*x8 - 4*x1*x2*x6 
    H += - 12*x1*x2*x7*x8 + 2*x1*x2*x7 + 4*x1*x2*x8 - 8*x1*x3*x4*x5*x6*x7*x8 + 8*x1*x3*x4*x5*x6*x7 + 4*x1*x3*x4*x5*x6*x8 - 4*x1*x3*x4*x5*x7*x8 - 4*x1*x3*x4*x5*x7 - 12*x1*x3*x4*x6*x7*x8 + 8*x1*x3*x4*x6*x7 + 4*x1*x3*x4*x6*x8 + 12*x1*x3*x4*x7*x8 - 4*x1*x3*x4*x7 - 4*x1*x3*x4*x8 - 4*x1*x3*x5*x6*x7*x8 - 4*x1*x3*x5*x6*x7 + 8*x1*x3*x5*x6*x8 - 4*x1*x3*x5*x6 + 12*x1*x3*x5*x7*x8 
    H += - 2*x1*x3*x5*x7 - 8*x1*x3*x5*x8 + 4*x1*x3*x5 + 12*x1*x3*x6*x7*x8 - 4*x1*x3*x6*x7 - 8*x1*x3*x6*x8 + 2*x1*x3*x6 - 10*x1*x3*x7*x8 + 4*x1*x3*x7 + 6*x1*x3*x8 - 2*x1*x3 + 8*x1*x4*x5*x6*x7*x8 - 8*x1*x4*x5*x6*x7 - 8*x1*x4*x5*x6*x8 + 2*x1*x4*x5*x6 + 4*x1*x4*x5*x7*x8 + 4*x1*x4*x6*x7*x8 + 4*x1*x4*x6*x8 - 4*x1*x4*x6 - 10*x1*x4*x7*x8 + 2*x1*x4*x7 + 2*x1*x4*x8 + x1*x4 + 4*x1*x5*x6*x7 
    H += - 4*x1*x5*x7*x8 + 4*x1*x5*x8 - 2*x1*x5 - 6*x1*x6*x7*x8 + 2*x1*x6*x8 + 2*x1*x6 + 6*x1*x7*x8 - x1*x7 - 5*x1*x8 + x1 + 8*x2*x3*x4*x5*x6*x7*x8 + 4*x2*x3*x4*x5*x6*x8 - 4*x2*x3*x4*x5*x6 - 16*x2*x3*x4*x5*x7*x8 + 4*x2*x3*x4*x5*x7 - 4*x2*x3*x4*x5*x8 + 6*x2*x3*x4*x5 - 8*x2*x3*x4*x6*x7*x8 - 4*x2*x3*x4*x6*x8 + 6*x2*x3*x4*x6 + 12*x2*x3*x4*x7*x8 - 4*x2*x3*x4*x7 - 2*x2*x3*x4 - 8*x2*x3*x5*x6*x7*x8 
    H += - 4*x2*x3*x5*x6*x7 + 4*x2*x3*x5*x6*x8 - 2*x2*x3*x5*x6 + 8*x2*x3*x5*x7*x8 + 2*x2*x3*x5*x7 + 2*x2*x3*x5*x8 - 4*x2*x3*x5 + 8*x2*x3*x6*x7*x8 + 4*x2*x3*x6*x7 - 4*x2*x3*x6*x8 - 2*x2*x3*x6 - 4*x2*x3*x7*x8 - 2*x2*x3*x8 + 3*x2*x3 - 4*x2*x4*x5*x6*x7 - 4*x2*x4*x5*x6*x8 + 2*x2*x4*x5*x6 + 12*x2*x4*x5*x7*x8 - 4*x2*x4*x5*x7 - 2*x2*x4*x5 + 4*x2*x4*x6*x7*x8 + 4*x2*x4*x6*x7 + 6*x2*x4*x6*x8 - 6*x2*x4*x6 - 14*x2*x4*x7*x8 + 4*x2*x4*x7 + 2*x2*x4*x8 + 4*x2*x5*x6*x7*x8 + 2*x2*x5*x6*x7 - 6*x2*x5*x7*x8 + x2*x5 - 8*x2*x6*x7*x8 - 2*x2*x6*x7 + 2*x2*x6*x8 + 4*x2*x6 + 8*x2*x7*x8 - 2*x2*x7 - 2*x2*x8 - x2 + 4*x3*x4*x5*x6*x7*x8 - 8*x3*x4*x5*x6*x7 + 4*x3*x4*x5*x6 + 4*x3*x4*x5*x7*x8 - 4*x3*x4*x5*x8 + 2*x3*x4*x6*x7 + 2*x3*x4*x6*x8 - 6*x3*x4*x6 - 4*x3*x4*x7*x8 + 2*x3*x4*x8 + 2*x3*x4 + 4*x3*x5*x6*x7*x8 + 6*x3*x5*x6*x7 - 8*x3*x5*x6*x8 + 2*x3*x5*x6 - 8*x3*x5*x7*x8 + 2*x3*x5*x7 + 8*x3*x5*x8 - 3*x3*x5 - 4*x3*x6*x7*x8 - 2*x3*x6*x7 + 4*x3*x6*x8 + x3*x6 + 2*x3*x7*x8 - x3*x7 - 2*x3*x8 + 4*x4*x5*x6*x7 - 2*x4*x5*x6 - 4*x4*x5*x7*x8 + 2*x4*x5*x7 + 4*x4*x5*x8 - x4*x5 + 2*x4*x6*x7*x8 - 4*x4*x6*x7 - 4*x4*x6*x8 + 5*x4*x6 + 4*x4*x7*x8 + x4*x7 - 2*x4*x8 - x4 - 2*x5*x6*x7*x8 - 2*x5*x6*x7 + 2*x5*x6*x8 + 2*x5*x7*x8 - x5*x7 - 4*x5*x8 + 2*x5 + 2*x6*x7*x8 + x6*x7 - x6*x8 - 2*x6 - 2*x7*x8 + 3*x8 + 4
    
    return(H) 

# i don't think i'll use, but let's save it
# better to use the symbolic_to_dwave in article_lib
def example_9_from_article(x1, x2, x3, x4, x5, x6, x7, x8, x9, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37):
    
    '''
        Build the 9-bits Hamiltonian from the article construction.

        Inputs:
            args (symbols || Binary variables):
                variables and ancillae to substitute, either with symbols or Binary variables from pyqubo.

        Outputs:
            H (expression || pyqubo.obj): Hamiltonian, either symbolic or as pyqubo Binary object.
    '''

    # broken for parse error
    H = - 86.0*a1*a10 - 82.0*a1*a11 + 157.0*a1*a2 - 6*a1*a27 - 6*a1*a28 - 4*a1*a29 - 314.0*a1*a3 + 4*a1*a31 + 4*a1*a32 + 8*a1*a33 + 2*a1*a34 - 4*a1*a35 - 4*a1*a36 - 4*a1*a37 + 67.0*a1*a5 - 134.0*a1*a6 + 63.0*a1*a7 - 126.0*a1*a8 - 82.0*a1*a9 - 848.0*a1*x1 - 848.0*a1*x2 + 41.0*a1*x3 + 43.0*a1*x4 + 41.0*a1*x5 + 2*a1*x6 + 2*a1*x7 + 4*a1*x9 + 1271.0*a1 + 12*a10*a27 + 8*a10*a28 - 12*a10*a31 - 12*a10*a33 - 4*a10*a34 + 4*a10*a35 + 8*a10*a36 + 8*a10*a37 - 86.0*a10*x4 - 2*a10*x6 - 6*a10*x7 - 2*a10*x8 - 4*a10*x9 + 131.0*a10 + 4*a11*a27 + 16*a11*a28 + 8*a11*a29 - 4*a11*a30 - 8*a11*a32 - 16*a11*a33 + 8*a11*a36 + 4*a11*a37 - 82.0*a11*x5 - 2*a11*x7 - 2*a11*x8 - 4*a11*x9 + 123.0*a11 - 190.0*a12*a13 + 95.0*a12*a2 - 4*a12*a27 - 8*a12*a28 - 12*a12*a29 + 8*a12*a30 + 8*a12*a31 + 8*a12*a32 + 4*a12*a33 - 6*a12*a34 - 6*a12*a35 - 2*a12*a36 - 272.0*a12*x1 - 272.0*a12*x5 + 4*a12*x6 - 2*a12*x7 + 2*a12*x8 + 4*a12*x9 + 407.0*a12 - 190.0*a13*a2 - 8*a13*a27 - 16*a13*a28 - 32*a13*a29 + 16*a13*a30 + 16*a13*a31 + 8*a13*a32 + 24*a13*a33 - 4*a13*a34 + 8*a13*a35  - 12*a13*a36 - 12*a13*a37 - 8*a13*x6 + 4*a13*x7 + 8*a13*x8 + 4*a13*x9 + 283.0*a13 - 146.0*a14*a15 + 73.0*a14*a2 + 2*a14*a27 - 10*a14*a28 + 8*a14*a29 - 8*a14*a30 - 8*a14*a31 + 4*a14*a32 + 4*a14*a33 + 4*a14*a37 - 206.0*a14*x2 - 206.0*a14*x5 - 4*a14*x7 + 2*a14*x8 + 2*a14*x9 + 310.0*a14 - 146.0*a15*a2 + 8*a15*a27 - 24*a15*a28 - 8*a15*a29 - 8*a15*a30 - 8*a15*a31 + 24*a15*a32 + 24*a15*a33 + 4*a15*a34 - 12*a15*a36 - 4*a15*a37 - 8*a15*x6 + 8*a15*x8 + 4*a15*x9 + 219.0*a15 - 90.0*a16*a2 + 12*a16*a27 + 4*a16*a28 + 8*a16*a29 - 8*a16*a30 - 8*a16*a31 - 4*a16*a32 - 12*a16*a33 + 4*a16*a34 - 4*a16*a35 + 4*a16*a36 + 8*a16*a37 - 90.0*a16*x1 - 4*a16*x7 - 4*a16*x8 + 4*a16*x9 + 135.0*a16 - 114.0*a17*a2 - 4*a17*a27 + 16*a17*a28 + 8*a17*a29 + 12*a17*a30 - 4*a17*a31 - 20*a17*a32 - 16*a17*a33 - 4*a17*a34 + 4*a17*a35 + 8*a17*a37 - 114.0*a17*x2 + 4*a17*x6 - 4*a17*x8 - 4*a17*x9 + 171.0*a17 - 86.0*a18*a2 - 4*a18*a27 + 12*a18*a28 + 8*a18*a29 + 4*a18*a31 - 16*a18*a32 - 8*a18*a33 + 4*a18*a34 
    H += - 4*a18*a35 - 4*a18*a37 - 86.0*a18*x5 + 6*a18*x6 + 6*a18*x7 - 4*a18*x8 + 127.0*a18 - 8*a19*a27 + 20*a19*a28 - 16*a19*a29 + 16*a19*a30 + 20*a19*a31 - 8*a19*a32 - 8*a19*a33 - 4*a19*a34 - 8*a19*a35 - 8*a19*a37 - 134.0*a19*a5 - 134.0*a19*x2 + 6*a19*x6 + 4*a19*x7 - 6*a19*x8 + 201.0*a19 - 2*a2*a27 - 4*a2*a28 - 8*a2*a29 - 314.0*a2*a3 - 4*a2*a30 + 4*a2*a31 + 8*a2*a32 + 8*a2*a33 + 2*a2*a34 + 2*a2*a35 + 4*a2*a36 - 4*a2*a37 + 45.0*a2*x1 + 57.0*a2*x2 - 974.0*a2*x3 - 974.0*a2*x4 + 43.0*a2*x5 - 4*a2*x6 - 2*a2*x7 - 2*a2*x9 + 1464.0*a2 + 12*a20*a28 - 8*a20*a29 + 8*a20*a30 + 4*a20*a31 - 4*a20*a32 - 12*a20*a33 - 4*a20*a34 + 8*a20*a36 + 4*a20*a37 - 74.0*a20*a7 - 74.0*a20*x2 - 4*a20*x8 - 4*a20*x9 + 111.0*a20 + 4*a21*a27 + 8*a21*a28 + 8*a21*a29 - 4*a21*a30 - 12*a21*a31 - 4*a21*a33 + 4*a21*a35 + 4*a21*a36 + 4*a21*a37 - 70.0*a21*a7 - 70.0*a21*x1 - 4*a21*x8 - 6*a21*x9 + 107.0*a21 + 4*a22*a27 + 20*a22*a28 + 16*a22*a29 - 8*a22*a30 - 4*a22*a31 - 16*a22*a32 - 16*a22*a33 + 8*a22*a34 + 4*a22*a36 - 114.0*a22*a5 - 114.0*a22*x1 + 2*a22*x7 - 6*a22*x8 - 4*a22*x9 + 173.0*a22 - 10*a23*a27 - 4*a23*a28 - 8*a23*a29 + 4*a23*a30 + 16*a23*a31 + 8*a23*a32 + 4*a23*a33 - 6*a23*a34 - 10*a23*a35 - 2*a23*a36 - 4*a23*a37 - 94.0*a23*x1 - 94.0*a23*x4 + 6*a23*x6 + 2*a23*x7 + 4*a23*x8 + 2*a23*x9 + 139.0*a23 - 4*a24*a27 - 10*a24*a28 - 4*a24*a29 - 4*a24*a30 + 8*a24*a31 + 12*a24*a32 + 12*a24*a33 - 2*a24*a34 - 8*a24*a35 - 4*a24*a36 - 8*a24*a37 - 98.0*a24*x2 - 98.0*a24*x4 + 4*a24*x6 + 2*a24*x7 + 4*a24*x8 + 6*a24*x9 + 145.0*a24 - 8*a25*a27 - 10*a25*a28 - 8*a25*a29 + 8*a25*a30 + 4*a25*a31 + 12*a25*a32 + 12*a25*a33 - 6*a25*a34 - 4*a25*a35 - 4*a25*a36 - 4*a25*a37 - 94.0*a25*x1 - 94.0*a25*x3 + 4*a25*x6 + 4*a25*x8 + 2*a25*x9 + 140.0*a25 + 4*a26*a27 - 12*a26*a28 + 4*a26*a29 - 12*a26*a30 - 4*a26*a31 + 8*a26*a32 + 8*a26*a33 + 4*a26*a34 + 4*a26*a36 - 2*a26*a37 - 78.0*a26*x2 - 78.0*a26*x3 - 2*a26*x6 - 4*a26*x7 + 2*a26*x8 + 4*a26*x9 + 117.0*a26 + 119.0*a27*a28 - 238.0*a27*a29 - 4*a27*a3 - 194.0*a27*a30 
    H += - 202.0*a27*a31 + 8*a27*a4 + 2*a27*a5 - 4*a27*a6 + 4*a27*a7 - 8*a27*a8 + 8*a27*a9 + 6*a27*x1 + 2*a27*x4 - 2*a27*x5 - 782.0*a27*x6 - 782.0*a27*x7 + 97.0*a27*x8 + 101.0*a27*x9 + 1173.0*a27 - 238.0*a28*a29 - 12*a28*a3 - 262.0*a28*a32 - 346.0*a28*a33 + 24*a28*a4 - 14*a28*a5 - 28*a28*a6 - 4*a28*a7 - 12*a28*a8 + 12*a28*a9 + 4*a28*x1 + 6*a28*x2 + 8*a28*x3 + 2*a28*x4 + 4*a28*x5 + 131.0*a28*x6 + 173.0*a28*x7 - 1186.0*a28*x8 - 1186.0*a28*x9 + 1777.0*a28 + 8*a29*a3 + 16*a29*a4 - 8*a29*a6 + 4*a29*a7 + 8*a29*x1 + 4*a29*x3 + 4*a29*x4 + 355.0*a29 - 16*a3*a30 + 16*a3*a33 + 16*a3*a34 + 8*a3*a35 - 4*a3*a36 - 12*a3*a37 - 202.0*a3*a4 + 101.0*a3*x5 - 8*a3*x6 + 8*a3*x7 + 471.0*a3 - 4*a30*a5 - 8*a30*a7 + 8*a30*a9 - 6*a30*x1 + 4*a30*x2 + 2*a30*x3 + 6*a30*x4 + 4*a30*x5 - 194.0*a30*x8 + 289.0*a30 - 8*a31*a4 - 4*a31*a5 - 8*a31*a6 + 8*a31*a8 - 4*a31*a9 - 8*a31*x1 - 8*a31*x4 - 202.0*a31*x9 + 305.0*a31 - 8*a32*a4 + 12*a32*a5 + 16*a32*a6 + 4*a32*a7 - 8*a32*a8 - 8*a32*a9 - 8*a32*x1 - 4*a32*x2 - 6*a32*x3 - 8*a32*x4 - 4*a32*x5 - 262.0*a32*x6 + 395.0*a32 - 32*a33*a4 + 4*a33*a5 + 24*a33*a6 + 16*a33*a8 - 12*a33*a9 - 2*a33*x1 - 6*a33*x2 - 8*a33*x3 - 4*a33*x4 + 2*a33*x5 - 346.0*a33*x7 + 521.0*a33 - 16*a34*a4 - 4*a34*a5 + 4*a34*a6 + 12*a34*a8 - 8*a34*a9 + 6*a34*x1 + 2*a34*x4 + 2*a34*x5 - 142.0*a34*x6 - 142.0*a34*x8 + 212.0*a34 - 8*a35*a4 + 8*a35*a6 - 4*a35*a7 + 4*a35*a8 + 8*a35*x1 + 4*a35*x2 + 8*a35*x4 + 4*a35*x5 - 134.0*a35*x6 - 134.0*a35*x9 + 197.0*a35 + 16*a36*a4 + 4*a36*a5 - 8*a36*a6 + 2*a36*a7 - 12*a36*a8 + 2*a36*x1 - 2*a36*x3 - 2*a36*x4 - 4*a36*x5 - 146.0*a36*x7 - 146.0*a36*x8 + 220.0*a36 + 24*a37*a4 + 6*a37*a5 - 4*a37*a6 + 2*a37*a7 - 16*a37*a8 + 8*a37*a9 + 2*a37*x1 + 2*a37*x2 + 2*a37*x3 + 4*a37*x4 - 4*a37*x5 - 184.0*a37*x7 - 184.0*a37*x9 + 275.0*a37 - 202.0*a4*x5 + 12*a4*x6 - 12*a4*x7 - 8*a4*x9 + 303.0*a4 - 134.0*a5*a6 + 57.0*a5*x1 + 67.0*a5*x2 - 442.0*a5*x3 - 442.0*a5*x5 - 2*a5*x6 - 6*a5*x7 + 4*a5*x8 + 2*a5*x9 + 663.0*a5 - 4*a6*x6 + 4*a6*x7 + 4*a6*x8 + 4*a6*x9 + 199.0*a6 - 126.0*a7*a8 + 35.0*a7*x1 + 37.0*a7*x2 - 302.0*a7*x4 - 302.0*a7*x5 - 6*a7*x7 + 2*a7*x9 + 455.0*a7 - 4*a8*x6 + 8*a8*x7 + 8*a8*x9 + 187.0*a8 - 82.0*a9*x3 - 4*a9*x7 - 4*a9*x9 + 125.0*a9 + 424.0*x1*x2 + 47.0*x1*x3 + 47.0*x1*x4 + 136.0*x1*x5 - 6*x1*x6 - 2*x1*x8 - 4*x1*x9 + 2*x1 + 39.0*x2*x3 + 49.0*x2*x4 + 103.0*x2*x5 - 2*x2*x6 + 2*x2*x7 - x2*x8 - 5*x2*x9 + x2 + 487.0*x3*x4 + 221.0*x3*x5 + x3*x6 + 4*x3*x7 - 2*x3*x8 - 2*x3*x9 - x3 + 151.0*x4*x5 - 3*x4*x6 + x4*x7 - x4*x8 - 2*x4*x9 - x5*x6 + 5*x5*x7 + x5*x8 - 2*x5*x9 - x5 + 391.0*x6*x7 + 71.0*x6*x8 + 67.0*x6*x9 + 2*x6 + 73.0*x7*x8 + 92.0*x7*x9 - 2*x7 + 593.0*x8*x9 + 3*x9 + 4
    
    return(H)

#################################### D-Wave Running

def single_run_from_pyqubo(H, topology, chosen_chainstrength):

    '''
        Function that runs *once* the sampling for the given Hamiltonian
        on D-Wave devices.

        Inputs:
            H (BQM model): problem Hamiltonian from pyqubo building.
            topology (str): string variable identifying the architecture on
                            which will run the problem. Either "Zephyr" or "Pegasus".
            chosen_chainstrength (str): choesen chain_stregth for the run. 
                                  Either "Article" (definition from the starting article) 
                                  or "Ocean_Doc" (definition from Ocean Documentation).

        Outputs:
            response: SampleSet object with the details of the run. 
    '''

    ## parameters (same as the article)
    numruns = 1000
    T = 20

    ## sampler
    if topology == "Zephyr":
        sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'zephyr'}))
        print('Running on Zephyr Topology.')
    elif topology == "Pegasus":
        sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
        print('Running on Pegasus Topology.')

    ## chainstrength
    if chosen_chainstrength == 'Ocean_Doc':
        chain_strength = uniform_torque_compensation(H, sampler)
        print(f'You chose the chainstrength from the Ocean Documentation: {chain_strength}')
    elif chosen_chainstrength == 'Article':
        chain_strength = 0
        H_qubo = H.to_qubo()
        for i in H_qubo[0].values():
            chain_strength += abs(i)
        chain_strength *= 3/len(H_qubo[0])
        print(f'You chose the Article chainstrength: {chain_strength}.')
    else:
        print('You can ony choose Ocean_Doc or Article.')

    ## sampling
    if type(H) != dimod.binary.binary_quadratic_model.BinaryQuadraticModel:
        print('You must use a BQM model.')
    else:
        response = sampler.sample(H, chain_strength=chain_strength, 
                                num_reads=numruns, annealing_time=T, 
                                answer_mode='histogram',
                                label = 'mq_on_zephyr')
    print('Finished running the experiment!')
        
    return(response)

class dwave_runs:
    ''' 
        Class that implements the iterative procedure 
        and the single run of a D-Wave annealing.
    '''

    def __init__(self, Q, offset, bits, topology, chosen_chainstrength, numruns = 1000, T=20):
        '''
            Initialize the variables for the class.

            Inputs:
                Q (dict): QUBO model.
                offset (float): offset of the QUBO model.
                bits (int): number of variables.
                topology (str): chosen topology, either "Zephyr" or "Pegasus".
                chosen_chainstrength (str): chosen definition, either "Article" or "Ocean_Doc".
                numruns (int): number of sampling per run (default = 1000).
                T (int): annealing time (default = 20).
        '''

        self.bits = bits
        self.numruns = numruns
        self.T = T
        self.topology = topology
        self.chosen_chainstrength = chosen_chainstrength
        self.Q = Q
        self.offset = offset

        self.into_BQM()
        self.define_topology()


    def into_BQM(self):
        '''
            Build the BQM model from the QUBO model in input.
        '''
        print('Building the BQM model.')
        self.H = BQM.from_qubo(self.Q, self.offset)


    def change_numruns(self, numruns):
        ''' 
            Change the number of sampling in a run from default = 1000 to a new value.

            Inputs:
                numruns (int): number of wanted samples.
        '''
        self.numruns = numruns


    def define_topology(self):
        '''
            Define the D-Wave sampler, with automated embedding, 
            specifying the wanted topology.
        '''

        if self.topology == "Zephyr":
            self.sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'zephyr'}))
            print('Running on Zephyr Topology.')
        elif self.topology == "Pegasus":
            self.sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
            print('Running on Pegasus Topology.')
        

    def define_chainstrength(self):
        '''
            Calculate the chosen_chainstrength following the definition from the original article
            or using the definition suggested from the Ocean documentation.
        '''

        if self.chosen_chainstrength == 'Ocean_Doc':
            self.chain_strength = uniform_torque_compensation(self.H, self.sampler)
            print(f'You chose the chainstrength from the Ocean Documentation: {self.chain_strength}')
        elif self.chosen_chainstrength == 'Article':
            self.chain_strength = 0
            for i in self.Q.values():
                self.chain_strength += abs(i)
            self.chain_strength *= 3/len(self.Q)
            print(f'You chose the Article chainstrength: {self.chain_strength}.')
        else:
            print('You can only choose Ocean_Doc or Article.')
    
    def decoding_response(self, response):
        '''
            Decode the SampleSet obtained from a D-Wave run and
            turn it into a list of values.

            Inputs:
                response: SampleSet object containing the data

            Outputs:
                solution (list(int)): best sample in list format
        '''

        best_sample = response.first.sample
        solution = []

        for i in range(self.bits):
            solution.append(best_sample[i])
        print(f"Solution found: {solution} with energy: {response.first.energy}")

        return(solution)
    
    def get_info_on_embedding(self, used_embedding):

        '''
            Get info on embedding of a single run on a QPU.

            Inputs:
                used_embedding (dict): mapping of logical qubits onto physical ones.

            Outputs:
                physical_qubits (int): number of physical qubits used in the embedding.
                sorted_chains (dict): logical qubits paired with their chainlength. Sorted by key.
        '''
        q_count = 0
        chains = {}

        for logical_q in used_embedding.keys():
            q_count += len(used_embedding[logical_q])
            chains[f'lq_{logical_q}'] = len(used_embedding[logical_q])
        
        sorted_chains = dict(sorted(chains.items()))

        return(q_count, sorted_chains)
    
    def single_run(self):
        '''
            Function that runs *once* the sampling for the given Hamiltonian
            on D-Wave devices.

            Outputs:
                response: SampleSet object with the details of the run. 
        '''

        self.define_chainstrength()      
        response = self.sampler.sample(self.H, chain_strength=self.chain_strength, 
                              num_reads=self.numruns, annealing_time=self.T, 
                              answer_mode='histogram',
                              label = f'mq_on_{self.topology}')
        print('Finished running the experiment!')

        return(response)

    def iterative(self, iterations=5 ,treshold=10):

        '''
            Run on QPU multiple times, each iteration fixing 
            ancillae that share the same value.

            Inputs: 
                iterations (int): number of single runs (default=5).
                treshold (int): number of samples to comparate to fix the ancillae (default=10).

            Outputs:
                solution (list): list of final found values for the variables.
                timing_info (dict): dictionary with total timings.
                physical_qubits (float): number of the physical qubits used.
        '''

        ## general setup     
        fixed = {}          # variable : value

        ###### iterating
        for it in range(iterations):

            ## info per run
            num_variables = self.H.num_variables
            print(f'Number of variables: {num_variables}')
                    
            ## running on QPU
            response = self.single_run()
            record = response.record
            order = np.argsort(record['energy'])

            best_sample, best_energy = response.first.sample, response.first.energy
            print(f'Energy of best sample at iteration {it}: {best_energy}')

            if best_energy == 0:
                print(f'Solution found with final iteration {it}.')
                break

            ## info
            if it == 0:                 # first iteration
                #timings
                timing_info = response.info['timing']
                #embedding
                embedding_info = response.info['embedding_context']['embedding']
                physical_qubits, chains = self.get_info_on_embedding(embedding_info)
            else:                        # later iterations
                # timings
                for key in timing_info.keys():
                    timing_info[key] += response.info['timing'][key]
                # embedding
                ph_info = response.info['embedding_context']['embedding']
                ph_qubits, ph_chains = self.get_info_on_embedding(ph_info)
                physical_qubits += ph_qubits

            ## fixing variables
            print("Fixing ancillae...")
            for i in range(self.bits, num_variables):
                flag = True
                ph_value = record.sample[order[0]][i]

                # checking for shared values
                for k in range(min(treshold, len(record.sample))):
                    if ph_value != record.sample[order[k]][i]:
                        flag = False
                        break
                if flag:
                    fixed[i] = ph_value
            
            print(f'Fixed {len(fixed.keys())} qubits:')
            for key in fixed.keys():
                print(f'{key} : {fixed[key]}')
            print('\n------------------------------------------------------------------------\n')

            ## modifying Hamiltonian
            for var in fixed.keys():
                if var not in self.H.variables:        # checking if it's a new fix or not
                    continue
                else:
                    self.H.fix_variable(var, fixed[var])

            ## creating new QUBO model
            self.Q, self.offset = self.H.to_qubo()

        ###### reconstruncting
        print('Reconstructing the final state...')
        solution = []
        for i in range(self.bits):
            solution.append(best_sample[i])

        print(f'Solution found: {solution}')

        return(solution, timing_info, physical_qubits, it)

def decoding_response_xi(bits:int, response):
    
    '''
        Decode the SampleSet obtained from a D-Wave run and
        turn it into a list of values.

        Inputs:
            bits (int): number of variables
            response: SampleSet object containing the data

        Outputs:
            solution (list(int)): best sample in list format
    '''

    best_sample = response.first.sample

    string_keys = [f'x{i}' for i in range(1,bits+1)]    # 1-9
    solution = []

    for key in best_sample:
        if key in string_keys:
            #print(f"{bs[key]} ", end='')
            solution.append(best_sample[key])
    print(f"Solution found:\n{solution}\nwith energy: {response.first.energy}")

    return(solution)
