'''
Examples file taken from the repository
    https://github.com/qiboteam/mq-problem-quantum-annealing
by Sergi Ramos-Calderer and Ruge Lin for the article 
    "Solving systems of Boolean multivariate equations with quantum annealing" 
(https://arxiv.org/abs/2111.13224) by Sergi Ramos-Calderer, Carlos Bravo-Prieto, 
Ruge Lin, Emanuele Bellini, Marc Manzano, Najwa Aaraj, and José I. Latorre.

All the code in this file is credited to the authors.

A few changes in the notation have been made to run smoothly with my code.
'''

def example_5(x1, x2, x3, x4, x5):
    p = [
        4*x1*x2*x3*x4*x5 - 2*x1*x2*x3*x4 - 4*x1*x2*x3*x5 + 
            2*x1*x2*x3 + 2*x1*x2*x5 - x1*x2 - 2*x1*x3*x4*x5 + 
            2*x1*x3*x4 + 2*x1*x3*x5 - x1*x3 - x1*x5 - 2*x2*x3*x4*x5 + 
            2*x2*x3*x5 - x2*x3 + x2*x4 - x2*x5 + x4*x5 - x4 + 1,
        -2*x1*x2*x3*x4 + 2*x1*x2*x3*x5 + 2*x1*x2*x4*x5 - 2*x1*x2*x5 +
            2*x1*x3*x4 - x1*x3 - x1*x4 + x1 - 2*x2*x3*x5 + x2*x3 + 
            x2*x5 - x3*x4 + x3*x5 - x4*x5 + x4,
        -4*x1*x2*x3*x4*x5 + 2*x1*x2*x3*x4 + 2*x1*x2*x3*x5 - 
            2*x1*x2*x3 + 2*x1*x2*x4*x5 - 2*x1*x2*x5 + x1*x2 + 
            2*x1*x3*x4*x5 - 2*x1*x3*x5 + x1*x3 - x1*x4 + x1*x5 + 
            x2*x3 - x2*x4 - x3*x4 + x4,
        4*x1*x2*x3*x4*x5 - 2*x1*x2*x3*x5 - 2*x1*x2*x4*x5 - 
            2*x1*x3*x4 + x1*x3 + x1*x4 - 2*x2*x3*x4*x5 + x2*x3 + 
            x2*x4 + x2*x5 - x2 + x3*x4 + x3*x5 - x3 + x4*x5 - x4 - x5 
            + 1,
        4*x1*x2*x3*x4*x5 - 4*x1*x2*x3*x4 - 4*x1*x2*x3*x5 + 
            4*x1*x2*x3 - 2*x1*x2*x4*x5 + 2*x1*x2*x4 + 2*x1*x2*x5 - 
            2*x1*x2 - 2*x1*x3*x4*x5 + 2*x1*x3*x4 + 2*x1*x3*x5 - 
            2*x1*x3 + 2*x1*x4*x5 - x1*x4 - x1*x5 + x1 - 2*x2*x3*x4*x5 
            + 2*x2*x3*x4 + 2*x2*x3*x5 - 2*x2*x3 + 2*x2*x4*x5 - x2*x4 - 
            2*x2*x5 + x2 - x3*x5 + x3 - x4*x5 + x5
            ]
    sol = [[ 1, 1, 0, 0, 1 ]]
    return p, sol


def example_9(x1, x2, x3, x4, x5, x6, x7, x8, x9):
    p = [
        -16*x1*x2*x3*x4*x5*x6*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x6*x7*x8 + 8*x1*x2*x3*x4*x5*x6*x7*x9 - 
        4*x1*x2*x3*x4*x5*x6 + 8*x1*x2*x3*x4*x5*x7*x8*x9 - 4*x1*x2*x3*x4*x5*x7 - 
        4*x1*x2*x3*x4*x5*x8 + 4*x1*x2*x3*x4*x5 + 16*x1*x2*x3*x4*x6*x7*x8*x9 - 
        8*x1*x2*x3*x4*x6*x7*x8 - 8*x1*x2*x3*x4*x6*x7*x9 - 8*x1*x2*x3*x4*x6*x8*x9 + 
        4*x1*x2*x3*x4*x6*x8 + 4*x1*x2*x3*x4*x6*x9 - 8*x1*x2*x3*x4*x7*x8*x9 + 4*x1*x2*x3*x4*x7*x8
        + 4*x1*x2*x3*x4*x7*x9 + 4*x1*x2*x3*x4*x8*x9 - 4*x1*x2*x3*x4*x9 + 
        16*x1*x2*x3*x5*x6*x7*x8*x9 - 8*x1*x2*x3*x5*x6*x7*x8 - 8*x1*x2*x3*x5*x6*x7*x9 + 
        4*x1*x2*x3*x5*x6 - 8*x1*x2*x3*x5*x7*x8*x9 + 4*x1*x2*x3*x5*x7*x8 + 4*x1*x2*x3*x5*x7*x9 - 
        2*x1*x2*x3*x5 - 16*x1*x2*x3*x6*x7*x8*x9 + 8*x1*x2*x3*x6*x7*x8 + 8*x1*x2*x3*x6*x7*x9 + 
        8*x1*x2*x3*x6*x8*x9 - 4*x1*x2*x3*x6*x8 - 4*x1*x2*x3*x6*x9 + 8*x1*x2*x3*x7*x8*x9 - 
        4*x1*x2*x3*x7*x8 - 4*x1*x2*x3*x7*x9 - 4*x1*x2*x3*x8*x9 + 2*x1*x2*x3*x8 + 2*x1*x2*x3*x9 + 
        16*x1*x2*x4*x5*x6*x7*x8*x9 - 8*x1*x2*x4*x5*x6*x7*x8 - 8*x1*x2*x4*x5*x6*x7*x9 + 
        4*x1*x2*x4*x5*x6 - 8*x1*x2*x4*x5*x7*x8*x9 + 4*x1*x2*x4*x5*x7 + 4*x1*x2*x4*x5*x8 - 
        4*x1*x2*x4*x5 - 16*x1*x2*x4*x6*x7*x8*x9 + 8*x1*x2*x4*x6*x7*x8 + 8*x1*x2*x4*x6*x7*x9 + 
        8*x1*x2*x4*x6*x8*x9 - 4*x1*x2*x4*x6*x8 - 4*x1*x2*x4*x6*x9 + 8*x1*x2*x4*x7*x8*x9 - 
        4*x1*x2*x4*x7*x8 - 4*x1*x2*x4*x7*x9 - 4*x1*x2*x4*x8*x9 + 4*x1*x2*x4*x9 - 
        16*x1*x2*x5*x6*x7*x8*x9 + 8*x1*x2*x5*x6*x7*x8 + 8*x1*x2*x5*x6*x7*x9 - 4*x1*x2*x5*x6 + 
        8*x1*x2*x5*x7*x8*x9 - 4*x1*x2*x5*x7*x8 - 4*x1*x2*x5*x7*x9 + 2*x1*x2*x5 + 
        16*x1*x2*x6*x7*x8*x9 - 8*x1*x2*x6*x7*x8 - 8*x1*x2*x6*x7*x9 - 8*x1*x2*x6*x8*x9 + 
        4*x1*x2*x6*x8 + 4*x1*x2*x6*x9 - 8*x1*x2*x7*x8*x9 + 4*x1*x2*x7*x8 + 4*x1*x2*x7*x9 + 
        4*x1*x2*x8*x9 - 2*x1*x2*x8 - 2*x1*x2*x9 + 8*x1*x3*x4*x5*x6*x7*x8*x9 - 
        8*x1*x3*x4*x5*x6*x7*x8 - 8*x1*x3*x4*x5*x6*x7*x9 + 4*x1*x3*x4*x5*x6*x7 + 
        4*x1*x3*x4*x5*x6*x9 - 4*x1*x3*x4*x5*x8*x9 + 4*x1*x3*x4*x5*x8 - 2*x1*x3*x4*x5 - 
        8*x1*x3*x4*x6*x7*x8*x9 + 4*x1*x3*x4*x6*x7*x8 + 4*x1*x3*x4*x6*x7*x9 + 4*x1*x3*x4*x6*x8*x9
        - 4*x1*x3*x4*x6*x9 - 2*x1*x3*x4*x8 + 2*x1*x3*x4*x9 - 8*x1*x3*x5*x6*x7*x8*x9 + 
        4*x1*x3*x5*x6*x7*x8 + 4*x1*x3*x5*x6*x7*x9 - 2*x1*x3*x5*x6 + 4*x1*x3*x5*x8*x9 - 
        2*x1*x3*x5*x8 - 2*x1*x3*x5*x9 + 2*x1*x3*x5 + 8*x1*x3*x6*x7*x8*x9 - 4*x1*x3*x6*x7*x8 - 
        4*x1*x3*x6*x7*x9 - 4*x1*x3*x6*x8*x9 + 2*x1*x3*x6*x8 + 2*x1*x3*x6*x9 - 
        8*x1*x4*x5*x6*x7*x8*x9 + 4*x1*x4*x5*x6*x7*x8 + 4*x1*x4*x5*x6*x7*x9 - 2*x1*x4*x5*x6 + 
        4*x1*x4*x5*x7*x8*x9 - 2*x1*x4*x5*x7 - 2*x1*x4*x5*x8 + 2*x1*x4*x5 + 8*x1*x4*x6*x7*x8*x9 - 
        4*x1*x4*x6*x7*x8 - 4*x1*x4*x6*x7*x9 - 4*x1*x4*x6*x8*x9 + 2*x1*x4*x6*x8 + 2*x1*x4*x6*x9 - 
        4*x1*x4*x7*x8*x9 + 2*x1*x4*x7*x8 + 2*x1*x4*x7*x9 + 2*x1*x4*x8*x9 - 2*x1*x4*x9 + 
        8*x1*x5*x6*x7*x8*x9 - 4*x1*x5*x6*x7*x8 - 4*x1*x5*x6*x7*x9 + 2*x1*x5*x6 - 4*x1*x5*x7*x8*x9
        + 2*x1*x5*x7*x8 + 2*x1*x5*x7*x9 - x1*x5 - 8*x1*x6*x7*x8*x9 + 4*x1*x6*x7*x8 + 4*x1*x6*x7*x9
        + 4*x1*x6*x8*x9 - 2*x1*x6*x8 - 2*x1*x6*x9 + 4*x1*x7*x8*x9 - 2*x1*x7*x8 - 2*x1*x7*x9 - 
        2*x1*x8*x9 + x1*x8 + x1*x9 - 4*x2*x3*x4*x5*x6*x7 + 4*x2*x3*x4*x5*x6 - 4*x2*x3*x4*x5*x7*x8
        + 4*x2*x3*x4*x5*x7 + 4*x2*x3*x4*x5*x8 - 4*x2*x3*x4*x5 + 4*x2*x3*x4*x6*x7 - 2*x2*x3*x4*x6 -
        2*x2*x3*x4*x7 - 2*x2*x3*x4*x8 + 2*x2*x3*x4 - 8*x2*x3*x5*x6*x7*x8*x9 + 4*x2*x3*x5*x6*x7*x8
        + 4*x2*x3*x5*x6*x7*x9 - 2*x2*x3*x5*x6 + 4*x2*x3*x5*x7*x8*x9 - 4*x2*x3*x5*x7*x9 - 
        2*x2*x3*x5*x8 + 2*x2*x3*x5 + 8*x2*x3*x6*x7*x8*x9 - 4*x2*x3*x6*x7*x8 - 4*x2*x3*x6*x7*x9 - 
        4*x2*x3*x6*x8*x9 + 2*x2*x3*x6*x8 + 2*x2*x3*x6*x9 - 4*x2*x3*x7*x8*x9 + 2*x2*x3*x7*x8 + 
        2*x2*x3*x7*x9 + 2*x2*x3*x8*x9 - x2*x3 + 4*x2*x4*x5*x6*x7 - 4*x2*x4*x5*x6 + 
        4*x2*x4*x5*x7*x8 - 4*x2*x4*x5*x7 - 4*x2*x4*x5*x8 + 4*x2*x4*x5 - 4*x2*x4*x6*x7 + 2*x2*x4*x6
        + 2*x2*x4*x7 + 2*x2*x4*x8 - 2*x2*x4 + 8*x2*x5*x6*x7*x8*x9 - 4*x2*x5*x6*x7*x8 - 
        4*x2*x5*x6*x7*x9 + 2*x2*x5*x6 - 4*x2*x5*x7*x8*x9 + 4*x2*x5*x7*x9 + 2*x2*x5*x8 - 2*x2*x5 - 
        8*x2*x6*x7*x8*x9 + 4*x2*x6*x7*x8 + 4*x2*x6*x7*x9 + 4*x2*x6*x8*x9 - 2*x2*x6*x8 - 2*x2*x6*x9
        + 4*x2*x7*x8*x9 - 2*x2*x7*x8 - 2*x2*x7*x9 - 2*x2*x8*x9 + x2 + 4*x3*x4*x5*x6*x7*x9 - 
        4*x3*x4*x5*x6*x9 - 4*x3*x4*x5*x7*x8*x9 + 4*x3*x4*x5*x7*x8 - 2*x3*x4*x5*x7 + 
        4*x3*x4*x5*x8*x9 - 4*x3*x4*x5*x8 + 2*x3*x4*x5 - 2*x3*x4*x6*x7 + 2*x3*x4*x6*x9 + 
        4*x3*x4*x7*x8*x9 - 2*x3*x4*x7*x8 - 2*x3*x4*x7*x9 + 2*x3*x4*x7 - 2*x3*x4*x8*x9 + 2*x3*x4*x8
        - x3*x4 + 4*x3*x5*x6*x7*x8*x9 - 4*x3*x5*x6*x7*x9 + 2*x3*x5*x6*x9 - 2*x3*x5*x7*x8 + 
        2*x3*x5*x7*x9 - 2*x3*x5*x8*x9 + 2*x3*x5*x8 - x3*x5 - 4*x3*x6*x7*x8*x9 + 2*x3*x6*x7*x8 + 
        2*x3*x6*x7*x9 + 2*x3*x6*x8*x9 - 2*x3*x6*x8 - 2*x3*x6*x9 + x3*x6 - 2*x4*x5*x6*x7 + 
        2*x4*x5*x6 - 2*x4*x5*x7*x8 + 2*x4*x5*x7 + 2*x4*x5*x8 - 2*x4*x5 + 2*x4*x6*x7 - x4*x6 - x4*x7
        - x4*x8 + x4 - 4*x5*x6*x7*x8*x9 + 2*x5*x6*x7*x8 + 2*x5*x6*x7*x9 - x5*x6 + 2*x5*x7*x8*x9 - 
        2*x5*x7*x9 - x5*x8 + x5 + 4*x6*x7*x8*x9 - 2*x6*x7*x8 - 2*x6*x7*x9 - 2*x6*x8*x9 + x6*x8 + 
        x6*x9 - 2*x7*x8*x9 + x7*x8 + x7*x9 + x8*x9,
        16*x1*x2*x3*x4*x5*x6*x7*x8*x9 - 16*x1*x2*x3*x4*x5*x6*x7*x8 - 8*x1*x2*x3*x4*x5*x6*x7*x9 + 
        8*x1*x2*x3*x4*x5*x6*x7 - 8*x1*x2*x3*x4*x5*x6*x8*x9 + 8*x1*x2*x3*x4*x5*x6*x8 - 
        16*x1*x2*x3*x4*x5*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x7*x8 + 8*x1*x2*x3*x4*x5*x7*x9 - 
        4*x1*x2*x3*x4*x5*x7 + 8*x1*x2*x3*x4*x5*x8*x9 - 4*x1*x2*x3*x4*x5*x8 - 
        16*x1*x2*x3*x4*x6*x7*x8*x9 + 8*x1*x2*x3*x4*x6*x7*x8 + 8*x1*x2*x3*x4*x6*x7*x9 - 
        4*x1*x2*x3*x4*x6*x7 + 8*x1*x2*x3*x4*x6*x8*x9 - 4*x1*x2*x3*x4*x6*x8 + 
        16*x1*x2*x3*x4*x7*x8*x9 - 8*x1*x2*x3*x4*x7*x8 - 8*x1*x2*x3*x4*x7*x9 + 4*x1*x2*x3*x4*x7 - 
        8*x1*x2*x3*x4*x8*x9 + 4*x1*x2*x3*x4*x8 - 8*x1*x2*x3*x5*x6*x7*x8*x9 + 
        8*x1*x2*x3*x5*x6*x7*x8 - 4*x1*x2*x3*x5*x6*x7 + 8*x1*x2*x3*x5*x6*x8*x9 - 
        4*x1*x2*x3*x5*x6*x8 + 8*x1*x2*x3*x5*x7*x8*x9 - 4*x1*x2*x3*x5*x7*x8 - 8*x1*x2*x3*x5*x8*x9
        + 4*x1*x2*x3*x5*x8 + 8*x1*x2*x3*x6*x7*x8*x9 - 4*x1*x2*x3*x6*x7*x8 - 4*x1*x2*x3*x6*x7*x9 +
        4*x1*x2*x3*x6*x7 - 4*x1*x2*x3*x6*x8*x9 - 8*x1*x2*x3*x7*x8*x9 + 4*x1*x2*x3*x7*x8 + 
        4*x1*x2*x3*x7*x9 - 2*x1*x2*x3*x7 + 4*x1*x2*x3*x8*x9 - 2*x1*x2*x3*x8 - 
        8*x1*x2*x4*x5*x6*x7*x8*x9 + 8*x1*x2*x4*x5*x6*x7*x8 + 4*x1*x2*x4*x5*x6*x9 - 
        4*x1*x2*x4*x5*x6 + 8*x1*x2*x4*x5*x7*x8*x9 - 4*x1*x2*x4*x5*x7*x8 - 4*x1*x2*x4*x5*x7*x9 - 
        4*x1*x2*x4*x5*x8*x9 + 2*x1*x2*x4*x5 + 8*x1*x2*x4*x6*x7*x8*x9 - 4*x1*x2*x4*x6*x7*x8 - 
        4*x1*x2*x4*x6*x7*x9 - 4*x1*x2*x4*x6*x8*x9 + 2*x1*x2*x4*x6 - 8*x1*x2*x4*x7*x8*x9 + 
        4*x1*x2*x4*x7*x8 + 4*x1*x2*x4*x7*x9 + 4*x1*x2*x4*x8*x9 - 2*x1*x2*x4 + 
        8*x1*x2*x5*x6*x7*x8*x9 - 4*x1*x2*x5*x6*x7*x8 - 4*x1*x2*x5*x6*x8*x9 + 2*x1*x2*x5*x6 - 
        4*x1*x2*x5*x7*x8*x9 + 2*x1*x2*x5*x7 + 4*x1*x2*x5*x8*x9 - 2*x1*x2*x5 - 8*x1*x2*x6*x7*x8*x9
        + 4*x1*x2*x6*x7*x8 + 4*x1*x2*x6*x7*x9 - 2*x1*x2*x6*x7 + 4*x1*x2*x6*x8*x9 - 2*x1*x2*x6*x9 +
        4*x1*x2*x7*x8*x9 - 2*x1*x2*x7*x8 - 2*x1*x2*x7*x9 - 2*x1*x2*x8*x9 + x1*x2 + 
        8*x1*x3*x4*x5*x6*x7*x8 - 4*x1*x3*x4*x5*x6*x7 - 4*x1*x3*x4*x5*x6*x8 + 
        8*x1*x3*x4*x6*x7*x8*x9 - 8*x1*x3*x4*x6*x7*x8 - 4*x1*x3*x4*x6*x7*x9 + 4*x1*x3*x4*x6*x7 - 
        4*x1*x3*x4*x6*x8*x9 + 4*x1*x3*x4*x6*x8 - 8*x1*x3*x4*x7*x8*x9 + 4*x1*x3*x4*x7*x8 + 
        4*x1*x3*x4*x7*x9 - 2*x1*x3*x4*x7 + 4*x1*x3*x4*x8*x9 - 2*x1*x3*x4*x8 - 4*x1*x3*x5*x6*x7*x8
        + 4*x1*x3*x5*x6*x7*x9 - 4*x1*x3*x5*x6*x8*x9 + 4*x1*x3*x5*x6*x8 - 4*x1*x3*x5*x7*x9 + 
        2*x1*x3*x5*x7 + 4*x1*x3*x5*x8*x9 - 2*x1*x3*x5*x8 - 4*x1*x3*x6*x7*x8*x9 + 4*x1*x3*x6*x7*x8
        - 2*x1*x3*x6*x7 + 4*x1*x3*x6*x8*x9 - 2*x1*x3*x6*x8 + 4*x1*x3*x7*x8*x9 - 2*x1*x3*x7*x8 - 
        4*x1*x3*x8*x9 + 2*x1*x3*x8 - 4*x1*x4*x5*x6*x7*x8 + 2*x1*x4*x5*x6 + 2*x1*x4*x5*x7 + 
        2*x1*x4*x5*x8 - 2*x1*x4*x5 - 4*x1*x4*x6*x7*x8*x9 + 4*x1*x4*x6*x7*x8 + 4*x1*x4*x6*x7*x9 - 
        2*x1*x4*x6*x7 + 4*x1*x4*x6*x8*x9 - 2*x1*x4*x6*x8 - 2*x1*x4*x6*x9 + 4*x1*x4*x7*x8*x9 - 
        2*x1*x4*x7*x8 - 2*x1*x4*x7*x9 - 2*x1*x4*x8*x9 + x1*x4 - 4*x1*x5*x6*x7*x8*x9 + 
        4*x1*x5*x6*x7*x8 + 4*x1*x5*x6*x8*x9 - 2*x1*x5*x6*x8 - 2*x1*x5*x6*x9 + 2*x1*x5*x7*x9 - 
        2*x1*x5*x7 - 2*x1*x5*x8*x9 + x1*x5 + 4*x1*x6*x7*x8*x9 - 4*x1*x6*x7*x8 - 2*x1*x6*x7*x9 + 
        2*x1*x6*x7 - 4*x1*x6*x8*x9 + 2*x1*x6*x8 + 2*x1*x6*x9 - x1*x6 - 2*x1*x7*x8*x9 + 2*x1*x7*x8 +
        2*x1*x8*x9 - x1*x8 - 8*x2*x3*x4*x5*x6*x7*x8*x9 + 8*x2*x3*x4*x5*x6*x7*x8 + 
        8*x2*x3*x4*x5*x6*x7*x9 - 4*x2*x3*x4*x5*x6*x7 + 8*x2*x3*x4*x5*x6*x8*x9 - 
        4*x2*x3*x4*x5*x6*x8 - 4*x2*x3*x4*x5*x6*x9 + 8*x2*x3*x4*x5*x7*x8*x9 - 4*x2*x3*x4*x5*x7*x8
        - 8*x2*x3*x4*x5*x7*x9 + 4*x2*x3*x4*x5*x7 - 8*x2*x3*x4*x5*x8*x9 + 4*x2*x3*x4*x5*x8 + 
        4*x2*x3*x4*x5*x9 - 2*x2*x3*x4*x5 + 8*x2*x3*x4*x6*x7*x8*x9 - 4*x2*x3*x4*x6*x7*x8 - 
        8*x2*x3*x4*x6*x7*x9 + 4*x2*x3*x4*x6*x7 - 8*x2*x3*x4*x6*x8*x9 + 4*x2*x3*x4*x6*x8 + 
        4*x2*x3*x4*x6*x9 - 2*x2*x3*x4*x6 - 8*x2*x3*x4*x7*x8*x9 + 4*x2*x3*x4*x7*x8 + 
        8*x2*x3*x4*x7*x9 - 4*x2*x3*x4*x7 + 8*x2*x3*x4*x8*x9 - 4*x2*x3*x4*x8 - 4*x2*x3*x4*x9 + 
        2*x2*x3*x4 - 4*x2*x3*x5*x6*x8*x9 + 2*x2*x3*x5*x6 + 4*x2*x3*x5*x8*x9 - 2*x2*x3*x5*x8 - 
        4*x2*x3*x6*x7*x8*x9 + 4*x2*x3*x6*x7*x9 - 2*x2*x3*x6*x7 + 4*x2*x3*x6*x8*x9 - 2*x2*x3*x6*x9
        + 4*x2*x3*x7*x8*x9 - 2*x2*x3*x7*x8 - 4*x2*x3*x7*x9 + 2*x2*x3*x7 - 4*x2*x3*x8*x9 + 
        2*x2*x3*x8 + 2*x2*x3*x9 - x2*x3 - 4*x2*x4*x5*x6*x7*x8 + 2*x2*x4*x5*x6 - 
        4*x2*x4*x5*x7*x8*x9 + 4*x2*x4*x5*x7*x8 + 4*x2*x4*x5*x7*x9 - 2*x2*x4*x5*x7 + 
        4*x2*x4*x5*x8*x9 - 2*x2*x4*x5*x8 - 2*x2*x4*x5*x9 - 4*x2*x4*x6*x7*x8*x9 + 4*x2*x4*x6*x7*x8
        + 4*x2*x4*x6*x7*x9 - 2*x2*x4*x6*x7 + 4*x2*x4*x6*x8*x9 - 2*x2*x4*x6*x8 - 2*x2*x4*x6*x9 + 
        4*x2*x4*x7*x8*x9 - 4*x2*x4*x7*x8 - 4*x2*x4*x7*x9 + 2*x2*x4*x7 - 4*x2*x4*x8*x9 + 2*x2*x4*x8
        + 2*x2*x4*x9 + 2*x2*x5*x6*x8 - 2*x2*x5*x6 - 2*x2*x5*x8*x9 + x2*x5 + 4*x2*x6*x7*x8*x9 - 
        2*x2*x6*x7*x8 - 4*x2*x6*x7*x9 + 2*x2*x6*x7 - 2*x2*x6*x8*x9 + 2*x2*x6*x9 - 2*x2*x7*x8*x9 + 
        2*x2*x7*x8 + 2*x2*x7*x9 - x2*x7 + 2*x2*x8*x9 - x2*x8 - x2*x9 - 4*x3*x4*x5*x6*x7*x8 - 
        4*x3*x4*x5*x6*x7*x9 + 4*x3*x4*x5*x6*x7 - 4*x3*x4*x5*x6*x8*x9 + 4*x3*x4*x5*x6*x8 + 
        4*x3*x4*x5*x6*x9 - 2*x3*x4*x5*x6 + 4*x3*x4*x5*x7*x9 - 2*x3*x4*x5*x7 + 4*x3*x4*x5*x8*x9 - 
        2*x3*x4*x5*x8 - 4*x3*x4*x5*x9 + 2*x3*x4*x5 - 4*x3*x4*x6*x7*x8*x9 + 4*x3*x4*x6*x7*x8 + 
        4*x3*x4*x6*x7*x9 - 4*x3*x4*x6*x7 + 4*x3*x4*x6*x8*x9 - 4*x3*x4*x6*x8 - 2*x3*x4*x6*x9 + 
        2*x3*x4*x6 + 4*x3*x4*x7*x8*x9 - 2*x3*x4*x7*x8 - 4*x3*x4*x7*x9 + 2*x3*x4*x7 - 4*x3*x4*x8*x9
        + 2*x3*x4*x8 + 2*x3*x4*x9 - x3*x4 + 4*x3*x5*x6*x8*x9 - 2*x3*x5*x6*x8 - 2*x3*x5*x6*x9 - 
        4*x3*x5*x8*x9 + 2*x3*x5*x8 + 2*x3*x5*x9 - x3*x5 + 4*x3*x6*x7*x8*x9 - 2*x3*x6*x7*x8 - 
        2*x3*x6*x7*x9 + 2*x3*x6*x7 - 4*x3*x6*x8*x9 + 2*x3*x6*x8 + 2*x3*x6*x9 - x3*x6 - 
        4*x3*x7*x8*x9 + 2*x3*x7*x8 + 2*x3*x7*x9 - x3*x7 + 4*x3*x8*x9 - 2*x3*x8 - 2*x3*x9 + x3 + 
        4*x4*x5*x6*x7*x8*x9 - 2*x4*x5*x6*x9 - 2*x4*x5*x7*x9 - 2*x4*x5*x8*x9 + 2*x4*x5*x9 - 
        2*x4*x6*x7*x8 - 2*x4*x6*x7*x9 + 2*x4*x6*x7 - 2*x4*x6*x8*x9 + 2*x4*x6*x8 + 2*x4*x6*x9 - 
        x4*x6 - 2*x4*x7*x8*x9 + 2*x4*x7*x8 + 2*x4*x7*x9 - x4*x7 + 2*x4*x8*x9 - x4*x8 - x4*x9 - 
        2*x5*x6*x8*x9 + 2*x5*x6*x9 + 2*x5*x8*x9 - x5*x9 - 2*x6*x7*x8*x9 + 2*x6*x7*x8 + 2*x6*x7*x9 -
        2*x6*x7 + 2*x6*x8*x9 - x6*x8 - 2*x6*x9 + x6 + 2*x7*x8*x9 - 2*x7*x8 - x7*x9 + x7 - 2*x8*x9 + 
        x8 + x9,
        16*x1*x2*x3*x4*x5*x6*x7*x8*x9 - 8*x1*x2*x3*x4*x5*x6*x7*x8 - 16*x1*x2*x3*x4*x5*x6*x7*x9 + 
        8*x1*x2*x3*x4*x5*x6*x7 - 16*x1*x2*x3*x4*x5*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x7*x8 + 
        16*x1*x2*x3*x4*x5*x7*x9 - 8*x1*x2*x3*x4*x5*x7 + 8*x1*x2*x3*x4*x5*x8*x9 - 
        4*x1*x2*x3*x4*x5*x8 - 8*x1*x2*x3*x4*x5*x9 + 4*x1*x2*x3*x4*x5 - 8*x1*x2*x3*x4*x6*x7*x8*x9
        + 8*x1*x2*x3*x4*x6*x7*x9 + 8*x1*x2*x3*x4*x7*x8*x9 - 4*x1*x2*x3*x4*x7*x8 - 
        8*x1*x2*x3*x4*x7*x9 + 4*x1*x2*x3*x4*x7 - 4*x1*x2*x3*x4*x8*x9 + 4*x1*x2*x3*x4*x8 + 
        4*x1*x2*x3*x4*x9 - 4*x1*x2*x3*x4 - 16*x1*x2*x3*x5*x6*x7*x8*x9 + 8*x1*x2*x3*x5*x6*x7*x8 + 
        8*x1*x2*x3*x5*x6*x7*x9 - 4*x1*x2*x3*x5*x6*x7 + 8*x1*x2*x3*x5*x6*x8*x9 - 
        4*x1*x2*x3*x5*x6*x8 + 8*x1*x2*x3*x5*x7*x8*x9 - 4*x1*x2*x3*x5*x7*x8 - 8*x1*x2*x3*x5*x7*x9
        + 4*x1*x2*x3*x5*x7 - 8*x1*x2*x3*x5*x8*x9 + 4*x1*x2*x3*x5*x8 + 4*x1*x2*x3*x5*x9 - 
        2*x1*x2*x3*x5 + 8*x1*x2*x3*x6*x7*x8*x9 - 4*x1*x2*x3*x6*x7*x9 - 4*x1*x2*x3*x6*x8*x9 - 
        4*x1*x2*x3*x7*x8*x9 + 4*x1*x2*x3*x7*x9 - 2*x1*x2*x3*x7 + 4*x1*x2*x3*x8*x9 - 2*x1*x2*x3*x8
        - 2*x1*x2*x3*x9 + 2*x1*x2*x3 - 8*x1*x2*x4*x5*x6*x7*x8*x9 + 8*x1*x2*x4*x5*x6*x7*x8 + 
        8*x1*x2*x4*x5*x6*x7*x9 - 4*x1*x2*x4*x5*x6*x7 - 4*x1*x2*x4*x5*x6*x8 + 
        8*x1*x2*x4*x5*x7*x8*x9 - 4*x1*x2*x4*x5*x7*x8 - 8*x1*x2*x4*x5*x7*x9 + 4*x1*x2*x4*x5*x7 - 
        4*x1*x2*x4*x5*x8*x9 + 4*x1*x2*x4*x5*x8 + 4*x1*x2*x4*x5*x9 - 2*x1*x2*x4*x5 + 
        8*x1*x2*x4*x6*x7*x8*x9 - 4*x1*x2*x4*x6*x7*x8 - 4*x1*x2*x4*x6*x7*x9 - 4*x1*x2*x4*x6*x8*x9
        + 4*x1*x2*x4*x6*x8 - 8*x1*x2*x4*x7*x8*x9 + 4*x1*x2*x4*x7*x8 + 4*x1*x2*x4*x7*x9 - 
        2*x1*x2*x4*x7 + 4*x1*x2*x4*x8*x9 - 4*x1*x2*x4*x8 - 2*x1*x2*x4*x9 + 2*x1*x2*x4 + 
        8*x1*x2*x5*x6*x7*x8*x9 - 4*x1*x2*x5*x6*x7*x8 - 4*x1*x2*x5*x6*x7*x9 - 4*x1*x2*x5*x6*x8*x9
        + 4*x1*x2*x5*x6*x8 - 4*x1*x2*x5*x7*x8*x9 + 4*x1*x2*x5*x7*x9 + 4*x1*x2*x5*x8*x9 - 
        2*x1*x2*x5*x8 - 2*x1*x2*x5*x9 - 4*x1*x2*x6*x7*x8*x9 + 2*x1*x2*x6*x7 + 4*x1*x2*x6*x8*x9 - 
        2*x1*x2*x6*x8 + 4*x1*x2*x7*x8*x9 - 2*x1*x2*x7*x9 - 4*x1*x2*x8*x9 + 2*x1*x2*x8 + 2*x1*x2*x9
        - x1*x2 - 8*x1*x3*x4*x5*x6*x8*x9 + 4*x1*x3*x4*x5*x6*x8 + 8*x1*x3*x4*x5*x6*x9 - 
        4*x1*x3*x4*x5*x6 + 4*x1*x3*x4*x6*x8*x9 - 4*x1*x3*x4*x6*x9 - 2*x1*x3*x4*x8 + 2*x1*x3*x4 - 
        4*x1*x3*x5*x6*x9 + 2*x1*x3*x5*x6 + 4*x1*x3*x5*x8*x9 - 2*x1*x3*x5*x8 + 2*x1*x3*x6*x9 - 
        2*x1*x3*x8*x9 + 2*x1*x3*x8 - x1*x3 + 4*x1*x4*x5*x6*x8*x9 - 4*x1*x4*x5*x6*x9 + 
        2*x1*x4*x5*x6 - 2*x1*x4*x5*x8 - 2*x1*x4*x6*x8 + 2*x1*x4*x6*x9 + 2*x1*x4*x8 - x1*x4 - 
        2*x1*x5*x6*x8 + 2*x1*x5*x6*x9 - 2*x1*x5*x8*x9 + 2*x1*x5*x8 - 2*x1*x6*x8*x9 + 2*x1*x6*x8 - 
        x1*x6 + 2*x1*x8*x9 - 2*x1*x8 - x1*x9 + x1 - 16*x2*x3*x4*x5*x6*x7*x8*x9 + 
        8*x2*x3*x4*x5*x6*x7*x8 + 8*x2*x3*x4*x5*x6*x7*x9 - 4*x2*x3*x4*x5*x6*x7 + 
        8*x2*x3*x4*x5*x6*x8*x9 - 4*x2*x3*x4*x5*x6*x8 + 16*x2*x3*x4*x5*x7*x8*x9 - 
        8*x2*x3*x4*x5*x7*x8 - 8*x2*x3*x4*x5*x7*x9 + 4*x2*x3*x4*x5*x7 - 8*x2*x3*x4*x5*x8*x9 + 
        4*x2*x3*x4*x5*x8 + 4*x2*x3*x4*x5*x9 - 2*x2*x3*x4*x5 + 8*x2*x3*x4*x6*x7*x8*x9 - 
        4*x2*x3*x4*x6*x7*x8 - 4*x2*x3*x4*x6*x7*x9 - 4*x2*x3*x4*x6*x8*x9 + 4*x2*x3*x4*x6*x8 - 
        8*x2*x3*x4*x7*x8*x9 + 4*x2*x3*x4*x7*x8 + 4*x2*x3*x4*x7*x9 - 2*x2*x3*x4*x7 + 
        4*x2*x3*x4*x8*x9 - 4*x2*x3*x4*x8 - 2*x2*x3*x4*x9 + 2*x2*x3*x4 + 8*x2*x3*x5*x6*x7*x8*x9 - 
        4*x2*x3*x5*x6*x7*x8 - 8*x2*x3*x5*x6*x8*x9 + 4*x2*x3*x5*x6*x8 - 8*x2*x3*x5*x7*x8*x9 + 
        4*x2*x3*x5*x7*x8 + 4*x2*x3*x5*x7*x9 - 2*x2*x3*x5*x7 + 8*x2*x3*x5*x8*x9 - 4*x2*x3*x5*x8 - 
        4*x2*x3*x5*x9 + 2*x2*x3*x5 - 4*x2*x3*x6*x7*x8*x9 + 2*x2*x3*x6*x7 + 4*x2*x3*x6*x8*x9 - 
        2*x2*x3*x6*x8 + 4*x2*x3*x7*x8*x9 - 2*x2*x3*x7*x9 - 4*x2*x3*x8*x9 + 2*x2*x3*x8 + 2*x2*x3*x9
        - x2*x3 + 8*x2*x4*x5*x6*x7*x8*x9 - 8*x2*x4*x5*x6*x7*x8 - 4*x2*x4*x5*x6*x7*x9 + 
        4*x2*x4*x5*x6*x7 - 4*x2*x4*x5*x6*x8*x9 + 4*x2*x4*x5*x6*x8 - 8*x2*x4*x5*x7*x8*x9 + 
        4*x2*x4*x5*x7*x8 + 4*x2*x4*x5*x7*x9 - 2*x2*x4*x5*x7 + 4*x2*x4*x5*x8*x9 - 2*x2*x4*x5*x8 - 
        2*x2*x4*x5*x9 - 4*x2*x4*x6*x7*x8*x9 + 4*x2*x4*x6*x7*x8 + 4*x2*x4*x6*x8*x9 - 4*x2*x4*x6*x8
        + 4*x2*x4*x7*x8*x9 - 2*x2*x4*x7*x8 - 2*x2*x4*x8*x9 + 2*x2*x4*x8 - 4*x2*x5*x6*x7*x8*x9 + 
        4*x2*x5*x6*x7*x8 + 4*x2*x5*x6*x8*x9 - 4*x2*x5*x6*x8 + 4*x2*x5*x7*x8*x9 - 2*x2*x5*x7*x8 - 
        2*x2*x5*x7*x9 - 4*x2*x5*x8*x9 + 2*x2*x5*x8 + 2*x2*x5*x9 + 2*x2*x6*x7*x9 - 2*x2*x6*x7 - 
        2*x2*x6*x8*x9 + 2*x2*x6*x8 - 2*x2*x7*x8*x9 + x2*x7 + 2*x2*x8*x9 - x2*x8 - x2*x9 - 
        4*x3*x4*x5*x6*x9 + 2*x3*x4*x5*x6 - 2*x3*x4*x6*x8 + 2*x3*x4*x6*x9 + 2*x3*x4*x8 - x3*x4 + 
        4*x3*x5*x6*x8*x9 - 2*x3*x5*x6*x8 - 4*x3*x5*x8*x9 + 2*x3*x5*x8 + 2*x3*x5*x9 - x3*x5 - 
        2*x3*x6*x8*x9 + 2*x3*x6*x8 - x3*x6 + 2*x3*x8*x9 - 2*x3*x8 - x3*x9 + x3 + 2*x4*x5*x6*x9 - 
        2*x4*x5*x6 + x4*x5 - 2*x4*x6*x8*x9 + 2*x4*x6*x8 - x4*x8 - 2*x5*x6*x8*x9 + 2*x5*x6*x8 + 
        2*x5*x8*x9 - x5*x8 - x5*x9 + 2*x6*x8*x9 - 2*x6*x8 - x6*x9 + x6 - x8*x9 + x8 + x9,
        -16*x1*x2*x3*x4*x5*x6*x7*x8*x9 + 16*x1*x2*x3*x4*x5*x6*x7*x8 + 16*x1*x2*x3*x4*x5*x6*x7*x9 - 
        8*x1*x2*x3*x4*x5*x6*x7 + 16*x1*x2*x3*x4*x5*x6*x8*x9 - 16*x1*x2*x3*x4*x5*x6*x8 - 
        16*x1*x2*x3*x4*x5*x6*x9 + 8*x1*x2*x3*x4*x5*x6 + 8*x1*x2*x3*x4*x5*x7*x8*x9 - 
        8*x1*x2*x3*x4*x5*x7*x8 - 8*x1*x2*x3*x4*x5*x7*x9 + 4*x1*x2*x3*x4*x5*x7 - 
        8*x1*x2*x3*x4*x5*x8*x9 + 8*x1*x2*x3*x4*x5*x8 + 8*x1*x2*x3*x4*x5*x9 - 4*x1*x2*x3*x4*x5 + 
        16*x1*x2*x3*x4*x6*x7*x8*x9 - 8*x1*x2*x3*x4*x6*x7*x8 - 8*x1*x2*x3*x4*x6*x7*x9 - 
        8*x1*x2*x3*x4*x6*x8*x9 + 8*x1*x2*x3*x4*x6*x8 + 8*x1*x2*x3*x4*x6*x9 - 4*x1*x2*x3*x4*x6 - 
        8*x1*x2*x3*x4*x7*x8*x9 + 4*x1*x2*x3*x4*x7*x8 + 4*x1*x2*x3*x4*x7*x9 + 4*x1*x2*x3*x4*x8*x9
        - 4*x1*x2*x3*x4*x8 - 4*x1*x2*x3*x4*x9 + 2*x1*x2*x3*x4 + 8*x1*x2*x3*x5*x6*x7*x8*x9 - 
        8*x1*x2*x3*x5*x6*x7*x8 - 8*x1*x2*x3*x5*x6*x7*x9 + 4*x1*x2*x3*x5*x6*x7 - 
        8*x1*x2*x3*x5*x6*x8*x9 + 8*x1*x2*x3*x5*x6*x8 + 8*x1*x2*x3*x5*x6*x9 - 4*x1*x2*x3*x5*x6 - 
        8*x1*x2*x3*x5*x7*x8*x9 + 4*x1*x2*x3*x5*x7*x8 + 8*x1*x2*x3*x5*x7*x9 - 4*x1*x2*x3*x5*x7 + 
        4*x1*x2*x3*x5*x8*x9 - 4*x1*x2*x3*x5*x8 - 4*x1*x2*x3*x5*x9 + 2*x1*x2*x3*x5 - 
        8*x1*x2*x3*x6*x7*x8*x9 + 4*x1*x2*x3*x6*x7*x8 + 4*x1*x2*x3*x6*x7*x9 + 4*x1*x2*x3*x6*x8*x9
        - 4*x1*x2*x3*x6*x8 - 4*x1*x2*x3*x6*x9 + 2*x1*x2*x3*x6 + 8*x1*x2*x3*x7*x8*x9 - 
        4*x1*x2*x3*x7*x8 - 4*x1*x2*x3*x7*x9 + 2*x1*x2*x3*x7 - 4*x1*x2*x3*x8*x9 + 4*x1*x2*x3*x8 + 
        2*x1*x2*x3*x9 - 2*x1*x2*x3 + 8*x1*x2*x4*x5*x6*x7*x8*x9 - 8*x1*x2*x4*x5*x6*x7*x8 - 
        8*x1*x2*x4*x5*x6*x7*x9 + 4*x1*x2*x4*x5*x6*x7 - 8*x1*x2*x4*x5*x6*x8*x9 + 
        8*x1*x2*x4*x5*x6*x8 + 8*x1*x2*x4*x5*x6*x9 - 4*x1*x2*x4*x5*x6 - 8*x1*x2*x4*x5*x7*x8*x9 + 
        4*x1*x2*x4*x5*x7*x8 + 8*x1*x2*x4*x5*x7*x9 - 4*x1*x2*x4*x5*x7 + 4*x1*x2*x4*x5*x8*x9 - 
        4*x1*x2*x4*x5*x8 - 4*x1*x2*x4*x5*x9 + 2*x1*x2*x4*x5 - 8*x1*x2*x4*x6*x7*x8*x9 + 
        4*x1*x2*x4*x6*x7*x8 + 4*x1*x2*x4*x6*x7*x9 + 4*x1*x2*x4*x6*x8*x9 - 4*x1*x2*x4*x6*x8 - 
        4*x1*x2*x4*x6*x9 + 2*x1*x2*x4*x6 + 8*x1*x2*x4*x7*x8*x9 - 4*x1*x2*x4*x7*x8 - 
        4*x1*x2*x4*x7*x9 + 2*x1*x2*x4*x7 - 4*x1*x2*x4*x8*x9 + 4*x1*x2*x4*x8 + 2*x1*x2*x4*x9 - 
        2*x1*x2*x4 + 4*x1*x2*x5*x7*x8*x9 - 4*x1*x2*x5*x7*x9 + 2*x1*x2*x5*x7 - 4*x1*x2*x7*x8*x9 + 
        2*x1*x2*x7*x8 + 2*x1*x2*x7*x9 - 2*x1*x2*x7 + 2*x1*x2*x8*x9 - 2*x1*x2*x8 + x1*x2 + 
        8*x1*x3*x4*x5*x6*x7*x8*x9 - 8*x1*x3*x4*x5*x6*x7*x8 - 8*x1*x3*x4*x5*x6*x7*x9 + 
        4*x1*x3*x4*x5*x6*x7 - 8*x1*x3*x4*x5*x6*x8*x9 + 8*x1*x3*x4*x5*x6*x8 + 8*x1*x3*x4*x5*x6*x9
        - 4*x1*x3*x4*x5*x6 + 4*x1*x3*x4*x5*x7*x9 - 4*x1*x3*x4*x5*x9 - 8*x1*x3*x4*x6*x7*x8*x9 + 
        4*x1*x3*x4*x6*x7*x8 + 4*x1*x3*x4*x6*x7*x9 + 4*x1*x3*x4*x6*x8*x9 - 4*x1*x3*x4*x6*x8 - 
        4*x1*x3*x4*x6*x9 + 2*x1*x3*x4*x6 - 2*x1*x3*x4*x7 + 2*x1*x3*x4*x9 + 4*x1*x3*x5*x6*x7*x8 + 
        4*x1*x3*x5*x6*x7*x9 - 4*x1*x3*x5*x6*x7 + 4*x1*x3*x5*x6*x8*x9 - 4*x1*x3*x5*x6*x8 - 
        4*x1*x3*x5*x6*x9 + 2*x1*x3*x5*x6 - 4*x1*x3*x5*x7*x9 + 2*x1*x3*x5*x7 + 2*x1*x3*x5*x9 + 
        4*x1*x4*x5*x6*x7*x8 + 4*x1*x4*x5*x6*x7*x9 - 4*x1*x4*x5*x6*x7 + 4*x1*x4*x5*x6*x8*x9 - 
        4*x1*x4*x5*x6*x8 - 4*x1*x4*x5*x6*x9 + 2*x1*x4*x5*x6 - 4*x1*x4*x5*x7*x9 + 2*x1*x4*x5*x7 + 
        2*x1*x4*x5*x9 - 4*x1*x5*x6*x7*x8*x9 + 2*x1*x5*x6*x7 + 2*x1*x5*x7*x9 - 2*x1*x5*x7 + 
        4*x1*x6*x7*x8*x9 - 2*x1*x6*x7*x8 - 2*x1*x6*x7*x9 - 2*x1*x6*x8*x9 + 2*x1*x6*x8 + 2*x1*x6*x9
        - x1*x6 + x1*x7 - x1*x9 + 8*x2*x3*x4*x5*x6*x7*x8*x9 - 8*x2*x3*x4*x5*x6*x7*x8 - 
        8*x2*x3*x4*x5*x6*x7*x9 + 4*x2*x3*x4*x5*x6*x7 - 8*x2*x3*x4*x5*x6*x8*x9 + 
        8*x2*x3*x4*x5*x6*x8 + 8*x2*x3*x4*x5*x6*x9 - 4*x2*x3*x4*x5*x6 + 4*x2*x3*x4*x5*x7*x8 + 
        4*x2*x3*x4*x5*x7*x9 - 4*x2*x3*x4*x5*x7 - 4*x2*x3*x4*x5*x8 - 4*x2*x3*x4*x5*x9 + 
        4*x2*x3*x4*x5 - 8*x2*x3*x4*x6*x7*x8*x9 + 4*x2*x3*x4*x6*x7*x8 + 4*x2*x3*x4*x6*x7*x9 + 
        4*x2*x3*x4*x6*x8*x9 - 4*x2*x3*x4*x6*x8 - 4*x2*x3*x4*x6*x9 + 2*x2*x3*x4*x6 + 
        4*x2*x3*x4*x7*x8*x9 - 4*x2*x3*x4*x7*x8 - 4*x2*x3*x4*x7*x9 + 2*x2*x3*x4*x7 + 2*x2*x3*x4*x8
        + 2*x2*x3*x4*x9 - 2*x2*x3*x4 + 4*x2*x3*x5*x6*x7*x9 + 4*x2*x3*x5*x6*x8*x9 - 
        4*x2*x3*x5*x6*x8 - 4*x2*x3*x5*x6*x9 + 2*x2*x3*x5*x6 - 4*x2*x3*x5*x7*x9 + 2*x2*x3*x5*x7 + 
        2*x2*x3*x5*x8 + 2*x2*x3*x5*x9 - 2*x2*x3*x5 + 4*x2*x3*x6*x7*x8*x9 - 4*x2*x3*x6*x7*x9 - 
        4*x2*x3*x6*x8*x9 + 2*x2*x3*x6*x8 + 4*x2*x3*x6*x9 - 2*x2*x3*x6 - 4*x2*x3*x7*x8*x9 + 
        2*x2*x3*x7*x8 + 4*x2*x3*x7*x9 - 2*x2*x3*x7 + 2*x2*x3*x8*x9 - 2*x2*x3*x8 - 2*x2*x3*x9 + 
        2*x2*x3 - 8*x2*x4*x5*x6*x7*x8*x9 + 8*x2*x4*x5*x6*x7*x8 + 4*x2*x4*x5*x6*x7*x9 - 
        4*x2*x4*x5*x6*x7 + 4*x2*x4*x5*x6*x8*x9 - 4*x2*x4*x5*x6*x8 - 4*x2*x4*x5*x6*x9 + 
        2*x2*x4*x5*x6 + 4*x2*x4*x5*x7*x8*x9 - 4*x2*x4*x5*x7*x8 - 4*x2*x4*x5*x7*x9 + 4*x2*x4*x5*x7
        + 2*x2*x4*x5*x8 + 2*x2*x4*x5*x9 - 2*x2*x4*x5 + 4*x2*x4*x6*x7*x8*x9 - 4*x2*x4*x6*x7*x8 + 
        2*x2*x4*x6*x8 - 4*x2*x4*x7*x8*x9 + 4*x2*x4*x7*x8 + 2*x2*x4*x7*x9 - 2*x2*x4*x7 - 2*x2*x4*x8
        + x2*x4 + 2*x2*x5*x7*x9 - 2*x2*x5*x7 - 2*x2*x5*x8*x9 + x2*x5 + 2*x2*x7*x8*x9 - 2*x2*x7*x8 -
        2*x2*x7*x9 + 2*x2*x7 + x2*x8 - x2 - 8*x3*x4*x5*x6*x7*x8*x9 + 4*x3*x4*x5*x6*x7*x8 + 
        8*x3*x4*x5*x6*x7*x9 - 4*x3*x4*x5*x6*x7 + 8*x3*x4*x5*x6*x8*x9 - 4*x3*x4*x5*x6*x8 - 
        8*x3*x4*x5*x6*x9 + 4*x3*x4*x5*x6 - 4*x3*x4*x5*x7*x9 + 2*x3*x4*x5*x7 + 4*x3*x4*x5*x9 - 
        2*x3*x4*x5 + 4*x3*x4*x6*x7*x8*x9 - 4*x3*x4*x6*x7*x9 - 4*x3*x4*x6*x8*x9 + 2*x3*x4*x6*x8 + 
        4*x3*x4*x6*x9 - 2*x3*x4*x6 + 2*x3*x4*x7*x9 - 2*x3*x4*x9 + x3*x4 - 4*x3*x5*x6*x7*x9 + 
        2*x3*x5*x6*x7 - 4*x3*x5*x6*x8*x9 + 2*x3*x5*x6*x8 + 4*x3*x5*x6*x9 - 2*x3*x5*x6 + 
        4*x3*x5*x7*x9 - 2*x3*x5*x7 - 2*x3*x5*x9 + x3*x5 - 2*x3*x6*x7*x8 + 2*x3*x6*x7*x9 + 
        2*x3*x6*x8*x9 - 2*x3*x6*x9 + x3*x6 - 2*x3*x7*x9 + x3*x7 + x3*x9 - x3 + 4*x4*x5*x6*x7*x8*x9
        - 4*x4*x5*x6*x7*x8 - 4*x4*x5*x6*x7*x9 + 4*x4*x5*x6*x7 - 4*x4*x5*x6*x8*x9 + 2*x4*x5*x6*x8 +
        4*x4*x5*x6*x9 - 2*x4*x5*x6 + 2*x4*x5*x7*x9 - 2*x4*x5*x7 - 2*x4*x5*x9 + x4*x5 + 
        2*x5*x6*x7*x9 - 2*x5*x6*x7 + 2*x5*x6*x8*x9 - 2*x5*x6*x9 + x5*x6 - 2*x5*x7*x9 + 2*x5*x7 + 
        x5*x9 - x5 - 2*x6*x7*x8*x9 + 2*x6*x7*x8 - x6*x8 + x7*x9 - x7 + 1,
        8*x1*x2*x3*x4*x5*x6*x7*x8 - 8*x1*x2*x3*x4*x5*x6*x7*x9 - 8*x1*x2*x3*x4*x5*x6*x8 + 
        4*x1*x2*x3*x4*x5*x6 - 8*x1*x2*x3*x4*x5*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x7*x9 + 
        8*x1*x2*x3*x4*x5*x8*x9 - 4*x1*x2*x3*x4*x5*x9 + 8*x1*x2*x3*x4*x6*x7*x8*x9 - 
        8*x1*x2*x3*x4*x6*x7*x8 + 4*x1*x2*x3*x4*x6*x7 + 4*x1*x2*x3*x4*x6*x8 - 4*x1*x2*x3*x4*x6 + 
        4*x1*x2*x3*x4*x7*x8 - 4*x1*x2*x3*x4*x7 - 4*x1*x2*x3*x4*x8*x9 + 2*x1*x2*x3*x4 - 
        8*x1*x2*x3*x5*x6*x7*x8 + 4*x1*x2*x3*x5*x6*x7 + 4*x1*x2*x3*x5*x6*x8 + 4*x1*x2*x3*x5*x6*x9
        - 4*x1*x2*x3*x5*x6 + 8*x1*x2*x3*x5*x7*x8*x9 - 4*x1*x2*x3*x5*x7*x9 - 4*x1*x2*x3*x5*x8*x9 +
        4*x1*x2*x3*x6*x7*x8 - 4*x1*x2*x3*x6*x7 - 4*x1*x2*x3*x6*x8*x9 + 2*x1*x2*x3*x6 - 
        4*x1*x2*x3*x7*x8*x9 + 2*x1*x2*x3*x7 + 4*x1*x2*x3*x8*x9 - 2*x1*x2*x3*x8 - 
        8*x1*x2*x4*x5*x6*x7*x8*x9 + 8*x1*x2*x4*x5*x6*x7*x9 - 4*x1*x2*x4*x5*x6*x7 + 
        8*x1*x2*x4*x5*x6*x8*x9 - 4*x1*x2*x4*x5*x6*x9 + 8*x1*x2*x4*x5*x7*x8*x9 - 
        4*x1*x2*x4*x5*x7*x9 - 8*x1*x2*x4*x5*x8*x9 + 4*x1*x2*x4*x5*x9 + 4*x1*x2*x4*x6*x7*x8 - 
        4*x1*x2*x4*x6*x7*x9 - 4*x1*x2*x4*x6*x8*x9 + 4*x1*x2*x4*x6*x9 - 4*x1*x2*x4*x7*x8 + 
        2*x1*x2*x4*x7 + 4*x1*x2*x4*x8*x9 - 2*x1*x2*x4*x9 + 4*x1*x2*x5*x6*x7*x8 - 
        4*x1*x2*x5*x6*x8*x9 - 4*x1*x2*x5*x7*x8*x9 + 4*x1*x2*x5*x8*x9 - 4*x1*x2*x6*x7*x8 + 
        2*x1*x2*x6*x7 + 4*x1*x2*x6*x8*x9 - 2*x1*x2*x6*x9 + 2*x1*x2*x7*x8 + 2*x1*x2*x7*x9 - 
        2*x1*x2*x7 - 2*x1*x2*x8*x9 - 8*x1*x3*x4*x5*x6*x7*x8*x9 + 8*x1*x3*x4*x5*x6*x7*x9 + 
        4*x1*x3*x4*x5*x6*x8 - 4*x1*x3*x4*x5*x6 + 8*x1*x3*x4*x5*x7*x8*x9 - 8*x1*x3*x4*x5*x7*x9 - 
        4*x1*x3*x4*x5*x8*x9 + 4*x1*x3*x4*x5*x9 + 4*x1*x3*x4*x6*x7*x8 - 4*x1*x3*x4*x6*x7 - 
        4*x1*x3*x4*x6*x8 + 4*x1*x3*x4*x6 - 4*x1*x3*x4*x7*x8 + 4*x1*x3*x4*x7 + 2*x1*x3*x4*x8 - 
        2*x1*x3*x4 + 8*x1*x3*x5*x6*x7*x8*x9 - 4*x1*x3*x5*x6*x7*x9 - 4*x1*x3*x5*x6*x8*x9 + 
        2*x1*x3*x5*x6 - 8*x1*x3*x5*x7*x8*x9 + 4*x1*x3*x5*x7*x9 + 4*x1*x3*x5*x8*x9 - 2*x1*x3*x5*x9
        - 4*x1*x3*x6*x7*x8*x9 + 2*x1*x3*x6*x7 + 4*x1*x3*x6*x8*x9 - 2*x1*x3*x6 + 4*x1*x3*x7*x8*x9 -
        2*x1*x3*x7 - 2*x1*x3*x8*x9 + x1*x3 + 8*x1*x4*x5*x6*x7*x8*x9 - 4*x1*x4*x5*x6*x7*x8 - 
        8*x1*x4*x5*x6*x7*x9 + 4*x1*x4*x5*x6*x7 - 4*x1*x4*x5*x6*x8*x9 + 4*x1*x4*x5*x6*x9 - 
        4*x1*x4*x5*x7*x8*x9 + 4*x1*x4*x5*x7*x9 + 4*x1*x4*x5*x8*x9 - 4*x1*x4*x5*x9 - 
        4*x1*x4*x6*x7*x8*x9 + 4*x1*x4*x6*x7*x9 + 4*x1*x4*x6*x8*x9 - 4*x1*x4*x6*x9 + 2*x1*x4*x7*x8
        - 2*x1*x4*x7 - 2*x1*x4*x8*x9 + 2*x1*x4*x9 - 4*x1*x5*x6*x7*x8*x9 + 4*x1*x5*x6*x7*x9 - 
        2*x1*x5*x6*x7 + 4*x1*x5*x6*x8*x9 - 2*x1*x5*x6*x9 + 4*x1*x5*x7*x8*x9 - 2*x1*x5*x7*x9 - 
        4*x1*x5*x8*x9 + 2*x1*x5*x9 + 4*x1*x6*x7*x8*x9 - 2*x1*x6*x7*x9 - 4*x1*x6*x8*x9 + 2*x1*x6*x9
        - 2*x1*x7*x8*x9 + x1*x7 + 2*x1*x8*x9 - x1*x9 + 8*x2*x3*x4*x5*x6*x7*x8*x9 - 
        8*x2*x3*x4*x5*x6*x7*x8 + 4*x2*x3*x4*x5*x6*x7 + 4*x2*x3*x4*x5*x6*x8 - 4*x2*x3*x4*x5*x6 + 
        4*x2*x3*x4*x5*x7*x8 - 4*x2*x3*x4*x5*x7 - 4*x2*x3*x4*x5*x8*x9 + 2*x2*x3*x4*x5 - 
        8*x2*x3*x4*x6*x7*x8*x9 + 8*x2*x3*x4*x6*x7*x8 - 4*x2*x3*x4*x6*x7 - 4*x2*x3*x4*x6*x8 + 
        4*x2*x3*x4*x6 - 4*x2*x3*x4*x7*x8 + 4*x2*x3*x4*x7 + 4*x2*x3*x4*x8*x9 - 2*x2*x3*x4 - 
        8*x2*x3*x5*x6*x7*x8*x9 + 8*x2*x3*x5*x6*x7*x8 + 4*x2*x3*x5*x6*x7*x9 - 4*x2*x3*x5*x6*x7 + 
        4*x2*x3*x5*x6*x8*x9 - 4*x2*x3*x5*x6*x8 - 4*x2*x3*x5*x6*x9 + 4*x2*x3*x5*x6 - 
        4*x2*x3*x5*x7*x8 + 2*x2*x3*x5*x7 + 2*x2*x3*x5*x8 + 2*x2*x3*x5*x9 - 2*x2*x3*x5 + 
        4*x2*x3*x6*x7*x8*x9 - 4*x2*x3*x6*x7*x8 + 2*x2*x3*x6*x7 + 2*x2*x3*x6*x8 - 2*x2*x3*x6 + 
        2*x2*x3*x7*x8 - 2*x2*x3*x7 - 2*x2*x3*x8*x9 + x2*x3 - 4*x2*x4*x5*x6*x8*x9 + 2*x2*x4*x5*x6 -
        4*x2*x4*x5*x7*x8*x9 + 2*x2*x4*x5*x7 + 4*x2*x4*x5*x8*x9 - 2*x2*x4*x5 + 4*x2*x4*x6*x7*x8*x9
        - 4*x2*x4*x6*x7*x8 + 2*x2*x4*x6*x7 + 2*x2*x4*x6*x8 - 2*x2*x4*x6 + 2*x2*x4*x7*x8 - 
        2*x2*x4*x7 - 2*x2*x4*x8*x9 + x2*x4 + 4*x2*x5*x6*x7*x8*x9 - 4*x2*x5*x6*x7*x8 - 
        4*x2*x5*x6*x7*x9 + 2*x2*x5*x6*x7 + 2*x2*x5*x6*x8 + 2*x2*x5*x6*x9 - 2*x2*x5*x6 + 
        2*x2*x5*x7*x8 + 2*x2*x5*x7*x9 - 2*x2*x5*x7 - 2*x2*x5*x8 - 2*x2*x5*x9 + 2*x2*x5 - 
        4*x2*x6*x7*x8*x9 + 4*x2*x6*x7*x8 + 2*x2*x6*x7*x9 - 2*x2*x6*x7 - 2*x2*x6*x8 + x2*x6 + 
        2*x2*x7*x8*x9 - 2*x2*x7*x8 - 2*x2*x7*x9 + 2*x2*x7 + x2*x8 + x2*x9 - x2 + 
        4*x3*x4*x5*x6*x7*x8 - 4*x3*x4*x5*x6*x7 - 4*x3*x4*x5*x6*x8 + 4*x3*x4*x5*x6 - 
        4*x3*x4*x5*x7*x8 + 4*x3*x4*x5*x7 + 2*x3*x4*x5*x8 - 2*x3*x4*x5 - 4*x3*x4*x6*x7*x8 + 
        4*x3*x4*x6*x7 + 4*x3*x4*x6*x8 - 4*x3*x4*x6 + 4*x3*x4*x7*x8 - 4*x3*x4*x7 - 2*x3*x4*x8 + 
        2*x3*x4 - 4*x3*x5*x6*x7*x8 + 2*x3*x5*x6*x7 + 2*x3*x5*x6*x8 - 2*x3*x5*x6 + 4*x3*x5*x7*x8 - 
        2*x3*x5*x7 - 2*x3*x5*x8 + x3*x5 + 2*x3*x6*x7*x8 - 2*x3*x6*x7 - 2*x3*x6*x8 + 2*x3*x6 - 
        2*x3*x7*x8 + 2*x3*x7 + x3*x8 - x3 + 2*x4*x5*x6*x8 - 2*x4*x5*x6 + 2*x4*x5*x7*x8 - 2*x4*x5*x7
        - 2*x4*x5*x8 + 2*x4*x5 + 2*x4*x6*x7*x8 - 2*x4*x6*x7 - 2*x4*x6*x8 + 2*x4*x6 - 2*x4*x7*x8 + 
        2*x4*x7 + x4*x8 - x4 + 2*x5*x6*x7*x8 - 2*x5*x6*x8 + x5*x6 - 2*x5*x7*x8 + x5*x7 + 2*x5*x8 - 
        x5 - 2*x6*x7*x8 + x6*x7 + 2*x6*x8 - x6 + x7*x8 - x7 - x8 + 1,
        16*x1*x2*x3*x4*x5*x6*x7*x8*x9 - 8*x1*x2*x3*x4*x5*x6*x7*x8 - 16*x1*x2*x3*x4*x5*x6*x8*x9 + 
        8*x1*x2*x3*x4*x5*x6*x8 - 8*x1*x2*x3*x4*x5*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x7*x8 + 
        8*x1*x2*x3*x4*x5*x8*x9 - 4*x1*x2*x3*x4*x5*x8 - 8*x1*x2*x3*x4*x6*x7*x8*x9 + 
        8*x1*x2*x3*x4*x6*x8*x9 + 8*x1*x2*x3*x4*x7*x8*x9 - 4*x1*x2*x3*x4*x7*x8 - 
        4*x1*x2*x3*x4*x8*x9 - 8*x1*x2*x3*x5*x6*x7*x8*x9 + 8*x1*x2*x3*x5*x6*x7*x8 - 
        4*x1*x2*x3*x5*x6*x7 + 8*x1*x2*x3*x5*x6*x8*x9 - 8*x1*x2*x3*x5*x6*x8 + 4*x1*x2*x3*x5*x6 + 
        8*x1*x2*x3*x5*x7*x8*x9 - 8*x1*x2*x3*x5*x7*x8 - 4*x1*x2*x3*x5*x7*x9 + 4*x1*x2*x3*x5*x7 - 
        4*x1*x2*x3*x5*x8*x9 + 4*x1*x2*x3*x5*x8 - 2*x1*x2*x3*x5 + 8*x1*x2*x3*x6*x7*x8*x9 - 
        4*x1*x2*x3*x6*x7*x8 - 4*x1*x2*x3*x6*x7*x9 + 4*x1*x2*x3*x6*x7 - 8*x1*x2*x3*x6*x8*x9 + 
        4*x1*x2*x3*x6*x8 + 4*x1*x2*x3*x6*x9 - 4*x1*x2*x3*x6 - 8*x1*x2*x3*x7*x8*x9 + 
        4*x1*x2*x3*x7*x8 + 4*x1*x2*x3*x7*x9 - 2*x1*x2*x3*x7 + 4*x1*x2*x3*x8*x9 - 2*x1*x2*x3*x8 - 
        2*x1*x2*x3*x9 + 2*x1*x2*x3 - 4*x1*x2*x4*x5*x7*x8 - 4*x1*x2*x4*x7*x8*x9 + 4*x1*x2*x4*x7*x8
        - 4*x1*x2*x5*x7*x8*x9 + 4*x1*x2*x5*x7*x8 + 4*x1*x2*x5*x7*x9 - 2*x1*x2*x5*x7 + 
        4*x1*x2*x7*x8*x9 - 2*x1*x2*x7*x8 - 2*x1*x2*x7*x9 - 16*x1*x3*x4*x5*x6*x7*x8*x9 + 
        8*x1*x3*x4*x5*x6*x7*x8 + 8*x1*x3*x4*x5*x6*x7*x9 - 4*x1*x3*x4*x5*x6*x7 + 
        8*x1*x3*x4*x5*x6*x8*x9 - 4*x1*x3*x4*x5*x6*x8 + 8*x1*x3*x4*x5*x7*x8*x9 - 
        8*x1*x3*x4*x5*x7*x8 - 4*x1*x3*x4*x5*x7*x9 + 4*x1*x3*x4*x5*x7 - 4*x1*x3*x4*x5*x8*x9 + 
        4*x1*x3*x4*x5*x8 - 2*x1*x3*x4*x5 + 8*x1*x3*x4*x6*x7*x8*x9 - 4*x1*x3*x4*x6*x7*x8 - 
        4*x1*x3*x4*x6*x7*x9 + 4*x1*x3*x4*x6*x7 - 4*x1*x3*x4*x6*x8*x9 - 4*x1*x3*x4*x7*x8*x9 + 
        4*x1*x3*x4*x7*x8 - 2*x1*x3*x4*x7 + 2*x1*x3*x4*x9 + 8*x1*x3*x5*x6*x7*x8*x9 - 
        4*x1*x3*x5*x6*x7*x8 - 4*x1*x3*x5*x6*x7*x9 + 4*x1*x3*x5*x6*x7 - 4*x1*x3*x5*x6*x8*x9 + 
        4*x1*x3*x5*x6*x8 - 2*x1*x3*x5*x6 - 4*x1*x3*x5*x7*x8*x9 + 4*x1*x3*x5*x7*x8 + 
        4*x1*x3*x5*x7*x9 - 4*x1*x3*x5*x7 - 2*x1*x3*x5*x8 + 2*x1*x3*x5 - 8*x1*x3*x6*x7*x8*x9 + 
        4*x1*x3*x6*x7*x8 + 4*x1*x3*x6*x7*x9 - 4*x1*x3*x6*x7 + 4*x1*x3*x6*x8*x9 - 2*x1*x3*x6*x8 - 
        2*x1*x3*x6*x9 + 2*x1*x3*x6 + 4*x1*x3*x7*x8*x9 - 2*x1*x3*x7*x8 - 2*x1*x3*x7*x9 + 2*x1*x3*x7
        - x1*x3 + 4*x1*x4*x5*x7*x8 - 2*x1*x4*x5*x7 - 2*x1*x4*x5*x8 + 2*x1*x4*x5 - 2*x1*x4*x7*x8 + 
        2*x1*x4*x7*x9 + 2*x1*x4*x8*x9 - 2*x1*x4*x9 - 2*x1*x5*x7*x8 - 2*x1*x5*x7*x9 + 2*x1*x5*x7 + 
        2*x1*x5*x8*x9 - x1*x5 - 2*x1*x8*x9 + x1*x8 + x1*x9 - 8*x2*x3*x4*x5*x6*x7*x9 + 
        4*x2*x3*x4*x5*x6*x7 + 8*x2*x3*x4*x5*x6*x8*x9 - 4*x2*x3*x4*x5*x6*x8 - 4*x2*x3*x4*x5*x7*x8
        + 4*x2*x3*x4*x5*x7*x9 - 4*x2*x3*x4*x5*x8*x9 + 4*x2*x3*x4*x5*x8 - 2*x2*x3*x4*x5 + 
        4*x2*x3*x4*x6*x7*x8 + 4*x2*x3*x4*x6*x7*x9 - 4*x2*x3*x4*x6*x7 - 4*x2*x3*x4*x6*x8*x9 - 
        4*x2*x3*x4*x7*x9 + 2*x2*x3*x4*x7 + 2*x2*x3*x4*x9 + 4*x2*x3*x5*x6*x7*x9 - 
        4*x2*x3*x5*x6*x8*x9 + 4*x2*x3*x5*x6*x8 - 2*x2*x3*x5*x6 - 4*x2*x3*x5*x7*x8*x9 + 
        4*x2*x3*x5*x7*x8 - 2*x2*x3*x5*x7 + 4*x2*x3*x5*x8*x9 - 4*x2*x3*x5*x8 + 2*x2*x3*x5 - 
        4*x2*x3*x6*x7*x8*x9 + 4*x2*x3*x6*x8*x9 - 2*x2*x3*x6*x8 - 2*x2*x3*x6*x9 + 2*x2*x3*x6 + 
        4*x2*x3*x7*x8*x9 - 2*x2*x3*x7*x8 - 2*x2*x3*x8*x9 + 2*x2*x3*x8 - x2*x3 - 
        8*x2*x4*x5*x6*x7*x8*x9 + 4*x2*x4*x5*x6*x7*x8 + 8*x2*x4*x5*x6*x7*x9 - 4*x2*x4*x5*x6*x7 + 
        4*x2*x4*x5*x7*x8*x9 - 4*x2*x4*x5*x7*x9 + 2*x2*x4*x5*x7 + 4*x2*x4*x6*x7*x8*x9 - 
        4*x2*x4*x6*x7*x8 - 4*x2*x4*x6*x7*x9 + 4*x2*x4*x6*x7 + 2*x2*x4*x7*x9 - 2*x2*x4*x7 + 
        4*x2*x5*x6*x7*x8*x9 - 4*x2*x5*x6*x7*x8 - 4*x2*x5*x6*x7*x9 + 2*x2*x5*x6*x7 + 2*x2*x6*x7*x8
        + 2*x2*x6*x7*x9 - 2*x2*x6*x7 - 2*x2*x7*x8*x9 + x2*x7 + 8*x3*x4*x5*x6*x7*x8*x9 - 
        4*x3*x4*x5*x6*x7*x8 - 8*x3*x4*x5*x6*x8*x9 + 4*x3*x4*x5*x6*x8 - 4*x3*x4*x5*x7*x8*x9 + 
        4*x3*x4*x5*x7*x8 - 2*x3*x4*x5*x7 + 4*x3*x4*x5*x8*x9 - 4*x3*x4*x5*x8 + 2*x3*x4*x5 - 
        4*x3*x4*x6*x7*x8*x9 + 4*x3*x4*x6*x8*x9 + 2*x3*x4*x7*x9 - 2*x3*x4*x9 - 4*x3*x5*x6*x7*x8*x9
        + 4*x3*x5*x6*x8*x9 - 2*x3*x5*x6*x8 + 4*x3*x5*x7*x8*x9 - 2*x3*x5*x7*x8 - 2*x3*x5*x7*x9 + 
        2*x3*x5*x7 - 2*x3*x5*x8*x9 + 2*x3*x5*x8 - x3*x5 + 4*x3*x6*x7*x8*x9 - 2*x3*x6*x8*x9 - 
        2*x3*x7*x8*x9 + x3*x9 - 4*x4*x5*x6*x7*x9 + 2*x4*x5*x6*x7 + 4*x4*x5*x6*x8*x9 - 
        2*x4*x5*x6*x8 - 2*x4*x5*x7*x8 + 2*x4*x5*x7*x9 - 2*x4*x5*x8*x9 + 2*x4*x5*x8 - x4*x5 + 
        2*x4*x6*x7*x8 + 2*x4*x6*x7*x9 - 2*x4*x6*x7 - 2*x4*x6*x8*x9 - 2*x4*x7*x9 + x4*x7 + x4*x9 + 
        2*x5*x6*x7*x8 + 2*x5*x6*x7*x9 - 2*x5*x6*x7 - 2*x5*x6*x8*x9 + x5*x6 - 2*x6*x7*x8 - 
        2*x6*x7*x9 + 2*x6*x7 + x6*x8 + x6*x9 - x6 + x7*x8 + x7*x9 - x7 + x8*x9 - x8 - x9 + 1,
        16*x1*x2*x3*x4*x5*x6*x7*x8*x9 - 16*x1*x2*x3*x4*x5*x6*x8*x9 + 8*x1*x2*x3*x4*x5*x6*x9 - 
        8*x1*x2*x3*x4*x5*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x8*x9 - 4*x1*x2*x3*x4*x5*x9 - 
        8*x1*x2*x3*x4*x6*x7*x8*x9 + 8*x1*x2*x3*x4*x6*x8*x9 - 4*x1*x2*x3*x4*x6*x9 + 
        8*x1*x2*x3*x4*x7*x8*x9 - 4*x1*x2*x3*x4*x7*x9 - 8*x1*x2*x3*x4*x8*x9 + 4*x1*x2*x3*x4*x9 - 
        8*x1*x2*x3*x5*x6*x7*x8*x9 + 8*x1*x2*x3*x5*x6*x8*x9 - 4*x1*x2*x3*x5*x6*x9 + 
        8*x1*x2*x3*x5*x7*x8*x9 - 8*x1*x2*x3*x5*x8*x9 + 4*x1*x2*x3*x5*x9 + 8*x1*x2*x3*x6*x7*x8*x9
        - 4*x1*x2*x3*x6*x7*x9 - 8*x1*x2*x3*x6*x8*x9 + 4*x1*x2*x3*x6*x9 - 8*x1*x2*x3*x7*x8*x9 + 
        4*x1*x2*x3*x7*x9 + 8*x1*x2*x3*x8*x9 - 4*x1*x2*x3*x9 - 8*x1*x2*x4*x5*x6*x7*x8*x9 + 
        8*x1*x2*x4*x5*x6*x8*x9 - 4*x1*x2*x4*x5*x6*x9 + 8*x1*x2*x4*x5*x7*x8*x9 - 
        4*x1*x2*x4*x5*x7*x8 - 4*x1*x2*x4*x5*x8*x9 + 2*x1*x2*x4*x5 + 8*x1*x2*x4*x6*x7*x8*x9 - 
        4*x1*x2*x4*x6*x7*x8 - 4*x1*x2*x4*x6*x7*x9 + 4*x1*x2*x4*x6*x7 - 4*x1*x2*x4*x6*x8*x9 + 
        4*x1*x2*x4*x6*x9 - 2*x1*x2*x4*x6 - 8*x1*x2*x4*x7*x8*x9 + 4*x1*x2*x4*x7*x8 + 
        4*x1*x2*x4*x7*x9 - 2*x1*x2*x4*x7 + 4*x1*x2*x4*x8*x9 - 2*x1*x2*x4*x9 + 
        8*x1*x2*x5*x6*x7*x8*x9 - 4*x1*x2*x5*x6*x7*x8 - 4*x1*x2*x5*x6*x8*x9 + 2*x1*x2*x5*x6 - 
        8*x1*x2*x5*x7*x8*x9 + 4*x1*x2*x5*x7*x8 + 4*x1*x2*x5*x8*x9 - 2*x1*x2*x5 - 
        8*x1*x2*x6*x7*x8*x9 + 4*x1*x2*x6*x7*x8 + 4*x1*x2*x6*x7*x9 - 2*x1*x2*x6*x7 + 
        4*x1*x2*x6*x8*x9 - 2*x1*x2*x6*x9 + 8*x1*x2*x7*x8*x9 - 4*x1*x2*x7*x8 - 4*x1*x2*x7*x9 + 
        2*x1*x2*x7 - 4*x1*x2*x8*x9 + 2*x1*x2*x9 - 16*x1*x3*x4*x5*x6*x7*x8*x9 + 
        8*x1*x3*x4*x5*x6*x7*x8 + 8*x1*x3*x4*x5*x6*x7*x9 - 8*x1*x3*x4*x5*x6*x7 + 
        16*x1*x3*x4*x5*x6*x8*x9 - 8*x1*x3*x4*x5*x6*x8 - 8*x1*x3*x4*x5*x6*x9 + 4*x1*x3*x4*x5*x6 + 
        8*x1*x3*x4*x5*x7*x8*x9 - 4*x1*x3*x4*x5*x7*x8 - 4*x1*x3*x4*x5*x7*x9 + 4*x1*x3*x4*x5*x7 - 
        8*x1*x3*x4*x5*x8*x9 + 4*x1*x3*x4*x5*x8 + 4*x1*x3*x4*x5*x9 - 2*x1*x3*x4*x5 + 
        8*x1*x3*x4*x6*x7*x8*x9 - 4*x1*x3*x4*x6*x7*x8 - 4*x1*x3*x4*x6*x7*x9 + 4*x1*x3*x4*x6*x7 - 
        8*x1*x3*x4*x6*x8*x9 + 4*x1*x3*x4*x6*x8 + 4*x1*x3*x4*x6*x9 - 2*x1*x3*x4*x6 - 
        4*x1*x3*x4*x7*x8*x9 + 4*x1*x3*x4*x7*x9 - 2*x1*x3*x4*x7 + 4*x1*x3*x4*x8*x9 - 2*x1*x3*x4*x9
        + 8*x1*x3*x5*x6*x7*x8*x9 - 4*x1*x3*x5*x6*x7*x8 - 4*x1*x3*x5*x6*x7*x9 + 4*x1*x3*x5*x6*x7 -
        8*x1*x3*x5*x6*x8*x9 + 4*x1*x3*x5*x6*x8 + 4*x1*x3*x5*x6*x9 - 2*x1*x3*x5*x6 - 
        4*x1*x3*x5*x7*x8*x9 + 4*x1*x3*x5*x8*x9 - 2*x1*x3*x5*x9 - 8*x1*x3*x6*x7*x8*x9 + 
        4*x1*x3*x6*x7*x8 + 4*x1*x3*x6*x7*x9 - 2*x1*x3*x6*x7 + 8*x1*x3*x6*x8*x9 - 4*x1*x3*x6*x8 - 
        4*x1*x3*x6*x9 + 2*x1*x3*x6 + 4*x1*x3*x7*x8*x9 - 2*x1*x3*x7*x9 - 4*x1*x3*x8*x9 + 2*x1*x3*x9
        + 8*x1*x4*x5*x6*x7*x8*x9 - 4*x1*x4*x5*x6*x7*x8 - 4*x1*x4*x5*x6*x7*x9 + 4*x1*x4*x5*x6*x7 -
        8*x1*x4*x5*x6*x8*x9 + 4*x1*x4*x5*x6*x8 + 4*x1*x4*x5*x6*x9 - 2*x1*x4*x5*x6 - 
        4*x1*x4*x5*x7*x8*x9 + 4*x1*x4*x5*x7*x8 - 2*x1*x4*x5*x7 + 4*x1*x4*x5*x8*x9 - 2*x1*x4*x5*x8
        - 8*x1*x4*x6*x7*x8*x9 + 4*x1*x4*x6*x7*x8 + 4*x1*x4*x6*x7*x9 - 4*x1*x4*x6*x7 + 
        4*x1*x4*x6*x8*x9 - 2*x1*x4*x6*x8 - 2*x1*x4*x6*x9 + 2*x1*x4*x6 + 4*x1*x4*x7*x8*x9 - 
        2*x1*x4*x7*x8 - 2*x1*x4*x7*x9 + 2*x1*x4*x7 - 2*x1*x4*x8*x9 - 8*x1*x5*x6*x7*x8*x9 + 
        4*x1*x5*x6*x7*x8 + 4*x1*x5*x6*x7*x9 - 2*x1*x5*x6*x7 + 4*x1*x5*x6*x8*x9 - 2*x1*x5*x6*x8 - 
        2*x1*x5*x6*x9 + 4*x1*x5*x7*x8*x9 - 2*x1*x5*x7*x8 - 2*x1*x5*x8*x9 + x1*x5 + 
        8*x1*x6*x7*x8*x9 - 4*x1*x6*x7*x8 - 4*x1*x6*x7*x9 + 2*x1*x6*x7 - 4*x1*x6*x8*x9 + 2*x1*x6*x8
        + 2*x1*x6*x9 - x1*x6 - 4*x1*x7*x8*x9 + 2*x1*x7*x8 + 2*x1*x7*x9 - x1*x7 + 2*x1*x8*x9 - x1*x9
        - 8*x2*x3*x4*x5*x6*x7*x8*x9 + 8*x2*x3*x4*x5*x6*x8*x9 - 4*x2*x3*x4*x5*x6*x9 + 
        8*x2*x3*x4*x6*x7*x8*x9 - 4*x2*x3*x4*x6*x7*x9 - 8*x2*x3*x4*x6*x8*x9 + 4*x2*x3*x4*x6*x9 - 
        4*x2*x3*x4*x7*x8*x9 + 4*x2*x3*x4*x7*x9 + 4*x2*x3*x4*x8*x9 - 2*x2*x3*x4*x9 - 
        4*x2*x3*x6*x7*x8*x9 + 4*x2*x3*x6*x7*x9 + 4*x2*x3*x6*x8*x9 - 2*x2*x3*x6*x9 + 
        4*x2*x3*x7*x8*x9 - 4*x2*x3*x7*x9 - 4*x2*x3*x8*x9 + 2*x2*x3*x9 + 4*x2*x4*x5*x6*x7*x8 - 
        4*x2*x4*x5*x6*x8*x9 + 4*x2*x4*x5*x6*x9 - 2*x2*x4*x5*x6 - 4*x2*x4*x6*x7*x8*x9 + 
        4*x2*x4*x6*x7*x9 - 2*x2*x4*x6*x7 + 4*x2*x4*x6*x8*x9 - 4*x2*x4*x6*x9 + 2*x2*x4*x6 + 
        4*x2*x4*x7*x8*x9 - 2*x2*x4*x7*x8 - 4*x2*x4*x7*x9 + 2*x2*x4*x7 - 2*x2*x4*x8*x9 + 2*x2*x4*x9
        - x2*x4 + 4*x2*x6*x7*x8*x9 - 2*x2*x6*x7*x8 - 4*x2*x6*x7*x9 + 2*x2*x6*x7 - 2*x2*x6*x8*x9 + 
        2*x2*x6*x9 - x2*x6 - 4*x2*x7*x8*x9 + 2*x2*x7*x8 + 4*x2*x7*x9 - 2*x2*x7 + 2*x2*x8*x9 - 
        2*x2*x9 + x2 + 8*x3*x4*x5*x6*x7*x8*x9 - 4*x3*x4*x5*x6*x7*x8 - 4*x3*x4*x5*x6*x7*x9 + 
        4*x3*x4*x5*x6*x7 - 8*x3*x4*x5*x6*x8*x9 + 4*x3*x4*x5*x6*x8 + 4*x3*x4*x5*x6*x9 - 
        2*x3*x4*x5*x6 - 4*x3*x4*x6*x7*x8*x9 + 4*x3*x4*x6*x7*x9 - 2*x3*x4*x6*x7 + 4*x3*x4*x6*x8*x9
        - 2*x3*x4*x6*x9 + 2*x3*x4*x7*x8 - 2*x3*x4*x7*x9 - 2*x3*x4*x8 + x3*x4 - 4*x3*x5*x6*x7*x8*x9
        + 4*x3*x5*x6*x7*x8 + 4*x3*x5*x6*x7*x9 - 4*x3*x5*x6*x7 + 4*x3*x5*x6*x8*x9 - 4*x3*x5*x6*x8 -
        2*x3*x5*x6*x9 + 2*x3*x5*x6 + 4*x3*x6*x7*x8*x9 - 2*x3*x6*x7*x8 - 4*x3*x6*x7*x9 + 2*x3*x6*x7
        - 4*x3*x6*x8*x9 + 2*x3*x6*x8 + 2*x3*x6*x9 - x3*x6 - 2*x3*x7*x8*x9 + 2*x3*x7*x9 + 2*x3*x8*x9
        - x3*x9 - 4*x4*x5*x6*x7*x8*x9 + 4*x4*x5*x6*x7*x9 - 2*x4*x5*x6*x7 + 4*x4*x5*x6*x8*x9 - 
        2*x4*x5*x6*x8 - 4*x4*x5*x6*x9 + 2*x4*x5*x6 + 4*x4*x6*x7*x8*x9 - 4*x4*x6*x7*x9 + 2*x4*x6*x7
        - 2*x4*x6*x8*x9 + 2*x4*x6*x9 - x4*x6 - 2*x4*x7*x8*x9 + 2*x4*x7*x9 - x4*x7 + x4*x8 + 
        4*x5*x6*x7*x8*x9 - 2*x5*x6*x7*x8 - 4*x5*x6*x7*x9 + 2*x5*x6*x7 - 2*x5*x6*x8*x9 + 2*x5*x6*x8
        + 2*x5*x6*x9 - x5*x6 - 4*x6*x7*x8*x9 + 2*x6*x7*x8 + 4*x6*x7*x9 - 2*x6*x7 + 2*x6*x8*x9 - 
        x6*x8 - 2*x6*x9 + x6 + 2*x7*x8*x9 - x7*x8 - 2*x7*x9 + x7 - x8*x9 + x9,
        -16*x1*x2*x3*x4*x5*x6*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x6*x7*x8 + 8*x1*x2*x3*x4*x5*x6*x7*x9 - 
        8*x1*x2*x3*x4*x5*x6*x7 + 8*x1*x2*x3*x4*x5*x6*x8*x9 - 8*x1*x2*x3*x4*x5*x6*x8 + 
        4*x1*x2*x3*x4*x5*x6 + 16*x1*x2*x3*x4*x5*x7*x8*x9 - 8*x1*x2*x3*x4*x5*x7*x8 - 
        8*x1*x2*x3*x4*x5*x7*x9 + 4*x1*x2*x3*x4*x5*x7 - 8*x1*x2*x3*x4*x5*x8*x9 + 
        8*x1*x2*x3*x4*x5*x8 + 4*x1*x2*x3*x4*x5*x9 - 4*x1*x2*x3*x4*x5 + 4*x1*x2*x3*x4*x6*x8 - 
        4*x1*x2*x3*x4*x6*x9 - 8*x1*x2*x3*x4*x7*x8*x9 + 4*x1*x2*x3*x4*x7*x8 + 4*x1*x2*x3*x4*x7*x9
        + 4*x1*x2*x3*x4*x8*x9 - 4*x1*x2*x3*x4*x8 + 16*x1*x2*x3*x5*x6*x7*x8*x9 - 
        8*x1*x2*x3*x5*x6*x7*x8 - 8*x1*x2*x3*x5*x6*x7*x9 + 4*x1*x2*x3*x5*x6*x7 - 
        8*x1*x2*x3*x5*x6*x8*x9 + 8*x1*x2*x3*x5*x6*x8 + 4*x1*x2*x3*x5*x6*x9 - 4*x1*x2*x3*x5*x6 - 
        8*x1*x2*x3*x5*x7*x8*x9 + 4*x1*x2*x3*x5*x7*x8 + 4*x1*x2*x3*x5*x7*x9 + 4*x1*x2*x3*x5*x8*x9
        - 4*x1*x2*x3*x5*x8 - 4*x1*x2*x3*x5*x9 + 2*x1*x2*x3*x5 - 8*x1*x2*x3*x6*x7*x8*x9 + 
        4*x1*x2*x3*x6*x7*x8 + 4*x1*x2*x3*x6*x7*x9 + 4*x1*x2*x3*x6*x8*x9 - 4*x1*x2*x3*x6*x8 + 
        8*x1*x2*x3*x7*x8*x9 - 4*x1*x2*x3*x7*x8 - 4*x1*x2*x3*x7*x9 - 4*x1*x2*x3*x8*x9 + 
        2*x1*x2*x3*x8 + 2*x1*x2*x3*x9 + 8*x1*x2*x4*x5*x6*x7*x8*x9 - 8*x1*x2*x4*x5*x6*x8*x9 + 
        4*x1*x2*x4*x5*x6*x8 - 8*x1*x2*x4*x5*x7*x8*x9 + 4*x1*x2*x4*x5*x7*x8 + 4*x1*x2*x4*x5*x8*x9
        - 4*x1*x2*x4*x5*x8 - 4*x1*x2*x4*x6*x7*x8 - 4*x1*x2*x4*x6*x7*x9 + 4*x1*x2*x4*x6*x7 + 
        4*x1*x2*x4*x6*x9 - 2*x1*x2*x4*x6 + 4*x1*x2*x4*x7*x8*x9 - 2*x1*x2*x4*x7 - 2*x1*x2*x4*x9 + 
        2*x1*x2*x4 - 8*x1*x2*x5*x6*x7*x8*x9 + 4*x1*x2*x5*x6*x7*x9 + 8*x1*x2*x5*x6*x8*x9 - 
        4*x1*x2*x5*x6*x8 - 4*x1*x2*x5*x6*x9 + 2*x1*x2*x5*x6 + 4*x1*x2*x5*x7*x8*x9 - 2*x1*x2*x5*x7
        - 4*x1*x2*x5*x8*x9 + 2*x1*x2*x5*x8 + 2*x1*x2*x5*x9 + 4*x1*x2*x6*x7*x8*x9 - 2*x1*x2*x6*x7 -
        4*x1*x2*x6*x8*x9 + 2*x1*x2*x6*x8 - 4*x1*x2*x7*x8*x9 + 2*x1*x2*x7 + 2*x1*x2*x8*x9 - x1*x2 +
        4*x1*x3*x4*x5*x6*x7 - 4*x1*x3*x4*x5*x6*x9 - 8*x1*x3*x4*x5*x7*x8*x9 + 4*x1*x3*x4*x5*x7*x8
        + 4*x1*x3*x4*x5*x7*x9 - 4*x1*x3*x4*x5*x7 + 4*x1*x3*x4*x5*x8*x9 - 4*x1*x3*x4*x5*x8 + 
        2*x1*x3*x4*x5 + 4*x1*x3*x4*x6*x9 - 2*x1*x3*x4*x6 + 8*x1*x3*x4*x7*x8*x9 - 4*x1*x3*x4*x7*x8
        - 4*x1*x3*x4*x7*x9 + 2*x1*x3*x4*x7 - 4*x1*x3*x4*x8*x9 + 2*x1*x3*x4*x8 - 
        8*x1*x3*x5*x6*x7*x8*x9 + 4*x1*x3*x5*x6*x7*x8 + 4*x1*x3*x5*x6*x7*x9 - 4*x1*x3*x5*x6*x7 + 
        4*x1*x3*x5*x6*x8*x9 - 4*x1*x3*x5*x6*x8 + 2*x1*x3*x5*x6 + 8*x1*x3*x5*x7*x8*x9 - 
        4*x1*x3*x5*x7*x8 - 4*x1*x3*x5*x7*x9 + 2*x1*x3*x5*x7 - 4*x1*x3*x5*x8*x9 + 4*x1*x3*x5*x8 + 
        2*x1*x3*x5*x9 - 2*x1*x3*x5 + 8*x1*x3*x6*x7*x8*x9 - 4*x1*x3*x6*x7*x8 - 4*x1*x3*x6*x7*x9 + 
        2*x1*x3*x6*x7 - 4*x1*x3*x6*x8*x9 + 2*x1*x3*x6*x8 - 8*x1*x3*x7*x8*x9 + 4*x1*x3*x7*x8 + 
        4*x1*x3*x7*x9 - 2*x1*x3*x7 + 4*x1*x3*x8*x9 - 2*x1*x3*x8 - 2*x1*x3*x9 + x1*x3 - 
        4*x1*x4*x5*x6*x7*x9 + 4*x1*x4*x5*x6*x9 - 2*x1*x4*x5*x6 + 4*x1*x4*x5*x7*x8*x9 - 
        4*x1*x4*x5*x7*x8 + 2*x1*x4*x5*x7 + 2*x1*x4*x5*x8 - 2*x1*x4*x5*x9 + 4*x1*x4*x6*x7*x9 - 
        2*x1*x4*x6*x7 - 4*x1*x4*x6*x9 + 2*x1*x4*x6 - 4*x1*x4*x7*x8*x9 + 2*x1*x4*x7*x8 + 2*x1*x4*x9
        - x1*x4 + 4*x1*x5*x6*x7*x8*x9 - 4*x1*x5*x6*x8*x9 + 2*x1*x5*x6*x8 - 4*x1*x5*x7*x8*x9 + 
        2*x1*x5*x7*x8 + 2*x1*x5*x8*x9 - 2*x1*x5*x8 - 4*x1*x6*x7*x8*x9 + 2*x1*x6*x7*x8 + 
        4*x1*x6*x8*x9 - 2*x1*x6*x8 + 4*x1*x7*x8*x9 - 2*x1*x7*x8 - 2*x1*x8*x9 + x1*x8 + 
        8*x2*x3*x4*x5*x6*x7*x8*x9 - 8*x2*x3*x4*x5*x6*x7*x8 - 8*x2*x3*x4*x5*x6*x7*x9 + 
        8*x2*x3*x4*x5*x6*x7 + 4*x2*x3*x4*x5*x6*x8 - 4*x2*x3*x4*x5*x6 - 8*x2*x3*x4*x5*x7*x8*x9 + 
        4*x2*x3*x4*x5*x7*x8 + 8*x2*x3*x4*x5*x7*x9 - 4*x2*x3*x4*x5*x7 + 4*x2*x3*x4*x5*x8*x9 - 
        4*x2*x3*x4*x5*x8 - 4*x2*x3*x4*x5*x9 + 4*x2*x3*x4*x5 + 4*x2*x3*x4*x6*x7*x8 + 
        4*x2*x3*x4*x6*x7*x9 - 4*x2*x3*x4*x6*x7 - 4*x2*x3*x4*x6*x8 + 2*x2*x3*x4*x6 + 
        4*x2*x3*x4*x7*x8*x9 - 4*x2*x3*x4*x7*x8 - 4*x2*x3*x4*x7*x9 + 2*x2*x3*x4*x7 - 
        4*x2*x3*x4*x8*x9 + 4*x2*x3*x4*x8 + 2*x2*x3*x4*x9 - 2*x2*x3*x4 - 8*x2*x3*x5*x6*x7*x8*x9 + 
        8*x2*x3*x5*x6*x7*x8 + 4*x2*x3*x5*x6*x7*x9 - 4*x2*x3*x5*x6*x7 - 4*x2*x3*x5*x6*x8 + 
        2*x2*x3*x5*x6 + 4*x2*x3*x5*x7*x8*x9 - 4*x2*x3*x5*x7*x8 - 4*x2*x3*x5*x7*x9 + 2*x2*x3*x5*x7
        + 2*x2*x3*x5*x8 + 2*x2*x3*x5*x9 - 2*x2*x3*x5 + 4*x2*x3*x6*x7*x8*x9 - 4*x2*x3*x6*x7*x8 - 
        4*x2*x3*x6*x7*x9 + 2*x2*x3*x6*x7 + 2*x2*x3*x6*x8 - 4*x2*x3*x7*x8*x9 + 4*x2*x3*x7*x8 + 
        4*x2*x3*x7*x9 - 2*x2*x3*x7 + 2*x2*x3*x8*x9 - 2*x2*x3*x8 - 2*x2*x3*x9 + x2*x3 + 
        4*x3*x4*x5*x6*x7*x8 - 4*x3*x4*x5*x6*x7 - 4*x3*x4*x5*x6*x8*x9 + 4*x3*x4*x5*x6*x9 + 
        4*x3*x4*x5*x7*x8*x9 - 4*x3*x4*x5*x7*x8 - 4*x3*x4*x5*x7*x9 + 4*x3*x4*x5*x7 + 2*x3*x4*x5*x8
        - 2*x3*x4*x5 - 4*x3*x4*x6*x7*x8 + 2*x3*x4*x6*x7 + 2*x3*x4*x6*x8 - 2*x3*x4*x6*x9 - 
        4*x3*x4*x7*x8*x9 + 4*x3*x4*x7*x8 + 2*x3*x4*x7*x9 - 2*x3*x4*x7 + 2*x3*x4*x8*x9 - 2*x3*x4*x8
        + x3*x4 + 4*x3*x5*x6*x7*x8*x9 - 4*x3*x5*x6*x7*x8 + 2*x3*x5*x6*x7 + 2*x3*x5*x6*x8 - 
        2*x3*x5*x6*x9 - 4*x3*x5*x7*x8*x9 + 4*x3*x5*x7*x8 + 2*x3*x5*x7*x9 - 2*x3*x5*x7 - 2*x3*x5*x8
        + x3*x5 - 4*x3*x6*x7*x8*x9 + 4*x3*x6*x7*x8 + 2*x3*x6*x7*x9 - 2*x3*x6*x7 + 2*x3*x6*x8*x9 - 
        2*x3*x6*x8 + 4*x3*x7*x8*x9 - 4*x3*x7*x8 - 2*x3*x7*x9 + 2*x3*x7 - 2*x3*x8*x9 + 2*x3*x8 + 
        x3*x9 - x3 - 4*x4*x5*x6*x7*x8*x9 + 4*x4*x5*x6*x7*x9 + 4*x4*x5*x6*x8*x9 - 2*x4*x5*x6*x8 - 
        4*x4*x5*x6*x9 + 2*x4*x5*x6 + 2*x4*x5*x7*x8 - 2*x4*x5*x7 - 2*x4*x5*x8*x9 + 2*x4*x5*x9 + 
        2*x4*x6*x7*x8 - 2*x4*x6*x7*x9 + 2*x4*x6*x9 - x4*x6 + 2*x4*x7*x8*x9 - 2*x4*x7*x8 + x4*x7 - 
        x4*x9 - 2*x5*x6*x7*x9 + 2*x5*x6*x9 - x5*x6 + 2*x5*x7*x8*x9 - 2*x5*x7*x8 + x5*x7 + x5*x8 - 
        x5*x9 + 2*x6*x7*x8*x9 - 2*x6*x7*x8 + x6*x7 - 2*x6*x8*x9 + x6*x8 - 2*x7*x8*x9 + 2*x7*x8 - x7
        + x8*x9 - x8 + 1,
        -8*x1*x2*x3*x4*x5*x6*x7*x8 - 8*x1*x2*x3*x4*x5*x6*x7*x9 + 8*x1*x2*x3*x4*x5*x6*x7 + 
        8*x1*x2*x3*x4*x5*x6*x8*x9 - 8*x1*x2*x3*x4*x5*x7*x8*x9 + 8*x1*x2*x3*x4*x5*x7*x8 + 
        8*x1*x2*x3*x4*x5*x7*x9 - 4*x1*x2*x3*x4*x5*x7 - 4*x1*x2*x3*x4*x5*x9 + 
        8*x1*x2*x3*x4*x6*x7*x8*x9 - 4*x1*x2*x3*x4*x6*x7 - 8*x1*x2*x3*x4*x6*x8*x9 + 
        4*x1*x2*x3*x4*x6*x9 - 4*x1*x2*x3*x4*x7*x8 - 4*x1*x2*x3*x4*x7*x9 + 4*x1*x2*x3*x4*x7 + 
        4*x1*x2*x3*x4*x8*x9 - 8*x1*x2*x3*x5*x6*x7*x8*x9 + 8*x1*x2*x3*x5*x6*x7*x8 + 
        8*x1*x2*x3*x5*x6*x7*x9 - 4*x1*x2*x3*x5*x6*x7 - 4*x1*x2*x3*x5*x6*x9 + 
        8*x1*x2*x3*x5*x7*x8*x9 - 4*x1*x2*x3*x5*x7*x8 - 4*x1*x2*x3*x5*x7*x9 - 4*x1*x2*x3*x5*x8*x9
        + 4*x1*x2*x3*x5*x9 - 4*x1*x2*x3*x6*x7*x8 - 4*x1*x2*x3*x6*x7*x9 + 4*x1*x2*x3*x6*x7 + 
        4*x1*x2*x3*x6*x8*x9 - 4*x1*x2*x3*x7*x8*x9 + 4*x1*x2*x3*x7*x8 + 4*x1*x2*x3*x7*x9 - 
        2*x1*x2*x3*x7 - 2*x1*x2*x3*x9 + 8*x1*x2*x4*x5*x6*x7*x9 - 4*x1*x2*x4*x5*x6*x7 - 
        8*x1*x2*x4*x5*x6*x8*x9 + 4*x1*x2*x4*x5*x6*x8 + 8*x1*x2*x4*x5*x7*x8*x9 - 
        4*x1*x2*x4*x5*x7*x8 - 8*x1*x2*x4*x5*x7*x9 + 4*x1*x2*x4*x5*x7 + 4*x1*x2*x4*x5*x9 - 
        2*x1*x2*x4*x5 - 4*x1*x2*x4*x6*x7*x9 + 4*x1*x2*x4*x6*x7 + 4*x1*x2*x4*x6*x8*x9 - 
        2*x1*x2*x4*x6 - 4*x1*x2*x4*x7*x8*x9 + 4*x1*x2*x4*x7*x8 + 4*x1*x2*x4*x7*x9 - 4*x1*x2*x4*x7
        - 2*x1*x2*x4*x8 - 2*x1*x2*x4*x9 + 2*x1*x2*x4 + 8*x1*x2*x5*x6*x7*x8*x9 - 
        4*x1*x2*x5*x6*x7*x8 - 8*x1*x2*x5*x6*x7*x9 + 4*x1*x2*x5*x6*x7 + 4*x1*x2*x5*x6*x9 - 
        2*x1*x2*x5*x6 - 8*x1*x2*x5*x7*x8*x9 + 4*x1*x2*x5*x7*x8 + 4*x1*x2*x5*x7*x9 - 2*x1*x2*x5*x7
        + 4*x1*x2*x5*x8*x9 - 2*x1*x2*x5*x8 - 4*x1*x2*x5*x9 + 2*x1*x2*x5 - 4*x1*x2*x6*x7*x8*x9 + 
        4*x1*x2*x6*x7*x8 + 4*x1*x2*x6*x7*x9 - 4*x1*x2*x6*x7 - 2*x1*x2*x6*x8 - 2*x1*x2*x6*x9 + 
        2*x1*x2*x6 + 4*x1*x2*x7*x8*x9 - 4*x1*x2*x7*x8 - 2*x1*x2*x7*x9 + 2*x1*x2*x7 - 2*x1*x2*x8*x9
        + 2*x1*x2*x8 + 2*x1*x2*x9 - x1*x2 - 8*x1*x3*x4*x5*x6*x7*x8*x9 + 8*x1*x3*x4*x5*x6*x7*x8 + 
        8*x1*x3*x4*x5*x6*x7*x9 - 4*x1*x3*x4*x5*x6*x7 - 4*x1*x3*x4*x5*x6*x8 + 
        8*x1*x3*x4*x5*x7*x8*x9 - 4*x1*x3*x4*x5*x7*x8 - 4*x1*x3*x4*x5*x7*x9 + 2*x1*x3*x4*x5 - 
        4*x1*x3*x4*x6*x7*x8 - 4*x1*x3*x4*x6*x7*x9 + 4*x1*x3*x4*x6*x7 + 4*x1*x3*x4*x6*x8 - 
        2*x1*x3*x4*x6 - 4*x1*x3*x4*x7*x8*x9 + 4*x1*x3*x4*x7*x8 + 4*x1*x3*x4*x7*x9 - 2*x1*x3*x4*x7
        - 2*x1*x3*x4*x8 + 8*x1*x3*x5*x6*x7*x8*x9 - 8*x1*x3*x5*x6*x7*x8 - 8*x1*x3*x5*x6*x7*x9 + 
        4*x1*x3*x5*x6*x7 - 4*x1*x3*x5*x6*x8*x9 + 4*x1*x3*x5*x6*x8 + 4*x1*x3*x5*x6*x9 - 
        2*x1*x3*x5*x6 - 8*x1*x3*x5*x7*x8*x9 + 4*x1*x3*x5*x7*x8 + 4*x1*x3*x5*x7*x9 + 
        4*x1*x3*x5*x8*x9 - 2*x1*x3*x5*x8 - 2*x1*x3*x5*x9 + 4*x1*x3*x6*x7*x8 + 4*x1*x3*x6*x7*x9 - 
        4*x1*x3*x6*x7 - 2*x1*x3*x6*x8 - 2*x1*x3*x6*x9 + 2*x1*x3*x6 + 4*x1*x3*x7*x8*x9 - 
        4*x1*x3*x7*x8 - 4*x1*x3*x7*x9 + 2*x1*x3*x7 - 2*x1*x3*x8*x9 + 2*x1*x3*x8 + 2*x1*x3*x9 - 
        x1*x3 - 4*x1*x4*x5*x6*x7*x9 + 4*x1*x4*x5*x6*x8*x9 - 4*x1*x4*x5*x7*x8*x9 + 
        4*x1*x4*x5*x7*x9 - 2*x1*x4*x5*x9 + 4*x1*x4*x6*x7*x9 - 2*x1*x4*x6*x7 - 2*x1*x4*x6*x8 - 
        2*x1*x4*x6*x9 + 2*x1*x4*x6 + 4*x1*x4*x7*x8*x9 - 2*x1*x4*x7*x8 - 4*x1*x4*x7*x9 + 2*x1*x4*x7
        - 2*x1*x4*x8*x9 + 2*x1*x4*x8 + 2*x1*x4*x9 - x1*x4 - 4*x1*x5*x6*x7*x8*x9 + 4*x1*x5*x6*x7*x8
        + 4*x1*x5*x6*x7*x9 - 2*x1*x5*x6*x7 - 2*x1*x5*x6*x8 - 2*x1*x5*x6*x9 + 2*x1*x5*x6 + 
        4*x1*x5*x7*x8*x9 - 2*x1*x5*x7*x8 - 2*x1*x5*x7*x9 - 2*x1*x5*x8*x9 + 2*x1*x5*x8 + 2*x1*x5*x9
        - x1*x5 - 2*x1*x6*x7*x8 - 2*x1*x6*x7*x9 + 2*x1*x6*x7 + 2*x1*x6*x8 + 2*x1*x6*x9 - 2*x1*x6 - 
        2*x1*x7*x8*x9 + 2*x1*x7*x8 + 2*x1*x7*x9 - x1*x7 + 2*x1*x8*x9 - 2*x1*x8 - 2*x1*x9 + x1 + 
        8*x2*x3*x4*x5*x7*x8*x9 - 4*x2*x3*x4*x5*x7*x8 - 4*x2*x3*x4*x5*x7*x9 - 4*x2*x3*x4*x5*x8*x9
        + 4*x2*x3*x4*x5*x9 - 4*x2*x3*x4*x7*x8*x9 + 4*x2*x3*x4*x7*x8 + 4*x2*x3*x4*x7*x9 - 
        2*x2*x3*x4*x7 - 2*x2*x3*x4*x9 - 4*x2*x3*x5*x7*x8*x9 + 2*x2*x3*x5*x7 + 4*x2*x3*x5*x8*x9 - 
        2*x2*x3*x5*x9 + 4*x2*x3*x7*x8*x9 - 2*x2*x3*x7*x8 - 2*x2*x3*x7*x9 - 2*x2*x3*x8*x9 + 
        2*x2*x3*x9 + 4*x2*x4*x5*x6*x7*x8 - 4*x2*x4*x5*x6*x7*x9 + 4*x2*x4*x5*x6*x8*x9 - 
        4*x2*x4*x5*x6*x8 - 4*x2*x4*x5*x7*x8*x9 + 4*x2*x4*x5*x7*x9 + 2*x2*x4*x5*x8 - 2*x2*x4*x5*x9
        - 4*x2*x4*x6*x7*x8*x9 + 4*x2*x4*x6*x7*x9 - 2*x2*x4*x6*x7 - 2*x2*x4*x6*x9 + 2*x2*x4*x6 + 
        4*x2*x4*x7*x8*x9 - 2*x2*x4*x7*x8 - 4*x2*x4*x7*x9 + 2*x2*x4*x7 + 2*x2*x4*x9 - x2*x4 - 
        4*x2*x5*x6*x7*x8*x9 + 4*x2*x5*x6*x7*x9 - 2*x2*x5*x6*x7 - 2*x2*x5*x6*x9 + 2*x2*x5*x6 + 
        4*x2*x5*x7*x8*x9 - 2*x2*x5*x7*x9 - 2*x2*x5*x8*x9 + 2*x2*x5*x9 - x2*x5 + 4*x2*x6*x7*x8*x9 -
        2*x2*x6*x7*x8 - 2*x2*x6*x7*x9 + 2*x2*x6*x7 - 2*x2*x6*x8*x9 + 2*x2*x6*x8 + 2*x2*x6*x9 - 
        2*x2*x6 - 4*x2*x7*x8*x9 + 2*x2*x7*x8 + 2*x2*x7*x9 - x2*x7 + 2*x2*x8*x9 - x2*x8 - 2*x2*x9 + 
        x2 - 4*x3*x4*x5*x7*x8*x9 + 2*x3*x4*x5*x7 + 2*x3*x4*x5*x8 - 2*x3*x4*x5 + 4*x3*x4*x7*x8*x9 -
        2*x3*x4*x7*x8 - 2*x3*x4*x7*x9 + x3*x4 + 4*x3*x5*x7*x8*x9 - 2*x3*x5*x7 - 2*x3*x5*x8*x9 + 
        x3*x5 - 4*x3*x7*x8*x9 + 2*x3*x7*x8 + 2*x3*x7*x9 + 2*x3*x8*x9 - x3*x8 - x3*x9 + 
        4*x4*x5*x6*x7*x8*x9 - 4*x4*x5*x6*x7*x8 + 2*x4*x5*x6*x7 - 4*x4*x5*x6*x8*x9 + 2*x4*x5*x6*x8
        + 2*x4*x5*x7*x8 - 2*x4*x5*x7 + 2*x4*x5*x8*x9 - 2*x4*x5*x8 + x4*x5 + 2*x4*x6*x7*x8 - 
        2*x4*x6*x7*x9 + 2*x4*x6*x9 - x4*x6 - 2*x4*x7*x8*x9 + 2*x4*x7*x9 - x4*x9 + 2*x5*x6*x8*x9 - 
        x5*x6 - 2*x5*x7*x8*x9 + x5*x7 - x6*x8 - x6*x9 + x6 + 2*x7*x8*x9 - x7*x8 - x7*x9 - x8*x9 + 
        x8 + x9
        ]
    sol = [ [0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 0, 1, 0, 1, 0, 1, 1] ]
    return p, sol
