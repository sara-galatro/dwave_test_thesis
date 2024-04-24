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

def example_4_anf(x1, x2, x3, x4):
    p = [
        x1*x2 + x1*x3 + x1*x4 + x1 + x2*x3 + x2*x4 + x2 + x3*x4 + x4,
        x1*x2 + x1*x3 + x2 + x3*x4 + x3,
        x1*x2 + x1*x3 + x2*x3 + x2 + x3 + x4,
        x1*x3 + x2*x4 + x4 + 1
        ]
    sol = [ [1, 0, 1, 0], [0, 0, 1, 1] ]
    return p, sol


def example_5_anf(x1, x2, x3, x4, x5):
    p = [
        x1*x2 + x1*x3 + x1*x5 + x2*x3 + x2*x4 + x2*x5 + x4*x5 + x4 + 1,
        x1*x3 + x1*x4 + x1 + x2*x3 + x2*x5 + x3*x4 + x3*x5 + x4*x5 + x4,
        x1*x2 + x1*x3 + x1*x4 + x1*x5 + x2*x3 + x2*x4 + x3*x4 + x4,
        x1*x3 + x1*x4 + x2*x3 + x2*x4 + x2*x5 + x2 + x3*x4 + x3*x5 + x3 + x4*x5 + x4 + x5 + 1,
        x1*x4 + x1*x5 + x1 + x2*x4 + x2 + x3*x5 + x3 + x4*x5 + x5
        ]
    sol = [ 1, 1, 0, 0, 1 ]
    return p, sol


def example_6_anf(x1, x2, x3, x4, x5, x6):
    p = [
        x1*x4 + x2*x6 + x3*x4 + x4*x6 + x6 + 1,
        x1*x3 + x1*x4 + x2*x3 + x2*x4 + x2*x5 + x3*x5 + x3*x6 + x3 + 
            x4 + x5 + x6 + 1,
        x1*x3 + x1*x6 + x2*x4 + x2*x5 + x2*x6 + x2 + x3*x5 + x3*x6 + 
            x4 + x5*x6 + x5 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x5 + x1 + x2*x4 + x3*x4 + x4*x6 + 
            x4 + 1,
        x1*x5 + x1*x6 + x2 + x3*x4 + x3 + x4*x6 + x4 + x5*x6 + x5,
        x1*x4 + x1*x5 + x1*x6 + x1 + x2*x3 + x2*x5 + x2*x6 + x2 + 
            x3*x5 + x3*x6 + x3 + x4 + x5*x6 + x5 + x6
        ]
    sol = [ 1, 1, 0, 1, 0, 0 ]
    return p, sol


def example_8_anf(x1, x2, x3, x4, x5, x6, x7, x8):
    p = [
        x1*x3 + x1*x7 + x1*x8 + x2*x4 + x2*x7 + x3*x4 + x3*x5 + 
            x3*x7 + x3*x8 + x3 + x4*x5 + x4*x7 + x4 + x5*x6 + x5*x7 + 
            x6*x7 + x6*x8 + x6 + x7*x8 + x8,
        x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x7 + x1 + x2*x3 + x2*x8 + 
            x3*x7 + x3*x8 + x4*x6 + x4*x7 + x5*x6 + x5 + x6*x8 + x6 + 
            1,
        x1*x4 + x1*x8 + x1 + x2*x3 + x2*x8 + x2 + x3*x6 + x4*x6 + 
            x4*x7 + x5*x7 + x6*x8 + x6 + x7 + x8,
        x1*x2 + x1*x5 + x1*x8 + x2*x3 + x2*x5 + x2*x6 + x3*x6 + x3 + 
            x4*x6 + x4*x7 + x4*x8 + x5*x6 + x5*x7 + x6*x7 + x6 + 
            x7*x8 + x7 + x8,
        x1*x4 + x1*x5 + x1*x6 + x1 + x2*x3 + x2*x4 + x2*x5 + x2*x6 + 
            x3*x5 + x3*x6 + x3*x7 + x3*x8 + x3 + x4*x5 + x4*x6 + 
            x4*x7 + x4*x8 + x4 + x5*x6 + x5*x7 + x5*x8 + x5 + x6 + 1,
        x1*x2 + x1*x7 + x1 + x2*x4 + x2*x5 + x2*x6 + x2 + x3*x4 + 
            x3*x5 + x3*x7 + x3*x8 + x3 + x4*x5 + x4*x6 + x4*x7 + 
            x4*x8 + x5*x7 + x5*x8 + x6*x7 + x6 + x8,
        x1*x2 + x1*x3 + x1*x4 + x1 + x2*x4 + x2*x5 + x2*x6 + x2*x8 + 
            x3*x6 + x3*x8 + x4*x6 + x4*x7 + x4 + x5*x7 + x5*x8 + x5 + 
            x6 + x7*x8 + x8,
        x1*x2 + x1*x3 + x1*x4 + x1*x6 + x1*x8 + x1 + x2*x3 + x2*x4 + 
            x2*x5 + x2 + x3*x6 + x3 + x4*x5 + x4*x6 + x5*x6 + x5*x8 + 
            x6*x8 + x7 + 1
        ]
    sol = [ 0, 1, 0, 0, 1, 1, 0, 0 ]
    return p, sol


def example_10_anf(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    p = [
        x1*x4 + x1*x6 + x1*x9 + x2*x4 + x2*x6 + x2*x8 + x2*x9 + 
            x2*x10 + x3*x7 + x4*x5 + x4 + x5*x7 + x5*x8 + x5*x10 + 
            x6*x7 + x6*x8 + x7*x9 + x8 + x10,
        x1*x2 + x1*x4 + x2*x3 + x2*x5 + x2*x6 + x2*x10 + x2 + x3*x4 +
            x3*x5 + x3*x7 + x3*x8 + x3*x9 + x3*x10 + x3 + x4*x6 + 
            x4*x8 + x4*x10 + x4 + x5*x6 + x5*x7 + x5*x8 + x5*x9 + 
            x6*x10 + x7*x8 + x7*x9 + x7 + x8*x9 + x8 + x9*x10 + x9 + 1,
        x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x9 + x2*x3 + x2*x4 + 
            x2*x6 + x3*x4 + x3*x5 + x3*x6 + x3*x8 + x3*x9 + x3*x10 + 
            x3 + x4*x5 + x4*x7 + x4 + x5*x7 + x5*x9 + x5 + x6*x8 + 
            x6*x9 + x6*x10 + x6 + x7*x8 + x8*x9 + x8 + x9*x10 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x6 + x1*x7 + x1*x9 + x1 + x2*x4 + 
            x2*x5 + x2*x7 + x2*x9 + x2*x10 + x2 + x3*x4 + x3*x5 + 
            x3*x6 + x3*x8 + x3*x9 + x3 + x4*x6 + x4*x7 + x4*x9 + x4 + 
            x5*x9 + x5 + x6*x7 + x6*x8 + x6 + x7*x8 + x7*x10 + x7 + 
            x8*x9 + x8*x10 + x9,
        x1*x3 + x1*x4 + x2*x4 + x2*x5 + x2*x7 + x2*x8 + x2*x9 + 
            x3*x4 + x3*x6 + x3*x8 + x3*x10 + x4*x6 + x4*x9 + x4 + 
            x5*x6 + x5*x10 + x5 + x6*x9 + x6*x10 + x6 + x7*x8 + x7*x9 
            + x7*x10 + x7 + x8 + x9*x10 + 1,
        x1*x2 + x1*x4 + x1*x6 + x1*x7 + x1*x8 + x2*x5 + x2*x6 + 
            x2*x7 + x2*x8 + x2*x9 + x2 + x3*x6 + x3*x8 + x3*x9 + 
            x3*x10 + x4*x6 + x4*x7 + x4*x9 + x4*x10 + x4 + x5*x10 + x5
            + x6*x8 + x6*x10 + x7*x9 + x7 + x8*x10 + x9*x10 + x9 + 1,
        x1*x2 + x1*x4 + x1*x9 + x1*x10 + x1 + x2*x4 + x2*x6 + x2*x8 +
            x2*x10 + x3*x4 + x3*x8 + x3*x9 + x3*x10 + x4*x6 + x4*x9 + 
            x5*x6 + x5*x8 + x5*x9 + x6*x8 + x6 + x7*x8 + x7*x9 + 
            x8*x10 + x8 + x9*x10 + x10,
        x1*x3 + x1*x4 + x1*x8 + x1*x9 + x2*x5 + x2*x6 + x2*x7 + 
            x2*x8 + x2*x9 + x2 + x3*x4 + x3*x5 + x3*x6 + x3*x10 + x3 +
            x4*x6 + x4*x7 + x4*x8 + x4 + x5*x8 + x5*x9 + x5*x10 + x5 +
            x6*x7 + x6*x8 + x6*x9 + x7*x8 + x7*x9 + x7*x10 + x8*x9 + 
            x8*x10 + x8 + x9,
        x1*x2 + x1*x9 + x1*x10 + x1 + x2*x3 + x2*x8 + x2 + x3*x5 + 
            x3*x6 + x3*x8 + x3*x9 + x3 + x4*x5 + x4*x6 + x4*x9 + 
            x4*x10 + x4 + x5*x6 + x5*x8 + x5*x10 + x6*x7 + x6*x8 + 
            x6*x10 + x6 + x7*x8 + x7*x9 + x7*x10 + x8*x9 + x8*x10 + x8
            + x9 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x8 + x2*x6 + 
            x2*x9 + x2*x10 + x2 + x3*x4 + x3*x5 + x3*x7 + x3*x8 + 
            x3*x10 + x4*x9 + x4*x10 + x5*x6 + x5*x7 + x5*x9 + x5*x10 +
            x5 + x6*x8 + x6*x9 + x7*x8 + x7*x10 + x8*x10 + x8 + x9*x10
            + x9 + 1
        ]
    sol = [ 0, 1, 1, 1, 0, 1, 1, 1, 0, 1 ]
    return p, sol


def example_12_anf(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    p = [
        x1*x3 + x1*x5 + x1*x8 + x1*x9 + x1*x10 + x1*x12 + x2*x7 + 
            x2*x8 + x2*x9 + x2*x10 + x2*x12 + x3*x4 + x3*x7 + x3*x10 +
            x3*x11 + x3*x12 + x4*x7 + x4*x12 + x5*x7 + x5*x10 + x5 + 
            x6*x8 + x6*x10 + x6*x11 + x6*x12 + x7*x9 + x7*x12 + x8*x9 
            + x8*x11 + x8*x12 + x8 + x9*x11 + x9*x12 + x10*x12 + x10 + 
            x11 + x12,
        x1*x4 + x1*x6 + x1*x7 + x1*x9 + x1*x11 + x1 + x2*x7 + x3*x4 +
            x3*x5 + x3*x6 + x3*x7 + x3*x8 + x3*x10 + x3*x12 + x4*x9 + 
            x4*x10 + x4 + x5*x8 + x5*x10 + x6*x10 + x6 + x7*x10 + 
            x7*x11 + x9*x10 + x11*x12,
        x1*x3 + x1*x5 + x1*x6 + x1*x8 + x1*x11 + x1*x12 + x1 + x2*x3 
            + x2*x5 + x2*x6 + x2*x7 + x2*x8 + x2*x10 + x2*x11 + x2*x12
            + x3*x5 + x3*x6 + x3*x8 + x3*x12 + x3 + x4*x6 + x4 + x5*x9
            + x5*x10 + x5*x11 + x5*x12 + x6*x7 + x6*x9 + x6*x11 + 
            x7*x9 + x7*x11 + x8*x10 + x8*x12 + x8 + x9*x11 + x10*x11 + 
            1,
        x1*x2 + x1*x4 + x1*x9 + x1*x10 + x1*x11 + x1*x12 + x1 + x2*x5
            + x2*x6 + x2*x7 + x2*x9 + x2*x10 + x2*x11 + x2*x12 + x2 + 
            x3*x4 + x3*x6 + x3*x7 + x3*x8 + x3*x10 + x3*x11 + x3 + 
            x4*x9 + x4*x10 + x4*x12 + x5*x7 + x5*x9 + x5*x11 + x5*x12 
            + x5 + x6*x7 + x6*x11 + x6*x12 + x6 + x7*x9 + x7*x10 + 
            x7*x11 + x7 + x8*x10 + x8*x12 + x8 + x9*x10 + x9*x12 + 
            x11*x12 + x12 + 1,
        x1*x2 + x1*x4 + x1*x7 + x1*x8 + x1*x9 + x1*x11 + x1*x12 + x1 
            + x2*x4 + x2*x5 + x2*x7 + x2*x8 + x2*x9 + x3*x12 + x4*x5 +
            x4 + x5*x6 + x5*x7 + x5*x9 + x5*x10 + x5*x12 + x6*x8 + 
            x6*x10 + x6*x11 + x6*x12 + x6 + x7*x8 + x7*x9 + x7*x10 + 
            x7*x12 + x7 + x8*x9 + x8*x11 + x9*x10 + x9 + x10*x11 + 
            x10*x12 + x10 + x11*x12 + x12 + 1,
        x1*x2 + x1*x5 + x1*x6 + x1*x7 + x1*x8 + x1*x9 + x1*x10 + 
            x1*x11 + x1*x12 + x2*x3 + x2*x6 + x2*x7 + x2*x9 + x2*x12 +
            x3*x4 + x3*x5 + x3*x9 + x3*x10 + x4*x7 + x4*x8 + x4*x9 + 
            x4*x10 + x4*x12 + x4 + x5*x7 + x5*x8 + x5*x10 + x5*x11 + 
            x5 + x6*x8 + x6*x9 + x6*x12 + x7*x8 + x7*x9 + x7 + x8*x12 
            + x9*x12 + x9 + x10 + x11*x12 + x11 + x12 + 1,
        x1*x2 + x1*x3 + x1*x5 + x1*x8 + x1*x9 + x1*x11 + x1*x12 + 
            x2*x5 + x2*x6 + x2*x8 + x2*x9 + x2*x10 + x2*x11 + x3*x5 + 
            x3*x8 + x3*x9 + x3*x12 + x4*x6 + x4*x7 + x4*x11 + x5*x10 +
            x6*x8 + x6*x10 + x6*x12 + x7 + x8*x10 + x8*x11 + x8*x12 + 
            x8 + x9*x11 + x9 + x10 + x11*x12 + x11 + 1,
        x1*x2 + x1*x5 + x1*x8 + x1*x9 + x1*x12 + x2*x3 + x2*x5 + 
            x2*x6 + x2*x8 + x2*x9 + x2*x11 + x2*x12 + x2 + x3*x4 + 
            x3*x5 + x3*x6 + x3*x7 + x3*x11 + x4*x8 + x4*x9 + x4*x10 + 
            x4*x11 + x5*x7 + x5*x8 + x5*x9 + x5 + x6*x7 + x6*x8 + 
            x6*x12 + x6 + x7*x8 + x8*x10 + x8*x11 + x8*x12 + x9*x10 + 
            x10*x11 + x10*x12 + x10 + x11*x12 + x11 + x12 + 1,
        x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x7 + x1*x8 + x1*x9 + 
            x1*x10 + x1*x11 + x1*x12 + x2*x8 + x2*x9 + x2*x12 + x3*x4 
            + x3*x5 + x3*x6 + x3*x10 + x4*x10 + x4*x11 + x4 + x5*x6 + 
            x5*x7 + x5*x8 + x5*x9 + x6*x9 + x6*x11 + x6*x12 + x6 + 
            x7*x8 + x8*x9 + x8*x10 + x10*x12 + x10,
        x1*x2 + x1*x3 + x1*x4 + x1*x7 + x1*x9 + x1 + x2*x3 + x2*x4 + 
            x2*x8 + x2*x9 + x2*x10 + x2*x11 + x2*x12 + x3*x4 + x3*x10 
            + x3*x11 + x3*x12 + x3 + x4*x5 + x4*x7 + x4*x9 + x4*x11 + 
            x4*x12 + x5*x6 + x5*x7 + x5*x11 + x6*x8 + x6*x9 + x6*x12 +
            x6 + x7*x8 + x7*x9 + x7 + x8*x9 + x8*x10 + x8*x11 + x8*x12
            + x8 + x9*x10 + x10*x11 + x10 + x11*x12 + x11 + 1,
        x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x7 + x1*x9 + x1*x11 + x1 +
            x2*x5 + x2*x8 + x2*x9 + x3*x4 + x3*x7 + x3*x8 + x3*x9 + 
            x3*x10 + x3 + x4*x5 + x4*x6 + x4*x9 + x4*x10 + x4*x11 + 
            x5*x7 + x5*x8 + x5*x9 + x5*x10 + x5*x11 + x5*x12 + x6*x9 +
            x6*x10 + x6*x11 + x6*x12 + x7*x10 + x7*x12 + x7 + x8*x9 + 
            x9*x10 + x11*x12 + x12 + 1,
        x1*x2 + x1*x3 + x1*x6 + x1*x8 + x1*x10 + x1*x12 + x2*x5 + 
            x2*x6 + x2*x9 + x2*x12 + x2 + x3*x4 + x3*x5 + x3*x6 + 
            x3*x7 + x3*x9 + x3 + x4*x7 + x4*x8 + x4*x10 + x4*x11 + 
            x5*x7 + x5*x8 + x5*x10 + x5 + x6*x7 + x6*x8 + x6*x10 + x6 
            + x7*x8 + x7*x10 + x7 + x8*x10 + x8*x11 + x8*x12 + x9*x11 +
            x10*x12 + x11 + 1
        ]
    sol = [ [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1] ]
    return p, sol


def example_14_anf(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14):
    p = [
        x1*x3 + x1*x4 + x1*x5 + x1*x9 + x1*x10 + x1*x11 + x1*x12 + 
            x1*x13 + x1 + x2*x3 + x2*x5 + x2*x7 + x2*x10 + x2*x12 + x2
            + x3*x5 + x3*x9 + x3*x12 + x3*x13 + x3*x14 + x4*x5 + x4*x7
            + x4*x8 + x4*x9 + x4*x10 + x5*x10 + x5*x11 + x5*x13 + x5 + 
            x6*x7 + x6*x11 + x6*x14 + x7*x8 + x7*x9 + x7*x10 + x7*x14 
            + x7 + x8*x13 + x8*x14 + x8 + x9*x10 + x9*x13 + x9*x14 + 
            x10*x12 + x10*x13 + x10*x14 + x11*x12 + x11*x13 + x12*x14 + 
            x13*x14 + x13 + x14,
        x1*x6 + x1*x7 + x1*x9 + x1*x13 + x2*x5 + x2*x8 + x2*x9 + 
            x2*x10 + x2*x12 + x2*x13 + x3*x4 + x3*x5 + x3*x8 + x3*x12 
            + x4*x5 + x4*x7 + x4*x8 + x4*x9 + x4*x10 + x4*x11 + x4*x13
            + x4 + x5*x7 + x5*x10 + x5*x11 + x5*x12 + x5*x14 + x5 + 
            x6*x9 + x6*x11 + x6*x13 + x7*x9 + x7*x11 + x7*x12 + x7*x14
            + x7 + x8*x9 + x8*x10 + x8*x12 + x8*x14 + x9*x10 + x9*x11 +
            x9*x12 + x9*x13 + x9 + x10*x11 + x10*x12 + x10*x14 + 
            x11*x12 + x11*x13 + x11*x14 + 1,
        x1*x2 + x1*x6 + x1*x7 + x1*x8 + x1*x12 + x1*x13 + x1*x14 + x1
            + x2*x3 + x2*x4 + x2*x10 + x2*x14 + x3*x7 + x3*x8 + x3*x12
            + x3*x13 + x3*x14 + x4*x5 + x4*x7 + x4*x8 + x4*x10 + x4 + 
            x5*x6 + x5*x8 + x5*x9 + x5*x11 + x5*x13 + x5 + x6*x7 + 
            x6*x10 + x6*x12 + x6*x13 + x6*x14 + x7*x8 + x7*x12 + 
            x7*x14 + x7 + x8*x11 + x8*x12 + x8 + x9*x10 + x9*x11 + 
            x9*x12 + x9*x13 + x9 + x10*x11 + x10*x12 + x10 + x11*x13 + 
            x11 + x12*x14 + x14,
        x1*x2 + x1*x3 + x1*x4 + x1*x9 + x1*x11 + x1*x12 + x1*x13 + x1
            + x2*x3 + x2*x8 + x2*x10 + x2*x11 + x2*x12 + x2 + x3*x4 + 
            x3*x5 + x3*x6 + x3*x8 + x3*x10 + x3*x11 + x3*x12 + x3 + 
            x4*x6 + x4*x8 + x4*x9 + x4*x12 + x4*x14 + x4 + x5*x7 + 
            x5*x8 + x5*x9 + x5*x12 + x5*x13 + x6*x7 + x6*x8 + x6*x12 +
            x6*x13 + x6 + x7*x8 + x7*x13 + x7*x14 + x7 + x8*x9 + 
            x8*x10 + x8*x12 + x8*x13 + x8 + x9*x11 + x10*x11 + x10*x12 
            + x10*x13 + x10*x14 + x10 + x11 + x12*x13 + x13*x14 + x13,
        x1*x2 + x1*x3 + x1*x4 + x1*x6 + x1*x10 + x1*x11 + x1*x12 + 
            x1*x13 + x1*x14 + x2*x4 + x2*x5 + x2*x6 + x2*x10 + x2*x13 
            + x3*x4 + x3*x5 + x3*x7 + x3*x8 + x3*x10 + x3*x12 + x3*x13
            + x3*x14 + x4*x7 + x4*x8 + x4*x9 + x4*x10 + x4*x12 + 
            x4*x14 + x5*x7 + x5*x8 + x5*x9 + x5*x10 + x5*x11 + x5*x12 
            + x5*x13 + x5 + x6*x9 + x6*x10 + x6*x11 + x6*x12 + x6 + 
            x7*x10 + x7*x11 + x7*x12 + x7*x13 + x8*x9 + x8*x10 + 
            x8*x11 + x8*x13 + x8*x14 + x9*x11 + x9*x12 + x10*x12 + 
            x10*x13 + x10*x14 + x11*x12 + x11*x14 + x11 + x12*x13 + 
            x13*x14 + x14,
        x1*x2 + x1*x3 + x1*x5 + x1*x7 + x1*x9 + x1*x12 + x1*x13 + 
            x2*x5 + x2*x7 + x2*x10 + x2*x12 + x2*x13 + x2*x14 + x2 + 
            x3*x5 + x3*x6 + x3*x9 + x3*x10 + x3*x13 + x4*x5 + x4*x7 + 
            x4*x11 + x4*x13 + x4*x14 + x4 + x5*x7 + x5*x8 + x5*x9 + 
            x5*x12 + x5*x14 + x6*x12 + x6*x13 + x6*x14 + x7*x10 + 
            x7*x12 + x8*x11 + x8*x13 + x8*x14 + x9*x10 + x9*x13 + x9 + 
            x10*x11 + x10*x13 + x10*x14 + x11*x12 + x11 + x12*x13 + x13,
        x1*x3 + x1*x7 + x1*x8 + x1*x10 + x1*x14 + x2*x7 + x2*x9 + 
            x2*x12 + x3*x4 + x3*x5 + x3*x6 + x3*x7 + x3*x8 + x3*x13 + 
            x3*x14 + x3 + x4*x5 + x4*x7 + x4*x9 + x4*x11 + x4*x12 + 
            x4*x14 + x5*x6 + x5*x7 + x5*x8 + x5*x9 + x5*x10 + x5*x11 +
            x5*x12 + x5 + x6*x7 + x6*x9 + x6*x10 + x6*x12 + x6 + x7*x8
            + x7*x11 + x7*x13 + x7 + x8*x10 + x8*x11 + x8*x13 + x9*x11 
            + x9*x12 + x9*x13 + x9*x14 + x9 + x10*x11 + x10*x12 + 
            x10*x13 + x10*x14 + x11*x12 + x11 + x12*x14 + x12 + x13*x14 
            + x13,
        x1*x2 + x1*x4 + x1*x7 + x1*x11 + x1*x13 + x1 + x2*x6 + x2*x7 
            + x2*x8 + x2*x10 + x2*x12 + x2*x13 + x3*x5 + x3*x6 + x3*x7
            + x3*x9 + x3*x10 + x3*x11 + x3*x12 + x3*x13 + x4*x10 + 
            x4*x14 + x5*x6 + x5*x10 + x6*x7 + x6*x8 + x6*x11 + x6*x13 
            + x6*x14 + x6 + x7*x10 + x7*x11 + x7*x12 + x7*x13 + x7 + 
            x8*x9 + x8*x11 + x8*x12 + x8*x14 + x9*x11 + x10*x12 + 
            x10*x14 + x10 + x12*x13 + x12*x14 + x13*x14 + 1,
        x1*x2 + x1*x8 + x1*x9 + x1*x10 + x1 + x2*x4 + x2*x5 + x2*x8 +
            x2*x10 + x2*x11 + x2*x14 + x3*x4 + x3*x6 + x3*x7 + x3*x8 +
            x3*x12 + x3*x13 + x3 + x4*x5 + x4*x6 + x4*x8 + x4*x9 + 
            x4*x12 + x4*x13 + x4*x14 + x5*x11 + x5 + x6*x8 + x6*x9 + 
            x6*x11 + x6*x13 + x6 + x7*x8 + x7*x10 + x7*x14 + x7 + 
            x8*x12 + x8*x14 + x9*x12 + x9*x14 + x9 + x10*x11 + x11*x12 
            + x11*x13 + x11*x14 + x12*x14 + x12 + x13*x14 + x13,
        x1*x4 + x1*x6 + x1*x11 + x1*x12 + x1*x13 + x1*x14 + x2*x4 + 
            x2*x5 + x2*x12 + x2*x13 + x3*x6 + x3*x8 + x3*x9 + x3*x11 +
            x3*x12 + x3 + x4*x6 + x4*x7 + x4*x11 + x4 + x5*x8 + x5*x11
            + x6*x10 + x6*x11 + x6*x12 + x6*x13 + x6 + x7*x9 + x7*x10 +
            x7*x11 + x7*x12 + x8*x9 + x8*x11 + x8*x14 + x8 + x9*x12 + 
            x9*x13 + x9 + x10*x13 + x10*x14 + x10 + x11*x12 + x11*x13 + 
            x12*x14 + x13*x14,
        x1*x2 + x1*x3 + x1*x6 + x1*x7 + x1*x10 + x1*x11 + x1*x13 + 
            x1*x14 + x1 + x2*x4 + x2*x5 + x2*x6 + x2*x8 + x2*x10 + 
            x2*x11 + x2*x13 + x3*x5 + x3*x6 + x3*x8 + x3*x10 + x3 + 
            x4*x5 + x4*x7 + x4*x11 + x4*x14 + x5*x8 + x5*x9 + x5*x10 +
            x5*x12 + x5*x13 + x5 + x6*x11 + x6*x14 + x7*x8 + x7*x9 + 
            x7*x10 + x7*x11 + x7*x13 + x7 + x8*x12 + x8*x14 + x8 + 
            x9*x11 + x9*x12 + x9*x13 + x9*x14 + x10*x12 + x10*x14 + x10
            + x11*x12 + x11*x13 + x11 + x12*x14 + x13*x14 + x13 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x7 + x1*x10 + x1*x12 + x1 + x2*x3 
            + x2*x4 + x2*x7 + x2*x9 + x2*x12 + x2*x13 + x2*x14 + x2 + 
            x3*x6 + x3*x7 + x3*x8 + x3*x9 + x3*x12 + x3*x14 + x4*x5 + 
            x4*x6 + x4*x9 + x4*x13 + x4 + x5*x6 + x5*x9 + x5*x10 + 
            x5*x11 + x5*x13 + x6*x13 + x6*x14 + x6 + x7*x9 + x7*x10 + 
            x7 + x8*x9 + x8*x10 + x8*x11 + x9*x10 + x10*x11 + x10*x14 +
            x10 + x11*x12 + x11*x14 + x11 + x12 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x5 + x1*x7 + x1*x8 + x1*x9 + 
            x1*x11 + x2*x3 + x2*x4 + x2*x5 + x2*x9 + x2*x10 + x2*x11 +
            x3*x6 + x3*x7 + x3*x8 + x3*x14 + x3 + x4*x5 + x4*x10 + 
            x4*x12 + x4 + x5*x6 + x5*x7 + x5*x8 + x5*x13 + x5*x14 + 
            x6*x9 + x6*x11 + x6*x12 + x6*x14 + x6 + x7*x8 + x7*x9 + 
            x7*x10 + x7*x11 + x7*x13 + x8*x10 + x8*x13 + x8*x14 + 
            x9*x10 + x9*x11 + x9*x13 + x9 + x10*x11 + x10*x12 + x10 + 
            x11*x14 + x11 + x12 + x13*x14 + x13,
        x1*x2 + x1*x3 + x1*x5 + x1*x10 + x1*x12 + x1*x14 + x2*x5 + 
            x2*x6 + x2*x7 + x2*x8 + x2*x9 + x2*x10 + x2*x13 + x2*x14 +
            x3*x4 + x3*x6 + x3*x10 + x3*x12 + x3*x13 + x3*x14 + x3 + 
            x4*x13 + x4*x14 + x4 + x5*x9 + x5*x10 + x5*x12 + x5*x14 + 
            x6*x7 + x6*x8 + x6*x11 + x6*x12 + x6 + x7*x11 + x7*x12 + 
            x7*x13 + x7*x14 + x7 + x8*x9 + x8*x10 + x8*x13 + x8*x14 + 
            x8 + x9*x11 + x9 + x10*x12 + x10*x13 + x10*x14 + x11*x13 + 
            x11 + x13 + x14 + 1
        ]
    sol = [ 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1 ]
    return p, sol


def example_16_anf(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16):
    p = [
        x1*x2 + x1*x3 + x1*x4 + x1*x7 + x1*x8 + x1*x10 + x1*x11 + 
            x1*x13 + x1*x14 + x2*x13 + x2*x14 + x2*x15 + x3*x4 + x3*x5
            + x3*x13 + x3*x14 + x3 + x4*x5 + x4*x6 + x4*x12 + x4*x16 + 
            x4 + x5*x8 + x5*x9 + x5*x10 + x5*x12 + x5*x14 + x5*x15 + 
            x5 + x6*x7 + x6*x12 + x6*x15 + x7*x11 + x7*x14 + x7*x15 + 
            x8*x9 + x8*x10 + x8*x15 + x8 + x9*x11 + x9*x13 + x9*x14 + 
            x9*x15 + x10*x12 + x10*x14 + x10*x15 + x10*x16 + x11*x12 + 
            x11*x13 + x11*x14 + x11 + x12*x13 + x12*x16 + x12 + x13*x14 
            + x13*x15 + x13*x16 + x14*x15 + x14*x16 + x14 + x16 + 1,
        x1*x3 + x1*x4 + x1*x5 + x1*x8 + x1*x11 + x1*x12 + x1*x13 + 
            x1*x14 + x1*x15 + x2*x5 + x2*x8 + x2*x12 + x2*x13 + x2*x16
            + x3*x4 + x3*x5 + x3*x7 + x3*x10 + x3*x11 + x3*x12 + 
            x3*x15 + x3 + x4*x5 + x4*x6 + x4*x7 + x4*x8 + x4*x9 + 
            x4*x11 + x4*x12 + x4*x14 + x4*x15 + x4*x16 + x4 + x5*x6 + 
            x5*x7 + x5*x8 + x5*x9 + x5*x12 + x5*x14 + x5*x16 + x6*x9 +
            x6*x13 + x6*x14 + x7*x9 + x7*x11 + x7*x13 + x7*x14 + 
            x7*x15 + x7*x16 + x8*x16 + x8 + x9*x11 + x9*x12 + x9*x13 + 
            x9*x14 + x9*x15 + x9*x16 + x10*x11 + x10*x14 + x10*x15 + 
            x10*x16 + x10 + x11*x12 + x11*x13 + x11*x14 + x11*x16 + 
            x12*x15 + x12 + x13*x14 + x13*x16 + x13 + x14*x15 + x14*x16 
            + x14 + x16 + 1,
        x1*x3 + x1*x4 + x1*x5 + x1*x11 + x1*x12 + x1*x14 + x1*x15 + 
            x2*x6 + x2*x7 + x2*x8 + x2*x9 + x2*x11 + x2*x13 + x2*x14 +
            x2*x16 + x3*x5 + x3*x8 + x3*x11 + x3*x13 + x3*x14 + x3*x16
            + x3 + x4*x9 + x4*x14 + x4 + x5*x8 + x5*x9 + x5*x16 + x5 + 
            x6*x8 + x6*x10 + x6*x11 + x6*x12 + x6*x13 + x6*x14 + x6 + 
            x7*x9 + x7*x11 + x7*x12 + x7*x16 + x8*x10 + x8*x12 + 
            x8*x13 + x8*x14 + x8*x15 + x8*x16 + x8 + x9*x10 + x9*x11 + 
            x9*x12 + x9*x13 + x9*x16 + x9 + x10*x11 + x10*x12 + x10*x13
            + x10*x14 + x10*x15 + x11*x12 + x11*x14 + x12*x14 + x12*x15 
            + x12*x16 + x12 + x13*x14 + x13*x16 + x14*x15 + x14*x16 + 
            x15 + 1,
        x1*x5 + x1*x6 + x1*x8 + x1*x14 + x1*x15 + x1*x16 + x2*x3 + 
            x2*x4 + x2*x7 + x2*x10 + x2*x11 + x2*x12 + x3*x5 + x3*x6 +
            x3*x12 + x3*x13 + x4*x7 + x4*x8 + x4*x9 + x4*x10 + x4*x12 
            + x4*x13 + x4*x14 + x5*x6 + x5*x7 + x5*x9 + x5*x10 + 
            x5*x11 + x5*x13 + x5*x14 + x5*x15 + x5 + x6*x7 + x6*x9 + 
            x6*x12 + x6*x13 + x6*x15 + x6*x16 + x6 + x7*x11 + x7*x12 + 
            x7*x14 + x7*x16 + x7 + x8*x9 + x8*x10 + x8*x11 + x8*x14 + 
            x8*x16 + x9*x11 + x9 + x10*x13 + x11*x13 + x11*x14 + 
            x11*x15 + x11 + x12*x13 + x12*x14 + x12*x15 + x12*x16 + x12 
            + x13*x14 + x13*x15 + x13 + x14*x15 + x14,
        x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x10 + x1*x12 + x1*x14 + x1
            + x2*x5 + x2*x6 + x2*x8 + x2*x9 + x2*x12 + x2*x14 + x2*x15
            + x2*x16 + x2 + x3*x5 + x3*x6 + x3*x8 + x3*x9 + x3*x10 + 
            x3*x11 + x3*x12 + x3*x13 + x3*x14 + x3 + x4*x5 + x4*x6 + 
            x4*x9 + x4*x11 + x4*x14 + x4*x15 + x4 + x5*x8 + x5*x12 + 
            x5*x13 + x5*x14 + x5*x16 + x5 + x6*x9 + x6*x10 + x6*x12 + 
            x6*x16 + x6 + x7*x9 + x7*x10 + x7*x11 + x7*x12 + x7*x13 + 
            x7*x14 + x7*x15 + x7 + x8*x9 + x8*x11 + x8*x14 + x8*x16 + 
            x8 + x9*x13 + x9*x16 + x10*x14 + x11*x13 + x11*x14 + 
            x11*x16 + x11 + x12*x14 + x13*x14 + x13 + x14*x16 + x15*x16 
            + 1,
        x1*x2 + x1*x5 + x1*x6 + x1*x9 + x1*x10 + x1*x12 + x1*x13 + 
            x1*x15 + x1*x16 + x2*x4 + x2*x5 + x2*x6 + x2*x7 + x2*x8 + 
            x2*x9 + x2*x12 + x2*x13 + x2*x15 + x2*x16 + x2 + x3*x4 + 
            x3*x5 + x3*x7 + x3*x8 + x3*x9 + x3*x11 + x3*x15 + x3*x16 +
            x4*x5 + x4*x7 + x4*x8 + x4*x9 + x4*x11 + x4*x15 + x4*x16 +
            x4 + x5*x8 + x5*x9 + x5*x11 + x5*x12 + x5*x13 + x5*x14 + 
            x5*x15 + x5*x16 + x5 + x6*x7 + x7*x8 + x7*x13 + x7*x14 + 
            x7*x16 + x8*x10 + x8*x11 + x8*x12 + x8*x13 + x8*x14 + 
            x8*x16 + x8 + x9*x10 + x9*x13 + x9*x15 + x9 + x10*x14 + 
            x11*x14 + x11*x15 + x11*x16 + x11 + x12*x13 + x12*x16 + 
            x13*x14 + x13*x15 + x13 + x14*x15 + x14*x16 + x15*x16 + x15 
            + 1,
        x1*x3 + x1*x5 + x1*x6 + x1*x15 + x2*x4 + x2*x6 + x2*x8 + 
            x2*x9 + x2*x11 + x2*x13 + x3*x4 + x3*x7 + x3*x8 + x3*x9 + 
            x3*x10 + x3*x12 + x3*x13 + x3 + x4*x5 + x4*x6 + x4*x7 + 
            x4*x9 + x4*x10 + x4*x14 + x4*x15 + x4*x16 + x5*x6 + x5*x7 
            + x5*x8 + x5*x10 + x5*x12 + x5*x13 + x5*x15 + x5 + x6*x11 +
            x6*x12 + x6*x13 + x6*x15 + x6*x16 + x6 + x7*x10 + x7*x11 + 
            x7*x12 + x7*x13 + x7*x15 + x7*x16 + x7 + x8*x9 + x8*x11 + 
            x8*x12 + x8*x14 + x8*x15 + x8*x16 + x9*x10 + x9*x12 + 
            x9*x13 + x9*x14 + x9 + x10*x13 + x10*x16 + x10 + x11*x12 + 
            x11*x15 + x11*x16 + x11 + x12*x13 + x13*x14 + x13 + x14*x15 
            + x14 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x6 + x1*x10 + x1*x12 + x1*x13 + 
            x1*x16 + x2*x3 + x2*x4 + x2*x7 + x2*x8 + x2*x9 + x2*x10 + 
            x2*x11 + x2*x12 + x2*x14 + x2*x15 + x2*x16 + x3*x5 + x3*x7
            + x3*x8 + x3*x9 + x3*x11 + x3*x12 + x3 + x4*x6 + x4*x7 + 
            x4*x10 + x4*x11 + x4*x12 + x4*x13 + x4*x14 + x4*x16 + x4 + 
            x5*x7 + x5*x11 + x5*x15 + x5*x16 + x5 + x6*x7 + x6*x8 + 
            x6*x11 + x6*x13 + x6*x15 + x6*x16 + x6 + x7*x8 + x7*x9 + 
            x7*x11 + x7*x13 + x7*x14 + x7*x16 + x7 + x8*x9 + x8*x10 + 
            x8*x12 + x8*x13 + x8*x15 + x8 + x9*x10 + x9*x11 + x9*x12 + 
            x9*x13 + x9*x16 + x9 + x10*x11 + x10*x12 + x10*x14 + 
            x10*x15 + x10 + x11*x12 + x11*x14 + x11*x16 + x12*x16 + 
            x13*x14 + x13*x16 + x13 + x14 + x15 + x16,
        x1*x5 + x1*x7 + x1*x9 + x1*x10 + x1*x14 + x2*x5 + x2*x8 + 
            x2*x10 + x2*x14 + x2*x15 + x3*x7 + x3*x8 + x3*x10 + x3*x13
            + x3*x14 + x3*x15 + x3 + x4*x5 + x4*x7 + x4*x8 + x4*x10 + 
            x4*x12 + x4*x13 + x4*x15 + x5*x9 + x5*x10 + x5*x13 + 
            x5*x15 + x5 + x6*x7 + x6*x8 + x6*x9 + x6*x10 + x6*x11 + 
            x6*x16 + x6 + x7*x8 + x7*x10 + x7*x12 + x7*x15 + x7*x16 + 
            x7 + x8*x9 + x8*x11 + x8*x12 + x8*x13 + x8*x14 + x9*x14 + 
            x9*x16 + x10*x15 + x11*x12 + x11*x14 + x11*x15 + x11*x16 + 
            x11 + x12*x13 + x12*x15 + x12 + x13*x15 + x14*x15 + x14*x16 
            + x15*x16 + x16,
        x1*x4 + x1*x7 + x1*x10 + x1*x13 + x1*x14 + x1*x15 + x1*x16 + 
            x1 + x2*x5 + x2*x6 + x2*x7 + x2*x8 + x2*x10 + x2*x11 + 
            x2*x12 + x2*x13 + x2 + x3*x5 + x3*x6 + x3*x8 + x3*x10 + 
            x3*x13 + x3*x15 + x3 + x4*x6 + x4*x10 + x4*x12 + x4*x13 + 
            x4 + x5*x6 + x5*x7 + x5*x8 + x5*x10 + x5*x11 + x5*x12 + 
            x5*x13 + x5 + x6*x14 + x6*x16 + x6 + x7*x9 + x7*x10 + 
            x7*x14 + x7*x15 + x8*x9 + x8*x10 + x8*x11 + x8*x12 + 
            x8*x14 + x8*x16 + x9*x10 + x9*x11 + x9*x15 + x10*x11 + 
            x10*x12 + x10*x16 + x11*x12 + x11*x13 + x11 + x12*x13 + 
            x12*x14 + x12*x15 + x12 + x14*x15 + x14 + x15*x16 + x15 + 
            x16,
        x1*x2 + x1*x3 + x1*x7 + x1*x10 + x1*x11 + x1*x13 + x1*x15 + 
            x2*x3 + x2*x4 + x2*x5 + x2*x7 + x2*x8 + x2*x10 + x2*x11 + 
            x2*x12 + x2*x14 + x2*x16 + x2 + x3*x6 + x3*x8 + x3*x10 + 
            x3*x11 + x3*x12 + x3*x13 + x3*x14 + x3*x15 + x3*x16 + 
            x4*x6 + x4*x8 + x4*x11 + x4*x12 + x4*x14 + x5*x6 + x5*x7 +
            x5*x8 + x5*x10 + x5*x13 + x5*x14 + x5*x15 + x5 + x6*x8 + 
            x6*x10 + x6*x11 + x6*x13 + x6*x14 + x6*x16 + x6 + x7*x9 + 
            x7*x11 + x7*x12 + x7*x13 + x8*x9 + x8*x12 + x8*x14 + 
            x9*x10 + x9*x13 + x9*x15 + x9 + x10*x11 + x10*x12 + x10*x13
            + x10*x14 + x10*x15 + x10*x16 + x11*x12 + x11*x14 + x11*x16 
            + x12*x13 + x12*x15 + x12*x16 + x12 + x13*x14 + x14*x16 + 
            x15*x16,
        x1*x2 + x1*x3 + x1*x4 + x1*x8 + x1*x9 + x1*x10 + x1*x11 + 
            x1*x16 + x1 + x2*x3 + x2*x4 + x2*x6 + x2*x7 + x2*x11 + 
            x2*x12 + x2*x14 + x2*x16 + x3*x5 + x3*x6 + x3*x11 + x3*x15
            + x3 + x4*x7 + x4*x10 + x4*x11 + x4*x14 + x5*x6 + x5*x7 + 
            x5*x10 + x5*x12 + x5*x13 + x5*x14 + x5*x15 + x6*x7 + x6*x9
            + x6*x12 + x6*x14 + x6*x16 + x6 + x7*x11 + x7*x12 + x7*x13 
            + x7*x14 + x7*x15 + x7*x16 + x7 + x8*x9 + x8*x10 + x8*x11 +
            x8*x13 + x8*x14 + x8*x16 + x9*x10 + x9*x11 + x9*x13 + 
            x9*x14 + x9*x15 + x9*x16 + x10*x13 + x10*x14 + x11*x13 + 
            x11*x15 + x12*x13 + x12*x14 + x12*x15 + x13*x15 + x14*x16 + 
            x15*x16 + x15 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x6 + x1*x12 + x1*x13 + x1*x15 + 
            x1*x16 + x1 + x2*x4 + x2*x6 + x2*x7 + x2*x8 + x2*x9 + 
            x2*x11 + x2*x12 + x2*x13 + x2*x15 + x2*x16 + x2 + x3*x6 + 
            x3*x9 + x3*x11 + x3*x13 + x3*x14 + x3*x15 + x3*x16 + x4*x5
            + x4*x6 + x4*x7 + x4*x8 + x4*x11 + x4*x12 + x4*x13 + 
            x4*x14 + x4*x15 + x4*x16 + x5*x6 + x5*x7 + x5*x9 + x5*x12 
            + x5*x16 + x6*x9 + x6*x10 + x6*x11 + x6*x16 + x6 + x7*x9 + 
            x7*x11 + x7*x13 + x7 + x8*x14 + x8*x15 + x8 + x9*x10 + 
            x9*x14 + x9*x15 + x9 + x10*x11 + x10 + x11*x12 + x11*x13 + 
            x11*x16 + x12 + x13*x15 + x13*x16 + x13 + x14*x15 + x15*x16 
            + x15 + 1,
        x1*x2 + x1*x9 + x1*x10 + x1*x11 + x1*x13 + x1*x14 + x1*x15 + 
            x2*x7 + x2*x8 + x2*x9 + x2*x10 + x2*x11 + x2*x12 + x2 + 
            x3*x4 + x3*x5 + x3*x8 + x3*x13 + x3*x15 + x3 + x4*x5 + 
            x4*x6 + x4*x8 + x4*x10 + x4*x11 + x4*x12 + x4*x13 + x4*x14
            + x4 + x5*x6 + x5*x7 + x5*x9 + x5*x12 + x5*x13 + x5*x14 + 
            x5*x15 + x5*x16 + x6*x8 + x6*x11 + x6*x12 + x6*x13 + x6 + 
            x7*x8 + x7*x10 + x7*x15 + x7 + x8*x13 + x8*x14 + x8 + 
            x9*x10 + x9*x11 + x9*x12 + x9*x13 + x9 + x10*x11 + x11*x12 
            + x11*x14 + x11*x15 + x11*x16 + x12*x13 + x12*x14 + x12 + 
            x13*x14 + x13 + x14*x15 + x14*x16 + x14 + x15*x16 + 1,
        x1*x2 + x1*x3 + x1*x5 + x1*x7 + x1*x9 + x1*x11 + x1*x12 + 
            x2*x3 + x2*x4 + x2*x8 + x2*x9 + x2*x10 + x2*x11 + x2*x15 +
            x2*x16 + x3*x6 + x3*x7 + x3*x8 + x3*x10 + x3*x12 + x3*x15 
            + x3*x16 + x4*x6 + x4*x9 + x4*x12 + x4*x14 + x4*x16 + x4 + 
            x5*x7 + x5*x8 + x5*x9 + x5*x10 + x5*x12 + x5*x13 + x6*x12 
            + x6*x13 + x6*x14 + x6*x15 + x6*x16 + x7*x9 + x7*x11 + 
            x7*x12 + x7*x13 + x8*x12 + x8*x13 + x8*x15 + x8*x16 + 
            x9*x11 + x9*x12 + x9*x13 + x9*x14 + x9*x15 + x9*x16 + 
            x10*x11 + x10*x13 + x10*x15 + x10 + x11*x13 + x11*x14 + 
            x11*x15 + x11 + x12*x13 + x12*x14 + x12*x15 + x12*x16 + x12 
            + x13*x14 + x15*x16 + x15 + 1,
        x1*x3 + x1*x4 + x1*x7 + x1*x8 + x1*x9 + x1*x11 + x1*x12 + 
            x1*x13 + x1*x15 + x1 + x2*x3 + x2*x8 + x2*x9 + x2*x10 + 
            x2*x11 + x2*x14 + x3*x6 + x3*x7 + x3*x8 + x3 + x4*x6 + 
            x4*x7 + x4*x11 + x4*x13 + x4*x14 + x4*x16 + x4 + x5*x6 + 
            x5*x7 + x5*x8 + x5*x9 + x5*x10 + x5*x12 + x5*x15 + x5 + 
            x6*x7 + x6*x8 + x6*x11 + x6*x15 + x6 + x7*x8 + x7*x11 + 
            x7*x13 + x7*x16 + x8*x9 + x8*x10 + x8*x11 + x8 + x9*x10 + 
            x9*x11 + x9*x12 + x9*x13 + x9 + x10*x11 + x10*x12 + x10*x14
            + x10*x15 + x11*x13 + x11*x15 + x11*x16 + x11 + x12*x13 + 
            x13*x15 + x13 + x14 + x15
        ]
    sol = [ 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0 ]
    return p, sol


def example_18_anf(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18):
    p = [
        x1*x3 + x1*x11 + x1*x12 + x1*x13 + x1*x17 + x1*x18 + x1 + 
            x2*x4 + x2*x5 + x2*x7 + x2*x8 + x2*x10 + x2*x11 + x2*x12 +
            x2*x14 + x2*x16 + x2*x17 + x2*x18 + x3*x4 + x3*x5 + x3*x6 
            + x3*x7 + x3*x9 + x3*x10 + x3*x11 + x3*x12 + x3*x15 + 
            x3*x18 + x3 + x4*x6 + x4*x12 + x4*x18 + x5*x6 + x5*x8 + 
            x5*x9 + x5*x11 + x5*x13 + x5*x14 + x5*x17 + x6*x11 + 
            x6*x13 + x6*x15 + x6*x16 + x6*x18 + x7*x9 + x7*x13 + 
            x7*x16 + x7*x17 + x7*x18 + x7 + x8*x9 + x8*x10 + x8*x11 + 
            x8*x12 + x8*x13 + x8*x17 + x8 + x9*x11 + x9*x12 + x9*x15 + 
            x10*x13 + x10*x14 + x10*x15 + x11*x13 + x11*x14 + x11*x15 + 
            x11*x18 + x11 + x12*x13 + x12*x14 + x12*x16 + x12*x17 + 
            x12*x18 + x13*x14 + x13*x15 + x13*x17 + x13 + x14*x15 + 
            x14*x16 + x15 + x16*x18 + x16 + x18,
        x1*x2 + x1*x4 + x1*x5 + x1*x6 + x1*x7 + x1*x10 + x1*x13 + 
            x1*x15 + x1*x18 + x2*x3 + x2*x6 + x2*x7 + x2*x11 + x2*x15 
            + x2*x16 + x3*x4 + x3*x5 + x3*x6 + x3*x9 + x3*x10 + x3*x11
            + x3*x12 + x3*x14 + x3*x15 + x3*x16 + x4*x5 + x4*x6 + 
            x4*x7 + x4*x8 + x4*x9 + x4*x11 + x4*x13 + x4*x15 + x4*x18 
            + x5*x7 + x5*x13 + x5*x14 + x5*x17 + x5*x18 + x6*x7 + 
            x6*x13 + x6*x14 + x6*x15 + x6*x16 + x6*x18 + x7*x8 + x7*x9
            + x7*x11 + x7*x12 + x7*x16 + x7*x17 + x7*x18 + x7 + x8*x9 +
            x8*x12 + x8*x14 + x8*x16 + x8*x18 + x9*x11 + x9*x12 + 
            x9*x14 + x9*x17 + x10*x12 + x10*x16 + x10*x17 + x10 + 
            x11*x12 + x11*x14 + x11*x18 + x12*x14 + x12*x17 + x13*x14 + 
            x13*x17 + x13*x18 + x13 + x14*x15 + x14 + x15*x16 + x15*x18 
            + x16*x17 + x16*x18 + x16 + x17 + 1,
        x1*x2 + x1*x3 + x1*x4 + x1*x5 + x1*x12 + x1*x14 + x1*x15 + 
            x1*x17 + x1 + x2*x9 + x2 + x3*x4 + x3*x6 + x3*x7 + x3*x9 +
            x3*x10 + x3*x11 + x3*x12 + x3*x13 + x3*x15 + x3*x17 + x3 + 
            x4*x6 + x4*x8 + x4*x9 + x4*x12 + x4*x13 + x4*x15 + x4*x16 
            + x5*x8 + x5*x11 + x5*x13 + x5*x14 + x6*x7 + x6*x8 + 
            x6*x10 + x6*x12 + x6*x16 + x7*x11 + x7*x16 + x7*x18 + x7 + 
            x8*x10 + x8*x13 + x8*x15 + x8*x17 + x8*x18 + x8 + x9*x10 + 
            x9*x13 + x9*x16 + x9*x17 + x9*x18 + x10*x12 + x10*x14 + 
            x10*x17 + x10 + x11*x12 + x11*x17 + x11*x18 + x12*x14 + 
            x12*x15 + x12*x16 + x13*x16 + x13*x17 + x13 + x14*x16 + 
            x14*x17 + x14 + x15*x17 + x15*x18 + x16*x17 + x16*x18 + 
            x17*x18 + x17 + x18,
        x1*x2 + x1*x5 + x1*x6 + x1*x7 + x1*x9 + x1*x10 + x1*x11 + 
            x1*x14 + x1*x16 + x1*x17 + x1*x18 + x1 + x2*x4 + x2*x5 + 
            x2*x8 + x2*x10 + x2*x16 + x2*x17 + x3*x7 + x3*x8 + x3*x9 +
            x3*x10 + x3*x13 + x3*x14 + x3*x15 + x3*x17 + x3*x18 + 
            x4*x5 + x4*x8 + x4*x14 + x4*x17 + x5*x6 + x5*x14 + x5*x17 
            + x6*x7 + x6*x8 + x6*x12 + x6*x13 + x6*x14 + x6*x17 + 
            x7*x8 + x7*x10 + x7*x11 + x7*x13 + x7*x15 + x7*x16 + 
            x7*x18 + x7 + x8*x10 + x8*x12 + x8*x15 + x8*x17 + x9*x10 + 
            x9*x13 + x9*x17 + x9 + x10*x12 + x10*x13 + x10*x15 + 
            x10*x17 + x10 + x11*x13 + x11*x17 + x11*x18 + x12*x14 + 
            x12*x15 + x12*x16 + x12*x17 + x12 + x13*x15 + x13*x16 + 
            x13*x18 + x13 + x14*x15 + x14*x16 + x14*x17 + x14*x18 + x14 
            + x15*x17 + x16*x17 + x16*x18 + x16 + x17 + 1,
        x1*x2 + x1*x3 + x1*x6 + x1*x7 + x1*x8 + x1*x9 + x1*x10 + 
            x1*x11 + x1*x12 + x1*x13 + x1*x15 + x1*x17 + x1*x18 + 
            x2*x3 + x2*x4 + x2*x6 + x2*x9 + x2*x13 + x2*x14 + x2*x15 +
            x2*x16 + x2*x17 + x2*x18 + x2 + x3*x7 + x3*x8 + x3*x10 + 
            x3*x11 + x3*x12 + x3*x13 + x3*x14 + x3*x15 + x3 + x4*x6 + 
            x4*x7 + x4*x9 + x4*x11 + x4*x14 + x4*x18 + x4 + x5*x6 + 
            x5*x7 + x5*x10 + x5*x11 + x5*x12 + x5*x14 + x5*x15 + 
            x5*x17 + x5 + x6*x7 + x6*x8 + x6*x13 + x6*x14 + x6*x16 + 
            x6*x17 + x7*x9 + x7*x10 + x7*x12 + x7*x15 + x7*x16 + 
            x8*x13 + x8*x14 + x8*x15 + x8*x16 + x8*x17 + x9*x11 + 
            x9*x16 + x9*x17 + x9 + x10*x11 + x10*x17 + x10*x18 + 
            x11*x16 + x11*x17 + x11 + x12*x14 + x12*x15 + x12*x17 + 
            x13*x14 + x13*x18 + x14*x17 + x15*x18 + x15 + x16*x17 + x16 
            + x17*x18 + 1,
        x1*x3 + x1*x4 + x1*x5 + x1*x6 + x1*x7 + x1*x8 + x1*x9 + 
            x1*x15 + x1*x17 + x1 + x2*x4 + x2*x5 + x2*x7 + x2*x8 + 
            x2*x9 + x2*x12 + x2*x13 + x2*x14 + x2*x15 + x3*x4 + x3*x5 
            + x3*x6 + x3*x7 + x3*x8 + x3*x9 + x3*x12 + x3*x13 + x3*x14
            + x3*x15 + x3*x17 + x3 + x4*x6 + x4*x8 + x4*x10 + x4*x15 + 
            x4*x16 + x4*x18 + x5*x6 + x5*x10 + x5*x13 + x5*x15 + 
            x5*x17 + x6*x7 + x6*x8 + x6*x11 + x6*x13 + x6*x16 + x7*x8 
            + x7*x11 + x7*x12 + x7*x13 + x7*x15 + x7*x16 + x7 + x8*x9 +
            x8*x14 + x8*x15 + x8 + x9*x10 + x9*x12 + x9*x14 + x9*x16 + 
            x9*x18 + x9 + x11*x12 + x11*x14 + x11*x16 + x11*x17 + 
            x11*x18 + x12*x14 + x12*x16 + x12*x17 + x13*x15 + x13*x18 + 
            x13 + x14*x15 + x14*x16 + x14*x18 + x14 + x15*x16 + x15*x18 
            + x15 + x16*x18 + x16 + x17*x18 + x17,
        x1*x2 + x1*x3 + x1*x4 + x1*x6 + x1*x8 + x1*x9 + x1*x10 + 
            x1*x11 + x1*x14 + x1*x17 + x1*x18 + x1 + x2*x3 + x2*x4 + 
            x2*x5 + x2*x6 + x2*x9 + x2*x10 + x2*x11 + x2*x12 + x2*x13 
            + x2*x14 + x2*x15 + x2*x16 + x3*x4 + x3*x5 + x3*x7 + x3*x8
            + x3*x9 + x3*x11 + x3*x12 + x3*x13 + x3*x14 + x3*x15 + 
            x4*x6 + x4*x7 + x4*x11 + x4*x14 + x4*x15 + x4*x16 + x4 + 
            x5*x8 + x5*x9 + x5*x10 + x5*x13 + x5*x14 + x5*x17 + x5*x18
            + x5 + x6*x9 + x6*x10 + x6*x13 + x6*x15 + x6*x16 + x6*x17 +
            x7*x9 + x7*x13 + x7*x14 + x7 + x8*x9 + x8*x10 + x8*x14 + 
            x8*x18 + x8 + x9*x10 + x9*x12 + x9*x15 + x9*x16 + x10*x13 +
            x10*x14 + x10*x15 + x10*x16 + x10*x17 + x11*x12 + x11*x18 + 
            x11 + x12*x14 + x12*x16 + x12 + x13*x14 + x13*x15 + x13*x16 
            + x13*x17 + x13*x18 + x14*x16 + x14 + x15*x17 + x16*x18 + 
            x17*x18 + x18 + 1,
        x1*x6 + x1*x7 + x1*x10 + x1*x14 + x1*x15 + x1*x17 + x1*x18 + 
            x2*x4 + x2*x5 + x2*x8 + x2*x11 + x2*x13 + x2*x18 + x2 + 
            x3*x4 + x3*x5 + x3*x7 + x3*x8 + x3*x9 + x3*x10 + x3*x12 + 
            x3*x13 + x3*x14 + x3*x18 + x3 + x4*x6 + x4*x8 + x4*x10 + 
            x4*x12 + x4*x16 + x4*x17 + x4 + x5*x6 + x5*x7 + x5*x8 + 
            x5*x9 + x5*x11 + x5*x15 + x5*x16 + x5 + x6*x8 + x6*x12 + 
            x6*x14 + x6*x15 + x6*x18 + x6 + x7*x8 + x7*x10 + x7*x11 + 
            x7*x12 + x7*x13 + x7*x17 + x7*x18 + x8*x9 + x8*x12 + 
            x8*x13 + x8*x16 + x9*x10 + x9*x11 + x9*x12 + x9*x13 + 
            x9*x16 + x9*x17 + x9*x18 + x10*x11 + x10*x12 + x10*x14 + 
            x10*x15 + x10*x16 + x10*x18 + x10 + x11*x13 + x11*x15 + 
            x11*x16 + x11 + x12*x14 + x12*x15 + x12*x18 + x12 + x13*x14 
            + x13*x15 + x13*x16 + x13*x17 + x13 + x14*x15 + x14*x16 + 
            x14*x18 + x14 + x15*x18 + x15 + x16*x18 + x17*x18 + x17 + 1,
        x1*x2 + x1*x4 + x1*x6 + x1*x10 + x1*x11 + x1*x16 + x1*x17 + 
            x1 + x2*x4 + x2*x5 + x2*x7 + x2*x13 + x2*x14 + x2*x16 + 
            x2*x18 + x2 + x3*x4 + x3*x5 + x3*x7 + x3*x10 + x3*x12 + 
            x3*x17 + x4*x5 + x4*x6 + x4*x10 + x4*x14 + x4*x18 + x4 + 
            x5*x6 + x5*x7 + x5*x8 + x5*x10 + x5*x11 + x5*x12 + x5*x14 
            + x5*x16 + x5*x17 + x5*x18 + x5 + x6*x8 + x6*x9 + x6*x11 + 
            x6*x13 + x6*x15 + x6 + x7*x8 + x7*x11 + x7*x12 + x7*x15 + 
            x7*x16 + x8*x9 + x8*x15 + x8*x18 + x8 + x9*x10 + x9*x15 + 
            x9*x17 + x9*x18 + x10*x12 + x10*x14 + x10 + x11*x12 + 
            x11*x14 + x11*x15 + x11*x18 + x11 + x12*x14 + x12*x15 + 
            x12*x17 + x12 + x13*x15 + x13 + x14*x15 + x14*x17 + x14 + 
            x15*x16 + x15*x17 + x15*x18 + x16*x17 + x16 + x17*x18 + x17,
        x1*x5 + x1*x9 + x1*x12 + x1*x13 + x1*x14 + x1*x15 + x1*x16 + 
            x1*x17 + x1 + x2*x5 + x2*x6 + x2*x8 + x2*x11 + x2*x15 + 
            x2*x16 + x3*x4 + x3*x5 + x3*x6 + x3*x7 + x3*x8 + x3*x12 + 
            x3*x13 + x3*x14 + x3*x16 + x3*x18 + x4*x5 + x4*x7 + x4*x8 
            + x4*x11 + x4*x12 + x4*x15 + x4*x17 + x4*x18 + x4 + x5*x6 +
            x5*x7 + x5*x8 + x5*x9 + x5*x10 + x5*x11 + x5*x13 + x5*x18 
            + x6*x7 + x6*x8 + x6*x9 + x6*x12 + x6*x13 + x6*x14 + 
            x6*x15 + x6*x16 + x6*x17 + x7*x8 + x7*x10 + x7*x13 + 
            x7*x14 + x7*x15 + x7*x17 + x7 + x8*x13 + x8*x15 + x8*x18 + 
            x8 + x9*x11 + x9*x12 + x9*x13 + x9*x16 + x9*x17 + x9*x18 + 
            x10*x11 + x10*x13 + x10*x15 + x10*x16 + x10*x17 + x10 + 
            x11*x12 + x11*x18 + x12*x14 + x12*x15 + x12*x17 + x12*x18 + 
            x13*x16 + x13*x18 + x14*x15 + x14*x16 + x14*x17 + x14*x18 + 
            x15*x17 + x15*x18 + x16*x18 + x17 + x18 + 1,
        x1*x4 + x1*x5 + x1*x10 + x1*x11 + x1*x14 + x1*x15 + x1*x16 + 
            x1*x17 + x1*x18 + x2*x3 + x2*x5 + x2*x6 + x2*x7 + x2*x9 + 
            x2*x10 + x2*x11 + x2*x13 + x2*x14 + x2*x17 + x2*x18 + x2 + 
            x3*x5 + x3*x7 + x3*x8 + x3*x10 + x3*x13 + x3*x15 + x3*x17 
            + x3 + x4*x8 + x4*x9 + x4*x12 + x4*x13 + x4*x15 + x4*x17 + 
            x4*x18 + x5*x6 + x5*x8 + x5*x9 + x5*x16 + x5*x17 + x6*x7 +
            x6*x8 + x6*x10 + x6*x12 + x6*x13 + x6*x14 + x6*x15 + x6 + 
            x7*x10 + x7*x11 + x7*x13 + x7*x14 + x7*x15 + x7*x17 + 
            x7*x18 + x7 + x8*x9 + x8*x15 + x8*x17 + x9*x10 + x9*x11 + 
            x9*x12 + x9*x13 + x9*x16 + x10*x13 + x10*x16 + x10 + 
            x11*x14 + x11*x18 + x11 + x12*x13 + x12*x14 + x12*x15 + 
            x12*x17 + x12*x18 + x12 + x13*x15 + x13*x16 + x13 + x14*x15 
            + x14*x16 + x14*x17 + x14*x18 + x15*x16 + x15*x17 + x16*x17 
            + x16*x18,
        x1*x2 + x1*x3 + x1*x5 + x1*x6 + x1*x7 + x1*x9 + x1*x10 + 
            x1*x11 + x1*x12 + x1 + x2*x5 + x2*x7 + x2*x9 + x2*x10 + 
            x2*x13 + x2*x14 + x2*x15 + x2*x16 + x2*x18 + x3*x4 + x3*x5
            + x3*x8 + x3*x12 + x3*x14 + x3*x15 + x3*x17 + x3*x18 + 
            x4*x6 + x4*x9 + x4*x10 + x4*x11 + x4*x12 + x4*x13 + x4*x15
            + x4*x16 + x4 + x5*x7 + x5*x9 + x5*x12 + x5*x14 + x5*x17 + 
            x6*x7 + x6*x11 + x6*x12 + x6*x15 + x6*x17 + x6*x18 + x6 + 
            x7*x9 + x7*x11 + x7*x12 + x7*x13 + x7*x15 + x7*x16 + x8*x9
            + x8*x15 + x9*x12 + x9*x14 + x9*x15 + x9*x17 + x10*x11 + 
            x10*x12 + x10*x13 + x10*x14 + x10*x17 + x10 + x11*x12 + 
            x11*x14 + x11*x16 + x11*x17 + x11*x18 + x11 + x12*x13 + 
            x12*x17 + x12*x18 + x13*x15 + x13*x18 + x13 + x14*x15 + 
            x14*x18 + x14 + x15*x17 + x15*x18 + x16*x17 + x17*x18 + x18,
        x1*x4 + x1*x5 + x1*x8 + x1*x10 + x1*x12 + x1*x13 + x1*x14 + 
            x1*x15 + x1 + x2*x4 + x2*x5 + x2*x7 + x2*x8 + x2*x9 + 
            x2*x11 + x2*x13 + x2*x16 + x2*x18 + x3*x5 + x3*x7 + x3*x9 
            + x3*x10 + x3*x11 + x3*x13 + x3*x14 + x3*x16 + x3*x17 + 
            x3*x18 + x3 + x4*x8 + x4*x10 + x4*x12 + x4*x13 + x4*x14 + 
            x4*x15 + x4*x17 + x5*x7 + x5*x8 + x5*x9 + x5*x12 + x5*x14 
            + x5*x17 + x6*x7 + x6*x9 + x6*x10 + x6*x11 + x6*x12 + 
            x6*x14 + x6*x15 + x6*x17 + x6*x18 + x6 + x7*x8 + x7*x10 + 
            x7*x11 + x7*x13 + x7*x15 + x7*x17 + x7*x18 + x7 + x8*x10 + 
            x8*x11 + x8*x13 + x8*x15 + x8*x18 + x8 + x9*x10 + x9*x11 + 
            x9*x14 + x9*x16 + x10*x11 + x10*x12 + x10*x13 + x10*x17 + 
            x11*x12 + x11*x13 + x11*x14 + x11*x15 + x11*x17 + x12*x15 + 
            x12*x16 + x12*x17 + x13*x15 + x13*x16 + x13*x17 + x13 + 
            x15*x16 + x15*x17 + x15 + x16*x18 + x16 + x17*x18 + x18,
        x1*x2 + x1*x3 + x1*x4 + x1*x8 + x1*x9 + x1*x10 + x1*x12 + 
            x1*x15 + x1*x16 + x1*x17 + x1*x18 + x2*x3 + x2*x5 + x2*x6 
            + x2*x7 + x2*x9 + x2*x11 + x2*x12 + x2*x13 + x2*x15 + 
            x2*x16 + x3*x4 + x3*x7 + x3*x9 + x3*x10 + x3*x12 + x3*x17 
            + x4*x7 + x4*x10 + x4*x12 + x4*x13 + x4*x17 + x5*x6 + 
            x5*x7 + x5*x8 + x5*x11 + x5*x12 + x5*x14 + x5*x15 + x5*x17
            + x5 + x6*x7 + x6*x13 + x6*x15 + x6*x16 + x6 + x7*x8 + 
            x7*x9 + x7*x12 + x7*x17 + x7 + x8*x10 + x8*x13 + x8*x14 + 
            x8*x18 + x9*x11 + x9*x12 + x9*x14 + x9*x16 + x9 + x10*x11 +
            x10*x12 + x10*x13 + x10*x17 + x10*x18 + x11*x12 + x11*x14 + 
            x11*x16 + x11*x17 + x11 + x12*x13 + x12*x14 + x12 + x13*x14 
            + x13*x15 + x13*x18 + x13 + x14*x16 + x14*x18 + x14 + 
            x15*x16 + x15 + x16*x17 + x16*x18 + x16 + x17*x18 + 1,
        x1*x2 + x1*x4 + x1*x6 + x1*x7 + x1*x8 + x1*x10 + x1*x14 + 
            x1*x15 + x1*x16 + x1*x17 + x1*x18 + x2*x3 + x2*x5 + x2*x8 
            + x2*x12 + x2*x13 + x2*x14 + x2*x16 + x2*x17 + x2 + x3*x4 +
            x3*x9 + x3*x13 + x3*x14 + x3*x16 + x3*x17 + x3*x18 + x4*x5
            + x4*x6 + x4*x12 + x4*x13 + x4*x17 + x4*x18 + x5*x6 + 
            x5*x7 + x5*x10 + x5*x11 + x5*x12 + x5*x13 + x5*x14 + 
            x5*x15 + x5*x16 + x5*x18 + x5 + x6*x7 + x6*x10 + x6*x12 + 
            x6*x14 + x6*x15 + x6*x17 + x7*x10 + x7*x11 + x7*x15 + 
            x7*x16 + x7*x17 + x8*x9 + x8*x10 + x8*x11 + x8*x12 + 
            x8*x13 + x8*x14 + x8*x16 + x8*x18 + x9*x10 + x9*x12 + 
            x9*x14 + x9*x16 + x9*x17 + x10*x11 + x10*x17 + x10*x18 + 
            x11*x12 + x11*x13 + x11*x14 + x11*x15 + x11*x17 + x11*x18 + 
            x12*x13 + x12*x15 + x12*x16 + x13*x14 + x13*x15 + x13*x17 + 
            x13*x18 + x14*x17 + x14*x18 + x14 + x15*x18 + x16*x17 + x16 
            + x18,
        x1*x2 + x1*x3 + x1*x6 + x1*x8 + x1*x10 + x1*x11 + x1*x12 + 
            x1*x17 + x1*x18 + x2*x6 + x2*x11 + x2*x12 + x2*x14 + 
            x2*x15 + x2*x18 + x2 + x3*x5 + x3*x6 + x3*x7 + x3*x8 + 
            x3*x9 + x3*x10 + x3*x12 + x3*x14 + x3*x15 + x3*x16 + x4*x8
            + x4*x9 + x4*x13 + x4*x17 + x4*x18 + x5*x7 + x5*x8 + x5*x9
            + x5*x14 + x5*x15 + x5*x17 + x6*x7 + x6*x10 + x6*x12 + 
            x6*x14 + x6*x16 + x6*x17 + x6*x18 + x7*x8 + x7*x11 + 
            x7*x13 + x7*x15 + x8*x9 + x8*x12 + x8*x13 + x8*x15 + 
            x8*x16 + x8*x17 + x8 + x9*x13 + x9*x14 + x9*x16 + x9*x18 + 
            x9 + x10*x11 + x10*x13 + x10*x14 + x10 + x11*x16 + x11 + 
            x12*x13 + x12*x16 + x13*x14 + x13*x16 + x13*x18 + x13 + x14 
            + x15*x16 + x16*x18 + x16 + x17*x18 + 1,
        x1*x3 + x1*x5 + x1*x7 + x1*x12 + x1*x13 + x1*x14 + x1*x15 + 
            x1*x16 + x1*x17 + x1*x18 + x2*x4 + x2*x7 + x2*x8 + x2*x12 
            + x2*x15 + x2*x16 + x2*x18 + x2 + x3*x4 + x3*x7 + x3*x8 + 
            x3*x10 + x3*x12 + x3*x13 + x3*x14 + x3*x16 + x3*x18 + 
            x4*x6 + x4*x7 + x4*x8 + x4*x10 + x4*x12 + x4*x15 + x4*x16 
            + x4*x18 + x5*x6 + x5*x8 + x5*x10 + x5*x13 + x5*x16 + 
            x5*x18 + x6*x8 + x6*x9 + x6*x15 + x6*x16 + x6*x17 + x6*x18
            + x6 + x7*x8 + x7*x11 + x7*x14 + x7*x15 + x7*x16 + x8*x9 + 
            x8*x10 + x8*x12 + x8*x15 + x8*x16 + x8*x17 + x9*x16 + 
            x10*x12 + x10*x13 + x10*x15 + x10*x16 + x10*x17 + x10 + 
            x11*x12 + x11*x14 + x11 + x12*x16 + x12*x17 + x12*x18 + 
            x13*x14 + x13*x15 + x13*x16 + x13*x18 + x14*x16 + x15*x17 + 
            x16*x17 + x16*x18 + x16 + x17*x18 + x17,
        x1*x2 + x1*x3 + x1*x4 + x1*x7 + x1*x14 + x1*x15 + x1*x16 + 
            x1*x18 + x1 + x2*x3 + x2*x4 + x2*x8 + x2*x9 + x2*x11 + 
            x2*x12 + x2*x13 + x2*x14 + x2*x15 + x3*x5 + x3*x8 + x3*x9 
            + x3*x10 + x3*x12 + x3*x13 + x4*x7 + x4*x8 + x4*x9 + 
            x4*x12 + x4*x14 + x4*x15 + x4*x16 + x4*x17 + x4*x18 + x4 + 
            x5*x6 + x5*x9 + x5*x14 + x5*x15 + x5*x17 + x5*x18 + x5 + 
            x6*x8 + x6*x10 + x6*x11 + x6*x12 + x6*x13 + x6*x14 + 
            x6*x15 + x6*x16 + x6*x17 + x6 + x7*x14 + x7*x15 + x8*x10 + 
            x8*x11 + x8*x12 + x8*x13 + x8*x15 + x8*x16 + x9*x11 + 
            x9*x12 + x9*x13 + x9*x16 + x9*x17 + x9*x18 + x10*x11 + 
            x10*x12 + x10*x14 + x10*x15 + x10*x16 + x10 + x11*x14 + x11 
            + x12*x14 + x12*x15 + x12*x16 + x12*x18 + x13*x15 + x14*x15 
            + x14 + x15*x16 + x16*x18 + x18 + 1
        ]
    sol = [ [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0] ]
    return p, sol