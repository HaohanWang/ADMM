__author__ = 'haohanwang'

import numpy as np
from ADMM.ADMM import ADMM

X = np.random.random((500, 100))
y = np.random.random((500, 1))
y[np.where(y <= 0.5)] = 0
y[np.where(y >= 0.5)] = 1
y = y.astype(int)
X[np.where(y == 0), :] -= 0.2
X[np.where(y == 1), :] += 0.2
beta = np.random.random((100, 1))
beta2 = np.random.random((100, 1))
lam = 0.01
rho = 0.01


def obj(b1, b2, b3):
    return 0.5 * ((np.square(y - np.dot(X, b1))).sum())**2 + \
           lam * (np.abs(b2)).sum() + rho / 2 * ((np.square(b1 - b2)).sum())**2 + np.dot(b3.T, const(b1, b2))


def const(b1, b2):
    return b1 - b2


def l_x(b1, b2, b3):
    return 0.5 * ((np.square(y - np.dot(X, b1))).sum())**2 + rho / 2 * ((np.square(b1 - b2)).sum())**2 + np.dot(b3.T, const(b1, b2))


def l_x_jac(b1, b2, b3):
    return np.dot(X.T, (y - np.dot(X, b1)))  + rho * (b1 - b2) + np.dot(b3.T, b1)


def l_z(b1, b2, b3):
    return lam * (np.abs(b2)).sum() + rho / 2 * ((b1 - b2).sum())**2 + np.dot(b3.T, const(b1, b2))


def l_z_jac(b1, b2, b3):
    return lam * b2 / np.abs(b2.sum()) - rho * (b1 - b2) - np.dot(b3.T, b2)


solver = ADMM(0.5, maxIter=1000)
(x, z, y, c) = solver.run(cost=obj, l_x=l_x, l_z=l_z, const=const, x=np.random.random((100, 1)), z=np.random.random((100, 1)),
                          l_x_jac = l_x_jac, l_z_jac=l_z_jac, step_size=1e-6, tol=1e-6, sub_iter=5)
# print x.T
# print z.T
# print y.T