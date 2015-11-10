__author__ = 'haohanwang'

import numpy as np


class ADMM:
    def __init__(self, rho, maxIter=1e4):
        self.rho = rho
        self.maxIter = int(maxIter)
        self.y = 0

    def run(self, cost, l_x, l_z, const, x, z, l_x_jac, l_z_jac, tol = 1e-3, l_x_hessian=None, l_z_hessian=None, sub_iter=1,
            step_size=1):
        self.y = np.zeros(x.shape)
        self.cost = cost
        self.lx = l_x
        self.lz = l_z
        self.const = const
        self.x = x
        self.z = z
        self.lxj = l_x_jac
        self.lzj = l_z_jac
        self.lxh = l_x_hessian
        self.lzh = l_z_hessian
        self.sub_iter = sub_iter
        self.ss = step_size

        prev = self.cost(self.x, self.z, self.y)
        print prev
        curr = 0
        for i in range(self.maxIter):
            self.update_f()
            self.update_g()
            self.update_Lagrangian()
            curr = self.cost(self.x, self.z, self.y)
            print curr,
            if prev - curr <= tol:
                print 'Early Stop, program converges'
                print 'Final cost', curr
                return self.x, self.z, self.y, curr
            prev = curr
        print 'run out of iterations'
        print 'Final cost', curr
        return self.x, self.z, self.y, curr

    def update_f(self):
        if self.lxh is None:
            for i in range(self.sub_iter):
                self.x += self.ss * self.lxj(self.x, self.z, self.y)
        else:
            for i in range(self.sub_iter):
                self.x += self.ss * np.linalg.inv(self.lxh(self.x, self.z, self.y))

    def update_g(self):
        if self.lzh is None:
            for i in range(self.sub_iter):
                self.z += self.ss * self.lzj(self.z, self.z, self.y)
        else:
            for i in range(self.sub_iter):
                self.z += self.ss * np.linalg.inv(self.lzh(self.x, self.z, self.y))

    def update_Lagrangian(self):
        self.y += self.rho * (self.const(self.x, self.z))