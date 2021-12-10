import numpy as np
import roughpath as rp
import sympy as sp


class VectorField:
    def __init__(self, vf, norm=rp.l1):
        """
        .
        :param vf: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param norm:
        """
        self.exact_der = len(vf)
        self.vf = [vf[i] for i in range(self.exact_der)]
        self.norm = norm
        self.global_norm = 0.
        self.local_norm = 0.

    def reset_local_norm(self):
        self.local_norm = 0.

    def vector_field(self, ls):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :return: Solution on partition points
        """
        return None


class VectorFieldNumeric(VectorField):
    def __init__(self, vf, h=1e-06, norm=rp.l1):
        """
        .
        :param vf: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param h:
        :param norm:
        """
        super().__init__(vf, norm)
        self.h = h

    def derivative(self, y, dx):
        """
        Computes the derivative of the vector field.
        :param y: Point at which the derivative is calculated
        :param dx: Direction to which the vector field is applied, an n-tensor
        :return: An approximation of the n-th derivative
        """
        N = len(dx.shape)
        if N <= self.exact_der:
            return self.vf[N - 1](y, dx)
        x_dim = np.shape(dx)[0]
        result = 0
        for i in range(x_dim):
            vec = np.zeros(x_dim)
            vec[i] = 1.
            direction = self.vf[0](y, vec)
            result += (self.derivative(y + self.h / 2 * direction, dx[..., i])
                       - self.derivative(y - self.h / 2 * direction, dx[..., i])) / self.h
        return result

    def vector_field(self, ls):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :return: Solution on partition points
        """
        deg = len(ls)

        if self.norm is None:
            return lambda y: np.sum(np.array([self.derivative(y, ls[i]) for i in range(deg)]), axis=0)

        def vf_norm(y):
            ls_norms = np.array([self.norm(ls[i]) for i in range(deg)])
            total_ls_norm = np.sum(ls_norms)
            summands = np.array([self.derivative(y, ls[i]) for i in range(deg)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                if ls_norms[i] > 1e-05*total_ls_norm:
                    local_local_norm = (self.norm(summands[i]) / ls_norms[i]) ** (1. / (i + 1))
                    if local_local_norm > 1e+3:
                        print(len(ls))
                        print(i)
                        print(ls)
                        print(ls[i])
                    self.local_norm = np.fmax(self.local_norm, local_local_norm)
                    self.global_norm = np.fmax(self.global_norm, local_local_norm)
            return vf

        return vf_norm


class VectorFieldSymbolic(VectorField):
    def __init__(self, vf, norm=rp.l1, variables=None):
        """
        .
        :param vf: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param norm:
        """
        super().__init__(vf, norm)
        self.variables = variables

    def new_derivative(self):
        """
        Computes the next derivative that has not yet been calculated.
        :return:
        """
        highest_der = self.vf[-1]
        base_func = self.vf[0]
        der_highest_der = sp.Array([sp.diff(highest_der, self.variables[i]) for i in range(len(self.variables))])
        permutations = [*range(der_highest_der.rank())]
        permutations[0] = 1
        permutations[1] = 0
        next_order = sp.tensorcontraction(sp.tensorproduct(base_func, der_highest_der), (0, 2), permutations)
        self.vf.append(next_order)
        return None

    def derivative(self, y, dx):
        rank = len(dx.shape)
        vec_field = self.vf[rank-1]
        eval_vec_field = sp.lambdify(self.variables, vec_field, modules='numpy')
        eval_vec_field = eval_vec_field(y)
        return np.tensordot(eval_vec_field, dx, axes=rank)

    def vector_field(self, ls):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :return: Solution on partition points
        """
        deg = len(ls)
        while len(self.vf) < deg:
            self.new_derivative()

        if self.norm is None:
            return lambda y: np.sum(np.array([self.derivative(y, ls[i]) for i in range(deg)]), axis=0)

        def vf_norm(y):
            ls_norms = np.array([self.norm(ls[i]) for i in range(deg)])
            total_ls_norm = np.sum(ls_norms)
            summands = np.array([self.derivative(y, ls[i]) for i in range(deg)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                if ls_norms[i] > 1e-05 * total_ls_norm:
                    local_local_norm = (self.norm(summands[i]) / ls_norms[i]) ** (1. / (i + 1))
                    if local_local_norm > 1e+3:
                        print(len(ls))
                        print(i)
                        print(ls)
                        print(ls[i])
                    self.local_norm = np.fmax(self.local_norm, local_local_norm)
                    self.global_norm = np.fmax(self.global_norm, local_local_norm)
            return vf

        return vf_norm
