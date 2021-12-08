import numpy as np
import roughpath as rp


class VectorField:
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
        self.exact_der = len(vf)
        self.vf = [vf[i] for i in range(self.exact_der)]
        self.h = h
        self.norm = norm
        self.global_norm = 0.
        self.local_norm = 0.

    def reset_local_norm(self):
        self.local_norm = 0.

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
            summands = np.array([self.derivative(y, ls[i]) for i in range(deg)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                norm_ls_i = self.norm(ls[i])
                if norm_ls_i > 0:
                    local_local_norm = (self.norm(summands[i]) / norm_ls_i) ** (1. / (i + 1))
                    self.local_norm = np.fmax(self.local_norm, local_local_norm)
                    self.global_norm = np.fmax(self.global_norm, local_local_norm)
            return vf

        return vf_norm
