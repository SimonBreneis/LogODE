import numpy as np
import sympy as sp
import tensoralgebra as ta


class VectorField:
    def __init__(self, f, norm=ta.l1):
        """
        Initialization.
        :param f: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f is a list of length smaller than deg), the derivatives
            are computed numerically or symbolically
            If the derivatives are specified, then f[i] is the ith derivative (i=0, ..., deg-1), and f[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param norm: Vector space norm used for estimating the norm of f
        """
        self.exact_der = len(f)
        self.f = [f[i] for i in range(self.exact_der)]
        self.norm = norm
        self.global_norm = [0.]  # the ith entry is a (lower) estimate of the norm \|f^{\circ i}\|_\infty
        self.local_norm = [0.]

    def reset_local_norm(self):
        """
        Resets the values of self.local_norm to zero.
        :return: Nothing
        """
        self.local_norm = [0.]*len(self.local_norm)

    def derivative(self, y, dx):
        """
        Computes the derivative of the vector field.
        :param y: Point at which the derivative is calculated
        :param dx: Direction to which the vector field is applied, an n-tensor
        :return: An approximation of the n-th derivative
        """
        return None

    def vector_field(self, ls):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :return: Solution on partition points
        """
        deg = ls.n_levels()

        if self.norm is None:
            return lambda y: np.sum(np.array([self.derivative(y, ls[i]) for i in range(1, deg+1)]), axis=0)

        def compute_vf_and_norm(y):
            while deg > len(self.local_norm):
                self.local_norm.append(0.)
                self.global_norm.append(0.)
            ls_norms = np.array([ls.norm(i, self.norm) for i in range(1, deg+1)])
            summands = np.array([self.derivative(y, ls[i]) for i in range(1, deg+1)])
            vf = np.sum(summands, axis=0)
            for i in range(deg):
                local_local_norm = (self.norm(summands[i]) / ls_norms[i]) ** (1. / (i + 1))
                if local_local_norm > self.local_norm[i]:
                    self.local_norm[i] = local_local_norm
                    if local_local_norm > self.global_norm[i]:
                        self.global_norm[i] = local_local_norm
            return vf

        return compute_vf_and_norm


class VectorFieldNumeric(VectorField):
    def __init__(self, f, h=1e-06, norm=ta.l1):
        """
        Initialization.
        :param f: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param h: Step size in numerical differentiation
        :param norm: Vector space norm used for estimating the norm of f
        """
        super().__init__(f, norm)
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
            return self.f[N - 1](y, dx)
        x_dim = np.shape(dx)[0]
        result = 0
        for i in range(x_dim):
            vec = np.zeros(x_dim)
            vec[i] = 1.
            direction = self.f[0](y, vec)
            result += (self.derivative(y + self.h / 2 * direction, dx[..., i])
                       - self.derivative(y - self.h / 2 * direction, dx[..., i])) / self.h
        return result


class VectorFieldSymbolic(VectorField):
    def __init__(self, f, norm=ta.l1, variables=None):
        """
        Initialization.
        :param f: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed symbolically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param norm: Vector space norm used for estimating the norm of f
        :param variables: The sympy variables with respect to which f is defined
        """
        super().__init__(f, norm)
        self.variables = variables

    def new_derivative(self):
        """
        Computes the next derivative that has not yet been calculated.
        :return: Nothing
        """
        highest_der = self.f[-1]
        base_func = self.f[0]
        der_highest_der = sp.Array([sp.diff(highest_der, self.variables[i]) for i in range(len(self.variables))])
        permutations = [*range(der_highest_der.rank())]
        permutations[0] = 1
        permutations[1] = 0
        self.f.append(sp.permutedims(sp.tensorcontraction(sp.tensorproduct(base_func, der_highest_der), (0, 2)),
                                     permutations))
        return None

    def derivative(self, y, dx):
        """
        Computes the len(dx.shape)-th term in the log-ODE method.
        :param y: The current position of the approximated solution
        :param dx: A level of the log-signature of the underlying path
        :return: The len(dx.shape)-th term in the log-ODE method
        """
        rank = len(dx.shape)
        return np.tensordot(sp.lambdify(self.variables, self.f[rank - 1], modules='numpy')(*list(y)), dx, axes=rank)

    def vector_field(self, ls):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :return: Solution on partition points
        """
        deg = ls.n_levels()
        while len(self.f) < deg:
            self.new_derivative()
        return super().vector_field(ls)
