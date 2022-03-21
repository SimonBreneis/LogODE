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

    def vector_field(self, ls, compute_norm=False):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :param compute_norm: If True, additionally computes the norm of the vector field
        :return: Solution on partition points
        """
        deg = ls.n_levels()

        if not compute_norm or self.norm is None:
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

    def extend(self, level_y):
        """
        Extends a vector field to a full vector field.
        :return: The full vector field
        """
        return VectorField(self.f, norm=self.norm)


class VectorFieldNumeric(VectorField):
    def __init__(self, f, dim_y, h=1e-06, norm=ta.l1):
        """
        Initialization.
        :param f: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param dim_y: Dimension of the solution y (dimension of the output of f)
        :param h: Step size in numerical differentiation
        :param norm: Vector space norm used for estimating the norm of f
        """
        super().__init__(f, norm)
        self.h = h
        self.dim_y = dim_y

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
            result += (self.derivative(y + self.h / 2 * direction, dx[i, ...])
                       - self.derivative(y - self.h / 2 * direction, dx[i, ...])) / self.h
        return result

    def extend(self, level_y):
        """
        Extends a vector field to a full vector field.
        :param level_y: The level to which to vector field should be extended (the level the solution should have)
        :return: The full vector field
        """

        def f(y, x):
            """
            The (first/zeroth derivative of the) extended vector field.
            :param y:
            :param x:
            :return:
            """
            y_tens = ta.array_to_tensor(y, self.dim_y)
            fyx = ta.tensor_algebra_embedding(self.f[0](y_tens[1], x), y_tens.n_levels())
            return (y_tens * fyx).to_array()

        if self.dim_y == 1:
            extended_dim = level_y + 1
        else:
            extended_dim = int(np.around((self.dim_y**(level_y+1) - 1)/(self.dim_y - 1)))
        return VectorFieldNumeric(f=[f], dim_y=extended_dim, h=self.h, norm=self.norm)


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
        self.f_num = [sp.lambdify(self.variables, self.f[i], modules='numpy') for i in range(len(f))]

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
        self.f_num.append(sp.lambdify(self.variables, self.f[-1], modules='numpy'))
        return None

    def derivative(self, y, dx):
        """
        Computes the len(dx.shape)-th term in the log-ODE method.
        :param y: The current position of the approximated solution
        :param dx: A level of the log-signature of the underlying path
        :return: The len(dx.shape)-th term in the log-ODE method
        """
        rank = len(dx.shape)
        return np.tensordot(self.f_num[rank-1](*list(y)), dx, axes=rank)

    def vector_field(self, ls, compute_norm=False):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :param compute_norm: If True, additionally computes the vector field norm
        :return: Solution on partition points
        """
        deg = ls.n_levels()
        while len(self.f) < deg:
            self.new_derivative()
        return super().vector_field(ls, compute_norm)

    def extend(self, level_y):
        """
        Extends a vector field to a full vector field.
        :return: The full vector field
        """
        dim_x = self.f[0].shape[1]
        dim_y = self.f[0].shape[0]
        if dim_y == 1:
            n_new_vars = level_y
        else:
            n_new_vars = (dim_y**(level_y+1)-1)/(dim_y-1) - dim_y
        new_vars = list(sp.symbols('a0:%d' % n_new_vars))
        if dim_y == 1 and level_y == 2:
            variables = [new_vars[0]] + self.variables + [new_vars[1]]
        else:
            variables = [new_vars[0]] + self.variables + new_vars[1:]

        y_tens = ta.array_to_tensor(sp.Array(variables), dim_y)
        '''
        f = [[0]*len(variables)]*dim_x
        for i in range(dim_x):
            f[i] = list((y_tens * ta.tensor_algebra_embedding(self.f[0][:, i], level_y)).to_array())
        # f = sp.transpose(sp.Array(f))
        '''
        f = [[0]*dim_x for _ in range(len(variables))]
        for i in range(dim_x):
            f_temp = list((y_tens * ta.tensor_algebra_embedding(self.f[0][:, i], level_y)).to_array())
            for j in range(len(variables)):
                f[j][i] = f_temp[j]
        f = sp.Array(f)
        return VectorFieldSymbolic(f=[f], norm=self.norm, variables=variables)
