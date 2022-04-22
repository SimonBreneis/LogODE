import numpy as np
import sympy as sp
import tensoralgebra as ta
import oneform as of


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
        self.f = [f[i] for i in range(self.exact_der)]  # vector field derivatives
        self.ordinary_derivatives = [f[0]]  # ordinary
        self.norm = norm
        self.global_norm = [0.]  # the ith entry is a (lower) estimate of the norm \|f^{\circ i}\|_\infty
        self.local_norm = [0.]
        self.flow_vf = None
        self.adj_vf = None
        self.full_vf = []
        self.dim_x = -1
        self.dim_y = -1

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
        pass

    def apply(self, ls, compute_norm=False):
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
        pass

    def adjoin(self):
        """
        Extends the vector field f to f_ext so that instead of solving dy = f(y)dx, one solves dz = f_ext(z)dx,
        where z = (x, y).
        :return: The vector field which is used to solve for z = (x, y)
        """
        pass

    def flow(self):
        """
        Returns the vector field associated with the flow, namely if f is in L(R^d, Lip(R^e, R^e)), this is the
        vector field f_1 in L(R^(d+e), Lip(R^(d+e+e^2), R^(d+e+e^2)) with
        f_1(x, y, h) = [[Id, 0], [0, Id], [f'(y), 0]].
        :return: The above vector field
        """
        pass

    def one_form(self):
        """
        Suppose one wants to compute z = int f(y) dx. This can be computed by first considering
        z = int f_1 ((x, y)) d(x, y), where we use the joint rough path (x, y) instead, and where f_1 is the
        obvious extension of f so that this works. Then, we can further extend f_1 to f_2, which is a vector field such
        that if we solve dv = f_2(v) d(x, y), then the solution is v = (x, y, z).
        :return:
        """
        pass


class VectorFieldNumeric(VectorField):
    def __init__(self, f, dim_x, dim_y, h=1e-06, norm=ta.l1):
        """
        Initialization.
        :param f: List, first element is the vector field. Further elements may be the derivatives of the vector field,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the vector field, given as a matrix-valued function
            that takes as input only the position vector y
        :param dim_x: Dimension of the driving path x
        :param dim_y: Dimension of the solution y (dimension of the output of f)
        :param h: Step size in numerical differentiation
        :param norm: Vector space norm used for estimating the norm of f
        """
        super().__init__(f, norm)
        self.h = h
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.flow_one_form = self.compute_flow_one_form()  # Suppose that f is in L(R^d, Lip(R^e, R^e)). Returns the
        # one-form g in L(R^(d+e), Lip(R^(d+e), R^(exe))) with g(x, y) = (f'(y) 0).

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
        result = 0
        for i in range(self.dim_x):
            vec = np.zeros(self.dim_x)
            vec[i] = 1.
            direction = self.f[0](y, vec)
            result += (self.derivative(y + self.h / 2 * direction, dx[i, ...])
                       - self.derivative(y - self.h / 2 * direction, dx[i, ...])) / self.h
        return result

    def extend(self, level_y):
        while len(self.full_vf) <= level_y:
            self.full_vf.append(None)
        if self.full_vf[level_y] is None:
            def f(y, x):
                """
                The (first/zeroth derivative of the) extended vector field.
                :param y: Current signature of the solution (in array form)
                :param x: Log-signature of the driving path
                :return: The vector field applied to the log-signature
                """
                y_tens = ta.array_to_tensor(y, self.dim_y)
                fyx = ta.tensor_algebra_embedding(self.f[0](y_tens[1], x), y_tens.n_levels())
                return (y_tens * fyx).to_array()

            if self.dim_y == 1:
                extended_dim = level_y + 1
            else:
                extended_dim = int(np.around((self.dim_y**(level_y+1) - 1)/(self.dim_y - 1)))
            self.full_vf[level_y] = VectorFieldNumeric(f=[f], dim_x=self.dim_x, dim_y=extended_dim, h=self.h,
                                                       norm=self.norm)
        return self.full_vf[level_y]

    def adjoin(self):
        if self.adj_vf is None:
            def f(y, x):
                """
                The (first/zeroth derivative of the) vector field that adjoins the path x to the solution y
                :param y: Current value of the solution
                :param x: Log-signature of the driving path
                :return: The vector field applied to the log-signature
                """
                result = np.empty(self.dim_x + self.dim_y)
                result[:self.dim_x] = x
                result[self.dim_x:] = self.f[0](y[self.dim_x:], x)
                return result

            self.adj_vf = VectorFieldNumeric(f=[f], dim_x=self.dim_x, dim_y=self.dim_x+self.dim_y, h=self.h,
                                             norm=self.norm)
        return self.adj_vf

    def compute_flow_one_form(self):
        def g(z, dz):
            y = z[self.dim_x:]
            dx = dz[:self.dim_x]
            result = np.empty(self.dim_y * self.dim_y)
            for i in range(self.dim_y):
                e_i = np.zeros(self.dim_y)
                e_i[i] = 1
                result[i*self.dim_y:(i+1)*self.dim_y] = (self.f[0](y + self.h/2 * e_i, dx)
                                                         - self.f[0](y - self.h/2 * e_i, dx))/self.h
            return result
        return g

    def flow(self):
        if self.flow_vf is None:
            def flow_f(v, dz):
                z = v[:(self.dim_x + self.dim_y)]
                result = np.empty(self.dim_x + self.dim_y + self.dim_y*self.dim_y)
                result[:(self.dim_x + self.dim_y)] = dz
                result[(self.dim_x + self.dim_y):] = self.flow_one_form(z, dz)
                return result
            self.flow_vf = VectorFieldNumeric(f=[flow_f], dim_x=self.dim_x + self.dim_y,
                                              dim_y=self.dim_x + self.dim_y + self.dim_y * self.dim_y, h=self.h,
                                              norm=self.norm)
        return self.flow_vf

    def one_form(self):
        def f_1(y, x):
            return self.f[0](y[self.dim_x:], x[:self.dim_x])

        def f_2(y, x):
            result = np.empty(self.dim_x + 2*self.dim_y)
            result[:(self.dim_x + self.dim_y)] = x
            result[(self.dim_x + self.dim_y):] = f_1(y[:(self.dim_x + self.dim_y)], x)
            return result

        return VectorFieldNumeric(f=[f_2], dim_x=self.dim_x+self.dim_y, dim_y=self.dim_x + 2*self.dim_y, h=self.h,
                                  norm=self.norm)


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
        self.dim_x = self.f[0].shape[1]
        self.dim_y = self.f[0].shape[0]

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

    def apply(self, ls, compute_norm=False):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :param compute_norm: If True, additionally computes the vector field norm
        :return: Solution on partition points
        """
        deg = ls.n_levels()
        while len(self.f) < deg:
            self.new_derivative()
        return super().apply(ls, compute_norm)

    def extend(self, level_y):
        while len(self.full_vf) <= level_y:
            self.full_vf.append(None)
        if self.full_vf[level_y] is None:
            if self.dim_y == 1:
                n_new_vars = level_y
            else:
                n_new_vars = (self.dim_y**(level_y+1)-1)/(self.dim_y-1) - self.dim_y
            new_vars = list(sp.symbols('a0:%d' % n_new_vars))
            if len(new_vars) == 2:
                variables = [new_vars[0]] + self.variables + [new_vars[1]]
            else:
                variables = [new_vars[0]] + self.variables + new_vars[1:]

            y_tens = ta.array_to_tensor(sp.Array(variables), self.dim_y)
            f = [[0]*self.dim_x for _ in range(len(variables))]
            for i in range(self.dim_x):
                f_temp = list((y_tens * ta.tensor_algebra_embedding(self.f[0][:, i], level_y)).to_array())
                for j in range(len(variables)):
                    f[j][i] = f_temp[j]
            f = sp.Array(f)
            self.full_vf[level_y] = VectorFieldSymbolic(f=[f], norm=self.norm, variables=variables)
        return self.full_vf[level_y]

    def adjoin(self):
        if self.adj_vf is None:
            f = [[0]*self.dim_x for _ in range(self.dim_x + self.dim_y)]
            for i in range(self.dim_x):
                f[i][i] = 1
            for i in range(self.dim_x, self.dim_x+self.dim_y):
                for j in range(self.dim_x):
                    f[i][j] = self.f[0][i-self.dim_x, j]
            f = sp.Array(f)
            new_vars = list(sp.symbols('b0:%d' % self.dim_x))
            self.adj_vf = VectorFieldSymbolic(f=[f], norm=self.norm, variables=new_vars + self.variables)
        return self.adj_vf

    def flow(self):
        if self.flow_vf is None:
            der_f_list = [sp.diff(self.f[0], self.variables[i]).tolist() for i in range(len(self.variables))]
            der_f_list_sum = []
            for summand in der_f_list:
                for i in range(len(summand)):
                    summand[i] = summand[i] + [0] * self.dim_y
                der_f_list_sum = der_f_list_sum + summand
            flow_one_form = sp.Array(der_f_list_sum)
            identity = np.eye(self.dim_x + self.dim_y).tolist()
            vf = sp.Array(identity + flow_one_form.tolist())
            new_vars_x = list(sp.symbols('e0:%d' % self.dim_x))
            new_vars_h = list(sp.symbols('f0:%d' % (self.dim_y*self.dim_y)))
            self.flow_vf = VectorFieldSymbolic(f=[vf], norm=self.norm,
                                               variables=new_vars_x + self.variables + new_vars_h)
        return self.flow_vf

    def one_form(self):
        f_1 = [[0]*(self.dim_x + self.dim_y) for _ in range(self.dim_y)]
        for i in range(self.dim_y):
            for j in range(self.dim_x):
                f_1[i][j] = self.f[0][i, j]

        f_2 = [[0]*(self.dim_x + self.dim_y) for _ in range(self.dim_x + 2*self.dim_y)]
        for i in range(self.dim_x + self.dim_y):
            f_2[i][i] = 1
        for i in range(self.dim_x + self.dim_y, self.dim_x + 2*self.dim_y):
            for j in range(self.dim_x + self.dim_y):
                f_2[i][j] = f_1[i-self.dim_x-self.dim_y][j]
        f_2 = sp.Array(f_2)
        new_vars = list(sp.symbols('c0:%d' % (self.dim_x + self.dim_y)))
        return VectorFieldSymbolic(f=[f_2], norm=self.norm, variables=new_vars + self.variables)


def matrix_multiplication_vector_field(d, norm=None):
    """
    Returns the vector field f in L(R^(d^2), Lip(R^(d^2), R^(d^2))) given by f(y)x = yx, where yx is the matrix
    multiplication of two dxd matrices.
    :param d: Dimension of the vector field
    :param norm: The norm that should be used
    :return: The vector field
    """
    variables = list(sp.symbols('g0:%d' % (d*d)))
    f = sp.MutableDenseNDimArray(np.zeros((d*d, d*d)))
    for i in range(d):
        for j in range(d):
            submatrix = [[0]*d for _ in range(d)]
            for k in range(d):
                submatrix[k][k] = variables[i*d + j]
            f[i*d:(i+1)*d, j*d:(j+1)*d] = submatrix
    f = sp.Array(f.tolist())
    return VectorFieldSymbolic(f=[f], norm=norm, variables=variables)
