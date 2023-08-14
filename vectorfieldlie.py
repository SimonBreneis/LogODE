import numpy as np
import sympy as sp
import tensoralgebra as ta
import esig


class VectorField:
    def __init__(self, f, norm=ta.l1, saving_accuracy=0.):
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
        :param saving_accuracy: If non-negative, stores previously evaluated vector field derivatives in a dictionary,
            and reuses them when the vector field is again evaluated at a point that is at most saving_accuracy away (in
            the specified norm). If saving_accuracy = 0., this implies that the new evaluation point must agree (as a
            numpy array with floats) completely with the previous evaluation point. If saving_accuracy is negative, no
            saving or lookup is performed. Changing the parameter after initialization can lead to undesired behaviour
            and is not recommended.
        """
        self.f = f.copy()  # vector field derivatives
        self.saving_accuracy = saving_accuracy
        self.rounding_digits = None if saving_accuracy <= 0 else int(np.around(-np.log10(saving_accuracy)))
        self.val_dictionary = {}
        self.norm = norm
        self.flow_vf = None
        self.adj_vf = None
        self.full_vf = []
        self.dim_x = -1
        self.dim_y = -1

    def get_degree_of_ls(self, ls):
        """
        Given the log-signature ls, determines the degree / level of the log-signature.
        :param ls: The log-signature, a numpy array
        :return: The degree N of the log-signature
        """
        N = 1
        while len(ls) != esig.logsigdim(self.dim_x, N):
            N = N + 1
        return N

    def get_key(self, y):
        """
        Returns the key of the value dictionary.
        :param y: Point of evaluation
        :return: The key for the dictionary
        """
        if self.rounding_digits is None:
            return y.tobytes()
        return np.round(y, self.rounding_digits).tobytes()

    def get_val(self, y):
        """
        Retrieves the value of the vector field derivatives from the dictionary
        :param y: Point of evaluation
        :return: The vector field derivatives at y. If nothing had been saved, returns an empty list
        """
        if self.saving_accuracy >= 0:
            dict_key = self.get_key(y)
            return self.val_dictionary.get(dict_key, [])

    def save_val(self, y, vec_field):
        """
        Saves the value of the vector field derivatives to the dictionary.
        :param y: Point of evaluation
        :param vec_field: The computed vector field derivatives
        :return: True if vec_field was saved, False otherwise
        """
        if self.saving_accuracy >= 0:
            previous_val = self.get_val(y)
            if len(vec_field) >= len(previous_val):
                dict_key = self.get_key(y)
                self.val_dictionary[dict_key] = vec_field
                return True
        return False

    def eval(self, y, N):
        """
        Evaluates all vector field derivatives up to level N.
        :param y: Point of evaluation
        :param N: Maximal derivative
        :return: List of vector field derivatives: [f^{circ 1}, ..., f^{circ N}]
        """
        pass

    def apply(self, ls):
        """
        Computes the vector field used in the Log-ODE method.
        :param ls: The log-signature of the driving path up to level deg
        :return: Solution on partition points
        """
        N = self.get_degree_of_ls(ls)
        return lambda y: ls @ self.eval(y, N)

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
    def __init__(self, f, dim_x, dim_y, h=1e-06, norm=ta.l1, saving_accuracy=0.):
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
        :param saving_accuracy: If non-negative, stores previously evaluated vector field derivatives in a dictionary,
            and reuses them when the vector field is again evaluated at a point that is at most saving_accuracy away (in
            the specified norm). If saving_accuracy = 0., this implies that the new evaluation point must agree (as a
            numpy array with floats) completely with the previous evaluation point. If saving_accuracy is negative, no
            saving or lookup is performed. Changing the parameter after initialization can lead to undesired behaviour
            and is not recommended.
        """
        super().__init__(f, norm, saving_accuracy)
        self.h = h
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.flow_one_form = self.compute_flow_one_form()  # Suppose that f is in L(R^d, Lip(R^e, R^e)). Returns the
        # one-form g in L(R^(d+e), Lip(R^(d+e), R^(exe))) with g(x, y) = (f'(y) 0).

    def derivative(self, y, dx, rank):
        if rank <= self.exact_der:
            return self.f[rank - 1](y, dx)
        result = 0
        for i in range(self.dim_x):
            vec = np.zeros(self.dim_x)
            vec[i] = 1.
            direction = self.f[0](y, vec)
            result += (self.derivative(y + self.h / 2 * direction, dx[i, ...], rank - 1)
                       - self.derivative(y - self.h / 2 * direction, dx[i, ...], rank - 1)) / self.h
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
                extended_dim = int(np.around((self.dim_y ** (level_y + 1) - 1) / (self.dim_y - 1)))
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

            self.adj_vf = VectorFieldNumeric(f=[f], dim_x=self.dim_x, dim_y=self.dim_x + self.dim_y, h=self.h,
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
                derivative = (self.f[0](y + self.h/2 * e_i, dx) - self.f[0](y - self.h/2 * e_i, dx)) / self.h
                result[(slice(i, i + self.dim_y * self.dim_y, self.dim_y),)] = derivative
            return result
        return g

    def flow(self):
        if self.flow_vf is None:
            def flow_f(v, dz):
                z = v[:(self.dim_x + self.dim_y)]
                result = np.empty(self.dim_x + self.dim_y + self.dim_y * self.dim_y)
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
            result = np.empty(self.dim_x + 2 * self.dim_y)
            result[:(self.dim_x + self.dim_y)] = x
            result[(self.dim_x + self.dim_y):] = f_1(y[:(self.dim_x + self.dim_y)], x)
            return result

        return VectorFieldNumeric(f=[f_2], dim_x=self.dim_x+self.dim_y, dim_y=self.dim_x + 2*self.dim_y, h=self.h,
                                  norm=self.norm)


class VectorFieldSymbolic(VectorField):
    def __init__(self, f, norm=ta.l1, variables=None, saving_accuracy=0.):
        """
        Initialization.
        :param f: List of sympy arrays. There are x_dim arrays, where x_dim is the dimension of the rough path x. Each
            array takes y_dim inputs and has a length of y_dim, where y_dim is the dimension of the solution path y.
        :param norm: Vector space norm used for estimating the norm of f
        :param variables: The sympy variables with respect to which f is defined
        :param
        :param saving_accuracy: If non-negative, stores previously evaluated vector field derivatives in a dictionary,
            and reuses them when the vector field is again evaluated at a point that is at most saving_accuracy away (in
            the specified norm). If saving_accuracy = 0., this implies that the new evaluation point must agree (as a
            numpy array with floats) completely with the previous evaluation point. If saving_accuracy is negative, no
            saving or lookup is performed. Changing the parameter after initialization can lead to undesired behaviour
            and is not recommended.
        """
        super().__init__(f, norm, saving_accuracy)
        self.variables = variables
        self.dim_x = len(self.f)
        self.dim_y = len(self.f[0])
        self.f_lie = {}  # symbolically computed Lie brackets
        self.f_lie_full = {}  # symbolically computed Lie brackets up to a certain degree
        self.f_num = {}  # numerical functions for evaluating Lie brackets
        self.f_num_full = {}  # numerical functions for evaluating all Lie brackets up to a certain degree
        for i in range(self.dim_x):
            self.f_lie[f'{i + 1}'] = self.f[i]
            self.f_num[f'{i + 1}'] = sp.lambdify(self.variables, self.f[i], modules=['numpy', 'sympy'])
        self.f_lie_full[1] = sp.Array([self.f[i] for i in range(self.dim_x)])
        self.f_num_full[1] = sp.lambdify(self.variables, self.f_lie_full[1], modules=['numpy', 'sympy'])
        self.computed_level = 1  # the level up to which Lie brackets have already been computed

    def new_lie_bracket(self, index):
        """
        Symbolically computes a new Lie bracket with index index. For example, if index = [1,[1,2]], computes the Lie
        bracket [1, [1, 2]].
        :param index: The index of the Lie bracket, a string
        :return: Nothing
        """
        if index in self.f_lie and index in self.f_num:
            return None
        orig_index = index.copy()
        index = index[1:-1]
        first_comma = index.find(',')
        left_index = index[:first_comma]
        right_index = index[first_comma + 1:]
        left_vf = self.f_lie[left_index]
        if right_index not in self.f_lie:
            self.new_lie_bracket(right_index)
        right_vf = self.f_lie[right_index]
        new_element = sp.Array([sp.Integer(0) for _ in range(self.dim_y)])
        for i in range(self.dim_y):
            expression = sp.Integer(0)
            for j in range(self.dim_y):
                expression = expression + left_vf[j] * sp.diff(right_vf[i], self.variables[j]) \
                    + right_vf[j] * sp.diff(left_vf[i], self.variables[j])
            new_element[i] = expression
        self.f_lie[orig_index] = new_element
        self.f_num[orig_index] = sp.lambdify(self.variables, new_element, modules=['numpy', 'sympy'])
        return None

    def compute_all_lie_brackets(self, N):
        """
        Symbolically computes all Lie brackets up to level N.
        :param N: The level up to which Lie brackets should be computed
        :return: Nothing
        """
        if self.computed_level >= N:
            return None
        if self.computed_level < N - 1:
            self.compute_all_lie_brackets(N - 1)
        indices = esig.logsigkeys(self.dim_x, N)
        indices = indices.split(' ')
        for index in indices:
            self.new_lie_bracket(index)
        self.f_lie_full[N] = sp.Array([self.f_lie[index] for index in indices])
        self.f_num_full[N] = sp.lambdify(self.variables, self.f_lie_full[N], modules=['numpy', 'sympy'])
        self.computed_level = N
        return None

    def eval(self, y, N):
        vec_field = self.get_val(y)
        if len(vec_field) < esig.logsigdim(self.dim_x, N):
            vec_field = np.array(self.f_num_full[N](*list(y)))
            self.save_val(y, vec_field)
        return vec_field

    def apply(self, ls):
        N = self.get_degree_of_ls(ls)
        self.compute_all_lie_brackets(N)
        return super().apply(ls)

    def extend(self, level_y):
        while len(self.full_vf) <= level_y:
            self.full_vf.append(None)
        if self.full_vf[level_y] is None:
            if self.dim_y == 1:
                n_new_vars = level_y
            else:
                n_new_vars = (self.dim_y ** (level_y + 1) - 1) / (self.dim_y - 1) - self.dim_y
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
            self.full_vf[level_y] = VectorFieldSymbolic(f=f, norm=self.norm, variables=variables)
        return self.full_vf[level_y]

    def adjoin(self):
        if self.adj_vf is None:
            f = [[0] * self.dim_x for _ in range(self.dim_x + self.dim_y)]
            for i in range(self.dim_x):
                f[i][i] = 1
            for i in range(self.dim_x, self.dim_x + self.dim_y):
                for j in range(self.dim_x):
                    f[i][j] = self.f[0][i - self.dim_x, j]
            f = sp.Array(f)
            new_vars = list(sp.symbols('b0:%d' % self.dim_x))
            self.adj_vf = VectorFieldSymbolic(f=f, norm=self.norm, variables=new_vars + self.variables)
        return self.adj_vf

    def flow(self):
        if self.flow_vf is None:
            der_f_list = [sp.diff(self.f[0], self.variables[i]).tolist() for i in range(len(self.variables))]
            der_f_list_sum = []
            for summand in der_f_list:
                for i in range(len(summand)):
                    summand[i] = summand[i] + [0] * self.dim_y
            for i in range(len(self.variables)):
                for summand in der_f_list:
                    der_f_list_sum = der_f_list_sum + [summand[i]]
            flow_one_form = sp.Array(der_f_list_sum)
            identity = np.eye(self.dim_x + self.dim_y).tolist()
            vf = sp.Array(identity + flow_one_form.tolist())
            new_vars_x = list(sp.symbols('e0:%d' % self.dim_x))
            new_vars_h = list(sp.symbols('f0:%d' % (self.dim_y * self.dim_y)))
            self.flow_vf = VectorFieldSymbolic(f=vf, norm=self.norm, variables=new_vars_x + self.variables + new_vars_h)
        return self.flow_vf

    def one_form(self):
        f_1 = [[0] * (self.dim_x + self.dim_y) for _ in range(self.dim_y)]
        for i in range(self.dim_y):
            for j in range(self.dim_x):
                f_1[i][j] = self.f[0][i, j]

        f_2 = [[0] * (self.dim_x + self.dim_y) for _ in range(self.dim_x + 2 * self.dim_y)]
        for i in range(self.dim_x + self.dim_y):
            f_2[i][i] = 1
        for i in range(self.dim_x + self.dim_y, self.dim_x + 2*self.dim_y):
            for j in range(self.dim_x + self.dim_y):
                f_2[i][j] = f_1[i - self.dim_x - self.dim_y][j]
        f_2 = sp.Array(f_2)
        new_vars = list(sp.symbols('c0:%d' % (self.dim_x + self.dim_y)))
        return VectorFieldSymbolic(f=f_2, norm=self.norm, variables=new_vars + self.variables)


def matrix_multiplication_vector_field(d, e=0, norm=None):
    """
    Returns the vector field f in L(R^(d^2), Lip(R^(d^2), R^(e^2))) given by f(y)x = yx, where yx is the matrix
    multiplication of the exd matrix y and the dxd matrix x.
    :param d: Dimension of the rough path
    :param e: Dimension of the solution. If e == 0, sets e = d
    :param norm: The norm that should be used
    :return: The vector field
    """
    if e == 0:
        e = d
    variables = list(sp.symbols('g0:%d' % (d * e)))
    f = sp.MutableDenseNDimArray(np.zeros((d * e, d * d)))
    for i in range(e):
        for j in range(d):
            submatrix = [[0] * d for _ in range(d)]
            for k in range(d):
                submatrix[k][k] = variables[i * d + j]
            f[i * d:(i + 1) * d, j * d:(j + 1) * d] = submatrix
    f = sp.Array(f.tolist())
    return VectorFieldSymbolic(f=f, norm=norm, variables=variables)
