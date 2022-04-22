import numpy as np
import sympy as sp
import tensoralgebra as ta
import vectorfield as vf


number_subpartitions_dict = {}
number_partitions_dict = {}
ordered_shuffle_dict = {}
tensorize_dict = {}


def k_vec_to_key(k):
    return " ".join(str(x) for x in k)


def Nn_to_key(N, n):
    return str(N) + ' ' + str(n)


def nkd_to_key(n, k, d):
    return str(n) + ' ' + str(k) + ' ' + str(d)


def number_partitions(N, n):
    """
    Finds all ways of partitioning N into n positive integers, i.e. all tuples (a_1, ..., a_n) with
    a_1 + ... + a_n = N. Includes all orderings of the (a_i), i.e. (1, 2) and (2, 1) are different partitions of 3.
    :param N: Number to be partitioned
    :param n: Number of summands into which N should be partitioned
    :return: All the partitions. Is a list of lists
    """
    if n > N:
        return []
    if n == N:
        return [[1]*n]
    if n == 1:
        return [[N]]
    if N == 3:
        return [[1, 2], [2, 1]]
    if N == 4:
        if n == 2:
            return [[1, 3], [2, 2], [3, 1]]
        if n == 3:
            return [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    if N == 5:
        if n == 2:
            return [[1, 4], [2, 3], [3, 2], [4, 1]]
        if n == 3:
            return [[3, 1, 1], [1, 3, 1], [1, 1, 3], [1, 2, 2], [2, 1, 2], [2, 2, 1]]
        if n == 4:
            return [[1, 1, 1, 2], [1, 1, 2, 1], [1, 2, 1, 1], [2, 1, 1, 1]]

    key = Nn_to_key(N, n)
    if key in number_partitions_dict:
        return number_partitions_dict[key]

    result = [[i] for i in range(1, N+2-n)]

    def update_result(curr_result, remaining_n):
        if remaining_n == 0:
            return curr_result
        new_result = []
        for part in curr_result:
            part_sum = np.sum(np.array(part))
            for i in range(1, N-part_sum-remaining_n+2):
                new_result.append(part + [i])
        return update_result(new_result, remaining_n-1)

    result = update_result(result, n-1)
    number_partitions_dict[key] = result
    return result


def number_subpartitions(N, n):
    """
    Finds all ways of subpartitioning N into n positive integers, i.e. all tuples (a_1, ..., a_n) with
    a_1 + ... + a_n <= N. Includes all orderings of the (a_i), i.e. (1, 2) and (2, 1) are different subpartitions of 7.
    :param N: Number to be subpartitioned
    :param n: Number of summands into which N should be subpartitioned
    :return: All the subpartitions. Is a list of lists
    """
    if n > N:
        return []
    if n == N:
        return [[1]*n]
    if n == 1:
        return [[i] for i in range(1, N+1)]
    if N == 3:
        return [[1, 1], [1, 2], [2, 1]]
    if N == 4:
        if n == 2:
            return [[1, 1], [1, 2], [2, 1], [1, 3], [2, 2], [3, 1]]
        if n == 3:
            return [[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1]]
    if N == 5:
        if n == 2:
            return [[1, 1], [1, 2], [2, 1], [1, 3], [2, 2], [3, 1], [1, 4], [2, 3], [3, 2], [4, 1]]
        if n == 3:
            return [[1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3], [1, 2, 2], [2, 1, 2],
                    [2, 2, 1]]
        if n == 4:
            return [[1, 1, 1, 1], [1, 1, 1, 2], [1, 1, 2, 1], [1, 2, 1, 1], [2, 1, 1, 1]]

    key = Nn_to_key(N, n)
    if key in number_subpartitions_dict:
        return number_subpartitions_dict[key]

    result = number_subpartitions(N - 1, n) + number_partitions(N, n)
    number_subpartitions_dict[key] = result
    return result


def ordered_shuffles(k):
    """
    Finds all ordered shuffles associated to the list k. For a definition of ordered shuffles, see e.g.
    Lyons, Caruana, Lévy - Differential equations driven by rough paths (2004), page 72.
    :param k: Vector determining the ordered shuffles. Is a list
    :return: A list of lists, where each list in the list is an ordered shuffle (i.e. a permutation)
    """
    n_k = len(k)
    K = np.sum(np.array(k))

    if n_k == 1:
        return [[i for i in range(K)]]
    if K == n_k:
        return [[i for i in range(K)]]
    if K == 3:  # then, n_k == 2
        if k[0] == 1:
            return [[0, 1, 2], [1, 0, 2]]
        return [[0, 1, 2]]
    if K == 4:
        if n_k == 2:
            if k[0] == 1:
                return [[0, 1, 2, 3], [1, 0, 2, 3], [2, 0, 1, 3]]
            if k[0] == 2:
                return [[0, 1, 2, 3], [0, 2, 1, 3], [1, 2, 0, 3]]
            return [[0, 1, 2, 3]]
        if k[0] == 2:
            return [[0, 1, 2, 3]]
        if k[1] == 2:
            return [[0, 1, 2, 3], [1, 0, 2, 3]]
        return [[0, 1, 2, 3], [0, 2, 1, 3], [1, 2, 0, 3]]
    if K == 5:
        if n_k == 2:
            if k[0] == 1:
                return [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [2, 0, 1, 3, 4], [3, 0, 1, 2, 4]]
            if k[0] == 2:
                return [[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [0, 3, 1, 2, 4], [1, 2, 0, 3, 4], [1, 3, 0, 2, 4],
                        [2, 3, 0, 1, 4]]
            if k[0] == 3:
                return [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 2, 3, 1, 4], [1, 2, 3, 0, 4]]
            return [[0, 1, 2, 3, 4]]
        if n_k == 3:
            if k[0] == 1:
                if k[1] == 1:
                    return [[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [0, 3, 1, 2, 4], [1, 2, 0, 3, 4], [1, 3, 0, 2, 4],
                            [2, 3, 0, 1, 4]]
                if k[1] == 2:
                    return [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 2, 3, 1, 4], [1, 0, 2, 3, 4], [1, 0, 3, 2, 4],
                            [1, 2, 3, 0, 4], [2, 0, 3, 1, 4], [2, 1, 3, 0, 4]]
                return [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4], [2, 0, 1, 3, 4]]
            if k[0] == 2:
                if k[1] == 1:
                    return [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 2, 3, 1, 4], [1, 2, 3, 0, 4]]
                return [[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [1, 2, 0, 3, 4]]
            if k[0] == 3:
                return [[0, 1, 2, 3, 4]]
        if k[0] == 2:
            return [[0, 1, 2, 3, 4]]
        if k[1] == 2:
            return [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]]
        if k[2] == 2:
            return [[0, 1, 2, 3, 4], [0, 2, 1, 3, 4], [1, 2, 0, 3, 4]]
        return [[0, 1, 2, 3, 4], [0, 1, 3, 2, 4], [0, 2, 3, 1, 4], [1, 2, 3, 0, 4]]

    key = " ".join(str(x) for x in k)
    if key in ordered_shuffle_dict:
        return ordered_shuffle_dict[key]

    if k[-1] == 1:
        result = [s + [K-1] for s in ordered_shuffles(k[:-1])]
        ordered_shuffle_dict[key] = result
        return result

    def update_possible_values(curr_possible_values, additional_params):
        if additional_params == 0:
            return curr_possible_values
        new_possible_values = []
        for v in curr_possible_values:
            for i in range(v[-1]+1, K-additional_params):
                new_possible_values.append(v + [i])
        return update_possible_values(new_possible_values, additional_params-1)

    possible_values = [[i] for i in range(K-k[-1]+1)]
    possible_values = update_possible_values(possible_values, k[-1]-2)

    shuffles = ordered_shuffles(k[:-1])
    all_shuffles = []
    for shuffle in shuffles:
        for value in possible_values:
            new_shuffle = shuffle.copy()
            for val in value:
                for i in range(len(new_shuffle)):
                    if new_shuffle[i] >= val:
                        new_shuffle[i] = new_shuffle[i] + 1
            all_shuffles.append(new_shuffle + value + [K-1])

    ordered_shuffle_dict[key] = all_shuffles
    return all_shuffles


def apply_permutation(g, p):
    """
    Applies the permutation p to the |p|-tensor g.
    :param g: The |p|-tensor
    :param p: The permutation
    :return: The permuted tensor
    """
    if isinstance(g, ta.NumericTensor):
        return g[len(p)].transpose(p)
    return sp.permutedims(g[len(p)], p)


def apply_all_permutations(g, ps):
    """
    Applies all the permutations in ps to g, and sums them all up.
    :param g: Tensor
    :param ps: List of permutations
    :return: The sum of each permutation in ps applied to g
    """
    result = 0
    for p in ps:
        result = result + apply_permutation(g, p)
    return result


def ordered_shuffles_on_tensor(subpartition, g):
    """
    Given a vector k=subpartition, computes the term in
    Lyons, Caruana, Lévy - Differential equations driven by rough paths (2004), equation (4.9).
    :param subpartition: The vector k in the above equation
    :param g: The signature
    :return: The expression on the above equation
    """
    return apply_all_permutations(g, ordered_shuffles(subpartition))


def tensorize(tensor, k):
    """
    Given an n-tensor tensor, writes tensor as a sum of terms a otimes b, where a is a k-tensor and b is a
    (n-k)-tensor.
    :param tensor: The tensor which is to be written as a sum of tensor products
    :param k: Level of the first factors of the summands
    :return: A list of lists of tensors. The lists in the list contain two tensors, a k-tensor and a (n-k)-tensor.
        More precisely, returns [[a_1, b_1], ..., [a_l, b_l]], and tensor = a_1 x b_1 + ... + a_l x b_l.
        May (or may not) return an empty list if the tensor is the 0 tensor!
    """
    n = len(tensor.shape)
    d = tensor.shape[0]
    result = []

    if k == 0:
        return [1, tensor]
    if k == n:
        return [tensor, 1]

    key = nkd_to_key(n, k, d)

    if True:  # 2*k >= n:
        index_init = (slice(None),) * k

        if key in tensorize_dict:
            indices_final = tensorize_dict[key]
        else:
            indices_final_current = [(i,) for i in range(d)]
            indices_final_next = []
            for i in range(n - k - 1):
                for index in indices_final_current:
                    for j in range(d):
                        indices_final_next.append(index + (j,))
                indices_final_current = indices_final_next.copy()
                indices_final_next = []
            indices_final = indices_final_current
            tensorize_dict[key] = indices_final

        for index in indices_final:
            a = tensor[index_init + index]
            if a.any():
                b = np.zeros((d,) * (n - k))
                b[index] = 1
                result.append([tensor[index_init + index], b])

    '''
    if 2*k < n:
        index_final = (slice(None),)*(n-k)

        if key in tensorize_dict:
            indices_init = tensorize_dict[key]
        else:
            indices_init_current = [(i,) for i in range(d)]
            indices_init_next = []
            for i in range(k - 1):
                for index in indices_init_current:
                    for j in range(d):
                        indices_init_next.append(index + (j,))
                    indices_init_current = indices_init_next.copy()
                    indices_init_next = []
            indices_init = indices_init_current
            tensorize_dict[key] = indices_init

        for index in indices_init:
            a = np.zeros((d,)*k)
            a[index] = 1
            result.append([a, tensor[index + index_final]])
    '''

    return result


class OneForm:
    def __init__(self, f, norm=ta.l1):
        """
        Initialization.
        :param f: List, first element is the one-form. Further elements may be the derivatives of the one-form,
            if the derivatives are not specified (i.e. if f is a list of length smaller than deg), the derivatives
            are computed numerically or symbolically
            If the derivatives are specified, then f[i] is the ith derivative (i=0, ..., deg-1), and f[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f[0] is the one-form, given as a matrix-valued function
            that takes as input only the position vector y
        :param norm: Vector space norm used for estimating the norm of f
        """
        self.exact_der = len(f)
        self.f = [f[i] for i in range(self.exact_der)]  # ordinary derivatives
        self.norm = norm
        self.global_norm = [0.]  # the ith entry is a (lower) estimate of the norm \|f^{\circ i}\|_\infty
        self.local_norm = [0.]
        self.dim_x = -1
        self.dim_y = -1

    def reset_local_norm(self):
        """
        Resets the values of self.local_norm to zero.
        :return: Nothing
        """
        self.local_norm = [0.] * len(self.local_norm)

    def derivative(self, y, dx):
        """
        Computes the derivative of the one-form.
        :param y: Point at which the derivative is calculated
        :param dx: Direction to which the vector field is applied, an n-tensor
        :return: An approximation of the n-th derivative
        """
        pass

    def higher_order_part_of_one_form(self, subpartition, y, dx):
        """
        Computes a summand of the n-th level of the integral int self(y) dx, as in
        Lyons, Caruana, Lévy - Differential equations driven by rough paths (2004), Theorem 4.6.
        :param subpartition: The vector k in the above cited formula
        :param y: Current path value
        :param dx: K-th level of the signature of the path (i.e. a K-tensor), where K = |subpartition|
        :return: Approximation of the corresponding summand of the increment of the n-th level of the integral
        """
        pass

    def higher_order_one_form(self, n, y, g):
        """
        Defines the n-th level of the integral int self(y) dx, as in
        Lyons, Caruana, Lévy - Differential equations driven by rough paths (2004), Theorem 4.6.
        :param n: Level of the integral
        :param y: Current path value
        :param g: Signature of the rough path
        :return: Approximation for the increment of the n-th level of the integral
        """
        N = g.n_levels()
        subpartitions = number_subpartitions(N, n)
        result = 0
        for subpartition in subpartitions:
            result = result + self.higher_order_part_of_one_form(subpartition, y, g[np.sum(np.array(subpartition))])
        return result

    def full_one_form(self, n, y, g, lie=True):
        """
        Computes the full approximation of the integral int self(y) dx, as in
        Lyons, Caruana, Lévy - Differential equations driven by rough paths (2004), Theorem 4.6.
        :param n: Level of the approximation
        :param y: Current path value
        :param g: Signature of the rough path
        :param lie: If True, projects onto the Lie group, i.e. returns a Lie group element
        :return: Approximation for the increment of the integral
        """
        result = ta.NumericTensor([1])
        if n >= 1:
            result.append(self.apply(g, compute_norm=False))
        if n >= 2:
            for i in range(2, n+1):
                result.append(self.higher_order_one_form(i, y, g))
        if lie:
            result = result.project_lie()
        return result

    def apply(self, g, compute_norm=False):
        """
        Computes the vector field used in the Log-ODE method.
        :param g: The log-signature of the driving path up to level deg
        :param compute_norm: If True, additionally computes the norm of the vector field
        :return: Solution on partition points
        """
        deg = g.n_levels()

        if not compute_norm or self.norm is None:
            return lambda y: np.sum(np.array([self.derivative(y, g[i]) for i in range(1, deg + 1)]), axis=0)

        def compute_vf_and_norm(y):
            while deg > len(self.local_norm):
                self.local_norm.append(0.)
                self.global_norm.append(0.)
            ls_norms = np.array([g.norm(i, self.norm) for i in range(1, deg + 1)])
            summands = np.array([self.derivative(y, g[i]) for i in range(1, deg + 1)])
            vec_field = np.sum(summands, axis=0)
            for i in range(deg):
                local_local_norm = (self.norm(summands[i]) / ls_norms[i]) ** (1. / (i + 1))
                if local_local_norm > self.local_norm[i]:
                    self.local_norm[i] = local_local_norm
                    if local_local_norm > self.global_norm[i]:
                        self.global_norm[i] = local_local_norm
            return vec_field

        return compute_vf_and_norm


class OneFormNumeric(OneForm):
    def __init__(self, f, dim_x, dim_y, h=1e-06, norm=ta.l1):
        """
        Initialization.
        :param f: List, first element is the one-form. Further elements may be the derivatives of the one-form,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed numerically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the one-form, given as a matrix-valued function
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

    def derivative(self, y, dx):
        N = len(dx.shape)
        if N <= self.exact_der:
            return self.f[N - 1](y, dx)
        result = 0
        for i in range(self.dim_x):
            vec = np.zeros(self.dim_x)
            vec[i] = 1.
            result += (self.derivative(y + self.h / 2 * vec, dx[i, ...])
                       - self.derivative(y - self.h / 2 * vec, dx[i, ...])) / self.h
        return result

    def higher_order_part_of_one_form(self, subpartition, y, dx):
        trans_dx = ordered_shuffles_on_tensor(subpartition, dx)

        def subroutine(k, rdx):
            if len(k) == 0:
                return 1
            if len(k) == 1:
                return self.derivative(y, rdx)
            tensor_sum = tensorize(rdx, k[0])
            result = 0
            for summand in tensor_sum:
                result = result + self.derivative(y, summand[0]) * subroutine(k[1:], summand[1])
            return result

        return subroutine(subpartition, trans_dx)


class OneFormSymbolic(OneForm):
    def __init__(self, f, norm=ta.l1, variables=None):
        """
        Initialization.
        :param f: List, first element is the one-form. Further elements may be the derivatives of the one-form,
            if the derivatives are not specified (i.e. if f_vec is a list of length smaller than deg), the derivatives
            are computed symbolically
            If the derivatives are specified, then f_vec[i] is the ith derivative (i=0, ..., deg-1), and f_vec[i] takes
            as input the position vector y and the (i+1)-tensor dx^(i+1)
            If the derivatives are not specified, then f_vec[0] is the one-form, given as a matrix-valued function
            that takes as input only the position vector y
        :param norm: Vector space norm used for estimating the norm of f
        :param variables: The sympy variables with respect to which f is defined
        """
        super().__init__(f, norm)
        self.variables = variables
        self.f_num = [sp.lambdify(self.variables, self.f[i], modules='numpy') for i in range(len(f))]
        self.higher_order_fs = {}
        self.dim_x = self.f[0].shape[1]
        self.dim_y = self.f[0].shape[0]

    def new_derivative(self):
        """
        Computes the next derivative that has not yet been calculated.
        :return: Nothing
        """
        der_highest_der = sp.Array([sp.diff(self.f[-1], self.variables[i]) for i in range(len(self.variables))])
        self.f.append(der_highest_der)
        self.f_num.append(sp.lambdify(self.variables, self.f[-1], modules='numpy'))
        return None

    def derivative(self, y, dx):
        rank = len(dx.shape)
        return np.tensordot(self.f_num[rank-1](*list(y)), dx, axes=rank)

    def higher_order_part_of_one_form(self, subpartition, y, dx):
        trans_dx = ordered_shuffles_on_tensor(subpartition, dx)
        key = k_vec_to_key(subpartition)
        if key in self.higher_order_fs:
            return np.tensordot(self.higher_order_fs[key](*list(y)), trans_dx, axes=len(trans_dx.shape))

        result = self.f[subpartition[0]-1]
        curr_tensor_level = subpartition[0]+1
        for i in range(1, len(subpartition)):
            add_tensor_level = subpartition[i]+1
            new_tensor_level = curr_tensor_level + add_tensor_level
            permutation = [j for j in range(new_tensor_level)]
            for j in range(1, curr_tensor_level):
                permutation[j] = j+1
            permutation[curr_tensor_level] = 2
            result = sp.permutedims(sp.tensorproduct(result, self.f[subpartition[1]-1]), permutation)
        result = sp.lambdify(self.variables, result, modules='numpy')
        self.higher_order_fs[key] = result
        return np.tensordot(result(*list(y)), trans_dx, axes=len(trans_dx.shape))

    def apply(self, g, compute_norm=False):
        deg = g.n_levels()
        while len(self.f) < deg:
            self.new_derivative()
        return super().apply(g, compute_norm)
