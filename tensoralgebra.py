import numpy as np
import sympy as sp
from esig import tosig as ts


def l1(x):
    """
    Implementation of the l^1-norm.
    :param x: A numpy-array or a Tensor
    :return: The l^1-norm
    """
    if isinstance(x, Tensor):
        return np.sum(np.array([x.norm(i, l1) for i in range(1, x.n_levels()+1)]))
    return np.sum(np.abs(x))


class Tensor:
    def __init__(self, tensor):
        """
        Initialization
        :param tensor: A list, the ith element is the ith level of the tensor
        """
        self.tensor = tensor

    def __copy__(self):
        return Tensor([self[0], *[self[i].copy() for i in range(1, len(self))]])

    def __add__(self, other):
        """
        Add two Tensors.
        :param other: Tensor to be added
        :return: A new Tensor which is the sum
        """
        pass

    def __mul__(self, other):
        """
        Multiply two tensors or a tensor with a scalar.
        :param other: Either a tensor or a scalar
        :return: A new Tensor which is the product
        """
        pass

    def __len__(self):
        """
        Length of the Tensor, the number of levels + 1.
        :return: Length
        """
        return len(self.tensor)

    def __str__(self):
        return self.tensor.__str__()

    def __getitem__(self, item):
        """
        Returns the item-th level of the Tensor.
        :param item: The level
        :return: The item-th level
        """
        return self.tensor[item]

    def __setitem__(self, key, value):
        """
        Sets the key-th level of the tensor (in place).
        :param key: The level
        :param value: The new key-th level
        :return: Nothing
        """
        self.tensor[key] = value

    def n_levels(self):
        """
        Returns the number of levels of the tensor.
        :return: Number of levels
        """
        return len(self) - 1

    def project(self, N):
        """
        Returns a new tensor which is the projection onto the first N levels. If N is greater than self.n_levels(),
        returns a new tensor which is the extension (extended by zero-levels).
        :param N: Level of new tensor
        :return: A new projected/extended tensor
        """
        pass

    def dim(self):
        """
        Returns the dimension of the underlying vector space.
        :return: Dimension of the underlying vector space
        """
        if len(self) <= 1:
            return 1
        else:
            return len(self[1])

    def inverse(self):
        """
        Returns a new tensor which is the inverse. Assumes that self[0] = 1.
        :return: A new tensor which is the inverse
        """
        N = self.n_levels()
        if N == 0:
            result = self.__copy__()
            result[0] = 1 / result[0]
            return result

        result = self * (-1 / self[0])
        result[0] = 1
        if N >= 2:
            factor = result.__copy__()
            factor[0] = 0
            product = factor.__copy__()

            for i in range(2, N + 1):
                product = product * factor
                result = result + product

        return result * (1/self[0])

    def exp(self):
        """
        Returns the exponent of the tensor. Assumes that self[0] = 0.
        :return: A new tensor which is the exponent
        """
        result = self.__copy__()
        result[0] += 1
        curr_tensor_prod = self
        for k in range(2, len(self)):
            curr_tensor_prod = curr_tensor_prod * self * (1/k)
            result = result + curr_tensor_prod
        return result

    def log(self):
        """
        Returns the logarithm of the tensor. Assumes that self[0] = 1.
        :return: A new tensor which is the logarithm
        """
        factor = self.__copy__()
        factor[0] -= 1
        result = factor.__copy__()
        curr_tensor_prod = factor
        for k in range(2, len(self)):
            curr_tensor_prod = curr_tensor_prod * factor
            result = result + curr_tensor_prod * ((-1)**(k+1) / k)
        return result

    def extend_sig(self, N):
        """
        Assumes that self is a signature of a rough path. Extends this signature up to level N in a natural way, so as
        to remain geometric. If N is smaller than self.n_levels(), projects instead.
        :param N: The level of the new signature tensor
        :return: A new naturally extended signature tensor
        """
        if N <= self.n_levels():
            return self.project(N)
        return self.log().project(N).exp()

    def norm(self, N, norm=l1):
        """
        Computes the norm of the Nth level of the tensor.
        :param N: The level
        :param norm: The norm
        :return: The norm of the Nth level
        """
        return norm(self[N])

    def append(self, tensor_level):
        """
        Appends (in place) tensor_level to self as the next level (i.e. as level self.n_levels()+1).
        :param tensor_level: The next level of the tensor
        :return: Nothing
        """
        self.tensor.append(tensor_level)

    def to_array(self):
        """
        Transforms the Tensor into an array.
        :return: The array
        """
        return None


class SymbolicTensor(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)

    def __copy__(self):
        return SymbolicTensor([self[0], *[self[i].copy() for i in range(1, len(self))]])

    def __add__(self, other):
        if isinstance(other, SymbolicTensor):
            return SymbolicTensor([self[i] + other[i] for i in range(min(len(self), len(other)))])
        return other * self

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if isinstance(other, SymbolicTensor):
                N = min(self.n_levels(), other.n_levels())
                x = self
                y = other
                if N == 0:
                    return SymbolicTensor([x[0] * y[0]])
                if N == 1:
                    return SymbolicTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0]])
                if N == 2:
                    return SymbolicTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                                           x[0] * y[2] + sp.tensorproduct(x[1], y[1]) + x[2] * y[0]])
                if N == 3:
                    return SymbolicTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                                           x[0] * y[2] + sp.tensorproduct(x[1], y[1]) + x[2] * y[0],
                                           x[0] * y[3] + sp.tensorproduct(x[1], y[2]) + sp.tensorproduct(x[2], y[1]) + x[3] * y[0]])
                # the following code also works for N in {0, 1, 2, 3}, but is considerably slower
                result = trivial_tens_sym(self.dim(), N)
                for i in range(N+1):
                    for j in range(i+1):
                        result[i] = result[i] + sp.tensorproduct(x[j], y[i - j])
                return result
        # assume other is scalar
        return SymbolicTensor([other * self[i] for i in range(len(self))])

    def project(self, N):
        if len(self) >= N+1:
            return SymbolicTensor([self.tensor[i] for i in range(N + 1)])
        else:
            new_tens = trivial_sig_sym(self.dim(), N)
            for i in range(len(self)):
                new_tens.tensor[i] = self.tensor[i]
            return new_tens

    def to_numeric_tensor(self):
        return NumericTensor([float(self.tensor[0]), *[np.array(self.tensor[i]).astype(np.float64) for i in range(1, len(self))]])

    def simplify(self):
        self.tensor = [sp.simplify(self.tensor[i]) for i in range(len(self))]

    def to_array(self):
        """
        Transforms the tensor into an array.
        :return: The array
        """
        dimension = self.dim()
        levels = self.n_levels()
        length = (dimension**(levels+1) - 1)/(dimension-1)
        result = sp.Array([0]*length)
        index = 0
        next_index = 1
        for level in range(levels):
            result[index:next_index] = self[level]
            index = next_index
            next_index += dimension**(level+1)
        return result


class NumericTensor(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)

    def __copy__(self):
        return NumericTensor([self.tensor[0], *[self.tensor[i].copy() for i in range(1, len(self))]])

    def __add__(self, other):
        if isinstance(other, SymbolicTensor):
            other = other.to_numeric_tensor()
        return NumericTensor([self.tensor[i] + other.tensor[i] for i in range(min(len(self), len(other)))])

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if isinstance(other, SymbolicTensor):
                other = other.to_numeric_tensor()
            N = min(self.n_levels(), other.n_levels())
            x = self.tensor
            y = other.tensor
            if N == 0:
                return NumericTensor([x[0] * y[0]])
            if N == 1:
                return NumericTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0]])
            if N == 2:
                return NumericTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                                      x[0] * y[2] + np.einsum('i,j->ij', x[1], y[1]) + x[2] * y[0]])
            if N == 3:
                return NumericTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                                      x[0] * y[2] + np.einsum('i,j->ij', x[1], y[1]) + x[2] * y[0],
                                      x[0] * y[3] + np.multiply.outer(x[1], y[2]) + np.multiply.outer(x[2], y[1]) + x[3] * y[0]])
            # the following code also works for N in {0, 1, 2, 3}, but is considerably slower
            return NumericTensor([np.sum(np.array([np.multiply.outer(x[j], y[i - j]) for j in range(i + 1)]), axis=0) for i in range(N + 1)])
        # assume other is scalar
        return NumericTensor([other * self.tensor[i] for i in range(len(self))])

    def project(self, N):
        if len(self) >= N+1:
            return NumericTensor([self.tensor[i] for i in range(N + 1)])
        else:
            new_tens = trivial_sig_num(self.dim(), N)
            for i in range(len(self)):
                new_tens.tensor[i] = self.tensor[i]
            return new_tens

    def to_array(self):
        """
        Transforms the tensor into an array.
        :return: The array
        """
        dimension = self.dim()
        levels = self.n_levels()
        length = (dimension**(levels+1) - 1)/(dimension-1)
        result = np.empty(length)
        index = 0
        next_index = 1
        for level in range(levels):
            result[index:next_index] = self[level]
            index = next_index
            next_index += dimension**(level+1)
        return result


def trivial_tens_num(dim, N):
    """
    Returns a trivial (all zero) NumericTensor.
    :param dim: The dimension of the underlying vector space
    :param N: The level of the tensor
    :return: The tensor
    """
    res = NumericTensor([np.zeros([dim] * i) for i in range(N + 1)])
    res[0] = 0
    return res


def trivial_sig_num(dim, N):
    """
    Returns the signature up to degree N of a trivial dim-dimensional path.
    :param dim: The dimension of the path
    :param N: The level of the signature
    :return: The signature tensor, instance of NumericTensor
    """
    result = trivial_tens_num(dim, N)
    result[0] = 1.
    return result


def trivial_tens_sym(dim, N):
    """
    Returns a trivial (all zero) SymbolicTensor.
    :param dim: The dimension of the underlying vector space
    :param N: The level of the tensor
    :return: The tensor
    """
    res = SymbolicTensor([sp.Array([0] * int(dim**i), (dim,)*i) for i in range(N + 1)])
    res[0] = sp.Integer(0)
    return res


def trivial_sig_sym(dim, N):
    """
    Returns the signature up to degree N of a trivial dim-dimensional path.
    :param dim: The dimension of the path
    :param N: The level of the signature
    :return: The signature tensor, instance of SymbolicTensor
    """
    result = trivial_tens_sym(dim, N)
    result.tensor[0] = sp.Integer(1)
    return result


def sig(x, N):
    """
    Computes the signature of the path x (given as a vector) up to degree N.
    :param x: The path
    :param N: The level of the signature
    :return: The signature, instance of NumericTensor
    """
    if N == 0:
        return NumericTensor([1.])
    if N == 1:
        return NumericTensor([1., x[-1, :] - x[0, :]])
    dim = x.shape[1]
    sig_vec = ts.stream2sig(x, N)
    indices = [int((dim ** (k + 1) - 1) / (dim - 1) + 0.1) for k in range(N + 1)]
    indices.insert(0, 0)
    res = [sig_vec[indices[i]:indices[i + 1]].reshape([dim] * i) for i in range(N + 1)]
    res[0] = float(res[0])
    return NumericTensor(res)


def logsig(x, N):
    """
    Computes the log-signature of the path x (given as a vector) up to degree N.
    :param x: The path
    :param N: The level of the log-signature
    :return: The log-signature, instance of NumericTensor
    """
    return sig(x, N).log()


def tensor_algebra_embedding(x, N):
    """
    Natural embedding of V^{otimes k} into T(V).
    :param x: The vector
    :param N: Level of the resulting tensor
    :return: The (truncated) tensor
    """
    level = len(x.shape)
    if level == 0:
        dim = 1
    elif level == 1:
        dim = len(x)
    else:
        dim = x.shape[0]
    if isinstance(x, np.ndarray):
        res = trivial_tens_num(dim, N)
    else:
        res = trivial_tens_sym(dim, N)
    if N >= level:
        res[level] = x
    return res


def sig_first_level_num(x_1, N):
    """
    Returns the signature associated to an increment x_1.
    :param x_1: Increment
    :param N: The level of the signature
    :return: The signature tensor, instance of NumericTensor
    """
    path = np.zeros((2, len(x_1)))
    path[1, :] = x_1
    return sig(path, N)


def array_to_tensor(x, dim):
    """
    Converts a numpy (or sympy) array to a NumericTensor (or SymbolicTensor, respectively).
    :param x: The array
    :param dim: The dimension of the underlying vector space (the dimension of the first component)
    :return: The Tensor
    """
    tensor = []
    index = 0
    next_index = 1
    level = 0
    while next_index <= len(x):
        tensor.append(x[index:next_index])
        index = next_index
        next_index += dim**(level+1)
        level += 1
    if isinstance(x, np.ndarray):
        return NumericTensor(tensor)
    return SymbolicTensor(tensor)


def tensor_multiplication_on_arrays(x, y, dim_x, dim_y):
    """
    Given two arrays x, y, interprets them as Tensors and computes the tensor product x*y.
    :param x: First array (numpy or sympy)
    :param y: Second array (numpy or sympy)
    :param dim_x: The dimension of the underlying vector space of x (the dimension of its first component)
    :param dim_y: The dimension of the underlying vector space of y (the dimension of its first component)
    :return: The tensor product x*y, again in array form
    """
    return (array_to_tensor(x, dim_x) * array_to_tensor(y, dim_y)).to_array()


def extend_path(x, N, x_init=None):
    """
    Given a path x, computes its signature path, i.e. the rough path associated to x.
    :param x: The path
    :param N: The level of the signature
    :param x_init: If True, computes x[0] * S(x)_{0,j}. If False, computes S(x)_{0,j}. If instance of Tensor,
        computes x_init * S(x)_{0,j}
    :return: The extended path, instance of a list of NumericTensor
    """
    length = x.shape[0]
    dim = x.shape[1]
    S_x = [trivial_sig_num(dim, N) for _ in range(length)]
    if isinstance(x_init, SymbolicTensor):
        S_x[0] = x_init.to_numeric_tensor().project(N)
    elif isinstance(x_init, NumericTensor):
        S_x[0] = x_init.project(N)
    elif x_init:
        S_x[0][1] = x[0, :]
    for i in range(1, length):
        S_x[i][1] = x[i, :] - x[i-1, :]
        S_x[i] = S_x[i-1] * S_x[i]
    return S_x
