import numpy as np
import scipy
from scipy import integrate, special
from esig import tosig as ts
import matplotlib.pyplot as plt
import p_var
import time
import logode as lo
import examples as ex
import roughpath as rp
import vectorfield as vf
import cProfile
import sympy as sp


def l1(x):
    return np.sum(np.abs(x))


class Tensor:
    def __init__(self, tensor):
        self.tensor = tensor

    def __copy__(self):
        return Tensor([self.tensor[0], *[self.tensor[i].copy() for i in range(1, len(self))]])

    def __add__(self, other):
        pass

    def __mul__(self, other):
        pass

    def __len__(self):
        return len(self.tensor)

    def n_levels(self):
        return len(self) - 1

    def project(self, N):
        pass

    def dim(self):
        if len(self) <= 1:
            return 1
        else:
            return len(self.tensor[1])

    def inverse(self):
        N = self.n_levels()
        if N == 0:
            result = self.__copy__()
            result.tensor[0] = 1 / result.tensor[0]
            return result

        result = self * (-1 / self.tensor[0])
        result.tensor[0] = 1
        if N >= 2:
            factor = result.__copy()
            factor.tensor[0] = 0
            product = factor.__copy__()

            for i in range(2, N + 1):
                product = product * factor
                result = result + product

        return result / self.tensor[0]

    def exp(self):
        """
        Assumes that self.tensor[0] = 0.
        :return:
        """
        result = self.__copy__()
        result.tensor[0] += 1
        curr_tensor_prod = self
        for k in range(2, len(self)):
            curr_tensor_prod = curr_tensor_prod * self / k
            result = result + curr_tensor_prod
        return result

    def log(self):
        """
        Assumes that self.tensor[0] = 1.
        :return:
        """
        factor = self.__copy__()
        factor.tensor[0] -= 1
        result = factor.__copy__()
        curr_tensor_prod = factor
        for k in range(2, len(self)):
            curr_tensor_prod = curr_tensor_prod * factor
            result = result + curr_tensor_prod / k * (-1)**(k+1)
        return result

    def extend_sig(self, N):
        if N <= self.n_levels():
            return self.project(N)
        return self.log().project(N).exp()


class SymbolicalTensor(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)

    def __copy__(self):
        return SymbolicalTensor([self.tensor[0], *[self.tensor[i].copy() for i in range(1, len(self))]])

    def __add__(self, other):
        if isinstance(other, SymbolicalTensor):
            return SymbolicalTensor([self.tensor[i] + other.tensor[i] for i in range(min(len(self), len(other)))])
        return other * self

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if isinstance(other, SymbolicalTensor):
                N = min(self.n_levels(), other.n_levels())
                x = self.tensor
                y = other.tensor
                if N == 0:
                    return SymbolicalTensor([x[0] * y[0]])
                if N == 1:
                    return SymbolicalTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0]])
                if N == 2:
                    return SymbolicalTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                            x[0] * y[2] + sp.tensorproduct(x[1], y[1]) + x[2] * y[0]])
                if N == 3:
                    return SymbolicalTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                            x[0] * y[2] + sp.tensorproduct(x[1], y[1]) + x[2] * y[0],
                            x[0] * y[3] + sp.tensorproduct(x[1], y[2]) + sp.tensorproduct(x[2], y[1]) + x[3] * y[0]])
                # the following code also works for N in {0, 1, 2, 3}, but is considerably slower
                result = trivial_tens_sym(self.dim(), N)
                for i in range(N+1):
                    for j in range(i+1):
                        result.tensor[i] = result.tensor[i] + sp.tensorproduct(x[j], y[i - j])
                return result
        # assume other is scalar
        return SymbolicalTensor([other*self.tensor[i] for i in range(len(self))])

    def project(self, N):
        if len(self) >= N+1:
            return SymbolicalTensor([self.tensor[i] for i in range(N+1)])
        else:
            new_tens = trivial_sig_sym(self.dim(), N)
            for i in range(len(self)):
                new_tens.tensor[i] = self.tensor[i]
            return new_tens

    def to_numerical_tensor(self):
        return NumericalTensor([float(self.tensor[0]), *[np.array(self.tensor[i]).astype(np.float64) for i in range(1, len(self))]])


class NumericalTensor(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor)

    def __copy__(self):
        return NumericalTensor([self.tensor[0], *[self.tensor[i].copy() for i in range(1, len(self))]])

    def __add__(self, other):
        if isinstance(other, SymbolicalTensor):
            other = other.to_numerical_tensor()
        return NumericalTensor([self.tensor[i] + other.tensor[i] for i in range(min(len(self), len(other)))])

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if isinstance(other, SymbolicalTensor):
                other = other.to_numerical_tensor()
            N = min(self.n_levels(), other.n_levels())
            x = self.tensor
            y = other.tensor
            if N == 0:
                return NumericalTensor([x[0] * y[0]])
            if N == 1:
                return NumericalTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0]])
            if N == 2:
                return NumericalTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                        x[0] * y[2] + np.einsum('i,j->ij', x[1], y[1]) + x[2] * y[0]])
            if N == 3:
                return NumericalTensor([x[0] * y[0], x[0] * y[1] + x[1] * y[0],
                        x[0] * y[2] + np.einsum('i,j->ij', x[1], y[1]) + x[2] * y[0],
                        x[0] * y[3] + np.multiply.outer(x[1], y[2]) + np.multiply.outer(x[2], y[1]) + x[3] * y[0]])
            # the following code also works for N in {0, 1, 2, 3}, but is considerably slower
            return NumericalTensor([np.sum(np.array([np.multiply.outer(x[j], y[i - j]) for j in range(i + 1)]), axis=0) for i in range(N+1)])
        # assume other is scalar
        return NumericalTensor([other*self.tensor[i] for i in range(len(self))])

    def project(self, N):
        if len(self) >= N+1:
            return NumericalTensor([self.tensor[i] for i in range(N+1)])
        else:
            new_tens = trivial_sig_num(self.dim(), N)
            for i in range(len(self)):
                new_tens.tensor[i] = self.tensor[i]
            return new_tens


def trivial_tens_num(dim, N):
    return NumericalTensor([np.zeros([dim] * i) for i in range(N + 1)])


def trivial_sig_num(dim, N):
    """
    Returns the signature up to degree N of a trivial dim-dimensional path.
    :param dim:
    :param N:
    :return:
    """
    result = trivial_tens_num(dim, N)
    result.tensor[0] = 1.
    return result


def trivial_tens_sym(dim, N):
    return SymbolicalTensor([sp.Array([dim]*i) for i in range(N+1)])


def trivial_sig_sym(dim, N):
    result = trivial_tens_sym(dim, N)
    result.tensor[0] = sp.Integer(1)
    return result


def sig(x, N):
    """
    Computes the signature of the path x (given as a vector) up to degree N.
    :param x:
    :param N:
    :return:
    """
    if N == 0:
        return NumericalTensor([1.])
    if N == 1:
        return NumericalTensor([1., x[-1, :] - x[0, :]])
    dim = x.shape[1]
    sig_vec = ts.stream2sig(x, N)
    indices = [int((dim ** (k + 1) - 1) / (dim - 1) + 0.1) for k in range(N + 1)]
    indices.insert(0, 0)
    res = [sig_vec[indices[i]:indices[i + 1]].reshape([dim] * i) for i in range(N + 1)]
    res[0] = float(res[0])
    return NumericalTensor(res)


def logsig(x, N):
    return sig(x, N).log()
