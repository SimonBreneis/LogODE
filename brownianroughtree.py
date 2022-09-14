import time
import numpy as np
import tensoralgebra as ta
import roughpath as rp
import os


def interpolate_Brownian_path(path, dt, has_time):
    new_path = np.empty((path.shape[0], path.shape[1]*2-1))
    if has_time:
        new_path[0, :] = np.linspace(path[0, 0], path[0, -1], new_path.shape[1])
        new_path[1:, ::2] = path[1:, :]
        new_path[1:, 1::2] = (path[1:, :-1] + path[1:, 1:]) / 2 \
            + np.random.normal(0, np.sqrt(dt) / 2, (path.shape[0] - 1, path.shape[1] - 1))
    else:
        new_path[:, ::2] = path
        new_path[:, 1::2] = (path[:, :-1] + path[:, 1:]) / 2 \
            + np.random.normal(0, np.sqrt(dt) / 2, (path.shape[0], path.shape[1] - 1))
    return new_path


class BrownianRoughNode:
    def __init__(self, tree, parent, path, depth):
        self.tree = tree
        self.parent = parent
        self.depth = depth
        self.is_leave = True
        self.path = path
        self.signature = None
        self.left = None
        self.right = None

    def refine_path(self, level, force=-1):
        if self.is_leave:
            if self.path is None:
                if self.signature is not None:
                    self.path = np.array([np.zeros(self.tree.full_dim), self.signature[1]]).T
                else:
                    if self.tree.has_time:
                        self.path = np.zeros((self.tree.full_dim, 2))
                        self.path[1, 0] = self.tree.T * 2 ** (-self.depth)
                        self.path[1, 1:] = np.random.normal(0, np.sqrt(self.tree.T * 2 ** (-self.depth)), self.tree.dim)
            if force == -1:
                while self.path.shape[1] < 2**(np.fmax(self.depth*(level-1), 0))+1:
                    self.path = interpolate_Brownian_path(self.path,
                                                          dt=self.tree.T / (2 ** self.depth * (self.path.shape[1] - 1)),
                                                          has_time=self.tree.has_time)
            else:
                while self.path.shape[1] < 2 ** force + 1:
                    self.path = interpolate_Brownian_path(self.path,
                                                          dt=self.tree.T / (2 ** self.depth * (self.path.shape[1] - 1)),
                                                          has_time=self.tree.has_time)
        else:
            self.left.refine_path(level)
            self.right.refine_path(level)

    def sig(self, N):
        if self.signature is not None and self.signature.n_levels() >= N:
            return self.signature.project_level(N)
        if not self.is_leave:
            self.signature = self.left.sig(N) * self.right.sig(N)
            return self.signature
        self.refine_path(N)
        self.signature = ta.sig(self.path.transpose(), N)
        return self.signature

    def split(self):
        if self.is_leave:
            if self.path.shape[1] == 2:
                self.path = interpolate_Brownian_path(self.path, self.tree.T/(2**self.depth),
                                                      has_time=self.tree.has_time)
            half = int((self.path.shape[1] + 1) / 2)
            self.left = BrownianRoughNode(tree=self.tree, parent=self, path=self.path[:, :half], depth=self.depth + 1)
            self.right = BrownianRoughNode(tree=self.tree, parent=self, path=self.path[:, half - 1:],
                                           depth=self.depth + 1)
            self.is_leave = False
            self.path = None

    def get_child(self, left):
        if self.is_leave:
            self.split()
        if left:
            return self.left
        return self.right

    def get_left(self):
        return self.get_child(True)

    def get_right(self):
        return self.get_child(False)

    def project_space(self, indices, new_tree, new_parent):
        if self.is_leave:
            new_node = BrownianRoughNode(new_tree, new_parent, self.path[indices, :], self.depth)
        else:
            new_node = BrownianRoughNode(new_tree, new_parent, None, self.depth)
            new_node.left = self.left.project_space(indices, new_tree, new_node)
            new_node.right = self.right.project_space(indices, new_tree, new_node)
            new_node.is_leave = False
        return new_node

    def get_all_leaves(self):
        if self.is_leave:
            return [self]
        else:
            return self.left.get_all_leaves() + self.right.get_all_leaves()

    def save(self, directory):
        if self.signature is not None:
            self.signature.save(directory + '/sig.npy')
        if self.is_leave and self.path is not None:
            with open(directory + '/path.npy') as f:
                np.save(f, self.path)
        elif not self.is_leave:
            os.mkdir(directory + '/0')
            os.mkdir(directory + '/1')
            self.left.save(directory + '/0')
            self.right.save(directory + '/1')


class BrownianRoughTree(rp.RoughPath):
    def __init__(self, dim, T, has_time):
        self.dim = dim
        self.T = T
        self.has_time = has_time
        if has_time:
            self.full_dim = dim + 1
            path = np.zeros((dim+1, 2))
            path[0, 1] = T
            path[1:, 1] = np.random.normal(0, np.sqrt(T), dim)
        else:
            self.full_dim = dim
            path = np.zeros((dim, 2))
            path[:, 1] = np.random.normal(0, np.sqrt(T), dim)
        self.root = BrownianRoughNode(tree=self, parent=None, path=path, depth=0)
        super().__init__(p=2.1, var_steps=15, norm=ta.l1, x_0=path[:, 0])

    def sig(self, s, t, N):
        s = s/self.T
        t = t/self.T
        dt = t-s
        depth = int(np.around(-np.log2(dt)))
        k = int(np.fmin(np.fmax(np.around(s * 2**depth), 0), 2**depth-1))
        if np.abs(depth + np.log2(dt)) > 0.01 or np.abs(np.fmin(np.fmax(s * 2**depth, 0), 2**depth-1) - k) > 0.01:
            print('Problem!')
        k_string = '0'
        if depth > 0:
            k_string_temp = '{0:b}'.format(k)
            k_string = '0' * (depth - len(k_string_temp)) + k_string_temp

        current_node = self.root
        for i in range(depth):
            if k_string[i] == '0':
                current_node = current_node.get_left()
            else:
                current_node = current_node.get_right()

        return current_node.sig(N)

    def project_space(self, indices):
        new_has_time = indices[0] == 0
        new_tree = BrownianRoughTree(dim=len(indices) - new_has_time, T=self.T, has_time=new_has_time)
        new_tree.root = self.root.project_space(indices, new_tree, None)
        return new_tree

    def get_all_leaves(self):
        return self.root.get_all_leaves()

    def save(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory, exist_ok=True)
        with open(directory + '/info.npy', 'wb') as f:
            np.save(f, np.array([self.dim, self.T, 1 if self.has_time else 0]))
        os.mkdir(directory + '/root')
        self.root.save(directory + '/root')


def initialize_brownian_rough_tree(dim=2, T=1, has_time=False, depth=10, accuracy=10, N=4, delete=True):
    tree = BrownianRoughTree(dim=dim, T=T, has_time=has_time)
    times = np.linspace(0, T, 2**depth+1)
    for i in range(len(times)-1):
        print(f'{i} of {len(times)}')
        tree.sig(times[i], times[i+1], 1)
    leaves = tree.get_all_leaves()
    i = 0
    init_time = time.perf_counter()
    tic = init_time
    for leave in leaves:
        if time.perf_counter() - tic > 10:
            print(f'{i/len(leaves)*100:.2g}% done, {int((time.perf_counter()-init_time)*(len(leaves)-i)/i)} '
                  f'sec remaining')
            tic = time.perf_counter()
        i = i+1
        leave.refine_path(level=1, force=accuracy)
        leave.signature = ta.sig(leave.path.T, N)
        if delete:
            leave.path = None
    tree.root.sig(N)
    return tree


def load_brownian_rough_node(directory, tree=None, parent=None, depth=None):
    node = BrownianRoughNode(tree=tree, parent=parent, path=None, depth=depth)
    if os.path.exists(directory + '/sig.npy'):
        node.signature = ta.load_tensor(directory + '/sig.npy')
    if os.path.exists(directory + '/path.npy'):
        with open(directory + '/path.npy') as f:
            node.path = np.load(f)
    if os.path.isdir(directory + '/0'):
        node.is_leave = False
        new_depth = depth
        if depth is not None:
            new_depth = depth + 1
        node.left = load_brownian_rough_node(directory + '/0', tree=tree, parent=node, depth=new_depth)
        node.right = load_brownian_rough_node(directory + '/1', tree=tree, parent=node, depth=new_depth)
    return node


def load_brownian_rough_tree(directory):
    with open(directory + '/info.npy', 'rb') as f:
        info = np.load(f)
    tree = BrownianRoughTree(dim=int(info[0]), T=info[1], has_time=True if info[2] == 1 else False)
    tree.root = load_brownian_rough_node(directory + '/root', tree=tree, parent=None, depth=0)
    return tree
