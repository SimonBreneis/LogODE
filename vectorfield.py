import numpy as np
import roughpath as rp


class VectorField:
    def __init__(self, h=1e-06, norm=rp.l1):
        self.h = h
        self.norm = norm
