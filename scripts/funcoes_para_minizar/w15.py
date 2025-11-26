import numpy as np
from .utils import schwefel, r1

# w15=z.*exp(cos(r1));
def w15(x, y):
    z = schwefel(x, y)

    x_normalizado = x/250
    y_normalizado = y/250
    r1_val = r1(x_normalizado, y_normalizado)

    return z * np.exp(np.cos(r1_val))
