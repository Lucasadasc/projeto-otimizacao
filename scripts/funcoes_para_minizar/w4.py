import numpy as np
from .utils import schwefel, rosenbrock, fobj

# w4=sqrt(r.^2+z.^2)+Fobj;
def w4(x, y):
    z = schwefel(x, y)
    valor_fobj = fobj(x, y)

    x_normalizado = x/250
    y_normalizado = y/250
    r = rosenbrock(x_normalizado, y_normalizado)

    return np.sqrt(r**2 + z**2) + valor_fobj
