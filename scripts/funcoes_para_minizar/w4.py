import numpy as np
from .utils import schwefel, rosenbrock, fobj

# w4=sqrt(r.^2+z.^2)+Fobj;
def w4(x, y):
    z = schwefel(x, y)
    valor_fobj = fobj(x, y)

    x = x/250
    y = y/250
    r = rosenbrock(x, y)

    return np.sqrt(r**2 + z**2) + valor_fobj
