import numpy as np
from utils import schwefel, rosenbrock

# w4=sqrt(r.^2+z.^2)+Fobj;
def w4(Fobj, x, y):
    r = rosenbrock(x, y)
    z = schwefel(x, y)

    return np.sqrt(r**2 + z**2) + Fobj
