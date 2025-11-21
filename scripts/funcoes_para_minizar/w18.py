from .w4 import w4
from .w15 import w15

# w18=-w15+w4;
def w18(r, z, Fobj, r1):
    return -w15(r1, z) + w4(r, z, Fobj)
