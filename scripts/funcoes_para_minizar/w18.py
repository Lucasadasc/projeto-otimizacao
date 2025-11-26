from .w4 import w4
from .w15 import w15

# w18=-w15+w4;
def w18(x, y):
    return -w15(x, y) + w4(x, y)
