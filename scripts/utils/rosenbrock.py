# r=100*(y-x.^2).^2+(1-x).^2;
def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2
