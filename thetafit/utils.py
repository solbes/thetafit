import numpy as np


class McmcOptions(object):

    def __init__(
            self,
            nsimu=10000,
            adaptint=100,
            printint=None,
            qcov=None,
    ):

        self.nsimu = nsimu
        self.adaptint = adaptint
        self.printint = adaptint if printint is None else printint
        self.qcov = qcov


class Parameter(object):

    def __init__(
            self, name, init, minimum=-np.inf, maximum=np.inf, target=True
    ):

        self.name = name
        self.init = init
        self.minimum = minimum
        self.maximum = maximum
        self.target = target


def vectorize_ssfun(ssfun, data, params):

    names_opt = [par.name for par in params if par.target]

    assert len(names_opt) > 0, "no target parameters"

    names_no_opt = [par.name for par in params if not par.target]
    init_no_opt = [par.init for par in params if not par.target]
    th_no_opt = dict(zip(names_no_opt, init_no_opt))

    def ssfun_vec(th):
        th_dict = {**dict(zip(names_opt, th)), **th_no_opt}
        return ssfun(th_dict, data)

    return ssfun_vec, names_opt, th_no_opt


def jacob(fun, x, params, hrel=1e-6, habs=1e-12, **fun_kwargs):

    th0 = {par.name: par.init for par in params}
    y0 = fun(th0, x, **fun_kwargs)

    names_target = [par.name for par in params if par.target]

    n = len(y0)
    p = len(names_target)
    J = np.zeros((n, p))

    for i, name in enumerate(names_target):
        hi = max(hrel*th0[name], habs)

        thi_plus, thi_minus = th0.copy(), th0.copy()
        thi_plus[name] += hi
        thi_minus[name] -= hi

        J[:, i] = (
            fun(thi_plus, x, **fun_kwargs) - fun(thi_minus, x, **fun_kwargs)
        ) / (2*hi)

    return J
