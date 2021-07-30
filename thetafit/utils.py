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
