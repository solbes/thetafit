import numpy as np
from scipy.optimize import minimize
from .utils import vectorize_ssfun


def optimize(ssfun, data, params, **kwargs):

    ssfun_vec, names_opt, th_no_opt = vectorize_ssfun(ssfun, data, params)

    init_opt = np.array([par.init for par in params if par.target])
    bounds = [(par.minimum, par.maximum) for par in params if par.target]

    res = minimize(ssfun_vec, init_opt, bounds=bounds, **kwargs)

    return {**dict(zip(names_opt, res.x)), **th_no_opt}, res
