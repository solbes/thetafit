import numpy as np
import pandas as pd
from .utils import vectorize_ssfun


def accept(acc_prob):

    if acc_prob <= 0:
        return False
    elif acc_prob >= 1:
        return True
    elif acc_prob > np.random.random():
        return True
    else:
        return False


def sample(ssfun, data, params, options):

    ssfun_vec, names_opt, th_no_opt = vectorize_ssfun(ssfun, data, params)

    oldpar = np.array([par.init for par in params if par.target])
    oldss = ssfun_vec(oldpar)

    bounds = [(par.minimum, par.maximum) for par in params if par.target]

    print('Sampling these parameters:\nname\tstart\t[min,max]')
    for name, init, bound in zip(names_opt, oldpar, bounds):
        print('{0}\t{1}\t[{2},{3}]'.format(name, init, bound[0], bound[1]))

    chain = np.zeros((options.nsimu, len(oldpar)))
    chain[0, :] = oldpar

    prop_cov = options.qcov

    for isimu in range(1, options.nsimu):
        newpar = np.random.multivariate_normal(oldpar, prop_cov)
        newss = ssfun_vec(newpar)

        acc_prob = np.exp(-0.5 * (newss - oldss))

        if accept(acc_prob):
            chain[isimu, :] = newpar
            oldpar = newpar.copy()
            oldss = newss
        else:
            chain[isimu, :] = oldpar

        adapt = np.mod(isimu, options.adaptint) == 0 \
            if options.adaptint > 0 else False

        if adapt:
            prop_cov = np.cov(chain[0:isimu, :], rowvar=False)
            if len(oldpar) == 1:
                prop_cov = prop_cov[np.newaxis, np.newaxis]

    chain_dict = dict(zip(names_opt, chain.T))

    return pd.DataFrame({**chain_dict, **th_no_opt})
