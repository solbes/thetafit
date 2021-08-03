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


def out_of_bounds(theta, bounds):

    th_out = [(th < bnd[0]) or (th > bnd[1]) for th, bnd in zip(theta, bounds)]
    return any(th_out)


def sample(ssfun, data, params, options):

    ssfun_vec, names_opt, th_no_opt = vectorize_ssfun(ssfun, data, params)
    bounds = [(par.minimum, par.maximum) for par in params if par.target]

    oldpar = np.array([par.init for par in params if par.target])
    assert not out_of_bounds(oldpar, bounds), "initial parameters out of bounds"
    oldss = ssfun_vec(oldpar)

    print('Sampling these parameters:\nname\tstart\t[min,max]')
    for name, init, bound in zip(names_opt, oldpar, bounds):
        print('{0}\t{1}\t[{2},{3}]'.format(name, init, bound[0], bound[1]))

    chain = np.zeros((options.nsimu, len(oldpar)))
    chain[0, :] = oldpar

    sschain = np.zeros(options.nsimu)
    sschain[0] = oldss

    prop_cov = options.qcov

    rej = 0
    rejb = 0
    for isimu in range(1, options.nsimu):

        newpar = np.random.multivariate_normal(oldpar, prop_cov)

        if out_of_bounds(newpar, bounds):
            rej += 1
            rejb += 1
            chain[isimu, :] = oldpar
            sschain[isimu] = oldss
        else:
            newss = ssfun_vec(newpar)
            acc_prob = np.exp(-0.5 * (newss - oldss))

            if accept(acc_prob):
                chain[isimu, :] = newpar
                sschain[isimu] = newss
                oldpar = newpar.copy()
                oldss = newss
            else:
                rej += 1
                chain[isimu, :] = oldpar
                sschain[isimu] = oldss

        adapt = np.mod(isimu, options.adaptint) == 0 \
            if options.adaptint > 0 else False
        if adapt:
            qcov_scale = 2.4**2/len(names_opt)
            prop_cov = qcov_scale*np.cov(chain[0:isimu, :], rowvar=False)
            if len(oldpar) == 1:
                prop_cov = prop_cov[np.newaxis, np.newaxis]

        doprint = np.mod(isimu, options.printint) == 0 \
            if options.printint > 0 else False
        if doprint:
            print('i: %i, rejected: %.1f%%, out of bounds: %.1f%%' %
                  (isimu, rej/isimu*100, rejb/isimu*100))

    chain_dict = dict(zip(names_opt, chain.T))

    # collect results
    results = {
        'N': options.nsimu,
        'accepted': (options.nsimu-rej)/options.nsimu*100,
        'rejected': rej/options.nsimu*100,
        'out_of_bounds': rejb/options.nsimu*100,
        'qcov': prop_cov,
        'last': chain[-1, :],
    }

    return results, pd.DataFrame({**chain_dict, **th_no_opt}), sschain
