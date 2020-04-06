from scipy import optimize
import numpy as np
from itertools import product

def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0, n_trials=100):

    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # n_trials random data sets are generated and fitted
    ps = []
    for i in range(n_trials):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap 


def calc_sup_inf(f, x, pars, errs):
    assert len(pars) == len (errs), f"`pars` and `errs` should have the same length"
    assert isinstance (errs, np.ndarray), f"errs should be a np.ndarray, instead is {type(errs)}"
    assert isinstance (pars, np.ndarray), f"pars should be a np.ndarray, instead is {type(pars)}"        
    # tutte le combinazioni di parametri + errori
    comb_errori = [pars+p*errs for p in product([1, -1, 0], repeat=len(errs))]
    # on each point check which is the max and the min err
    max_f = f(x, pars)
    min_f = f(x, pars)
    for i, xi in enumerate(x):
        
        for c in comb_errori:
            temp_val = f(xi, c)
            if np.isinf(temp_val):
                continue
            if temp_val>max_f[i]:
                max_f[i] = temp_val
            elif temp_val<min_f[i]:
                min_f[i] = temp_val
    return max_f, min_f

def sigmoid(x, p0, p1=None, p2=None):
    if p1==None and p2==None:
        
        crescita = p0[0]
        massimo = p0[1]
        flesso = p0[2]
    else:
        crescita = p0
        massimo = p1
        flesso = p2
    return massimo / (1 + np.exp(-(x-flesso)*crescita)) 

def gompertz(x, p0, p1=None, p2=None, p3=None):
    # ex. pars [100000,9,0.09,1]
    if p1==None and p2==None and p3==None:
        return p0[0]* np.exp(-p0[1]*np.exp(-p0[2]*(x-p0[3])))
    else:
        return p0* np.exp(-p1*np.exp(-p2*(x-p3)))
    
    
def mydiff(arr):
    toret = []
    for i, v in enumerate(arr):
        if i == 0:
            toret.append(v)
            continue
        toret.append(v-arr[i-1])
    return toret

def calc_sup_inf_bay(predictions):
    y_sup = np.empty_like(predictions[0])
    y_inf = np.empty_like(predictions[0])
    y_mean = np.empty_like(predictions[0])

    for i, arr  in enumerate(predictions.T):
        mean = np.mean(arr)
        y_mean[i] = mean
        arr_sup = arr[arr>mean]
        arr_inf = arr[arr<mean]
        sigma_sup = np.std(arr_sup)
        sigma_inf = np.std(arr_inf)
        y_sup[i] = mean + sigma_sup
        y_inf[i] = mean - sigma_inf
        
    return y_mean, y_sup, y_inf


def my_g(x, mu, sigma, scale, skew = 'L'):
    t = (x-mu)/sigma
    if skew == 'L':
        return scale * np.exp(t - np.exp(t))  # L
    elif skew == 'R':
        return scale * np.exp(-(t + np.exp(-t)))  # R
    else:
        raise ValueError("skew must be \'L\' or \'R\'")
        
def scala_x(x, inf, sup):
    assert inf<sup, "inf deve essere minore di sup"
    xp = (x - x[0])
    xp = xp / xp[-1]
    scaled =  xp * (sup-inf)
    return scaled + inf