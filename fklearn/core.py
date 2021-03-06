"""
    core.py
"""

from copy import copy
import numpy as np
from twutils import word_ngrams
from numba import autojit

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _score_l2(X, i=0):
    """ 
        Feature counts -> predictions
        X consists of (count, idf_weight, model_weight_1, model_weight_2, ...) tuples
        i is a pointer to which model_weight_* you care about
        
        !! Need to test
    """
    
    num = 0
    denom = 0
    for x in X:
        num += (x[0] * x[1] * x[2 + i])
        denom += ((x[0] * x[1]) ** 2)
    
    denom = np.sqrt(denom)
    
    return num / denom

def _make_weights(ngrams, lookup, sublinear_tf=False):
    if not sublinear_tf:
        weights = [(ngrams.count(z), ) + lookup['coef'][z] for z in set(ngrams) if z in lookup['coef']]
    else:
        weights = [(1 + np.log(ngrams.count(z)), ) + lookup['coef'][z] for z in set(ngrams) if z in lookup['coef']]
    
    return weights

sigmoid = autojit(_sigmoid)
score_l2 = autojit(_score_l2)
make_weights = _make_weights

# --
# Model re-implementations

def tfidf_svc_predict(x, lookup):
    """ Binary classification from coefficient plus intercept """
    params = lookup['params']
    ngrams = word_ngrams(x.lower(), minn=params['minn'], maxn=params['maxn'])
    weights = make_weights(ngrams, lookup, sublinear_tf=params['sublinear_tf'])
    
    score = lookup['intercept']
    if weights:
        score += score_l2(weights)
    
    score = score * lookup['calibration_coef'] + lookup['calibration_intercept']
    return sigmoid(score)


def tfidf_svc_predict_multi(x, lookup):
    """ Multi-class sigmoid classification from coefficient plus intercept """
    params = lookup['params']
    ngrams = word_ngrams(x.lower(), minn=params['minn'], maxn=params['maxn'])
    weights = make_weights(ngrams, lookup, sublinear_tf=params['sublinear_tf'])
    
    scores = copy(lookup['intercept'])
    if weights:
        for i in range(len(scores)):
            scores[i] += score_l2(weights, i)
            scores[i] = sigmoid(scores[i] * lookup['calibration_coef'][i] + lookup['calibration_intercept'][i])
    
    return scores


def tfidf_svr_predict(x, lookup):
    """ Regression from coefficient plus intercept """
    params = lookup['params']
    ngrams = word_ngrams(x.lower(), minn=params['minn'], maxn=params['maxn'])
    weights = make_weights(ngrams, lookup, sublinear_tf=params['sublinear_tf'])
    
    score = lookup['intercept']
    if weights:
        score += score_l2(weights)
    
    return score


def vector_svc_predict(x, lookup):
    params = lookup['params']
    if params['norm']:
        x /= np.sqrt((x ** 2).sum())
    
    raw_score = ((x * lookup['coef']).sum() + lookup['intercept']) 
    cal_score = raw_score * lookup['calibration_coef'] + lookup['calibration_intercept']
    return sigmoid(cal_score)


models = {
    "tfidf_svc_predict" : tfidf_svc_predict,
    "tfidf_svc_predict_multi" : tfidf_svc_predict_multi,
    "tfidf_svr_predict" : tfidf_svr_predict,
    "vector_svc_predict" : vector_svc_predict,
}

# --
# Load model wrapper

def load_model(lookup):
    if 'params' not in lookup:
        raise Exception('!! need `params` to be set in the lookup')
    
    predict_function = models[lookup['params']['model']]
    def _fklearn_load_model_predict(x):
        return predict_function(x, lookup)
    
    return _fklearn_load_model_predict



