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

sigmoid = autojit(_sigmoid)
score_l2 = autojit(_score_l2)

# --
# Model re-implementations

def tfidf_svc_predict(x, lookup):
    ngrams = word_ngrams(x.lower())
    weights = [(ngrams.count(z), ) + lookup['coef'][z] for z in set(ngrams) if z in lookup['coef']]
    
    score = lookup['intercept']
    if weights:
        score += score_l2(weights)
    
    return sigmoid(score * lookup['calibration_coef'] + lookup['calibration_intercept'])


def tfidf_svc_predict_multi(x, lookup):
    ngrams = word_ngrams(x.lower())
    weights = [(ngrams.count(z), ) + lookup['coef'][z] for z in set(ngrams) if z in lookup['coef']]
    
    scores = copy(lookup['intercept'])
    if weights:
        for i in range(len(scores)):
            scores[i] += score_l2(weights, i)
            scores[i] = sigmoid(scores[i] * lookup['calibration_coef'][i] + lookup['calibration_intercept'][i])
    
    return scores
