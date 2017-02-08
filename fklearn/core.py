"""
    core.py
    
    Some functions for running sklearn models much faster on individual 
    observations.
    
    Regular `sklearn` uses lots of vectorized operations via `numpy`. 
    However this appears to introduce a lot of overhead, which hurts 
    us a lot when we're evaluating on single messages as in our pipeline.
    
    By writing some of these functions we can also reduce a lot of the 
    dependencies and (possibly) make the models more portable between languages.
"""

import numpy as np
from twutils import word_ngrams
from numba import autojit

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _score_l2(X):
    """ 
        Feature counts -> predictions
        X consists of (count, idf_weight, model_weight) triplets
        
        !! Need to test
    """
    
    num = 0
    denom = 0
    for x in X:
        num += (x[0] * x[1] * x[2])
        denom += ((x[0] * x[1]) ** 2)
    
    denom = np.sqrt(denom)
    
    return num / denom

sigmoid = autojit(_sigmoid)
score_l2 = autojit(_score_l2)

def tfidf_svc_predict(x, lookup):
    ngrams = word_ngrams(x)
    weights = [(ngrams.count(z), ) + lookup['coef'][z] for z in set(ngrams) if z in lookup['coef']]
    
    score = lookup['intercept']
    if weights:
        score += score_l2_(weights)
    
    return sigmoid(score * lookup['calibration_coef'] + lookup['calibration_intercept'])

