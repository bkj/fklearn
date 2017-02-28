
#### Example of how to extract coefficients from a simple model

`lookup` is in the format expected by `fklearn`

```
    vocab = sorted(est.named_steps['vect'].vocabulary_.iteritems(), key=lambda x: x[1])

    coefs = np.vstack([
        est.named_steps['tfidf'].idf_,
        est.named_steps['svc'].coef_,
    ]).T

    lookup = {
        "coef" : dict(zip([v[0] for v in vocab], map(tuple, coefs))),
        "intercept" : float(est.named_steps['svc'].intercept_),
    }
```