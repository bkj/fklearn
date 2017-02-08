### fklearn

Some functions for running sklearn models much faster on individual 
observations.

Regular `sklearn` uses lots of vectorized operations via `numpy`. 
However this appears to introduce a lot of overhead, which hurts 
us a lot when we're evaluating on single messages as in our pipeline.

By writing some of these functions we can also reduce a lot of the 
dependencies and (possibly) make the models more portable between languages.
