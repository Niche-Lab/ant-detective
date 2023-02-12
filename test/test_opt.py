# native imports
import os

# Bayesian Optimization
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

def loss_function(a, b, c):
    return a**2 + (b - c)

def main():
    # config
    params = {}
    params["a"] = (-10, 10)
    params["b"] = (-10, 10)
    params["c"] = (-10, 10)

    optimizer = BayesianOptimization(f=loss_function, pbounds=params)
    
    # check if logger exists
    path_log = os.path.join("out", "logs.json")
    if not os.path.exists(path_log):
        with open(path_log, "w") as f:
            pass
    else:
        load_logs(optimizer, logs=[path_log])

    logger = JSONLogger(path=path_log, reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    utility = UtilityFunction()
    next_point_to_probe = optimizer.suggest(utility)
    print("Current probe is:", next_point_to_probe, flush=True)
    target = loss_function(**next_point_to_probe)
    print("Found the target value to be:", target, flush=True)
    optimizer.register(
        params=next_point_to_probe,
        target=target,
    )
    next_point_to_probe = optimizer.suggest(utility)
    print("Next point to probe is:", next_point_to_probe, flush=True)
  
if __name__ == "__main__":
    main()
