# native imports
import os

# Bayesian Optimization
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# local imports
from main import find_ants


def loss_function(lr, batch, delta):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PATH_MODEL = os.path.join(ROOT, "models")
    PATH_OUT = os.path.join(ROOT, "out")

    return -find_ants(
        weights="model_0.717.pt",
        data="peptone_sucrose",
        epochs=20,
        batch=int(batch),
        lr=lr,
        inference=False,
        demo=False,
        path_model=PATH_MODEL,
        path_out=PATH_OUT,
        delta=delta,
    )


def main():
    # config
    params = {}
    params["lr"] = (1e-5, 1e-1)
    params["batch"] = (1, 64) # RATE = 4
    params["delta"] = (3, 20)


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
    print("Next point to probe is:", next_point_to_probe)
    target = loss_function(**next_point_to_probe)
    print("Found the target value to be:", target)
    optimizer.register(
        params=next_point_to_probe,
        target=target,
    )

if __name__ == "__main__":
    main()
