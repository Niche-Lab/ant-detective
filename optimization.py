# native imports
import os

# Bayesian Optimization
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

# local imports
from main import find_ants


def loss_function(lr, batch, delta):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    PATH_MODEL = os.path.join(ROOT, "models")
    PATH_OUT = os.path.join(ROOT, "out")

    return -find_ants(
        weights=None,
        data="peptone_sucrose",
        epochs=20,
        batch=int(batch),
        lr=lr,
        inference=False,
        demo=True,
        path_model=PATH_MODEL,
        path_out=PATH_OUT,
        delta=delta,
    )


def main():
    # config
    params = {}
    params["lr"] = (1e-5, 1e-1)
    params["batch"] = (4, 256)
    params["delta"] = (3, 20)
    optimizer = BayesianOptimization(f=loss_function, pbounds=params)

    # logger
    logger = JSONLogger(path=os.path.join("out", "logs.json"))
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # run
    optimizer.maximize(init_points=2, n_iter=2)


if __name__ == "__main__":
    main()
