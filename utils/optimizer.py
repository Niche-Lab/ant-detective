from bayes_opt import BayesianOptimization
import pandas as pd

class BayesianOptimizer:
    def __init__(self, func, search_bounds, random_state=None):
        self.optimizer = BayesianOptimization(
            f=func,
            pbounds=search_bounds,
            random_state=random_state,
        )

    def maximize(self, init_points=10, n_iter=50):
        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )

    def max(self):
        return self.optimizer.max
    
    def out(self):
        ls_y = [r["target"] for r in self.optimizer.res]
        map_x = [r["params"] for r in self.optimizer.res]
        df_y = pd.Series(ls_y, name="target")
        df_x = pd.DataFrame([[k[i] for i in k.keys()] for k in map_x])
        keys = map_x[0].keys()
        df_merged = pd.concat([df_y, df_x], axis=1)
        df_merged.columns = ["target"] + list(keys)
        return df_merged

    def __repr__(self):
        return f"BayesianOptimizer(func={self.optimizer.f}, pbounds={self.optimizer.pbounds})"


# # example --------------------------
# def black_box_function(x, y):
#     """Function with unknown internals we wish to maximize.

#     This is just serving as an example, for all intents and
#     purposes think of the internals of this function, i.e.: the process
#     which generates its output values, as unknown.
#     """
#     return -x ** 2 - (y - 1) ** 2 + 1

# optimizer = BayesianOptimizer(
#     func=black_box_function,
#     search_bounds={'x': (2, 4), 'y': (-3, 3)},
#     random_state=1,
# )
# optimizer.maximize()
# optimizer.max()
# optimizer.out()
