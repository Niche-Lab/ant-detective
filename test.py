import numpy as np

pred = []
for i in range(3):
    pred.append(np.array([[1, 2], [3, 4], [5, 6]]))

pred

p = pred[0].shape[1]
np.array(pred, dtype=object).reshape((-1, p)).astype(float)
