import numpy as np
X=np.zeros((1,4))
print(X)
print(len(X[0]))
X_b = np.hstack([np.ones((len(X), 1)), X])
print(X_b)