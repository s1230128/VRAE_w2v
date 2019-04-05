import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA


with open('../pickle/hidden.pickle', 'rb') as f:
    hs, ls = pickle.load(f)

print(hs[0].shape)
hs = np.stack(hs)
print(hs.shape)
#hs = np.stack([h[0] for h in hs])

xs = np.array([x for x, y in hs])
ys = np.array([y for x, y in hs])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter([xs[0]], [ys[0]], s=1, c=[[1.0, 0.0, 0.0]], label=0)
ax.scatter([xs[1]], [ys[1]], s=1, c=[[0.5, 0.0, 0.0]], label=1)
ax.scatter([xs[2]], [ys[2]], s=1, c=[[0.0, 1.0, 0.0]], label=2)
ax.scatter([xs[3]], [ys[3]], s=1, c=[[0.0, 0.5, 0.0]], label=3)
ax.scatter([xs[4]], [ys[4]], s=1, c=[[0.0, 0.0, 1.0]], label=4)
ax.scatter([xs[5]], [ys[5]], s=1, c=[[0.0, 0.0, 0.5]], label=5)
ax.scatter([xs[6]], [ys[6]], s=1, c=[[0.0, 0.0, 0.0]], label=6)

ax.set_title('third scatter plot')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

fig.show()
plt.savefig('figure.png')
