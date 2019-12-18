import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


data = np.array(((20,30), (60,75), (80,30), (90,70), (95,90), (40,60), (80,90), (30,40), (25,55), (35,25), (80,45), (50,10), (25,80)))
y = np.array((0,1,0,1,1,1,1,0,0,0,1,0,1))

for multiplier in [1, 2, 4, 8]:
    plt.clf()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    print("multiplier = {0}".format(multiplier))
    w = np.array([0.1, 0.3, -20]) * multiplier

    ax.set_xlabel('Report score', size=10)
    ax.set_ylabel('Test score', size=10)

    t0 = np.linspace(0,100,1001)
    t1 = np.linspace(0,100,1001)

    g0, g1 = np.meshgrid(t0, t1)
    # convert two (N x N) matrices into a single (N x N x 2) matrix
    x = np.stack([g0, g1], axis=2)
    # add 1 dimension 2 to transform (N x N x 2) to a (N x N x 3) matrix
    # where the third dimension will change from [x, y] to [x, y, 1]
    x = np.insert(x, 2, 1, axis=2)
    # compute activation: (N x N x 3) ・ (3) -> (N x N)
    activation = np.matmul(x, w)
    g2 = sigmoid(activation)

    # Plot the surface.
    surf = ax.plot_surface(g0, g1, g2, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
