# PART2
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def high_d_EPMF(d, n):
    matx = np.zeros((n, 1))
    X = []
    for i in range(n):
        diag1 = np.zeros((d, 1))
        diag2 = np.zeros((d, 1))
        for k in range(d):
            diag1[k] = random.randrange(-1, 2, 2)
            diag2[k] = random.randrange(-1, 2, 2)
        matx[i] = math.degrees(math.acos(np.dot(diag1.transpose(), diag2) / float(d)))
    for j in matx:
        X.append(j[0])

    angles = list(set(X))

    def EPMF(x):
        dic_epmf = []
        for a in angles:
            dic_epmf.append(X.count(a))
        return dic_epmf

    minx = np.min(matx)
    maxx = np.max(matx)
    rge = maxx - minx
    mea = np.mean(matx)
    varx = np.var(matx)
    print 'For %d dimensions:\n\n1. The minimun of X is %f,\n2. The maximun of X is %f,\n3. The value range of X is %f,\n4. The mean of X is %f,\n5. The variance of X is %f\n\n\n' % (d, minx, maxx, rge, mea, varx)

    x_axis = angles
    y_axis = EPMF(x_axis)
    if d <= 100: plt.bar(x_axis, y_axis)
    else: plt.bar(x_axis, y_axis, width = 0.1)
    plt.title('Table EPMF for %d Dimensions' % d)
    if d <= 100: plt.xticks(x_axis, rotation = 45)
    else: plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 0.5), rotation = 45)
    plt.xlabel('Angles(degrees)')
    plt.ylabel('EPMF')
    plt.show()


high_d_EPMF(10, 100000)
high_d_EPMF(100, 100000)
high_d_EPMF(1000, 100000)



