import sys

#PART 1

import numpy as np
import matplotlib.pyplot as plt
import math
data = np.loadtxt(str(sys.argv[1]), delimiter=',')
#means = np.mean(data, axis = 0)
#data -= means

n = data.shape[0]
d = data.shape[1]

def linear(data):
    # compute kernel matrix
    kmat = np.zeros((n, n))
    i = 0
    for x in data:
        j = 0
        for y in data:
            kmat[(i, j)] = np.dot(x, y.transpose())
            j += 1
        i += 1
    return kmat

def kpca(kmat, alpha):
    # center the kernel matrix
    kmat_cent = np.mat(np.dot(np.dot((np.eye(n)-np.ones((n, n))/n),kmat),
                              (np.eye(n)-np.ones((n, n))/n)))
                              
    # compute the eigenvalues and eigenvectors
    eigval, eigvec = np.linalg.eigh(kmat_cent)
    delst = []
    for x in eigval:
        if x < 0:
            x = 0
        delst.append(x)
    delta = np.array(delst)

    # compute variance for each component
    lamda = delta / n

    # ensure that u has unit length
    for id in range(1030):
        if delta[-id] != 0:
            eigvec[:,-id] = eigvec[:,-id] / np.sqrt(delta[-id])

    # fraction of total variance and choose a smallest r
    sumr = 0
    suml = np.sum(lamda)
    for id in range(1030):
        if sumr / suml < alpha:
            sumr += lamda[-id-1]
            r = abs(-id-1)
        else:
            print 'The fraction of total variance: ', sumr / suml
            #print sumr
            print 'The the smallest r: ', r
            break

    # project
    c1 = eigvec[:,-1]
    c2 = eigvec[:,-2]
    a1 = np.array(np.dot(kmat_cent, c1))
    a2 = np.array(np.dot(kmat_cent, c2))
    
    # plot
    plt.scatter(a1, a2)
    #plt.title('Table %s' % name)
    plt.xlabel('A1')
    plt.ylabel('A2')
    plt.show()

# PCA
def pca(data):
    means = np.mean(data, axis = 0)
    data -= means
    cov_pca = np.cov(data.transpose())
    eigval_p, eigvec_p = np.linalg.eigh(cov_pca)
    u1 = eigvec_p[:,-1]
    u2 = eigvec_p[:,-2]
    a3 = np.array(np.dot(data, u1))
    a4 = np.array(np.dot(data, u2))
    # plot
    plt.scatter(a3, a4)
    #plt.title('Table 2.2')
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.show()


# Gaussian Kernel
def gaussian(data, sigma):
    gkmatrix = np.zeros((n, n))
    gi = 0
    for x in data:
        gj = 0
        for y in data:
            gkmatrix[(gi, gj)] = math.exp((- np.dot(x.transpose(), x) - np.dot(y.transpose(), y) + 2 * np.dot(x.transpose(), y)) / (2 * sigma))
            gj += 1
        gi += 1
    #name = 'Gaussian Kernel Matrix'
    return gkmatrix

kpca(linear(data), 0.95)
pca(data)
kpca(gaussian(data, float(sys.argv[2])), 0.95)




