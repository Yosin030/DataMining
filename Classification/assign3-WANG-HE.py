import sys
import numpy as np
import math
import copy
import datetime
import random

#assign3.py FILENAME C eps [linear OR quadratic OR gaussian ]

#import data
dataset = np.loadtxt(str(sys.argv[1]), delimiter = ',')
datetime1 = datetime.datetime.now()
#dataset = np.loadtxt('Concrete_Data_RNorm_Class.txt', delimiter = ',')

n = dataset.shape[0]
d = dataset.shape[1]

tol = 0.00001
#kernel
def kernel(type, data, sigma):
    # if type = linear or quadratic, sigma = 0
    if type == 'linear':
        #linear
        kmat = np.zeros((n, n))
        ki = 0
        for x in data[:,:d-1]:
            kj = 0
            for y in data[:,:d-1]:
                kmat[(ki, kj)] = np.dot(x, y.transpose())
                kj += 1
            ki += 1
        return kmat

    #quadratic
    elif type == 'quadratic':
        kmat = np.zeros((n, n))
        ki = 0
        for x in data[:,:d-1]:
            kj = 0
            for y in data[:,:d-1]:
                kmat[(ki, kj)] = np.dot(x, y.transpose()) ** 2
                kj += 1
            ki += 1
        return kmat

    #gaussian
    elif type == 'gaussian':
        kmat = np.zeros((n, n))
        ki = 0
        for x in data[:,:d-1]:
            kj = 0
            for y in data[:,:d-1]:
                kmat[(ki, kj)] = math.exp((2 * np.dot(x, y.transpose()) - np.dot(x,x.transpose()) - np.dot(y,y.transpose())) / (2 * sigma ** 2))
                kj += 1
            ki += 1
        print 'sigma =\n %f' % sigma
        return kmat

#lh
def low(data, i, j, a_i, a_j, C):
    if data[(i, -1)] != data[(j, -1)]:
        low = max(0, a_j - a_i)
    else:
        low = max(0, a_i + a_j - C)
    return low

def high(data, i, j, a_i, a_j, C):
    if data[(i, -1)] != data[(j, -1)]:
        high = min(C, C - a_i + a_j)
    else:
        high = min(C, a_i + a_j)
    return high

#error1-error2
def error(kernel, data, i, j, alpha):
    error = 0
    ei = 0
    ej = 0
    ei = np.dot(alpha.transpose() * data[:, -1], kernel[:, i])
    ej = np.dot(alpha.transpose() * data[:, -1], kernel[:, j])
    error = ei - ej - data[(i, -1)] + data[(j, -1)]
    return error

#kij
def kij(kernel, i, j):
    kij = kernel[(i, i)] + kernel[(j, j)] - 2 * kernel[(i, j)]
    return kij

#norm
def norm(x, y):
    z = x - y
    zn = 0
    for zi in z:
        zn += zi*zi
    zns = math.sqrt(zn)
    return zns

#feature space of quadratic kernel
def feature(data, i):
    fs = []
    for i_f in range(d-1):
        for j_f in range(d-1):
            if i_f == j_f:
                fs.append(data[i, i_f] * data[i, j_f])
            elif i_f > j_f:
                fs.append(math.sqrt(2) * data[i, i_f] * data[i, j_f])
    fs = np.array(fs)
    return fs


def smo(kernel, data, C, eps, type):
    alpha = np.zeros((n,1))
    count_a = 0
    a_j = 0
    a_i = 0
    lst = []
    while True:
        alpha_p = copy.copy(alpha)
        for j in range(n):
            #print j
            if count_a != 0 and (alpha[j] - tol < 0 or alpha[j] + tol > C):
                continue
            lst_tmp = range(0, n)
            random.shuffle(lst_tmp)
            for i in lst_tmp:
                if i == j:
                    continue
                if count_a != 0 and (alpha[i] - tol < 0 or alpha[i] + tol > C):
                    continue
                if kij(kernel, i, j) == 0:
                    continue
                a_j = float(alpha[j])
                a_i = float(alpha[i])
                if low(data, i, j, a_i, a_j, C) == high(data, i, j, a_i, a_j, C):
                    continue
                alpha[j] = a_j + (data[(j, -1)] * error(kernel, data, i, j, alpha)) / kij(kernel, i, j)
                if alpha[j] < low(data, i, j, a_i, a_j, C):
                    alpha[j] = low(data, i, j, a_i, a_j, C)
                elif alpha[j] > high(data, i, j, a_i, a_j, C):
                    alpha[j] = high(data, i, j, a_i, a_j, C)
                alpha[i] = a_i + data[(i, -1)] * data[(j, -1)] * (a_j - alpha[j])
        count_a +=1
        
        if norm(alpha, alpha_p) <= eps:
            break

    #bias & Support Vector
    bias = 0
    count = 0
    print '\nSupport Vectors'
    for i_a in range(n):
        if alpha[i_a] - tol > 0 and alpha[i_a] + tol < C:
            count += 1
            print data[i_a,:d-1]
            bias += data[(i_a, -1)] - np.dot(alpha.transpose() * data[:, -1], kernel[:, i_a])
    bias = np.sum(bias) / count
    print '\nBias\n',bias

    #weight
    #linear
    if type == 'linear':
        w = np.zeros((1, d))
        w = np.dot(alpha.transpose() * data[:, -1], data)
        print '\nWeight\n', w[:,:d-1]

    #quadratic
    elif type == 'quadratic':
        w = np.zeros((1, (d - 1 + (d - 1) * (d - 2) / 2)))
        for i_w in range(n):
            if alpha[i_w] - tol > 0:
                w += alpha[i_w] * data[(i_w, -1)] * feature(data, i_w)
        print '\nWeight\n', w

    #sign
    y = np.zeros((n,1))
    for j_s in range(n):
        y[j_s] = np.sign(bias + np.dot(alpha.transpose() * data[:, -1], kernel[:, j_s]))
        y[j_s] = np.sign(y[j_s])

    classes = data[:,-1:]
    difference = np.zeros((n, 1))
    for i_d in range(n):
        if classes[i_d] == y[i_d]:
            difference[i_d] = 1
    acc = np.sum(difference) / n
    print '\nAccuracy\n', acc
    print '\nTime Cost\n', datetime.datetime.now() - datetime1

smo(kernel(str(sys.argv[4]), dataset, 0.1), dataset, float(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4]))
#smo(kernel('gaussian', dataset, 0.1), dataset, 20, 0.001, 'gaussian')
