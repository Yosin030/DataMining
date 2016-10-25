#python 2.7
#epsilon = 0.0000000001


#------------------------------------------------------------------
'''
    Mean vector
    
    Compute the mean vector miu for the 6-dimensional data matrix
'''

import numpy as np
file_d = open('airfoil_self_noise.dat')
for line in file_d:
    matrix = np.array(file_d.readline())
print matrix
#matrix = np.array(file_d)
#print matrix
#print dir(np)
import pandas as pd
#df = pd.read_csv('airfoil_self_noise.dat')
#print df

'''
list = []
for line in file_d:
    alst = line.split()
    tmp = []
    for num in alst:
        tmp.append(float(num))
    list.append(tmp)
mat_a = np.mat(list)
mat_at = mat_a.transpose()  #create a matrix

average = []
for lst in mat_at:
    #loop rows
    row = np.array(lst)
    sum = 0
    count = 0
    for x in row[0]:
        #loop for every x in one row
        sum += x
        count += 1
    avg = sum / count
    average.append(avg)
average_a = np.array(average)
avg_a = np.mat(average).transpose()

print 'the average vector is:\n', avg_a
'''

#------------------------------------------------------------------
'''
    total variance
    
    and then compute the total variance var(D)
    '''

sum_var = 0
count_var = 0
for lst in list:
    x_i = np.array(lst)
    z = x_i - average_a  #centalize: vecort x - vector mean
    det_z = np.dot(z.transpose(),z)
    sum_var += det_z
    count_var += 1
var = sum_var / count_var

print '\nthe variance is: ', var


#------------------------------------------------------------------
'''
    Covariance matrix (inner and outer product form)
    
    Compute the sample covariance matrix sigma as inner products between the attributes of the centered data matrix.
    Next compute the sample covariance matrix as sum of the outer products between the centered points
'''

i = -1  #use index to trace the mean
cent_z = []
for lst in mat_at:
    row = np.array(lst)
    i += 1  #index
    for x in row:
        x -= average_a[i]  #xij - mean_j
        cent_z.append(x)
cov_z_t = np.mat(cent_z)  #centered data matrix


#inner products
sigma_inner = np.dot(cov_z_t, cov_z_t.transpose()) / cov_z_t.shape[1]
print '\nsigma_inner is: \n', sigma_inner


#outer products
cov_z = cov_z_t.transpose()
sigma_outer = np.repeat(0.0, 36).reshape(6, 6)  #set 0 matrix
for z in cov_z:
    cov_tmp = np.dot(z.transpose(), z)  #dot product of each z_i
    sigma_outer += cov_tmp

sigma_outer = np.mat(sigma_outer / cov_z_t.shape[1])
print '\nsigma_outer is: \n', sigma_outer


#------------------------------------------------------------------
'''
    Correlation matrix as pair-wise cosines
    
    Compute the correlation matrix for this dataset using the formula for the cosine between centered attribute vectors.
    Which attributes are the most correlated, the most anti-correlated, and least correlated?
    Create the scatter plots for these interesting pairs using matplotlib and visually confirm the trends.
    '''

dict = {}  #for tracing the location of rho
lst_z_i = []
for i in range(6):
    z_i = cent_z[i] / np.dot(cent_z[i].transpose(), cent_z[i]) ** 0.5
    lst_z_i.append(z_i)

cor = {}
for p in range(6):
    for q in range(6):
        rho = np.dot(lst_z_i[p].transpose(), lst_z_i[q])
        dict[(p, q)] = rho
        cor[(p, q)] = rho
        if p == q:
            dict[(p, q)] -= 1
#save rho into a dictionary


cor_lst = []
for p in range(6):
    for q in range(6):
        cor_lst.append(cor[(p, q)])
cor_mat = np.mat(cor_lst).reshape(6,6)
print '\nThe correlation matrix is:\n', cor_mat


min_cor = min(dict, key = dict.get)
max_cor = max(dict, key = dict.get)

print '\nThe most correlated: Attribute %d and Attribute %d, the rho is %f' % (max_cor[0]+1, max_cor[1]+1, dict[max_cor[0], max_cor[1]])
print 'The most anti-correlated: Attribute %d and Attribute %d, the rho is %f' % (min_cor[0]+1, min_cor[1]+1, dict[min_cor[0], min_cor[1]])

for p in range(6):
    for q in range(6):
        if p == q:
            dict[(p, q)] += 1
        dict[(p, q)] = abs(dict[(p, q)])  #using absolute value to find the least correlated attributes

lea_cor = min(dict, key = dict.get)

print 'The least correlated: Attribute %d and Attribute %d, the rho is %f' % (lea_cor[0]+1, lea_cor[1]+1, cor[lea_cor[0], lea_cor[1]])

print '\nThe obeservations:', np.shape(mat_a)[0]

#scartter plot
import matplotlib.pyplot as plt
'''
#The most correlated
plt.plot(np.array(mat_at[1]), np.array(mat_at[4]), 'o')
plt.title('The most correlated')
plt.xlabel('Attribute 2')
plt.ylabel('Attribute 5')
plt.show()

#The most anti-correlated
plt.plot(np.array(mat_at[1]), np.array(mat_at[2]), 'o')
plt.title('The most anti-correlated')
plt.xlabel('Attribute 2')
plt.ylabel('Attribute 3')
plt.show()

#The least correlated
plt.plot(np.array(mat_at[0]), np.array(mat_at[2]), 'o')
plt.title('The least correlated')
plt.xlabel('Attribute 1')
plt.ylabel('Attribute 3')
plt.show()
'''


#------------------------------------------------------------------
'''
    First Two Eigenvectors and Eigenvalues
    
    Compute the first two eigenvectors of the covariance matrix sigma
    Once you have obtained the two eigenvectors: u1 and u2, project each of the original data points xi onto those two vectors,
    to obtain the new projected points in 2D. Plot these projected points in the two new dimensions.
    '''

def norm(x):
    #norm of vector x
    sum_norm = 0
    for col in x.transpose():
        for i in col.flat:
            sum_norm += i ** 2.0
        nrm = sum_norm ** 0.5
    return nrm

def dis(x, y):
    #distance of matrix x-y
    count_n2 = 0
    sum_n2 = 0
    di = 0
    z = x - y
    for col in z.transpose():
        for i in col.flat:
            count_n2 += 1
            sum_n2 += i ** 2.0
        nrm2 = sum_n2 ** 0.5
        di += nrm2
    return di

x_0_r = np.mat(np.random.randint(1, 10, size=(6, 2)))
x_0 = x_0_r / norm(x_0_r)  #create a random 6*2 matrix and make it into unit length

x_bfr = x_0
x_eig = x_0
count_x = 0
while True:
    #iterative
    #print 'before', x_bfr
    lamda1 = np.max(x_eig.transpose()[0]) / np.max(x_bfr.transpose()[0])
    lamda2 = np.max(x_eig.transpose()[1]) / np.max(x_bfr.transpose()[1])
    x_aft = np.dot(sigma_inner.transpose(),x_bfr)  #after multiplying
    
    x_aft_t = x_aft.transpose()
    a = x_aft_t[0].transpose()
    b = x_aft_t[1].transpose()
    x_aft_t[1] = (b - float(np.dot(b.transpose(), a) / np.dot(a.transpose(), a)) * a).transpose()
    x_aft_t[0] = x_aft_t[0] / norm(x_aft_t[0])
    x_aft_t[1] = x_aft_t[1] / norm(x_aft_t[1])
    x_eig = x_aft_t.transpose()
    #x_eig = norm1(x_aft_t.transpose())
    count_x += 1
    #change b, make it into unit length
    #it now has 2 eigen vectors a and b
    
    if dis(x_eig, x_bfr) <= 0.0000000001:  #epsilon: 0.0000000001
        #for ending the loop
        u1 = x_eig.transpose()[0]
        u2 = x_eig.transpose()[1]
        u1_m = np.mat(u1).transpose()
        u2_m = np.mat(u2).transpose()
        print '\nEigenvector u1:\n', u1_m, '\nEigenvector u2: \n', u2_m
        print '\nThe eigenvalue:'
        print 'lamda1:', lamda1, '\nlamda2:', lamda2,
        break
    
    x_bfr = x_eig  #new round of iterative

#create new axis
vec = []
for vector in mat_a:
    #the projection on u1
    vector1 = float(np.dot(u1, vector.transpose()) / np.dot(u1, u1.transpose())) * u1
    ##the projection on u2
    vector2 = float(np.dot(u2, vector.transpose()) / np.dot(u2, u2.transpose())) * u2
    vec.append((norm(vector1), norm(vector2)))  #change 6-d into 2-d
new_mat = np.mat(vec)


#scartter plot
x = np.array(new_mat.transpose()[0])
y = np.array(new_mat.transpose()[1])
'''
plt.plot(x, y, 'o')
plt.title('New projected points')
plt.xlabel('u1')
plt.ylabel('u2')
plt.show()
'''
