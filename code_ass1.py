#Poisson Equation Part
#Poisson Equation with numerical method
#Hyperparameters
from math import sin, pi
import numpy as np

def f(x,y,h):
    return sin(pi*h*x)*sin(pi*h*y)

def Poisson(M):
    #initialize
    h = 1 / M
    F = np.zeros(shape=((M-1),(M-1)),dtype='float64')
    for i in range(M-1):
        for j in range(M-1):
            F[i][j] = f(i+1,j+1,h)
    #Coefficient matrix, use internal dot matrix only
    A = np.zeros(shape=((M-1)*(M-1),(M-1)*(M-1)),dtype='float64')
    #B vector
    B = np.zeros((M-1)*(M-1),dtype='float64')
    #iterate all the internal dot
    eqa_index = 0
    for i in range(M-1):
        for j in range(M-1):
            tmp_A = np.zeros(shape=((M-1),(M-1)),dtype='float64')
            B[eqa_index] = F[i][j]
            tmp_A[i,j] = 4
            if i >= 1:
                tmp_A[i-1,j] = -1
            if i + 1 <= M - 2:
                tmp_A[i+1,j] = -1
            if j >= 1:
                tmp_A[i,j-1] = -1
            if j + 1 <= M - 2:
                tmp_A[i,j+1] = -1
            A[eqa_index] = tmp_A.reshape(-1)
            eqa_index+=1
    return [A,B]

#Methods





#LU
from numpy import dot
def LUdecomp(a):
    n = len(a)
    for k in range(n-1):
        for i in range(k+1,n):
            if a[i,k] != 0:
                lam = a[i,k] / a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam * a[k,k+1:n]
                a[i,k] = lam
    return a

def LUsolve(a,b):
    #forward substitution
    n = len(b)
    for k in range(1,n):
        b[k] = b[k] - dot(a[k,0:k], b[:k])
    #backward substitution
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - dot(a[k,k+1:n],b[k+1:n])) / a[k,k]
    return b






#Choleski

from numpy import dot
from math import sqrt

def choleski(a):
    n = len(a)
    for k in range(n):
#         print(k)
#         print(a[k,k] - dot(a[k,0:k],a[k,0:k]))
        a[k,k] = sqrt(a[k,k] - dot(a[k,0:k],a[k,0:k]))
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n):
        a[0:k,k] = 0.0
    return a

def choleskiSol(L,b):
    n = len(b)
    for k in range(n):
        b[k] = (b[k] - dot(L[k,0:k], b[:k])) / L[k,k]
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    return b






#Banded LU
def BLUDecompose(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def BLUSolve(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1] 
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b
#Banded Solution
length = len(A)
ones = np.ones(length-1)
c = ones * -1
e = ones * -1
d = np.ones(length)
d = d * 4
c,d,e = BLUDecompose(c,d,e)
BLUSolve(c,d,e,B)




#Banded Choleski
def LDLT(A):
    n = len(A)
    L = np.zeros(shape=(n,n),dtype='float64')
    D = np.zeros(shape=(n,),dtype='float64')
    for i in range(n):
        L[i,i] = 1
    for i in range(n):
        D[i] = A[i,i] - dot(L[i,0:i]**2, D[:i])
        for j in range(i+1,n):
            L[j,i] = A[j,i] - dot(L[j,0:i]*L[i,0:i], D[:i]) / D[i]
    return L,D


n = len(A)
l,d = LDLT(A)
#A = LDL^t
#Ax = y
z = np.zeros(shape=(n,1))
y = np.zeros(shape=(n,1))
x = np.zeros(shape=(n,1))
#Ly = b
y[0] = B[0]
for i in range(1,n):
    y[i] = B[i] - np.dot(l[i,0:i], y[:i])
#Dz = y
for i in range(n):
    z[i] = y[i] / d[i]
#L^t x = z
x[n-1] = z[n-1]
for i in range(n-1,-1,-1):
    x[i] = z[i] - np.dot(l[i+1:n-1,i].T,x[i+1:n-1]) 





#PART II

def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i], v[j] = v[j],v[i]
    else:
        temp = v[i].copy()
        v[i] = v[j]
        v[j] = temp

# with col pivotting
import numpy as np

def LUdecomp_withPivot(a):
    n = len(a)
    seq = np.array(range(n))

    for k in range(n-1):
        #row change
        p = int(np.argmax(a[k:n,k])) + k
        if p != k:
            swapRows(a,k,p)
            swapRows(seq,k,p)

        for i in range(k+1,n):
            if a[i,k] != 0:
                lam = a[i,k] / a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam * a[k,k+1:n]
                a[i,k] = lam
    return a, seq

def LUsolve_withPivot(a,b,seq):
    n = len(a)
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

    for k in range(1,n):
        x[k] = x[k] - dot(a[k,0:k], x[:k])
    for k in range(n-1,-1,-1):
        x[k] = (x[k] - dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x


#Solving the equation
m = np.zeros(shape=(84,84))
m[0][0] = 6
m[0][1] = 1
m[83][82] = 8
m[83][83] = 6

for i in range(1,83):
    m[i][i] = 6
    m[i][i-1] = 8
    m[i][i+1] = 1

b = np.zeros(84)
b[0] = 7
b[83] = 14
for i in range(1,83):
    b[i] = 15

a = LUdecomp(m)
ans = LUsolve(a,b)

lu,seq = LUdecomp_withPivot(m)
sol = LUsolve_withPivot(lu,b,seq)

#Hilbert Matrix
H = np.zeros(shape=(40,40),dtype='float64')
for i in range(40):
    for j in range(40):
        H[i][j] = 1 / (i+j+1)
H_b = np.zeros(40,dtype='float64')
for i in range(40):
    temp_sum = sum(1 / (i+j+1) for j in range(40))
    H_b[i] = temp_sum

choleski_H = choleski(H)
sol_Cho = choleskiSol(choleski_H,H_b)


L,D = LDLT(H)
#A = LDL^t
#Ax = y
n = len(D)
z = np.zeros(shape=(n,1))
y = np.zeros(shape=(n,1))
x = np.zeros(shape=(n,1))
#Ly = b
y[0] = b[0]
for i in range(1,n):
    y[i] = b[i] - dot(L[i,0:i], y[:i])
#Dz = y
for i in range(n):
    z[i] = y[i] / D[i]
#L^t x = z
x[n-1] = z[n-1]
for i in range(n-1,-1,-1):
    x[i] = z[i] - dot(L[i+1:n-1,i].T,x[i+1:n-1])

#ass2

#Cal the inf Norm of Matrix
def Norm(x):
    return max(sum(x))

for i in range(5,21):
    H = Hilbert(i)
    print(Norm(H))



def Mat(N):
    M = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            if(i == j):
                M[i][j] = 1
            if(i > j):
                M[i][j] = -1
    M[:,N-1] = 1
    return M

for i in range(5,31):
    b = np.random.rand(i)
    M = Mat(i)
    M = LUdecomp(M)
    b = LUsolve(M,b)
    print(f"Rank {i} matrix's solution is :\n")
    print(b)
    print('\n')
