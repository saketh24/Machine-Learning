import numpy as np


def question2(N):
    X1 = np.random.rand(N)
    X2 = np.random.rand(N)
    X = [max(a,b) for a,b in zip(X1, X2)]
    X_mean = np.mean(X)
    X1_mean = np.mean(X1)
    sum1 = 0
    sum2 = 0
    for a,b in zip(X1,X):
        sum1 += (b - X_mean) * (b - X_mean)
        sum2 += (a - X1_mean) * (b - X_mean)
    X_var = sum1/N
    X_X1_cov = sum2/N
    final = [X_mean, X_var, X_X1_cov]
    return final

E_list = []
V_list = []
C_list= []
iters = 30
N = 100
for i in range(iters):
    [E,V,C] = question2(N)
    E_list.append(E)
    V_list.append(V)
    C_list.append(C)
E_mean = np.mean(E_list)
V_mean = np.mean(V_list)
C_mean = np.mean(C_list)
sum1 = 0
sum2 = 0
sum3 = 0
for a,b,c in zip(E_list, V_list, C_list):
    sum1 += (a - E_mean) * (a-E_mean)
    sum2 += (b - V_mean) * (b-V_mean)
    sum3 += (c - C_mean) * (c-C_mean)
E_sd = (np.sqrt(sum1))/iters
V_sd = (np.sqrt(sum2))/iters
C_sd = (np.sqrt(sum3))/iters
print(E_mean)
print(V_mean)
print(C_mean)
print(E_sd)
print(V_sd)
print(C_sd)


