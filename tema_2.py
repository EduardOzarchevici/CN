import numpy as np
import scipy.linalg as la

# SETUP

n = 150 # System dimension (can be tested with sizes > 100)
eps = 1e-9 # Precision for calculations

B = np.random.randn(n, n)

A = np.dot(B, B.T) 

b = np.random.randn(n) 

# LIBRARY SOLUTION

# P, L_lib, U_lib = la.lu(A) # type: ignore
x_lib = np.linalg.solve(A, b)

print("Library solution x_lib computed successfully.\n")


# CHOLESKY LDL^T DECOMPOSITION

d = np.zeros(n)

for p in range(n):
    # d_p = a_pp - sum(d_k * l_pk^2) for k from 0 to p-1
    sum_d = 0.0
    for k in range(p):
        sum_d += d[k] * (A[p, k] ** 2)
    
    d[p] = A[p, p] - sum_d
    
    # Epsilon check to prevent division by zero in the next step
    if abs(d[p]) <= eps:
        print("Division by zero! The matrix might not be positive definite.")
        break
        
    # calculete lower triangular elements L[i, p] for i > p using:
    # l_ip = (a_ip - sum(d_k * l_ik * l_pk)) / d_p
    for i in range(p + 1, n):
        sum_l = 0.0
        for k in range(p):
            sum_l += d[k] * A[i, k] * A[p, k]
            
        # Store L elements directly into the lower part of A
        A[i, p] = (A[i, p] - sum_l) / d[p]

print("LDL^T Decomposition completed (Matrix A overwritten).\n")


# CALCULATING THE DETERMINANT

# det(A) = det(L) * det(D) * det(L^T). Since det(L) = 1, det(A) is just the product of d.
det_A = np.prod(d)
print(f"Determinant of A: {det_A}\n")

# SOLVING THE SYSTEM USING SUBSTITUTION

# L has 1s on the diagonal (implicitly).
# Calculate z from L (lower triangular)
z = np.zeros(n)
for i in range(n):
    sum_z = 0.0
    for j in range(i):
        sum_z += A[i, j] * z[j] # A[i, j] contains L elements here
    z[i] = b[i] - sum_z

# Solve -> D * y = z (Page 2)
# Compute y by dividing z by d, with an epsilon check to prevent division by zero
y = np.zeros(n)
for i in range(n):
    if abs(d[i]) > eps:
        y[i] = z[i] / d[i]

# Backward Substitution -> L^T * x = y 
# L^T has 1s on the diagonal. L^T_ij is L_ji, which is stored in A[j, i].
x_chol = np.zeros(n)
for i in range(n - 1, -1, -1):
    sum_x = 0.0
    for j in range(i + 1, n):
        sum_x += A[j, i] * x_chol[j] #A[j, i] instead of A[i, j]
    x_chol[i] = y[i] - sum_x

print("System solved. x_chol computed.\n")


# MATRIX-VECTOR MULTIPLICATION (A^init * x_chol)

# Compute A * x 
A_init_x = np.zeros(n)
for i in range(n):
    sum_ax = 0.0
    for j in range(n):
        if j >= i:
            original_a_ij = A[i, j]
        else:
            original_a_ij = A[j, i]
            
        sum_ax += original_a_ij * x_chol[j]
        
    A_init_x[i] = sum_ax


# VERIFICATION (NORMS) 

# Calculate Euclidean norms (L2 norms)
norm1 = np.linalg.norm(A_init_x - b, ord=2)
norm2 = np.linalg.norm(x_chol - x_lib, ord=2)

print("VERIFICATION")
print(f"||A^init * x_chol - b||_2 : {norm1}")
print(f"||x_chol - x_lib||_2      : {norm2}")

if norm1 < 1e-8 and norm2 < 1e-8:
    print("SUCCESS! The norms are within the acceptable precision limits.")
else:
    print("WARNING: The norms are larger than expected.")