# nume: Bondor
# prenume: Ricardo Filipe
# nr matricol: 310910401ESL231009
# email: b.ryky.filipe@gmail.com
# nume discord: Bondor Ricardo-Filipe 3E1

# nume: Ozarchevici
# prenume: Eduard-Iosua
# email: iosuaozarchevici@gmail.com
# nr matricol: 310910402ESL231055
# nume discord: Ozarchevici Eduard-Iosua 3E1

# procent AI: 100%

import numpy as np
import os


# ==========================================
# FILE READING UTILITY
# ==========================================
def load_matrix_from_file(x):
    """
    Reads a matrix from a text file named 'hw5_{x}.txt'.
    Values in the text file should be separated by spaces.
    """
    filename = f"hw5_{x}.txt"

    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return None

    try:
        # numpy.loadtxt automatically parses space-separated rows into a 2D array
        A = np.loadtxt(filename)
        # Ensure it's explicitly a 2D array (in case of a 1D vector file)
        if len(A.shape) == 1:
            A = A.reshape(1, -1)
        return A
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


# ==========================================
# PART 1: SQUARE MATRICES (p = n)
# ==========================================
def jacobi_method(A, epsilon=1e-9, max_iter=1000):
    print("\n--- Running Jacobi Method ---")
    n = A.shape[0]
    A_k = A.copy()
    A_init = A.copy()
    U = np.eye(n)

    for k in range(max_iter):
        max_val = 0
        p, q = 0, 0
        for i in range(n):
            for j in range(i):
                if abs(A_k[i, j]) > max_val:
                    max_val = abs(A_k[i, j])
                    p = i
                    q = j

        if max_val < epsilon:
            print(f"Converged after {k} iterations.")
            break

        alpha = (A_k[p, p] - A_k[q, q]) / (2 * A_k[p, q]) 
        sign_alpha = 1 if alpha >= 0 else -1
        t = -alpha + sign_alpha * np.sqrt(alpha ** 2 + 1) #tangenta
        c = 1 / np.sqrt(1 + t ** 2) # cos
        s = t / np.sqrt(1 + t ** 2) # sin

        # $A_{nou} = R_{pq}(\theta) * A_{vechi} * R_{pq}^T(\theta)$

        for j in range(n):
            if j != p and j != q:
                a_pj = c * A_k[p, j] + s * A_k[q, j]
                a_qj = -s * A_k[p, j] + c * A_k[q, j]

                # pentru simetrie, actualizam ambele A[p, j] si A[j, p]

                A_k[p, j] = A_k[j, p] = a_pj
                A_k[q, j] = A_k[j, q] = a_qj

        # actualizam elementele diagonale

        a_pp = A_k[p, p] + t * A_k[p, q]
        a_qq = A_k[q, q] - t * A_k[p, q]

        A_k[p, p] = a_pp
        A_k[q, q] = a_qq

        # elementul pivot devine 0
        A_k[p, q] = A_k[q, p] = 0


        # actualizam matricea U 
        for i in range(n):
            u_ip = c * U[i, p] + s * U[i, q]
            u_iq = -s * U[i, p] + c * U[i, q]

            U[i, p] = u_ip
            U[i, q] = u_iq

    print(A_k)

    Lambda = np.diag(np.diag(A_k)) # valori proprii pe diagonala, restul 0

    print(Lambda)
    # $||A^{init}*U-U*\Lambda||$
    verification_norm = np.linalg.norm(A_init @ U - U @ A_k)
    print("Eigenvalues (Diagonal of Lambda):", np.diag(Lambda))
    print(f"Verification Norm || A_init * U - U * Lambda ||: {verification_norm}")

    return np.diag(A_k), U


def cholesky_sequence(A, epsilon=1e-9, max_iter=1000):
    print("\n--- Running Cholesky Sequence ---")
    A_k = A.copy()
    for k in range(max_iter):
        try:
            L = np.linalg.cholesky(A_k)
            A_next = L.T @ L

            if np.linalg.norm(A_next - A_k) < epsilon:
                print(f"Sequence converged after {k} iterations.")
                A_k = A_next
                break
            A_k = A_next
        except np.linalg.LinAlgError:
            print("Notice: Matrix became non-positive-definite; Cholesky failed at iteration", k)
            break

    print("Final Sequence Matrix (approximate diagonal):")
    print(A_k)
    return A_k


# ==========================================
# PART 2: RECTANGULAR MATRICES (p > n)
# ==========================================
def solve_svd_requirements(A):
    print("\n--- Running SVD Analysis ---")
    p, n = A.shape

    U, SingularValues, VT = np.linalg.svd(A, full_matrices=True)
    V = VT.T
    print(f"Singular Values: {SingularValues}")

    epsilon = 1e-9
    rank_A = np.sum(SingularValues > 0)
    print(f"Rank of A: {rank_A}")

    valid_sigmas = SingularValues[SingularValues > 0]
    if len(valid_sigmas) > 0:
        cond_A = valid_sigmas[0] / valid_sigmas[-1]
        print(f"Condition Number: {cond_A}")

    S_I = np.zeros((n, p))
    for i in range(int(rank_A)):
        S_I[i, i] = 1.0 / SingularValues[i]
    A_I = V @ S_I @ U.T
    print("\nMoore-Penrose Pseudoinverse (A^I):")
    print(A_I)

    try:
        A_J = np.linalg.inv(A.T @ A) @ A.T
        print("\nLeast Squares Pseudoinverse (A^J):")
        print(A_J)

        norm_diff = np.linalg.norm(A_I - A_J, ord=1)
        print(f"\nNorm ||A^I - A^J||_1: {norm_diff}")
    except np.linalg.LinAlgError:
        print("\nCould not compute Least Squares Pseudoinverse (A^T * A is singular).")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    file_index = "5"

    print(f"Loading matrix from hw5_{file_index}.txt...")
    A = load_matrix_from_file(file_index)

    if A is not None:
        print("Matrix loaded successfully:")
        print(A)

        p, n = A.shape
        print(f"\nMatrix dimensions: {p} rows (p), {n} columns (n).")

        if p == n:
            print("Matrix is square (p = n). Executing Part 1 requirements...")
            if np.allclose(A, A.T, atol=1e-8):
                jacobi_method(A)
            else:
                print("Warning: Matrix is not perfectly symmetric. Jacobi might not yield standard results.")
                jacobi_method(A)

            cholesky_sequence(A)

        elif p > n:
            print("Matrix is overdetermined (p > n). Executing Part 2 requirements...")
            solve_svd_requirements(A)

        else:
            print("Matrix has p < n. The homework requirements strictly specify logic for p = n or p > n.")