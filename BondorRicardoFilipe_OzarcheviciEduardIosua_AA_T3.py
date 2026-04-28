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

def householder(A, eps=1e-8):
    # Descompunere QR cu Householder
    n = A.shape[0]
    R = A.copy()
    Qt = np.eye(n)

    for r in range(n - 1):
        sigma = np.sum(R[r:, r]**2)
        if sigma <= eps: 
            continue

        k = np.sqrt(sigma)
        if R[r, r] > 0: 
            k = -k

        beta = sigma - k * R[r, r]
        
        u = np.zeros(n)
        u[r] = R[r, r] - k
        u[r+1:] = R[r+1:, r]

        # Transformare R
        for j in range(r + 1, n):
            gamma = np.dot(u[r:], R[r:, j]) / beta
            R[r:, j] -= gamma * u[r:]

        R[r, r] = k
        R[r+1:, r] = 0

        # Transformare Qt
        for j in range(n):
            gamma = np.dot(u[r:], Qt[r:, j]) / beta
            Qt[r:, j] -= gamma * u[r:]

    return Qt, R

def solve_triangular(R, b, eps=1e-8):
    # Substitutie inversa
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) <= eps:
            raise ValueError("Matrice singulara!")
        # Folosim np.dot pentru un calcul mai scurt si rapid
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
        
    return x

def main():
    # C6: Initializare random
    n = 5 
    eps = 1e-8
    np.random.seed(42)
    
    A = np.random.rand(n, n) * 10
    s = np.random.rand(n) * 10
    
    print(f"Dimensiune sistem: n = {n}\n")

    # C1: Calcul vector b
    b = A @ s

    # C2: Algoritmul Householder
    Qt, R = householder(A, eps)

    # C3: Rezolvare sistem liniar
    # Varianta NOASTRA (Householder)
    x_hh = solve_triangular(R, Qt @ b, eps) 
    
    # Varianta BIBLIOTECA (Descompunere QR)
    # Extragem Q_lib si R_lib folosind numpy.linalg.qr
    Q_lib, R_lib = np.linalg.qr(A)
    # Ax = b <=> Q_lib * R_lib * x = b <=> R_lib * x = Q_lib.T * b
    # Putem folosi functia noastra solve_triangular pentru a gasi solutia
    x_qr = solve_triangular(R_lib, Q_lib.T @ b, eps) 
    
    print(f"Norma ||x_QR - x_hh||: {np.linalg.norm(x_qr - x_hh):.5e}")

    # C4: Calcul erori
    norm_s = np.linalg.norm(s)
    
    err1 = np.linalg.norm(A @ x_hh - b)
    err2 = np.linalg.norm(A @ x_qr - b)
    err3 = np.linalg.norm(x_hh - s) / norm_s
    err4 = np.linalg.norm(x_qr - s) / norm_s

    print("\nErori (ar trebui sa fie < 10^-6):")
    print(f"||A*x_hh - b||_2 = {err1:.5e}")
    print(f"||A*x_QR - b||_2 = {err2:.5e}")
    print(f"eroare relativa x_hh = {err3:.5e}")
    print(f"eroare relativa x_QR = {err4:.5e}")

    # C5: Inversa matricei
    A_inv_hh = np.zeros((n, n))
    for j in range(n):
        # coloana j din inversa = solutia sistemului Rx = Qt * e_j
        A_inv_hh[:, j] = solve_triangular(R, Qt[:, j], eps)

    A_inv_lib = np.linalg.inv(A)
    
    inv_diff = np.linalg.norm(A_inv_hh - A_inv_lib)
    print(f"\nNorma diferentei inverselor: {inv_diff:.5e}")

if __name__ == "__main__":
    main()
