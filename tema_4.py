import math


def read_vector(filename):
    try:
        with open(filename, 'r') as f:
            return [float(word) for line in f for word in line.split()]
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return []


def solve_sparse_system(prefix, epsilon=1e-8, max_iter=10000):
    d0 = read_vector(f"d0_{prefix}.txt")
    d1 = read_vector(f"d1_{prefix}.txt")
    d2 = read_vector(f"d2_{prefix}.txt")
    b = read_vector(f"b_{prefix}.txt")

    if not d0 or not d1 or not d2 or not b:
        return

    n = len(d0)

    p = n - len(d1)
    q = n - len(d2)

    print(f"--- System {prefix} ---")
    print(f"Dimension n: {n}, Offset p: {p}, Offset q: {q}")

    for i in range(n):
        if abs(d0[i]) <= epsilon:
            print(f"Error: Zero found on main diagonal at index {i}. Cannot use Gauss-Seidel.")
            return

    x_curr = [0.0] * n
    x_prev = [0.0] * n

    k = 0
    dx_norm = 0.0

    while k <= max_iter:
        x_prev = x_curr[:]
        dx_norm = 0.0

        for i in range(n):
            sum_ax = 0.0

            if i >= p:
                sum_ax += d1[i - p] * x_curr[i - p]
            if i + p < n:
                sum_ax += d1[i] * x_prev[i + p]
            if i >= q:
                sum_ax += d2[i - q] * x_curr[i - q]
            if i + q < n:
                sum_ax += d2[i] * x_prev[i + q]

            x_curr[i] = (b[i] - sum_ax) / d0[i]

            diff = abs(x_curr[i] - x_prev[i])
            if diff > dx_norm:
                dx_norm = diff

        k += 1

        if dx_norm < epsilon or dx_norm > 1e10:
            break

    if dx_norm < epsilon:
        print(f"Solution approximated in {k} iterations.")

        y = [0.0] * n
        max_error = 0.0

        for i in range(n):
            y[i] = d0[i] * x_curr[i]

            if i >= p:     y[i] += d1[i - p] * x_curr[i - p]
            if i + p < n:  y[i] += d1[i] * x_curr[i + p]
            if i >= q:     y[i] += d2[i - q] * x_curr[i - q]
            if i + q < n:  y[i] += d2[i] * x_curr[i + q]

            error = abs(y[i] - b[i])
            if error > max_error:
                max_error = error

        print(f"Infinity Norm ||Ax_GS - b||: {max_error:e}")

        print("Sample of solution x_GS (first 5 elements):")
        for i in range(min(n, 5)):
            print(f"x[{i}] = {x_curr[i]:.5f}")

    elif dx_norm > 1e10:
        print("Algorithm diverged!")
    else:
        print("Hit maximum iterations without converging.")
    print("-" * 30)


if __name__ == "__main__":
    solve_sparse_system("1")
    solve_sparse_system("2")
    solve_sparse_system("3")
    solve_sparse_system("4")
    solve_sparse_system("5")
