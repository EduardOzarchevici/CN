import math


# Helper function to read a list of floats from a text file
def read_vector(filename):
    try:
        with open(filename, 'r') as f:
            # Read all lines, split by whitespace, and convert to floats
            return [float(word) for line in f for word in line.split()]
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return []


def solve_sparse_system(prefix, epsilon=1e-8, max_iter=10000):
    # 1. Read the input files [cite: 25]
    d0 = read_vector(f"d0_{prefix}.txt")
    d1 = read_vector(f"d1_{prefix}.txt")
    d2 = read_vector(f"d2_{prefix}.txt")
    b = read_vector(f"b_{prefix}.txt")

    if not d0 or not d1 or not d2 or not b:
        return

    # 2. Calculate system dimensions and offsets [cite: 29, 30]
    n = len(d0)

    # The sizes of d1 and d2 are x+1 and y+1[cite: 22, 23].
    # Therefore: p = n - (x + 1) and q = n - (y + 1)
    p = n - len(d1)
    q = n - len(d2)

    print(f"--- System {prefix} ---")
    print(f"Dimension n: {n}, Offset p: {p}, Offset q: {q}")

    # 3. Verify that all elements on the main diagonal are non-zero [cite: 31, 53]
    for i in range(n):
        if abs(d0[i]) <= epsilon:
            print(f"Error: Zero found on main diagonal at index {i}. Cannot use Gauss-Seidel.")
            return

    # 4. Gauss-Seidel Algorithm Implementation [cite: 32, 93-107]
    x_curr = [0.0] * n  # Current guess, initialized to 0 [cite: 49, 94]
    x_prev = [0.0] * n  # Previous guess

    k = 0
    dx_norm = 0.0

    while k <= max_iter:
        # Copy current to previous for the new iteration
        x_prev = x_curr[:]
        dx_norm = 0.0

        for i in range(n):
            sum_ax = 0.0

            # --- THE MAGIC INDEXING --- [cite: 76, 77]
            # Lower diagonal p (uses newly calculated x_curr)
            if i >= p:
                sum_ax += d1[i - p] * x_curr[i - p]
            # Upper diagonal p (uses older x_prev)
            if i + p < n:
                sum_ax += d1[i] * x_prev[i + p]
            # Lower diagonal q (uses newly calculated x_curr)
            if i >= q:
                sum_ax += d2[i - q] * x_curr[i - q]
            # Upper diagonal q (uses older x_prev)
            if i + q < n:
                sum_ax += d2[i] * x_prev[i + q]

            # Apply the Gauss-Seidel formula [cite: 58-63]
            x_curr[i] = (b[i] - sum_ax) / d0[i]

            # Track the maximum difference (infinity norm) [cite: 89, 101]
            diff = abs(x_curr[i] - x_prev[i])
            if diff > dx_norm:
                dx_norm = diff

        k += 1

        # Check stopping criteria [cite: 103]
        if dx_norm < epsilon or dx_norm > 1e10:
            break

    # 5 & 6. Validation and Error Calculation [cite: 35-39]
    if dx_norm < epsilon:
        print(f"Solution approximated in {k} iterations.")

        y = [0.0] * n
        max_error = 0.0

        # Calculate y = Ax in a single pass [cite: 36, 37]
        for i in range(n):
            y[i] = d0[i] * x_curr[i]

            if i >= p:     y[i] += d1[i - p] * x_curr[i - p]
            if i + p < n:  y[i] += d1[i] * x_curr[i + p]
            if i >= q:     y[i] += d2[i - q] * x_curr[i - q]
            if i + q < n:  y[i] += d2[i] * x_curr[i + q]

            # Calculate infinity norm ||Ax_GS - b|| [cite: 38, 39]
            error = abs(y[i] - b[i])
            if error > max_error:
                max_error = error

        print(f"Infinity Norm ||Ax_GS - b||: {max_error:e}")

        # Print a sample to verify against PDF answers [cite: 148-160]
        print("Sample of solution x_GS (first 5 elements):")
        for i in range(min(n, 5)):
            print(f"x[{i}] = {x_curr[i]:.5f}")

    elif dx_norm > 1e10:
        print("Algorithm diverged! [cite: 106]")
    else:
        print("Hit maximum iterations without converging.")
    print("-" * 30)


# Run the solver for the first dataset
if __name__ == "__main__":
    solve_sparse_system("1")
    solve_sparse_system("2")
    solve_sparse_system("3")
    solve_sparse_system("4")
    solve_sparse_system("5")
