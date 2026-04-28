import random

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def horner(coeffs, x):
    """
    Evaluates the polynomial at a given point 'x' using Horner's scheme.
    """
    result = coeffs[0]
    for i in range(1, len(coeffs)):
        result = result * x + coeffs[i]
    return result

def get_derivative_coeffs(coeffs):
    """
    Calculates the coefficients of the derivative of a polynomial.
    """
    n = len(coeffs) - 1
    der_coeffs = []
    for i in range(n):
        # Multiply the coefficient by its power
        der_coeffs.append(coeffs[i] * (n - i))
    return der_coeffs

def calculate_R(coeffs):
    """
    Calculates the radius R. All real roots are guaranteed to be in [-R, R].
    """
    a0 = abs(coeffs[0])
    # Find the maximum absolute value among the rest of the coefficients (a1 to an)
    A = max([abs(c) for c in coeffs[1:]])
    return (a0 + A) / a0

def is_distinct(new_root, found_roots, epsilon):
    """
    Checks if a root is considered new/distinct based on the epsilon precision.
    """
    for r in found_roots:
        if abs(new_root - r) <= epsilon:
            return False # It's too close to an existing root
    return True

# ==========================================
# 2. NUMERICAL METHODS
# ==========================================

def newton_method(coeffs, der_coeffs, x0, epsilon, max_steps=1000):
    """
    Approximates a root using Newton's method.
    Returns: (root_value, number_of_steps) or (None, steps) if it fails.
    """
    x = x0
    for k in range(max_steps):
        p_val = horner(coeffs, x)
        p_der_val = horner(der_coeffs, x)
        
        # Prevent division by zero
        if abs(p_der_val) <= epsilon:
            return None, k
            
        delta_x = p_val / p_der_val
        x = x - delta_x
        
        # Stop condition: precision reached
        if abs(delta_x) < epsilon:
            return x, k + 1
            
        # Divergence check
        if abs(delta_x) >= 1e8:
            return None, k + 1
            
    return None, max_steps

def olver_method(coeffs, der_coeffs, der2_coeffs, x0, epsilon, max_steps=1000):
    """
    Approximates a root using Olver's method.
    Returns: (root_value, number_of_steps) or (None, steps) if it fails.
    """
    x = x0
    for k in range(max_steps):
        p_val = horner(coeffs, x)
        p_der_val = horner(der_coeffs, x)
        p_der2_val = horner(der2_coeffs, x)
        
        # Prevent division by zero
        if abs(p_der_val) <= epsilon:
            return None, k
            
        # Olver's specific correction factor (c_k)
        c_k = (p_val**2 * p_der2_val) / (p_der_val**3)
        delta_x = (p_val / p_der_val) + 0.5 * c_k
        
        x = x - delta_x
        
        # Stop condition: precision reached
        if abs(delta_x) < epsilon:
            return x, k + 1
            
        # Divergence check
        if abs(delta_x) >= 1e8:
            return None, k + 1
            
    return None, max_steps

# ==========================================
# 3. MAIN PROGRAM
# ==========================================

def main():
    # Example Polynomial: P(x) = x^3 - 6x^2 + 11x - 6 (Roots are 1, 2, and 3)
    coeffs = [1.0, -6.0, 11.0, -6.0]
    epsilon = 1e-6
    
    # Get derivatives
    der1_coeffs = get_derivative_coeffs(coeffs)
    der2_coeffs = get_derivative_coeffs(der1_coeffs)
    
    # Calculate search interval
    R = calculate_R(coeffs)
    print(f"Searching for roots in the interval: [{-R}, {R}]\n")
    
    # Generate random starting points within [-R, R]
    num_starting_points = 50
    start_points = [random.uniform(-R, R) for _ in range(num_starting_points)]
    
    # Lists to keep track of distinct roots
    distinct_roots = []
    
    print(f"{'Start Point (x0)':>16} | {'Newton (Root / Steps)':>25} | {'Olver (Root / Steps)':>25}")
    print("-" * 75)
    
    for x0 in start_points:
        # Run both methods from the same starting point
        root_n, steps_n = newton_method(coeffs, der1_coeffs, x0, epsilon)
        root_o, steps_o = olver_method(coeffs, der1_coeffs, der2_coeffs, x0, epsilon)
        
        # We process the results. We will use Newton's roots for saving, 
        # but we display both to compare the number of steps.
        if root_n is not None:
            # Check if it's a completely new root
            if is_distinct(root_n, distinct_roots, epsilon):
                distinct_roots.append(root_n)
                
                # Format the output nicely for the console
                str_n = f"{root_n:10.6f} (in {steps_n:2} steps)"
                str_o = f"{root_o:10.6f} (in {steps_o:2} steps)" if root_o is not None else "Failed"
                
                print(f"{x0:16.4f} | {str_n:>25} | {str_o:>25}")

    # Sort the roots in ascending order for a cleaner output file
    distinct_roots.sort()
    
    # ==========================================
    # 4. SAVE TO FILE
    # ==========================================
    filename = "distinct_roots.txt"
    with open(filename, "w") as file:
        file.write("Distinct real roots found:\n")
        file.write("-" * 30 + "\n")
        for root in distinct_roots:
            file.write(f"{root:.6f}\n")
            
    print(f"\nFinished! Found {len(distinct_roots)} distinct roots.")
    print(f"The roots have been saved to the file: '{filename}'")

if __name__ == "__main__":
    main()