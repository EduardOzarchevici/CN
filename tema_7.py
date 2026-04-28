import random

# Helper Functions

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
        der_coeffs.append(coeffs[i] * (n - i))
    return der_coeffs

def calculate_R(coeffs):
    """
    Calculates the radius R. All real roots are guaranteed to be in [-R, R].
    """
    a0 = abs(coeffs[0])
    A = max([abs(c) for c in coeffs[1:]])
    return (a0 + A) / a0

def is_distinct(new_root, found_roots, epsilon):
    """
    Checks if a root is considered new/distinct based on the epsilon precision.
    """
    for r in found_roots:
        if abs(new_root - r) <= epsilon:
            return False
    return True

# Numerical Methods

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

# Main Program

def main():
    coeffs = [1.0, -6.0, 11.0, -6.0]
    # coeffs = [1.0, -6.0, 13.0, -12.0, 4.0]
    
    epsilon = 1e-6
    
    der1_coeffs = get_derivative_coeffs(coeffs)
    der2_coeffs = get_derivative_coeffs(der1_coeffs) 
    
    R = calculate_R(coeffs)
    print(f"Searching for roots in the interval: [{-R}, {R}]\n")
    
    num_starting_points = 50
    start_points = [random.uniform(-R, R) for _ in range(num_starting_points)]
    
    distinct_roots = []
    
    print(f"{'Start Point (x0)':>16} | {'Newton (Root / Steps)':>25} | {'Olver (Root / Steps)':>25}")
    print("-" * 75)
    
    for x0 in start_points:
        root_n, steps_n = newton_method(coeffs, der1_coeffs, x0, epsilon)
        root_o, steps_o = olver_method(coeffs, der1_coeffs, der2_coeffs, x0, epsilon)
        
        if root_n is not None:
            str_n = f"{root_n:10.6f} (in {steps_n:2} steps)"
            str_o = f"{root_o:10.6f} (in {steps_o:2} steps)" if root_o is not None else "Failed"
            print(f"{x0:16.4f} | {str_n:>25} | {str_o:>25}")


            if is_distinct(root_n, distinct_roots, epsilon):
                distinct_roots.append(root_n)
                
                
                

    distinct_roots.sort()
    
    # Save results
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