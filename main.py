import math
import random
import time
from typing import List


def get_machine_epsilon() -> float:
    m = 0
    # Căutăm m astfel încât 1.0 + 10^(-m) == 1.0
    while 1.0 + (10.0 ** -m) != 1.0:
        m += 1

    # Pasul anterior (m-1) este cel pentru care inegalitatea s-a menținut
    u = 10.0 ** -(m - 1)
    return u

def test_associativity(u: float) -> None:
    x = 1.0
    y = u / 10.0
    z = u / 10.0

    add1 = (x + y) + z
    add2 = x + (y + z)

    print(f"Asociativitate adunare: {'DA' if add1 == add2 else 'NU'}")


def find_non_associative_numbers() -> None:
    val = 1.0

    while val * val != 0.0:
        val = val / 10.0

    x = val
    y = val
    z = 1.0 / val

    inmultire1 = (x * y) * z
    inmultire2 = x * (y * z)

    print("Programul a gasit numerele:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")

    if inmultire1 != inmultire2:
        print("\nDemonstratie finalizata. Operatia NU este asociativa:")
        print(f"(x * y) * z = {inmultire1}")
        print(f"x * (y * z) = {inmultire2}")

# Functie auxiliara pentru reducerea unghiului in intervalul dorit
def reduce_angle(x: float) -> float:
    # Reducere la perioada [-pi/2, pi/2]
    x = math.fmod(x, math.pi)
    if x > math.pi / 2.0:
        x -= math.pi
    if x < -math.pi / 2.0:
        x += math.pi
    return x



# f0 = b0 ; mic = 10−12 ;
# if (f0 = 0) then f0 = mic ;
# C0 = f0 ;
# D0 = 0 ;
# j = 1 ;

# do
# Dj = bj + ajDj−1 ;
# if (Dj = 0) then Dj = mic ;
# Cj = bj + aj / Cj−1

# if (Cj = 0) then Cj = mic ;
# Dj = 1 / Dj
# ∆j = Cj Dj
# fj = ∆j fj−1 ;
# j = j + 1 ;
# while (|∆j − 1| ≥ ϵ) ;


# 1. Metoda Lentz
def my_tan_lentz(x: float, epsilon: float = 1e-10) -> float:
    x = reduce_angle(x)
    if abs(x) == math.pi / 2.0:
        return math.inf  # Tratare multipli de pi/2

    mic = 1e-12
    b0 = 0.0
    f0 = b0
    if f0 == 0.0:
        f0 = mic

    C = f0
    D = 0.0
    f = f0

    j = 1

    while True:
        a = x if j == 1 else -(x * x)
        b = 2 * j - 1

        D = b + a * D
        if D == 0.0:
            D = mic

        C = b + a / C
        if C == 0.0:
            C = mic

        D = 1.0 / D
        delta = C * D
        f = delta * f

        j += 1

        # Conditia de oprire a buclei do-while
        if abs(delta - 1.0) < epsilon:
            break

    return f


# 2. Metoda Polinomiala
def my_tan_poly(x: float) -> float:
    x = reduce_angle(x)
    if abs(x) == math.pi / 2.0:
        return math.inf

    invert = False
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x  # Antisimetrie

    # Reducere la [0, pi/4]
    if x > math.pi / 4.0:
        x = math.pi / 2.0 - x
        invert = True

    # Constante precalculate
    c1 = 0.33333333333333333
    c2 = 0.13333333333333333
    c3 = 0.053968253968254
    c4 = 0.0218694885361552

    x_2 = x * x
    x_3 = x_2 * x
    x_5 = x_3 * x_2
    x_7 = x_5 * x_2
    x_9 = x_7 * x_2

    rez = x + c1 * x_3 + c2 * x_5 + c3 * x_7 + c4 * x_9

    if invert:
        rez = 1.0 / rez

    return sign * rez


# 3. Testarea pe 10.000 de numere
def test_performance() -> None:
    lower_bound = -math.pi / 2.0 + 0.001
    upper_bound = math.pi / 2.0 - 0.001

    values: List[float] = [random.uniform(lower_bound, upper_bound) for _ in range(10000)]

    err_lentz = 0.0
    err_poly = 0.0

    # Testare Lentz
    start_lentz = time.perf_counter()
    for val in values:
        err_lentz += abs(math.tan(val) - my_tan_lentz(val))
    end_lentz = time.perf_counter()

    # Testare Poly
    start_poly = time.perf_counter()
    for val in values:
        err_poly += abs(math.tan(val) - my_tan_poly(val))
    end_poly = time.perf_counter()

    print(f"Eroare totala Lentz: {err_lentz}")
    print(f"Timp Lentz: {end_lentz - start_lentz:.6f} s")

    print(f"Eroare totala Poly: {err_poly}")
    print(f"Timp Poly: {end_poly - start_poly:.6f} s")


if __name__ == "__main__":
    print("Ex 1 -----------------------------------------------")
    epsilon = get_machine_epsilon()
    print(f"Decimal:    {epsilon:.30f}")

    print("Ex 2 a -----------------------------------------------")
    test_associativity(epsilon)

    print("Ex 2 b -----------------------------------------------")
    find_non_associative_numbers()

    print("Ex 3 -----------------------------------------------")
    test_performance()
