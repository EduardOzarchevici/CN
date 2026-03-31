import math

def citeste_vector(nume_fisier):
    """Citește un vector dintr-un fișier text și îl returnează ca listă de float-uri."""
    vector = []
    try:
        with open(nume_fisier, 'r') as f:
            for linie in f:
                elemente = linie.split()
                for el in elemente:
                    vector.append(float(el))
    except FileNotFoundError:
        return None
    return vector

def rezolva_sistem(index_sistem, epsilon=1e-6, k_max=10000):
    print(f"\n{'='*40}")
    print(f"Sistemul {index_sistem}")
    print(f"{'='*40}")
    
    # Numele normale, fix cum apar in folderul tau
    nume_d0 = f"d0_{index_sistem}.txt"
    nume_d1 = f"d1_{index_sistem}.txt"
    nume_d2 = f"d2_{index_sistem}.txt"
    nume_b  = f"b_{index_sistem}.txt"
    
    # 1. Citirea datelor
    d0 = citeste_vector(nume_d0)
    d1 = citeste_vector(nume_d1)
    d2 = citeste_vector(nume_d2)
    b  = citeste_vector(nume_b)
    
    if not (d0 and d1 and d2 and b):
        print(f"Eroare: Nu am putut găsi toate fișierele pentru sistemul {index_sistem}.")
        print(f"Am cautat: {nume_d0}, {nume_d1}, {nume_d2}, {nume_b}")
        return

    # 2. Determinarea dimensiunilor și a lui p, q
    n = len(d0)
    p = n - len(d1)
    q = n - len(d2)
    
    print(f"Dimensiunea sistemului (n): {n}")
    print(f"Diagonala secundară 1 (p): {p}")
    print(f"Diagonala secundară 2 (q): {q}")
    
    # 3. Validarea diagonalei principale
    for val in d0:
        if abs(val) <= epsilon:
            print("Eroare: Matricea are elemente nule pe diagonala principală. Oprim.")
            return

    # 4. Metoda iterativă Gauss-Seidel
    xp = [0.0] * n  # x precedent
    xc = [0.0] * n  # x curent
    
    k = 0
    solutie_gasita = False
    
    while k <= k_max:
        for i in range(n):
            suma = 0.0
            
            # Adunăm elementele folosind vectorii comprimați
            if i - q >= 0:
                suma += d2[i - q] * xc[i - q]
            if i - p >= 0:
                suma += d1[i - p] * xc[i - p]
            if i + p < n:
                suma += d1[i] * xp[i + p]
            if i + q < n:
                suma += d2[i] * xp[i + q]
                
            xc[i] = (b[i] - suma) / d0[i]
            
        # Varianta sigură pentru calculul diferenței maxime (fără list comprehension inline)
        diferente = []
        for i in range(n):
            diferente.append(abs(xc[i] - xp[i]))
        delta_x = max(diferente)
        
        if delta_x < epsilon:
            solutie_gasita = True
            break
        elif delta_x > 1e10:
            print(f"Divergență la iterația {k}!")
            break
            
        # Actualizăm vectorul precedent
        xp = xc.copy()
        k += 1

    if solutie_gasita:
        print(f"Soluție aproximată cu succes în {k} iterații.")
        
        # 5. Calculul vectorului y = A * x_gs
        y = [0.0] * n
        for i in range(n):
            y[i] = d0[i] * xc[i]
            if i - q >= 0:
                y[i] += d2[i - q] * xc[i - q]
            if i - p >= 0:
                y[i] += d1[i - p] * xc[i - p]
            if i + p < n:
                y[i] += d1[i] * xc[i + p]
            if i + q < n:
                y[i] += d2[i] * xc[i + q]
                
        # 6. Calculul normei
        erori = []
        for i in range(n):
            erori.append(abs(y[i] - b[i]))
        norma = max(erori)
        print(f"Norma erorii ||Ax_GS - b||_inf: {norma}")
        
        # Varianta sigură pentru afișare
        afisare_limita = min(5, n)
        valori_afisare = []
        for val in xc[:afisare_limita]:
            valori_afisare.append(round(val, 4))
            
        print(f"Primele {afisare_limita} elemente ale soluției x_GS: {valori_afisare}")
        
    else:
        print("Nu s-a putut aproxima soluția (număr maxim de iterații depășit).")

if __name__ == "__main__":
    precizie = 10**(-6) 
    
    # Rulam pentru toate cele 5 sisteme
    for i in range(1, 6):
        rezolva_sistem(i, epsilon=precizie)