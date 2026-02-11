import numpy as np

np.set_printoptions(precision=8, suppress=True)

# Matrices and Vectors

A = np.array([
    [-2.74125009,  2.24215689, -0.60553211, -0.16755625],
    [-0.34868395,  0.29538923, -0.45259498,  0.50015934],
    [ 2.49664208,  0.27798324,  2.00739274,  0.21978030]
], dtype=float)

yA = np.array([0.61339829, 0.11012282, -0.06426754], dtype=float)

B = np.array([
    [-2.74125009, -0.34868395,  2.49664208],
    [ 2.24215689,  0.29538923,  0.27798324],
    [-0.60553211, -0.45259498,  2.00739274],
    [-0.16755625,  0.50015934,  0.21978030]
], dtype=float)

yB  = np.array([0.66761214, 0.35931116, 0.74289966, 0.02979187], dtype=float)
yB2 = np.array([0.24982762, -0.45768269, 0.22778277, 0.63413920], dtype=float)

C = np.array([
    [0.31997336, 0.43316234, -0.33457014, -0.34017903],
    [1.12969075, 1.52931319, -1.18122581, -1.20102843],
    [0.20087760, 0.27193705, -0.21004138, -0.21356262]
], dtype=float)

yC  = np.array([0.14216640, 0.50192948, 0.08925132], dtype=float)
yC2 = np.array([-1.01480112, 0.41152110, -0.45229071], dtype=float)

D = np.array([
    [ 0.07999334,  0.28242269,  0.05021940],
    [ 0.10829058,  0.38232830,  0.06708426],
    [-0.08364254, -0.29530645, -0.05251035],
    [-0.08504476, -0.30025711, -0.05339065]
], dtype=float)

yD  = np.array([0.41615372, 0.56336601, -0.43513813, -0.44243299], dtype=float)
yD2 = np.array([0.47277025, -0.64357627, 1.30059591, 1.42694800], dtype=float)

# Cases

cases = [
    ("Case 1", "A", A, yA, "yA"),
    ("Case 2", "B", B, yB, "yB"),
    ("Case 3", "B", B, yB2, "yB2"),
    ("Case 4", "C", C, yC, "yC"),
    ("Case 5", "C", C, yC2, "yC2"),
    ("Case 6", "D", D, yD, "yD"),
    ("Case 7", "D", D, yD2, "yD2"),
]

# Orthogonal Projection Matrix onto Range(G)
def projection_range(G, tol=1e-12):
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    r = np.sum(s > tol)
    U_r = U[:, :r]  # m x r
    P = U_r @ U_r.T # m x m
    return P, r, s

def analyze_case(case_num, G_name, G, f, f_name, tolerance=1e-8):
    m, n = G.shape

    # Projection onto Range(G)
    projection_matrix, rankG, svals = projection_range(G)

    # Test f in Range(G) -- Pf = f
    projected_f = projection_matrix @ f
    projection_error = f - projected_f
    distance = np.linalg.norm(projection_error, ord=2)
    in_range = distance < tolerance

    # Over/Underdetermined  
    if m > n:
        determined_type = "Overdetermined (m > n)"
    elif m < n:
        determined_type = "Underdetermined (m < n)"
    else:
        determined_type = "Square (m = n)"

    # Solution
    if not in_range:
        solution_type = "No Solution"
        w = None
    else:
        if rankG == n:
            solution_type = "Unique Solution"
        else:
            solution_type = "Infinitely Many Solutions"

        w, *_ = np.linalg.lstsq(G, f, rcond=None)

    # Print
    print("\n" + "-"*50)
    print(f"\n{case_num}: G = {G_name}, f = {f_name}")
    print(f"G shape: {m} x {n}")
    print(f"Rank of G: {rankG}")
    print("Singular values:", svals)

    print("\nProjection matrix onto Range(G):")
    print(projection_matrix)

    print(f"\n||f - Pf||_2 = {distance:.6e}")
    print(f"f in Range(G)? {'YES' if in_range else 'NO'}")

    print(f"\nSystem type: {determined_type}")
    print(f"Solution status: {solution_type}")

    if w is not None:
        print("\nOne solution w (using first square):")
        print(w)

# Run Cases

if __name__ == "__main__":
    for (case_name, G_name, G, f, f_name) in cases:
        analyze_case(case_name, G_name, G, f, f_name, tolerance=1e-8)
