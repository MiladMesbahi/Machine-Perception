
import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinates of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinates of the April tag corners in (x,y,z) format
        K: 3x3 numpy array representing camera intrinsic matrix (default is identity)
    
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)
    """
    print("Initial Pixel Coordinates (Pc):\n", Pc)
    print("Initial World Coordinates (Pw):\n", Pw)
    print("Intrinsic Matrix (K):\n", K)

    f = K[0, 0]
    p1, p2, p3, p4 = 3, 2, 0, 1
    Pw_selected = np.array([Pw[p1], Pw[p2], Pw[p3]])

    Pc_h = np.concatenate([Pc, np.ones((4, 1))], axis=1)  # make homogeneous
    print(f"Homogeneous Pixel Coordinates:\n{Pc_h}")
    normalized_coords = (np.linalg.inv(K) @ Pc_h.T).T  # 3x3 matrix (normalize by K)
    normalized_coords /= normalized_coords[:, 2][:, np.newaxis]

    print("Normalized Pixel Coordinates:\n", normalized_coords)

    a = np.linalg.norm(Pw[p2] - Pw[p3])
    b = np.linalg.norm(Pw[p1] - Pw[p3])
    c = np.linalg.norm(Pw[p1] - Pw[p2])

    print(f"Computed Distances a: {a}, b: {b}, c: {c}")

    j1, j2, j3 = calculate_unit_vectors(normalized_coords,p1,p2,p3)

    cos_alpha, cos_beta, cos_gamma = calculate_cosines(j1, j2, j3)
    
    print("Unit Vectors:\nj1:", j1, "\nj2:", j2, "\nj3:", j3)
    print(f"Cos(alpha): {cos_alpha}, Cos(beta): {cos_beta}, Cos(gamma): {cos_gamma}")

    v_roots = compute_grunert_coefficients([a, b, c, cos_alpha, cos_beta, cos_gamma])

    print("Grunert Roots:\n", v_roots)
    best_solution = None
    min_error = float('inf')

    solutions = []
    for v in v_roots:
        u = (-1 + (a**2 - c**2) / b**2) * v**2 - 2 * (a**2 - c**2) / b**2 * cos_beta * v + 1 + (a**2 - c**2) / b**2
        u /= 2 * (cos_gamma - (v * cos_alpha))

        s1 = np.sqrt(c**2 / (1 + u**2 - 2 * u * cos_gamma))
        s2 = u * s1
        s3 = v * s1
        solutions.append([s1, s2, s3])
        print(f"Root: {v}\nSolution s1: {s1}, s2: {s2}, s3: {s3}")

    
    for s1, s2, s3 in solutions:
        Pc_camera = np.array([s1 * j1, s2 * j2, s3 * j3])
        R, t = Procrustes(Pc_camera, Pw_selected)
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (t):\n", t)
        print("Determinant of R:", np.linalg.det(R))

        projP = K @ (R.T @ (Pw[p4].T - t))
        projP /= projP[2]  # Normalize
        projP = np.array(projP.flatten())

        print(f"Projected Point (for Pw[{p4}]):\n", projP)
        print(f"Original Pixel Point (Pc[{p4}]):\n", Pc[p4])

        error = np.sum((projP[:2] - Pc[p4])**2)
        if error < min_error:
            min_error = error
            best_solution = (R, t)
    
    if best_solution is not None:
        R, t = best_solution
        print(f"Best solution found with SSE = {min_error}")
        return R, t
    
    print("No valid solution found.")
    return None  # Return None if no valid solution is found
def calculate_unit_vectors(imgC, p1,p2,p3):
# Unit vectors from the center of perspectivity to the observed points
    j1 = (1/np.linalg.norm(imgC[p1])) * imgC[p1]
    j2 = (1/np.linalg.norm(imgC[p2])) * imgC[p2]
    j3 = (1/np.linalg.norm(imgC[p3])) * imgC[p3]
    
    return j1, j2, j3
def calculate_cosines(j1, j2, j3):
    # Calculate cosines  of the angles between the unit vectors
    cos_alpha = np.dot(j2, j3)
    cos_beta = np.dot(j1, j3)
    cos_gamma = np.dot(j1, j2)
    
    return cos_alpha, cos_beta, cos_gamma

def compute_grunert_coefficients(constants):
    a = constants[0]
    b = constants[1]
    c = constants[2]
    cos_alpha = constants[3]
    cos_beta = constants[4]
    cos_gamma = constants[5]
    
    # coefficients for Grunert's method based on the distances and angles.
    A4 = (((a**2 - c**2) / b**2) - 1)**2 - (4 * c**2 / b**2) * cos_alpha**2
    
    A3 = 4 * (((a**2 - c**2) / b**2) * (1 - ((a**2 - c**2) / b**2)) * cos_beta
              - (1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma
              + 2 * (c**2 / b**2) * cos_alpha**2 * cos_beta)
    
    A2 = 2 * (((a**2 - c**2) / b**2)**2 - 1 + 2 * ((a**2 - c**2) / b**2)**2 * cos_beta**2
              + 2 * ((b**2 - c**2) / b**2) * cos_alpha**2
              - 4 * ((a**2 + c**2) / b**2) * cos_alpha * cos_beta * cos_gamma
              + 2 * ((b**2 - a**2) / b**2) * cos_gamma**2)
    
    A1 = 4 * (-((a**2 - c**2) / b**2) * (1 + ((a**2 - c**2) / b**2)) * cos_beta
              + ((2 * a**2)/ b**2) * cos_gamma**2 * cos_beta
              - (1 - ((a**2 + c**2) / b**2)) * cos_alpha * cos_gamma)
    
    A0 = (1 + ((a**2 - c**2) / b**2))**2 - (4 * a**2 / b**2) * cos_gamma**2
    
    roots = np.roots([A4, A3, A2, A1, A0])

    return roots.real[abs(roots.imag) < 1e-5]

def Procrustes(X, Y):
    X = np.array(X)
    Y = np.array(Y)

    X_centroid = np.mean(X, axis=0)
    Y_centroid = np.mean(Y, axis=0)

    # Center the points
    X_centered = X - X_centroid
    Y_centered = Y - Y_centroid

    # Compute covariance matrix
    H = X_centered.T @ Y_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correct for reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = Y_centroid - R @ X_centroid
    return R, t

# if __name__ == "__main__":
    pc = np.array([[304.28, 346.36], [449.04, 308.92], [363.24, 240.72], [232.29, 266.60]])
    pw = np.array([[-0.07, -0.07, 0], [0.07, -0.07, 0], [0.07, 0.07, 0], [-0.07, 0.07, 0]])
    K = np.array([[823.8, 0, 304.8], [0, 823.8, 236.3], [0, 0, 1]])

    # Call the P3P function and print the result
    print("Calling P3P function with test data...")
    R, t = P3P(pc, pw, K)

    # Print the returned Rotation and Translation if a valid solution is found
    if R is not None and t is not None:
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (t):\n", t)
    else:
        print("No valid solution found.")
