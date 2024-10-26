from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####
    H = est_homography(Pw[:, :2], Pc)
    K_inv = np.linalg.inv(K)
    
    # Normalize H by K
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    r1 = np.dot(K_inv, h1)
    r2 = np.dot(K_inv, h2)
    t = np.dot(K_inv, h3)

    # Normalize r1 and r2 -> orthonormal basis
    norm_r1 = np.linalg.norm(r1)

    # scale
    r1 /= norm_r1
    r2 /= norm_r1
    t /= norm_r1  

    r3 = np.cross(r1, r2)
    R_wc = np.column_stack((r1, r2, r3))

    U, _, Vt = np.linalg.svd(R_wc) # orthogonality using SVD
    R_wc = np.dot(U, Vt)

    R_cw = R_wc.T  # inverse rotation
    t_cw = -np.dot(R_cw, t)

    return R_cw, t_cw

    ##### STUDENT CODE END #####