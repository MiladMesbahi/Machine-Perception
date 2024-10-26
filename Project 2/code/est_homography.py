import numpy as np

def est_homography(X, Y):
    """ 
    Calculates the homography H of two planes such that Y ~ H*X
    If you want to use this function for hw5, you need to figure out 
    what X and Y should be. 
    Input:
        X: 4x2 matrix of (x,y) coordinates 
        Y: 4x2 matrix of (x,y) coordinates
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X
        
    """
    
    ##### STUDENT CODE START #####
    # Copy your HW1 code here
    #(8x9)
    A = []
    
    for i in range(4):
        x, y = X[i][0], X[i][1]
        x_prime, y_prime = Y[i][0], Y[i][1]
        
        # 2 sets 4 eqns
        A.append([-x, -y, -1, 0, 0, 0, x_prime * x, x_prime * y, x_prime])
        A.append([0, 0, 0, -x, -y, -1, y_prime * x, y_prime * y, y_prime])
    
    A = np.array(A)
    
    # SVD
    U, S, V = np.linalg.svd(A)
    
    H = V[-1].reshape((3, 3))
    
    # normalization
    H = H / H[2, 2]
    
    return H
    ##### STUDENT CODE END #####
 
