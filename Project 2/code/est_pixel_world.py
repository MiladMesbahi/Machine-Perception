import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world (z=0).
    
    Input:
        pixels: N x 2 coordinates of pixels (u, v) in the image
        R_wc: (3, 3) Rotation of the camera with respect to world coordinates
        t_wc: (3, ) translation of the camera in the world
        K: (3, 3) camera intrinsic matrix
        
    Returns:
        Pw: N x 3 world coordinates corresponding to the given pixel coordinates
    """
    pixels_h = np.concatenate([pixels, np.ones((pixels.shape[0], 1))], axis=1).T 
    
    normalized_coords = np.linalg.inv(K) @ pixels_h
    normalized_coords /= normalized_coords[2, :]  
    
    Pw = []
    
    for i in range(normalized_coords.shape[1]):
        norm_pixel = normalized_coords[:, i]
        direction = R_wc @ norm_pixel
        scale = -t_wc[2] / direction[2]

        X_w = t_wc[0] + scale * direction[0]
        Y_w = t_wc[1] + scale * direction[1]
        
        Pw.append([X_w, Y_w, 0]) 
    
    Pw = np.array(Pw)
    return Pw

