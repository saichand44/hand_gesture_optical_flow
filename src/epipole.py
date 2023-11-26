import numpy as np

def epipole(flow_x, flow_y, smin, threshmin=10):
    '''
    Compute the epipole from the optical flow.

    Here only translation is considered and using the equation, p = (1/Z) * P --> [1]
    where p - pixel position, P - world point, Z - coordinate to convert (by normalization) P to p

    Differentiating [1], we get p_dot (in pixels) = (1/Z) * (xVz-Vx, yVz-Vy).T
                                x, y in pixels; Vx, Vy, Vz --> world velocity

                                p_dot.T (p x V) = 0 => V.T (p x p_dot) = 0 --> [2]

    Parameters
    ----------
    flow_x : numpy array
        optical flow on the x-direction; shape: (H, W)
    flow_y : numpy array
        optical flow on the y-direction; shape: (H, W)
    smin : numpy array
        confidence of the flow estimates; shape: (H, W)
    threshmin : int
        threshold for condidence (optional)

    Returns
    -------
    ep : numpy array
        epipole; shape: (3, )
    '''

    # filter the values
    good_idx = np.flatnonzero(smin>threshmin)
    permuted_indices = np.random.RandomState(seed=10).permutation(good_idx)
    valid_idx = permuted_indices[:3000]

    # get the x and y dimensions
    y_dim, x_dim = flow_x.shape[0], flow_x.shape[1]

    # generate the meshgrid and considering the origin at the center of the image
    xp = np.linspace(-x_dim//2, x_dim//2, num=x_dim, endpoint=False, dtype=int)
    yp = np.linspace(-y_dim//2, y_dim//2, num=y_dim, endpoint=False, dtype=int)

    Xp, Yp = np.meshgrid(xp, yp)

    # get only permissible pixel positions according to the thresholding
    Xp = np.ravel(Xp)[valid_idx]
    Yp = np.ravel(Yp)[valid_idx]

    # get only permissible flow values according to the thresholding
    U = np.ravel(flow_x)[valid_idx]
    V = np.ravel(flow_y)[valid_idx]

    # initialize the matrices for homogeneous pixel coodinates, flow velocities
    pixel_positions = np.zeros((len(valid_idx), 3))
    flow_vectors = np.zeros((len(valid_idx), 3))

    # create homogeneous coordinates of the pixel position, flow vectors
    for i, (row, col) in enumerate(zip(range(len(valid_idx)), range(len(valid_idx)))):
        pixel_positions[i, :] = Xp[row], Yp[col], 1
        flow_vectors[i, :] = U[row], V[col], 0

    # Solving for V.T (p x p_dot) = 0 (V in world coordinates)
    epipolar_lines = np.cross(pixel_positions, flow_vectors)
    U, S, Vt = np.linalg.svd(epipolar_lines)
    ep = Vt[-1]

    return ep