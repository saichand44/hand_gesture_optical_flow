import numpy as np
import matplotlib.pyplot as plt

def plot_flow(image, image_flow, confidence, threshmin=10):
    '''
    Plot the optical flow field for one frame of the data.

    Parameters
    ----------
    image: numpy array
        grayscale image; shape: (H, W)
    image_flow : numpy array
        flow velocities of the pixels in the image in x, y directions; shape:(H, W, 2)
    confidence : 
        confidence of the flow estimates; shape: (H, W)
    threshmin : int
        threshold for condidence (optional)
    '''

    # get the x and y dimensions
    y_dim, x_dim = image.shape[0], image.shape[1]

    # generate the meshgrid
    x = np.linspace(0, x_dim, num=x_dim, endpoint=False)
    y = np.linspace(0, y_dim, num=y_dim, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # get the gradients for the arrow direction
    U = image_flow[:, :, 0]
    V = image_flow[:, :, 1]

    # filter the values
    good_idx = np.flatnonzero(confidence>threshmin)
    permuted_indices = np.random.RandomState(seed=10).permutation(good_idx)
    valid_idx=permuted_indices[:3000]
    X = np.ravel(X)[valid_idx]
    Y = np.ravel(Y)[valid_idx]
    U = np.ravel(U)[valid_idx]
    V = np.ravel(V)[valid_idx]

    # plot the original image
    plt.imshow(image, cmap='gray', alpha=0.5)

    # create an optical flow plot
    plt.quiver(X, Y, U, V, scale=15, width=0.003, color='red', alpha=0.5, angles='xy', headwidth=10)