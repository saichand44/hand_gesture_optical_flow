import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from src.compute_grad import compute_Ix, compute_Iy, compute_It
from src.compute_flow import flow_lk
from src.vis_flow import plot_flow
from src.epipole import epipole

# define the intrinsic (K) matrix
K = np.array([[1118,    0, 357],
              [   0, 1121, 268],
              [   0,    0,   1]])

if __name__ == "__main__":
    pass