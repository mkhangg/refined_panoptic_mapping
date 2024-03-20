from numpy import *
from math import sqrt

def find_rotation_translation(A, B, weights):
    assert len(A) == len(B)

    N = A.shape[0]

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    H = transpose(AA) * BB
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    if linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = Vt.T * U.T
    t = -R*centroid_A.T + centroid_B.T

    return R, t
