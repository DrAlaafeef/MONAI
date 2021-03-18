from __future__ import print_function


import site
print(site.getsitepackages())
site.addsitedir('./')  # Replace with your MONAI directory
import torch
import monai
import numpy as np

import time
from PIL import Image
import matplotlib.pyplot as plt
import itertools as it
from scipy import ndimage
import os.path as osp

try:
    from numba import jit
    print("with numba")
except ImportError:
    print("no numba")
    def jit(ob):
        return ob

@jit
def sweep(A, Cost):
    max_diff = 0.0
    for i in range(1, A.shape[0]):
        for j in range(1, A.shape[1]):
            t1, t2 = A[i, j-1], A[i-1, j]
            C = Cost[i, j]
            if abs(t1-t2) > C:
                t0 = min(t1, t2) + C  # handle degenerate case
            else:
                t0 = 0.5*(t1 + t2 + np.sqrt(2*C*C - (t1-t2)**2))
            max_diff = max(max_diff, A[i, j] - t0)
            A[i, j] = min(A[i, j], t0)
    return max_diff

def geodesic2d_fsm(I, S, max_iter_n=80, max_diff=0.1):
    # create distance accumulation array
    # fill it with large values to mark
    # cells with unknown distance
    A = (1-S.copy())*1e10
    C = ndimage.gaussian_gradient_magnitude(I,1)

    sweeps = [A, A[:,::-1], A[::-1], A[::-1,::-1]]
    costs = [C, C[:,::-1], C[::-1], C[::-1,::-1]]
    for i, (a, c) in enumerate(it.cycle(zip(sweeps, costs))):
        r = sweep(a, c)
        if r < max_diff or i >= max_iter_n:
            break
    return A


def test_geodesic_distance2d():
    imgpath = osp.join(osp.join(osp.dirname(__file__), 'data'), 'img2d.png')
    I = np.asarray(Image.open(imgpath).convert('L'), np.float32)
    #I = np.zeros_like(I, np.uint8)
    S = np.zeros_like(I, np.uint8)
    S[100][100] = 1
    #I = np.asarray(ndimage.distance_transform_edt(1-S),np.float32)
    #dry run to make sure loading time is factored out
    #geodesic_distance.geodesic2d_fast_marching(I,S)
    chann = torch.tensor(1, dtype=torch.int32)

    print(I)
    It = torch.from_numpy(I)
    St = torch.from_numpy(S)

    dist = monai._C.geodesic2d_fast_marching(It, St, chann)
    print('--after--')
    print(It)
    print(dist)    

    geodesic2d_fsm(I,S)

    t0 = time.time()
    #D1 = geodesic_distance.geodesic2d_fast_marching(I, S)
    D1t = monai._C.geodesic2d_fast_marching(It, St, chann)
    print(D1t)
    t1 = time.time()
    #D2 = geodesic_distance.geodesic2d_raster_scan(I,S,1.0,2)
    lamb = torch.tensor(1.0, dtype=torch.float32)
    iter = torch.tensor(1, dtype=torch.int32)
    D2t = monai._C.geodesic2d_raster_scan(It, St, lamb, chann, iter)
    print(D2t)

    t2 = time.time()
    D3 = geodesic2d_fsm(I,S)
    print(D3)
    t3 = time.time()
    #D4 = geodesic_distance.geodesic2d_raster_scan(I,S,1.0,2)
    lamb = torch.tensor(1.0, dtype=torch.float32)
    iter = torch.tensor(2, dtype=torch.int32)
    D4t = monai._C.geodesic2d_raster_scan(It, St, lamb, chann, iter)
    print(D4t)
    t4 = time.time()

    dt1 = t1 - t0
    dt2 = t2 - t1
    dt3 = t3 - t2
    dt4 = t4 - t3
    print("runtime(s) of fast marching    {0:}".format(dt1))
    print("runtime(s) of raster scan      {0:}".format(dt2))
    print("runtime(s) of fast sweep       {0:}".format(dt3))
    print("runtime(s) of raster scan eucl {0:}".format(dt4))

    # Convert tensor to numpy array
    D1 = D1t.numpy()
    D2 = D2t.numpy()
    #D3 = D3t.numpy() D3 isnt tensor array
    D4 = D4t.numpy()
    plt.subplot(1,5,1); plt.imshow(I); plt.plot([100], [100], 'ro'); plt.title('input image')
    plt.subplot(1,5,2); plt.imshow(D1); plt.title('fast marching')
    plt.subplot(1,5,3); plt.imshow(D2); plt.title('raster scan')
    plt.subplot(1,5,4); plt.imshow(D3); plt.title('fast sweep')
    plt.subplot(1,5,4); plt.imshow(D4); plt.title('Euclidean')

    #plt.show()
    plt.savefig('test_geodis_2d.png')


if __name__ == '__main__':
    test_geodesic_distance2d()
