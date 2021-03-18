from __future__ import print_function

import site
print(site.getsitepackages())
site.addsitedir('./') # Replace with your MONAI directory
import torch
import monai

import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt
import nibabel
import os.path as osp


def load_nifty_volume_as_array(filename):
    # input shape [W, H, D]
    # output shape [D, H, W]
    img = nibabel.load(filename)
    data = img.get_fdata()
    data = np.transpose(data, [2,1,0])
    return data

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

def geodesic_distance_3d(It, St, lambt, itert, chann):
    '''
    get 3d geodesic disntance by raser scanning.
    I: input image
    S: binary image where non-zero pixels are used as seeds
    lamb: weighting betwween 0 and 1.
          0: spatial euclidean distance without considering gradient
          1: distance based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''

    '''
    torch::Tensor
    geodesic3d_raster_scan_torch(torch::Tensor I,  /* img float* 
                                         [depth,width,height,channel]*/
			     torch::Tensor S,  /* seeds uint8* */
			     torch::Tensor lamb,  /* float */
			     torch::Tensor chann, /* int depth for 3d, 1 for 2d */
			     torch::Tensor iter   /* int */
			     )
    '''
    #return geodesic_distance.geodesic3d_raster_scan(I, S, lamb, iter)
    #chann = torch.tensor(It.size(2), dtype=torch.int32)
    return monai._C.geodesic3d_raster_scan(It, St, lambt, chann, itert)

def test_geodesic_distance3d():
    imgpath = osp.join(osp.join(osp.dirname(__file__), 'data'), 'img3d.nii')
    I = load_nifty_volume_as_array(imgpath)

    I = np.asarray(I, np.float32)
    I = I[18:38, 63:183, 93:233 ]
    S = np.zeros_like(I, np.uint8)
    S[10][60][70] = 1

    #dry run to make sure loading time is factored out
    It = torch.from_numpy(I)
    St = torch.from_numpy(S)    
    #geodesic_distance.geodesic3d_raster_scan(I,S,0.0,1)
    # I,S,spacing,lamb,chann,iter
    lambt = torch.tensor(0.0, dtype=torch.float32)
    itert = torch.tensor(1, dtype=torch.int32)
    channt = torch.tensor(2, dtype=torch.int32)
    print(It)

    monai._C.geodesic3d_raster_scan(It, St, lambt, channt, itert)

    t0 = time.time()
    #D1 = geodesic_distance.geodesic3d_fast_marching(I,S)
    D1t = monai._C.geodesic3d_fast_marching(It, St, channt)
    print(D1t)

    #D1 = geodesic_distance_3d(I,S, 0, 4)
    t1 = time.time()
    #D2 = geodesic_distance_3d(I,S, 1.0, 4)
    lambt = torch.tensor(1.0, dtype=torch.float32)
    itert = torch.tensor(4, dtype=torch.int32)
    D2t = geodesic_distance_3d(It, St, lambt, itert, channt)
    t2 = time.time()
    print(D2t)
    #D3 = geodesic_distance_3d(I,S, 0.0, 4)
    lambt = torch.tensor(0.0, dtype=torch.float32)
    itert = torch.tensor(4, dtype=torch.int32)
    D3t = geodesic_distance_3d(It, St, lambt, itert, channt)
    print(D3t)
    t3 = time.time()

    dt1 = t1 - t0
    dt2 = t2 - t1
    dt3 = t3 - t2
    print("runtime(s) fast marching    {0:}".format(dt1))
    print("runtime(s) raster scan      {0:}".format(dt2))
    print("runtime(s) raster scan eucl {0:}".format(dt3))

    # save_array_as_nifty_volume(D1, "./data/image3d_dis1.nii")
    # save_array_as_nifty_volume(D2, "./data/image3d_dis2.nii")
    # save_array_as_nifty_volume(D3, "./data/image3d_dis3.nii")
    # save_array_as_nifty_volume(I, "./data/image3d_sub.nii")

    I = It.numpy()
    D1 = D1t.numpy()
    D2 = D2t.numpy()
    D3 = D3t.numpy()
    
    I = I*255/I.max()
    I_slice = I[10]
    D1_slice = D1[10]
    D2_slice = D2[10]
    D3_slice = D3[10]

    plt.subplot(1,4,1); plt.imshow(I_slice, cmap='gray')
    plt.autoscale(False);  plt.plot([70], [60], 'ro')
    plt.axis('off'); plt.title('input image')

    plt.subplot(1,4,2); plt.imshow(D1_slice)
    plt.axis('off'); plt.title('fast marching')

    plt.subplot(1,4,3); plt.imshow(D2_slice)
    plt.axis('off'); plt.title('raster scan')

    plt.subplot(1,4,4); plt.imshow(D3_slice)
    plt.axis('off'); plt.title('Euclidean distance')
    #plt.show()
    plt.savefig('test_geodis_3d.png')


    
if __name__ == '__main__':
    test_geodesic_distance3d()
