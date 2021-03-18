'''
geodesic2d_fast_marching_pytorch(torch::Tensor I,  /*img float* */
				 torch::Tensor S,  /* seeds uint8* */
				 torch::Tensor chann /* int value */
				 )
geodesic3d_fast_marching_pytorch(torch::Tensor I,  /*img float* */
				 torch::Tensor S,  /* seeds uint8* */
				 torch::Tensor spacing, /* float vec */
				 torch::Tensor chann  /* int value */
				 )
geodesic2d_raster_scan_pytorch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor lamb,  /* float */
			       torch::Tensor chann,  /* int */
			       torch::Tensor iter   /* int */
			       )
geodesic2d_raster_scan_pytorch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor lamb,  /* float */
			       torch::Tensor chann,  /* int */
			       torch::Tensor iter   /* int */
			       )
'''

import torch
torch.ops.load_library("..//build/lib.linux-x86_64-3.8/monai/_C.so")

print(torch.ops.my_ops.geodesic2d_raster_scan_pytorch)
print(torch.ops.my_ops.geodesic3d_raster_scan_pytorch)
print(torch.ops.my_ops.geodesic2d_fast_marching_pytorch)
print(torch.ops.my_ops.geodesic3d_fast_marching_pytorch)

I = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
S = torch.tensor([[1,1,1],[1,1,1]], dtype=torch.uint8)
chann = torch.tensor(1, dtype=torch.int32)
iter = torch.tensor(1, dtype=torch.int32)
print(I)
print(S)
print(chann.item())
print(iter.item())


dist = torch.ops.my_ops.geodesic2d_fast_marching_pytorch(I, S, chann)
print("geodesic2d_fast_marching_pytorch", dist)

I = torch.tensor([[[1,2,3], [4,5,6]]], dtype=torch.float32)
S = torch.tensor([[[1,1,1], [1,1,1]]], dtype=torch.uint8)
spacing = torch.tensor([[1,1,1]], dtype=torch.float32)

dist = torch.ops.my_ops.geodesic3d_fast_marching_pytorch(I, S, spacing, chann)
print("geodesic2d_fast_marching_pytorch", dist)

lamb = torch.tensor(0.5, dtype=torch.float32)
dist = torch.ops.my_ops.geodesic3d_raster_scan_pytorch(I, S, spacing, lamb, chann, iter)
print("geodesic3d_raster_scan_pytorch", dist)

dist = torch.ops.my_ops.geodesic2d_raster_scan_pytorch(I, S, lamb, chann, iter)
print("geodesic2d_raster_scan_pytorch", dist)
