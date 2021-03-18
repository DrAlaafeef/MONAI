/*
  PyTorch interface functions got GeodisTK distance funcs

*/
#include <assert.h>
#include "geodesic_distance_2d.h"
#include "geodesic_distance_3d.h"
#include <iostream>
#include <cstdint>
#include "torch/script.h"


using namespace std;

//using at::TensorOptions;
//using c10::IntArrayRef;

#define HAVE_GEODISTK 1


/*
  C++ API
  Some tips:
  Create s torch int value: 

  x = torch.tenso(100); x.item() -> 100
  y = torch.tenso(99, dtype=torch.int32); x.item() -> 99
  z = torch.tenso(99, dtype=torch.float32); x.item() -> 99

  void geodesic3d_fast_marching(const float * img, const unsigned char * seeds,
  float * distance, int depth, int height, int width, int channel, 
  std::vector<float> spacing);
*/
torch::Tensor
geodesic3d_fast_marching_torch(torch::Tensor I,  /*img float* */
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor chann  /* int value */
			       )
{
  const float*   arr_I = I.data_ptr<float>();
  const uint8_t* arr_S = S.data_ptr<uint8_t>();
  int64_t  dim_I[3] = {I.size(0), I.size(1), I.size(2)};

  int channel = chann.item<int>();
  int depth =  I.size(0);
  int height = I.size(1);
  int width =  I.size(2);

  // Need to use std::vector as this is a parameter to the API func
  std::vector<float> sp_vec(3);
  for(int i = 0; i < 3; i++){
    sp_vec[i] = (float)dim_I[i];   // arr_spacing[i]; //sp[i];
  }
  torch::Tensor distance = torch::zeros(dim_I, torch::kFloat32);
  // return tensor
#ifdef HAVE_GEODISTK
  geodesic3d_fast_marching(arr_I, arr_S, distance.data_ptr<float>(),
			   depth, height, width, channel, sp_vec);
#endif
  return distance;
}


/*
  def geodesic_distance_2d_fast_marching(I, S):
  ''' 
  get 2d geodesic disntance by raser scanning.
  I: input image, can have multiple channels. Type should be np.float32.
  S: binary image where non-zero pixels are used as seeds. Type should be np.uint8.
  lamb: weighting betwween 0.0 and 1.0
  if lamb==0.0, return spatial euclidean distance without considering gradient
  if lamb==1.0, the distance is based on gradient only without using spatial distance
  iter: number of iteration for raster scanning.

  C++
  void geodesic2d_fast_marching(const float * img,
  const unsigned char * seeds,
  float * distance,
  int height, int width, int channel);

*/
torch::Tensor
geodesic2d_fast_marching_torch(torch::Tensor I,  /*img float* */
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor chann /* int value */
			       )
{
  const float*   arr_I = I.data_ptr<float>();
  const uint8_t* arr_S = S.data_ptr<uint8_t>();
  int channel = chann.item<int>();  
  int height = I.size(0);
  int width =  I.size(1);
  torch::Tensor distance = torch::zeros({I.size(0),I.size(1)}, torch::kFloat32);

  //cout << "height=" << height << "width=" << width << endl;
#ifdef HAVE_GEODISTK
  geodesic2d_fast_marching(arr_I, arr_S, distance.data_ptr<float>(),
			   height, width, channel);
#endif
  return distance;
}

/*

  void geodesic2d_raster_scan(const float * img, const unsigned char * seeds, 
  float * distance,int height, int width, int channel, float lambda, int iteration);
*/
torch::Tensor
geodesic2d_raster_scan_torch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			     torch::Tensor S,  /* seeds uint8* */
			     torch::Tensor lamb,  /* float */
			     torch::Tensor chann,  /* int */
			     torch::Tensor iter   /* int */
			     )
{
  const float*   arr_I = I.data_ptr<float>();
  const uint8_t* arr_S = S.data_ptr<uint8_t>();
  float lambda = lamb.item<float>();
  int   channel = chann.item<int>();
  int   iteration = iter.item<int>();
  torch::Tensor distance = torch::zeros({I.size(0),I.size(1)}, torch::kFloat32);  // return tensor torch::Float32
  int height = I.size(0);
  int width =  I.size(1);
#ifdef HAVE_GEODISTK
  geodesic2d_raster_scan(arr_I, arr_S, distance.data_ptr<float>(),
			 height, width, channel, lambda, (int)iteration);
#endif
  return distance;
}

/*
  def geodesic_distance_3d(I, S, spacing, lamb, iter):
  '''
  Get 3D geodesic disntance by raser scanning.
  I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
  S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
  Type should be np.uint8.
  spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
  lamb: weighting betwween 0.0 and 1.0
  if lamb==0.0, return spatial euclidean distance without considering gradient
  if lamb==1.0, the distance is based on gradient only without using spatial distance
  iter: number of iteration for raster scanning.
  '''
  return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)

  (Wrapper)return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)
  (C++API)void geodesic3d_raster_scan(const float * img,
  const unsigned char * seeds,
  float* distance,
  int depth, int height, int width, int channel,
  std::vector<float> spacing,
  float lambda,
  int iteration);

*/
torch::Tensor
geodesic3d_raster_scan_torch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			     torch::Tensor S,  /* seeds uint8* */
			     torch::Tensor lamb,  /* float */
			     torch::Tensor chann, /* int */
			     torch::Tensor iter   /* int */
			     )
{
  const float*   arr_I = I.data_ptr<float>();
  const uint8_t* arr_S = S.data_ptr<uint8_t>();
  float lambda = lamb.item<float>();
  int   channel = chann.item<int>();    
  int   iteration = iter.item<int>();
  int64_t  dim_I[3] = {I.size(0),I.size(1),I.size(2)};

  // Need to use std::vector as this is a parameter to the API func
  int depth = I.size(0);
  int height = I.size(1);
  int width =  I.size(2);

  std::vector<float> sp_vec(3);
  for(int i = 0; i < 3; i++){
    sp_vec[i] = (float)dim_I[i]; // arr_spacing[i]; //sp[i];
  }
  torch::Tensor distance = torch::zeros(dim_I, torch::kFloat32);  // return tensor torch::Float32
  //cout << "lambda=" << lambda << "channel=" << channel << "itera=" << iteration << endl;
#ifdef HAVE_GEODISTK  
  geodesic3d_raster_scan(arr_I, arr_S, distance.data_ptr<float>(),
			 depth, height, width, channel, sp_vec, lambda, (int) iteration);
#endif
  return distance;
}



/*
  TORCH_LIBRARY(my_ops, m) {
  m.def("geodesic3d_fast_marching_torch", geodesic3d_fast_marching_torch);
  m.def("geodesic2d_fast_marching_torch", geodesic2d_fast_marching_torch);
  m.def("geodesic3d_raster_scan_torch", geodesic3d_raster_scan_torch);
  m.def("geodesic2d_raster_scan_torch", geodesic2d_raster_scan_torch);
  }
*/
