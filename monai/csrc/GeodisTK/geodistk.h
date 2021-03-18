/*
Copyright 2020 - 2021 MONAI Consortium
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <torch/extension.h>
#include <vector>
#include "util.h"


  
torch::Tensor
geodesic2d_fast_marching_torch(torch::Tensor I,  /*img float* */
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor chann /* int value */
			       );

torch::Tensor
geodesic3d_fast_marching_torch(torch::Tensor I,  /*img float* */
			       torch::Tensor S,  /* seeds uint8* */
			       torch::Tensor chann  /* int value */
			       );

torch::Tensor
geodesic2d_raster_scan_torch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			     torch::Tensor S,  /* seeds uint8* */
			     torch::Tensor lamb,  /* float */
			     torch::Tensor chann,  /* int */
			     torch::Tensor iter   /* int */
			     );

torch::Tensor
geodesic3d_raster_scan_torch(torch::Tensor I,  /* img float* [depth,width,height,channel]*/
			     torch::Tensor S,  /* seeds uint8* */
			     torch::Tensor lamb,  /* float */
			     torch::Tensor chann, /* int */
			     torch::Tensor iter   /* int */
			     );


