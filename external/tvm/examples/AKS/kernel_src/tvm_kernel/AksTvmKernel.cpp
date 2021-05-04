/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 i* Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Caffe Kernel Implementation
// TODO : Make it thread-safe

// #define CPU_ONLY 1

#include <fstream>
#include <iterator>
#include <algorithm>
#include <typeinfo>

#include <dlpack/dlpack.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

//#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <bits/stdc++.h>
#include <vector>
#include "aks/AksKernelBase.h"
#include "aks/AksNodeParams.h"
#include "aks/AksDataDescriptor.h"

#include <cstdio>
#include <future>
#include <unistd.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

vector<vector<int64_t>> get_out_shape(vector<int> &out_dim, vector<int> &out_order)
{
  list<vector<int64_t>> listOfVec;
  vector<int64_t> temp;
  int counter = 0;
  int counter1 = 0;

  for (int i = 0; i < out_order.size(); i++)
  {

    for (int j = 0; j < out_order[counter1]; j++)
    {
      temp.push_back(out_dim[counter]);
      counter += 1;
    }
    counter1 += 1;
    listOfVec.push_back(temp);
    temp.clear();
  }

  vector<vector<int64_t>> v(listOfVec.begin(), listOfVec.end());
  return v;
}

struct TVMNodeObject
{
  std::vector<tvm::runtime::Module> tvmrunners;
  std::string lib;
  std::string input_name;
  DLContext ctx{kDLCPU, 0};
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;
  int in_ndim = 4;
  std::atomic<unsigned int> core_id{0};
  unsigned int core_count;
  vector<vector<int64_t>> out_shape;
};

class TvmKernelBase : public AKS::KernelBase
{
public:
  int exec_async(
      std::vector<AKS::DataDescriptor *> &in,
      std::vector<AKS::DataDescriptor *> &out,
      AKS::NodeParams *params,
      AKS::DynamicParamValues *dynParams);
  void nodeInit(AKS::NodeParams *);

private:
  std::map<AKS::NodeParams *, TVMNodeObject> nodes;
  string _device = "CPU";
};

extern "C"
{
  AKS::KernelBase *getKernel(AKS::NodeParams *params)
  {
    return new TvmKernelBase();
  }
} // extern C

void TvmKernelBase::nodeInit(AKS::NodeParams *params)
{
  nodes.emplace(std::piecewise_construct,
                       std::forward_as_tuple(params),
                       std::forward_as_tuple());

  auto num_runners = params->hasKey<int>("num_runners") ? params->getValue<int>("num_runners") : 1;
  nodes[params].core_count = num_runners;
  auto lib = params->_stringParams["lib"];
  //tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(lib);
  for (int i = 0; i < num_runners; i++)
  {

  //  auto lib = params->_stringParams["lib"];
    auto input_name = params->_stringParams["input_name"];
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(lib);
    DLDevice ctx{kDLCPU, 0};
    tvm::runtime::Module mod = mod_factory.GetFunction("default")(ctx);

    nodes[params].tvmrunners.push_back(std::move(mod));
    nodes[params].lib = lib;
    nodes[params].input_name = input_name;

    auto indimIter = params->_intVectorParams.find("in_dim");
    int64_t in_dim[4];
    in_dim[0] = indimIter->second[0]; //N
    in_dim[1] = indimIter->second[1]; //C;
    in_dim[2] = indimIter->second[2]; //H;
    in_dim[3] = indimIter->second[3]; //W;

    vector<int> out_dim = params->_intVectorParams["out_dim"];
    vector<int> out_order = params->_intVectorParams["out_order"];

    nodes[params].out_shape = get_out_shape(out_dim, out_order);
  }
  nodes[params].core_count = num_runners;
}

int TvmKernelBase::exec_async(
    vector<AKS::DataDescriptor *> &in, vector<AKS::DataDescriptor *> &out,
    AKS::NodeParams *params, AKS::DynamicParamValues *dynParams)
{
  auto &curNode = nodes[params];
  std::vector<tvm::runtime::Module> &curRunners1 = curNode.tvmrunners;

  unsigned int tmpID = curNode.core_id++;
  unsigned int runnerID = tmpID % curNode.core_count;
  tvm::runtime::Module curRunners = curRunners1[runnerID];
  auto indimIter = params->_intVectorParams.find("in_dim");
  int64_t in_dim[4];
  in_dim[0] = indimIter->second[0]; //N
  in_dim[1] = indimIter->second[1]; //C;
  in_dim[2] = indimIter->second[2]; //H;
  in_dim[3] = indimIter->second[3]; //W;
  vector<int64_t> in_dim_vec{in_dim[0], in_dim[1], in_dim[2], in_dim[3]};
  DLTensor *in_data;

  for (int i = 0; i < 1; ++i)
  {
    float *inData = static_cast<float *>(in[i]->data());
    tvm::runtime::PackedFunc set_input = curRunners.GetFunction("set_input");
    TVMArrayAlloc(in_dim, curNode.in_ndim, curNode.dtype_code,
                  curNode.dtype_bits, curNode.dtype_lanes,
                  curNode.device_type, curNode.device_id, &in_data);

    int64_t size = std::accumulate(in_dim_vec.begin(), in_dim_vec.end(), 1, std::multiplies<int64_t>());
    TVMArrayCopyFromBytes(in_data, inData, size * 4);
    set_input(curNode.input_name, in_data);
  }
  
  tvm::runtime::PackedFunc run = curRunners.GetFunction("run");
  run();

  for (int i = 0; i < curNode.out_shape.size(); ++i)
  {
    vector<int> out_shape(curNode.out_shape[i].begin(), curNode.out_shape[i].end());
    int64_t shape_arr[curNode.out_shape[i].size()];
    std::copy(curNode.out_shape[i].begin(), curNode.out_shape[i].end(), shape_arr);
    AKS::DataDescriptor *outDD = new AKS::DataDescriptor(out_shape, AKS::DataType::FLOAT32);
    out.push_back(outDD);
    float *outData = outDD->data<float>();
    tvm::runtime::PackedFunc get_output = curRunners.GetFunction("get_output");
    tvm::runtime::NDArray res = get_output(i);
    int64_t size = std::accumulate(curNode.out_shape[i].begin(), curNode.out_shape[i].end(), 1, std::multiplies<int64_t>());
    memcpy(outData, res->data, size * 4);
  }

  TVMArrayFree(in_data);

  return -1;
}

