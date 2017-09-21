/*******************************************************************************
//
//  SYCL 1.2.1 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "math_builtin.h"

#define TEST_NAME math_builtins

namespace TEST_NAMESPACE {
using namespace sycl_cts;
using namespace cl::sycl;

class TEST_NAME : public util::test_base {
public:
  /** return information about this test
     */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void run(util::logger &log) override { test_function<0, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7755374812,0.7063635224,0.4364572647);
cl::sycl::float3 inputData_1(0.3071334002,0.5090197771,0.42394731);
return cl::sycl::cross(inputData_0,inputData_1);

});

test_function<1, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7270388712,0.3426501809,0.4812775633,0.5667056316);
cl::sycl::float4 inputData_1(0.8264903082,0.5037494847,0.3254702755,0.7046433633);
return cl::sycl::cross(inputData_0,inputData_1);

});

test_function<2, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5946951973,0.3004050731,0.8277970048);
cl::sycl::double3 inputData_1(0.8862283808,0.7481737888,0.8217327604);
return cl::sycl::cross(inputData_0,inputData_1);

});

test_function<3, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3481180555,0.6838653986,0.8190706304,0.6471871455);
cl::sycl::double4 inputData_1(0.4777141724,0.1805609665,0.4473374684,0.5887095788);
return cl::sycl::cross(inputData_0,inputData_1);

});

test_function<4, float>(
[=](){
float inputData_0(0.8304088426);
float inputData_1(0.8732850942);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<5, float>(
[=](){
cl::sycl::float2 inputData_0(0.4816078212,0.7922479422);
cl::sycl::float2 inputData_1(0.3083938483,0.7440222616);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<6, float>(
[=](){
cl::sycl::float3 inputData_0(0.5389594431,0.1112333601,0.6757637491);
cl::sycl::float3 inputData_1(0.4190588338,0.7598759817,0.634522561);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<7, float>(
[=](){
cl::sycl::float4 inputData_0(0.1009142555,0.4948622932,0.7940822204,0.2951287015);
cl::sycl::float4 inputData_1(0.3601634902,0.7963769857,0.2528536732,0.5540085925);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<8, double>(
[=](){
double inputData_0(0.2908927429);
double inputData_1(0.8740322002);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<9, double>(
[=](){
cl::sycl::double2 inputData_0(0.7425435754,0.4583756571);
cl::sycl::double2 inputData_1(0.1643566548,0.3560436837);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<10, double>(
[=](){
cl::sycl::double3 inputData_0(0.506352514,0.8462670594,0.1872462767);
cl::sycl::double3 inputData_1(0.5410137969,0.6652491279,0.5379527291);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<11, double>(
[=](){
cl::sycl::double4 inputData_0(0.7515734906,0.5322268856,0.8710708368,0.5825485024);
cl::sycl::double4 inputData_1(0.5700936513,0.455991221,0.5770294893,0.4079209168);
return cl::sycl::dot(inputData_0,inputData_1);

});

test_function<12, float>(
[=](){
float inputData_0(0.5605208113);
float inputData_1(0.3322636019);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<13, float>(
[=](){
cl::sycl::float2 inputData_0(0.2515130628,0.2493836226);
cl::sycl::float2 inputData_1(0.5902185439,0.6253275112);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<14, float>(
[=](){
cl::sycl::float3 inputData_0(0.4812247936,0.171859489,0.7060831376);
cl::sycl::float3 inputData_1(0.8014162967,0.8387048128,0.7739681785);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<15, float>(
[=](){
cl::sycl::float4 inputData_0(0.8185384971,0.8384659519,0.53247994,0.4130368402);
cl::sycl::float4 inputData_1(0.6642267199,0.320507297,0.7493029668,0.7795887721);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<16, double>(
[=](){
double inputData_0(0.8160311739);
double inputData_1(0.5718409468);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<17, double>(
[=](){
cl::sycl::double2 inputData_0(0.8598118986,0.5637560086);
cl::sycl::double2 inputData_1(0.4604504853,0.6281963029);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<18, double>(
[=](){
cl::sycl::double3 inputData_0(0.8970062715,0.8335529744,0.7346600673);
cl::sycl::double3 inputData_1(0.1658983906,0.590226484,0.4891553616);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<19, double>(
[=](){
cl::sycl::double4 inputData_0(0.6041178723,0.7760620605,0.2944284976,0.6851913766);
cl::sycl::double4 inputData_1(0.1937074346,0.2763684295,0.7356663774,0.3660289194);
return cl::sycl::distance(inputData_0,inputData_1);

});

test_function<20, float>(
[=](){
float inputData_0(0.7527304772);
return cl::sycl::length(inputData_0);

});

test_function<21, float>(
[=](){
cl::sycl::float2 inputData_0(0.1804860162,0.2170867911);
return cl::sycl::length(inputData_0);

});

test_function<22, float>(
[=](){
cl::sycl::float3 inputData_0(0.6581365122,0.1361872543,0.5590928294);
return cl::sycl::length(inputData_0);

});

test_function<23, float>(
[=](){
cl::sycl::float4 inputData_0(0.8280128118,0.5273583746,0.644471306,0.1213574357);
return cl::sycl::length(inputData_0);

});

test_function<24, double>(
[=](){
double inputData_0(0.6079999279);
return cl::sycl::length(inputData_0);

});

test_function<25, double>(
[=](){
cl::sycl::double2 inputData_0(0.5850707342,0.5607623584);
return cl::sycl::length(inputData_0);

});

test_function<26, double>(
[=](){
cl::sycl::double3 inputData_0(0.4129675275,0.3961119523,0.8844133205);
return cl::sycl::length(inputData_0);

});

test_function<27, double>(
[=](){
cl::sycl::double4 inputData_0(0.1291136301,0.1173092079,0.8688250242,0.2479775531);
return cl::sycl::length(inputData_0);

});

test_function<28, double>(
[=](){
double inputData_0(0.1991161315);
return cl::sycl::normalize(inputData_0);

});

test_function<29, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2684612079,0.7405972723);
return cl::sycl::normalize(inputData_0);

});

test_function<30, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8495753269,0.1182260605,0.4404950656);
return cl::sycl::normalize(inputData_0);

});

test_function<31, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1812001755,0.3079359118,0.2766634171,0.6175405759);
return cl::sycl::normalize(inputData_0);

});

test_function<32, float>(
[=](){
float inputData_0(0.3802351739);
float inputData_1(0.2442543212);
return cl::sycl::fast_distance(inputData_0,inputData_1);

});

test_function<33, float>(
[=](){
cl::sycl::float2 inputData_0(0.5029092042,0.1315029657);
cl::sycl::float2 inputData_1(0.180736993,0.890588119);
return cl::sycl::fast_distance(inputData_0,inputData_1);

});

test_function<34, float>(
[=](){
cl::sycl::float3 inputData_0(0.2594846324,0.386844241,0.685278645);
cl::sycl::float3 inputData_1(0.7706612522,0.8347856496,0.2355396849);
return cl::sycl::fast_distance(inputData_0,inputData_1);

});

test_function<35, float>(
[=](){
cl::sycl::float4 inputData_0(0.6381124509,0.8732391224,0.1464407551,0.6409614274);
cl::sycl::float4 inputData_1(0.776339675,0.3738500329,0.3005498714,0.5774331148);
return cl::sycl::fast_distance(inputData_0,inputData_1);

});

test_function<36, float>(
[=](){
float inputData_0(0.453851227);
return cl::sycl::fast_length(inputData_0);

});

test_function<37, float>(
[=](){
cl::sycl::float2 inputData_0(0.2398555876,0.4773003321);
return cl::sycl::fast_length(inputData_0);

});

test_function<38, float>(
[=](){
cl::sycl::float3 inputData_0(0.4279243165,0.5552901916,0.5068801041);
return cl::sycl::fast_length(inputData_0);

});

test_function<39, float>(
[=](){
cl::sycl::float4 inputData_0(0.3491568008,0.3857213461,0.7701289395,0.3007461319);
return cl::sycl::fast_length(inputData_0);

});

test_function<40, float>(
[=](){
float inputData_0(0.5484801751);
return cl::sycl::fast_normalize(inputData_0);

});

test_function<41, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1099490551,0.6932595019);
return cl::sycl::fast_normalize(inputData_0);

});

test_function<42, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3687332436,0.1365571949,0.3247065314);
return cl::sycl::fast_normalize(inputData_0);

});

test_function<43, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2921043263,0.8625034719,0.3817804492,0.3303023319);
return cl::sycl::fast_normalize(inputData_0);

});

 }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}