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

class TEST_NAME : public util::test_base {
public:
  /** return information about this test
     */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  void run(util::logger &log) override { test_function<0, float>(
[=](){
float inputData_0(0.7755374812);
return cl::sycl::acos(inputData_0);

});

test_function<1, double>(
[=](){
double inputData_0(0.7063635224);
return cl::sycl::acos(inputData_0);

});

test_function<2, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4364572647);
return cl::sycl::acos(inputData_0);

});

test_function<3, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3071334002,0.5090197771);
return cl::sycl::acos(inputData_0);

});

test_function<4, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.42394731,0.7270388712,0.3426501809);
return cl::sycl::acos(inputData_0);

});

test_function<5, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4812775633,0.5667056316,0.8264903082,0.5037494847);
return cl::sycl::acos(inputData_0);

});

test_function<6, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.3254702755,0.7046433633,0.5946951973,0.3004050731,0.8277970048,0.8862283808,0.7481737888,0.8217327604);
return cl::sycl::acos(inputData_0);

});

test_function<7, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3481180555,0.6838653986,0.8190706304,0.6471871455,0.4777141724,0.1805609665,0.4473374684,0.5887095788,0.8304088426,0.8732850942,0.4816078212,0.7922479422,0.3083938483,0.7440222616,0.5389594431,0.1112333601);
return cl::sycl::acos(inputData_0);

});

test_function<8, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6757637491,0.4190588338);
return cl::sycl::acos(inputData_0);

});

test_function<9, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7598759817,0.634522561,0.1009142555);
return cl::sycl::acos(inputData_0);

});

test_function<10, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4948622932,0.7940822204,0.2951287015,0.3601634902);
return cl::sycl::acos(inputData_0);

});

test_function<11, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7963769857,0.2528536732,0.5540085925,0.2908927429,0.8740322002,0.7425435754,0.4583756571,0.1643566548);
return cl::sycl::acos(inputData_0);

});

test_function<12, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3560436837,0.506352514,0.8462670594,0.1872462767,0.5410137969,0.6652491279,0.5379527291,0.7515734906,0.5322268856,0.8710708368,0.5825485024,0.5700936513,0.455991221,0.5770294893,0.4079209168,0.5605208113);
return cl::sycl::acos(inputData_0);

});

test_function<13, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3322636019,0.2515130628);
return cl::sycl::acos(inputData_0);

});

test_function<14, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2493836226,0.5902185439,0.6253275112);
return cl::sycl::acos(inputData_0);

});

test_function<15, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4812247936,0.171859489,0.7060831376,0.8014162967);
return cl::sycl::acos(inputData_0);

});

test_function<16, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8387048128,0.7739681785,0.8185384971,0.8384659519,0.53247994,0.4130368402,0.6642267199,0.320507297);
return cl::sycl::acos(inputData_0);

});

test_function<17, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.7493029668,0.7795887721,0.8160311739,0.5718409468,0.8598118986,0.5637560086,0.4604504853,0.6281963029,0.8970062715,0.8335529744,0.7346600673,0.1658983906,0.590226484,0.4891553616,0.6041178723,0.7760620605);
return cl::sycl::acos(inputData_0);

});

test_function<18, float>(
[=](){
float inputData_0(0.2944284976);
return cl::sycl::acosh(inputData_0);

});

test_function<19, double>(
[=](){
double inputData_0(0.6851913766);
return cl::sycl::acosh(inputData_0);

});

test_function<20, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1937074346);
return cl::sycl::acosh(inputData_0);

});

test_function<21, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2763684295,0.7356663774);
return cl::sycl::acosh(inputData_0);

});

test_function<22, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3660289194,0.7527304772,0.1804860162);
return cl::sycl::acosh(inputData_0);

});

test_function<23, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2170867911,0.6581365122,0.1361872543,0.5590928294);
return cl::sycl::acosh(inputData_0);

});

test_function<24, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8280128118,0.5273583746,0.644471306,0.1213574357,0.6079999279,0.5850707342,0.5607623584,0.4129675275);
return cl::sycl::acosh(inputData_0);

});

test_function<25, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3961119523,0.8844133205,0.1291136301,0.1173092079,0.8688250242,0.2479775531,0.1991161315,0.2684612079,0.7405972723,0.8495753269,0.1182260605,0.4404950656,0.1812001755,0.3079359118,0.2766634171,0.6175405759);
return cl::sycl::acosh(inputData_0);

});

test_function<26, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3802351739,0.2442543212);
return cl::sycl::acosh(inputData_0);

});

test_function<27, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5029092042,0.1315029657,0.180736993);
return cl::sycl::acosh(inputData_0);

});

test_function<28, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.890588119,0.2594846324,0.386844241,0.685278645);
return cl::sycl::acosh(inputData_0);

});

test_function<29, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7706612522,0.8347856496,0.2355396849,0.6381124509,0.8732391224,0.1464407551,0.6409614274,0.776339675);
return cl::sycl::acosh(inputData_0);

});

test_function<30, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3738500329,0.3005498714,0.5774331148,0.453851227,0.2398555876,0.4773003321,0.4279243165,0.5552901916,0.5068801041,0.3491568008,0.3857213461,0.7701289395,0.3007461319,0.5484801751,0.1099490551,0.6932595019);
return cl::sycl::acosh(inputData_0);

});

test_function<31, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3687332436,0.1365571949);
return cl::sycl::acosh(inputData_0);

});

test_function<32, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.3247065314,0.2921043263,0.8625034719);
return cl::sycl::acosh(inputData_0);

});

test_function<33, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3817804492,0.3303023319,0.3873609578,0.8575246685);
return cl::sycl::acosh(inputData_0);

});

test_function<34, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6069982818,0.5968614765,0.6724954802,0.4104137883,0.4315343906,0.6206662898,0.1012193775,0.253847633);
return cl::sycl::acosh(inputData_0);

});

test_function<35, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3675213525,0.2915327681,0.6099195209,0.4029184563,0.8003387134,0.5545211367,0.4315251173,0.4218136601,0.6614636991,0.4345812426,0.6297567112,0.1374237488,0.4562817518,0.3073815388,0.2261492577,0.5220585041);
return cl::sycl::acosh(inputData_0);

});

test_function<36, float>(
[=](){
float inputData_0(0.4898124809);
return cl::sycl::acospi(inputData_0);

});

test_function<37, double>(
[=](){
double inputData_0(0.5491239405);
return cl::sycl::acospi(inputData_0);

});

test_function<38, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7043878138);
return cl::sycl::acospi(inputData_0);

});

test_function<39, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8071001234,0.4956661363);
return cl::sycl::acospi(inputData_0);

});

test_function<40, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3496465971,0.4735137883,0.7472366859);
return cl::sycl::acospi(inputData_0);

});

test_function<41, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8000130652,0.7499319459,0.2504010352,0.8995362876);
return cl::sycl::acospi(inputData_0);

});

test_function<42, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6064710079,0.1667736401,0.6804434844,0.8894571842,0.4214534578,0.6428120042,0.3529417098,0.2708197297);
return cl::sycl::acospi(inputData_0);

});

test_function<43, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6738593146,0.1018860518,0.7581851284,0.5226767815,0.1782274734,0.1951231158,0.6194123399,0.7989230591,0.3239861947,0.8828121494,0.1801445513,0.7831504877,0.4173569419,0.1650763334,0.3197710747,0.4623825479);
return cl::sycl::acospi(inputData_0);

});

test_function<44, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7338732249,0.7890879229);
return cl::sycl::acospi(inputData_0);

});

test_function<45, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2067364434,0.5166924227,0.6206265905);
return cl::sycl::acospi(inputData_0);

});

test_function<46, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3776424117,0.7974910686,0.3227278522,0.114859462);
return cl::sycl::acospi(inputData_0);

});

test_function<47, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1325306189,0.6447974161,0.5466845889,0.8572020433,0.8507510398,0.8278809419,0.1336036256,0.6993078587);
return cl::sycl::acospi(inputData_0);

});

test_function<48, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6610598541,0.6242894917,0.669886122,0.8221681205,0.6121129598,0.3979594104,0.530343027,0.266275283,0.5697004038,0.1071176656,0.2208185391,0.3667267104,0.7316985271,0.6747995382,0.370604776,0.5964304867);
return cl::sycl::acospi(inputData_0);

});

test_function<49, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1329623596,0.2310884365);
return cl::sycl::acospi(inputData_0);

});

test_function<50, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8855312561,0.3316246829,0.4158335864);
return cl::sycl::acospi(inputData_0);

});

test_function<51, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5387874373,0.3347256012,0.4824517353,0.2917648669);
return cl::sycl::acospi(inputData_0);

});

test_function<52, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1386050898,0.2436694792,0.5184401854,0.1566903073,0.4225353172,0.362816568,0.4317772872,0.1795202706);
return cl::sycl::acospi(inputData_0);

});

test_function<53, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8269260435,0.4792037209,0.7726786661,0.8809835661,0.3749212749,0.4832692153,0.6596762329,0.4412282588,0.341522493,0.687800793,0.8155198226,0.8357510755,0.6013936374,0.4004570771,0.8796484172,0.611102814);
return cl::sycl::acospi(inputData_0);

});

test_function<54, float>(
[=](){
float inputData_0(0.1526677418);
return cl::sycl::asin(inputData_0);

});

test_function<55, double>(
[=](){
double inputData_0(0.1677356553);
return cl::sycl::asin(inputData_0);

});

test_function<56, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6998956574);
return cl::sycl::asin(inputData_0);

});

test_function<57, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1489249252,0.1062808043);
return cl::sycl::asin(inputData_0);

});

test_function<58, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4150463614,0.515202983,0.4588354285);
return cl::sycl::asin(inputData_0);

});

test_function<59, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4908950435,0.5679109616,0.6434420539,0.4384304588);
return cl::sycl::asin(inputData_0);

});

test_function<60, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.3946651651,0.8907672465,0.3087332284,0.7216801236,0.4449768197,0.3868163056,0.1510863592,0.7908631554);
return cl::sycl::asin(inputData_0);

});

test_function<61, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6616033198,0.822408566,0.4612894341,0.6415367735,0.1951282292,0.4183628813,0.2657855787,0.1336811423,0.858369081,0.2727154948,0.2170835918,0.2583760348,0.4024255715,0.5371130099,0.2210674948,0.8909519112);
return cl::sycl::asin(inputData_0);

});

test_function<62, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8863913684,0.2187216137);
return cl::sycl::asin(inputData_0);

});

test_function<63, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4247255065,0.6439435865,0.8021252663);
return cl::sycl::asin(inputData_0);

});

test_function<64, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4963247399,0.8336373382,0.3579682519,0.4987527132);
return cl::sycl::asin(inputData_0);

});

test_function<65, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4989172735,0.6360545211,0.261593047,0.5878164883,0.2750184775,0.372176252,0.8700531706,0.8192064304);
return cl::sycl::asin(inputData_0);

});

test_function<66, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.7544947047,0.1283746095,0.218693506,0.3055055297,0.7273332546,0.7738666617,0.5663585442,0.6745053214,0.745644304,0.1530873048,0.1677145095,0.7951162512,0.1315326635,0.2800725229,0.1325056213,0.112228112);
return cl::sycl::asin(inputData_0);

});

test_function<67, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7751637486,0.3644754938);
return cl::sycl::asin(inputData_0);

});

test_function<68, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2285520482,0.2190555922,0.6248669294);
return cl::sycl::asin(inputData_0);

});

test_function<69, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8748786174,0.5039997541,0.8208723815,0.5019428792);
return cl::sycl::asin(inputData_0);

});

test_function<70, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.559097982,0.6428570854,0.7440879912,0.7062771058,0.8924260502,0.6975723113,0.8246245787,0.2648838657);
return cl::sycl::asin(inputData_0);

});

test_function<71, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5283330435,0.5788914109,0.7605572937,0.4857708505,0.7328321694,0.4108551121,0.5691107645,0.781053286,0.7384475769,0.6255876415,0.1001925572,0.2455751377,0.5054862295,0.3035675188,0.1524966746,0.7879067377);
return cl::sycl::asin(inputData_0);

});

test_function<72, float>(
[=](){
float inputData_0(0.8543576171);
return cl::sycl::asinh(inputData_0);

});

test_function<73, double>(
[=](){
double inputData_0(0.3422439025);
return cl::sycl::asinh(inputData_0);

});

test_function<74, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4264585339);
return cl::sycl::asinh(inputData_0);

});

test_function<75, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7480300271,0.1498070071);
return cl::sycl::asinh(inputData_0);

});

test_function<76, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.61278789,0.2018566503,0.329670672);
return cl::sycl::asinh(inputData_0);

});

test_function<77, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7639525493,0.1444216367,0.1287470667,0.4342928358);
return cl::sycl::asinh(inputData_0);

});

test_function<78, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4934647673,0.7906601465,0.6737509971,0.6388350469,0.2210990179,0.8893647394,0.428912157,0.5894166915);
return cl::sycl::asinh(inputData_0);

});

test_function<79, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4093464044,0.1376263326,0.4767113672,0.2210942031,0.1259723699,0.5939203389,0.603973033,0.1842342597,0.539315013,0.3773343813,0.4067312585,0.721135919,0.492255742,0.8050212923,0.5880957943,0.473750732);
return cl::sycl::asinh(inputData_0);

});

test_function<80, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.605850112,0.3702923039);
return cl::sycl::asinh(inputData_0);

});

test_function<81, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.199459034,0.646023695,0.5976299542);
return cl::sycl::asinh(inputData_0);

});

test_function<82, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7308531931,0.2016873,0.8294266545,0.7394729691);
return cl::sycl::asinh(inputData_0);

});

test_function<83, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8335099265,0.7980277774,0.6448051571,0.7482006795,0.5152058474,0.7283913195,0.2513019743,0.7256912851);
return cl::sycl::asinh(inputData_0);

});

test_function<84, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4556636832,0.705292977,0.4643761894,0.7316469826,0.1602716682,0.1357127243,0.8474316659,0.4889320806,0.8208571197,0.8558266015,0.633208922,0.5574374609,0.2727835073,0.1747809754,0.7555153721,0.8110176541);
return cl::sycl::asinh(inputData_0);

});

test_function<85, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7235165686,0.6588019462);
return cl::sycl::asinh(inputData_0);

});

test_function<86, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.4360888929,0.344249272,0.1907559165);
return cl::sycl::asinh(inputData_0);

});

test_function<87, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4407761985,0.5528103794,0.8383044665,0.8486038155);
return cl::sycl::asinh(inputData_0);

});

test_function<88, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4325129572,0.1793687905,0.719054986,0.6874234733,0.1245606768,0.4573748793,0.6491344834,0.1241073876);
return cl::sycl::asinh(inputData_0);

});

test_function<89, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8354258827,0.8697939892,0.6780342177,0.1628308317,0.1562635727,0.3874026519,0.1235020062,0.3783021818,0.1079713931,0.8794588103,0.7552053593,0.1564140892,0.8147480735,0.266382432,0.2638326386,0.6390073164);
return cl::sycl::asinh(inputData_0);

});

test_function<90, float>(
[=](){
float inputData_0(0.8506098145);
return cl::sycl::asinpi(inputData_0);

});

test_function<91, double>(
[=](){
double inputData_0(0.198550497);
return cl::sycl::asinpi(inputData_0);

});

test_function<92, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1057476538);
return cl::sycl::asinpi(inputData_0);

});

test_function<93, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3953041177,0.1197200115);
return cl::sycl::asinpi(inputData_0);

});

test_function<94, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5838785901,0.7873404869,0.2495933619);
return cl::sycl::asinpi(inputData_0);

});

test_function<95, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.1899128287,0.3755596859,0.8673372165,0.2041261555);
return cl::sycl::asinpi(inputData_0);

});

test_function<96, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8732154084,0.389791896,0.4786963222,0.3341055888,0.8497014754,0.866518316,0.6087325652,0.2472364401);
return cl::sycl::asinpi(inputData_0);

});

test_function<97, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8943614309,0.1820643516,0.5646795053,0.2251224481,0.8181402513,0.8565427132,0.7435122384,0.3527131349,0.294270952,0.7038867306,0.3328476153,0.4358283023,0.1370045415,0.2057870483,0.1164396965,0.1623368961);
return cl::sycl::asinpi(inputData_0);

});

test_function<98, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.1585689195,0.4361853617);
return cl::sycl::asinpi(inputData_0);

});

test_function<99, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5406217421,0.6927030559,0.2138267791);
return cl::sycl::asinpi(inputData_0);

});

test_function<100, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4377509969,0.6095728299,0.1676445559,0.4558489241);
return cl::sycl::asinpi(inputData_0);

});

test_function<101, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.3954048314,0.8591455432,0.1462856911,0.4269009769,0.4337803838,0.6825444037,0.3565368023,0.2631922208);
return cl::sycl::asinpi(inputData_0);

});

test_function<102, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3346493241,0.476710034,0.8602146637,0.7372136182,0.3215761966,0.5465452707,0.6505602429,0.7365257245,0.4569315072,0.4190215241,0.7141125943,0.4453731965,0.2983661351,0.4627576252,0.849683717,0.2140539906);
return cl::sycl::asinpi(inputData_0);

});

test_function<103, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4699482836,0.6098428195);
return cl::sycl::asinpi(inputData_0);

});

test_function<104, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.4866303906,0.2629119235,0.1014745285);
return cl::sycl::asinpi(inputData_0);

});

test_function<105, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6591933694,0.5949884144,0.1062213195,0.3388480968);
return cl::sycl::asinpi(inputData_0);

});

test_function<106, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7149074076,0.6031363028,0.5361664928,0.2249768878,0.6650352344,0.4771479374,0.642542997,0.7080718694);
return cl::sycl::asinpi(inputData_0);

});

test_function<107, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2858901772,0.7095960105,0.3240707078,0.8872121097,0.1966652886,0.806974415,0.1324377,0.3052606547,0.520881527,0.5652929468,0.416987988,0.1816253826,0.3020864687,0.3267172031,0.7041782836,0.8270194602);
return cl::sycl::asinpi(inputData_0);

});

test_function<108, float>(
[=](){
float inputData_0(0.5763279324);
return cl::sycl::atan(inputData_0);

});

test_function<109, double>(
[=](){
double inputData_0(0.1283607726);
return cl::sycl::atan(inputData_0);

});

test_function<110, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7337891773);
return cl::sycl::atan(inputData_0);

});

test_function<111, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3444831463,0.3719123251);
return cl::sycl::atan(inputData_0);

});

test_function<112, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5241483501,0.2992376381,0.8359824703);
return cl::sycl::atan(inputData_0);

});

test_function<113, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2308438067,0.4318643204,0.3317535596,0.5158672818);
return cl::sycl::atan(inputData_0);

});

test_function<114, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.5591854425,0.6017117513,0.5251006431,0.4286436019,0.6076752099,0.4227303013,0.7228402072,0.7305419402);
return cl::sycl::atan(inputData_0);

});

test_function<115, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3338033345,0.3974434588,0.6030487248,0.2256559737,0.6576255448,0.4051422024,0.5728499798,0.2116264794,0.6346067089,0.3832462885,0.478132461,0.4320859207,0.4813721984,0.6557565063,0.3545921415,0.6216435847);
return cl::sycl::atan(inputData_0);

});

test_function<116, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.148177686,0.340148122);
return cl::sycl::atan(inputData_0);

});

test_function<117, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6961677521,0.1419247024,0.5969137562);
return cl::sycl::atan(inputData_0);

});

test_function<118, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1204374394,0.4772230946,0.810836035,0.1080880752);
return cl::sycl::atan(inputData_0);

});

test_function<119, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.5214624165,0.1531654637,0.7936878209,0.6490372178,0.6935630853,0.635206064,0.105138763,0.1329422898);
return cl::sycl::atan(inputData_0);

});

test_function<120, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5967014432,0.8997481005,0.7985177913,0.6597486454,0.6816799635,0.2813496181,0.7012911473,0.3303392839,0.1843682136,0.4687159164,0.364156618,0.2346043189,0.437367914,0.8177607816,0.4482162186,0.4578335162);
return cl::sycl::atan(inputData_0);

});

test_function<121, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.667062206,0.5193294961);
return cl::sycl::atan(inputData_0);

});

test_function<122, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2033784283,0.828313918,0.4552994689);
return cl::sycl::atan(inputData_0);

});

test_function<123, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7314701914,0.411100104,0.7454768151,0.4116291328);
return cl::sycl::atan(inputData_0);

});

test_function<124, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2761276173,0.2569557335,0.8520277155,0.5692242069,0.139834612,0.4106780769,0.2872234084,0.1677256517);
return cl::sycl::atan(inputData_0);

});

test_function<125, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2494046948,0.145592384,0.6104589026,0.2386990919,0.588623901,0.5900053983,0.6639389686,0.5096949205,0.3275391923,0.8019659631,0.3824568654,0.46663546,0.6055035454,0.5128994385,0.8651746789,0.863774142);
return cl::sycl::atan(inputData_0);

});

test_function<126, float>(
[=](){
float inputData_0(0.8438078805);
return cl::sycl::atan2(inputData_0);

});

test_function<127, double>(
[=](){
double inputData_0(0.8472610797);
return cl::sycl::atan2(inputData_0);

});

test_function<128, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5647681085);
return cl::sycl::atan2(inputData_0);

});

test_function<129, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.492161651,0.6632934539);
return cl::sycl::atan2(inputData_0);

});

test_function<130, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2723356744,0.3126976314,0.1350458029);
return cl::sycl::atan2(inputData_0);

});

test_function<131, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.230286034,0.10309964,0.6237020612,0.2123255912);
return cl::sycl::atan2(inputData_0);

});

test_function<132, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7293434765,0.6444031967,0.8765406347,0.4172115896,0.8371135308,0.4629633379,0.3716029919,0.1818710959);
return cl::sycl::atan2(inputData_0);

});

test_function<133, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8062657481,0.7358321269,0.3583431812,0.4645955079,0.3601147727,0.1230632932,0.1354820203,0.3949633007,0.2676730625,0.5196116826,0.2502280285,0.2612972692,0.6381343051,0.6884821254,0.3497856767,0.7879955195);
return cl::sycl::atan2(inputData_0);

});

test_function<134, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3037113397,0.375152301);
return cl::sycl::atan2(inputData_0);

});

test_function<135, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6699843123,0.1356023211,0.8473467681);
return cl::sycl::atan2(inputData_0);

});

test_function<136, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1578701854,0.4687448472,0.6796838608,0.137974828);
return cl::sycl::atan2(inputData_0);

});

test_function<137, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7472021485,0.8831146746,0.4684093382,0.194498909,0.1651815965,0.1789843489,0.7123530993,0.4312102788);
return cl::sycl::atan2(inputData_0);

});

test_function<138, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8353873266,0.4525118209,0.1617146481,0.44154847,0.7038623147,0.7634707415,0.1314813492,0.244311513,0.4920107616,0.2024683824,0.7968741136,0.8475687108,0.3556775987,0.447874946,0.5456432515,0.3284046329);
return cl::sycl::atan2(inputData_0);

});

test_function<139, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.532860558,0.2609480364);
return cl::sycl::atan2(inputData_0);

});

test_function<140, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.337313001,0.4534269066,0.5837359218);
return cl::sycl::atan2(inputData_0);

});

test_function<141, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5289320209,0.3087903814,0.2854303003,0.1949841894);
return cl::sycl::atan2(inputData_0);

});

test_function<142, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7267949087,0.1791206132,0.6863080049,0.2990189565,0.3276455872,0.6888667464,0.6276966334,0.6935372444);
return cl::sycl::atan2(inputData_0);

});

test_function<143, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.512226447,0.7872766557,0.1974351131,0.6161575691,0.19459545,0.6898266945,0.3871237292,0.6399056835,0.6627871308,0.6284867661,0.2772463843,0.7654399091,0.2921088699,0.5145226378,0.6397166033,0.2868825398);
return cl::sycl::atan2(inputData_0);

});

test_function<144, float>(
[=](){
float inputData_0(0.6028093784);
return cl::sycl::atanh(inputData_0);

});

test_function<145, double>(
[=](){
double inputData_0(0.3294648384);
return cl::sycl::atanh(inputData_0);

});

test_function<146, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.2371059009);
return cl::sycl::atanh(inputData_0);

});

test_function<147, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7477990628,0.5424982161);
return cl::sycl::atanh(inputData_0);

});

test_function<148, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3623077653,0.5683447578,0.1202291179);
return cl::sycl::atanh(inputData_0);

});

test_function<149, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2038582854,0.4164646814,0.8806052636,0.5083796143);
return cl::sycl::atanh(inputData_0);

});

test_function<150, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1611649641,0.7120324922,0.7251550967,0.7198417395,0.5555984304,0.6565589903,0.270766349,0.6860484727);
return cl::sycl::atanh(inputData_0);

});

test_function<151, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7529391899,0.7079732322,0.3827699221,0.5728224405,0.603191486,0.8206478829,0.1864111162,0.7671470167,0.5211484468,0.3868912964,0.4644823212,0.1101083991,0.2760588739,0.6222107361,0.6286794238,0.4957591522);
return cl::sycl::atanh(inputData_0);

});

test_function<152, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8626607045,0.4847320708);
return cl::sycl::atanh(inputData_0);

});

test_function<153, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.3511549276,0.7782246714,0.3073266395);
return cl::sycl::atanh(inputData_0);

});

test_function<154, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5834447944,0.6627350819,0.757357039,0.7282950001);
return cl::sycl::atanh(inputData_0);

});

test_function<155, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4072738644,0.1473442448,0.1306302924,0.6811683103,0.8693531051,0.3745322994,0.4529560785,0.6806384126);
return cl::sycl::atanh(inputData_0);

});

test_function<156, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6262649967,0.3080852708,0.6372678766,0.3439219357,0.3850863252,0.5316106442,0.6858510591,0.2209729693,0.1175897687,0.6022639636,0.1196517422,0.1359705926,0.2806204614,0.6231014986,0.1532360781,0.1499246141);
return cl::sycl::atanh(inputData_0);

});

test_function<157, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8776745955,0.438122315);
return cl::sycl::atanh(inputData_0);

});

test_function<158, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8139431472,0.2732194272,0.4481705436);
return cl::sycl::atanh(inputData_0);

});

test_function<159, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3864281077,0.2415484288,0.3630505486,0.889436655);
return cl::sycl::atanh(inputData_0);

});

test_function<160, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6978472078,0.4061346233,0.4274275475,0.3109927209,0.5250693429,0.6885095297,0.6493172926,0.4701198683);
return cl::sycl::atanh(inputData_0);

});

test_function<161, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1335512374,0.8372062452,0.4271470425,0.4122390936,0.1024880916,0.2105817713,0.795082734,0.5111476769,0.6859478754,0.2185343091,0.3640408053,0.7721092452,0.7565268169,0.2974354145,0.1175802467,0.7451735788);
return cl::sycl::atanh(inputData_0);

});

test_function<162, float>(
[=](){
float inputData_0(0.235075204);
return cl::sycl::atanpi(inputData_0);

});

test_function<163, double>(
[=](){
double inputData_0(0.7301451137);
return cl::sycl::atanpi(inputData_0);

});

test_function<164, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6469273839);
return cl::sycl::atanpi(inputData_0);

});

test_function<165, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2346518082,0.1627909149);
return cl::sycl::atanpi(inputData_0);

});

test_function<166, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8421195439,0.5783027178,0.5964081384);
return cl::sycl::atanpi(inputData_0);

});

test_function<167, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4660094423,0.2200567819,0.5815759304,0.301978304);
return cl::sycl::atanpi(inputData_0);

});

test_function<168, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7447157248,0.6861751638,0.121813748,0.8459384077,0.1290528387,0.1716954551,0.3341876487,0.2206472484);
return cl::sycl::atanpi(inputData_0);

});

test_function<169, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2889160663,0.3846475909,0.6883997724,0.4237690886,0.3158718038,0.4938505229,0.4140745998,0.348611358,0.8204333263,0.5403587608,0.8818620088,0.7183299275,0.5563994381,0.3099572714,0.649474925,0.4647341752);
return cl::sycl::atanpi(inputData_0);

});

test_function<170, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.677110172,0.4230230471);
return cl::sycl::atanpi(inputData_0);

});

test_function<171, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4968040291,0.116547014,0.6919668019);
return cl::sycl::atanpi(inputData_0);

});

test_function<172, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1274188355,0.6445803087,0.5656029564,0.7207340892);
return cl::sycl::atanpi(inputData_0);

});

test_function<173, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.3318220794,0.6488886521,0.2656783805,0.5234176011,0.3722243034,0.8827636411,0.8774932459,0.2671757884);
return cl::sycl::atanpi(inputData_0);

});

test_function<174, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5528305887,0.3635541487,0.8748305496,0.8396207585,0.5689166824,0.6760675641,0.6450598054,0.382684506,0.8330892555,0.8195628295,0.3645267716,0.6979159285,0.1072737013,0.7530872884,0.5518954763,0.8618453702);
return cl::sycl::atanpi(inputData_0);

});

test_function<175, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3905544596,0.6005704599);
return cl::sycl::atanpi(inputData_0);

});

test_function<176, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.3584019452,0.7262283051,0.5805623974);
return cl::sycl::atanpi(inputData_0);

});

test_function<177, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8899768184,0.1008102345,0.2126069937,0.1348811057);
return cl::sycl::atanpi(inputData_0);

});

test_function<178, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2006782791,0.8435082377,0.8588866396,0.4843300278,0.8573515157,0.7547100882,0.7228941873,0.6978255606);
return cl::sycl::atanpi(inputData_0);

});

test_function<179, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2501236681,0.5391018089,0.4391033845,0.8598304381,0.2390668305,0.2358870748,0.6270894029,0.2259214317,0.1880429384,0.5031385579,0.737328777,0.5840365373,0.7038031783,0.3126068253,0.3279702664,0.4429629134);
return cl::sycl::atanpi(inputData_0);

});

test_function<180, float>(
[=](){
float inputData_0(0.8926783074);
float inputData_1(0.6743346052);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<181, double>(
[=](){
double inputData_0(0.8570031658);
double inputData_1(0.5302963646);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<182, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5436478813);
cl::sycl::half inputData_1(0.8920722684);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<183, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2519906205,0.7260723497);
cl::sycl::float2 inputData_1(0.7332110592,0.7757933021);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<184, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7000421674,0.2242664146,0.6289021057);
cl::sycl::float3 inputData_1(0.8389625477,0.550628122,0.3887532083);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<185, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8596161189,0.5492789204,0.4293091156,0.5913067984);
cl::sycl::float4 inputData_1(0.7433000133,0.2826416725,0.1125536344,0.5232759177);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<186, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8530859361,0.6442063701,0.6047264001,0.602252118,0.4975917698,0.6847354158,0.299355552,0.8134034112);
cl::sycl::float8 inputData_1(0.3195781242,0.8559560106,0.841197368,0.1623396192,0.458543761,0.6952290279,0.4597232572,0.5071192199);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<187, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7454591501,0.6639937288,0.8664033782,0.2315887954,0.8388474289,0.842389002,0.6077991509,0.8523126618,0.3021484699,0.8054298267,0.7187834344,0.5877511998,0.1725033957,0.124107483,0.1087755964,0.300446446);
cl::sycl::float16 inputData_1(0.709881928,0.4093000259,0.7203573801,0.6005139927,0.4114095193,0.804117303,0.1307337915,0.4722503922,0.7638818715,0.2014507864,0.6683900492,0.3624926735,0.1194410229,0.4789799447,0.5173541912,0.1332690005);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<188, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.5527354828,0.3779470704);
cl::sycl::double2 inputData_1(0.1035945658,0.2526187045);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<189, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1886485378,0.5324975638,0.1344961304);
cl::sycl::double3 inputData_1(0.8425060561,0.7760495779,0.856238115);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<190, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3518408262,0.8242139108,0.887449938,0.7117851474);
cl::sycl::double4 inputData_1(0.3200660909,0.6367114433,0.576530523,0.4233626417);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<191, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.3448782832,0.1478785525,0.2003059798,0.2071649248,0.4847142917,0.6135147078,0.711254762,0.137371008);
cl::sycl::double8 inputData_1(0.7590078981,0.1347769786,0.543957464,0.6953182798,0.6049770974,0.8597429405,0.3757586825,0.5687066842);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<192, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1662392502,0.5478372703,0.7506390409,0.2612836111,0.3087716003,0.6603245122,0.3031055735,0.3073963792,0.8484122304,0.8988344247,0.2241587446,0.8201299098,0.5421811888,0.1308809139,0.5684021722,0.6132397205);
cl::sycl::double16 inputData_1(0.127036559,0.7061535377,0.7542401132,0.1573145937,0.6187199521,0.4652379847,0.2909770299,0.4669363053,0.2275117678,0.3669327254,0.6241658398,0.4811884449,0.5447360757,0.534754235,0.7564753921,0.3747062385);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<193, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7503696726,0.1639896697);
cl::sycl::half2 inputData_1(0.4421864367,0.3818560932);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<194, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.461264511,0.7668078564,0.5099195204);
cl::sycl::half3 inputData_1(0.889797317,0.7891685762,0.1950773963);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<195, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3535132284,0.1181804012,0.6870027371,0.1153606435);
cl::sycl::half4 inputData_1(0.8087508119,0.2546742919,0.4310690322,0.1496314448);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<196, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3490039098,0.4116119916,0.1417847788,0.7140405225,0.6690797756,0.3863068993,0.7681540425,0.1619374419);
cl::sycl::half8 inputData_1(0.1432051208,0.3839842355,0.8214730657,0.7051741615,0.6378541428,0.5501885882,0.743012431,0.4297813546);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<197, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1245508638,0.7419234292,0.2523947474,0.4101270799,0.3860874778,0.1986925007,0.3806274952,0.2416695023,0.5928110641,0.6227474862,0.1109172423,0.4651806819,0.5432421252,0.7973303955,0.4968250482,0.1643603495);
cl::sycl::half16 inputData_1(0.1413791033,0.7896872664,0.7325835275,0.7867583449,0.3097941341,0.6183978886,0.1765744378,0.7612584894,0.3668903491,0.8641177899,0.4771060524,0.1264541576,0.8272447021,0.6004257191,0.3296650288,0.129443115);
return cl::sycl::atan2pi(inputData_0,inputData_1);

});

test_function<198, float>(
[=](){
float inputData_0(0.4013491174);
return cl::sycl::cbrt(inputData_0);

});

test_function<199, double>(
[=](){
double inputData_0(0.2254884478);
return cl::sycl::cbrt(inputData_0);

});

test_function<200, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5386242491);
return cl::sycl::cbrt(inputData_0);

});

test_function<201, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2175069401,0.2396914231);
return cl::sycl::cbrt(inputData_0);

});

test_function<202, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8366955886,0.6120960276,0.2940651294);
return cl::sycl::cbrt(inputData_0);

});

test_function<203, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8031170245,0.5997726639,0.8564794721,0.4863334355);
return cl::sycl::cbrt(inputData_0);

});

test_function<204, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8103206672,0.6427550459,0.1353348447,0.292232392,0.3252611239,0.2360133467,0.2905495605,0.2808321188);
return cl::sycl::cbrt(inputData_0);

});

test_function<205, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8026749994,0.4703190348,0.8012094651,0.2103983068,0.5519347892,0.1107741425,0.8442411279,0.1045096915,0.4119261188,0.741268767,0.8999052474,0.1156077923,0.7592683871,0.5080703676,0.1305456164,0.7216954167);
return cl::sycl::cbrt(inputData_0);

});

test_function<206, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.1895219301,0.5891793498);
return cl::sycl::cbrt(inputData_0);

});

test_function<207, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7226601729,0.6388727414,0.4038994598);
return cl::sycl::cbrt(inputData_0);

});

test_function<208, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1211533095,0.4490111742,0.8309555869,0.3663386924);
return cl::sycl::cbrt(inputData_0);

});

test_function<209, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2983669753,0.2102646671,0.508201964,0.5266786182,0.1584385939,0.4262067888,0.6269451639,0.8728405481);
return cl::sycl::cbrt(inputData_0);

});

test_function<210, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4452328974,0.4488282695,0.4769071781,0.280026793,0.4158701136,0.6162117808,0.4176473672,0.5651005987,0.7684658304,0.8983740585,0.8080317469,0.3974373015,0.1173817075,0.5892836796,0.4796405666,0.2896136896);
return cl::sycl::cbrt(inputData_0);

});

test_function<211, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1322432793,0.3572561912);
return cl::sycl::cbrt(inputData_0);

});

test_function<212, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7384570503,0.8712952055,0.1853281113);
return cl::sycl::cbrt(inputData_0);

});

test_function<213, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8021115294,0.138974137,0.6707806551,0.1214365707);
return cl::sycl::cbrt(inputData_0);

});

test_function<214, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4368397458,0.7961846708,0.4144865181,0.8396514541,0.6705561129,0.5833474246,0.2291032384,0.3723966269);
return cl::sycl::cbrt(inputData_0);

});

test_function<215, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4288769314,0.5721638913,0.8968305282,0.3269677982,0.5028503127,0.8467583261,0.376336635,0.6028838298,0.712905231,0.60421578,0.7027445439,0.2565544002,0.8658701495,0.2415182455,0.5669449408,0.3368340873);
return cl::sycl::cbrt(inputData_0);

});

test_function<216, float>(
[=](){
float inputData_0(0.6075384202);
return cl::sycl::ceil(inputData_0);

});

test_function<217, double>(
[=](){
double inputData_0(0.3328883323);
return cl::sycl::ceil(inputData_0);

});

test_function<218, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4449706855);
return cl::sycl::ceil(inputData_0);

});

test_function<219, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6457780386,0.3152550004);
return cl::sycl::ceil(inputData_0);

});

test_function<220, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.682300706,0.3775021382,0.2057248778);
return cl::sycl::ceil(inputData_0);

});

test_function<221, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5905029735,0.2326064231,0.4444619571,0.4187179295);
return cl::sycl::ceil(inputData_0);

});

test_function<222, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1609350779,0.6686158699,0.6446588521,0.722236004,0.5359305127,0.5431334206,0.2353864023,0.2659711192);
return cl::sycl::ceil(inputData_0);

});

test_function<223, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2825995924,0.520242823,0.755186066,0.3855792934,0.8054975904,0.6887026148,0.6731577146,0.3681377034,0.1947819936,0.8702323849,0.7836885085,0.4270943926,0.7905745522,0.819373692,0.3739788987,0.501249194);
return cl::sycl::ceil(inputData_0);

});

test_function<224, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3654318722,0.6561260113);
return cl::sycl::ceil(inputData_0);

});

test_function<225, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8297338508,0.8876352831,0.6950232599);
return cl::sycl::ceil(inputData_0);

});

test_function<226, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3441938831,0.8043946321,0.8940957032,0.377220931);
return cl::sycl::ceil(inputData_0);

});

test_function<227, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.858969882,0.5092371244,0.8717083538,0.8966847921,0.7503536767,0.6467496394,0.2232115754,0.1039338266);
return cl::sycl::ceil(inputData_0);

});

test_function<228, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5763766803,0.6635679244,0.8484304361,0.5136959202,0.6574772822,0.6178847772,0.2639361,0.6154400742,0.8853769691,0.1889479653,0.6508345946,0.591444094,0.400683779,0.7346782031,0.1083886869,0.8139292977);
return cl::sycl::ceil(inputData_0);

});

test_function<229, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7538911624,0.4845638652);
return cl::sycl::ceil(inputData_0);

});

test_function<230, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1865113239,0.4621028445,0.5674023193);
return cl::sycl::ceil(inputData_0);

});

test_function<231, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3031067828,0.4892251719,0.7205830111,0.8381854365);
return cl::sycl::ceil(inputData_0);

});

test_function<232, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5493160221,0.761793428,0.1623465704,0.7850944371,0.8366516524,0.234401101,0.7619898894,0.7796529363);
return cl::sycl::ceil(inputData_0);

});

test_function<233, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8029270947,0.5137116159,0.5866033951,0.2664665964,0.6665052395,0.4240138427,0.1169352686,0.2074136908,0.4105744253,0.8081438448,0.5519538346,0.8330056272,0.8435870755,0.1694359937,0.570572331,0.3676224305);
return cl::sycl::ceil(inputData_0);

});

test_function<234, float>(
[=](){
float inputData_0(0.5054360978);
float inputData_1(0.464419842);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<235, double>(
[=](){
double inputData_0(0.4839546181);
double inputData_1(0.1814446465);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<236, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.766528067);
cl::sycl::half inputData_1(0.4922239694);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<237, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.615990045,0.4781430022);
cl::sycl::float2 inputData_1(0.2448147013,0.532800468);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<238, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2276317912,0.7817434049,0.7652832205);
cl::sycl::float3 inputData_1(0.2149110225,0.1550751624,0.1547935336);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<239, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4145952199,0.8624331499,0.5449123328,0.3124212592);
cl::sycl::float4 inputData_1(0.2837190617,0.1886985558,0.2128569684,0.7494906133);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<240, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2109067718,0.7912492457,0.7583984593,0.2094470417,0.546979816,0.1056442144,0.7896289075,0.5466216964);
cl::sycl::float8 inputData_1(0.7042723155,0.4922762615,0.6523375994,0.8449912993,0.5476366481,0.7997643815,0.3744363539,0.1780260365);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<241, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1041157141,0.2813202232,0.7708694695,0.3491963701,0.2796929181,0.4965043387,0.857523298,0.5071827586,0.3726973301,0.1620014328,0.5589335467,0.2810055836,0.3939993022,0.4049298934,0.7065474698,0.2853030758);
cl::sycl::float16 inputData_1(0.8487137806,0.6939104544,0.4848956326,0.804379593,0.3873343842,0.4074718989,0.2034953048,0.7228448756,0.4209541223,0.5002024226,0.4767749323,0.6249454541,0.3991507449,0.8326890609,0.4455378072,0.3873711823);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<242, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4207024441,0.7130365772);
cl::sycl::double2 inputData_1(0.894445272,0.793211717);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<243, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4837819988,0.3330874748,0.4567896435);
cl::sycl::double3 inputData_1(0.3752124425,0.2948256418,0.2495527323);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<244, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.8647005876,0.4994441523,0.1879798989,0.4071252881);
cl::sycl::double4 inputData_1(0.4109735338,0.5108276216,0.8840330597,0.8813067973);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<245, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.5527152886,0.5944732202,0.6405032599,0.5017777461,0.4893424465,0.3516191513,0.6471373916,0.1735162227);
cl::sycl::double8 inputData_1(0.353716197,0.8127828476,0.2819025208,0.8740659024,0.8873357776,0.5603061305,0.1323487842,0.1747825579);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<246, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2602413102,0.3614492546,0.1904865693,0.7377686185,0.391323656,0.2869869587,0.1349550963,0.4061374875,0.1036053844,0.1931931604,0.583716408,0.8479563291,0.2594927375,0.6928489653,0.2581644168,0.1011961551);
cl::sycl::double16 inputData_1(0.8172304369,0.7768869902,0.1534229728,0.2417082305,0.2874407424,0.8426570917,0.4055432765,0.7459054053,0.4486508263,0.4049957334,0.7122784438,0.5926087973,0.3154541554,0.5662484786,0.66308228,0.7616624733);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<247, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6417432633,0.6125976571);
cl::sycl::half2 inputData_1(0.576721874,0.1736407593);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<248, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8561512476,0.6718735284,0.3182969035);
cl::sycl::half3 inputData_1(0.6538805553,0.5966539489,0.6270811566);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<249, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4031271768,0.5585406838,0.6280217845,0.2613248552);
cl::sycl::half4 inputData_1(0.5064097315,0.1962733242,0.1844243985,0.8288484602);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<250, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1996377796,0.8146135774,0.4758393596,0.4639220926,0.3718522556,0.4329741732,0.4017859046,0.5519863576);
cl::sycl::half8 inputData_1(0.3684746553,0.7575806908,0.2868494001,0.2987760982,0.4844412373,0.8480650271,0.1191325393,0.6787308925);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<251, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1048052702,0.4238881705,0.7113657998,0.4568632974,0.4435911431,0.3025734632,0.4800765106,0.2826075997,0.3268170319,0.6226348489,0.5795576449,0.8436364123,0.8750952651,0.5179041546,0.17004521,0.3399224754);
cl::sycl::half16 inputData_1(0.5142439165,0.6385303147,0.8569577995,0.2240859469,0.1293477615,0.7960285462,0.7441314945,0.7125986212,0.4748806142,0.6422245633,0.4291753798,0.2536413262,0.4127150121,0.7296372768,0.7414844976,0.8689075729);
return cl::sycl::copysign(inputData_0,inputData_1);

});

test_function<252, float>(
[=](){
float inputData_0(0.8101337001);
return cl::sycl::cos(inputData_0);

});

test_function<253, double>(
[=](){
double inputData_0(0.6456676294);
return cl::sycl::cos(inputData_0);

});

test_function<254, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5167296182);
return cl::sycl::cos(inputData_0);

});

test_function<255, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6791416188,0.2465628715);
return cl::sycl::cos(inputData_0);

});

test_function<256, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.838467625,0.6700612596,0.5755884444);
return cl::sycl::cos(inputData_0);

});

test_function<257, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4472333738,0.6068332713,0.5941429823,0.8190833012);
return cl::sycl::cos(inputData_0);

});

test_function<258, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.5565890487,0.2707017219,0.4531034889,0.2943748086,0.823960135,0.7748206515,0.5446552916,0.2571132541);
return cl::sycl::cos(inputData_0);

});

test_function<259, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1348336104,0.207335563,0.4545754246,0.6393632627,0.2791984836,0.6476162798,0.7895595072,0.7057928653,0.4404219802,0.6165823259,0.8906939464,0.8083292823,0.3705196052,0.6483576428,0.2305690251,0.5458947768);
return cl::sycl::cos(inputData_0);

});

test_function<260, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3852271719,0.4505170547);
return cl::sycl::cos(inputData_0);

});

test_function<261, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4511190831,0.6305863519,0.7767968058);
return cl::sycl::cos(inputData_0);

});

test_function<262, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4748574738,0.2172678305,0.7033234186,0.7013144085);
return cl::sycl::cos(inputData_0);

});

test_function<263, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8630761918,0.4152448055,0.4711032075,0.5324765933,0.8136987172,0.6633732143,0.1170224989,0.2658581501);
return cl::sycl::cos(inputData_0);

});

test_function<264, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.7831158782,0.5683790539,0.7991267728,0.42911931,0.2683743097,0.1033125815,0.8968407671,0.2091052266,0.6143750126,0.4917671282,0.4041195127,0.5297617201,0.1626268482,0.8760273619,0.494189553,0.1122316134);
return cl::sycl::cos(inputData_0);

});

test_function<265, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4354747451,0.7057615242);
return cl::sycl::cos(inputData_0);

});

test_function<266, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.3496669357,0.6960179271,0.7138901759);
return cl::sycl::cos(inputData_0);

});

test_function<267, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2912965752,0.874377989,0.1223109996,0.7908843872);
return cl::sycl::cos(inputData_0);

});

test_function<268, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5101192973,0.2227035908,0.3067143566,0.5748138335,0.3227657311,0.7707368611,0.2756228112,0.4072490379);
return cl::sycl::cos(inputData_0);

});

test_function<269, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5054505344,0.371818371,0.7593142425,0.3111057635,0.1711817386,0.2238281507,0.6015563642,0.5508501201,0.1506386575,0.8944393309,0.4835525062,0.355549761,0.6833299212,0.1194334872,0.4473993159,0.6315310713);
return cl::sycl::cos(inputData_0);

});

test_function<270, float>(
[=](){
float inputData_0(0.8697089799);
return cl::sycl::cosh(inputData_0);

});

test_function<271, double>(
[=](){
double inputData_0(0.7093102225);
return cl::sycl::cosh(inputData_0);

});

test_function<272, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8081273677);
return cl::sycl::cosh(inputData_0);

});

test_function<273, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1951247257,0.4438164845);
return cl::sycl::cosh(inputData_0);

});

test_function<274, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1254323401,0.3175953583,0.4074374921);
return cl::sycl::cosh(inputData_0);

});

test_function<275, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3750568646,0.3989926375,0.7424640038,0.2516349062);
return cl::sycl::cosh(inputData_0);

});

test_function<276, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7595964881,0.5335368696,0.3709961028,0.5417886139,0.2291386658,0.4963637997,0.1175626376,0.7903800622);
return cl::sycl::cosh(inputData_0);

});

test_function<277, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3652648278,0.3752343595,0.8961215979,0.5907645855,0.4341229555,0.732525377,0.1541317657,0.5564033634,0.5165607695,0.7889825344,0.568960274,0.4882179322,0.5161806872,0.7255178465,0.3778566366,0.5462315311);
return cl::sycl::cosh(inputData_0);

});

test_function<278, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6659122182,0.8964443635);
return cl::sycl::cosh(inputData_0);

});

test_function<279, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6549473564,0.869496937,0.4192261291);
return cl::sycl::cosh(inputData_0);

});

test_function<280, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5870247942,0.6962358859,0.3787327597,0.315339951);
return cl::sycl::cosh(inputData_0);

});

test_function<281, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8782664889,0.3788271783,0.8999221417,0.7818167877,0.2728544919,0.7625753779,0.8869017013,0.3214561794);
return cl::sycl::cosh(inputData_0);

});

test_function<282, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.631563531,0.7156713784,0.166625599,0.7554654439,0.3466885857,0.6651054369,0.8601105769,0.1280872171,0.5893703044,0.3339237023,0.1917270313,0.6694838421,0.8832372499,0.5101684007,0.3770753674,0.4592716759);
return cl::sycl::cosh(inputData_0);

});

test_function<283, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4316943079,0.5255215277);
return cl::sycl::cosh(inputData_0);

});

test_function<284, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.4273406692,0.1642979717,0.8835421888);
return cl::sycl::cosh(inputData_0);

});

test_function<285, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8973658223,0.2393076438,0.292831973,0.4495650332);
return cl::sycl::cosh(inputData_0);

});

test_function<286, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6589863496,0.1250759484,0.7683980406,0.6107466987,0.3154348497,0.7966937719,0.6289673705,0.353539425);
return cl::sycl::cosh(inputData_0);

});

test_function<287, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5382767955,0.8833900469,0.1387463736,0.6667696133,0.7795308881,0.6538534791,0.2120147303,0.5777196828,0.7287641966,0.4348767297,0.5659426226,0.3027743635,0.3501988406,0.7468561145,0.4915985101,0.45904939);
return cl::sycl::cosh(inputData_0);

});

test_function<288, float>(
[=](){
float inputData_0(0.1983070697);
return cl::sycl::cospi(inputData_0);

});

test_function<289, double>(
[=](){
double inputData_0(0.3995767186);
return cl::sycl::cospi(inputData_0);

});

test_function<290, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5165768448);
return cl::sycl::cospi(inputData_0);

});

test_function<291, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2848098678,0.7463484495);
return cl::sycl::cospi(inputData_0);

});

test_function<292, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.406960565,0.2907918339,0.3466379946);
return cl::sycl::cospi(inputData_0);

});

test_function<293, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7595708262,0.8233147808,0.8682382789,0.112155264);
return cl::sycl::cospi(inputData_0);

});

test_function<294, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7031153045,0.5203872288,0.1996480616,0.2972268611,0.3253526644,0.4233733501,0.4765770179,0.8494310852);
return cl::sycl::cospi(inputData_0);

});

test_function<295, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1466840456,0.667335444,0.7832848759,0.3858399344,0.2993737587,0.2770467925,0.3406713414,0.2162383925,0.5413423895,0.3003196033,0.1218012284,0.2861067466,0.7565056844,0.4338961776,0.8068290037,0.8548925052);
return cl::sycl::cospi(inputData_0);

});

test_function<296, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2946786628,0.5479779609);
return cl::sycl::cospi(inputData_0);

});

test_function<297, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8048535277,0.5651362698,0.2344002291);
return cl::sycl::cospi(inputData_0);

});

test_function<298, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.298362598,0.8900998639,0.3395094919,0.7941623858);
return cl::sycl::cospi(inputData_0);

});

test_function<299, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7360098702,0.6935877622,0.6775538055,0.7319854909,0.7779261482,0.1498931492,0.2342478596,0.5044234597);
return cl::sycl::cospi(inputData_0);

});

test_function<300, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2699916207,0.5265744181,0.4945458716,0.2014171592,0.1687689217,0.1093222374,0.760028904,0.1653933347,0.869252263,0.8870654836,0.6965571546,0.4603066246,0.3206307708,0.4299619375,0.3762346128,0.4170361117);
return cl::sycl::cospi(inputData_0);

});

test_function<301, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6809566264,0.8140209661);
return cl::sycl::cospi(inputData_0);

});

test_function<302, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2261720215,0.2941364631,0.2679175249);
return cl::sycl::cospi(inputData_0);

});

test_function<303, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1362767921,0.7833604637,0.5090204347,0.153627823);
return cl::sycl::cospi(inputData_0);

});

test_function<304, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4570042089,0.4604886336,0.7223647648,0.7091179589,0.2075911915,0.6015004926,0.5077490347,0.1107934213);
return cl::sycl::cospi(inputData_0);

});

test_function<305, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2181886634,0.633478739,0.393620645,0.8709483164,0.5014002898,0.6506265357,0.2068941118,0.4835587669,0.6872971769,0.7667853406,0.2596859432,0.4175253769,0.4788216072,0.4522979149,0.4803533364,0.3367201801);
return cl::sycl::cospi(inputData_0);

});

test_function<306, float>(
[=](){
float inputData_0(0.7469765943);
return cl::sycl::erfc(inputData_0);

});

test_function<307, double>(
[=](){
double inputData_0(0.8304623564);
return cl::sycl::erfc(inputData_0);

});

test_function<308, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.3792005828);
return cl::sycl::erfc(inputData_0);

});

test_function<309, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6102880573,0.4045647347);
return cl::sycl::erfc(inputData_0);

});

test_function<310, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5630015934,0.6564311558,0.5012130219);
return cl::sycl::erfc(inputData_0);

});

test_function<311, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6396656562,0.7057165954,0.7746365084,0.2510479118);
return cl::sycl::erfc(inputData_0);

});

test_function<312, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2731079177,0.5114970813,0.5077256151,0.7461803949,0.513906698,0.8200419756,0.7220822687,0.505052649);
return cl::sycl::erfc(inputData_0);

});

test_function<313, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7610608974,0.4806847687,0.3733721272,0.4467387515,0.4649664098,0.6204255612,0.141725676,0.6836069417,0.8745862201,0.4670540553,0.155017835,0.2610051071,0.1825712315,0.305082929,0.7351256844,0.1008395483);
return cl::sycl::erfc(inputData_0);

});

test_function<314, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7988634659,0.8516377204);
return cl::sycl::erfc(inputData_0);

});

test_function<315, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2480024673,0.2388691479,0.8726103703);
return cl::sycl::erfc(inputData_0);

});

test_function<316, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3883041239,0.7494210438,0.1072085421,0.8926332632);
return cl::sycl::erfc(inputData_0);

});

test_function<317, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.113192165,0.5860564763,0.8427602838,0.7650086872,0.3483220456,0.7576643302,0.4144381095,0.4998451337);
return cl::sycl::erfc(inputData_0);

});

test_function<318, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3906224004,0.3837744539,0.5656601812,0.7256543882,0.6595929028,0.7144623647,0.1114186144,0.5253547148,0.3822308823,0.2668918662,0.8366814577,0.25744644,0.2475799727,0.2430512756,0.6264759252,0.5893919444);
return cl::sycl::erfc(inputData_0);

});

test_function<319, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.5045067752,0.5694414057);
return cl::sycl::erfc(inputData_0);

});

test_function<320, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8524610514,0.7889680697,0.8247476161);
return cl::sycl::erfc(inputData_0);

});

test_function<321, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1433262672,0.8179302192,0.1250805178,0.6181143561);
return cl::sycl::erfc(inputData_0);

});

test_function<322, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8446657427,0.5019707387,0.4354565341,0.3653943636,0.8328988947,0.8407754113,0.5953004073,0.6715431873);
return cl::sycl::erfc(inputData_0);

});

test_function<323, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3713016329,0.2105398137,0.8832007824,0.6256176933,0.3195100396,0.881666842,0.5871759469,0.3644680695,0.8166504004,0.1623203424,0.7433229313,0.2276632124,0.1861395568,0.3071552548,0.671860435,0.5864066389);
return cl::sycl::erfc(inputData_0);

});

test_function<324, float>(
[=](){
float inputData_0(0.4370227019);
return cl::sycl::erf(inputData_0);

});

test_function<325, double>(
[=](){
double inputData_0(0.227236954);
return cl::sycl::erf(inputData_0);

});

test_function<326, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8390014912);
return cl::sycl::erf(inputData_0);

});

test_function<327, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7130281396,0.6490133027);
return cl::sycl::erf(inputData_0);

});

test_function<328, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.750327253,0.7193962085,0.189940009);
return cl::sycl::erf(inputData_0);

});

test_function<329, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7186833746,0.7709803473,0.6974046884,0.4858218211);
return cl::sycl::erf(inputData_0);

});

test_function<330, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6491565152,0.1800060909,0.7114762868,0.3097822146,0.7281011171,0.608227267,0.5072627653,0.5288137245);
return cl::sycl::erf(inputData_0);

});

test_function<331, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1597846449,0.1327183283,0.1118598519,0.720433974,0.2107962542,0.1982932294,0.4080502318,0.8821623771,0.8087394358,0.3506312229,0.7558387586,0.1680622202,0.4136322716,0.5633646294,0.8889975323,0.1389655662);
return cl::sycl::erf(inputData_0);

});

test_function<332, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4299368138,0.835688352);
return cl::sycl::erf(inputData_0);

});

test_function<333, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1220885458,0.5792659031,0.4195149966);
return cl::sycl::erf(inputData_0);

});

test_function<334, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5482147737,0.662679531,0.4253563309,0.8136417924);
return cl::sycl::erf(inputData_0);

});

test_function<335, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.864571618,0.888047305,0.1438227996,0.7694635656,0.8026980075,0.2163242075,0.8531435961,0.201691083);
return cl::sycl::erf(inputData_0);

});

test_function<336, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2658512627,0.8644373725,0.7646265274,0.56124095,0.3302466647,0.3021723879,0.4225108247,0.1071947984,0.6090873472,0.1412400873,0.7190817104,0.1561309151,0.1082035043,0.3277256021,0.7181536325,0.7661096201);
return cl::sycl::erf(inputData_0);

});

test_function<337, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.5058422097,0.8506378352);
return cl::sycl::erf(inputData_0);

});

test_function<338, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1916136202,0.3657755256,0.6922762324);
return cl::sycl::erf(inputData_0);

});

test_function<339, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3586559346,0.2162885984,0.5626886115,0.1500591939);
return cl::sycl::erf(inputData_0);

});

test_function<340, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3984134567,0.3030949917,0.3654498916,0.4881457774,0.5285619588,0.1674859648,0.3524224904,0.4069573926);
return cl::sycl::erf(inputData_0);

});

test_function<341, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4226435217,0.4840282831,0.4406976194,0.1592220658,0.2754596659,0.6154288256,0.7630112104,0.5091524918,0.2185518138,0.156373603,0.2247153582,0.4072377184,0.5521651335,0.6314057891,0.5194874541,0.5522325295);
return cl::sycl::erf(inputData_0);

});

test_function<342, float>(
[=](){
float inputData_0(0.3817867275);
return cl::sycl::exp(inputData_0);

});

test_function<343, double>(
[=](){
double inputData_0(0.6324775271);
return cl::sycl::exp(inputData_0);

});

test_function<344, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6814881949);
return cl::sycl::exp(inputData_0);

});

test_function<345, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4217003202,0.7519299824);
return cl::sycl::exp(inputData_0);

});

test_function<346, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6951456229,0.8237951912,0.4733986468);
return cl::sycl::exp(inputData_0);

});

test_function<347, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3761775448,0.7217629343,0.1300747438,0.4071010279);
return cl::sycl::exp(inputData_0);

});

test_function<348, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8817545674,0.3738111066,0.5098579109,0.2998156284,0.1615575819,0.1887613974,0.4481680618,0.5952018514);
return cl::sycl::exp(inputData_0);

});

test_function<349, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5365987243,0.5149457723,0.1893207489,0.1322074268,0.3869569542,0.8539879105,0.2438540419,0.3154091089,0.4865127728,0.8313657634,0.8575794901,0.1010428596,0.6183164821,0.288926498,0.6235853581,0.6942773087);
return cl::sycl::exp(inputData_0);

});

test_function<350, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.809711512,0.6467339862);
return cl::sycl::exp(inputData_0);

});

test_function<351, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7777676751,0.7275839312,0.2285729106);
return cl::sycl::exp(inputData_0);

});

test_function<352, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1349542127,0.6910229116,0.5207351187,0.898292404);
return cl::sycl::exp(inputData_0);

});

test_function<353, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2319124331,0.4082159501,0.3302268606,0.8029578542,0.4869566096,0.8309196653,0.6657520023,0.8990449346);
return cl::sycl::exp(inputData_0);

});

test_function<354, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5798232128,0.8809273223,0.2387250698,0.4533440489,0.5627129634,0.8826367285,0.5543038593,0.7922119848,0.6028044471,0.5099208202,0.4131531074,0.3949072384,0.3361744299,0.2690959767,0.8700616195,0.5291689729);
return cl::sycl::exp(inputData_0);

});

test_function<355, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7926956292,0.8079710088);
return cl::sycl::exp(inputData_0);

});

test_function<356, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8538398889,0.2905334449,0.370183114);
return cl::sycl::exp(inputData_0);

});

test_function<357, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6065119037,0.3576432362,0.2151424756,0.7078873709);
return cl::sycl::exp(inputData_0);

});

test_function<358, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5403135191,0.5292184156,0.6683768308,0.1917922014,0.8375272062,0.4838584674,0.6534631144,0.5802594087);
return cl::sycl::exp(inputData_0);

});

test_function<359, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5841863409,0.6679293442,0.1710365149,0.4973865742,0.2682315367,0.4117470817,0.5093966317,0.3831589093,0.4253703255,0.6846958579,0.1346618127,0.8652748716,0.5831340184,0.23084371,0.5457750109,0.1647484496);
return cl::sycl::exp(inputData_0);

});

test_function<360, float>(
[=](){
float inputData_0(0.5011789637);
return cl::sycl::exp2(inputData_0);

});

test_function<361, double>(
[=](){
double inputData_0(0.6509051412);
return cl::sycl::exp2(inputData_0);

});

test_function<362, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4358223487);
return cl::sycl::exp2(inputData_0);

});

test_function<363, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3513450554,0.6389712855);
return cl::sycl::exp2(inputData_0);

});

test_function<364, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8482281488,0.7989086729,0.4082945708);
return cl::sycl::exp2(inputData_0);

});

test_function<365, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7907217915,0.1920273298,0.1469764789,0.8865495751);
return cl::sycl::exp2(inputData_0);

});

test_function<366, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7103269772,0.5919907434,0.5470246043,0.3471031579,0.8192657317,0.7822003598,0.4853720032,0.2763062414);
return cl::sycl::exp2(inputData_0);

});

test_function<367, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6411161428,0.6809513931,0.8964050113,0.7323025324,0.1729974896,0.8919526757,0.7836102525,0.5660191507,0.3647449193,0.6863421608,0.5772970438,0.176612466,0.5508101,0.1161575439,0.7309573179,0.7588686045);
return cl::sycl::exp2(inputData_0);

});

test_function<368, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6840924043,0.1731827308);
return cl::sycl::exp2(inputData_0);

});

test_function<369, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5704749755,0.4131164533,0.2027144418);
return cl::sycl::exp2(inputData_0);

});

test_function<370, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.8136723553,0.8538352962,0.837826953,0.5249742625);
return cl::sycl::exp2(inputData_0);

});

test_function<371, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7915427916,0.2582689719,0.3363871524,0.8269689929,0.5722991762,0.2808126458,0.2040779583,0.2826495365);
return cl::sycl::exp2(inputData_0);

});

test_function<372, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4968055057,0.342791278,0.6875373937,0.3170488072,0.162750084,0.8185701182,0.6310714448,0.8792514321,0.2455976677,0.7963100468,0.1137564382,0.5302577058,0.4837939482,0.200957112,0.7520949706,0.3166815465);
return cl::sycl::exp2(inputData_0);

});

test_function<373, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8187473221,0.6595307566);
return cl::sycl::exp2(inputData_0);

});

test_function<374, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7818884414,0.7931847708,0.7334263837);
return cl::sycl::exp2(inputData_0);

});

test_function<375, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6891906811,0.1032564157,0.2147440065,0.2657056316);
return cl::sycl::exp2(inputData_0);

});

test_function<376, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5619859318,0.1026996935,0.2017252215,0.4879733654,0.1328514189,0.3549971101,0.2759906749,0.2395256427);
return cl::sycl::exp2(inputData_0);

});

test_function<377, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3532845179,0.8049734142,0.2851541277,0.6193270602,0.6871985452,0.640224467,0.2509963697,0.3798405796,0.3177256307,0.5309170639,0.8752281265,0.2742980616,0.5418965542,0.1523813079,0.4004537351,0.8633383607);
return cl::sycl::exp2(inputData_0);

});

test_function<378, float>(
[=](){
float inputData_0(0.826781849);
return cl::sycl::exp10(inputData_0);

});

test_function<379, double>(
[=](){
double inputData_0(0.1761649103);
return cl::sycl::exp10(inputData_0);

});

test_function<380, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7820207008);
return cl::sycl::exp10(inputData_0);

});

test_function<381, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6725651137,0.8343387658);
return cl::sycl::exp10(inputData_0);

});

test_function<382, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4686523685,0.4380440072,0.816922377);
return cl::sycl::exp10(inputData_0);

});

test_function<383, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5289258492,0.7090939986,0.2421136841,0.1547466711);
return cl::sycl::exp10(inputData_0);

});

test_function<384, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4523950635,0.3616412245,0.5094987634,0.3754586687,0.7900704109,0.6884839518,0.4071654889,0.2006931768);
return cl::sycl::exp10(inputData_0);

});

test_function<385, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6678756578,0.5328684424,0.2215562263,0.1278278011,0.5933601465,0.512998359,0.5603642225,0.4326889706,0.4749694717,0.4131409704,0.1701802796,0.5282440984,0.1972833842,0.6387908963,0.6995345393,0.2343432309);
return cl::sycl::exp10(inputData_0);

});

test_function<386, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2612093582,0.2925688988);
return cl::sycl::exp10(inputData_0);

});

test_function<387, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5788080953,0.4251940672,0.8100242426);
return cl::sycl::exp10(inputData_0);

});

test_function<388, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5383737338,0.5205090177,0.274795112,0.1726858031);
return cl::sycl::exp10(inputData_0);

});

test_function<389, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8397679917,0.1797111899,0.2041800177,0.2562466893,0.5614610941,0.6112267531,0.4453092015,0.4159180693);
return cl::sycl::exp10(inputData_0);

});

test_function<390, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6128218288,0.3096755604,0.7403603582,0.6120397844,0.5822900998,0.1231012842,0.3762994348,0.7150878357,0.2647261125,0.6155953409,0.8797757757,0.4524323834,0.5143578093,0.269632106,0.1056134113,0.289280906);
return cl::sycl::exp10(inputData_0);

});

test_function<391, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4774719721,0.5834170954);
return cl::sycl::exp10(inputData_0);

});

test_function<392, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7687435427,0.3320150816,0.3632355555);
return cl::sycl::exp10(inputData_0);

});

test_function<393, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6765189107,0.6314250105,0.6716179985,0.8018278102);
return cl::sycl::exp10(inputData_0);

});

test_function<394, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.170945046,0.1995891024,0.4970740221,0.5900836743,0.6232572144,0.2840779952,0.2088767851,0.8371486895);
return cl::sycl::exp10(inputData_0);

});

test_function<395, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2920566848,0.1142625897,0.3263300308,0.5137599399,0.6066832207,0.691367219,0.2164710111,0.5063009281,0.3561553161,0.679746813,0.3875953847,0.7489146938,0.2532957029,0.8957616049,0.5171517056,0.4390761057);
return cl::sycl::exp10(inputData_0);

});

test_function<396, float>(
[=](){
float inputData_0(0.6805257662);
return cl::sycl::expm1(inputData_0);

});

test_function<397, double>(
[=](){
double inputData_0(0.4030762636);
return cl::sycl::expm1(inputData_0);

});

test_function<398, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1283830472);
return cl::sycl::expm1(inputData_0);

});

test_function<399, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4527011399,0.3302353062);
return cl::sycl::expm1(inputData_0);

});

test_function<400, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6289693126,0.5213716578,0.7640784955);
return cl::sycl::expm1(inputData_0);

});

test_function<401, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.491423299,0.224276713,0.218883466,0.5580985274);
return cl::sycl::expm1(inputData_0);

});

test_function<402, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.3118991896,0.269279427,0.8534261713,0.2114370074,0.832856392,0.5289820279,0.8495265554,0.7712264659);
return cl::sycl::expm1(inputData_0);

});

test_function<403, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3391026363,0.476157317,0.1682264011,0.3933370808,0.8413818246,0.1807997435,0.2965097935,0.1341811361,0.7886626107,0.6468700695,0.5714761739,0.4725222637,0.3080348655,0.5686748282,0.6621724639,0.7343997203);
return cl::sycl::expm1(inputData_0);

});

test_function<404, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2299793496,0.6002428658);
return cl::sycl::expm1(inputData_0);

});

test_function<405, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6430606355,0.5649390039,0.682203369);
return cl::sycl::expm1(inputData_0);

});

test_function<406, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5142269084,0.8637120515,0.6201482948,0.6024381414);
return cl::sycl::expm1(inputData_0);

});

test_function<407, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1105262114,0.2148334307,0.5808572273,0.7135293238,0.2154064945,0.609462863,0.2234771004,0.7105632353);
return cl::sycl::expm1(inputData_0);

});

test_function<408, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.7569641804,0.5964911889,0.1543311185,0.3235507494,0.3156250335,0.4742668305,0.7240521089,0.5626542653,0.8935648921,0.6665977184,0.2130872537,0.8833033171,0.1468902251,0.3662857607,0.6098010207,0.4123892421);
return cl::sycl::expm1(inputData_0);

});

test_function<409, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1177379965,0.3371876696);
return cl::sycl::expm1(inputData_0);

});

test_function<410, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2934933496,0.7209192281,0.5740431471);
return cl::sycl::expm1(inputData_0);

});

test_function<411, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2151967183,0.7980845781,0.270398846,0.3557426809);
return cl::sycl::expm1(inputData_0);

});

test_function<412, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7997787178,0.7121730879,0.4366648708,0.5154173943,0.8833352156,0.6686450859,0.6725833212,0.6246257397);
return cl::sycl::expm1(inputData_0);

});

test_function<413, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8905907773,0.8392071438,0.3385914927,0.456126366,0.6085488932,0.289520334,0.6178271607,0.8226182865,0.345020551,0.3943380305,0.4599517109,0.409054148,0.6146579352,0.14157537,0.7213888632,0.3282738299);
return cl::sycl::expm1(inputData_0);

});

test_function<414, float>(
[=](){
float inputData_0(0.5976048694);
return cl::sycl::fabs(inputData_0);

});

test_function<415, double>(
[=](){
double inputData_0(0.4390946357);
return cl::sycl::fabs(inputData_0);

});

test_function<416, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5890312722);
return cl::sycl::fabs(inputData_0);

});

test_function<417, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5551589593,0.5140462768);
return cl::sycl::fabs(inputData_0);

});

test_function<418, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2282585254,0.1060181991,0.185581409);
return cl::sycl::fabs(inputData_0);

});

test_function<419, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4079147143,0.3056779873,0.4878376179,0.4766650931);
return cl::sycl::fabs(inputData_0);

});

test_function<420, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.5123106683,0.2061677611,0.4979270236,0.8603894405,0.2375898939,0.1124346924,0.3711106912,0.6667994959);
return cl::sycl::fabs(inputData_0);

});

test_function<421, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7886006642,0.187414358,0.1249622073,0.3481093962,0.5976275785,0.8363254797,0.3643388296,0.7234898384,0.2020133553,0.6127780445,0.2999708137,0.7089426203,0.8297006933,0.4531252991,0.6497746816,0.3832300867);
return cl::sycl::fabs(inputData_0);

});

test_function<422, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7792580452,0.4281069956);
return cl::sycl::fabs(inputData_0);

});

test_function<423, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5672622925,0.8891635781,0.5461097527);
return cl::sycl::fabs(inputData_0);

});

test_function<424, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4622641988,0.1768481971,0.8596966434,0.5199197701);
return cl::sycl::fabs(inputData_0);

});

test_function<425, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6606330537,0.6236657469,0.2899548375,0.6096565409,0.1773030366,0.1458307247,0.7727306562,0.5806256364);
return cl::sycl::fabs(inputData_0);

});

test_function<426, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3410980599,0.521471674,0.5463219861,0.6422226151,0.1001074253,0.2155586958,0.1745358666,0.7024688383,0.4618677726,0.2587172821,0.3995888465,0.635563675,0.4689792589,0.536687238,0.8500266162,0.4202585579);
return cl::sycl::fabs(inputData_0);

});

test_function<427, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1826031387,0.1857107599);
return cl::sycl::fabs(inputData_0);

});

test_function<428, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6797738862,0.3499756607,0.1920304447);
return cl::sycl::fabs(inputData_0);

});

test_function<429, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7225449282,0.8110556548,0.1818854722,0.5932110684);
return cl::sycl::fabs(inputData_0);

});

test_function<430, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6923906866,0.2966677184,0.7693351037,0.6472099314,0.455655838,0.2325724987,0.3062662298,0.7635745269);
return cl::sycl::fabs(inputData_0);

});

test_function<431, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2342462625,0.6637066771,0.5567540647,0.5485743962,0.1130653379,0.1981610304,0.3478436535,0.6042633937,0.4051483346,0.3051934822,0.4070726587,0.4704926052,0.5756665498,0.547797965,0.3944144141,0.4403698184);
return cl::sycl::fabs(inputData_0);

});

test_function<432, float>(
[=](){
float inputData_0(0.7449913979);
float inputData_1(0.5712122585);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<433, double>(
[=](){
double inputData_0(0.8763877567);
double inputData_1(0.5819789722);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<434, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.3266325018);
cl::sycl::half inputData_1(0.5104924638);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<435, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4783214666,0.7823408709);
cl::sycl::float2 inputData_1(0.6899996368,0.8044702157);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<436, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6742634827,0.3026206335,0.3182522403);
cl::sycl::float3 inputData_1(0.2310467254,0.8032336129,0.8200662462);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<437, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3588515082,0.118578913,0.4794140555,0.7307035621);
cl::sycl::float4 inputData_1(0.6623652296,0.6404829755,0.1168118953,0.181532499);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<438, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6833727271,0.7548409158,0.2457740259,0.7531041288,0.8610243303,0.5813077648,0.5413761326,0.1263378828);
cl::sycl::float8 inputData_1(0.4311544233,0.4735963153,0.862763267,0.4526889525,0.1095287752,0.5536800874,0.1544967144,0.8936156758);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<439, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6277539974,0.6752715619,0.6547897498,0.8527968976,0.4239578462,0.3231524361,0.1629717733,0.1182568285,0.4817404075,0.694961383,0.6913929577,0.1022999537,0.5933279949,0.7654257403,0.7935597711,0.7156998801);
cl::sycl::float16 inputData_1(0.4351045993,0.6631661088,0.6612268107,0.1513627528,0.1311107223,0.3769060873,0.6146845902,0.4047730433,0.6159917754,0.709210197,0.717102076,0.32654666,0.8773897374,0.543035686,0.6023228671,0.6081668335);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<440, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6386769443,0.2232063436);
cl::sycl::double2 inputData_1(0.6396638443,0.4452299769);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<441, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8749517345,0.6713201399,0.8791921073);
cl::sycl::double3 inputData_1(0.8928481267,0.7671999989,0.568954484);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<442, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5805407661,0.4757481786,0.3957396466,0.4349925746);
cl::sycl::double4 inputData_1(0.8307941889,0.6170438693,0.2359417817,0.1298085178);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<443, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.45132094,0.4526930112,0.1526597779,0.2814057111,0.3651204183,0.4013863088,0.5999229869,0.2247929286);
cl::sycl::double8 inputData_1(0.7574296528,0.4976890378,0.155300294,0.1796694587,0.8545388675,0.125588174,0.6191865205,0.2431896628);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<444, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6233641715,0.8900749261,0.8345632653,0.4498051743,0.4452028106,0.331551882,0.4523312051,0.8663389655,0.1302877316,0.4830683002,0.8165806824,0.1852888981,0.1933765559,0.7518166884,0.3262609764,0.7383488494);
cl::sycl::double16 inputData_1(0.3414457344,0.1258452033,0.7558672444,0.3654556499,0.4690444533,0.1478813515,0.6206663711,0.7631622988,0.2758047755,0.8465650764,0.6068760592,0.538557988,0.2646808528,0.8008547429,0.4391592386,0.1341293621);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<445, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8406227594,0.3950779856);
cl::sycl::half2 inputData_1(0.335671993,0.1464001715);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<446, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1264766134,0.1344830761,0.7999686753);
cl::sycl::half3 inputData_1(0.7943041156,0.4757855752,0.1649979805);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<447, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2114023746,0.8580181226,0.5485689529,0.7364733583);
cl::sycl::half4 inputData_1(0.1614655895,0.1522552705,0.7217012266,0.1926470471);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<448, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7565611607,0.8475594609,0.4462648516,0.1921046881,0.6740977815,0.4565489351,0.5011290176,0.8103892992);
cl::sycl::half8 inputData_1(0.5318850643,0.2105573467,0.398009084,0.7797291781,0.5091566809,0.1645975508,0.5052347777,0.1280760809);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<449, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8034454557,0.2984918779,0.6833424735,0.8957143508,0.8170918129,0.5229033167,0.359317771,0.6589981352,0.5372890555,0.8270417709,0.2937402342,0.6547641164,0.2219883061,0.4240737429,0.721912515,0.3914799555);
cl::sycl::half16 inputData_1(0.3268994388,0.3823218316,0.3989232541,0.1063817024,0.214498432,0.692460994,0.6542298031,0.6219859117,0.209482808,0.1926656474,0.4916416272,0.4871233503,0.200639637,0.8726583318,0.7538753816,0.7977735522);
return cl::sycl::fdim(inputData_0,inputData_1);

});

test_function<450, float>(
[=](){
float inputData_0(0.2377536113);
return cl::sycl::floor(inputData_0);

});

test_function<451, double>(
[=](){
double inputData_0(0.6252291104);
return cl::sycl::floor(inputData_0);

});

test_function<452, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7505846545);
return cl::sycl::floor(inputData_0);

});

test_function<453, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3588118625,0.8882774152);
return cl::sycl::floor(inputData_0);

});

test_function<454, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1760597915,0.7345832318,0.2156889715);
return cl::sycl::floor(inputData_0);

});

test_function<455, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3010506768,0.251614783,0.1925917364,0.636714099);
return cl::sycl::floor(inputData_0);

});

test_function<456, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2943673512,0.7394023681,0.7155558294,0.8659372585,0.7197534438,0.8920246624,0.5458606384,0.1255924084);
return cl::sycl::floor(inputData_0);

});

test_function<457, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3656067683,0.415263957,0.8730525837,0.4369038793,0.3161369847,0.7397970507,0.8358259157,0.1373398706,0.8847775773,0.6781446581,0.8966016432,0.618845966,0.1062278019,0.6426598624,0.2813842455,0.8907713531);
return cl::sycl::floor(inputData_0);

});

test_function<458, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8456671764,0.7886065887);
return cl::sycl::floor(inputData_0);

});

test_function<459, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6669284831,0.3027148984,0.841312426);
return cl::sycl::floor(inputData_0);

});

test_function<460, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7580920429,0.6071427292,0.2367621392,0.2593978278);
return cl::sycl::floor(inputData_0);

});

test_function<461, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.5535354293,0.5746397678,0.4264090317,0.7508149069,0.6427018049,0.3064444363,0.1240353155,0.5854300508);
return cl::sycl::floor(inputData_0);

});

test_function<462, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3635183643,0.8118687654,0.2873636579,0.5389066447,0.2583430822,0.8733831646,0.6143516334,0.7024666837,0.8126850616,0.4631054156,0.5373923341,0.4597113108,0.5952625153,0.3678030906,0.7575708827,0.3921740956);
return cl::sycl::floor(inputData_0);

});

test_function<463, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7171064066,0.1133590068);
return cl::sycl::floor(inputData_0);

});

test_function<464, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.390947951,0.4839154474,0.8208412899);
return cl::sycl::floor(inputData_0);

});

test_function<465, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6778929989,0.6870434157,0.6093155969,0.8205987915);
return cl::sycl::floor(inputData_0);

});

test_function<466, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6799357825,0.6051797146,0.8366848074,0.7113561726,0.1118308405,0.7948955055,0.2505539692,0.6987000852);
return cl::sycl::floor(inputData_0);

});

test_function<467, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4307057843,0.4138439486,0.2230832915,0.4430915378,0.8995551553,0.7618324218,0.6893931372,0.6143498942,0.2717405453,0.1505347771,0.8461766014,0.8268376623,0.1017751294,0.8288688361,0.6990884673,0.1789962649);
return cl::sycl::floor(inputData_0);

});

test_function<468, float>(
[=](){
float inputData_0(0.223378022);
float inputData_1(0.2410957479);
float inputData_2(0.3052864821);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<469, double>(
[=](){
double inputData_0(0.8222590664);
double inputData_1(0.4369439383);
double inputData_2(0.5307817503);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<470, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4394748538);
cl::sycl::half inputData_1(0.4712778397);
cl::sycl::half inputData_2(0.3006452895);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<471, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3520967618,0.3019595308);
cl::sycl::float2 inputData_1(0.3279179417,0.5863366634);
cl::sycl::float2 inputData_2(0.1186508566,0.8498295346);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<472, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.510992628,0.5016983462,0.4375022845);
cl::sycl::float3 inputData_1(0.68480467,0.7090437019,0.6778958621);
cl::sycl::float3 inputData_2(0.8723172189,0.4315975222,0.1685146865);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<473, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7386688189,0.6699782599,0.1203602256,0.4942562144);
cl::sycl::float4 inputData_1(0.7478193783,0.8586155148,0.3797446462,0.4773999007);
cl::sycl::float4 inputData_2(0.4559235606,0.6952727362,0.6627633864,0.4799219941);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<474, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2480146635,0.5697127626,0.1199919975,0.3845975173,0.3462894465,0.1533479704,0.1779148469,0.7306529856);
cl::sycl::float8 inputData_1(0.4625251152,0.2029723166,0.8471925288,0.3525179377,0.452181547,0.6558514832,0.2236248344,0.7259980139);
cl::sycl::float8 inputData_2(0.7762561396,0.4930015882,0.6675649376,0.6787032712,0.8934955074,0.7201982084,0.3886160548,0.5085528926);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<475, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4150712539,0.5280843595,0.8166120982,0.1769559163,0.8153703995,0.5031263002,0.2163208965,0.8471128722,0.2556513431,0.1215820461,0.4766964917,0.5589354427,0.140432752,0.6030987291,0.5504965552,0.5505197898);
cl::sycl::float16 inputData_1(0.1647605258,0.3853979762,0.2044629328,0.47433325,0.3964967432,0.3638675755,0.8054387873,0.599395905,0.1880449492,0.6258946616,0.896997032,0.3922335755,0.7164185226,0.6257186994,0.4800416553,0.4236811974);
cl::sycl::float16 inputData_2(0.2711592739,0.4342089764,0.3134502527,0.4890447523,0.6355003923,0.8094935738,0.2163402225,0.7478225482,0.7095688726,0.6024555789,0.2536596731,0.1573656905,0.2094812898,0.3574803195,0.3330981958,0.7483176675);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<476, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3748730555,0.2298155574);
cl::sycl::double2 inputData_1(0.8805767186,0.6639147302);
cl::sycl::double2 inputData_2(0.7426468207,0.4977765748);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<477, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1775191697,0.5690074301,0.455794446);
cl::sycl::double3 inputData_1(0.3544397782,0.1699853638,0.3528896163);
cl::sycl::double3 inputData_2(0.3178483244,0.3850097517,0.6706672549);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<478, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1933067448,0.4030728598,0.7070926447,0.3477506896);
cl::sycl::double4 inputData_1(0.78477792,0.7912480593,0.2513779322,0.6196702379);
cl::sycl::double4 inputData_2(0.6115331661,0.2358041695,0.5462213645,0.4801936088);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<479, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.292590553,0.6703986259,0.81630366,0.81718754,0.184614754,0.4030817493,0.1714004421,0.4710896759);
cl::sycl::double8 inputData_1(0.6305444666,0.6655086585,0.146164238,0.1904339084,0.6057308025,0.7679451072,0.5860101097,0.1751300054);
cl::sycl::double8 inputData_2(0.1654704844,0.5644886142,0.2735237232,0.6981052333,0.3417041233,0.7607512438,0.8216795336,0.5248494235);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<480, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6621319787,0.7112531146,0.601336969,0.6174364257,0.4229453758,0.8142498785,0.2417618508,0.2294734885,0.679508478,0.2691373386,0.5160968889,0.8211709843,0.386065207,0.6884898153,0.7952535802,0.6269227554);
cl::sycl::double16 inputData_1(0.172636046,0.5130811895,0.7376340019,0.8982151169,0.4669277418,0.1450092503,0.3793138514,0.4995987965,0.8435221149,0.1521443076,0.8326475129,0.4519797113,0.1796256802,0.7012933983,0.1968635757,0.2071841133);
cl::sycl::double16 inputData_2(0.2894909299,0.5377921064,0.6282547027,0.7882958909,0.5717896102,0.6099915556,0.4849550599,0.5034436463,0.6303726368,0.3159559534,0.3367935122,0.5983830552,0.5973919089,0.2441567974,0.2565682024,0.5387470943);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<481, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7665935821,0.3494261589);
cl::sycl::half2 inputData_1(0.2391959257,0.1388376084);
cl::sycl::half2 inputData_2(0.524132198,0.5227638218);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<482, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5901311789,0.5372527286,0.6271627806);
cl::sycl::half3 inputData_1(0.7961051787,0.353999517,0.7368335923);
cl::sycl::half3 inputData_2(0.2813913107,0.3646967828,0.4955069133);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<483, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5157067819,0.6901468745,0.1208417235,0.6907310574);
cl::sycl::half4 inputData_1(0.7449030884,0.1011187022,0.5495931259,0.1149817339);
cl::sycl::half4 inputData_2(0.2106550024,0.7969515151,0.1439693232,0.8259201486);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<484, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6339923572,0.7127407495,0.6401808146,0.7067049279,0.6855705033,0.8158876168,0.3878762243,0.406250894);
cl::sycl::half8 inputData_1(0.6195409116,0.5241313673,0.6428181051,0.3945274228,0.434980054,0.4137094243,0.1317078566,0.6894763841);
cl::sycl::half8 inputData_2(0.55516994,0.8087586012,0.6903255106,0.4319233069,0.7203567159,0.1507282768,0.6089008339,0.8449529941);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<485, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8132893788,0.3531265661,0.3244546585,0.3358865009,0.8093207863,0.3738007517,0.4306795507,0.3637533594,0.3129808099,0.7960788387,0.4549334752,0.5273719152,0.4210703175,0.368272471,0.4228737711,0.8159086392);
cl::sycl::half16 inputData_1(0.3242541428,0.6573826612,0.2712451329,0.8273697538,0.4735405967,0.3390211189,0.4428564354,0.6918058027,0.7851268526,0.750076453,0.2038478847,0.873844869,0.4818585824,0.8608836233,0.2366500057,0.8527310307);
cl::sycl::half16 inputData_2(0.2543708031,0.7955126103,0.3165447146,0.7727605075,0.2310523009,0.7838064264,0.6290916829,0.1971190779,0.8494889632,0.4472183226,0.2469539172,0.774935424,0.1435188086,0.1305866641,0.6897942543,0.1718969273);
return cl::sycl::fma(inputData_0,inputData_1,inputData_2);

});

test_function<486, float>(
[=](){
float inputData_0(0.6971291585);
float inputData_1(0.1755738549);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<487, double>(
[=](){
double inputData_0(0.8517330086);
double inputData_1(0.2098735997);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<488, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.550355078);
cl::sycl::half inputData_1(0.6168708763);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<489, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3929657048,0.1766417824);
cl::sycl::float2 inputData_1(0.4906333871,0.3685226534);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<490, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8127489836,0.2625322557,0.2790231452);
cl::sycl::float3 inputData_1(0.577754036,0.4551919913,0.329583705);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<491, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2122732843,0.7718500673,0.1764387235,0.8231723081);
cl::sycl::float4 inputData_1(0.8523542507,0.2042018007,0.8397095662,0.3125681953);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<492, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.73397579,0.3597293685,0.4564754605,0.6683941802,0.5369914895,0.7036695724,0.4170466364,0.1498448979);
cl::sycl::float8 inputData_1(0.374768677,0.8077080145,0.7418629726,0.8409457828,0.5713530905,0.3625684682,0.485371492,0.4945523101);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<493, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2960092329,0.797013889,0.1472565612,0.3652552972,0.882514668,0.5482751148,0.8550612775,0.1049727129,0.7493169464,0.6004190931,0.7851244994,0.6876900647,0.89989045,0.3701967459,0.6951958445,0.7756952218);
cl::sycl::float16 inputData_1(0.6546268765,0.7330136859,0.7553301333,0.3174278764,0.326586342,0.4235813728,0.1190195489,0.8240400461,0.7488557763,0.2054600823,0.7548885661,0.3636473197,0.585450227,0.4913582075,0.4557485581,0.5872091968);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<494, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8258970568,0.4312833238);
cl::sycl::double2 inputData_1(0.1213288206,0.7811045598);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<495, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8615205501,0.7457414497,0.2253380899);
cl::sycl::double3 inputData_1(0.4369431607,0.7055059106,0.7908888367);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<496, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7958901573,0.8389092954,0.6132910119,0.2109313423);
cl::sycl::double4 inputData_1(0.4361262788,0.8675965687,0.2057806235,0.846359292);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<497, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.387662506,0.5727033506,0.7507620578,0.1743352964,0.1898878408,0.4826942026,0.8234592004,0.8762206128);
cl::sycl::double8 inputData_1(0.6234423396,0.2531856419,0.7347372583,0.400976034,0.3114483974,0.571929045,0.4543402547,0.5153529152);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<498, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8763563749,0.1307017881,0.7988407281,0.4467921907,0.7765880121,0.8008602426,0.6894837364,0.4790056528,0.5173904945,0.5768628522,0.4164118647,0.6708331513,0.7838016439,0.3695572329,0.8636037169,0.5501765749);
cl::sycl::double16 inputData_1(0.3349081147,0.2301913391,0.6810712559,0.3248700094,0.7471076069,0.8575609929,0.3819983824,0.4023859212,0.3414398838,0.3045330507,0.8989299692,0.7621688614,0.7920667136,0.1221543233,0.8619132394,0.7590958898);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<499, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7586159896,0.8412913703);
cl::sycl::half2 inputData_1(0.4293733303,0.1614974479);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<500, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5555571571,0.5285539989,0.2727448125);
cl::sycl::half3 inputData_1(0.6939152387,0.888146076,0.6730175055);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<501, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7353140598,0.7181894968,0.7484047466,0.2276771651);
cl::sycl::half4 inputData_1(0.5844761466,0.5866541389,0.7992260573,0.4503208326);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<502, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3081367267,0.5292973522,0.4948803727,0.1134617665,0.1390208935,0.2504851173,0.4940573201,0.6268347216);
cl::sycl::half8 inputData_1(0.5167865651,0.5599072322,0.8935996447,0.1701146373,0.3981610945,0.5362935949,0.7325222373,0.8069708883);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<503, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5961919268,0.6244554896,0.8154815766,0.7335913433,0.5646166343,0.1097440873,0.4300869605,0.1567087253,0.8703820186,0.7613072918,0.3767779876,0.1803580242,0.5512691089,0.3816865387,0.678612423,0.76983758);
cl::sycl::half16 inputData_1(0.577582317,0.754654541,0.1067796018,0.2867669154,0.4445214723,0.4720196086,0.7501915157,0.2404037576,0.558313742,0.5303412024,0.6183121608,0.1109533886,0.1349844205,0.1727141746,0.4147154771,0.5782012115);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<504, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6872054471);
float inputData_1(0.6207263741);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<505, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.186009598,0.7926613989);
float inputData_1(0.4156063938);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<506, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7530869084,0.236927562,0.4042333724);
float inputData_1(0.6363481839);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<507, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6040703024,0.5150323999,0.8385108442,0.7492470166);
float inputData_1(0.5766820863);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<508, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.642132205,0.432159397,0.510321468,0.6042064252,0.7699264576,0.2599823733,0.4661792739,0.2472326015);
float inputData_1(0.2577601088);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<509, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1285438371,0.1953455287,0.7058803041,0.3639610215,0.3438004839,0.2151071023,0.5611054285,0.4320686238,0.1442118701,0.4060692954,0.7460869855,0.7054181431,0.879791336,0.7859805579,0.4981524298,0.7339008712);
float inputData_1(0.7283111109);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<510, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4280204699,0.4097192295);
double inputData_1(0.5480760016);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<511, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.454049366,0.5983507197,0.7510180853);
double inputData_1(0.8351755443);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<512, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.6146611429,0.5198463096,0.2726597697,0.4196614354);
double inputData_1(0.2793971453);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<513, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6830774729,0.5322328038,0.2226278834,0.2321764547,0.1225390155,0.1910317846,0.5953346787,0.610114187);
double inputData_1(0.5226126469);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<514, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.7315483938,0.5371909464,0.3051304341,0.7124416041,0.3931459614,0.135590753,0.2729331655,0.4557987014,0.5904924886,0.7236091281,0.7891636249,0.7194308761,0.7011669353,0.5714757935,0.2545597514,0.6329136557);
double inputData_1(0.655504482);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<515, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3144754797,0.3968823006);
float inputData_1(0.3350626234);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<516, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.896520491,0.6576935477,0.3873378115);
float inputData_1(0.255808347);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<517, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.372435136,0.292944908,0.7646242452,0.2504124027);
float inputData_1(0.4027622588);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<518, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5279459774,0.3011499504,0.2351084886,0.8691047749,0.8229378664,0.1214912884,0.5268634121,0.1349855574);
float inputData_1(0.4469779711);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<519, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1876164273,0.7436187548,0.2607947201,0.3404832109,0.1686544055,0.8745756215,0.5000992952,0.4377529244,0.5422817192,0.2435744235,0.7380038686,0.3296917244,0.7775411653,0.6913402901,0.3357555698,0.4888309473);
float inputData_1(0.335575141);
return cl::sycl::fmax(inputData_0,inputData_1);

});

test_function<520, float>(
[=](){
float inputData_0(0.8521575386);
float inputData_1(0.77917996);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<521, double>(
[=](){
double inputData_0(0.5311922585);
double inputData_1(0.6517901375);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<522, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1621495812);
cl::sycl::half inputData_1(0.7405864369);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<523, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3866373982,0.3148516651);
cl::sycl::float2 inputData_1(0.3621672566,0.8113618076);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<524, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.182637124,0.8892045208,0.4708639611);
cl::sycl::float3 inputData_1(0.6019113442,0.5053157635,0.1501586254);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<525, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.712587866,0.4208391414,0.4856938739,0.7572211898);
cl::sycl::float4 inputData_1(0.349258831,0.2127679186,0.1364993895,0.6519742358);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<526, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1530680446,0.5747571121,0.1545136736,0.1903249673,0.4151152863,0.3335018553,0.1857925917,0.8534966327);
cl::sycl::float8 inputData_1(0.8571776127,0.8701330858,0.8843385073,0.8276246192,0.5528272393,0.4234132587,0.6934586788,0.519014617);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<527, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2818650299,0.7205694928,0.230443863,0.6214289076,0.8598131246,0.5282404399,0.2163321095,0.3866678267,0.5086104169,0.4456656681,0.2192736085,0.5785442984,0.8040100466,0.8773466607,0.2451060519,0.6266467802);
cl::sycl::float16 inputData_1(0.1174111327,0.2198555714,0.5107035656,0.585136705,0.2983016735,0.6877781912,0.8136410926,0.8760574753,0.1831414565,0.6466710596,0.5183922029,0.658463722,0.8467303067,0.3291674396,0.8050497688,0.1724878979);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<528, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.1330631771,0.8400690162);
cl::sycl::double2 inputData_1(0.3071434767,0.6685185333);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<529, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2758123636,0.5706316644,0.3722282544);
cl::sycl::double3 inputData_1(0.6783991383,0.695848034,0.8218683068);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<530, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7426978936,0.4601282704,0.2018505158,0.6371129221);
cl::sycl::double4 inputData_1(0.4513997253,0.6670433598,0.890554534,0.3678185299);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<531, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.454939741,0.3746140404,0.3570342102,0.2940016914,0.3822937174,0.3896248935,0.3867404694,0.8583333475);
cl::sycl::double8 inputData_1(0.4001576436,0.1139353716,0.8718775548,0.5648230644,0.3465608323,0.7350143051,0.1176080557,0.7836866431);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<532, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1950542946,0.1193584945,0.2147003716,0.7681717463,0.1405852476,0.8310274873,0.3334598122,0.2575939939,0.6240315101,0.2939386666,0.5521893387,0.7111203922,0.6341497089,0.3539066949,0.6066624206,0.6030468183);
cl::sycl::double16 inputData_1(0.1165890817,0.4686713831,0.606180143,0.4052646232,0.2283424003,0.8301093665,0.10869279,0.792220074,0.2333621178,0.8931096296,0.8285489085,0.1734644375,0.7141124729,0.6342504515,0.7917126311,0.4971128479);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<533, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3496754561,0.732626655);
cl::sycl::half2 inputData_1(0.6891272643,0.5291062553);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<534, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8883221129,0.7155798358,0.4977126538);
cl::sycl::half3 inputData_1(0.8406474269,0.5046558012,0.7744870053);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<535, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1399622514,0.2039270445,0.2703526554,0.2856873101);
cl::sycl::half4 inputData_1(0.2123442299,0.1332267022,0.5476955956,0.5719766428);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<536, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6812238145,0.5918843891,0.4625254074,0.4599295103,0.1112506314,0.5931195066,0.6087287874,0.6005880408);
cl::sycl::half8 inputData_1(0.8693585832,0.6626207979,0.2685862489,0.5884924132,0.5431393879,0.6382448399,0.5758610399,0.7059491518);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<537, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4664223526,0.5534886315,0.8405196034,0.3111284349,0.6299871344,0.6903563983,0.7569380242,0.5586485189,0.8793508128,0.8703424316,0.5109802977,0.4617704835,0.1826019759,0.5697314192,0.3488356192,0.650399435);
cl::sycl::half16 inputData_1(0.6729730572,0.792820402,0.6944799602,0.7475186891,0.7728788255,0.8966970784,0.3632633318,0.5710776037,0.5457891893,0.5567002515,0.6342205704,0.8495404061,0.1146332086,0.3481008336,0.4700744516,0.5134601037);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<538, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8255037011);
float inputData_1(0.4997138187);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<539, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2892149405,0.7858165438);
float inputData_1(0.5117627877);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<540, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7274403269,0.3112838757,0.8559693738);
float inputData_1(0.4211061275);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<541, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4574829156,0.4814065773,0.3647869409,0.7068640462);
float inputData_1(0.5609832214);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<542, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.582792252,0.5807624684,0.3355264849,0.373377971,0.1729355225,0.4079976288,0.3098082591,0.3789389448);
float inputData_1(0.5997858996);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<543, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6479986009,0.6987939225,0.2772039694,0.2464514348,0.8745345083,0.4731793777,0.4806048756,0.1664272006,0.6496570587,0.2379794988,0.1417737888,0.3007820091,0.8729212037,0.3659499488,0.1456192457,0.6362359902);
float inputData_1(0.2392175673);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<544, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3277848714,0.4808980289);
double inputData_1(0.5324755162);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<545, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.3422720809,0.8398154632,0.7062705673);
double inputData_1(0.6014455695);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<546, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1600850402,0.7544205448,0.7633330048,0.7467060295);
double inputData_1(0.7648459293);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<547, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2390347085,0.8191602565,0.3356656235,0.3115437835,0.1014621225,0.2262110129,0.3780671177,0.2866531498);
double inputData_1(0.7993081906);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<548, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1381324195,0.4561405468,0.403087894,0.7917995131,0.7683523299,0.3595412077,0.6425476969,0.4673705695,0.3000361634,0.6572711061,0.4721707827,0.6564438031,0.6903580153,0.5495744232,0.3114661044,0.2097454373);
double inputData_1(0.108392711);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<549, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4384863289,0.6284707116);
float inputData_1(0.7613411044);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<550, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.3279174078,0.154589661,0.485294721);
float inputData_1(0.3707784609);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<551, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.367509432,0.1378254338,0.5159453585,0.4537656875);
float inputData_1(0.6659324409);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<552, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6426203717,0.4060953755,0.1341092081,0.8772549026,0.2623636229,0.691229207,0.5491347149,0.1240943719);
float inputData_1(0.6865443917);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<553, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8779572088,0.2197542655,0.6722775772,0.5576388031,0.5325387865,0.4468091022,0.4573639289,0.2375731828,0.3885557351,0.6403730052,0.7792737356,0.6889038464,0.6010013001,0.2832095587,0.1043780867,0.7866253667);
float inputData_1(0.7130931716);
return cl::sycl::fmin(inputData_0,inputData_1);

});

test_function<554, float>(
[=](){
float inputData_0(0.8574289268);
float inputData_1(0.208605206);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<555, double>(
[=](){
double inputData_0(0.8791399276);
double inputData_1(0.3640832477);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<556, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5510332594);
cl::sycl::half inputData_1(0.1154956462);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<557, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7355508657,0.8479700575);
cl::sycl::float2 inputData_1(0.8570016538,0.2239391034);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<558, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2116395423,0.5382056545,0.4259589576);
cl::sycl::float3 inputData_1(0.1149825297,0.3305704793,0.7229604635);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<559, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7113948644,0.2897973215,0.8354715586,0.3940652238);
cl::sycl::float4 inputData_1(0.7156991802,0.3738828426,0.8890464161,0.6409702461);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<560, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.3967654012,0.1517716432,0.2501785359,0.857711446,0.5601256088,0.7398520538,0.2702401211,0.5194958102);
cl::sycl::float8 inputData_1(0.5802306899,0.8511924018,0.5044641601,0.8391314632,0.3549607669,0.5661854956,0.600222329,0.1768505133);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<561, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2323965034,0.3043309299,0.3098149863,0.2479383611,0.4348735712,0.1779221315,0.2197717444,0.7845162439,0.2434611937,0.4580955849,0.1647253655,0.7372832797,0.3144591741,0.6299776277,0.7554673263,0.4312185107);
cl::sycl::float16 inputData_1(0.2655436945,0.226167277,0.4677863717,0.5756822492,0.6271088588,0.5692971868,0.2953825102,0.1012588821,0.2091843845,0.4528699363,0.8751285787,0.5540888596,0.8838199204,0.7728652898,0.8967847251,0.3315167107);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<562, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6518982601,0.4447000563);
cl::sycl::double2 inputData_1(0.7968042893,0.5906602992);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<563, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4231670595,0.854579237,0.7729354557);
cl::sycl::double3 inputData_1(0.4949600375,0.4401432156,0.422618198);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<564, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4240365866,0.3479767691,0.2338770799,0.7642960746);
cl::sycl::double4 inputData_1(0.1974863082,0.5983186033,0.2273276602,0.7942787432);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<565, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2822472198,0.2005933419,0.1142225382,0.8237847773,0.7309351729,0.3508876948,0.7915906331,0.5064203844);
cl::sycl::double8 inputData_1(0.2752380151,0.5540668239,0.1228535622,0.6565571078,0.3169995022,0.3990433098,0.7902214187,0.5987051393);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<566, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8674200161,0.2812930698,0.1053485943,0.7815918672,0.5872951948,0.2064568809,0.3027990742,0.857679806,0.5617920983,0.5429821444,0.7166909643,0.8210297643,0.4121534977,0.7595989386,0.7952659485,0.8640566393);
cl::sycl::double16 inputData_1(0.4843227289,0.8141092114,0.7586499791,0.4756956614,0.6743269193,0.2080324532,0.6955822086,0.4169803301,0.7623279478,0.3759771366,0.7380381354,0.3320995447,0.4989861444,0.4177443904,0.8971049404,0.2928849923);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<567, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7736276952,0.7463056977);
cl::sycl::half2 inputData_1(0.8573950362,0.6541990708);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<568, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.4972703383,0.8533697334,0.1302085146);
cl::sycl::half3 inputData_1(0.8261896123,0.6936451936,0.2619947644);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<569, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3339736741,0.7911299373,0.1829968888,0.3162353049);
cl::sycl::half4 inputData_1(0.2205402538,0.2674508232,0.2043052266,0.3776827825);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<570, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7984981035,0.5743790019,0.2273474578,0.1837989755,0.8983872589,0.3019071077,0.3292871969,0.2485789958);
cl::sycl::half8 inputData_1(0.3232236539,0.855690669,0.3968757457,0.4309285835,0.3674686892,0.5766184959,0.5196965195,0.308528029);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<571, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.186136925,0.5437680122,0.5459098983,0.1452619852,0.3731835468,0.4393217957,0.4871252405,0.3540941828,0.6355274593,0.7410142018,0.1954579259,0.4374870118,0.6340944832,0.5740747272,0.7979324415,0.7963812695);
cl::sycl::half16 inputData_1(0.5363963624,0.8103160723,0.7452321708,0.2202053967,0.1074036308,0.409178697,0.392037325,0.1225468837,0.7483395835,0.5130812501,0.8807834144,0.7667939832,0.7462623989,0.4192141863,0.8547392602,0.6706236786);
return cl::sycl::fmod(inputData_0,inputData_1);

});

test_function<572, float>(
[=](){
float inputData_0(0.1840296842);
float * inputData_1 = new float(0.5430571687);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<573, double>(
[=](){
double inputData_0(0.2689905305);
double * inputData_1 = new double(0.6139712069);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<574, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.421951563);
cl::sycl::half * inputData_1 = new cl::sycl::half(0.1328275673);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<575, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8195902293,0.83559379);
cl::sycl::float2 * inputData_1 = new cl::sycl::float2(0.6093784234,0.2662488871);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<576, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4293700967,0.6557195225,0.1926874245);
cl::sycl::float3 * inputData_1 = new cl::sycl::float3(0.2657491048,0.1724183566,0.8472059234);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<577, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7039347755,0.2099410848,0.2885174985,0.6258159443);
cl::sycl::float4 * inputData_1 = new cl::sycl::float4(0.8789639066,0.2723070259,0.7785952472,0.448192693);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<578, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7859745991,0.1091252384,0.5178370433,0.4349272949,0.7298228694,0.8689285368,0.1813940573,0.7025927759);
cl::sycl::float8 * inputData_1 = new cl::sycl::float8(0.3903662613,0.8052497902,0.7911225637,0.8979169324,0.5074689104,0.4990258669,0.2696347747,0.1409243975);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<579, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8243656198,0.1865430192,0.445762805,0.1811611282,0.1299549824,0.3622618457,0.1916070081,0.326414114,0.8942340618,0.2924260311,0.7303054291,0.7311457549,0.1519208675,0.4132510989,0.2193447823,0.3597314192);
cl::sycl::float16 * inputData_1 = new cl::sycl::float16(0.2043609455,0.2103335323,0.6715188001,0.143349853,0.481297985,0.54141642,0.4394524566,0.2980039669,0.598547988,0.1778425833,0.2397416515,0.3362914032,0.4593422678,0.2614985939,0.6194652943,0.3094868051);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<580, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4831939351,0.7439172414);
cl::sycl::double2 * inputData_1 = new cl::sycl::double2(0.6352647076,0.5821626557);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<581, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7658602927,0.4231887004,0.4440065161);
cl::sycl::double3 * inputData_1 = new cl::sycl::double3(0.4124747107,0.2047072633,0.577248964);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<582, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4921472854,0.7384003176,0.5868540818,0.3762202959);
cl::sycl::double4 * inputData_1 = new cl::sycl::double4(0.7874239478,0.3212179482,0.3444610861,0.7955127535);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<583, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8839360946,0.2693822586,0.2797940271,0.5634530921,0.1195322535,0.7520442309,0.4411499615,0.6849017605);
cl::sycl::double8 * inputData_1 = new cl::sycl::double8(0.1202335809,0.7356581904,0.5387196707,0.1175444886,0.1385989848,0.425608581,0.5985705334,0.4624312778);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<584, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6149884464,0.8203146729,0.405583728,0.4365994351,0.7361133381,0.1349292624,0.371219671,0.1499877354,0.6229070533,0.6394098745,0.6312069542,0.5963958603,0.6968063704,0.1111343615,0.6041627438,0.6043162352);
cl::sycl::double16 * inputData_1 = new cl::sycl::double16(0.7674172793,0.7034983548,0.1275344682,0.5736525272,0.4294977893,0.3236953443,0.1866404723,0.1972707383,0.2243941161,0.2912852957,0.4864578957,0.5837912316,0.307933315,0.2632842603,0.5176419409,0.4904252759);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<585, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.160127186,0.2828090914);
cl::sycl::half2 * inputData_1 = new cl::sycl::half2(0.4150539141,0.8473724524);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<586, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.4665891771,0.2969987287,0.7990082131);
cl::sycl::half3 * inputData_1 = new cl::sycl::half3(0.8985502131,0.2243440892,0.797895662);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<587, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4926971997,0.3050638235,0.7497798037,0.5395198407);
cl::sycl::half4 * inputData_1 = new cl::sycl::half4(0.5419392585,0.6578699271,0.4123922,0.6604293018);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<588, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6932710563,0.74703644,0.1385810606,0.3453742229,0.5663960843,0.632962766,0.5771135017,0.7423576499);
cl::sycl::half8 * inputData_1 = new cl::sycl::half8(0.3656712817,0.6119802169,0.526371391,0.773917431,0.3391006461,0.5277380735,0.2898812612,0.839771675);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<589, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8703587039,0.3429725661,0.7901584894,0.4474740998,0.3335978197,0.4044773538,0.6946123977,0.8870794883,0.764290042,0.7745337075,0.3426259297,0.2560785717,0.294001993,0.8542954894,0.2310765345,0.2702655042);
cl::sycl::half16 * inputData_1 = new cl::sycl::half16(0.7349281577,0.7936322501,0.2958274381,0.5182094332,0.3916394555,0.7905326192,0.5682304856,0.4811918166,0.6437165504,0.8879183362,0.611005003,0.7465190144,0.7491618232,0.3119599224,0.4121997201,0.2591012539);
return cl::sycl::fract(inputData_0,inputData_1);
delete inputData_1;

});

test_function<590, float>(
[=](){
float inputData_0(0.5831008563);
int * inputData_1 = new int(18790);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<591, double>(
[=](){
double inputData_0(0.432752433);
int * inputData_1 = new int(-10936);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<592, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4415669966);
int * inputData_1 = new int(14519);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<593, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1325020806,0.8395662813);
cl::sycl::int2 * inputData_1 = new cl::sycl::int2(10563,4755);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<594, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1113461608,0.1689677479,0.2207859931);
cl::sycl::int3 * inputData_1 = new cl::sycl::int3(7628,3880,-22008);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<595, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7875172658,0.3657536413,0.2117931992,0.3334990557);
cl::sycl::int4 * inputData_1 = new cl::sycl::int4(16440,-888,9034,-15113);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<596, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.5962007713,0.2971451946,0.1105968529,0.6736452462,0.4534361379,0.3412686272,0.5208766521,0.4048119752);
cl::sycl::int8 * inputData_1 = new cl::sycl::int8(20157,30826,9971,-30247,10082,-32164,-6589,-26764);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<597, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7907898206,0.1208876607,0.6423394077,0.4697353048,0.5073015406,0.3071277439,0.4016209546,0.4463213316,0.4601853785,0.4907269965,0.7679364217,0.6046959764,0.7882375571,0.5125751203,0.518867792,0.6030005501);
cl::sycl::int16 * inputData_1 = new cl::sycl::int16(13893,15377,3875,-6564,-14597,15564,-28877,29961,32100,26002,25377,29969,6616,22002,-20613,-2686);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<598, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.11833373,0.3835038677);
cl::sycl::int2 * inputData_1 = new cl::sycl::int2(-4263,-21409);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<599, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6704646559,0.2411116353,0.2283619191);
cl::sycl::int3 * inputData_1 = new cl::sycl::int3(16348,26870,27868);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<600, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7934746071,0.2589857894,0.584164339,0.5255605613);
cl::sycl::int4 * inputData_1 = new cl::sycl::int4(-29413,-17695,4739,23517);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<601, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7040057502,0.7497891263,0.4638489176,0.8629316433,0.5891026693,0.323215899,0.3648079037,0.3995302736);
cl::sycl::int8 * inputData_1 = new cl::sycl::int8(-24201,9385,2406,21205,-26387,-22706,-10620,-18553);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<602, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3009054942,0.3816269375,0.3831499353,0.5061123006,0.6917523325,0.5157736886,0.8129792269,0.1505307194,0.6250325764,0.6290229618,0.8456918783,0.5605756988,0.6685775961,0.4675611841,0.1592733785,0.1797437532);
cl::sycl::int16 * inputData_1 = new cl::sycl::int16(-2908,8353,-7742,-20670,24108,30020,-10210,-14522,-23521,31052,4367,3019,-12218,19697,-20414,10876);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<603, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6360352913,0.8486772957);
cl::sycl::int2 * inputData_1 = new cl::sycl::int2(21845,-3638);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<604, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6787102916,0.5056299532,0.2091586742);
cl::sycl::int3 * inputData_1 = new cl::sycl::int3(2612,14376,28004);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<605, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5446575177,0.312154353,0.5107656143,0.6560897751);
cl::sycl::int4 * inputData_1 = new cl::sycl::int4(-14124,-24156,-20126,-13077);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<606, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2656162944,0.3355004251,0.1337321605,0.4490704986,0.6530838546,0.1082706672,0.2332101207,0.1826012693);
cl::sycl::int8 * inputData_1 = new cl::sycl::int8(-5306,24536,-15153,-6451,-9559,-22598,-26788,-18679);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<607, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1335646091,0.4146885686,0.7078002177,0.3227702935,0.8913459849,0.5032768888,0.7051002636,0.5543956851,0.5541967811,0.7252692192,0.7275541618,0.2642706411,0.2285490088,0.2473347095,0.2852241876,0.1627054512);
cl::sycl::int16 * inputData_1 = new cl::sycl::int16(-17582,-6728,8529,-30960,7513,13768,25657,10211,4414,-13234,-7719,7420,19844,-9915,10869,-8930);
return cl::sycl::frexp(inputData_0,inputData_1);
delete inputData_1;

});

test_function<608, float>(
[=](){
float inputData_0(0.5952162777);
float inputData_1(0.1354383814);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<609, double>(
[=](){
double inputData_0(0.1809051475);
double inputData_1(0.6271213499);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<610, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7859431278);
cl::sycl::half inputData_1(0.3482093351);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<611, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6924631959,0.7176288963);
cl::sycl::float2 inputData_1(0.154453688,0.6364816046);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<612, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4481292061,0.2604968279,0.2635182648);
cl::sycl::float3 inputData_1(0.7572193656,0.6416319712,0.4242714138);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<613, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4782219108,0.228802366,0.89135077,0.5198867885);
cl::sycl::float4 inputData_1(0.1591710559,0.4302402295,0.8023477238,0.3560586688);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<614, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6861106049,0.1640626345,0.7141563115,0.5898756703,0.4643034793,0.2543080253,0.4914250355,0.7363442625);
cl::sycl::float8 inputData_1(0.140817065,0.3003156661,0.3334149572,0.335535424,0.6344724502,0.8436012374,0.5720536053,0.6248715101);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<615, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.494457889,0.4774208325,0.6768931476,0.647277591,0.3933362182,0.4838628329,0.1559270898,0.3465817285,0.7020188804,0.8849268879,0.4706947884,0.4539766529,0.2157842921,0.6515797167,0.8108135543,0.1514917687);
cl::sycl::float16 inputData_1(0.6182408247,0.2673548793,0.2083572387,0.6321725638,0.3288460294,0.1648700188,0.6836993873,0.7981271956,0.5675177141,0.7138740937,0.6527442866,0.5303093609,0.4646669317,0.544255641,0.3222658764,0.1041506161);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<616, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6134865695,0.8205014732);
cl::sycl::double2 inputData_1(0.3815678376,0.4497744563);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<617, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6612028355,0.1262108567,0.298926648);
cl::sycl::double3 inputData_1(0.7163501499,0.6215407437,0.1373260615);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<618, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4003350286,0.5775487733,0.1936425404,0.8524074556);
cl::sycl::double4 inputData_1(0.239061877,0.584684408,0.1536086262,0.3011731204);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<619, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2164483991,0.7096409201,0.8835628281,0.8135415662,0.2736320867,0.6125378741,0.8259892198,0.6265227498);
cl::sycl::double8 inputData_1(0.1491733837,0.459252116,0.3655219326,0.6167925481,0.1493738763,0.6271996653,0.1383557969,0.1449404993);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<620, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2826929865,0.7906980717,0.7143411861,0.3113082634,0.1077318459,0.3090002525,0.84651658,0.544522793,0.1640250705,0.6800549765,0.6031338961,0.7779968956,0.884008975,0.7775351199,0.6428452051,0.8019028115);
cl::sycl::double16 inputData_1(0.596572678,0.269121835,0.5118272541,0.3391346069,0.2983567247,0.2482507429,0.1538920647,0.328214475,0.801352575,0.5068336204,0.5239019582,0.1887005292,0.8658360743,0.1964032639,0.8721427317,0.2490842451);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<621, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1393880034,0.8526981924);
cl::sycl::half2 inputData_1(0.124824718,0.317556661);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<622, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6274802908,0.5581609188,0.6860325571);
cl::sycl::half3 inputData_1(0.3522927179,0.7019289971,0.144231182);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<623, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1265108169,0.2867598223,0.7618635555,0.1896469213);
cl::sycl::half4 inputData_1(0.6880559314,0.8539952555,0.8767583829,0.1730501788);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<624, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3046879439,0.1460939287,0.7096307463,0.4386643006,0.3118676083,0.8213501917,0.5727912879,0.8808085604);
cl::sycl::half8 inputData_1(0.6447243139,0.1885704451,0.2079985651,0.1943808756,0.3894495184,0.7307104902,0.8341036461,0.1431806508);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<625, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4269430714,0.8533716082,0.7393117509,0.323850625,0.8179240227,0.2102596253,0.3683641149,0.2635784809,0.4393378636,0.5271245698,0.1583672295,0.2407783725,0.5947837665,0.7720764551,0.171032943,0.8651898903);
cl::sycl::half16 inputData_1(0.5006463156,0.7198390864,0.1468052782,0.385844518,0.4694924569,0.5517750871,0.8702775506,0.5042537636,0.8308793929,0.8742133733,0.8153028619,0.3812115608,0.7942033685,0.3505002936,0.6748934982,0.6576862027);
return cl::sycl::hypot(inputData_0,inputData_1);

});

test_function<626, int>(
[=](){
float inputData_0(0.5817955412);
return cl::sycl::ilogb(inputData_0);

});

test_function<627, int>(
[=](){
double inputData_0(0.820072221);
return cl::sycl::ilogb(inputData_0);

});

test_function<628, int>(
[=](){
cl::sycl::half inputData_0(0.7356343152);
return cl::sycl::ilogb(inputData_0);

});

test_function<629, cl::sycl::int2>(
[=](){
cl::sycl::float2 inputData_0(0.7507744366,0.4392121479);
return cl::sycl::ilogb(inputData_0);

});

test_function<630, cl::sycl::int3>(
[=](){
cl::sycl::float3 inputData_0(0.8086805203,0.6833283192,0.3524837815);
return cl::sycl::ilogb(inputData_0);

});

test_function<631, cl::sycl::int4>(
[=](){
cl::sycl::float4 inputData_0(0.4377061926,0.2912125357,0.7806961291,0.3986812111);
return cl::sycl::ilogb(inputData_0);

});

test_function<632, cl::sycl::int8>(
[=](){
cl::sycl::float8 inputData_0(0.4001456675,0.41024494,0.5474577778,0.3257261021,0.75568798,0.6355439436,0.7336245788,0.1546920884);
return cl::sycl::ilogb(inputData_0);

});

test_function<633, cl::sycl::int16>(
[=](){
cl::sycl::float16 inputData_0(0.4013393869,0.2659973036,0.2546767476,0.3696304012,0.7821255204,0.4021698573,0.1504330689,0.7781918651,0.7291190183,0.5388559378,0.137158098,0.2222620437,0.3861350087,0.4321962973,0.3400392469,0.7139780598);
return cl::sycl::ilogb(inputData_0);

});

test_function<634, cl::sycl::int2>(
[=](){
cl::sycl::double2 inputData_0(0.8739330847,0.4420566186);
return cl::sycl::ilogb(inputData_0);

});

test_function<635, cl::sycl::int3>(
[=](){
cl::sycl::double3 inputData_0(0.2367465496,0.4612821083,0.2103523175);
return cl::sycl::ilogb(inputData_0);

});

test_function<636, cl::sycl::int4>(
[=](){
cl::sycl::double4 inputData_0(0.4711761125,0.4273814651,0.8857884683,0.3818885703);
return cl::sycl::ilogb(inputData_0);

});

test_function<637, cl::sycl::int8>(
[=](){
cl::sycl::double8 inputData_0(0.1975502433,0.8447437518,0.6628889391,0.6569511196,0.3603730837,0.2627905257,0.6699225898,0.3316176734);
return cl::sycl::ilogb(inputData_0);

});

test_function<638, cl::sycl::int16>(
[=](){
cl::sycl::double16 inputData_0(0.5921428922,0.1750624818,0.4503298573,0.6085413276,0.540453773,0.610984788,0.2843741239,0.4639686494,0.1524706795,0.1608351362,0.2656491632,0.6181494711,0.7360614771,0.5786669803,0.3848706506,0.10584984);
return cl::sycl::ilogb(inputData_0);

});

test_function<639, cl::sycl::int2>(
[=](){
cl::sycl::half2 inputData_0(0.6558330553,0.7802138905);
return cl::sycl::ilogb(inputData_0);

});

test_function<640, cl::sycl::int3>(
[=](){
cl::sycl::half3 inputData_0(0.1077144405,0.5938123945,0.3756430934);
return cl::sycl::ilogb(inputData_0);

});

test_function<641, cl::sycl::int4>(
[=](){
cl::sycl::half4 inputData_0(0.4285841708,0.6866055648,0.4468590553,0.5396563405);
return cl::sycl::ilogb(inputData_0);

});

test_function<642, cl::sycl::int8>(
[=](){
cl::sycl::half8 inputData_0(0.6545366296,0.6280249861,0.4806226074,0.5293976108,0.6158852059,0.1107577942,0.1185375159,0.4724018841);
return cl::sycl::ilogb(inputData_0);

});

test_function<643, cl::sycl::int16>(
[=](){
cl::sycl::half16 inputData_0(0.6658839465,0.8607919052,0.5211264614,0.6392249054,0.836750301,0.1795989121,0.2019510439,0.4813064967,0.5537602598,0.3431023513,0.409023279,0.7956610493,0.5250334324,0.5585038138,0.842621095,0.5760922025);
return cl::sycl::ilogb(inputData_0);

});

test_function<644, float>(
[=](){
float inputData_0(0.4696408867);
int inputData_1(-21088);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<645, double>(
[=](){
double inputData_0(0.2528575421);
int inputData_1(348);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<646, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1362507697);
int inputData_1(593);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<647, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8046405821,0.4107361988);
cl::sycl::int2 inputData_1(22184,16484);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<648, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7508075562,0.3704129911,0.5878067466);
cl::sycl::int3 inputData_1(10017,-26256,-21617);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<649, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6375034133,0.6767376641,0.1324773517,0.4882051103);
cl::sycl::int4 inputData_1(-1333,-4952,17244,30847);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<650, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1250377341,0.8436880645,0.6891186912,0.4058073609,0.2883332587,0.5701119782,0.5279489733,0.8588645887);
cl::sycl::int8 inputData_1(31097,18267,24770,12577,18706,2198,13166,-4134);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<651, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8373250104,0.8324568278,0.8972920638,0.3050097488,0.2736232821,0.5932211693,0.3957001556,0.4173745164,0.7178181637,0.6408629236,0.7830394591,0.7466423424,0.4918687396,0.1835450963,0.4377385816,0.7314820936);
cl::sycl::int16 inputData_1(-8713,22445,-6612,20835,16981,11244,-32305,27901,-8720,-26708,-31937,23652,-5342,5667,31764,28509);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<652, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6038781696,0.6852104705);
cl::sycl::int2 inputData_1(9396,-5288);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<653, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1927956165,0.2656341591,0.3383580093);
cl::sycl::int3 inputData_1(-18740,28292,-19332);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<654, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2883379315,0.6567013489,0.8766092744,0.6100683649);
cl::sycl::int4 inputData_1(-13560,20656,-22523,9766);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<655, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4952884436,0.7015554078,0.4887141743,0.5970699135,0.102440967,0.774860492,0.6797196166,0.2580166013);
cl::sycl::int8 inputData_1(29250,32096,8411,22016,32098,-12781,-20646,-11557);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<656, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8734894318,0.3366776785,0.4607464938,0.7808916474,0.2419797158,0.463120873,0.3112489686,0.4355881491,0.8687338222,0.4498549073,0.1493444463,0.814187287,0.7859310179,0.3743842341,0.139179929,0.4955371402);
cl::sycl::int16 inputData_1(9230,-4012,-28354,-26641,16281,2281,9805,-13393,-4170,-10399,10155,-2943,-25461,11716,31033,-5741);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<657, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1263431096,0.5333431622);
cl::sycl::int2 inputData_1(9017,-9547);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<658, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8195737542,0.8605448102,0.4762750622);
cl::sycl::int3 inputData_1(256,-4872,12090);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<659, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1471587,0.3441368313,0.739914039,0.2557694173);
cl::sycl::int4 inputData_1(2508,-18517,-25246,-6233);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<660, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5393915418,0.3098863859,0.371810067,0.4612204593,0.683425288,0.2195000083,0.5706809999,0.5782369354);
cl::sycl::int8 inputData_1(31058,6912,-6128,-1169,7436,2120,11407,-13519);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<661, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2972498889,0.8825696659,0.8532357484,0.6915124073,0.885250999,0.8946993754,0.3166056324,0.3763928782,0.8370660399,0.6734352191,0.2192453746,0.7306453489,0.2549074457,0.2644371678,0.5909993088,0.2668809404);
cl::sycl::int16 inputData_1(-7783,-22071,12419,7548,26683,-26511,31858,-12601,-30866,-9880,29327,-13747,-500,-14720,11576,-8413);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<662, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8397516048,0.1873559529);
int inputData_1(8294);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<663, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8899788582,0.3953306026,0.5280697106);
int inputData_1(17920);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<664, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2164868928,0.4800666607,0.7838862156,0.8041750871);
int inputData_1(1757);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<665, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7384288327,0.5271105954,0.6453000554,0.7482314644,0.1608805844,0.1747847155,0.1357823234,0.8184648135);
int inputData_1(-18367);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<666, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5044191356,0.3847443028,0.4117130909,0.3788289663,0.7938042208,0.1844987732,0.6930014002,0.5025048179,0.5685643895,0.476015854,0.7193902981,0.5344359386,0.4285428168,0.4297549049,0.440173993,0.5419661615);
int inputData_1(-16773);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<667, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8817871277,0.7881582915);
int inputData_1(-10683);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<668, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5255152363,0.6188639832,0.4946620492);
int inputData_1(-21457);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<669, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.6150274211,0.6336385285,0.5695725518,0.516788617);
int inputData_1(-2036);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<670, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8824037752,0.1783563306,0.808249466,0.1014849445,0.706903412,0.4674655029,0.5257696168,0.1409305934);
int inputData_1(29834);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<671, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2447278666,0.7612914472,0.8967299083,0.8095931667,0.6093640954,0.2217951665,0.5739863459,0.6408147826,0.8432965547,0.6488925022,0.2293505577,0.1304010339,0.7460430057,0.3818782672,0.3686489906,0.3797514154);
int inputData_1(27718);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<672, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7689942027,0.7326921328);
int inputData_1(-949);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<673, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8874741709,0.5627020211,0.8107064789);
int inputData_1(30640);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<674, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.581327979,0.574456772,0.870510115,0.124223282);
int inputData_1(2522);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<675, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1947790543,0.7550362567,0.8485067713,0.5486188812,0.3391913525,0.8209352548,0.1821670032,0.7312135301);
int inputData_1(-30043);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<676, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.721094886,0.7107591938,0.7328294176,0.6302814834,0.2457350096,0.7581009655,0.2792336248,0.7630564948,0.6014972743,0.7738825542,0.530454265,0.1108567737,0.8384677609,0.5538574987,0.4570749474,0.4657703722);
int inputData_1(4327);
return cl::sycl::ldexp(inputData_0,inputData_1);

});

test_function<677, float>(
[=](){
float inputData_0(0.7255671535);
return cl::sycl::lgamma(inputData_0);

});

test_function<678, double>(
[=](){
double inputData_0(0.3530654396);
return cl::sycl::lgamma(inputData_0);

});

test_function<679, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4428795545);
return cl::sycl::lgamma(inputData_0);

});

test_function<680, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2524249206,0.4841752294);
return cl::sycl::lgamma(inputData_0);

});

test_function<681, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7215364766,0.1341176957,0.7780476451);
return cl::sycl::lgamma(inputData_0);

});

test_function<682, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2275941736,0.2307255299,0.2694688659,0.69257536);
return cl::sycl::lgamma(inputData_0);

});

test_function<683, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6491859622,0.7782582487,0.2512097092,0.8233877064,0.2289846085,0.2284615345,0.4367257587,0.3683218987);
return cl::sycl::lgamma(inputData_0);

});

test_function<684, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7078784658,0.6404623715,0.1477220959,0.1878296149,0.5201099164,0.183285105,0.5495310541,0.674983043,0.5949984734,0.1075528976,0.8488164912,0.3035417076,0.4662762915,0.5197541828,0.8819002279,0.3473454355);
return cl::sycl::lgamma(inputData_0);

});

test_function<685, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6261133331,0.7707367693);
return cl::sycl::lgamma(inputData_0);

});

test_function<686, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5516570605,0.2782835682,0.3224780367);
return cl::sycl::lgamma(inputData_0);

});

test_function<687, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2046068128,0.6856128909,0.4962782691,0.3634674068);
return cl::sycl::lgamma(inputData_0);

});

test_function<688, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8345942597,0.2597835495,0.201429443,0.5964471693,0.4149799375,0.4033292476,0.5713366043,0.587312262);
return cl::sycl::lgamma(inputData_0);

});

test_function<689, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1153485017,0.1787220706,0.1380715455,0.1271204302,0.2690592343,0.828292318,0.1612718264,0.25463807,0.5613584784,0.4028962757,0.632614973,0.2656250076,0.6514783661,0.1598268671,0.3827437192,0.441828992);
return cl::sycl::lgamma(inputData_0);

});

test_function<690, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.5913350931,0.1140438775);
return cl::sycl::lgamma(inputData_0);

});

test_function<691, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1257135144,0.1920868015,0.5812494604);
return cl::sycl::lgamma(inputData_0);

});

test_function<692, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4517357123,0.6545709125,0.4580657155,0.10597976);
return cl::sycl::lgamma(inputData_0);

});

test_function<693, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2966038945,0.8455744399,0.7661471247,0.7011632616,0.3530136036,0.6771948129,0.3212659285,0.1265050945);
return cl::sycl::lgamma(inputData_0);

});

test_function<694, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4127861224,0.6542064262,0.2948949989,0.8906817686,0.3527018489,0.5997369332,0.1842327955,0.1707904531,0.7408906421,0.5783037288,0.1849590333,0.4132453279,0.6024834943,0.7431870174,0.4506952134,0.4668114973);
return cl::sycl::lgamma(inputData_0);

});

test_function<695, float>(
[=](){
float inputData_0(0.800707137);
int * inputData_1 = new int(5763);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<696, double>(
[=](){
double inputData_0(0.7171776444);
int * inputData_1 = new int(3177);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<697, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6538371511);
int * inputData_1 = new int(20114);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<698, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.76699898,0.2473593776);
cl::sycl::int2 * inputData_1 = new cl::sycl::int2(13413,14003);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<699, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4465313667,0.3429316102,0.265469087);
cl::sycl::int3 * inputData_1 = new cl::sycl::int3(28917,12172,27706);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<700, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6113715959,0.3477863923,0.7257965091,0.7925654775);
cl::sycl::int4 * inputData_1 = new cl::sycl::int4(20694,2891,-51,20939);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<701, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6652681538,0.1669572479,0.707328583,0.7543528594,0.5040595487,0.2777009793,0.4621556592,0.4975987546);
cl::sycl::int8 * inputData_1 = new cl::sycl::int8(21326,-287,23900,-4963,-21129,-5908,19616,7180);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<702, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8305592811,0.6674192973,0.657303431,0.8387349926,0.6751190873,0.5356305015,0.7934861345,0.8365010154,0.8412157676,0.2810619749,0.4997447301,0.6109063919,0.1398971365,0.4307825445,0.583764724,0.772606167);
cl::sycl::int16 * inputData_1 = new cl::sycl::int16(-30015,-11718,-12105,-22076,-11455,32202,-614,-27975,18723,31840,27627,7211,-6394,21040,4797,-21078);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<703, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6314988256,0.4803493864);
cl::sycl::int2 * inputData_1 = new cl::sycl::int2(13025,-19269);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<704, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.3320995783,0.3437169906,0.3322720528);
cl::sycl::int3 * inputData_1 = new cl::sycl::int3(13334,-9765,10842);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<705, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1122347509,0.1199973339,0.4746179276,0.8333770749);
cl::sycl::int4 * inputData_1 = new cl::sycl::int4(-29687,-27149,32756,-12385);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<706, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1591567035,0.1500139583,0.1640804409,0.2791588951,0.5704480506,0.1276212978,0.7721134641,0.8532438476);
cl::sycl::int8 * inputData_1 = new cl::sycl::int8(-10902,21627,7101,-31204,21536,31137,15300,14961);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<707, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4233267742,0.784268514,0.1207476153,0.3847533843,0.7630539079,0.2365341212,0.2521964037,0.1421751368,0.1233614103,0.8095000265,0.716276989,0.5351241423,0.7758176941,0.2213600671,0.88807804,0.6776688004);
cl::sycl::int16 * inputData_1 = new cl::sycl::int16(20887,-15879,-31513,7362,16378,28051,-23001,-18079,7411,20634,-13305,-7123,12887,-27657,-1537,-15204);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<708, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1268441131,0.3926619803);
cl::sycl::int2 * inputData_1 = new cl::sycl::int2(-13253,25861);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<709, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5522533134,0.2867161018,0.3456464489);
cl::sycl::int3 * inputData_1 = new cl::sycl::int3(1611,32019,-14816);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<710, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4022665708,0.1896187764,0.1615917111,0.2577029644);
cl::sycl::int4 * inputData_1 = new cl::sycl::int4(9792,12101,21185,21411);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<711, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3193500212,0.3972068766,0.8561632573,0.519369426,0.616876569,0.6617064871,0.8382721947,0.4281026734);
cl::sycl::int8 * inputData_1 = new cl::sycl::int8(18030,-3633,-10351,15531,7470,-12461,5746,-25865);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<712, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8320685448,0.1252454277,0.1570411127,0.6568262917,0.7279969587,0.8470510218,0.4381679574,0.8526873727,0.501962716,0.1728462904,0.3364545984,0.4963965623,0.8049628852,0.1477393646,0.5337825218,0.5399315474);
cl::sycl::int16 * inputData_1 = new cl::sycl::int16(23967,-8455,-1704,10785,-17924,8911,-3311,-17013,-16385,6079,11666,-14187,-7466,30532,-10123,5997);
return cl::sycl::lgamma_r(inputData_0,inputData_1);
delete inputData_1;

});

test_function<713, float>(
[=](){
float inputData_0(0.4655666478);
return cl::sycl::log(inputData_0);

});

test_function<714, double>(
[=](){
double inputData_0(0.4318021247);
return cl::sycl::log(inputData_0);

});

test_function<715, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.2848299863);
return cl::sycl::log(inputData_0);

});

test_function<716, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3989315239,0.7638392726);
return cl::sycl::log(inputData_0);

});

test_function<717, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4631627328,0.8972729604,0.8628677209);
return cl::sycl::log(inputData_0);

});

test_function<718, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3076177335,0.2039215549,0.6650805729,0.1289094625);
return cl::sycl::log(inputData_0);

});

test_function<719, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2563242645,0.3339061037,0.290263093,0.4670009739,0.8618943365,0.4896450266,0.356774693,0.2441639382);
return cl::sycl::log(inputData_0);

});

test_function<720, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.502638436,0.1027021003,0.2767079792,0.5305021339,0.7926654886,0.3984157849,0.3661592259,0.3479824137,0.7908899286,0.5112058148,0.8191566351,0.6392489635,0.3052968221,0.4099009888,0.1673962452,0.1461109519);
return cl::sycl::log(inputData_0);

});

test_function<721, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.843419778,0.5685515818);
return cl::sycl::log(inputData_0);

});

test_function<722, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7572591193,0.2822215193,0.5483088434);
return cl::sycl::log(inputData_0);

});

test_function<723, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5314734979,0.2020112192,0.7504107334,0.8523294406);
return cl::sycl::log(inputData_0);

});

test_function<724, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8373013953,0.2379260848,0.5672778874,0.8517859759,0.8923360089,0.4974860406,0.8088311095,0.7893521759);
return cl::sycl::log(inputData_0);

});

test_function<725, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8026991918,0.6429156585,0.8499392324,0.6446955523,0.1958509433,0.2681012382,0.2765018591,0.2239653187,0.5127045541,0.3448493388,0.3360225451,0.3745489371,0.208016303,0.482583152,0.8106565664,0.5096329106);
return cl::sycl::log(inputData_0);

});

test_function<726, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7102268841,0.4113343077);
return cl::sycl::log(inputData_0);

});

test_function<727, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8930638803,0.3785687783,0.3455582885);
return cl::sycl::log(inputData_0);

});

test_function<728, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6428067871,0.6595894184,0.4014314976,0.427132432);
return cl::sycl::log(inputData_0);

});

test_function<729, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2687865728,0.5920724931,0.3044213448,0.517667596,0.1401709775,0.4852868115,0.7919986668,0.4300266422);
return cl::sycl::log(inputData_0);

});

test_function<730, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5779092666,0.2393306019,0.1122251963,0.3016164058,0.2407630567,0.7545123817,0.3182126125,0.8730390908,0.6667694765,0.7778567218,0.1820899569,0.4255067631,0.3645733163,0.26448165,0.8369455924,0.7267047003);
return cl::sycl::log(inputData_0);

});

test_function<731, float>(
[=](){
float inputData_0(0.2313655369);
return cl::sycl::log2(inputData_0);

});

test_function<732, double>(
[=](){
double inputData_0(0.5820855584);
return cl::sycl::log2(inputData_0);

});

test_function<733, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7587149136);
return cl::sycl::log2(inputData_0);

});

test_function<734, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4077985663,0.7591269758);
return cl::sycl::log2(inputData_0);

});

test_function<735, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5279189228,0.6895335683,0.750129411);
return cl::sycl::log2(inputData_0);

});

test_function<736, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.1780247673,0.7223918216,0.4620983578,0.7755405493);
return cl::sycl::log2(inputData_0);

});

test_function<737, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2092376696,0.3429588913,0.527777417,0.6224551827,0.3727630721,0.6434228027,0.8450676356,0.3683043824);
return cl::sycl::log2(inputData_0);

});

test_function<738, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1540904932,0.3906229525,0.1116300411,0.2177809843,0.6316503957,0.3212048345,0.2016706693,0.8296828403,0.2294939122,0.3729242664,0.5010346068,0.7901375291,0.7042432382,0.5720055702,0.8829357853,0.5115955631);
return cl::sycl::log2(inputData_0);

});

test_function<739, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2559298607,0.490956371);
return cl::sycl::log2(inputData_0);

});

test_function<740, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.398334987,0.4157361714,0.496468426);
return cl::sycl::log2(inputData_0);

});

test_function<741, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7571599331,0.5370598756,0.6621791798,0.8515925361);
return cl::sycl::log2(inputData_0);

});

test_function<742, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1967391402,0.6822221664,0.7330260242,0.5968898884,0.6859146909,0.3528611486,0.1780669794,0.2204437711);
return cl::sycl::log2(inputData_0);

});

test_function<743, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.7678271963,0.4959919425,0.2439096752,0.4677628588,0.5210422322,0.4845706649,0.1207892313,0.2494121861,0.5802250149,0.3724162063,0.7211777895,0.3477314589,0.7308158715,0.5519581636,0.3584470337,0.3420331158);
return cl::sycl::log2(inputData_0);

});

test_function<744, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4239818268,0.1590944212);
return cl::sycl::log2(inputData_0);

});

test_function<745, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7243488997,0.8346284665,0.6119503279);
return cl::sycl::log2(inputData_0);

});

test_function<746, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.571239408,0.7283205773,0.175334841,0.6974704869);
return cl::sycl::log2(inputData_0);

});

test_function<747, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6489915372,0.222669998,0.5522725842,0.7422123962,0.2936299147,0.163318083,0.1825348953,0.3457635249);
return cl::sycl::log2(inputData_0);

});

test_function<748, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4072536009,0.7673455019,0.4817759398,0.7109732151,0.2886042075,0.659383583,0.880842181,0.4309829461,0.78967308,0.1449335818,0.1622848033,0.1312649984,0.7879114125,0.7656202861,0.6965168658,0.8045942182);
return cl::sycl::log2(inputData_0);

});

test_function<749, float>(
[=](){
float inputData_0(0.3417678749);
return cl::sycl::log10(inputData_0);

});

test_function<750, double>(
[=](){
double inputData_0(0.718838282);
return cl::sycl::log10(inputData_0);

});

test_function<751, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8066248704);
return cl::sycl::log10(inputData_0);

});

test_function<752, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5806502035,0.7390959843);
return cl::sycl::log10(inputData_0);

});

test_function<753, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.767172934,0.62398161,0.7395278056);
return cl::sycl::log10(inputData_0);

});

test_function<754, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8161181006,0.2718156089,0.2655969688,0.5932107714);
return cl::sycl::log10(inputData_0);

});

test_function<755, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8051038884,0.7535395383,0.7534888071,0.2242962365,0.3111848339,0.8486573752,0.6797893523,0.6792150997);
return cl::sycl::log10(inputData_0);

});

test_function<756, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.775252219,0.6432425101,0.3344405808,0.8061851706,0.3648242407,0.8569877792,0.6179961143,0.2558785118,0.784422052,0.7358431206,0.8061172414,0.6175082888,0.2159163315,0.1587883589,0.2055549516,0.5230809637);
return cl::sycl::log10(inputData_0);

});

test_function<757, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.486739232,0.5679894033);
return cl::sycl::log10(inputData_0);

});

test_function<758, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4561747131,0.6724203076,0.5313782224);
return cl::sycl::log10(inputData_0);

});

test_function<759, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.6202405185,0.7846337878,0.1685132838,0.7795591927);
return cl::sycl::log10(inputData_0);

});

test_function<760, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2902164031,0.8758904272,0.4659065463,0.4333416495,0.87891077,0.5452418877,0.1352381815,0.8139656366);
return cl::sycl::log10(inputData_0);

});

test_function<761, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3200362838,0.421133427,0.2155900611,0.8959221973,0.6134630728,0.8234691492,0.5958468462,0.7906268275,0.7987582296,0.8660542539,0.7080133449,0.8622007558,0.633365393,0.6922408297,0.8405884324,0.8540789848);
return cl::sycl::log10(inputData_0);

});

test_function<762, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8828339969,0.1032406642);
return cl::sycl::log10(inputData_0);

});

test_function<763, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5678599066,0.8683207098,0.6192644479);
return cl::sycl::log10(inputData_0);

});

test_function<764, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2536125317,0.2230965031,0.3689725125,0.5221793944);
return cl::sycl::log10(inputData_0);

});

test_function<765, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4266707306,0.3953512883,0.170477863,0.6151198196,0.4105600841,0.5086890419,0.1416621034,0.8960628214);
return cl::sycl::log10(inputData_0);

});

test_function<766, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3352137047,0.6686852219,0.7415605419,0.2358068055,0.3986631033,0.3821581679,0.2372578767,0.6253139164,0.703464533,0.620284742,0.6893536165,0.1747997291,0.4321398566,0.498357222,0.8747247369,0.4630223386);
return cl::sycl::log10(inputData_0);

});

test_function<767, float>(
[=](){
float inputData_0(0.5313896399);
return cl::sycl::log1p(inputData_0);

});

test_function<768, double>(
[=](){
double inputData_0(0.5199706609);
return cl::sycl::log1p(inputData_0);

});

test_function<769, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.2426732168);
return cl::sycl::log1p(inputData_0);

});

test_function<770, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1000186248,0.8128802714);
return cl::sycl::log1p(inputData_0);

});

test_function<771, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8451332456,0.1139507978,0.2229823125);
return cl::sycl::log1p(inputData_0);

});

test_function<772, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8045428236,0.5565702918,0.1685132712,0.51796536);
return cl::sycl::log1p(inputData_0);

});

test_function<773, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7429862328,0.3078871151,0.4404361672,0.6901761226,0.4139551919,0.1033745154,0.3278711677,0.5418952818);
return cl::sycl::log1p(inputData_0);

});

test_function<774, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6503505188,0.4324342466,0.3540391415,0.6637768168,0.8801792088,0.7934740902,0.4948211296,0.2021798916,0.6284566106,0.2680938736,0.2816177196,0.6024163814,0.2785495396,0.6697527003,0.5033969122,0.4465411919);
return cl::sycl::log1p(inputData_0);

});

test_function<775, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7429645648,0.469920122);
return cl::sycl::log1p(inputData_0);

});

test_function<776, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6766901522,0.5956287263,0.6954475139);
return cl::sycl::log1p(inputData_0);

});

test_function<777, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.8137932326,0.2709948254,0.6423339119,0.103562097);
return cl::sycl::log1p(inputData_0);

});

test_function<778, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2727832876,0.693030567,0.3460072803,0.2738858106,0.5459261563,0.8100674534,0.5326637348,0.2704971917);
return cl::sycl::log1p(inputData_0);

});

test_function<779, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3345791285,0.2037247959,0.3647242482,0.1402489551,0.623209149,0.5247158832,0.1888748987,0.1375173874,0.217491784,0.7886238274,0.8703478492,0.3440579237,0.4528886086,0.2525254941,0.6815221704,0.5073422227);
return cl::sycl::log1p(inputData_0);

});

test_function<780, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.109856819,0.2866218159);
return cl::sycl::log1p(inputData_0);

});

test_function<781, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1064265703,0.4500802056,0.7969391244);
return cl::sycl::log1p(inputData_0);

});

test_function<782, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8045369065,0.5822108498,0.8676600825,0.829127281);
return cl::sycl::log1p(inputData_0);

});

test_function<783, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5084250128,0.4146742219,0.3523273176,0.2335175283,0.5406157741,0.4790731261,0.1080698705,0.1270694572);
return cl::sycl::log1p(inputData_0);

});

test_function<784, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2738052316,0.3078253404,0.4599454193,0.8218215752,0.8388893796,0.3158141904,0.6428342378,0.5448804226,0.8201600698,0.3420123277,0.4553167237,0.5759149763,0.4355867273,0.4345164978,0.2604749389,0.6933920942);
return cl::sycl::log1p(inputData_0);

});

test_function<785, float>(
[=](){
float inputData_0(0.7509620883);
return cl::sycl::logb(inputData_0);

});

test_function<786, double>(
[=](){
double inputData_0(0.6210649768);
return cl::sycl::logb(inputData_0);

});

test_function<787, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5502360299);
return cl::sycl::logb(inputData_0);

});

test_function<788, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8043045918,0.7880933507);
return cl::sycl::logb(inputData_0);

});

test_function<789, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.115847078,0.6433024589,0.1165000005);
return cl::sycl::logb(inputData_0);

});

test_function<790, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4256768486,0.2607202513,0.2639933725,0.4485331661);
return cl::sycl::logb(inputData_0);

});

test_function<791, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1366154544,0.6796455497,0.4931221034,0.7981701631,0.3185728216,0.8739199973,0.8703885393,0.406290326);
return cl::sycl::logb(inputData_0);

});

test_function<792, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2234515704,0.2000536741,0.1550550948,0.1337476272,0.7947177506,0.391299841,0.4739711752,0.4488243936,0.7415363562,0.4208763901,0.3856521056,0.6429182142,0.4069906651,0.6261679927,0.4664931655,0.5545986748);
return cl::sycl::logb(inputData_0);

});

test_function<793, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4220553909,0.3026731839);
return cl::sycl::logb(inputData_0);

});

test_function<794, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.854807303,0.5709781594,0.6088726412);
return cl::sycl::logb(inputData_0);

});

test_function<795, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3292739104,0.6088048438,0.5421214983,0.56509801);
return cl::sycl::logb(inputData_0);

});

test_function<796, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.5312946344,0.3059921392,0.6235613738,0.164587509,0.3487989273,0.34289013,0.4475180842,0.1021893769);
return cl::sycl::logb(inputData_0);

});

test_function<797, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2637285621,0.7831942218,0.7335635853,0.1106184885,0.5853847067,0.5120545644,0.1779380525,0.6933730409,0.3897865577,0.4057717021,0.392671345,0.5866484447,0.7669632696,0.7739474249,0.3500544008,0.8180550576);
return cl::sycl::logb(inputData_0);

});

test_function<798, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.1469167119,0.5983674199);
return cl::sycl::logb(inputData_0);

});

test_function<799, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6603934207,0.8645993152,0.4337731428);
return cl::sycl::logb(inputData_0);

});

test_function<800, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4316966434,0.6021590544,0.3721337636,0.8749293787);
return cl::sycl::logb(inputData_0);

});

test_function<801, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3782634638,0.217514189,0.3716025477,0.3036593577,0.8988132823,0.8106380214,0.6872233727,0.3193685868);
return cl::sycl::logb(inputData_0);

});

test_function<802, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3253447585,0.221495303,0.5678103709,0.2765876747,0.1836470975,0.8207631347,0.3152789262,0.6008058363,0.2035057831,0.3277104016,0.2107461747,0.3208444002,0.10752265,0.3894289394,0.8510878185,0.6007438845);
return cl::sycl::logb(inputData_0);

});

test_function<803, float>(
[=](){
float inputData_0(0.859345172);
float inputData_1(0.4610178032);
float inputData_2(0.1476207213);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<804, double>(
[=](){
double inputData_0(0.222842836);
double inputData_1(0.4897679401);
double inputData_2(0.2823252698);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<805, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1733211404);
cl::sycl::half inputData_1(0.3101154357);
cl::sycl::half inputData_2(0.4745723564);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<806, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4365040648,0.6995960496);
cl::sycl::float2 inputData_1(0.8524418448,0.5345792364);
cl::sycl::float2 inputData_2(0.1727684067,0.193923182);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<807, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4226749019,0.1687057188,0.5346577441);
cl::sycl::float3 inputData_1(0.6676058999,0.3317765823,0.5174436688);
cl::sycl::float3 inputData_2(0.8832645356,0.2331781906,0.8435211746);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<808, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4318220912,0.8042782023,0.399119188,0.8345225112);
cl::sycl::float4 inputData_1(0.805001687,0.7645372078,0.3634646784,0.3074208451);
cl::sycl::float4 inputData_2(0.1786970692,0.7235697948,0.4202673873,0.471943651);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<809, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1762680273,0.1554595168,0.7732766679,0.5469244137,0.2322096673,0.251522169,0.3881329641,0.3126502397);
cl::sycl::float8 inputData_1(0.2091839659,0.1060792786,0.8641429744,0.4554378944,0.6290315673,0.5524461526,0.6613594927,0.8125917305);
cl::sycl::float8 inputData_2(0.3632256578,0.6256427584,0.1118820881,0.3373805831,0.1209190812,0.4548047834,0.7855771132,0.8145481195);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<810, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6367153448,0.7767582693,0.4782604979,0.1658403339,0.4719747464,0.6745955434,0.2801646368,0.2023162102,0.8877550988,0.4571058874,0.6186614088,0.8938208124,0.6585574266,0.4313054313,0.5321917731,0.497111675);
cl::sycl::float16 inputData_1(0.4608729322,0.5541457947,0.3472903733,0.1918082422,0.8703355078,0.8230282262,0.6501475002,0.4529857997,0.8122752284,0.4722790351,0.6142871262,0.8402488026,0.2426115402,0.7125261891,0.8470811585,0.4289525142);
cl::sycl::float16 inputData_2(0.5488455396,0.3647702763,0.8448719841,0.3684296498,0.2946308343,0.413637657,0.1774659,0.2274770466,0.1775134859,0.6691759119,0.3656665431,0.1298943214,0.6527196116,0.339172836,0.7484457032,0.8343068729);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<811, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.5682385782,0.845665651);
cl::sycl::double2 inputData_1(0.8384761167,0.4141980259);
cl::sycl::double2 inputData_2(0.6701842997,0.1159174935);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<812, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6061536299,0.8748290265,0.555652172);
cl::sycl::double3 inputData_1(0.3376071296,0.5317640099,0.4055469888);
cl::sycl::double3 inputData_2(0.3363041406,0.4592108232,0.2803194302);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<813, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4791382254,0.2907872785,0.114431519,0.1177762111);
cl::sycl::double4 inputData_1(0.4389579781,0.3388053178,0.1313680715,0.7532456445);
cl::sycl::double4 inputData_2(0.1784771672,0.3465559928,0.7578901309,0.6880934306);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<814, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4427909091,0.7134949646,0.3075477923,0.4374863063,0.7492251665,0.3492899631,0.743078485,0.4495373979);
cl::sycl::double8 inputData_1(0.6628505744,0.2434145073,0.4637370842,0.2072834974,0.5507594477,0.2033578351,0.8540496217,0.8381055431);
cl::sycl::double8 inputData_2(0.3216008848,0.3534394152,0.1034281206,0.8963690642,0.56141694,0.3543734701,0.8967486027,0.3131741142);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<815, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.621305105,0.4419565552,0.196112604,0.3055899601,0.6642592263,0.2853718078,0.3510655221,0.2439428379,0.1299170906,0.1419217706,0.2563530855,0.1757820828,0.6496944609,0.8076624926,0.4961255615,0.3452506673);
cl::sycl::double16 inputData_1(0.5035476944,0.6727103108,0.6053615595,0.227871745,0.7183175221,0.8191057545,0.5019885536,0.6555034492,0.5653068419,0.2328585175,0.4077267301,0.8712307384,0.1429281386,0.7114678658,0.5949602808,0.5550117178);
cl::sycl::double16 inputData_2(0.4942570981,0.2200387919,0.5877689292,0.8553146095,0.5062705812,0.7791243249,0.6476176386,0.2378785287,0.7569237699,0.235444184,0.8541451193,0.2282140113,0.4291720086,0.7611340691,0.3885141947,0.5931201245);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<816, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6304808582,0.3322493685);
cl::sycl::half2 inputData_1(0.4240737641,0.5680712188);
cl::sycl::half2 inputData_2(0.5005308849,0.8502750827);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<817, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6791320649,0.1686331809,0.6479936164);
cl::sycl::half3 inputData_1(0.5738306193,0.5211599471,0.3597483357);
cl::sycl::half3 inputData_2(0.721827768,0.8272020221,0.3594264983);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<818, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4869654157,0.6731233691,0.7087610298,0.3260883019);
cl::sycl::half4 inputData_1(0.5596310071,0.8001769515,0.1570406858,0.7618440811);
cl::sycl::half4 inputData_2(0.6612227279,0.7617676627,0.8108452932,0.6951535239);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<819, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6328230803,0.1757904049,0.5570086669,0.1272087771,0.4493718699,0.1675646336,0.5866063591,0.1261269988);
cl::sycl::half8 inputData_1(0.6892466732,0.6331506981,0.167366996,0.2048530263,0.7187299943,0.6958375907,0.8265648774,0.381011713);
cl::sycl::half8 inputData_2(0.2656014489,0.8637709035,0.4784825848,0.5531924959,0.1726860933,0.7453068168,0.2784628144,0.8602793426);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<820, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4703319721,0.2263824967,0.5826468728,0.2999839076,0.1905334908,0.4888661318,0.5032696585,0.6690927926,0.8203107023,0.2738457407,0.7601505155,0.462838695,0.2007796632,0.7007962097,0.4088284554,0.5481365906);
cl::sycl::half16 inputData_1(0.1025756508,0.2978584143,0.5633227132,0.1247642444,0.2052352851,0.5197274632,0.2763742846,0.663446462,0.8141191778,0.2316823267,0.3973376167,0.1572911525,0.8516773295,0.6699159429,0.2150855166,0.2860468118);
cl::sycl::half16 inputData_2(0.591150483,0.3392906925,0.2260352298,0.7681511862,0.3207272965,0.6607352369,0.4176032089,0.2908517178,0.7918155744,0.6627116513,0.4304864327,0.6016204293,0.2094344461,0.8800451474,0.701455688,0.7587036008);
return cl::sycl::mad(inputData_0,inputData_1,inputData_2);

});

test_function<821, float>(
[=](){
float inputData_0(0.2649811695);
float inputData_1(0.7717266562);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<822, double>(
[=](){
double inputData_0(0.3544012944);
double inputData_1(0.3723471576);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<823, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6340065456);
cl::sycl::half inputData_1(0.7990337129);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<824, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6459830703,0.3322223019);
cl::sycl::float2 inputData_1(0.7596505969,0.2079543927);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<825, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8541381362,0.6645692716,0.5492339482);
cl::sycl::float3 inputData_1(0.2672135256,0.4685926433,0.4389763607);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<826, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5231728263,0.7121933054,0.758866487,0.4858918639);
cl::sycl::float4 inputData_1(0.5924164533,0.7501581856,0.6715069772,0.5786848607);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<827, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4218940683,0.14960938,0.7654232434,0.2619664709,0.5554286562,0.2688010781,0.1030304419,0.4226031216);
cl::sycl::float8 inputData_1(0.532616014,0.3084933011,0.140493833,0.7166817445,0.4869172505,0.1434283524,0.2099325411,0.6620113619);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<828, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8882115286,0.8503748334,0.8692898235,0.4650416632,0.6271581594,0.1669678104,0.5705128769,0.6901603623,0.8181236029,0.16610136,0.4310833381,0.3172532273,0.1923139175,0.2875006457,0.7404652743,0.5441016638);
cl::sycl::float16 inputData_1(0.537993468,0.2845744216,0.7604763989,0.4876522033,0.7676361979,0.4507058215,0.1013906729,0.281351145,0.5871301741,0.2334823197,0.4876106546,0.578475722,0.412526084,0.1228484328,0.1495835547,0.747644729);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<829, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4960309124,0.1761668329);
cl::sycl::double2 inputData_1(0.2640121424,0.357030441);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<830, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1441737279,0.5855916299,0.5006181335);
cl::sycl::double3 inputData_1(0.7223026853,0.4812409551,0.535600619);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<831, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7431451668,0.7616448145,0.3610952177,0.1781380991);
cl::sycl::double4 inputData_1(0.8180417737,0.7167162851,0.1156940476,0.1480234066);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<832, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7908938231,0.4226242801,0.1043565838,0.5527464709,0.8970216209,0.8867034208,0.2953895972,0.6905470486);
cl::sycl::double8 inputData_1(0.2483782977,0.7556895828,0.592032517,0.7534575836,0.6499440696,0.7650699762,0.2099816514,0.3162371625);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<833, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2494549912,0.1989856163,0.2776309618,0.1667110876,0.4594281225,0.3761679986,0.1070969709,0.2564067444,0.6618556097,0.134735995,0.1148789958,0.7852489542,0.8320345436,0.1295555379,0.307887858,0.3867279821);
cl::sycl::double16 inputData_1(0.5647808141,0.6017894545,0.8715334795,0.3163167794,0.4212870201,0.4009539685,0.7708611984,0.2545141013,0.827435084,0.6296592218,0.4913618794,0.6756157808,0.2573741975,0.1352428751,0.792368906,0.3853342829);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<834, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7987515876,0.5987284526);
cl::sycl::half2 inputData_1(0.5656803934,0.8060818789);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<835, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.3430750067,0.4065664783,0.7758392092);
cl::sycl::half3 inputData_1(0.825968361,0.7286293039,0.835170512);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<836, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5403608577,0.2386338765,0.778588363,0.8066628335);
cl::sycl::half4 inputData_1(0.3433533228,0.6334531979,0.1150444968,0.8408808825);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<837, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3777078635,0.5511414073,0.6472997865,0.7587512107,0.6655813208,0.3616248709,0.8445238924,0.2783367558);
cl::sycl::half8 inputData_1(0.6975318603,0.5056805327,0.3027545632,0.5102114866,0.351934169,0.51668946,0.2668443002,0.3115313193);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<838, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2956913648,0.7892782617,0.413227435,0.7855385276,0.4030072622,0.3563621779,0.6312862124,0.4236529249,0.7549669339,0.6177485081,0.6598485118,0.8758560316,0.7759463077,0.3733094265,0.4675290043,0.7928094753);
cl::sycl::half16 inputData_1(0.814227074,0.5311083167,0.5365584483,0.1666243568,0.270204098,0.1946516291,0.3602461609,0.112736698,0.5061808463,0.1004768619,0.3877576559,0.1817052996,0.2945329332,0.6312998428,0.3115877691,0.104619143);
return cl::sycl::maxmag(inputData_0,inputData_1);

});

test_function<839, float>(
[=](){
float inputData_0(0.760462161);
float inputData_1(0.5312184785);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<840, double>(
[=](){
double inputData_0(0.8060735922);
double inputData_1(0.3148436062);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<841, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.355844229);
cl::sycl::half inputData_1(0.124675047);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<842, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5060848044,0.815811584);
cl::sycl::float2 inputData_1(0.1014288125,0.5620177531);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<843, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2922667412,0.6494889419,0.84037187);
cl::sycl::float3 inputData_1(0.4874857801,0.8014099844,0.4085644545);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<844, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2741532578,0.4381522164,0.6268808846,0.7418813353);
cl::sycl::float4 inputData_1(0.6415900845,0.1574740609,0.7363906783,0.2060167934);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<845, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1920480318,0.2698782927,0.2393692719,0.2545313653,0.2604065296,0.5282399796,0.7882428591,0.520328398);
cl::sycl::float8 inputData_1(0.6259145256,0.7258377785,0.8920853476,0.3194338189,0.8536360664,0.2564322084,0.254236825,0.7367054989);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<846, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1631692124,0.8209833253,0.8224885918,0.6617034542,0.6572937658,0.4836449697,0.15659307,0.3766557683,0.7085819125,0.5701540997,0.344677041,0.2377587432,0.268528855,0.5273368397,0.6600250231,0.8554323642);
cl::sycl::float16 inputData_1(0.7245064661,0.664890436,0.6062534304,0.8984024605,0.5875942756,0.7253263487,0.4216690823,0.2114766872,0.3937433705,0.7338826687,0.5064840267,0.4964480226,0.2664267391,0.3899559526,0.6995126283,0.4337445215);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<847, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2405122086,0.8670452535);
cl::sycl::double2 inputData_1(0.8464628331,0.788469682);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<848, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4914200381,0.8893768551,0.1570458388);
cl::sycl::double3 inputData_1(0.6239425881,0.1256939101,0.3755808343);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<849, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5478923972,0.486377584,0.4491441776,0.8499018822);
cl::sycl::double4 inputData_1(0.806699703,0.8124964087,0.7240594812,0.2612403981);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<850, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6289921763,0.3959696376,0.8110676841,0.37551609,0.577414312,0.7615566322,0.3630858132,0.1970406337);
cl::sycl::double8 inputData_1(0.385486949,0.3163142199,0.380841192,0.881392414,0.7875972588,0.5676459977,0.7718961143,0.504359551);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<851, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.844997972,0.3025987289,0.8207000742,0.5512689773,0.1887458157,0.4785577243,0.3514943115,0.2771438554,0.7295488682,0.2346453394,0.6002408099,0.5428434582,0.5383779016,0.2496919959,0.661021901,0.4785898058);
cl::sycl::double16 inputData_1(0.5464461907,0.8426804564,0.8949118715,0.5877757947,0.5018777476,0.8822083753,0.1264927817,0.2315247122,0.6638714125,0.3514784478,0.4981898749,0.5983258219,0.4226095202,0.5728531953,0.3542580315,0.227821453);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<852, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6104277116,0.2304394995);
cl::sycl::half2 inputData_1(0.235036975,0.7798971144);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<853, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7203508247,0.1799031353,0.6891248724);
cl::sycl::half3 inputData_1(0.7584521569,0.7604297155,0.5984408452);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<854, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5303850093,0.5877568686,0.3563035973,0.5641022177);
cl::sycl::half4 inputData_1(0.73174927,0.4556391867,0.1868006253,0.7711429797);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<855, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3339728256,0.4535342369,0.6267114511,0.7428825854,0.611900707,0.4088665215,0.1850462786,0.8741174646);
cl::sycl::half8 inputData_1(0.4979936807,0.4533930729,0.7098619533,0.6929858039,0.6635717101,0.8837883331,0.7658146304,0.6023293397);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<856, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2025454866,0.7640629445,0.512573732,0.5679624238,0.5455357876,0.2777336313,0.5453740681,0.4758643784,0.8177001666,0.8528774443,0.8864066638,0.5234662549,0.3330154747,0.1755410957,0.1170318275,0.3211993362);
cl::sycl::half16 inputData_1(0.1316273325,0.6740681723,0.4824188533,0.7558564397,0.7669927296,0.4781374678,0.4854635869,0.5241353711,0.6240791898,0.4807602073,0.8678930315,0.3204572417,0.1304807596,0.8775610451,0.1701219751,0.2504327608);
return cl::sycl::minmag(inputData_0,inputData_1);

});

test_function<857, float>(
[=](){
float inputData_0(0.3191190221);
float * inputData_1 = new float(0.7727820614);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<858, double>(
[=](){
double inputData_0(0.3529388129);
double * inputData_1 = new double(0.624722031);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<859, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7770386468);
cl::sycl::half * inputData_1 = new cl::sycl::half(0.5346414709);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<860, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.282458587,0.510705257);
cl::sycl::float2 * inputData_1 = new cl::sycl::float2(0.1183190511,0.392917347);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<861, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2441397745,0.4217382817,0.1480865368);
cl::sycl::float3 * inputData_1 = new cl::sycl::float3(0.6360728556,0.6514408014,0.7453915284);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<862, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5767240106,0.198666634,0.216470065,0.8873984317);
cl::sycl::float4 * inputData_1 = new cl::sycl::float4(0.6566445137,0.6299732259,0.4899204401,0.3589899523);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<863, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7800873075,0.8517652164,0.2997718208,0.3750458652,0.354680071,0.8880666743,0.5607536459,0.6939558469);
cl::sycl::float8 * inputData_1 = new cl::sycl::float8(0.3668227183,0.3613520315,0.6959032747,0.7522562409,0.2143656399,0.8003528838,0.6213816803,0.343119707);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<864, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2563610913,0.8574300748,0.5174363778,0.4513556136,0.3446387351,0.6282520471,0.1071312953,0.8290932494,0.5929768372,0.1004923097,0.1493352293,0.5799833415,0.2924551816,0.3651499531,0.7988438448,0.729276145);
cl::sycl::float16 * inputData_1 = new cl::sycl::float16(0.512340443,0.8014488154,0.5653791537,0.3241954268,0.8808340558,0.1474513876,0.1551777658,0.8168744258,0.5406979462,0.8009991174,0.7347273027,0.6860934035,0.8648447582,0.3756660568,0.6006157552,0.8249043503);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<865, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.2967856741,0.1048738804);
cl::sycl::double2 * inputData_1 = new cl::sycl::double2(0.5365275522,0.4422338053);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<866, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4179949503,0.4077462164,0.6408209152);
cl::sycl::double3 * inputData_1 = new cl::sycl::double3(0.5213293016,0.4534913436,0.6430631764);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<867, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3871814077,0.4922728773,0.3677414768,0.5221229577);
cl::sycl::double4 * inputData_1 = new cl::sycl::double4(0.8248509787,0.1809190365,0.5995210111,0.2436751688);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<868, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8386549034,0.7569109554,0.4240631482,0.8624102892,0.1460135385,0.5876486669,0.3832495521,0.7323565302);
cl::sycl::double8 * inputData_1 = new cl::sycl::double8(0.8958971826,0.3326003013,0.5926537066,0.6221147043,0.3501904057,0.3387464562,0.1809618674,0.8567889224);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<869, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4952687683,0.6611516324,0.7292072024,0.3142919562,0.2556176484,0.3984165802,0.5878888779,0.5230203151,0.1923988831,0.8863240289,0.8913580798,0.7785449344,0.2362160148,0.4353934931,0.7449238516,0.8906564616);
cl::sycl::double16 * inputData_1 = new cl::sycl::double16(0.5932261792,0.2424111937,0.8234385515,0.3367310655,0.7521311615,0.6283858196,0.3756758013,0.5857690293,0.4006791925,0.5808362883,0.7262956134,0.674558019,0.5654880489,0.6755333708,0.7428307123,0.4021013971);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<870, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.2641380849,0.6529109049);
cl::sycl::half2 * inputData_1 = new cl::sycl::half2(0.2668044656,0.626562404);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<871, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6893982099,0.3698330373,0.2824916336);
cl::sycl::half3 * inputData_1 = new cl::sycl::half3(0.6986421898,0.8801160722,0.7029880387);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<872, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7975005671,0.4813947317,0.2996946857,0.1876749404);
cl::sycl::half4 * inputData_1 = new cl::sycl::half4(0.2198747917,0.3170872019,0.7256868024,0.2029813455);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<873, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8988996853,0.1335282742,0.555732743,0.5602647883,0.5012982689,0.2327858053,0.471660798,0.7784800353);
cl::sycl::half8 * inputData_1 = new cl::sycl::half8(0.3055008022,0.3950002708,0.3445859006,0.4122116025,0.5897912766,0.5177908305,0.8660337175,0.4753192231);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<874, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3434489444,0.1145509387,0.3348812795,0.7656586435,0.8965548629,0.558058778,0.4859664797,0.4089721601,0.3653937562,0.380048374,0.7472463973,0.3759875197,0.5610994843,0.2141749949,0.8041387332,0.3457217698);
cl::sycl::half16 * inputData_1 = new cl::sycl::half16(0.4659693965,0.3010855622,0.2315336428,0.8513385207,0.6629788904,0.8421072518,0.1325005173,0.5420365389,0.5797117721,0.8713133733,0.6731729771,0.6067660284,0.3145876825,0.8165675976,0.5730442128,0.1007841567);
return cl::sycl::modf(inputData_0,inputData_1);
delete inputData_1;

});

test_function<875, float>(
[=](){
unsigned int inputData_0(41907);
return cl::sycl::nan(inputData_0);

});

test_function<876, cl::sycl::float2>(
[=](){
cl::sycl::uint2 inputData_0(56381,6476);
return cl::sycl::nan(inputData_0);

});

test_function<877, cl::sycl::float3>(
[=](){
cl::sycl::uint3 inputData_0(12588,35645,2743);
return cl::sycl::nan(inputData_0);

});

test_function<878, cl::sycl::float4>(
[=](){
cl::sycl::uint4 inputData_0(25398,37885,16112,55762);
return cl::sycl::nan(inputData_0);

});

test_function<879, cl::sycl::float8>(
[=](){
cl::sycl::uint8 inputData_0(6496,52541,13726,54667,42960,48894,34049,53826);
return cl::sycl::nan(inputData_0);

});

test_function<880, cl::sycl::float16>(
[=](){
cl::sycl::uint16 inputData_0(10618,49402,1272,16448,5587,29627,18104,57834,2663,49947,57478,40612,12925,113,5643,54207);
return cl::sycl::nan(inputData_0);

});

test_function<881, double>(
[=](){
unsigned long int inputData_0(2759787404);
return cl::sycl::nan(inputData_0);

});

test_function<882, double>(
[=](){
unsigned long long int inputData_0(1994376607944073277);
return cl::sycl::nan(inputData_0);

});

test_function<883, cl::sycl::double2>(
[=](){
cl::sycl::ulong2 inputData_0(2758634439,1355987330);
return cl::sycl::nan(inputData_0);

});

test_function<884, cl::sycl::double3>(
[=](){
cl::sycl::ulong3 inputData_0(2690102885,770768145,3421225031);
return cl::sycl::nan(inputData_0);

});

test_function<885, cl::sycl::double4>(
[=](){
cl::sycl::ulong4 inputData_0(3026493551,2254607539,2588849666,1855560786);
return cl::sycl::nan(inputData_0);

});

test_function<886, cl::sycl::double8>(
[=](){
cl::sycl::ulong8 inputData_0(2204585127,2186327126,1437053372,1164934448,1639984974,469759186,704660,3239807469);
return cl::sycl::nan(inputData_0);

});

test_function<887, cl::sycl::double16>(
[=](){
cl::sycl::ulong16 inputData_0(3752661570,1480194649,1264535153,3178856065,3805220603,201672779,784180479,424464103,1541396496,3573015006,149270424,1424581841,4038619495,2458609351,1361084290,3725821525);
return cl::sycl::nan(inputData_0);

});

test_function<888, cl::sycl::double2>(
[=](){
cl::sycl::ulonglong2 inputData_0(1489377237086296447,5925785984464595076);
return cl::sycl::nan(inputData_0);

});

test_function<889, cl::sycl::double3>(
[=](){
cl::sycl::ulonglong3 inputData_0(9179160800590787390,11638889601609357592,2434160271507755190);
return cl::sycl::nan(inputData_0);

});

test_function<890, cl::sycl::double4>(
[=](){
cl::sycl::ulonglong4 inputData_0(10017889486061892910,7622509332469655053,6576185652128899781,8579477230558592049);
return cl::sycl::nan(inputData_0);

});

test_function<891, cl::sycl::double8>(
[=](){
cl::sycl::ulonglong8 inputData_0(14306706572938804579,10662565509919780003,12439670355792672329,1987437020109751758,8831118384766845619,8005280309944873398,3773146546575771502,17688941246882835084);
return cl::sycl::nan(inputData_0);

});

test_function<892, cl::sycl::double16>(
[=](){
cl::sycl::ulonglong16 inputData_0(14387254946208345514,9643348999196867261,14431344772925073643,15784351505856318893,1368481936904082541,565390085250520260,15627543936165021037,12676804996553490604,7802594814111162896,10179414252430648925,13529435358268779779,10489174560805472993,9150288657028054665,17070558288353573897,9321800431614250203,16683625454395587635);
return cl::sycl::nan(inputData_0);

});

test_function<893, float>(
[=](){
float inputData_0(0.283817916);
float inputData_1(0.2982278783);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<894, double>(
[=](){
double inputData_0(0.8669644229);
double inputData_1(0.3028616037);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<895, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.449685009);
cl::sycl::half inputData_1(0.3254715987);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<896, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1196784928,0.2082096732);
cl::sycl::float2 inputData_1(0.4979555679,0.5498350261);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<897, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5423320285,0.6310181886,0.8176528987);
cl::sycl::float3 inputData_1(0.4999856654,0.5895032564,0.5408276944);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<898, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7964184406,0.1140972,0.6780789889,0.1804956179);
cl::sycl::float4 inputData_1(0.7752154722,0.5601603334,0.5313058271,0.2211286391);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<899, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.554812775,0.6201190694,0.7140533048,0.7283800592,0.2133881666,0.6240699721,0.3076092301,0.5912437016);
cl::sycl::float8 inputData_1(0.8196633982,0.7707769726,0.4674007672,0.3243951031,0.6498378352,0.2811708304,0.6689709509,0.2651308074);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<900, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7681085806,0.1318599195,0.7235776313,0.7941675203,0.5713489435,0.2779783861,0.8321976539,0.7406091763,0.7804858077,0.6958705546,0.5189024186,0.6678111558,0.4944691734,0.5313599765,0.2735098066,0.3855043732);
cl::sycl::float16 inputData_1(0.4358263698,0.544157251,0.4422182781,0.8240883337,0.7276924355,0.7227012054,0.323724434,0.3325238278,0.6389988919,0.5616823691,0.1029457851,0.7690755904,0.3923028631,0.1995588265,0.7740801502,0.3833158129);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<901, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.5767224461,0.7264262109);
cl::sycl::double2 inputData_1(0.6626301966,0.5443741786);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<902, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2248296211,0.1145004952,0.572564049);
cl::sycl::double3 inputData_1(0.7719410187,0.8951204763,0.2120414369);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<903, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.8257216202,0.8249967289,0.3419149415,0.4720171964);
cl::sycl::double4 inputData_1(0.6779833928,0.2810092348,0.1492030711,0.21561759);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<904, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2829735255,0.3819153605,0.8238294383,0.7671262132,0.6274135385,0.8861289156,0.4866569288,0.6064699322);
cl::sycl::double8 inputData_1(0.3022434068,0.8693770194,0.4275033298,0.8504638221,0.1489214026,0.1565434066,0.7653208006,0.2657516756);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<905, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6368028779,0.8810592653,0.7715344497,0.4540571183,0.8784004026,0.6868436125,0.6426777388,0.8605719775,0.1286393233,0.2196041036,0.2766451924,0.7030248289,0.7853470857,0.13658011,0.5302185656,0.8461927712);
cl::sycl::double16 inputData_1(0.8137021216,0.8896555682,0.6308061246,0.3895143571,0.2731266169,0.5385431652,0.5630927813,0.7317269668,0.7057372648,0.3466437103,0.2777155635,0.8362419661,0.4879651704,0.242857127,0.2900512929,0.8243128082);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<906, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8769365586,0.8983938514);
cl::sycl::half2 inputData_1(0.1429122571,0.7727278642);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<907, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6186975148,0.3534760411,0.5297787331);
cl::sycl::half3 inputData_1(0.1371295097,0.8276958735,0.1126179512);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<908, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2479741666,0.1572181418,0.490580837,0.7272437605);
cl::sycl::half4 inputData_1(0.8991037998,0.5008124008,0.1469735278,0.4539547384);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<909, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2003214775,0.8445546672,0.1350076491,0.234870204,0.8124404981,0.8139325707,0.1765362678,0.6332150332);
cl::sycl::half8 inputData_1(0.6725142701,0.3557239811,0.5260499622,0.6182706807,0.4548376665,0.5559491529,0.2978457991,0.3139939471);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<910, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4005748599,0.4136265794,0.2442442732,0.8599836571,0.3341719083,0.8492869532,0.3094765485,0.2697491635,0.7860366358,0.2846829919,0.1055142646,0.7685377979,0.3578886443,0.1544600099,0.1079800648,0.3928842989);
cl::sycl::half16 inputData_1(0.627429232,0.139674391,0.5512960713,0.3172740173,0.2383447641,0.5806702485,0.5470363215,0.3018702955,0.170740628,0.7356726817,0.739349193,0.8935388172,0.8724269651,0.5228022568,0.8400057433,0.1506223486);
return cl::sycl::nextafter(inputData_0,inputData_1);

});

test_function<911, float>(
[=](){
float inputData_0(0.378279666);
float inputData_1(0.5952881531);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<912, double>(
[=](){
double inputData_0(0.7814675705);
double inputData_1(0.4437361744);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<913, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4586917186);
cl::sycl::half inputData_1(0.4271034998);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<914, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8859475042,0.3029632025);
cl::sycl::float2 inputData_1(0.7277856535,0.8201856622);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<915, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8320331977,0.8011078351,0.5317386838);
cl::sycl::float3 inputData_1(0.8116675231,0.5903656219,0.435187595);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<916, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6637131503,0.1148859103,0.2467086637,0.7906973644);
cl::sycl::float4 inputData_1(0.8226082485,0.8853495987,0.6707042593,0.1282447976);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<917, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7500054227,0.3588382761,0.6959478492,0.1099714162,0.110396184,0.3931363058,0.1595417929,0.7767774763);
cl::sycl::float8 inputData_1(0.272830365,0.5761660492,0.8384922456,0.377298412,0.5342466073,0.4351124036,0.1827255171,0.8478183537);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<918, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8002817493,0.2210439794,0.4405314005,0.1058457217,0.8642213236,0.5688514738,0.3667540789,0.100597568,0.2685576828,0.5597231659,0.7016082834,0.6708385133,0.5855378954,0.5510807386,0.4927176666,0.278343781);
cl::sycl::float16 inputData_1(0.4866482193,0.571990212,0.8272850188,0.4446377964,0.3360956247,0.4487814483,0.3010443526,0.491964922,0.1750308975,0.6598064962,0.1036605641,0.3142386042,0.5187097767,0.6140107238,0.8958874595,0.8951944808);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<919, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6746219102,0.4009477292);
cl::sycl::double2 inputData_1(0.7977305324,0.6256258429);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<920, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6433526497,0.3600065917,0.7560353294);
cl::sycl::double3 inputData_1(0.3904138895,0.3645680407,0.5721696652);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<921, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.149426089,0.1272735311,0.453049391,0.4273708433);
cl::sycl::double4 inputData_1(0.6814971642,0.3260304427,0.7734002417,0.8213220955);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<922, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7918307022,0.1719420961,0.7667775101,0.2895599159,0.7864572927,0.8810535346,0.3590442731,0.2314920812);
cl::sycl::double8 inputData_1(0.7336335266,0.7140490658,0.8055856011,0.7599469244,0.7483121994,0.1748952517,0.7973202756,0.1995499364);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<923, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.7474450535,0.8652343622,0.7373911253,0.4068550496,0.4089072476,0.3560396343,0.8487326385,0.5257926987,0.6678046178,0.347215726,0.6468515014,0.7532539347,0.875205107,0.235793712,0.5026043808,0.3368044392);
cl::sycl::double16 inputData_1(0.1832959226,0.411929533,0.7076830962,0.8995954803,0.3760038261,0.6763046016,0.8168908016,0.3992553664,0.5741652351,0.8860328283,0.6419366464,0.3771941965,0.1018025018,0.8672214179,0.6550432547,0.7743593217);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<924, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.2950749044,0.7726700881);
cl::sycl::half2 inputData_1(0.3343291402,0.638871736);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<925, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5527183369,0.4465594772,0.3021256436);
cl::sycl::half3 inputData_1(0.7109778589,0.6612226112,0.5610344611);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<926, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2676435443,0.8691262327,0.8409914625,0.2218303373);
cl::sycl::half4 inputData_1(0.465800554,0.5319191061,0.6990983533,0.1973984927);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<927, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.2451233716,0.7753623731,0.4578899456,0.239439015,0.6943758328,0.7897434219,0.225748626,0.1033407551);
cl::sycl::half8 inputData_1(0.6716214913,0.448692968,0.7747060052,0.136793102,0.5496402089,0.7836809862,0.4967347256,0.8575681876);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<928, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8418147911,0.6893734826,0.4675397384,0.7423720639,0.6804491488,0.1100808227,0.1895473532,0.3097347034,0.8056954717,0.4093573283,0.8342232342,0.1710106498,0.1957414862,0.4231313818,0.6957020519,0.6861440106);
cl::sycl::half16 inputData_1(0.2661684331,0.3497885943,0.192929133,0.74281967,0.2758764778,0.7909141602,0.412360876,0.1303217836,0.1829942592,0.6533503239,0.456749683,0.394040407,0.8744072474,0.8887318367,0.2319195431,0.6659882467);
return cl::sycl::pow(inputData_0,inputData_1);

});

test_function<929, float>(
[=](){
float inputData_0(0.3141789962);
int inputData_1(23334);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<930, double>(
[=](){
double inputData_0(0.7836557308);
int inputData_1(12275);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<931, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.3565747781);
int inputData_1(-13714);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<932, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8878794033,0.8578871582);
cl::sycl::int2 inputData_1(3173,-25242);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<933, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6496197753,0.2918023896,0.8348057056);
cl::sycl::int3 inputData_1(-5930,-23489,23937);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<934, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6940488412,0.2782740968,0.8492438061,0.6958029833);
cl::sycl::int4 inputData_1(-13830,-29519,8340,-7025);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<935, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.324324268,0.845085733,0.120827924,0.4220107502,0.1050020284,0.5508400143,0.3282154384,0.1153468428);
cl::sycl::int8 inputData_1(-7646,3443,7888,17326,-782,18760,-26811,-29542);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<936, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4267073047,0.2978575899,0.7507078515,0.1000883464,0.1758027235,0.6887658669,0.7208700815,0.7355778741,0.267854535,0.4518008998,0.3972952934,0.37144386,0.8328164181,0.6818027476,0.6721412218,0.108379766);
cl::sycl::int16 inputData_1(-7282,6133,-5492,-11332,-22793,-18517,-19053,-6727,29365,15991,17971,17433,10733,9663,2703,-29844);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<937, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7495616336,0.2510958009);
cl::sycl::int2 inputData_1(-31643,-11631);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<938, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1928865054,0.2930308863,0.3539598162);
cl::sycl::int3 inputData_1(-6881,-11904,-20789);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<939, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7128487384,0.2987481583,0.4001976914,0.4084920273);
cl::sycl::int4 inputData_1(19848,5976,27648,6314);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<940, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2981688603,0.7027135628,0.1064007986,0.2465179538,0.3407439134,0.3580364163,0.5070923229,0.8847895717);
cl::sycl::int8 inputData_1(-5958,15672,-3809,-20781,-27155,-8822,-9051,19300);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<941, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1311771472,0.5124883284,0.6501944061,0.6074191328,0.1753617249,0.787051262,0.5533440568,0.8857086541,0.1678355276,0.8572388427,0.2082978871,0.8660875218,0.7920637663,0.3359078458,0.1586068901,0.1617749369);
cl::sycl::int16 inputData_1(-23631,5708,31745,2103,6269,-24254,26481,12366,-26468,18808,16764,21498,21750,-9843,-15694,-1682);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<942, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6706570463,0.2955104452);
cl::sycl::int2 inputData_1(23972,16956);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<943, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.335670234,0.3793336761,0.4061182528);
cl::sycl::int3 inputData_1(-11307,28990,6240);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<944, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2160433964,0.2843608219,0.8130693971,0.3399049231);
cl::sycl::int4 inputData_1(-16550,-22824,-27102,27295);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<945, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4996799072,0.7505488015,0.1683950115,0.2865783076,0.6410386401,0.2282238843,0.3205914975,0.1214600732);
cl::sycl::int8 inputData_1(-27601,-15782,-11929,25428,-23883,10822,12439,9296);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<946, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5751976122,0.4326997069,0.3862500089,0.5564065783,0.1818981402,0.4552235883,0.7988830796,0.4410157317,0.1900216996,0.449774403,0.3442988661,0.3123338411,0.4125315225,0.2052450035,0.4264466476,0.2103336344);
cl::sycl::int16 inputData_1(1135,27535,-17709,-19982,-13248,-13335,-2632,26969,6385,-10354,-13500,13409,-20,-31468,32337,-31533);
return cl::sycl::pown(inputData_0,inputData_1);

});

test_function<947, float>(
[=](){
float inputData_0(0.6776691419);
float inputData_1(0.1629703298);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<948, double>(
[=](){
double inputData_0(0.3285093227);
double inputData_1(0.5058845995);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<949, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6216613843);
cl::sycl::half inputData_1(0.8249632718);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<950, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8849658574,0.84223944);
cl::sycl::float2 inputData_1(0.5720744582,0.2790244278);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<951, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1662137405,0.6318444198,0.5639935457);
cl::sycl::float3 inputData_1(0.6318428916,0.4016198509,0.197955164);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<952, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.1694509209,0.31534078,0.7454321931,0.3865161091);
cl::sycl::float4 inputData_1(0.2888190103,0.3342217764,0.2540953921,0.2206758329);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<953, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6224569172,0.2702751428,0.8204216134,0.2224935909,0.3189707439,0.1124679873,0.308772688,0.6060638671);
cl::sycl::float8 inputData_1(0.604241642,0.5078023525,0.293604895,0.8454442486,0.2915470925,0.6144313248,0.1458049517,0.3946736402);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<954, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4413217919,0.554233781,0.1530968602,0.6143676072,0.3799656224,0.1692525398,0.1548559489,0.4473363824,0.3731534278,0.3582023449,0.6474346588,0.8193478068,0.8959794301,0.5248885983,0.8533096022,0.7704256388);
cl::sycl::float16 inputData_1(0.2081893286,0.4306377925,0.2488279395,0.6559642562,0.133923014,0.3444092439,0.2601397708,0.821103503,0.4504138291,0.5152684227,0.5257066118,0.197124616,0.8537312903,0.1105155361,0.2479491335,0.5569875476);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<955, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4262541594,0.2046183911);
cl::sycl::double2 inputData_1(0.7007690513,0.454019174);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<956, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4682891058,0.35528125,0.7115067898);
cl::sycl::double3 inputData_1(0.5180261771,0.683274601,0.4507705121);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<957, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1709276275,0.1519209172,0.611017127,0.723865865);
cl::sycl::double4 inputData_1(0.6181653981,0.1668700358,0.796689183,0.4076666516);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<958, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4143226615,0.6707894708,0.2508424124,0.8803689468,0.855481407,0.133973505,0.5016908773,0.344646551);
cl::sycl::double8 inputData_1(0.5591219945,0.655949396,0.8567992516,0.1450577727,0.2341415081,0.3197274347,0.4225296712,0.301352098);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<959, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6741359903,0.2567026933,0.4858269561,0.6085623673,0.3586246561,0.1242851905,0.703663923,0.2183389351,0.3186370326,0.8176604616,0.682789837,0.510580294,0.5692741612,0.7303598607,0.8303215007,0.7762702115);
cl::sycl::double16 inputData_1(0.4360694339,0.467606559,0.4255530457,0.4311995869,0.7701034844,0.2689070005,0.1045010223,0.5656966438,0.8915333431,0.6063854914,0.8677369942,0.7190067776,0.2446923428,0.401991881,0.2608273293,0.8113405466);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<960, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6801915206,0.777686152);
cl::sycl::half2 inputData_1(0.8803270998,0.8279867719);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<961, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1138003999,0.1259627676,0.6615307041);
cl::sycl::half3 inputData_1(0.3215174811,0.3155832217,0.5779747365);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<962, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4266061735,0.1617020672,0.1178883034,0.2749151083);
cl::sycl::half4 inputData_1(0.5212712805,0.1143001656,0.3605674607,0.8931070657);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<963, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1503782327,0.5256707926,0.2829667962,0.5314504866,0.4570214122,0.4835002515,0.8059101995,0.5591516537);
cl::sycl::half8 inputData_1(0.865885637,0.5531089988,0.5512155269,0.4704912975,0.7238621818,0.5640606332,0.1862456175,0.8658230278);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<964, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1383148527,0.6176941742,0.1307945314,0.4403264141,0.8329358602,0.8441323275,0.4840290201,0.1193335552,0.1035444703,0.6462298667,0.265763998,0.7268418759,0.5680685331,0.1724380229,0.1842212694,0.7994838811);
cl::sycl::half16 inputData_1(0.4260660418,0.66575823,0.252501617,0.8900113406,0.6999078973,0.2450889945,0.4461620602,0.2752324503,0.8590509377,0.6401198331,0.6350843975,0.418247598,0.854032549,0.4155759281,0.406873213,0.3406020883);
return cl::sycl::powr(inputData_0,inputData_1);

});

test_function<965, float>(
[=](){
float inputData_0(0.3178125276);
float inputData_1(0.2132348918);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<966, double>(
[=](){
double inputData_0(0.5623357172);
double inputData_1(0.629652525);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<967, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.3575551198);
cl::sycl::half inputData_1(0.2755904347);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<968, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1489894591,0.4209181722);
cl::sycl::float2 inputData_1(0.4305214586,0.2898619537);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<969, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3976675192,0.2297580748,0.7674497648);
cl::sycl::float3 inputData_1(0.5333189553,0.6693671736,0.7206734186);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<970, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8174573629,0.5173712216,0.4752044836,0.1162675033);
cl::sycl::float4 inputData_1(0.782334915,0.2426438744,0.4501362515,0.6599298686);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<971, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2061306085,0.3389045232,0.5038299326,0.7450296311,0.2314340055,0.2509898006,0.7794394284,0.8980416064);
cl::sycl::float8 inputData_1(0.2047012292,0.5591686303,0.8230969257,0.2790818207,0.4268634067,0.4946523446,0.620503791,0.5221729405);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<972, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1566594286,0.4503371562,0.6545579478,0.750973507,0.2521248111,0.3804746329,0.8425922875,0.8242819668,0.3090263944,0.6544649913,0.3519832557,0.7463798889,0.8460471253,0.2755242795,0.4254195097,0.8517209487);
cl::sycl::float16 inputData_1(0.1061231182,0.7998644535,0.7186583972,0.4405348651,0.5751386706,0.4041411431,0.5443761024,0.1359345187,0.6789688862,0.1536392915,0.7071939058,0.2534943942,0.6539166952,0.6754393912,0.5641829237,0.5622039661);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<973, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7598176124,0.8081559324);
cl::sycl::double2 inputData_1(0.3544786429,0.6626686775);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<974, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6261538744,0.4929666253,0.3421161733);
cl::sycl::double3 inputData_1(0.4769709868,0.1443986413,0.3007198207);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<975, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7333236218,0.3371842699,0.2986012052,0.6263493079);
cl::sycl::double4 inputData_1(0.8762240347,0.275038192,0.8865598982,0.8945025068);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<976, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8063683104,0.6123774586,0.2872977701,0.7976550663,0.7597099073,0.8164568782,0.5078822084,0.2367027765);
cl::sycl::double8 inputData_1(0.745140432,0.2473798298,0.5392636944,0.5390081684,0.257428899,0.8690773184,0.8609872449,0.4821351898);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<977, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2862546805,0.3425665804,0.7147696779,0.592201017,0.1448340143,0.3202153496,0.3333280952,0.5485139718,0.3161829802,0.5940719895,0.6417948599,0.4771926009,0.6683137611,0.5405100902,0.4020612518,0.7988822801);
cl::sycl::double16 inputData_1(0.6734740262,0.6961506879,0.7664017445,0.7739016347,0.4783073251,0.2028283616,0.3935844783,0.8648591126,0.6595933692,0.6792800629,0.2438735623,0.8195841999,0.3434530293,0.7700521771,0.517676993,0.516239799);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<978, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4810382974,0.7690933075);
cl::sycl::half2 inputData_1(0.6063489835,0.1331019873);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<979, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6638760012,0.2671051162,0.5667872181);
cl::sycl::half3 inputData_1(0.3209378119,0.1576293762,0.6842068531);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<980, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7300152634,0.462444888,0.2295327288,0.58449104);
cl::sycl::half4 inputData_1(0.1670833845,0.1461359213,0.2410944309,0.374245214);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<981, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3674608231,0.1618919083,0.6204985301,0.3936142428,0.8534084878,0.8055546011,0.1202689431,0.6601470772);
cl::sycl::half8 inputData_1(0.7685378049,0.5683120748,0.6213878452,0.5756487032,0.216883829,0.1861420809,0.7167699739,0.7467978519);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<982, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8371506795,0.6460175551,0.5430487683,0.4580955811,0.7767780526,0.4570564829,0.4856282089,0.3469636789,0.6870541379,0.638210235,0.488714051,0.2699338998,0.3526614965,0.4896199761,0.7742642606,0.2184386468);
cl::sycl::half16 inputData_1(0.3351860096,0.7917630958,0.351341821,0.1884379041,0.2015730425,0.5705466712,0.1316781499,0.7119492694,0.4935005228,0.3709790691,0.31533403,0.7669420161,0.487107068,0.1848339737,0.7164557456,0.1240988098);
return cl::sycl::remainder(inputData_0,inputData_1);

});

test_function<983, float>(
[=](){
float inputData_0(0.616788526);
float inputData_1(0.4368688684);
int * inputData_2 = new int(5133);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<984, double>(
[=](){
double inputData_0(0.4101734353);
double inputData_1(0.5623155934);
int * inputData_2 = new int(-28329);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<985, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1785114625);
cl::sycl::half inputData_1(0.3174231046);
int * inputData_2 = new int(5727);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<986, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7655625546,0.7540103155);
cl::sycl::float2 inputData_1(0.6680648734,0.3998043859);
cl::sycl::int2 * inputData_2 = new cl::sycl::int2(-20726,-8076);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<987, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.843616346,0.6707812555,0.7468760061);
cl::sycl::float3 inputData_1(0.6912498709,0.5484062479,0.4402777513);
cl::sycl::int3 * inputData_2 = new cl::sycl::int3(-15826,31205,7795);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<988, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4551072054,0.2908367705,0.1787690845,0.660371388);
cl::sycl::float4 inputData_1(0.103119425,0.5991981637,0.3849765169,0.6934891108);
cl::sycl::int4 * inputData_2 = new cl::sycl::int4(15723,-17287,-27319,15471);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<989, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4842927731,0.1652343665,0.5676090498,0.1236398163,0.7885973294,0.8886010356,0.1813047528,0.432099977);
cl::sycl::float8 inputData_1(0.8302581121,0.8501240781,0.4032279526,0.3788790183,0.5815931414,0.5537269819,0.1282185271,0.6082617716);
cl::sycl::int8 * inputData_2 = new cl::sycl::int8(-19960,10560,3161,15922,1265,31509,-32430,319);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<990, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8121763826,0.1597753953,0.6159307978,0.4686288029,0.7468194339,0.1753374384,0.7696829224,0.2278746601,0.5399221407,0.8168502686,0.8673504164,0.6832792266,0.7523426441,0.2484449625,0.776749935,0.4726162503);
cl::sycl::float16 inputData_1(0.1528354697,0.7934729842,0.7500620536,0.1220314471,0.407763361,0.5981217806,0.8571327332,0.7175599185,0.7947536518,0.5669487586,0.1660209126,0.5573416719,0.7809998552,0.4515830467,0.6834331844,0.1276747771);
cl::sycl::int16 * inputData_2 = new cl::sycl::int16(-29899,-8298,-24760,-5561,7509,-26060,-20127,24205,-21158,-32068,-5425,-23825,8427,15086,-23593,29278);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<991, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6474929925,0.4224540061);
cl::sycl::double2 inputData_1(0.1994225413,0.5279172911);
cl::sycl::int2 * inputData_2 = new cl::sycl::int2(13813,6358);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<992, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1919126908,0.774812852,0.1660699481);
cl::sycl::double3 inputData_1(0.1313324259,0.6238586284,0.3616540786);
cl::sycl::int3 * inputData_2 = new cl::sycl::int3(-20239,-25284,-24364);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<993, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2792718982,0.7818221214,0.6487109037,0.5346721202);
cl::sycl::double4 inputData_1(0.1808753048,0.204701906,0.6295922618,0.5704722186);
cl::sycl::int4 * inputData_2 = new cl::sycl::int4(21269,-22729,-21042,-799);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<994, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7657844385,0.2081074291,0.8479458275,0.6514366399,0.654576935,0.3778461573,0.7802583627,0.3715656653);
cl::sycl::double8 inputData_1(0.3134195422,0.4306206542,0.6173548218,0.3545544423,0.1136076971,0.6257947322,0.4812498498,0.3348988708);
cl::sycl::int8 * inputData_2 = new cl::sycl::int8(18941,11839,-19974,-31704,-678,-12586,30026,16451);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<995, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1064971036,0.3960274799,0.7443599596,0.7957831717,0.6182352517,0.2919100844,0.2548832396,0.7218895578,0.8476321763,0.2412062458,0.3041231444,0.1796979027,0.2347905014,0.7845798262,0.5621470436,0.7926610497);
cl::sycl::double16 inputData_1(0.4302729331,0.4313483771,0.2852516581,0.549745368,0.7629840839,0.1643291099,0.1745518202,0.4426121969,0.3458846028,0.4986120547,0.8907793507,0.4293361863,0.4222776143,0.418395628,0.1258981652,0.3254437287);
cl::sycl::int16 * inputData_2 = new cl::sycl::int16(1544,26617,-12926,-643,7539,-24455,-30576,13511,-6939,-32615,-17644,-26180,-7443,-14687,18313,-31780);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<996, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3304736554,0.543784994);
cl::sycl::half2 inputData_1(0.1566689123,0.6713462502);
cl::sycl::int2 * inputData_2 = new cl::sycl::int2(5902,21295);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<997, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2802643788,0.2902834134,0.7249485497);
cl::sycl::half3 inputData_1(0.1671555571,0.8089406206,0.4710754935);
cl::sycl::int3 * inputData_2 = new cl::sycl::int3(-31254,16806,16786);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<998, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7976744644,0.7389586798,0.4401481506,0.406036208);
cl::sycl::half4 inputData_1(0.3664220207,0.7190044014,0.2086964763,0.6150992668);
cl::sycl::int4 * inputData_2 = new cl::sycl::int4(-32274,17300,-19905,21158);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<999, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4481935276,0.709867054,0.6228809898,0.8933601813,0.5183131325,0.7776083827,0.7253883488,0.5835142452);
cl::sycl::half8 inputData_1(0.824495211,0.6428604268,0.6654422772,0.4043572262,0.1817733932,0.4264471832,0.1512661863,0.3554785093);
cl::sycl::int8 * inputData_2 = new cl::sycl::int8(-12569,-21937,-24368,-30905,8438,9577,30342,-26836);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<1000, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8369947078,0.2828700195,0.4487892447,0.7897004702,0.5590937937,0.7800517004,0.729435732,0.6144403703,0.4783313191,0.8369061703,0.7727737372,0.6296581112,0.6151032059,0.6925328887,0.1388420936,0.8365130534);
cl::sycl::half16 inputData_1(0.1861079274,0.2354346125,0.6565162981,0.7184721694,0.3097237183,0.6615552033,0.3489056597,0.1684368378,0.4642267036,0.6063757123,0.3315259022,0.2311733354,0.794926649,0.7766185989,0.6108049024,0.7721956429);
cl::sycl::int16 * inputData_2 = new cl::sycl::int16(-24474,-22829,12998,-7320,-20066,-878,26695,8150,-31695,13290,-7495,23314,23082,-16122,-11345,14013);
return cl::sycl::remquo(inputData_0,inputData_1,inputData_2);
delete inputData_2;

});

test_function<1001, float>(
[=](){
float inputData_0(0.7481375727);
return cl::sycl::rint(inputData_0);

});

test_function<1002, double>(
[=](){
double inputData_0(0.5487550026);
return cl::sycl::rint(inputData_0);

});

test_function<1003, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.2656829259);
return cl::sycl::rint(inputData_0);

});

test_function<1004, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.689227395,0.6313968662);
return cl::sycl::rint(inputData_0);

});

test_function<1005, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8822998902,0.125994997,0.2991077751);
return cl::sycl::rint(inputData_0);

});

test_function<1006, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6168720942,0.7652581597,0.7044417565,0.5843922714);
return cl::sycl::rint(inputData_0);

});

test_function<1007, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1528144628,0.1613034583,0.34405126,0.4106604041,0.5621744152,0.7399554902,0.8530894429,0.4157175148);
return cl::sycl::rint(inputData_0);

});

test_function<1008, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7513785189,0.3769116153,0.8648490297,0.2714828292,0.7790762353,0.7031120568,0.7210463633,0.7368004511,0.1286064808,0.250695667,0.2094358537,0.3151035701,0.4505685462,0.248387351,0.4820435531,0.3167616237);
return cl::sycl::rint(inputData_0);

});

test_function<1009, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.314489328,0.6940150121);
return cl::sycl::rint(inputData_0);

});

test_function<1010, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4312517403,0.1560580053,0.6228040274);
return cl::sycl::rint(inputData_0);

});

test_function<1011, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5483690072,0.5079409593,0.4043228052,0.5899270659);
return cl::sycl::rint(inputData_0);

});

test_function<1012, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.178397412,0.8360471048,0.3333183753,0.2762002077,0.8108980584,0.5725256245,0.2539554847,0.4992241692);
return cl::sycl::rint(inputData_0);

});

test_function<1013, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5998508763,0.2573578493,0.4462260202,0.5374046895,0.7008870616,0.3043149923,0.4828351654,0.5927756119,0.2109322468,0.8840490685,0.6464615255,0.2978093089,0.615535893,0.1602249898,0.8560035019,0.8853273983);
return cl::sycl::rint(inputData_0);

});

test_function<1014, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.424256904,0.4749416532);
return cl::sycl::rint(inputData_0);

});

test_function<1015, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1273304727,0.6688768162,0.2785190573);
return cl::sycl::rint(inputData_0);

});

test_function<1016, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5992652806,0.7743307752,0.8037300685,0.7083775117);
return cl::sycl::rint(inputData_0);

});

test_function<1017, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6606510663,0.1043580702,0.6152476206,0.4780052479,0.754151522,0.2729688377,0.722426376,0.6507890553);
return cl::sycl::rint(inputData_0);

});

test_function<1018, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.6617826624,0.4073289373,0.69210773,0.3821087332,0.4912338072,0.7614500408,0.128266977,0.49828754,0.297209345,0.8308637865,0.2608802588,0.4746532865,0.3386681239,0.1916951512,0.6840732557,0.8885586659);
return cl::sycl::rint(inputData_0);

});

test_function<1019, float>(
[=](){
float inputData_0(0.632809678);
int inputData_1(3665);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1020, double>(
[=](){
double inputData_0(0.148691493);
int inputData_1(11090);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1021, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1322506445);
int inputData_1(-23766);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1022, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.619442234,0.7875223998);
cl::sycl::int2 inputData_1(684,18017);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1023, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5192239491,0.5592931903,0.56300561);
cl::sycl::int3 inputData_1(-27986,30041,-9300);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1024, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6988615526,0.735870336,0.1158345228,0.7711589248);
cl::sycl::int4 inputData_1(-7979,5518,22395,-10966);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1025, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4423959782,0.4649544532,0.37328275,0.4180995628,0.4870758639,0.6857483511,0.4463414723,0.5436192615);
cl::sycl::int8 inputData_1(-16028,-20183,30745,-22710,-8825,8784,28671,12207);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1026, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7126532403,0.8255163744,0.3394950315,0.1384206116,0.4543513914,0.3812692695,0.5780987873,0.386804241,0.6083412715,0.2741242413,0.3808690871,0.2372564002,0.2538806682,0.4114718531,0.3602119461,0.5603085896);
cl::sycl::int16 inputData_1(22626,-20812,-7523,26539,21266,-32645,13445,15952,18757,-26570,-2007,29031,7,-18012,-2485,15779);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1027, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8341121429,0.789165597);
cl::sycl::int2 inputData_1(-9548,3433);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1028, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.3760281344,0.7331577754,0.719309534);
cl::sycl::int3 inputData_1(-28323,-12477,10356);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1029, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2056498409,0.2628337909,0.7063103845,0.3720411895);
cl::sycl::int4 inputData_1(-14121,-2660,28826,-22229);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1030, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1984768852,0.7934881956,0.6682590704,0.7553924058,0.8846951038,0.703162617,0.2233820879,0.4252028906);
cl::sycl::int8 inputData_1(-30055,10641,5426,-7850,-21112,23584,-9551,7595);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1031, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6137921579,0.6710064323,0.8208372641,0.701712318,0.399414497,0.5908777107,0.780358323,0.5188691483,0.7561337854,0.3083636748,0.7282159051,0.6247997914,0.1133962119,0.1277454751,0.5731112448,0.6514413322);
cl::sycl::int16 inputData_1(27295,29861,-24099,-25205,-16655,19816,-24314,-15838,1928,13296,15670,-20330,-12502,27516,246,9816);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1032, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.2882358911,0.6870548464);
cl::sycl::int2 inputData_1(-13010,6472);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1033, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5302776533,0.5401948356,0.8585356856);
cl::sycl::int3 inputData_1(-9501,-32727,-24369);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1034, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7442039495,0.6322299231,0.1748969508,0.3150317772);
cl::sycl::int4 inputData_1(-4410,-687,9098,-5926);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1035, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1112496479,0.3979754096,0.467067073,0.327752341,0.3657816795,0.4187218994,0.1110308305,0.3922847678);
cl::sycl::int8 inputData_1(28047,27214,-11482,-13770,7697,-19673,-18180,6907);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1036, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2528895398,0.3073129881,0.1063258542,0.4794381405,0.7862571862,0.3692031892,0.6859326251,0.4956396625,0.2124903082,0.1602691092,0.5154037219,0.5648261443,0.6500989926,0.6881025276,0.3594334798,0.1675063754);
cl::sycl::int16 inputData_1(22944,-27096,22107,-6138,-16169,-23926,-20600,-17535,30617,-30255,-26192,-5952,7454,18824,18309,-28431);
return cl::sycl::rootn(inputData_0,inputData_1);

});

test_function<1037, float>(
[=](){
float inputData_0(0.6799374466);
return cl::sycl::round(inputData_0);

});

test_function<1038, double>(
[=](){
double inputData_0(0.8188690873);
return cl::sycl::round(inputData_0);

});

test_function<1039, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.232648152);
return cl::sycl::round(inputData_0);

});

test_function<1040, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6300402047,0.6321568756);
return cl::sycl::round(inputData_0);

});

test_function<1041, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5924059285,0.2471494851,0.4431329335);
return cl::sycl::round(inputData_0);

});

test_function<1042, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3294901407,0.7381517553,0.8880051043,0.4894632762);
return cl::sycl::round(inputData_0);

});

test_function<1043, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2096008795,0.1538170806,0.1925963429,0.451634012,0.8277726369,0.4679287482,0.8571256496,0.8684337556);
return cl::sycl::round(inputData_0);

});

test_function<1044, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5718933006,0.5414112165,0.7356142596,0.2362679808,0.7611561409,0.1064608205,0.4493658072,0.8119606776,0.6211463342,0.2827129286,0.1159815997,0.4790012928,0.6252795625,0.6108072422,0.3798368186,0.60389382);
return cl::sycl::round(inputData_0);

});

test_function<1045, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6860203003,0.6442216007);
return cl::sycl::round(inputData_0);

});

test_function<1046, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7220239897,0.6160137372,0.2777231141);
return cl::sycl::round(inputData_0);

});

test_function<1047, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1340535984,0.5259720672,0.5932440241,0.1744498781);
return cl::sycl::round(inputData_0);

});

test_function<1048, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8281772146,0.4545170684,0.2088271178,0.5281234529,0.4631212873,0.7393453876,0.1645158568,0.2006306319);
return cl::sycl::round(inputData_0);

});

test_function<1049, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2172823135,0.8414787939,0.4431026297,0.4728950861,0.764026657,0.3625155181,0.4324383917,0.882105729,0.3253295527,0.6712641929,0.6310450658,0.738545187,0.7616309677,0.6753236814,0.6467171431,0.5512251564);
return cl::sycl::round(inputData_0);

});

test_function<1050, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8990663877,0.3022874157);
return cl::sycl::round(inputData_0);

});

test_function<1051, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2147391005,0.823379709,0.4587681022);
return cl::sycl::round(inputData_0);

});

test_function<1052, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5807708651,0.3396446825,0.4948048956,0.2055185984);
return cl::sycl::round(inputData_0);

});

test_function<1053, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1930341843,0.7442486659,0.4285658183,0.2579929255,0.6689549994,0.161657992,0.2711422818,0.8819843145);
return cl::sycl::round(inputData_0);

});

test_function<1054, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.5124990261,0.5352862882,0.2661583103,0.6875031197,0.7482619088,0.2126961524,0.6887359554,0.1658835629,0.8863874443,0.8364126301,0.4453536009,0.2906153309,0.3496371124,0.1087373263,0.881826048,0.8356503633);
return cl::sycl::round(inputData_0);

});

test_function<1055, float>(
[=](){
float inputData_0(0.6422099804);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1056, double>(
[=](){
double inputData_0(0.495834311);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1057, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5912895644);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1058, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.146042222,0.2516547293);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1059, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1968099268,0.1366536786,0.2195204697);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1060, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3713746261,0.1227802812,0.6639288119,0.2684880094);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1061, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2169864473,0.1816267917,0.3838907547,0.8201612402,0.7402883829,0.7921039105,0.6143700657,0.579149497);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1062, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3191441163,0.7574697809,0.4488471244,0.7989152698,0.3510580369,0.3986920065,0.8685684122,0.5123235417,0.2670022938,0.608907653,0.2000441539,0.7930329059,0.6414410108,0.7699500995,0.2139182457,0.3306579885);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1063, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6896795011,0.1475955915);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1064, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7667457295,0.668091322,0.4052886587);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1065, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.614808415,0.3098783392,0.5453148795,0.1276578734);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1066, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2066854529,0.2579541485,0.4022838236,0.5629159858,0.4661263173,0.4408639199,0.8665141803,0.8003308866);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1067, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6076917114,0.3963204587,0.8721201673,0.1971366232,0.5463224022,0.204358364,0.5377726759,0.465022914,0.3172415111,0.6645766188,0.1487045934,0.5766664942,0.4248854315,0.3673781347,0.3842500052,0.5350242997);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1068, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4785237206,0.4649482417);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1069, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8438424805,0.3950056946,0.4197136426);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1070, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2514814028,0.4169276072,0.290276469,0.3467064249);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1071, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1110814933,0.4433883654,0.5414035289,0.1824152394,0.3737353852,0.1369548776,0.3310957648,0.268021388);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1072, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.7775478734,0.1716139475,0.3192578014,0.608338234,0.8219910205,0.7174649826,0.1618246431,0.1473837443,0.348766967,0.8541271908,0.1804191675,0.2857320217,0.7540282476,0.1370873618,0.2382371516,0.6024506371);
return cl::sycl::rsqrt(inputData_0);

});

test_function<1073, float>(
[=](){
float inputData_0(0.71250238);
return cl::sycl::sin(inputData_0);

});

test_function<1074, double>(
[=](){
double inputData_0(0.5950179314);
return cl::sycl::sin(inputData_0);

});

test_function<1075, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7143480219);
return cl::sycl::sin(inputData_0);

});

test_function<1076, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2493490056,0.2877212645);
return cl::sycl::sin(inputData_0);

});

test_function<1077, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8896371176,0.5377243064,0.4624339162);
return cl::sycl::sin(inputData_0);

});

test_function<1078, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.124904526,0.1481198346,0.6145310342,0.3627359932);
return cl::sycl::sin(inputData_0);

});

test_function<1079, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7933566243,0.3997809142,0.4123991691,0.3038900034,0.7641889058,0.259306508,0.5513901353,0.4963168689);
return cl::sycl::sin(inputData_0);

});

test_function<1080, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6295335555,0.8966270582,0.7898803929,0.7917807749,0.339608738,0.7876594977,0.797186963,0.5880322822,0.4082525105,0.8535474001,0.8430329546,0.1559160646,0.5698364137,0.8922111336,0.637931018,0.3994918494);
return cl::sycl::sin(inputData_0);

});

test_function<1081, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3343517257,0.1840300706);
return cl::sycl::sin(inputData_0);

});

test_function<1082, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5365244518,0.6498549855,0.8903829606);
return cl::sycl::sin(inputData_0);

});

test_function<1083, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2158512781,0.8756965924,0.3790542348,0.370087284);
return cl::sycl::sin(inputData_0);

});

test_function<1084, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4961868851,0.2524595843,0.1597164788,0.2426372503,0.2751971836,0.7447266008,0.6187259418,0.2194543531);
return cl::sycl::sin(inputData_0);

});

test_function<1085, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4945412697,0.6168317699,0.8575021038,0.2720981294,0.5825031431,0.1579656231,0.4157679621,0.5435208534,0.1198411892,0.506281647,0.3441685277,0.5400135371,0.4846818096,0.6018052613,0.1126501681,0.257635144);
return cl::sycl::sin(inputData_0);

});

test_function<1086, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3765711364,0.1446741651);
return cl::sycl::sin(inputData_0);

});

test_function<1087, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8621710882,0.519607042,0.5576603691);
return cl::sycl::sin(inputData_0);

});

test_function<1088, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7962807646,0.4596863901,0.1588554723,0.5164721985);
return cl::sycl::sin(inputData_0);

});

test_function<1089, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3554867978,0.5007363183,0.8571841339,0.3862911483,0.7501655738,0.3172952151,0.3943598833,0.8742374871);
return cl::sycl::sin(inputData_0);

});

test_function<1090, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.2632954888,0.638947859,0.1258683284,0.3922809521,0.4332363373,0.6222796371,0.6187184745,0.7236827237,0.7744381062,0.8896882163,0.6760832441,0.2045773306,0.4854174051,0.8026971562,0.8297398534,0.3412936257);
return cl::sycl::sin(inputData_0);

});

test_function<1091, float>(
[=](){
float inputData_0(0.549964591);
float * inputData_1 = new float(0.535073937);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1092, double>(
[=](){
double inputData_0(0.7268009307);
double * inputData_1 = new double(0.8635780766);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1093, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.6661038325);
cl::sycl::half * inputData_1 = new cl::sycl::half(0.7321656815);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1094, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3291739447,0.7458248695);
cl::sycl::float2 * inputData_1 = new cl::sycl::float2(0.1643094888,0.8976523665);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1095, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2090442059,0.4428909527,0.5030049866);
cl::sycl::float3 * inputData_1 = new cl::sycl::float3(0.8769196105,0.1543130098,0.5093875478);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1096, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7532497887,0.3143765831,0.7150592907,0.2668462962);
cl::sycl::float4 * inputData_1 = new cl::sycl::float4(0.5195621972,0.361694132,0.1996258656,0.1065189873);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1097, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7197282384,0.223097539,0.2022318532,0.2223694038,0.3338922929,0.2733121727,0.8012921012,0.7015022519);
cl::sycl::float8 * inputData_1 = new cl::sycl::float8(0.5780015718,0.3610861713,0.3612609053,0.5095165543,0.5727509063,0.6826496587,0.5135160159,0.7037706533);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1098, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6709058497,0.3536457867,0.7668395179,0.8183024117,0.4127588491,0.4653483145,0.8771442897,0.8817707966,0.306453012,0.7071047911,0.6941682449,0.609523978,0.8442588713,0.3448605731,0.1351394859,0.4120214819);
cl::sycl::float16 * inputData_1 = new cl::sycl::float16(0.686973374,0.6039204002,0.3461255442,0.5049726823,0.1720757699,0.8778055146,0.3810017302,0.8105728041,0.3104315048,0.3693818699,0.8196697165,0.84251428,0.6153034205,0.5742513185,0.1772821669,0.6689124921);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1099, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.8717661786,0.547820646);
cl::sycl::double2 * inputData_1 = new cl::sycl::double2(0.8517588054,0.2720160627);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1100, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8814634398,0.5385969124,0.6341936356);
cl::sycl::double3 * inputData_1 = new cl::sycl::double3(0.1654464997,0.7411065013,0.681422766);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1101, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5334702333,0.577076682,0.5748936795,0.2834811834);
cl::sycl::double4 * inputData_1 = new cl::sycl::double4(0.1357542817,0.7791767939,0.7969538097,0.1237661626);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1102, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7771754539,0.6279221734,0.6460339177,0.7377001293,0.633128758,0.1918399434,0.7356900987,0.160270364);
cl::sycl::double8 * inputData_1 = new cl::sycl::double8(0.6301733147,0.2785051676,0.1319598356,0.7205453544,0.7111394293,0.7479480457,0.2596075204,0.2960040782);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1103, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2163576595,0.695067711,0.6635356501,0.2560048203,0.7400977622,0.8954525662,0.1627389862,0.2263045398,0.1923310439,0.68897675,0.733800127,0.4681499681,0.7925584444,0.7336606513,0.2340405821,0.3004922644);
cl::sycl::double16 * inputData_1 = new cl::sycl::double16(0.6175836839,0.3370489034,0.343218594,0.2822915253,0.218193997,0.1141849402,0.4615776489,0.4945472041,0.6225841859,0.2013159938,0.3647802196,0.4495822143,0.1670016782,0.2536280701,0.5429825532,0.4942287449);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1104, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4723209083,0.5585207729);
cl::sycl::half2 * inputData_1 = new cl::sycl::half2(0.6439658779,0.1255215704);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1105, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.416059216,0.1219210582,0.8417170061);
cl::sycl::half3 * inputData_1 = new cl::sycl::half3(0.2610221727,0.6408734982,0.8299900427);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1106, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2237688862,0.7376537494,0.6207185593,0.1972935681);
cl::sycl::half4 * inputData_1 = new cl::sycl::half4(0.5780679512,0.8490482261,0.4042323965,0.572938225);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1107, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3091756024,0.7405719092,0.5726298137,0.6163475192,0.5343308061,0.4481122597,0.4936915143,0.5595264778);
cl::sycl::half8 * inputData_1 = new cl::sycl::half8(0.7052736381,0.8141676615,0.6669251456,0.4803567821,0.1445958322,0.6556487548,0.581202652,0.6789254497);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1108, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1979241607,0.7880838122,0.1567557921,0.4824264834,0.7687913918,0.2415273956,0.6108225349,0.6770608933,0.2863624155,0.5007793557,0.1626799084,0.4278024118,0.6966046909,0.2417089759,0.234044983,0.3188215396);
cl::sycl::half16 * inputData_1 = new cl::sycl::half16(0.4461735143,0.3305638264,0.7144716913,0.5237185647,0.5501241363,0.4376261299,0.6583742633,0.2412264419,0.7729566963,0.3527586835,0.1163117866,0.5875145281,0.7417085183,0.8571160095,0.360495536,0.5776986508);
return cl::sycl::sincos(inputData_0,inputData_1);
delete inputData_1;

});

test_function<1109, float>(
[=](){
float inputData_0(0.8600583147);
return cl::sycl::sinh(inputData_0);

});

test_function<1110, double>(
[=](){
double inputData_0(0.5943047538);
return cl::sycl::sinh(inputData_0);

});

test_function<1111, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.5060023011);
return cl::sycl::sinh(inputData_0);

});

test_function<1112, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1644436576,0.8208361361);
return cl::sycl::sinh(inputData_0);

});

test_function<1113, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5887391778,0.6689511563,0.1459256427);
return cl::sycl::sinh(inputData_0);

});

test_function<1114, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6517653985,0.362436867,0.874313295,0.8443562853);
return cl::sycl::sinh(inputData_0);

});

test_function<1115, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7660671918,0.6577238005,0.4142688034,0.6165534599,0.4206300331,0.427029155,0.5883003376,0.2138077641);
return cl::sycl::sinh(inputData_0);

});

test_function<1116, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7092964846,0.3934732638,0.1575437687,0.2283951922,0.7536079173,0.6184122732,0.3905493355,0.518480499,0.8584885996,0.4032457831,0.8944602853,0.823873703,0.2744287002,0.2158558689,0.2312411381,0.1880971788);
return cl::sycl::sinh(inputData_0);

});

test_function<1117, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.873619911,0.6612406655);
return cl::sycl::sinh(inputData_0);

});

test_function<1118, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2909149062,0.412520672,0.1058976792);
return cl::sycl::sinh(inputData_0);

});

test_function<1119, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.503552014,0.5864714481,0.1200919201,0.8528694159);
return cl::sycl::sinh(inputData_0);

});

test_function<1120, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4762626572,0.2396738444,0.5611205745,0.7522295278,0.8596950756,0.4716574551,0.814621998,0.3064456634);
return cl::sycl::sinh(inputData_0);

});

test_function<1121, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4781364963,0.5065133945,0.5199346348,0.2689730665,0.5892370394,0.4332871527,0.7307761971,0.1715437634,0.5528790743,0.1660901393,0.6121678494,0.5167628617,0.2819870952,0.1533862558,0.4097897754,0.8704463331);
return cl::sycl::sinh(inputData_0);

});

test_function<1122, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7506752104,0.7652810234);
return cl::sycl::sinh(inputData_0);

});

test_function<1123, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7976679675,0.5894307572,0.6180300783);
return cl::sycl::sinh(inputData_0);

});

test_function<1124, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.723269362,0.6282304628,0.1269475743,0.2192815291);
return cl::sycl::sinh(inputData_0);

});

test_function<1125, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6574473381,0.2277975279,0.1347566862,0.554407736,0.7258789267,0.2298540118,0.2300546135,0.1534146703);
return cl::sycl::sinh(inputData_0);

});

test_function<1126, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3645925159,0.8768444927,0.8920235424,0.5510561168,0.6130018219,0.8744019145,0.3723632486,0.6538796417,0.2293839046,0.1046299864,0.4891001295,0.2875780749,0.1011940103,0.6983815748,0.2577355415,0.7377632531);
return cl::sycl::sinh(inputData_0);

});

test_function<1127, float>(
[=](){
float inputData_0(0.8386110183);
return cl::sycl::sinpi(inputData_0);

});

test_function<1128, double>(
[=](){
double inputData_0(0.5591309458);
return cl::sycl::sinpi(inputData_0);

});

test_function<1129, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.3269226414);
return cl::sycl::sinpi(inputData_0);

});

test_function<1130, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5848220658,0.1506784511);
return cl::sycl::sinpi(inputData_0);

});

test_function<1131, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.312473799,0.2014806798,0.7479185101);
return cl::sycl::sinpi(inputData_0);

});

test_function<1132, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5369361997,0.3081292933,0.4841181006,0.575905404);
return cl::sycl::sinpi(inputData_0);

});

test_function<1133, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.3623007805,0.5912076335,0.2577025173,0.7430714847,0.5053074883,0.8129776523,0.8640580756,0.8399919227);
return cl::sycl::sinpi(inputData_0);

});

test_function<1134, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3203611261,0.3402940316,0.4184425652,0.5169674442,0.4126432053,0.1643037329,0.2976847543,0.223901524,0.5157084309,0.2854767977,0.4221207784,0.4432911475,0.3813634887,0.6240727024,0.3342020474,0.2309012117);
return cl::sycl::sinpi(inputData_0);

});

test_function<1135, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7913043965,0.1513794207);
return cl::sycl::sinpi(inputData_0);

});

test_function<1136, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4616807295,0.2875611683,0.5540961213);
return cl::sycl::sinpi(inputData_0);

});

test_function<1137, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2191174887,0.2390923402,0.7481815512,0.6151240265);
return cl::sycl::sinpi(inputData_0);

});

test_function<1138, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2862595284,0.631721057,0.2924896628,0.7884290698,0.4426540712,0.1459945878,0.6638223999,0.5019745642);
return cl::sycl::sinpi(inputData_0);

});

test_function<1139, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8741394762,0.7013494395,0.6411421305,0.3131284518,0.6094997158,0.1021409079,0.5898321373,0.1067936319,0.4579409287,0.4644404603,0.1441605888,0.4132565556,0.7542605929,0.538729589,0.405383516,0.1924245323);
return cl::sycl::sinpi(inputData_0);

});

test_function<1140, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.7423311571,0.1746122648);
return cl::sycl::sinpi(inputData_0);

});

test_function<1141, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.2836116799,0.5497381972,0.4056927738);
return cl::sycl::sinpi(inputData_0);

});

test_function<1142, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8097327926,0.2736514987,0.6793695854,0.599303265);
return cl::sycl::sinpi(inputData_0);

});

test_function<1143, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8777575497,0.5529360124,0.3260912375,0.6818498979,0.8444880547,0.1619344436,0.2954992835,0.2678695404);
return cl::sycl::sinpi(inputData_0);

});

test_function<1144, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.7435998069,0.4557607771,0.1248794242,0.4045108513,0.271153479,0.1507650231,0.1830764268,0.5286646242,0.3046689457,0.4689705755,0.360945828,0.5295411133,0.3635115208,0.4439918986,0.862296933,0.4390733763);
return cl::sycl::sinpi(inputData_0);

});

test_function<1145, float>(
[=](){
float inputData_0(0.5067732305);
return cl::sycl::sqrt(inputData_0);

});

test_function<1146, double>(
[=](){
double inputData_0(0.5847882785);
return cl::sycl::sqrt(inputData_0);

});

test_function<1147, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.718188423);
return cl::sycl::sqrt(inputData_0);

});

test_function<1148, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1428446591,0.2841460662);
return cl::sycl::sqrt(inputData_0);

});

test_function<1149, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6039168229,0.2703012999,0.1135999698);
return cl::sycl::sqrt(inputData_0);

});

test_function<1150, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.1888388869,0.8849313181,0.2783444311,0.5947184549);
return cl::sycl::sqrt(inputData_0);

});

test_function<1151, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.296321978,0.5498934887,0.4966974547,0.415768653,0.5637083153,0.6108743808,0.6720142186,0.6951978268);
return cl::sycl::sqrt(inputData_0);

});

test_function<1152, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1568218863,0.7142505543,0.5643209057,0.4475481739,0.5183023611,0.2287272539,0.4280560279,0.8472562927,0.5962013595,0.3113351942,0.3109589028,0.674984952,0.718311442,0.3644124766,0.4692580662,0.457164261);
return cl::sycl::sqrt(inputData_0);

});

test_function<1153, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6485450294,0.2456572697);
return cl::sycl::sqrt(inputData_0);

});

test_function<1154, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.671285765,0.4913930233,0.8389092015);
return cl::sycl::sqrt(inputData_0);

});

test_function<1155, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3780378777,0.2487963807,0.5599390719,0.4863931726);
return cl::sycl::sqrt(inputData_0);

});

test_function<1156, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8275333429,0.7964656831,0.7084525564,0.1043322889,0.2269344762,0.5258361056,0.4598879783,0.6935276842);
return cl::sycl::sqrt(inputData_0);

});

test_function<1157, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.417591358,0.6501698816,0.7065427358,0.1521095188,0.1151665779,0.3883749023,0.5150263042,0.3642564128,0.4394905587,0.2989693387,0.1683335807,0.1242167571,0.8993697696,0.1778214275,0.6028639517,0.8907460024);
return cl::sycl::sqrt(inputData_0);

});

test_function<1158, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.134498014,0.4231977566);
return cl::sycl::sqrt(inputData_0);

});

test_function<1159, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.183910203,0.5678398202,0.6078009707);
return cl::sycl::sqrt(inputData_0);

});

test_function<1160, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1727583135,0.5407230283,0.1890180673,0.4041182231);
return cl::sycl::sqrt(inputData_0);

});

test_function<1161, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4169380729,0.4391491731,0.5638667549,0.8858548233,0.353459747,0.4067003519,0.4221657229,0.1750306898);
return cl::sycl::sqrt(inputData_0);

});

test_function<1162, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.766613121,0.3543065369,0.4921317321,0.4018380965,0.6386932992,0.8055800647,0.4900256958,0.2020048764,0.8380277295,0.3125347072,0.8777405629,0.6433935319,0.5395483419,0.2850853492,0.6234038399,0.2411476378);
return cl::sycl::sqrt(inputData_0);

});

test_function<1163, float>(
[=](){
float inputData_0(0.4062416564);
return cl::sycl::tan(inputData_0);

});

test_function<1164, double>(
[=](){
double inputData_0(0.8799087374);
return cl::sycl::tan(inputData_0);

});

test_function<1165, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8482876788);
return cl::sycl::tan(inputData_0);

});

test_function<1166, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2636305861,0.3447348758);
return cl::sycl::tan(inputData_0);

});

test_function<1167, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6295787596,0.7598786584,0.7881822238);
return cl::sycl::tan(inputData_0);

});

test_function<1168, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2311936714,0.7324257913,0.2304204297,0.1894095902);
return cl::sycl::tan(inputData_0);

});

test_function<1169, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8315084594,0.3765430478,0.2384801776,0.1646452632,0.5586230886,0.4375531697,0.8200004719,0.8964617975);
return cl::sycl::tan(inputData_0);

});

test_function<1170, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5752474041,0.5908431851,0.1837913345,0.2819440871,0.4495504999,0.1237198249,0.2254153395,0.6748986932,0.8027821808,0.7561137329,0.1918921582,0.5269568116,0.7292501132,0.3030630495,0.5127937879,0.5956498224);
return cl::sycl::tan(inputData_0);

});

test_function<1171, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6576961824,0.3355749281);
return cl::sycl::tan(inputData_0);

});

test_function<1172, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4216945171,0.4086544277,0.3265958533);
return cl::sycl::tan(inputData_0);

});

test_function<1173, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7901962311,0.1282764887,0.7799520285,0.3017347111);
return cl::sycl::tan(inputData_0);

});

test_function<1174, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1477272076,0.7826774744,0.7189334664,0.1722282385,0.573286954,0.3919894579,0.8740581142,0.5186473472);
return cl::sycl::tan(inputData_0);

});

test_function<1175, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5598520706,0.8079110814,0.2887999217,0.8769429054,0.3562134683,0.3076726379,0.2943016923,0.8386012065,0.5151065249,0.1524237153,0.7105215564,0.5757570763,0.2491455162,0.3520394508,0.3070936599,0.3401075464);
return cl::sycl::tan(inputData_0);

});

test_function<1176, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3569554435,0.6954543894);
return cl::sycl::tan(inputData_0);

});

test_function<1177, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.5250580211,0.124186722,0.8460287782);
return cl::sycl::tan(inputData_0);

});

test_function<1178, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3339251736,0.4739696173,0.1776505582,0.8953151519);
return cl::sycl::tan(inputData_0);

});

test_function<1179, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.4600669848,0.4796436015,0.113363094,0.4285126458,0.7510777586,0.1905500412,0.2913317764,0.7015356777);
return cl::sycl::tan(inputData_0);

});

test_function<1180, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.4299605843,0.6003552784,0.6906535359,0.1058361253,0.7132781609,0.7466368771,0.5236435454,0.5283252452,0.7603875965,0.7293480955,0.2126525811,0.4815597895,0.7343542732,0.271698276,0.403816465,0.3196762632);
return cl::sycl::tan(inputData_0);

});

test_function<1181, float>(
[=](){
float inputData_0(0.8897280838);
return cl::sycl::tanh(inputData_0);

});

test_function<1182, double>(
[=](){
double inputData_0(0.6778976171);
return cl::sycl::tanh(inputData_0);

});

test_function<1183, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1253888765);
return cl::sycl::tanh(inputData_0);

});

test_function<1184, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4704248935,0.8634400075);
return cl::sycl::tanh(inputData_0);

});

test_function<1185, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6032043757,0.5161592339,0.3806187019);
return cl::sycl::tanh(inputData_0);

});

test_function<1186, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6594291279,0.7354765018,0.3654712485,0.777949479);
return cl::sycl::tanh(inputData_0);

});

test_function<1187, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.3939443775,0.7347140895,0.4191565479,0.8040965029,0.1442329036,0.8114763616,0.4252474029,0.1453505977);
return cl::sycl::tanh(inputData_0);

});

test_function<1188, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6701366643,0.5709551543,0.6153317909,0.6271952419,0.6082344207,0.4611486086,0.7634484231,0.182125313,0.7455133641,0.8774324364,0.7475095233,0.5741570844,0.1097230159,0.1200894004,0.5002750796,0.5252941749);
return cl::sycl::tanh(inputData_0);

});

test_function<1189, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7010380861,0.5057056426);
return cl::sycl::tanh(inputData_0);

});

test_function<1190, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.3817136946,0.82000988,0.5335631425);
return cl::sycl::tanh(inputData_0);

});

test_function<1191, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.7806404701,0.4385333793,0.3917197084,0.8483145171);
return cl::sycl::tanh(inputData_0);

});

test_function<1192, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7825211638,0.8129809908,0.4461560838,0.1789903326,0.1752695318,0.2276761885,0.3780952326,0.5866617641);
return cl::sycl::tanh(inputData_0);

});

test_function<1193, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.4346112795,0.4444608814,0.544766728,0.6439705078,0.8117103982,0.3414500116,0.7473899465,0.6051101083,0.2364201653,0.5240329435,0.4112300822,0.4299367859,0.3304578672,0.1970362694,0.6504919964,0.5842010946);
return cl::sycl::tanh(inputData_0);

});

test_function<1194, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.64710781,0.7249347115);
return cl::sycl::tanh(inputData_0);

});

test_function<1195, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8176520254,0.1886854108,0.7789220082);
return cl::sycl::tanh(inputData_0);

});

test_function<1196, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.7285819067,0.227475234,0.1662103529,0.2132030084);
return cl::sycl::tanh(inputData_0);

});

test_function<1197, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8031556099,0.8221602043,0.7177780602,0.4553020911,0.462065013,0.4624569715,0.112414891,0.4890022949);
return cl::sycl::tanh(inputData_0);

});

test_function<1198, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.6333999925,0.5563950437,0.447323904,0.1735526721,0.3916403888,0.2927443721,0.2795362556,0.3287374463,0.3771993315,0.3691388611,0.4575230267,0.6164827179,0.3568863645,0.6952025727,0.7736576903,0.3622840922);
return cl::sycl::tanh(inputData_0);

});

test_function<1199, float>(
[=](){
float inputData_0(0.4176455714);
return cl::sycl::tanpi(inputData_0);

});

test_function<1200, double>(
[=](){
double inputData_0(0.3830050643);
return cl::sycl::tanpi(inputData_0);

});

test_function<1201, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8327566102);
return cl::sycl::tanpi(inputData_0);

});

test_function<1202, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.8468412807,0.3095318167);
return cl::sycl::tanpi(inputData_0);

});

test_function<1203, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.5205671606,0.8275135497,0.4831132449);
return cl::sycl::tanpi(inputData_0);

});

test_function<1204, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6513932989,0.8371040007,0.4356475882,0.1844177834);
return cl::sycl::tanpi(inputData_0);

});

test_function<1205, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6854058491,0.8185038907,0.7836211504,0.8951831521,0.3492814293,0.8524468793,0.2739047624,0.5496794948);
return cl::sycl::tanpi(inputData_0);

});

test_function<1206, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7750610601,0.1763327389,0.5126465043,0.4290339089,0.3937938056,0.3982875971,0.4672728557,0.8637178226,0.3948544981,0.4314406993,0.1320195943,0.3604876472,0.5379002312,0.8591511161,0.6364228495,0.8519004191);
return cl::sycl::tanpi(inputData_0);

});

test_function<1207, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4144773711,0.2222538435);
return cl::sycl::tanpi(inputData_0);

});

test_function<1208, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6399479667,0.3434798131,0.2099235129);
return cl::sycl::tanpi(inputData_0);

});

test_function<1209, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.2197220659,0.7721924972,0.4192695893,0.864371573);
return cl::sycl::tanpi(inputData_0);

});

test_function<1210, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6835514339,0.5197870375,0.7635185883,0.1856583671,0.4791466737,0.6146985786,0.2906970868,0.2148377815);
return cl::sycl::tanpi(inputData_0);

});

test_function<1211, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6277001306,0.752009702,0.3984469257,0.3475233658,0.3035205772,0.7292025565,0.152091831,0.3510113186,0.8802644612,0.648208845,0.2687999663,0.2016768707,0.4218544965,0.8904697255,0.8427777089,0.5475155825);
return cl::sycl::tanpi(inputData_0);

});

test_function<1212, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.2169223898,0.7751927846);
return cl::sycl::tanpi(inputData_0);

});

test_function<1213, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.425994061,0.1302403497,0.5046837462);
return cl::sycl::tanpi(inputData_0);

});

test_function<1214, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.2353886633,0.1012331613,0.3737891396,0.6487979498);
return cl::sycl::tanpi(inputData_0);

});

test_function<1215, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.364090101,0.6519735434,0.7483132211,0.6875033636,0.6206784112,0.3664551895,0.7602393479,0.8892990411);
return cl::sycl::tanpi(inputData_0);

});

test_function<1216, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8409407371,0.5399747466,0.5173990261,0.2891383015,0.2374137767,0.4164871635,0.1378894077,0.5528863812,0.3020172147,0.2901708167,0.140132525,0.4290776458,0.1455799841,0.2700291658,0.4352763517,0.8910279275);
return cl::sycl::tanpi(inputData_0);

});

test_function<1217, float>(
[=](){
float inputData_0(0.4923368169);
return cl::sycl::tgamma(inputData_0);

});

test_function<1218, double>(
[=](){
double inputData_0(0.2037283697);
return cl::sycl::tgamma(inputData_0);

});

test_function<1219, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8029534021);
return cl::sycl::tgamma(inputData_0);

});

test_function<1220, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1632144086,0.7239608292);
return cl::sycl::tgamma(inputData_0);

});

test_function<1221, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.6917633359,0.3848288719,0.6838441071);
return cl::sycl::tgamma(inputData_0);

});

test_function<1222, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.1339240436,0.5489710223,0.349408842,0.4792628601);
return cl::sycl::tgamma(inputData_0);

});

test_function<1223, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2652137379,0.2816791554,0.5846488397,0.3897668464,0.1596885498,0.8821963921,0.7146064081,0.5554316193);
return cl::sycl::tgamma(inputData_0);

});

test_function<1224, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4973804481,0.1971533366,0.4136029556,0.7720674056,0.513426299,0.3233908979,0.4427028609,0.6691351682,0.5561623494,0.8480399657,0.4517067515,0.8478681922,0.7445187508,0.2159275017,0.5450541433,0.5892413812);
return cl::sycl::tgamma(inputData_0);

});

test_function<1225, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7813876022,0.7165297574);
return cl::sycl::tgamma(inputData_0);

});

test_function<1226, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2779915692,0.3013432514,0.5958438192);
return cl::sycl::tgamma(inputData_0);

});

test_function<1227, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3442081164,0.2567372582,0.8495676773,0.8185395064);
return cl::sycl::tgamma(inputData_0);

});

test_function<1228, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.1663913062,0.6552357181,0.7709398622,0.7378002552,0.8957428068,0.6983272437,0.111042214,0.7080195459);
return cl::sycl::tgamma(inputData_0);

});

test_function<1229, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1094449176,0.8278779753,0.8776443874,0.8588022472,0.3853720525,0.8524193536,0.1042580552,0.5572006248,0.194360142,0.5415753353,0.2036778905,0.5207702193,0.6668237468,0.646363367,0.8078057232,0.7190483265);
return cl::sycl::tgamma(inputData_0);

});

test_function<1230, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8270458032,0.6136124774);
return cl::sycl::tgamma(inputData_0);

});

test_function<1231, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1233034867,0.4924126415,0.1460435675);
return cl::sycl::tgamma(inputData_0);

});

test_function<1232, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1316947095,0.4400904236,0.535311558,0.6839205111);
return cl::sycl::tgamma(inputData_0);

});

test_function<1233, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8632168338,0.6084408949,0.165984602,0.5143588496,0.4942797954,0.2397094062,0.5316211988,0.1904130226);
return cl::sycl::tgamma(inputData_0);

});

test_function<1234, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8043451422,0.4500376341,0.2147751452,0.5658183277,0.2225511324,0.7857922927,0.6648539621,0.2766883103,0.5666713163,0.3582856553,0.2241956461,0.1573033773,0.7066728235,0.522979127,0.7978314429,0.2864919947);
return cl::sycl::tgamma(inputData_0);

});

test_function<1235, float>(
[=](){
float inputData_0(0.1474725189);
return cl::sycl::trunc(inputData_0);

});

test_function<1236, double>(
[=](){
double inputData_0(0.6261761152);
return cl::sycl::trunc(inputData_0);

});

test_function<1237, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1742237006);
return cl::sycl::trunc(inputData_0);

});

test_function<1238, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5952561672,0.4949523837);
return cl::sycl::trunc(inputData_0);

});

test_function<1239, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3688368078,0.6394794796,0.8735138861);
return cl::sycl::trunc(inputData_0);

});

test_function<1240, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.398913743,0.2650297806,0.133157389,0.6053205491);
return cl::sycl::trunc(inputData_0);

});

test_function<1241, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2677927037,0.7137759168,0.6335528685,0.8373989257,0.4730250617,0.3644020448,0.2204924032,0.5721156427);
return cl::sycl::trunc(inputData_0);

});

test_function<1242, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5878059348,0.2731338197,0.4016341017,0.8335798463,0.8685739125,0.7045566342,0.6079432529,0.7815374314,0.3968283997,0.3680807889,0.1088007214,0.8986209534,0.8521296379,0.7932289814,0.5093116902,0.658518677);
return cl::sycl::trunc(inputData_0);

});

test_function<1243, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.6243154735,0.6098600313);
return cl::sycl::trunc(inputData_0);

});

test_function<1244, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7439015779,0.2306190803,0.2884744149);
return cl::sycl::trunc(inputData_0);

});

test_function<1245, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3364910257,0.6664152033,0.8922920621,0.7320226551);
return cl::sycl::trunc(inputData_0);

});

test_function<1246, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4785040645,0.737448378,0.8705201292,0.3102192963,0.4493637051,0.3810119038,0.7941216143,0.1369239048);
return cl::sycl::trunc(inputData_0);

});

test_function<1247, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3112526948,0.7458100205,0.5877981296,0.4384380854,0.4653607091,0.6828034135,0.5363226625,0.334741852,0.8481348984,0.4452182744,0.2990696196,0.4766950234,0.8245571261,0.8407496499,0.3493253737,0.2322225223);
return cl::sycl::trunc(inputData_0);

});

test_function<1248, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8616964061,0.3349002938);
return cl::sycl::trunc(inputData_0);

});

test_function<1249, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.3685764303,0.8217832602,0.349493526);
return cl::sycl::trunc(inputData_0);

});

test_function<1250, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.5848753344,0.28193048,0.4203620264,0.2874774868);
return cl::sycl::trunc(inputData_0);

});

test_function<1251, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.187845683,0.67640028,0.1558777194,0.5975197141,0.1639810632,0.4938553912,0.7007878438,0.6582340303);
return cl::sycl::trunc(inputData_0);

});

test_function<1252, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8900271346,0.5814315029,0.4312856861,0.1667910271,0.7559838889,0.5999508245,0.6573574084,0.3067646183,0.885051105,0.4865847159,0.6374883868,0.6226754525,0.261223567,0.782905874,0.1914740248,0.4027308708);
return cl::sycl::trunc(inputData_0);

});

 }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}