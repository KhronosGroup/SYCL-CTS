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

  void run(util::logger &log) override { test_function<0, float>(
[=](){
float inputData_0(0.7755374812);
return cl::sycl::half_precision::cos(inputData_0);

});

test_function<1, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7063635224,0.4364572647);
return cl::sycl::half_precision::cos(inputData_0);

});

test_function<2, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3071334002,0.5090197771,0.42394731);
return cl::sycl::half_precision::cos(inputData_0);

});

test_function<3, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7270388712,0.3426501809,0.4812775633,0.5667056316);
return cl::sycl::half_precision::cos(inputData_0);

});

test_function<4, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8264903082,0.5037494847,0.3254702755,0.7046433633,0.5946951973,0.3004050731,0.8277970048,0.8862283808);
return cl::sycl::half_precision::cos(inputData_0);

});

test_function<5, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7481737888,0.8217327604,0.3481180555,0.6838653986,0.8190706304,0.6471871455,0.4777141724,0.1805609665,0.4473374684,0.5887095788,0.8304088426,0.8732850942,0.4816078212,0.7922479422,0.3083938483,0.7440222616);
return cl::sycl::half_precision::cos(inputData_0);

});

test_function<6, float>(
[=](){
float inputData_0(0.5389594431);
float inputData_1(0.1112333601);
return cl::sycl::half_precision::divide(inputData_0,inputData_1);

});

test_function<7, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6757637491,0.4190588338);
cl::sycl::float2 inputData_1(0.7598759817,0.634522561);
return cl::sycl::half_precision::divide(inputData_0,inputData_1);

});

test_function<8, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1009142555,0.4948622932,0.7940822204);
cl::sycl::float3 inputData_1(0.2951287015,0.3601634902,0.7963769857);
return cl::sycl::half_precision::divide(inputData_0,inputData_1);

});

test_function<9, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2528536732,0.5540085925,0.2908927429,0.8740322002);
cl::sycl::float4 inputData_1(0.7425435754,0.4583756571,0.1643566548,0.3560436837);
return cl::sycl::half_precision::divide(inputData_0,inputData_1);

});

test_function<10, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.506352514,0.8462670594,0.1872462767,0.5410137969,0.6652491279,0.5379527291,0.7515734906,0.5322268856);
cl::sycl::float8 inputData_1(0.8710708368,0.5825485024,0.5700936513,0.455991221,0.5770294893,0.4079209168,0.5605208113,0.3322636019);
return cl::sycl::half_precision::divide(inputData_0,inputData_1);

});

test_function<11, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2515130628,0.2493836226,0.5902185439,0.6253275112,0.4812247936,0.171859489,0.7060831376,0.8014162967,0.8387048128,0.7739681785,0.8185384971,0.8384659519,0.53247994,0.4130368402,0.6642267199,0.320507297);
cl::sycl::float16 inputData_1(0.7493029668,0.7795887721,0.8160311739,0.5718409468,0.8598118986,0.5637560086,0.4604504853,0.6281963029,0.8970062715,0.8335529744,0.7346600673,0.1658983906,0.590226484,0.4891553616,0.6041178723,0.7760620605);
return cl::sycl::half_precision::divide(inputData_0,inputData_1);

});

test_function<12, float>(
[=](){
float inputData_0(0.2944284976);
return cl::sycl::half_precision::exp(inputData_0);

});

test_function<13, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6851913766,0.1937074346);
return cl::sycl::half_precision::exp(inputData_0);

});

test_function<14, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2763684295,0.7356663774,0.3660289194);
return cl::sycl::half_precision::exp(inputData_0);

});

test_function<15, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7527304772,0.1804860162,0.2170867911,0.6581365122);
return cl::sycl::half_precision::exp(inputData_0);

});

test_function<16, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1361872543,0.5590928294,0.8280128118,0.5273583746,0.644471306,0.1213574357,0.6079999279,0.5850707342);
return cl::sycl::half_precision::exp(inputData_0);

});

test_function<17, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5607623584,0.4129675275,0.3961119523,0.8844133205,0.1291136301,0.1173092079,0.8688250242,0.2479775531,0.1991161315,0.2684612079,0.7405972723,0.8495753269,0.1182260605,0.4404950656,0.1812001755,0.3079359118);
return cl::sycl::half_precision::exp(inputData_0);

});

test_function<18, float>(
[=](){
float inputData_0(0.2766634171);
return cl::sycl::half_precision::exp2(inputData_0);

});

test_function<19, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6175405759,0.3802351739);
return cl::sycl::half_precision::exp2(inputData_0);

});

test_function<20, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2442543212,0.5029092042,0.1315029657);
return cl::sycl::half_precision::exp2(inputData_0);

});

test_function<21, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.180736993,0.890588119,0.2594846324,0.386844241);
return cl::sycl::half_precision::exp2(inputData_0);

});

test_function<22, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.685278645,0.7706612522,0.8347856496,0.2355396849,0.6381124509,0.8732391224,0.1464407551,0.6409614274);
return cl::sycl::half_precision::exp2(inputData_0);

});

test_function<23, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.776339675,0.3738500329,0.3005498714,0.5774331148,0.453851227,0.2398555876,0.4773003321,0.4279243165,0.5552901916,0.5068801041,0.3491568008,0.3857213461,0.7701289395,0.3007461319,0.5484801751,0.1099490551);
return cl::sycl::half_precision::exp2(inputData_0);

});

test_function<24, float>(
[=](){
float inputData_0(0.6932595019);
return cl::sycl::half_precision::exp10(inputData_0);

});

test_function<25, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3687332436,0.1365571949);
return cl::sycl::half_precision::exp10(inputData_0);

});

test_function<26, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3247065314,0.2921043263,0.8625034719);
return cl::sycl::half_precision::exp10(inputData_0);

});

test_function<27, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3817804492,0.3303023319,0.3873609578,0.8575246685);
return cl::sycl::half_precision::exp10(inputData_0);

});

test_function<28, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6069982818,0.5968614765,0.6724954802,0.4104137883,0.4315343906,0.6206662898,0.1012193775,0.253847633);
return cl::sycl::half_precision::exp10(inputData_0);

});

test_function<29, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3675213525,0.2915327681,0.6099195209,0.4029184563,0.8003387134,0.5545211367,0.4315251173,0.4218136601,0.6614636991,0.4345812426,0.6297567112,0.1374237488,0.4562817518,0.3073815388,0.2261492577,0.5220585041);
return cl::sycl::half_precision::exp10(inputData_0);

});

test_function<30, float>(
[=](){
float inputData_0(0.4898124809);
return cl::sycl::half_precision::log(inputData_0);

});

test_function<31, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5491239405,0.7043878138);
return cl::sycl::half_precision::log(inputData_0);

});

test_function<32, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8071001234,0.4956661363,0.3496465971);
return cl::sycl::half_precision::log(inputData_0);

});

test_function<33, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4735137883,0.7472366859,0.8000130652,0.7499319459);
return cl::sycl::half_precision::log(inputData_0);

});

test_function<34, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.2504010352,0.8995362876,0.6064710079,0.1667736401,0.6804434844,0.8894571842,0.4214534578,0.6428120042);
return cl::sycl::half_precision::log(inputData_0);

});

test_function<35, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.3529417098,0.2708197297,0.6738593146,0.1018860518,0.7581851284,0.5226767815,0.1782274734,0.1951231158,0.6194123399,0.7989230591,0.3239861947,0.8828121494,0.1801445513,0.7831504877,0.4173569419,0.1650763334);
return cl::sycl::half_precision::log(inputData_0);

});

test_function<36, float>(
[=](){
float inputData_0(0.3197710747);
return cl::sycl::half_precision::log2(inputData_0);

});

test_function<37, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4623825479,0.7338732249);
return cl::sycl::half_precision::log2(inputData_0);

});

test_function<38, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7890879229,0.2067364434,0.5166924227);
return cl::sycl::half_precision::log2(inputData_0);

});

test_function<39, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.6206265905,0.3776424117,0.7974910686,0.3227278522);
return cl::sycl::half_precision::log2(inputData_0);

});

test_function<40, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.114859462,0.1325306189,0.6447974161,0.5466845889,0.8572020433,0.8507510398,0.8278809419,0.1336036256);
return cl::sycl::half_precision::log2(inputData_0);

});

test_function<41, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6993078587,0.6610598541,0.6242894917,0.669886122,0.8221681205,0.6121129598,0.3979594104,0.530343027,0.266275283,0.5697004038,0.1071176656,0.2208185391,0.3667267104,0.7316985271,0.6747995382,0.370604776);
return cl::sycl::half_precision::log2(inputData_0);

});

test_function<42, float>(
[=](){
float inputData_0(0.5964304867);
return cl::sycl::half_precision::log10(inputData_0);

});

test_function<43, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1329623596,0.2310884365);
return cl::sycl::half_precision::log10(inputData_0);

});

test_function<44, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8855312561,0.3316246829,0.4158335864);
return cl::sycl::half_precision::log10(inputData_0);

});

test_function<45, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5387874373,0.3347256012,0.4824517353,0.2917648669);
return cl::sycl::half_precision::log10(inputData_0);

});

test_function<46, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1386050898,0.2436694792,0.5184401854,0.1566903073,0.4225353172,0.362816568,0.4317772872,0.1795202706);
return cl::sycl::half_precision::log10(inputData_0);

});

test_function<47, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8269260435,0.4792037209,0.7726786661,0.8809835661,0.3749212749,0.4832692153,0.6596762329,0.4412282588,0.341522493,0.687800793,0.8155198226,0.8357510755,0.6013936374,0.4004570771,0.8796484172,0.611102814);
return cl::sycl::half_precision::log10(inputData_0);

});

test_function<48, float>(
[=](){
float inputData_0(0.1526677418);
float inputData_1(0.1677356553);
return cl::sycl::half_precision::powr(inputData_0,inputData_1);

});

test_function<49, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.6998956574,0.1489249252);
cl::sycl::float2 inputData_1(0.1062808043,0.4150463614);
return cl::sycl::half_precision::powr(inputData_0,inputData_1);

});

test_function<50, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.515202983,0.4588354285,0.4908950435);
cl::sycl::float3 inputData_1(0.5679109616,0.6434420539,0.4384304588);
return cl::sycl::half_precision::powr(inputData_0,inputData_1);

});

test_function<51, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.3946651651,0.8907672465,0.3087332284,0.7216801236);
cl::sycl::float4 inputData_1(0.4449768197,0.3868163056,0.1510863592,0.7908631554);
return cl::sycl::half_precision::powr(inputData_0,inputData_1);

});

test_function<52, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6616033198,0.822408566,0.4612894341,0.6415367735,0.1951282292,0.4183628813,0.2657855787,0.1336811423);
cl::sycl::float8 inputData_1(0.858369081,0.2727154948,0.2170835918,0.2583760348,0.4024255715,0.5371130099,0.2210674948,0.8909519112);
return cl::sycl::half_precision::powr(inputData_0,inputData_1);

});

test_function<53, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8863913684,0.2187216137,0.4247255065,0.6439435865,0.8021252663,0.4963247399,0.8336373382,0.3579682519,0.4987527132,0.4989172735,0.6360545211,0.261593047,0.5878164883,0.2750184775,0.372176252,0.8700531706);
cl::sycl::float16 inputData_1(0.8192064304,0.7544947047,0.1283746095,0.218693506,0.3055055297,0.7273332546,0.7738666617,0.5663585442,0.6745053214,0.745644304,0.1530873048,0.1677145095,0.7951162512,0.1315326635,0.2800725229,0.1325056213);
return cl::sycl::half_precision::powr(inputData_0,inputData_1);

});

test_function<54, float>(
[=](){
float inputData_0(0.112228112);
return cl::sycl::half_precision::recip(inputData_0);

});

test_function<55, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7751637486,0.3644754938);
return cl::sycl::half_precision::recip(inputData_0);

});

test_function<56, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2285520482,0.2190555922,0.6248669294);
return cl::sycl::half_precision::recip(inputData_0);

});

test_function<57, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8748786174,0.5039997541,0.8208723815,0.5019428792);
return cl::sycl::half_precision::recip(inputData_0);

});

test_function<58, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.559097982,0.6428570854,0.7440879912,0.7062771058,0.8924260502,0.6975723113,0.8246245787,0.2648838657);
return cl::sycl::half_precision::recip(inputData_0);

});

test_function<59, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5283330435,0.5788914109,0.7605572937,0.4857708505,0.7328321694,0.4108551121,0.5691107645,0.781053286,0.7384475769,0.6255876415,0.1001925572,0.2455751377,0.5054862295,0.3035675188,0.1524966746,0.7879067377);
return cl::sycl::half_precision::recip(inputData_0);

});

test_function<60, float>(
[=](){
float inputData_0(0.8543576171);
return cl::sycl::half_precision::rsqrt(inputData_0);

});

test_function<61, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3422439025,0.4264585339);
return cl::sycl::half_precision::rsqrt(inputData_0);

});

test_function<62, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7480300271,0.1498070071,0.61278789);
return cl::sycl::half_precision::rsqrt(inputData_0);

});

test_function<63, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2018566503,0.329670672,0.7639525493,0.1444216367);
return cl::sycl::half_precision::rsqrt(inputData_0);

});

test_function<64, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1287470667,0.4342928358,0.4934647673,0.7906601465,0.6737509971,0.6388350469,0.2210990179,0.8893647394);
return cl::sycl::half_precision::rsqrt(inputData_0);

});

test_function<65, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.428912157,0.5894166915,0.4093464044,0.1376263326,0.4767113672,0.2210942031,0.1259723699,0.5939203389,0.603973033,0.1842342597,0.539315013,0.3773343813,0.4067312585,0.721135919,0.492255742,0.8050212923);
return cl::sycl::half_precision::rsqrt(inputData_0);

});

test_function<66, float>(
[=](){
float inputData_0(0.5880957943);
return cl::sycl::half_precision::sin(inputData_0);

});

test_function<67, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.473750732,0.605850112);
return cl::sycl::half_precision::sin(inputData_0);

});

test_function<68, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3702923039,0.199459034,0.646023695);
return cl::sycl::half_precision::sin(inputData_0);

});

test_function<69, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5976299542,0.7308531931,0.2016873,0.8294266545);
return cl::sycl::half_precision::sin(inputData_0);

});

test_function<70, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7394729691,0.8335099265,0.7980277774,0.6448051571,0.7482006795,0.5152058474,0.7283913195,0.2513019743);
return cl::sycl::half_precision::sin(inputData_0);

});

test_function<71, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7256912851,0.4556636832,0.705292977,0.4643761894,0.7316469826,0.1602716682,0.1357127243,0.8474316659,0.4889320806,0.8208571197,0.8558266015,0.633208922,0.5574374609,0.2727835073,0.1747809754,0.7555153721);
return cl::sycl::half_precision::sin(inputData_0);

});

test_function<72, float>(
[=](){
float inputData_0(0.8110176541);
return cl::sycl::half_precision::sqrt(inputData_0);

});

test_function<73, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.7235165686,0.6588019462);
return cl::sycl::half_precision::sqrt(inputData_0);

});

test_function<74, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4360888929,0.344249272,0.1907559165);
return cl::sycl::half_precision::sqrt(inputData_0);

});

test_function<75, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4407761985,0.5528103794,0.8383044665,0.8486038155);
return cl::sycl::half_precision::sqrt(inputData_0);

});

test_function<76, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4325129572,0.1793687905,0.719054986,0.6874234733,0.1245606768,0.4573748793,0.6491344834,0.1241073876);
return cl::sycl::half_precision::sqrt(inputData_0);

});

test_function<77, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8354258827,0.8697939892,0.6780342177,0.1628308317,0.1562635727,0.3874026519,0.1235020062,0.3783021818,0.1079713931,0.8794588103,0.7552053593,0.1564140892,0.8147480735,0.266382432,0.2638326386,0.6390073164);
return cl::sycl::half_precision::sqrt(inputData_0);

});

test_function<78, float>(
[=](){
float inputData_0(0.8506098145);
return cl::sycl::half_precision::tan(inputData_0);

});

test_function<79, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.198550497,0.1057476538);
return cl::sycl::half_precision::tan(inputData_0);

});

test_function<80, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3953041177,0.1197200115,0.5838785901);
return cl::sycl::half_precision::tan(inputData_0);

});

test_function<81, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7873404869,0.2495933619,0.1899128287,0.3755596859);
return cl::sycl::half_precision::tan(inputData_0);

});

test_function<82, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8673372165,0.2041261555,0.8732154084,0.389791896,0.4786963222,0.3341055888,0.8497014754,0.866518316);
return cl::sycl::half_precision::tan(inputData_0);

});

test_function<83, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6087325652,0.2472364401,0.8943614309,0.1820643516,0.5646795053,0.2251224481,0.8181402513,0.8565427132,0.7435122384,0.3527131349,0.294270952,0.7038867306,0.3328476153,0.4358283023,0.1370045415,0.2057870483);
return cl::sycl::half_precision::tan(inputData_0);

});

 }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}