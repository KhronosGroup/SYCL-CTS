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
float inputData_1(0.7063635224);
float inputData_2(0.4364572647);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<1, double>(
[=](){
double inputData_0(0.3071334002);
double inputData_1(0.5090197771);
double inputData_2(0.42394731);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<2, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7270388712);
cl::sycl::half inputData_1(0.3426501809);
cl::sycl::half inputData_2(0.4812775633);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<3, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5667056316,0.8264903082);
cl::sycl::float2 inputData_1(0.5037494847,0.3254702755);
cl::sycl::float2 inputData_2(0.7046433633,0.5946951973);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<4, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3004050731,0.8277970048,0.8862283808);
cl::sycl::float3 inputData_1(0.7481737888,0.8217327604,0.3481180555);
cl::sycl::float3 inputData_2(0.6838653986,0.8190706304,0.6471871455);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<5, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.4777141724,0.1805609665,0.4473374684,0.5887095788);
cl::sycl::float4 inputData_1(0.8304088426,0.8732850942,0.4816078212,0.7922479422);
cl::sycl::float4 inputData_2(0.3083938483,0.7440222616,0.5389594431,0.1112333601);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<6, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6757637491,0.4190588338,0.7598759817,0.634522561,0.1009142555,0.4948622932,0.7940822204,0.2951287015);
cl::sycl::float8 inputData_1(0.3601634902,0.7963769857,0.2528536732,0.5540085925,0.2908927429,0.8740322002,0.7425435754,0.4583756571);
cl::sycl::float8 inputData_2(0.1643566548,0.3560436837,0.506352514,0.8462670594,0.1872462767,0.5410137969,0.6652491279,0.5379527291);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<7, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7515734906,0.5322268856,0.8710708368,0.5825485024,0.5700936513,0.455991221,0.5770294893,0.4079209168,0.5605208113,0.3322636019,0.2515130628,0.2493836226,0.5902185439,0.6253275112,0.4812247936,0.171859489);
cl::sycl::float16 inputData_1(0.7060831376,0.8014162967,0.8387048128,0.7739681785,0.8185384971,0.8384659519,0.53247994,0.4130368402,0.6642267199,0.320507297,0.7493029668,0.7795887721,0.8160311739,0.5718409468,0.8598118986,0.5637560086);
cl::sycl::float16 inputData_2(0.4604504853,0.6281963029,0.8970062715,0.8335529744,0.7346600673,0.1658983906,0.590226484,0.4891553616,0.6041178723,0.7760620605,0.2944284976,0.6851913766,0.1937074346,0.2763684295,0.7356663774,0.3660289194);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<8, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7527304772,0.1804860162);
cl::sycl::double2 inputData_1(0.2170867911,0.6581365122);
cl::sycl::double2 inputData_2(0.1361872543,0.5590928294);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<9, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.8280128118,0.5273583746,0.644471306);
cl::sycl::double3 inputData_1(0.1213574357,0.6079999279,0.5850707342);
cl::sycl::double3 inputData_2(0.5607623584,0.4129675275,0.3961119523);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<10, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.8844133205,0.1291136301,0.1173092079,0.8688250242);
cl::sycl::double4 inputData_1(0.2479775531,0.1991161315,0.2684612079,0.7405972723);
cl::sycl::double4 inputData_2(0.8495753269,0.1182260605,0.4404950656,0.1812001755);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<11, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.3079359118,0.2766634171,0.6175405759,0.3802351739,0.2442543212,0.5029092042,0.1315029657,0.180736993);
cl::sycl::double8 inputData_1(0.890588119,0.2594846324,0.386844241,0.685278645,0.7706612522,0.8347856496,0.2355396849,0.6381124509);
cl::sycl::double8 inputData_2(0.8732391224,0.1464407551,0.6409614274,0.776339675,0.3738500329,0.3005498714,0.5774331148,0.453851227);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<12, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2398555876,0.4773003321,0.4279243165,0.5552901916,0.5068801041,0.3491568008,0.3857213461,0.7701289395,0.3007461319,0.5484801751,0.1099490551,0.6932595019,0.3687332436,0.1365571949,0.3247065314,0.2921043263);
cl::sycl::double16 inputData_1(0.8625034719,0.3817804492,0.3303023319,0.3873609578,0.8575246685,0.6069982818,0.5968614765,0.6724954802,0.4104137883,0.4315343906,0.6206662898,0.1012193775,0.253847633,0.3675213525,0.2915327681,0.6099195209);
cl::sycl::double16 inputData_2(0.4029184563,0.8003387134,0.5545211367,0.4315251173,0.4218136601,0.6614636991,0.4345812426,0.6297567112,0.1374237488,0.4562817518,0.3073815388,0.2261492577,0.5220585041,0.4898124809,0.5491239405,0.7043878138);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<13, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.8071001234,0.4956661363);
cl::sycl::half2 inputData_1(0.3496465971,0.4735137883);
cl::sycl::half2 inputData_2(0.7472366859,0.8000130652);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<14, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7499319459,0.2504010352,0.8995362876);
cl::sycl::half3 inputData_1(0.6064710079,0.1667736401,0.6804434844);
cl::sycl::half3 inputData_2(0.8894571842,0.4214534578,0.6428120042);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<15, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.3529417098,0.2708197297,0.6738593146,0.1018860518);
cl::sycl::half4 inputData_1(0.7581851284,0.5226767815,0.1782274734,0.1951231158);
cl::sycl::half4 inputData_2(0.6194123399,0.7989230591,0.3239861947,0.8828121494);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<16, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1801445513,0.7831504877,0.4173569419,0.1650763334,0.3197710747,0.4623825479,0.7338732249,0.7890879229);
cl::sycl::half8 inputData_1(0.2067364434,0.5166924227,0.6206265905,0.3776424117,0.7974910686,0.3227278522,0.114859462,0.1325306189);
cl::sycl::half8 inputData_2(0.6447974161,0.5466845889,0.8572020433,0.8507510398,0.8278809419,0.1336036256,0.6993078587,0.6610598541);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<17, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.6242894917,0.669886122,0.8221681205,0.6121129598,0.3979594104,0.530343027,0.266275283,0.5697004038,0.1071176656,0.2208185391,0.3667267104,0.7316985271,0.6747995382,0.370604776,0.5964304867,0.1329623596);
cl::sycl::half16 inputData_1(0.2310884365,0.8855312561,0.3316246829,0.4158335864,0.5387874373,0.3347256012,0.4824517353,0.2917648669,0.1386050898,0.2436694792,0.5184401854,0.1566903073,0.4225353172,0.362816568,0.4317772872,0.1795202706);
cl::sycl::half16 inputData_2(0.8269260435,0.4792037209,0.7726786661,0.8809835661,0.3749212749,0.4832692153,0.6596762329,0.4412282588,0.341522493,0.687800793,0.8155198226,0.8357510755,0.6013936374,0.4004570771,0.8796484172,0.611102814);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<18, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1526677418,0.1677356553);
float inputData_1(0.6998956574);
float inputData_2(0.1489249252);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<19, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1062808043,0.4150463614,0.515202983);
float inputData_1(0.4588354285);
float inputData_2(0.4908950435);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<20, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5679109616,0.6434420539,0.4384304588,0.3946651651);
float inputData_1(0.8907672465);
float inputData_2(0.3087332284);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<21, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7216801236,0.4449768197,0.3868163056,0.1510863592,0.7908631554,0.6616033198,0.822408566,0.4612894341);
float inputData_1(0.6415367735);
float inputData_2(0.1951282292);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<22, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4183628813,0.2657855787,0.1336811423,0.858369081,0.2727154948,0.2170835918,0.2583760348,0.4024255715,0.5371130099,0.2210674948,0.8909519112,0.8863913684,0.2187216137,0.4247255065,0.6439435865,0.8021252663);
float inputData_1(0.4963247399);
float inputData_2(0.8336373382);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<23, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3579682519,0.4987527132);
double inputData_1(0.4989172735);
double inputData_2(0.6360545211);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<24, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.261593047,0.5878164883,0.2750184775);
double inputData_1(0.372176252);
double inputData_2(0.8700531706);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<25, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.8192064304,0.7544947047,0.1283746095,0.218693506);
double inputData_1(0.3055055297);
double inputData_2(0.7273332546);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<26, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7738666617,0.5663585442,0.6745053214,0.745644304,0.1530873048,0.1677145095,0.7951162512,0.1315326635);
double inputData_1(0.2800725229);
double inputData_2(0.1325056213);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<27, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.112228112,0.7751637486,0.3644754938,0.2285520482,0.2190555922,0.6248669294,0.8748786174,0.5039997541,0.8208723815,0.5019428792,0.559097982,0.6428570854,0.7440879912,0.7062771058,0.8924260502,0.6975723113);
double inputData_1(0.8246245787);
double inputData_2(0.2648838657);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<28, float>(
[=](){
float inputData_0(0.5283330435);
return cl::sycl::degrees(inputData_0);

});

test_function<29, double>(
[=](){
double inputData_0(0.5788914109);
return cl::sycl::degrees(inputData_0);

});

test_function<30, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7605572937);
return cl::sycl::degrees(inputData_0);

});

test_function<31, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.4857708505,0.7328321694);
return cl::sycl::degrees(inputData_0);

});

test_function<32, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.4108551121,0.5691107645,0.781053286);
return cl::sycl::degrees(inputData_0);

});

test_function<33, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7384475769,0.6255876415,0.1001925572,0.2455751377);
return cl::sycl::degrees(inputData_0);

});

test_function<34, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.5054862295,0.3035675188,0.1524966746,0.7879067377,0.8543576171,0.3422439025,0.4264585339,0.7480300271);
return cl::sycl::degrees(inputData_0);

});

test_function<35, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.1498070071,0.61278789,0.2018566503,0.329670672,0.7639525493,0.1444216367,0.1287470667,0.4342928358,0.4934647673,0.7906601465,0.6737509971,0.6388350469,0.2210990179,0.8893647394,0.428912157,0.5894166915);
return cl::sycl::degrees(inputData_0);

});

test_function<36, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4093464044,0.1376263326);
return cl::sycl::degrees(inputData_0);

});

test_function<37, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4767113672,0.2210942031,0.1259723699);
return cl::sycl::degrees(inputData_0);

});

test_function<38, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5939203389,0.603973033,0.1842342597,0.539315013);
return cl::sycl::degrees(inputData_0);

});

test_function<39, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.3773343813,0.4067312585,0.721135919,0.492255742,0.8050212923,0.5880957943,0.473750732,0.605850112);
return cl::sycl::degrees(inputData_0);

});

test_function<40, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.3702923039,0.199459034,0.646023695,0.5976299542,0.7308531931,0.2016873,0.8294266545,0.7394729691,0.8335099265,0.7980277774,0.6448051571,0.7482006795,0.5152058474,0.7283913195,0.2513019743,0.7256912851);
return cl::sycl::degrees(inputData_0);

});

test_function<41, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.4556636832,0.705292977);
return cl::sycl::degrees(inputData_0);

});

test_function<42, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.4643761894,0.7316469826,0.1602716682);
return cl::sycl::degrees(inputData_0);

});

test_function<43, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1357127243,0.8474316659,0.4889320806,0.8208571197);
return cl::sycl::degrees(inputData_0);

});

test_function<44, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.8558266015,0.633208922,0.5574374609,0.2727835073,0.1747809754,0.7555153721,0.8110176541,0.7235165686);
return cl::sycl::degrees(inputData_0);

});

test_function<45, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.6588019462,0.4360888929,0.344249272,0.1907559165,0.4407761985,0.5528103794,0.8383044665,0.8486038155,0.4325129572,0.1793687905,0.719054986,0.6874234733,0.1245606768,0.4573748793,0.6491344834,0.1241073876);
return cl::sycl::degrees(inputData_0);

});

test_function<46, float>(
[=](){
float inputData_0(0.8354258827);
float inputData_1(0.8697939892);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<47, double>(
[=](){
double inputData_0(0.6780342177);
double inputData_1(0.1628308317);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<48, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.1562635727);
cl::sycl::half inputData_1(0.3874026519);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<49, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.1235020062,0.3783021818);
cl::sycl::float2 inputData_1(0.1079713931,0.8794588103);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<50, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.7552053593,0.1564140892,0.8147480735);
cl::sycl::float3 inputData_1(0.266382432,0.2638326386,0.6390073164);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<51, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8506098145,0.198550497,0.1057476538,0.3953041177);
cl::sycl::float4 inputData_1(0.1197200115,0.5838785901,0.7873404869,0.2495933619);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<52, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.1899128287,0.3755596859,0.8673372165,0.2041261555,0.8732154084,0.389791896,0.4786963222,0.3341055888);
cl::sycl::float8 inputData_1(0.8497014754,0.866518316,0.6087325652,0.2472364401,0.8943614309,0.1820643516,0.5646795053,0.2251224481);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<53, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.8181402513,0.8565427132,0.7435122384,0.3527131349,0.294270952,0.7038867306,0.3328476153,0.4358283023,0.1370045415,0.2057870483,0.1164396965,0.1623368961,0.1585689195,0.4361853617,0.5406217421,0.6927030559);
cl::sycl::float16 inputData_1(0.2138267791,0.4377509969,0.6095728299,0.1676445559,0.4558489241,0.3954048314,0.8591455432,0.1462856911,0.4269009769,0.4337803838,0.6825444037,0.3565368023,0.2631922208,0.3346493241,0.476710034,0.8602146637);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<54, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7372136182,0.3215761966);
cl::sycl::double2 inputData_1(0.5465452707,0.6505602429);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<55, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.7365257245,0.4569315072,0.4190215241);
cl::sycl::double3 inputData_1(0.7141125943,0.4453731965,0.2983661351);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<56, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4627576252,0.849683717,0.2140539906,0.4699482836);
cl::sycl::double4 inputData_1(0.6098428195,0.4866303906,0.2629119235,0.1014745285);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<57, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6591933694,0.5949884144,0.1062213195,0.3388480968,0.7149074076,0.6031363028,0.5361664928,0.2249768878);
cl::sycl::double8 inputData_1(0.6650352344,0.4771479374,0.642542997,0.7080718694,0.2858901772,0.7095960105,0.3240707078,0.8872121097);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<58, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1966652886,0.806974415,0.1324377,0.3052606547,0.520881527,0.5652929468,0.416987988,0.1816253826,0.3020864687,0.3267172031,0.7041782836,0.8270194602,0.5763279324,0.1283607726,0.7337891773,0.3444831463);
cl::sycl::double16 inputData_1(0.3719123251,0.5241483501,0.2992376381,0.8359824703,0.2308438067,0.4318643204,0.3317535596,0.5158672818,0.5591854425,0.6017117513,0.5251006431,0.4286436019,0.6076752099,0.4227303013,0.7228402072,0.7305419402);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<59, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3338033345,0.3974434588);
cl::sycl::half2 inputData_1(0.6030487248,0.2256559737);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<60, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6576255448,0.4051422024,0.5728499798);
cl::sycl::half3 inputData_1(0.2116264794,0.6346067089,0.3832462885);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<61, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.478132461,0.4320859207,0.4813721984,0.6557565063);
cl::sycl::half4 inputData_1(0.3545921415,0.6216435847,0.148177686,0.340148122);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<62, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.6961677521,0.1419247024,0.5969137562,0.1204374394,0.4772230946,0.810836035,0.1080880752,0.5214624165);
cl::sycl::half8 inputData_1(0.1531654637,0.7936878209,0.6490372178,0.6935630853,0.635206064,0.105138763,0.1329422898,0.5967014432);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<63, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8997481005,0.7985177913,0.6597486454,0.6816799635,0.2813496181,0.7012911473,0.3303392839,0.1843682136,0.4687159164,0.364156618,0.2346043189,0.437367914,0.8177607816,0.4482162186,0.4578335162,0.667062206);
cl::sycl::half16 inputData_1(0.5193294961,0.2033784283,0.828313918,0.4552994689,0.7314701914,0.411100104,0.7454768151,0.4116291328,0.2761276173,0.2569557335,0.8520277155,0.5692242069,0.139834612,0.4106780769,0.2872234084,0.1677256517);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<64, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2494046948,0.145592384);
float inputData_1(0.6104589026);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<65, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2386990919,0.588623901,0.5900053983);
float inputData_1(0.6639389686);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<66, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5096949205,0.3275391923,0.8019659631,0.3824568654);
float inputData_1(0.46663546);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<67, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6055035454,0.5128994385,0.8651746789,0.863774142,0.8438078805,0.8472610797,0.5647681085,0.492161651);
float inputData_1(0.6632934539);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<68, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.2723356744,0.3126976314,0.1350458029,0.230286034,0.10309964,0.6237020612,0.2123255912,0.7293434765,0.6444031967,0.8765406347,0.4172115896,0.8371135308,0.4629633379,0.3716029919,0.1818710959,0.8062657481);
float inputData_1(0.7358321269);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<69, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.3583431812,0.4645955079);
double inputData_1(0.3601147727);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<70, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1230632932,0.1354820203,0.3949633007);
double inputData_1(0.2676730625);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<71, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5196116826,0.2502280285,0.2612972692,0.6381343051);
double inputData_1(0.6884821254);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<72, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.3497856767,0.7879955195,0.3037113397,0.375152301,0.6699843123,0.1356023211,0.8473467681,0.1578701854);
double inputData_1(0.4687448472);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<73, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6796838608,0.137974828,0.7472021485,0.8831146746,0.4684093382,0.194498909,0.1651815965,0.1789843489,0.7123530993,0.4312102788,0.8353873266,0.4525118209,0.1617146481,0.44154847,0.7038623147,0.7634707415);
double inputData_1(0.1314813492);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<74, float>(
[=](){
float inputData_0(0.244311513);
float inputData_1(0.4920107616);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<75, double>(
[=](){
double inputData_0(0.2024683824);
double inputData_1(0.7968741136);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<76, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8475687108);
cl::sycl::half inputData_1(0.3556775987);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<77, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.447874946,0.5456432515);
cl::sycl::float2 inputData_1(0.3284046329,0.532860558);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<78, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.2609480364,0.337313001,0.4534269066);
cl::sycl::float3 inputData_1(0.5837359218,0.5289320209,0.3087903814);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<79, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.2854303003,0.1949841894,0.7267949087,0.1791206132);
cl::sycl::float4 inputData_1(0.6863080049,0.2990189565,0.3276455872,0.6888667464);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<80, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.6276966334,0.6935372444,0.512226447,0.7872766557,0.1974351131,0.6161575691,0.19459545,0.6898266945);
cl::sycl::float8 inputData_1(0.3871237292,0.6399056835,0.6627871308,0.6284867661,0.2772463843,0.7654399091,0.2921088699,0.5145226378);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<81, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.6397166033,0.2868825398,0.6028093784,0.3294648384,0.2371059009,0.7477990628,0.5424982161,0.3623077653,0.5683447578,0.1202291179,0.2038582854,0.4164646814,0.8806052636,0.5083796143,0.1611649641,0.7120324922);
cl::sycl::float16 inputData_1(0.7251550967,0.7198417395,0.5555984304,0.6565589903,0.270766349,0.6860484727,0.7529391899,0.7079732322,0.3827699221,0.5728224405,0.603191486,0.8206478829,0.1864111162,0.7671470167,0.5211484468,0.3868912964);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<82, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.4644823212,0.1101083991);
cl::sycl::double2 inputData_1(0.2760588739,0.6222107361);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<83, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.6286794238,0.4957591522,0.8626607045);
cl::sycl::double3 inputData_1(0.4847320708,0.3511549276,0.7782246714);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<84, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.3073266395,0.5834447944,0.6627350819,0.757357039);
cl::sycl::double4 inputData_1(0.7282950001,0.4072738644,0.1473442448,0.1306302924);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<85, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6811683103,0.8693531051,0.3745322994,0.4529560785,0.6806384126,0.6262649967,0.3080852708,0.6372678766);
cl::sycl::double8 inputData_1(0.3439219357,0.3850863252,0.5316106442,0.6858510591,0.2209729693,0.1175897687,0.6022639636,0.1196517422);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<86, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.1359705926,0.2806204614,0.6231014986,0.1532360781,0.1499246141,0.8776745955,0.438122315,0.8139431472,0.2732194272,0.4481705436,0.3864281077,0.2415484288,0.3630505486,0.889436655,0.6978472078,0.4061346233);
cl::sycl::double16 inputData_1(0.4274275475,0.3109927209,0.5250693429,0.6885095297,0.6493172926,0.4701198683,0.1335512374,0.8372062452,0.4271470425,0.4122390936,0.1024880916,0.2105817713,0.795082734,0.5111476769,0.6859478754,0.2185343091);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<87, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3640408053,0.7721092452);
cl::sycl::half2 inputData_1(0.7565268169,0.2974354145);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<88, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.1175802467,0.7451735788,0.235075204);
cl::sycl::half3 inputData_1(0.7301451137,0.6469273839,0.2346518082);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<89, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1627909149,0.8421195439,0.5783027178,0.5964081384);
cl::sycl::half4 inputData_1(0.4660094423,0.2200567819,0.5815759304,0.301978304);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<90, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7447157248,0.6861751638,0.121813748,0.8459384077,0.1290528387,0.1716954551,0.3341876487,0.2206472484);
cl::sycl::half8 inputData_1(0.2889160663,0.3846475909,0.6883997724,0.4237690886,0.3158718038,0.4938505229,0.4140745998,0.348611358);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<91, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8204333263,0.5403587608,0.8818620088,0.7183299275,0.5563994381,0.3099572714,0.649474925,0.4647341752,0.677110172,0.4230230471,0.4968040291,0.116547014,0.6919668019,0.1274188355,0.6445803087,0.5656029564);
cl::sycl::half16 inputData_1(0.7207340892,0.3318220794,0.6488886521,0.2656783805,0.5234176011,0.3722243034,0.8827636411,0.8774932459,0.2671757884,0.5528305887,0.3635541487,0.8748305496,0.8396207585,0.5689166824,0.6760675641,0.6450598054);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<92, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.382684506,0.8330892555);
float inputData_1(0.8195628295);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<93, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3645267716,0.6979159285,0.1072737013);
float inputData_1(0.7530872884);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<94, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5518954763,0.8618453702,0.3905544596,0.6005704599);
float inputData_1(0.3584019452);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<95, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7262283051,0.5805623974,0.8899768184,0.1008102345,0.2126069937,0.1348811057,0.2006782791,0.8435082377);
float inputData_1(0.8588866396);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<96, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4843300278,0.8573515157,0.7547100882,0.7228941873,0.6978255606,0.2501236681,0.5391018089,0.4391033845,0.8598304381,0.2390668305,0.2358870748,0.6270894029,0.2259214317,0.1880429384,0.5031385579,0.737328777);
float inputData_1(0.5840365373);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<97, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7038031783,0.3126068253);
double inputData_1(0.3279702664);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<98, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.4429629134,0.8926783074,0.6743346052);
double inputData_1(0.8570031658);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<99, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5302963646,0.5436478813,0.8920722684,0.2519906205);
double inputData_1(0.7260723497);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<100, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.7332110592,0.7757933021,0.7000421674,0.2242664146,0.6289021057,0.8389625477,0.550628122,0.3887532083);
double inputData_1(0.8596161189);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<101, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5492789204,0.4293091156,0.5913067984,0.7433000133,0.2826416725,0.1125536344,0.5232759177,0.8530859361,0.6442063701,0.6047264001,0.602252118,0.4975917698,0.6847354158,0.299355552,0.8134034112,0.3195781242);
double inputData_1(0.8559560106);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<102, float>(
[=](){
float inputData_0(0.841197368);
float inputData_1(0.1623396192);
float inputData_2(0.458543761);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<103, double>(
[=](){
double inputData_0(0.6952290279);
double inputData_1(0.4597232572);
double inputData_2(0.5071192199);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<104, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.7454591501);
cl::sycl::half inputData_1(0.6639937288);
cl::sycl::half inputData_2(0.8664033782);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<105, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.2315887954,0.8388474289);
cl::sycl::float2 inputData_1(0.842389002,0.6077991509);
cl::sycl::float2 inputData_2(0.8523126618,0.3021484699);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<106, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8054298267,0.7187834344,0.5877511998);
cl::sycl::float3 inputData_1(0.1725033957,0.124107483,0.1087755964);
cl::sycl::float3 inputData_2(0.300446446,0.709881928,0.4093000259);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<107, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.7203573801,0.6005139927,0.4114095193,0.804117303);
cl::sycl::float4 inputData_1(0.1307337915,0.4722503922,0.7638818715,0.2014507864);
cl::sycl::float4 inputData_2(0.6683900492,0.3624926735,0.1194410229,0.4789799447);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<108, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.5173541912,0.1332690005,0.5527354828,0.3779470704,0.1035945658,0.2526187045,0.1886485378,0.5324975638);
cl::sycl::float8 inputData_1(0.1344961304,0.8425060561,0.7760495779,0.856238115,0.3518408262,0.8242139108,0.887449938,0.7117851474);
cl::sycl::float8 inputData_2(0.3200660909,0.6367114433,0.576530523,0.4233626417,0.3448782832,0.1478785525,0.2003059798,0.2071649248);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<109, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4847142917,0.6135147078,0.711254762,0.137371008,0.7590078981,0.1347769786,0.543957464,0.6953182798,0.6049770974,0.8597429405,0.3757586825,0.5687066842,0.1662392502,0.5478372703,0.7506390409,0.2612836111);
cl::sycl::float16 inputData_1(0.3087716003,0.6603245122,0.3031055735,0.3073963792,0.8484122304,0.8988344247,0.2241587446,0.8201299098,0.5421811888,0.1308809139,0.5684021722,0.6132397205,0.127036559,0.7061535377,0.7542401132,0.1573145937);
cl::sycl::float16 inputData_2(0.6187199521,0.4652379847,0.2909770299,0.4669363053,0.2275117678,0.3669327254,0.6241658398,0.4811884449,0.5447360757,0.534754235,0.7564753921,0.3747062385,0.7503696726,0.1639896697,0.4421864367,0.3818560932);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<110, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.461264511,0.7668078564);
cl::sycl::double2 inputData_1(0.5099195204,0.889797317);
cl::sycl::double2 inputData_2(0.7891685762,0.1950773963);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<111, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.3535132284,0.1181804012,0.6870027371);
cl::sycl::double3 inputData_1(0.1153606435,0.8087508119,0.2546742919);
cl::sycl::double3 inputData_2(0.4310690322,0.1496314448,0.3490039098);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<112, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.4116119916,0.1417847788,0.7140405225,0.6690797756);
cl::sycl::double4 inputData_1(0.3863068993,0.7681540425,0.1619374419,0.1432051208);
cl::sycl::double4 inputData_2(0.3839842355,0.8214730657,0.7051741615,0.6378541428);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<113, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.5501885882,0.743012431,0.4297813546,0.1245508638,0.7419234292,0.2523947474,0.4101270799,0.3860874778);
cl::sycl::double8 inputData_1(0.1986925007,0.3806274952,0.2416695023,0.5928110641,0.6227474862,0.1109172423,0.4651806819,0.5432421252);
cl::sycl::double8 inputData_2(0.7973303955,0.4968250482,0.1643603495,0.1413791033,0.7896872664,0.7325835275,0.7867583449,0.3097941341);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<114, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.6183978886,0.1765744378,0.7612584894,0.3668903491,0.8641177899,0.4771060524,0.1264541576,0.8272447021,0.6004257191,0.3296650288,0.129443115,0.4013491174,0.2254884478,0.5386242491,0.2175069401,0.2396914231);
cl::sycl::double16 inputData_1(0.8366955886,0.6120960276,0.2940651294,0.8031170245,0.5997726639,0.8564794721,0.4863334355,0.8103206672,0.6427550459,0.1353348447,0.292232392,0.3252611239,0.2360133467,0.2905495605,0.2808321188,0.8026749994);
cl::sycl::double16 inputData_2(0.4703190348,0.8012094651,0.2103983068,0.5519347892,0.1107741425,0.8442411279,0.1045096915,0.4119261188,0.741268767,0.8999052474,0.1156077923,0.7592683871,0.5080703676,0.1305456164,0.7216954167,0.1895219301);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<115, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.5891793498,0.7226601729);
cl::sycl::half2 inputData_1(0.6388727414,0.4038994598);
cl::sycl::half2 inputData_2(0.1211533095,0.4490111742);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<116, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.8309555869,0.3663386924,0.2983669753);
cl::sycl::half3 inputData_1(0.2102646671,0.508201964,0.5266786182);
cl::sycl::half3 inputData_2(0.1584385939,0.4262067888,0.6269451639);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<117, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.8728405481,0.4452328974,0.4488282695,0.4769071781);
cl::sycl::half4 inputData_1(0.280026793,0.4158701136,0.6162117808,0.4176473672);
cl::sycl::half4 inputData_2(0.5651005987,0.7684658304,0.8983740585,0.8080317469);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<118, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.3974373015,0.1173817075,0.5892836796,0.4796405666,0.2896136896,0.1322432793,0.3572561912,0.7384570503);
cl::sycl::half8 inputData_1(0.8712952055,0.1853281113,0.8021115294,0.138974137,0.6707806551,0.1214365707,0.4368397458,0.7961846708);
cl::sycl::half8 inputData_2(0.4144865181,0.8396514541,0.6705561129,0.5833474246,0.2291032384,0.3723966269,0.4288769314,0.5721638913);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<119, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.8968305282,0.3269677982,0.5028503127,0.8467583261,0.376336635,0.6028838298,0.712905231,0.60421578,0.7027445439,0.2565544002,0.8658701495,0.2415182455,0.5669449408,0.3368340873,0.6075384202,0.3328883323);
cl::sycl::half16 inputData_1(0.4449706855,0.6457780386,0.3152550004,0.682300706,0.3775021382,0.2057248778,0.5905029735,0.2326064231,0.4444619571,0.4187179295,0.1609350779,0.6686158699,0.6446588521,0.722236004,0.5359305127,0.5431334206);
cl::sycl::half16 inputData_2(0.2353864023,0.2659711192,0.2825995924,0.520242823,0.755186066,0.3855792934,0.8054975904,0.6887026148,0.6731577146,0.3681377034,0.1947819936,0.8702323849,0.7836885085,0.4270943926,0.7905745522,0.819373692);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<120, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3739788987,0.501249194);
cl::sycl::float2 inputData_1(0.3654318722,0.6561260113);
float inputData_2(0.8297338508);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<121, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.8876352831,0.6950232599,0.3441938831);
cl::sycl::float3 inputData_1(0.8043946321,0.8940957032,0.377220931);
float inputData_2(0.858969882);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<122, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5092371244,0.8717083538,0.8966847921,0.7503536767);
cl::sycl::float4 inputData_1(0.6467496394,0.2232115754,0.1039338266,0.5763766803);
float inputData_2(0.6635679244);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<123, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.8484304361,0.5136959202,0.6574772822,0.6178847772,0.2639361,0.6154400742,0.8853769691,0.1889479653);
cl::sycl::float8 inputData_1(0.6508345946,0.591444094,0.400683779,0.7346782031,0.1083886869,0.8139292977,0.7538911624,0.4845638652);
float inputData_2(0.1865113239);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<124, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4621028445,0.5674023193,0.3031067828,0.4892251719,0.7205830111,0.8381854365,0.5493160221,0.761793428,0.1623465704,0.7850944371,0.8366516524,0.234401101,0.7619898894,0.7796529363,0.8029270947,0.5137116159);
cl::sycl::float16 inputData_1(0.5866033951,0.2664665964,0.6665052395,0.4240138427,0.1169352686,0.2074136908,0.4105744253,0.8081438448,0.5519538346,0.8330056272,0.8435870755,0.1694359937,0.570572331,0.3676224305,0.5054360978,0.464419842);
float inputData_2(0.4839546181);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<125, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.1814446465,0.766528067);
cl::sycl::double2 inputData_1(0.4922239694,0.615990045);
double inputData_2(0.4781430022);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<126, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.2448147013,0.532800468,0.2276317912);
cl::sycl::double3 inputData_1(0.7817434049,0.7652832205,0.2149110225);
double inputData_2(0.1550751624);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<127, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1547935336,0.4145952199,0.8624331499,0.5449123328);
cl::sycl::double4 inputData_1(0.3124212592,0.2837190617,0.1886985558,0.2128569684);
double inputData_2(0.7494906133);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<128, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.2109067718,0.7912492457,0.7583984593,0.2094470417,0.546979816,0.1056442144,0.7896289075,0.5466216964);
cl::sycl::double8 inputData_1(0.7042723155,0.4922762615,0.6523375994,0.8449912993,0.5476366481,0.7997643815,0.3744363539,0.1780260365);
double inputData_2(0.1041157141);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<129, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.2813202232,0.7708694695,0.3491963701,0.2796929181,0.4965043387,0.857523298,0.5071827586,0.3726973301,0.1620014328,0.5589335467,0.2810055836,0.3939993022,0.4049298934,0.7065474698,0.2853030758,0.8487137806);
cl::sycl::double16 inputData_1(0.6939104544,0.4848956326,0.804379593,0.3873343842,0.4074718989,0.2034953048,0.7228448756,0.4209541223,0.5002024226,0.4767749323,0.6249454541,0.3991507449,0.8326890609,0.4455378072,0.3873711823,0.4207024441);
double inputData_2(0.7130365772);
return cl::sycl::mix(inputData_0,inputData_1,inputData_2);

});

test_function<130, float>(
[=](){
float inputData_0(0.894445272);
return cl::sycl::radians(inputData_0);

});

test_function<131, double>(
[=](){
double inputData_0(0.793211717);
return cl::sycl::radians(inputData_0);

});

test_function<132, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.4837819988);
return cl::sycl::radians(inputData_0);

});

test_function<133, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3330874748,0.4567896435);
return cl::sycl::radians(inputData_0);

});

test_function<134, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.3752124425,0.2948256418,0.2495527323);
return cl::sycl::radians(inputData_0);

});

test_function<135, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.8647005876,0.4994441523,0.1879798989,0.4071252881);
return cl::sycl::radians(inputData_0);

});

test_function<136, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4109735338,0.5108276216,0.8840330597,0.8813067973,0.5527152886,0.5944732202,0.6405032599,0.5017777461);
return cl::sycl::radians(inputData_0);

});

test_function<137, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.4893424465,0.3516191513,0.6471373916,0.1735162227,0.353716197,0.8127828476,0.2819025208,0.8740659024,0.8873357776,0.5603061305,0.1323487842,0.1747825579,0.2602413102,0.3614492546,0.1904865693,0.7377686185);
return cl::sycl::radians(inputData_0);

});

test_function<138, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.391323656,0.2869869587);
return cl::sycl::radians(inputData_0);

});

test_function<139, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1349550963,0.4061374875,0.1036053844);
return cl::sycl::radians(inputData_0);

});

test_function<140, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.1931931604,0.583716408,0.8479563291,0.2594927375);
return cl::sycl::radians(inputData_0);

});

test_function<141, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.6928489653,0.2581644168,0.1011961551,0.8172304369,0.7768869902,0.1534229728,0.2417082305,0.2874407424);
return cl::sycl::radians(inputData_0);

});

test_function<142, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.8426570917,0.4055432765,0.7459054053,0.4486508263,0.4049957334,0.7122784438,0.5926087973,0.3154541554,0.5662484786,0.66308228,0.7616624733,0.6417432633,0.6125976571,0.576721874,0.1736407593,0.8561512476);
return cl::sycl::radians(inputData_0);

});

test_function<143, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.6718735284,0.3182969035);
return cl::sycl::radians(inputData_0);

});

test_function<144, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.6538805553,0.5966539489,0.6270811566);
return cl::sycl::radians(inputData_0);

});

test_function<145, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.4031271768,0.5585406838,0.6280217845,0.2613248552);
return cl::sycl::radians(inputData_0);

});

test_function<146, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.5064097315,0.1962733242,0.1844243985,0.8288484602,0.1996377796,0.8146135774,0.4758393596,0.4639220926);
return cl::sycl::radians(inputData_0);

});

test_function<147, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.3718522556,0.4329741732,0.4017859046,0.5519863576,0.3684746553,0.7575806908,0.2868494001,0.2987760982,0.4844412373,0.8480650271,0.1191325393,0.6787308925,0.1048052702,0.4238881705,0.7113657998,0.4568632974);
return cl::sycl::radians(inputData_0);

});

test_function<148, float>(
[=](){
float inputData_0(0.4435911431);
float inputData_1(0.3025734632);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<149, double>(
[=](){
double inputData_0(0.4800765106);
double inputData_1(0.2826075997);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<150, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.3268170319);
cl::sycl::half inputData_1(0.6226348489);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<151, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.5795576449,0.8436364123);
cl::sycl::float2 inputData_1(0.8750952651,0.5179041546);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<152, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.17004521,0.3399224754,0.5142439165);
cl::sycl::float3 inputData_1(0.6385303147,0.8569577995,0.2240859469);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<153, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.1293477615,0.7960285462,0.7441314945,0.7125986212);
cl::sycl::float4 inputData_1(0.4748806142,0.6422245633,0.4291753798,0.2536413262);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<154, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.4127150121,0.7296372768,0.7414844976,0.8689075729,0.8101337001,0.6456676294,0.5167296182,0.6791416188);
cl::sycl::float8 inputData_1(0.2465628715,0.838467625,0.6700612596,0.5755884444,0.4472333738,0.6068332713,0.5941429823,0.8190833012);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<155, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.5565890487,0.2707017219,0.4531034889,0.2943748086,0.823960135,0.7748206515,0.5446552916,0.2571132541,0.1348336104,0.207335563,0.4545754246,0.6393632627,0.2791984836,0.6476162798,0.7895595072,0.7057928653);
cl::sycl::float16 inputData_1(0.4404219802,0.6165823259,0.8906939464,0.8083292823,0.3705196052,0.6483576428,0.2305690251,0.5458947768,0.3852271719,0.4505170547,0.4511190831,0.6305863519,0.7767968058,0.4748574738,0.2172678305,0.7033234186);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<156, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.7013144085,0.8630761918);
cl::sycl::double2 inputData_1(0.4152448055,0.4711032075);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<157, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.5324765933,0.8136987172,0.6633732143);
cl::sycl::double3 inputData_1(0.1170224989,0.2658581501,0.7831158782);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<158, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5683790539,0.7991267728,0.42911931,0.2683743097);
cl::sycl::double4 inputData_1(0.1033125815,0.8968407671,0.2091052266,0.6143750126);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<159, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.4917671282,0.4041195127,0.5297617201,0.1626268482,0.8760273619,0.494189553,0.1122316134,0.4354747451);
cl::sycl::double8 inputData_1(0.7057615242,0.3496669357,0.6960179271,0.7138901759,0.2912965752,0.874377989,0.1223109996,0.7908843872);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<160, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.5101192973,0.2227035908,0.3067143566,0.5748138335,0.3227657311,0.7707368611,0.2756228112,0.4072490379,0.5054505344,0.371818371,0.7593142425,0.3111057635,0.1711817386,0.2238281507,0.6015563642,0.5508501201);
cl::sycl::double16 inputData_1(0.1506386575,0.8944393309,0.4835525062,0.355549761,0.6833299212,0.1194334872,0.4473993159,0.6315310713,0.8697089799,0.7093102225,0.8081273677,0.1951247257,0.4438164845,0.1254323401,0.3175953583,0.4074374921);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<161, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.3750568646,0.3989926375);
cl::sycl::half2 inputData_1(0.7424640038,0.2516349062);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<162, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.7595964881,0.5335368696,0.3709961028);
cl::sycl::half3 inputData_1(0.5417886139,0.2291386658,0.4963637997);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<163, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.1175626376,0.7903800622,0.3652648278,0.3752343595);
cl::sycl::half4 inputData_1(0.8961215979,0.5907645855,0.4341229555,0.732525377);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<164, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.1541317657,0.5564033634,0.5165607695,0.7889825344,0.568960274,0.4882179322,0.5161806872,0.7255178465);
cl::sycl::half8 inputData_1(0.3778566366,0.5462315311,0.6659122182,0.8964443635,0.6549473564,0.869496937,0.4192261291,0.5870247942);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<165, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.6962358859,0.3787327597,0.315339951,0.8782664889,0.3788271783,0.8999221417,0.7818167877,0.2728544919,0.7625753779,0.8869017013,0.3214561794,0.631563531,0.7156713784,0.166625599,0.7554654439,0.3466885857);
cl::sycl::half16 inputData_1(0.6651054369,0.8601105769,0.1280872171,0.5893703044,0.3339237023,0.1917270313,0.6694838421,0.8832372499,0.5101684007,0.3770753674,0.4592716759,0.4316943079,0.5255215277,0.4273406692,0.1642979717,0.8835421888);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<166, cl::sycl::float2>(
[=](){
float inputData_0(0.8973658223);
cl::sycl::float2 inputData_1(0.2393076438,0.292831973);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<167, cl::sycl::float3>(
[=](){
float inputData_0(0.4495650332);
cl::sycl::float3 inputData_1(0.6589863496,0.1250759484,0.7683980406);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<168, cl::sycl::float4>(
[=](){
float inputData_0(0.6107466987);
cl::sycl::float4 inputData_1(0.3154348497,0.7966937719,0.6289673705,0.353539425);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<169, cl::sycl::float8>(
[=](){
float inputData_0(0.5382767955);
cl::sycl::float8 inputData_1(0.8833900469,0.1387463736,0.6667696133,0.7795308881,0.6538534791,0.2120147303,0.5777196828,0.7287641966);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<170, cl::sycl::float16>(
[=](){
float inputData_0(0.4348767297);
cl::sycl::float16 inputData_1(0.5659426226,0.3027743635,0.3501988406,0.7468561145,0.4915985101,0.45904939,0.1983070697,0.3995767186,0.5165768448,0.2848098678,0.7463484495,0.406960565,0.2907918339,0.3466379946,0.7595708262,0.8233147808);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<171, cl::sycl::double2>(
[=](){
double inputData_0(0.8682382789);
cl::sycl::double2 inputData_1(0.112155264,0.7031153045);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<172, cl::sycl::double3>(
[=](){
double inputData_0(0.5203872288);
cl::sycl::double3 inputData_1(0.1996480616,0.2972268611,0.3253526644);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<173, cl::sycl::double4>(
[=](){
double inputData_0(0.4233733501);
cl::sycl::double4 inputData_1(0.4765770179,0.8494310852,0.1466840456,0.667335444);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<174, cl::sycl::double8>(
[=](){
double inputData_0(0.7832848759);
cl::sycl::double8 inputData_1(0.3858399344,0.2993737587,0.2770467925,0.3406713414,0.2162383925,0.5413423895,0.3003196033,0.1218012284);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<175, cl::sycl::double16>(
[=](){
double inputData_0(0.2861067466);
cl::sycl::double16 inputData_1(0.7565056844,0.4338961776,0.8068290037,0.8548925052,0.2946786628,0.5479779609,0.8048535277,0.5651362698,0.2344002291,0.298362598,0.8900998639,0.3395094919,0.7941623858,0.7360098702,0.6935877622,0.6775538055);
return cl::sycl::step(inputData_0,inputData_1);

});

test_function<176, float>(
[=](){
float inputData_0(0.7319854909);
float inputData_1(0.7779261482);
float inputData_2(0.1498931492);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<177, cl::sycl::float2>(
[=](){
float inputData_0(0.2342478596);
float inputData_1(0.5044234597);
cl::sycl::float2 inputData_2(0.2699916207,0.5265744181);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<178, cl::sycl::float3>(
[=](){
float inputData_0(0.4945458716);
float inputData_1(0.2014171592);
cl::sycl::float3 inputData_2(0.1687689217,0.1093222374,0.760028904);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<179, cl::sycl::float4>(
[=](){
float inputData_0(0.1653933347);
float inputData_1(0.869252263);
cl::sycl::float4 inputData_2(0.8870654836,0.6965571546,0.4603066246,0.3206307708);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<180, cl::sycl::float8>(
[=](){
float inputData_0(0.4299619375);
float inputData_1(0.3762346128);
cl::sycl::float8 inputData_2(0.4170361117,0.6809566264,0.8140209661,0.2261720215,0.2941364631,0.2679175249,0.1362767921,0.7833604637);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<181, cl::sycl::float16>(
[=](){
float inputData_0(0.5090204347);
float inputData_1(0.153627823);
cl::sycl::float16 inputData_2(0.4570042089,0.4604886336,0.7223647648,0.7091179589,0.2075911915,0.6015004926,0.5077490347,0.1107934213,0.2181886634,0.633478739,0.393620645,0.8709483164,0.5014002898,0.6506265357,0.2068941118,0.4835587669);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<182, double>(
[=](){
double inputData_0(0.6872971769);
double inputData_1(0.7667853406);
double inputData_2(0.2596859432);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<183, cl::sycl::double2>(
[=](){
double inputData_0(0.4175253769);
double inputData_1(0.4788216072);
cl::sycl::double2 inputData_2(0.4522979149,0.4803533364);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<184, cl::sycl::double3>(
[=](){
double inputData_0(0.3367201801);
double inputData_1(0.7469765943);
cl::sycl::double3 inputData_2(0.8304623564,0.3792005828,0.6102880573);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<185, cl::sycl::double4>(
[=](){
double inputData_0(0.4045647347);
double inputData_1(0.5630015934);
cl::sycl::double4 inputData_2(0.6564311558,0.5012130219,0.6396656562,0.7057165954);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<186, cl::sycl::double8>(
[=](){
double inputData_0(0.7746365084);
double inputData_1(0.2510479118);
cl::sycl::double8 inputData_2(0.2731079177,0.5114970813,0.5077256151,0.7461803949,0.513906698,0.8200419756,0.7220822687,0.505052649);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<187, cl::sycl::double16>(
[=](){
double inputData_0(0.7610608974);
double inputData_1(0.4806847687);
cl::sycl::double16 inputData_2(0.3733721272,0.4467387515,0.4649664098,0.6204255612,0.141725676,0.6836069417,0.8745862201,0.4670540553,0.155017835,0.2610051071,0.1825712315,0.305082929,0.7351256844,0.1008395483,0.7988634659,0.8516377204);
return cl::sycl::smoothstep(inputData_0,inputData_1,inputData_2);

});

test_function<188, float>(
[=](){
float inputData_0(0.2480024673);
return cl::sycl::sign(inputData_0);

});

test_function<189, double>(
[=](){
double inputData_0(0.2388691479);
return cl::sycl::sign(inputData_0);

});

test_function<190, cl::sycl::half>(
[=](){
cl::sycl::half inputData_0(0.8726103703);
return cl::sycl::sign(inputData_0);

});

test_function<191, cl::sycl::float2>(
[=](){
cl::sycl::float2 inputData_0(0.3883041239,0.7494210438);
return cl::sycl::sign(inputData_0);

});

test_function<192, cl::sycl::float3>(
[=](){
cl::sycl::float3 inputData_0(0.1072085421,0.8926332632,0.113192165);
return cl::sycl::sign(inputData_0);

});

test_function<193, cl::sycl::float4>(
[=](){
cl::sycl::float4 inputData_0(0.5860564763,0.8427602838,0.7650086872,0.3483220456);
return cl::sycl::sign(inputData_0);

});

test_function<194, cl::sycl::float8>(
[=](){
cl::sycl::float8 inputData_0(0.7576643302,0.4144381095,0.4998451337,0.3906224004,0.3837744539,0.5656601812,0.7256543882,0.6595929028);
return cl::sycl::sign(inputData_0);

});

test_function<195, cl::sycl::float16>(
[=](){
cl::sycl::float16 inputData_0(0.7144623647,0.1114186144,0.5253547148,0.3822308823,0.2668918662,0.8366814577,0.25744644,0.2475799727,0.2430512756,0.6264759252,0.5893919444,0.5045067752,0.5694414057,0.8524610514,0.7889680697,0.8247476161);
return cl::sycl::sign(inputData_0);

});

test_function<196, cl::sycl::double2>(
[=](){
cl::sycl::double2 inputData_0(0.1433262672,0.8179302192);
return cl::sycl::sign(inputData_0);

});

test_function<197, cl::sycl::double3>(
[=](){
cl::sycl::double3 inputData_0(0.1250805178,0.6181143561,0.8446657427);
return cl::sycl::sign(inputData_0);

});

test_function<198, cl::sycl::double4>(
[=](){
cl::sycl::double4 inputData_0(0.5019707387,0.4354565341,0.3653943636,0.8328988947);
return cl::sycl::sign(inputData_0);

});

test_function<199, cl::sycl::double8>(
[=](){
cl::sycl::double8 inputData_0(0.8407754113,0.5953004073,0.6715431873,0.3713016329,0.2105398137,0.8832007824,0.6256176933,0.3195100396);
return cl::sycl::sign(inputData_0);

});

test_function<200, cl::sycl::double16>(
[=](){
cl::sycl::double16 inputData_0(0.881666842,0.5871759469,0.3644680695,0.8166504004,0.1623203424,0.7433229313,0.2276632124,0.1861395568,0.3071552548,0.671860435,0.5864066389,0.4370227019,0.227236954,0.8390014912,0.7130281396,0.6490133027);
return cl::sycl::sign(inputData_0);

});

test_function<201, cl::sycl::half2>(
[=](){
cl::sycl::half2 inputData_0(0.750327253,0.7193962085);
return cl::sycl::sign(inputData_0);

});

test_function<202, cl::sycl::half3>(
[=](){
cl::sycl::half3 inputData_0(0.189940009,0.7186833746,0.7709803473);
return cl::sycl::sign(inputData_0);

});

test_function<203, cl::sycl::half4>(
[=](){
cl::sycl::half4 inputData_0(0.6974046884,0.4858218211,0.6491565152,0.1800060909);
return cl::sycl::sign(inputData_0);

});

test_function<204, cl::sycl::half8>(
[=](){
cl::sycl::half8 inputData_0(0.7114762868,0.3097822146,0.7281011171,0.608227267,0.5072627653,0.5288137245,0.1597846449,0.1327183283);
return cl::sycl::sign(inputData_0);

});

test_function<205, cl::sycl::half16>(
[=](){
cl::sycl::half16 inputData_0(0.1118598519,0.720433974,0.2107962542,0.1982932294,0.4080502318,0.8821623771,0.8087394358,0.3506312229,0.7558387586,0.1680622202,0.4136322716,0.5633646294,0.8889975323,0.1389655662,0.4299368138,0.835688352);
return cl::sycl::sign(inputData_0);

});

 }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}