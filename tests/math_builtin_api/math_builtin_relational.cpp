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

  void run(util::logger &log) override {
    test_function<0, int32_t>([=]() {
      float inputData_0(0.7755374812);
      float inputData_1(0.7063635224);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<1, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.4364572647, 0.3071334002);
      cl::sycl::float2 inputData_1(0.5090197771, 0.42394731);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<2, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.7270388712, 0.3426501809, 0.4812775633);
      cl::sycl::float3 inputData_1(0.5667056316, 0.8264903082, 0.5037494847);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<3, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.3254702755, 0.7046433633, 0.5946951973,
                                   0.3004050731);
      cl::sycl::float4 inputData_1(0.8277970048, 0.8862283808, 0.7481737888,
                                   0.8217327604);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<4, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.3481180555, 0.6838653986, 0.8190706304,
                                   0.6471871455, 0.4777141724, 0.1805609665,
                                   0.4473374684, 0.5887095788);
      cl::sycl::float8 inputData_1(0.8304088426, 0.8732850942, 0.4816078212,
                                   0.7922479422, 0.3083938483, 0.7440222616,
                                   0.5389594431, 0.1112333601);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<5, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.6757637491, 0.4190588338, 0.7598759817, 0.634522561, 0.1009142555,
          0.4948622932, 0.7940822204, 0.2951287015, 0.3601634902, 0.7963769857,
          0.2528536732, 0.5540085925, 0.2908927429, 0.8740322002, 0.7425435754,
          0.4583756571);
      cl::sycl::float16 inputData_1(
          0.1643566548, 0.3560436837, 0.506352514, 0.8462670594, 0.1872462767,
          0.5410137969, 0.6652491279, 0.5379527291, 0.7515734906, 0.5322268856,
          0.8710708368, 0.5825485024, 0.5700936513, 0.455991221, 0.5770294893,
          0.4079209168);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<6, int64_t>([=]() {
      double inputData_0(0.5605208113);
      double inputData_1(0.3322636019);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<7, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.2515130628, 0.2493836226);
      cl::sycl::double2 inputData_1(0.5902185439, 0.6253275112);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<8, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.4812247936, 0.171859489, 0.7060831376);
      cl::sycl::double3 inputData_1(0.8014162967, 0.8387048128, 0.7739681785);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<9, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.8185384971, 0.8384659519, 0.53247994,
                                    0.4130368402);
      cl::sycl::double4 inputData_1(0.6642267199, 0.320507297, 0.7493029668,
                                    0.7795887721);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<10, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.8160311739, 0.5718409468, 0.8598118986,
                                    0.5637560086, 0.4604504853, 0.6281963029,
                                    0.8970062715, 0.8335529744);
      cl::sycl::double8 inputData_1(0.7346600673, 0.1658983906, 0.590226484,
                                    0.4891553616, 0.6041178723, 0.7760620605,
                                    0.2944284976, 0.6851913766);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<11, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.1937074346, 0.2763684295, 0.7356663774, 0.3660289194, 0.7527304772,
          0.1804860162, 0.2170867911, 0.6581365122, 0.1361872543, 0.5590928294,
          0.8280128118, 0.5273583746, 0.644471306, 0.1213574357, 0.6079999279,
          0.5850707342);
      cl::sycl::double16 inputData_1(
          0.5607623584, 0.4129675275, 0.3961119523, 0.8844133205, 0.1291136301,
          0.1173092079, 0.8688250242, 0.2479775531, 0.1991161315, 0.2684612079,
          0.7405972723, 0.8495753269, 0.1182260605, 0.4404950656, 0.1812001755,
          0.3079359118);
      return cl::sycl::isequal(inputData_0, inputData_1);

    });

    test_function<12, int32_t>([=]() {
      float inputData_0(0.2766634171);
      float inputData_1(0.6175405759);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<13, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.3802351739, 0.2442543212);
      cl::sycl::float2 inputData_1(0.5029092042, 0.1315029657);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<14, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.180736993, 0.890588119, 0.2594846324);
      cl::sycl::float3 inputData_1(0.386844241, 0.685278645, 0.7706612522);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<15, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.8347856496, 0.2355396849, 0.6381124509,
                                   0.8732391224);
      cl::sycl::float4 inputData_1(0.1464407551, 0.6409614274, 0.776339675,
                                   0.3738500329);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<16, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.3005498714, 0.5774331148, 0.453851227,
                                   0.2398555876, 0.4773003321, 0.4279243165,
                                   0.5552901916, 0.5068801041);
      cl::sycl::float8 inputData_1(0.3491568008, 0.3857213461, 0.7701289395,
                                   0.3007461319, 0.5484801751, 0.1099490551,
                                   0.6932595019, 0.3687332436);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<17, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.1365571949, 0.3247065314, 0.2921043263, 0.8625034719, 0.3817804492,
          0.3303023319, 0.3873609578, 0.8575246685, 0.6069982818, 0.5968614765,
          0.6724954802, 0.4104137883, 0.4315343906, 0.6206662898, 0.1012193775,
          0.253847633);
      cl::sycl::float16 inputData_1(
          0.3675213525, 0.2915327681, 0.6099195209, 0.4029184563, 0.8003387134,
          0.5545211367, 0.4315251173, 0.4218136601, 0.6614636991, 0.4345812426,
          0.6297567112, 0.1374237488, 0.4562817518, 0.3073815388, 0.2261492577,
          0.5220585041);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<18, int64_t>([=]() {
      double inputData_0(0.4898124809);
      double inputData_1(0.5491239405);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<19, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.7043878138, 0.8071001234);
      cl::sycl::double2 inputData_1(0.4956661363, 0.3496465971);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<20, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.4735137883, 0.7472366859, 0.8000130652);
      cl::sycl::double3 inputData_1(0.7499319459, 0.2504010352, 0.8995362876);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<21, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.6064710079, 0.1667736401, 0.6804434844,
                                    0.8894571842);
      cl::sycl::double4 inputData_1(0.4214534578, 0.6428120042, 0.3529417098,
                                    0.2708197297);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<22, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.6738593146, 0.1018860518, 0.7581851284,
                                    0.5226767815, 0.1782274734, 0.1951231158,
                                    0.6194123399, 0.7989230591);
      cl::sycl::double8 inputData_1(0.3239861947, 0.8828121494, 0.1801445513,
                                    0.7831504877, 0.4173569419, 0.1650763334,
                                    0.3197710747, 0.4623825479);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<23, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.7338732249, 0.7890879229, 0.2067364434, 0.5166924227, 0.6206265905,
          0.3776424117, 0.7974910686, 0.3227278522, 0.114859462, 0.1325306189,
          0.6447974161, 0.5466845889, 0.8572020433, 0.8507510398, 0.8278809419,
          0.1336036256);
      cl::sycl::double16 inputData_1(
          0.6993078587, 0.6610598541, 0.6242894917, 0.669886122, 0.8221681205,
          0.6121129598, 0.3979594104, 0.530343027, 0.266275283, 0.5697004038,
          0.1071176656, 0.2208185391, 0.3667267104, 0.7316985271, 0.6747995382,
          0.370604776);
      return cl::sycl::isnotequal(inputData_0, inputData_1);

    });

    test_function<24, int32_t>([=]() {
      float inputData_0(0.5964304867);
      float inputData_1(0.1329623596);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<25, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.2310884365, 0.8855312561);
      cl::sycl::float2 inputData_1(0.3316246829, 0.4158335864);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<26, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.5387874373, 0.3347256012, 0.4824517353);
      cl::sycl::float3 inputData_1(0.2917648669, 0.1386050898, 0.2436694792);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<27, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.5184401854, 0.1566903073, 0.4225353172,
                                   0.362816568);
      cl::sycl::float4 inputData_1(0.4317772872, 0.1795202706, 0.8269260435,
                                   0.4792037209);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<28, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.7726786661, 0.8809835661, 0.3749212749,
                                   0.4832692153, 0.6596762329, 0.4412282588,
                                   0.341522493, 0.687800793);
      cl::sycl::float8 inputData_1(0.8155198226, 0.8357510755, 0.6013936374,
                                   0.4004570771, 0.8796484172, 0.611102814,
                                   0.1526677418, 0.1677356553);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<29, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.6998956574, 0.1489249252, 0.1062808043, 0.4150463614, 0.515202983,
          0.4588354285, 0.4908950435, 0.5679109616, 0.6434420539, 0.4384304588,
          0.3946651651, 0.8907672465, 0.3087332284, 0.7216801236, 0.4449768197,
          0.3868163056);
      cl::sycl::float16 inputData_1(
          0.1510863592, 0.7908631554, 0.6616033198, 0.822408566, 0.4612894341,
          0.6415367735, 0.1951282292, 0.4183628813, 0.2657855787, 0.1336811423,
          0.858369081, 0.2727154948, 0.2170835918, 0.2583760348, 0.4024255715,
          0.5371130099);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<30, int64_t>([=]() {
      double inputData_0(0.2210674948);
      double inputData_1(0.8909519112);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<31, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.8863913684, 0.2187216137);
      cl::sycl::double2 inputData_1(0.4247255065, 0.6439435865);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<32, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.8021252663, 0.4963247399, 0.8336373382);
      cl::sycl::double3 inputData_1(0.3579682519, 0.4987527132, 0.4989172735);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<33, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.6360545211, 0.261593047, 0.5878164883,
                                    0.2750184775);
      cl::sycl::double4 inputData_1(0.372176252, 0.8700531706, 0.8192064304,
                                    0.7544947047);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<34, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.1283746095, 0.218693506, 0.3055055297,
                                    0.7273332546, 0.7738666617, 0.5663585442,
                                    0.6745053214, 0.745644304);
      cl::sycl::double8 inputData_1(0.1530873048, 0.1677145095, 0.7951162512,
                                    0.1315326635, 0.2800725229, 0.1325056213,
                                    0.112228112, 0.7751637486);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<35, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.3644754938, 0.2285520482, 0.2190555922, 0.6248669294, 0.8748786174,
          0.5039997541, 0.8208723815, 0.5019428792, 0.559097982, 0.6428570854,
          0.7440879912, 0.7062771058, 0.8924260502, 0.6975723113, 0.8246245787,
          0.2648838657);
      cl::sycl::double16 inputData_1(
          0.5283330435, 0.5788914109, 0.7605572937, 0.4857708505, 0.7328321694,
          0.4108551121, 0.5691107645, 0.781053286, 0.7384475769, 0.6255876415,
          0.1001925572, 0.2455751377, 0.5054862295, 0.3035675188, 0.1524966746,
          0.7879067377);
      return cl::sycl::isgreater(inputData_0, inputData_1);

    });

    test_function<36, int32_t>([=]() {
      float inputData_0(0.8543576171);
      float inputData_1(0.3422439025);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<37, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.4264585339, 0.7480300271);
      cl::sycl::float2 inputData_1(0.1498070071, 0.61278789);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<38, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.2018566503, 0.329670672, 0.7639525493);
      cl::sycl::float3 inputData_1(0.1444216367, 0.1287470667, 0.4342928358);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<39, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.4934647673, 0.7906601465, 0.6737509971,
                                   0.6388350469);
      cl::sycl::float4 inputData_1(0.2210990179, 0.8893647394, 0.428912157,
                                   0.5894166915);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<40, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.4093464044, 0.1376263326, 0.4767113672,
                                   0.2210942031, 0.1259723699, 0.5939203389,
                                   0.603973033, 0.1842342597);
      cl::sycl::float8 inputData_1(0.539315013, 0.3773343813, 0.4067312585,
                                   0.721135919, 0.492255742, 0.8050212923,
                                   0.5880957943, 0.473750732);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<41, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.605850112, 0.3702923039, 0.199459034, 0.646023695, 0.5976299542,
          0.7308531931, 0.2016873, 0.8294266545, 0.7394729691, 0.8335099265,
          0.7980277774, 0.6448051571, 0.7482006795, 0.5152058474, 0.7283913195,
          0.2513019743);
      cl::sycl::float16 inputData_1(
          0.7256912851, 0.4556636832, 0.705292977, 0.4643761894, 0.7316469826,
          0.1602716682, 0.1357127243, 0.8474316659, 0.4889320806, 0.8208571197,
          0.8558266015, 0.633208922, 0.5574374609, 0.2727835073, 0.1747809754,
          0.7555153721);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<42, int64_t>([=]() {
      double inputData_0(0.8110176541);
      double inputData_1(0.7235165686);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<43, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.6588019462, 0.4360888929);
      cl::sycl::double2 inputData_1(0.344249272, 0.1907559165);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<44, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.4407761985, 0.5528103794, 0.8383044665);
      cl::sycl::double3 inputData_1(0.8486038155, 0.4325129572, 0.1793687905);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<45, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.719054986, 0.6874234733, 0.1245606768,
                                    0.4573748793);
      cl::sycl::double4 inputData_1(0.6491344834, 0.1241073876, 0.8354258827,
                                    0.8697939892);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<46, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.6780342177, 0.1628308317, 0.1562635727,
                                    0.3874026519, 0.1235020062, 0.3783021818,
                                    0.1079713931, 0.8794588103);
      cl::sycl::double8 inputData_1(0.7552053593, 0.1564140892, 0.8147480735,
                                    0.266382432, 0.2638326386, 0.6390073164,
                                    0.8506098145, 0.198550497);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<47, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.1057476538, 0.3953041177, 0.1197200115, 0.5838785901, 0.7873404869,
          0.2495933619, 0.1899128287, 0.3755596859, 0.8673372165, 0.2041261555,
          0.8732154084, 0.389791896, 0.4786963222, 0.3341055888, 0.8497014754,
          0.866518316);
      cl::sycl::double16 inputData_1(
          0.6087325652, 0.2472364401, 0.8943614309, 0.1820643516, 0.5646795053,
          0.2251224481, 0.8181402513, 0.8565427132, 0.7435122384, 0.3527131349,
          0.294270952, 0.7038867306, 0.3328476153, 0.4358283023, 0.1370045415,
          0.2057870483);
      return cl::sycl::isgreaterequal(inputData_0, inputData_1);

    });

    test_function<48, int32_t>([=]() {
      float inputData_0(0.1164396965);
      float inputData_1(0.1623368961);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<49, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.1585689195, 0.4361853617);
      cl::sycl::float2 inputData_1(0.5406217421, 0.6927030559);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<50, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.2138267791, 0.4377509969, 0.6095728299);
      cl::sycl::float3 inputData_1(0.1676445559, 0.4558489241, 0.3954048314);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<51, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.8591455432, 0.1462856911, 0.4269009769,
                                   0.4337803838);
      cl::sycl::float4 inputData_1(0.6825444037, 0.3565368023, 0.2631922208,
                                   0.3346493241);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<52, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.476710034, 0.8602146637, 0.7372136182,
                                   0.3215761966, 0.5465452707, 0.6505602429,
                                   0.7365257245, 0.4569315072);
      cl::sycl::float8 inputData_1(0.4190215241, 0.7141125943, 0.4453731965,
                                   0.2983661351, 0.4627576252, 0.849683717,
                                   0.2140539906, 0.4699482836);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<53, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.6098428195, 0.4866303906, 0.2629119235, 0.1014745285, 0.6591933694,
          0.5949884144, 0.1062213195, 0.3388480968, 0.7149074076, 0.6031363028,
          0.5361664928, 0.2249768878, 0.6650352344, 0.4771479374, 0.642542997,
          0.7080718694);
      cl::sycl::float16 inputData_1(
          0.2858901772, 0.7095960105, 0.3240707078, 0.8872121097, 0.1966652886,
          0.806974415, 0.1324377, 0.3052606547, 0.520881527, 0.5652929468,
          0.416987988, 0.1816253826, 0.3020864687, 0.3267172031, 0.7041782836,
          0.8270194602);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<54, int64_t>([=]() {
      double inputData_0(0.5763279324);
      double inputData_1(0.1283607726);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<55, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.7337891773, 0.3444831463);
      cl::sycl::double2 inputData_1(0.3719123251, 0.5241483501);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<56, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.2992376381, 0.8359824703, 0.2308438067);
      cl::sycl::double3 inputData_1(0.4318643204, 0.3317535596, 0.5158672818);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<57, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.5591854425, 0.6017117513, 0.5251006431,
                                    0.4286436019);
      cl::sycl::double4 inputData_1(0.6076752099, 0.4227303013, 0.7228402072,
                                    0.7305419402);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<58, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.3338033345, 0.3974434588, 0.6030487248,
                                    0.2256559737, 0.6576255448, 0.4051422024,
                                    0.5728499798, 0.2116264794);
      cl::sycl::double8 inputData_1(0.6346067089, 0.3832462885, 0.478132461,
                                    0.4320859207, 0.4813721984, 0.6557565063,
                                    0.3545921415, 0.6216435847);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<59, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.148177686, 0.340148122, 0.6961677521, 0.1419247024, 0.5969137562,
          0.1204374394, 0.4772230946, 0.810836035, 0.1080880752, 0.5214624165,
          0.1531654637, 0.7936878209, 0.6490372178, 0.6935630853, 0.635206064,
          0.105138763);
      cl::sycl::double16 inputData_1(
          0.1329422898, 0.5967014432, 0.8997481005, 0.7985177913, 0.6597486454,
          0.6816799635, 0.2813496181, 0.7012911473, 0.3303392839, 0.1843682136,
          0.4687159164, 0.364156618, 0.2346043189, 0.437367914, 0.8177607816,
          0.4482162186);
      return cl::sycl::isless(inputData_0, inputData_1);

    });

    test_function<60, int32_t>([=]() {
      float inputData_0(0.4578335162);
      float inputData_1(0.667062206);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<61, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.5193294961, 0.2033784283);
      cl::sycl::float2 inputData_1(0.828313918, 0.4552994689);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<62, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.7314701914, 0.411100104, 0.7454768151);
      cl::sycl::float3 inputData_1(0.4116291328, 0.2761276173, 0.2569557335);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<63, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.8520277155, 0.5692242069, 0.139834612,
                                   0.4106780769);
      cl::sycl::float4 inputData_1(0.2872234084, 0.1677256517, 0.2494046948,
                                   0.145592384);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<64, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.6104589026, 0.2386990919, 0.588623901,
                                   0.5900053983, 0.6639389686, 0.5096949205,
                                   0.3275391923, 0.8019659631);
      cl::sycl::float8 inputData_1(0.3824568654, 0.46663546, 0.6055035454,
                                   0.5128994385, 0.8651746789, 0.863774142,
                                   0.8438078805, 0.8472610797);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<65, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.5647681085, 0.492161651, 0.6632934539, 0.2723356744, 0.3126976314,
          0.1350458029, 0.230286034, 0.10309964, 0.6237020612, 0.2123255912,
          0.7293434765, 0.6444031967, 0.8765406347, 0.4172115896, 0.8371135308,
          0.4629633379);
      cl::sycl::float16 inputData_1(
          0.3716029919, 0.1818710959, 0.8062657481, 0.7358321269, 0.3583431812,
          0.4645955079, 0.3601147727, 0.1230632932, 0.1354820203, 0.3949633007,
          0.2676730625, 0.5196116826, 0.2502280285, 0.2612972692, 0.6381343051,
          0.6884821254);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<66, int64_t>([=]() {
      double inputData_0(0.3497856767);
      double inputData_1(0.7879955195);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<67, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.3037113397, 0.375152301);
      cl::sycl::double2 inputData_1(0.6699843123, 0.1356023211);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<68, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.8473467681, 0.1578701854, 0.4687448472);
      cl::sycl::double3 inputData_1(0.6796838608, 0.137974828, 0.7472021485);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<69, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.8831146746, 0.4684093382, 0.194498909,
                                    0.1651815965);
      cl::sycl::double4 inputData_1(0.1789843489, 0.7123530993, 0.4312102788,
                                    0.8353873266);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<70, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.4525118209, 0.1617146481, 0.44154847,
                                    0.7038623147, 0.7634707415, 0.1314813492,
                                    0.244311513, 0.4920107616);
      cl::sycl::double8 inputData_1(0.2024683824, 0.7968741136, 0.8475687108,
                                    0.3556775987, 0.447874946, 0.5456432515,
                                    0.3284046329, 0.532860558);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<71, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.2609480364, 0.337313001, 0.4534269066, 0.5837359218, 0.5289320209,
          0.3087903814, 0.2854303003, 0.1949841894, 0.7267949087, 0.1791206132,
          0.6863080049, 0.2990189565, 0.3276455872, 0.6888667464, 0.6276966334,
          0.6935372444);
      cl::sycl::double16 inputData_1(
          0.512226447, 0.7872766557, 0.1974351131, 0.6161575691, 0.19459545,
          0.6898266945, 0.3871237292, 0.6399056835, 0.6627871308, 0.6284867661,
          0.2772463843, 0.7654399091, 0.2921088699, 0.5145226378, 0.6397166033,
          0.2868825398);
      return cl::sycl::islessequal(inputData_0, inputData_1);

    });

    test_function<72, int32_t>([=]() {
      float inputData_0(0.6028093784);
      float inputData_1(0.3294648384);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<73, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.2371059009, 0.7477990628);
      cl::sycl::float2 inputData_1(0.5424982161, 0.3623077653);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<74, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.5683447578, 0.1202291179, 0.2038582854);
      cl::sycl::float3 inputData_1(0.4164646814, 0.8806052636, 0.5083796143);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<75, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.1611649641, 0.7120324922, 0.7251550967,
                                   0.7198417395);
      cl::sycl::float4 inputData_1(0.5555984304, 0.6565589903, 0.270766349,
                                   0.6860484727);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<76, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.7529391899, 0.7079732322, 0.3827699221,
                                   0.5728224405, 0.603191486, 0.8206478829,
                                   0.1864111162, 0.7671470167);
      cl::sycl::float8 inputData_1(0.5211484468, 0.3868912964, 0.4644823212,
                                   0.1101083991, 0.2760588739, 0.6222107361,
                                   0.6286794238, 0.4957591522);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<77, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.8626607045, 0.4847320708, 0.3511549276, 0.7782246714, 0.3073266395,
          0.5834447944, 0.6627350819, 0.757357039, 0.7282950001, 0.4072738644,
          0.1473442448, 0.1306302924, 0.6811683103, 0.8693531051, 0.3745322994,
          0.4529560785);
      cl::sycl::float16 inputData_1(
          0.6806384126, 0.6262649967, 0.3080852708, 0.6372678766, 0.3439219357,
          0.3850863252, 0.5316106442, 0.6858510591, 0.2209729693, 0.1175897687,
          0.6022639636, 0.1196517422, 0.1359705926, 0.2806204614, 0.6231014986,
          0.1532360781);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<78, int64_t>([=]() {
      double inputData_0(0.1499246141);
      double inputData_1(0.8776745955);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<79, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.438122315, 0.8139431472);
      cl::sycl::double2 inputData_1(0.2732194272, 0.4481705436);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<80, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.3864281077, 0.2415484288, 0.3630505486);
      cl::sycl::double3 inputData_1(0.889436655, 0.6978472078, 0.4061346233);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<81, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.4274275475, 0.3109927209, 0.5250693429,
                                    0.6885095297);
      cl::sycl::double4 inputData_1(0.6493172926, 0.4701198683, 0.1335512374,
                                    0.8372062452);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<82, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.4271470425, 0.4122390936, 0.1024880916,
                                    0.2105817713, 0.795082734, 0.5111476769,
                                    0.6859478754, 0.2185343091);
      cl::sycl::double8 inputData_1(0.3640408053, 0.7721092452, 0.7565268169,
                                    0.2974354145, 0.1175802467, 0.7451735788,
                                    0.235075204, 0.7301451137);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<83, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.6469273839, 0.2346518082, 0.1627909149, 0.8421195439, 0.5783027178,
          0.5964081384, 0.4660094423, 0.2200567819, 0.5815759304, 0.301978304,
          0.7447157248, 0.6861751638, 0.121813748, 0.8459384077, 0.1290528387,
          0.1716954551);
      cl::sycl::double16 inputData_1(
          0.3341876487, 0.2206472484, 0.2889160663, 0.3846475909, 0.6883997724,
          0.4237690886, 0.3158718038, 0.4938505229, 0.4140745998, 0.348611358,
          0.8204333263, 0.5403587608, 0.8818620088, 0.7183299275, 0.5563994381,
          0.3099572714);
      return cl::sycl::islessgreater(inputData_0, inputData_1);

    });

    test_function<84, int32_t>([=]() {
      float inputData_0(0.649474925);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<85, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.4647341752, 0.677110172);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<86, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.4230230471, 0.4968040291, 0.116547014);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<87, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.6919668019, 0.1274188355, 0.6445803087,
                                   0.5656029564);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<88, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.7207340892, 0.3318220794, 0.6488886521,
                                   0.2656783805, 0.5234176011, 0.3722243034,
                                   0.8827636411, 0.8774932459);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<89, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.2671757884, 0.5528305887, 0.3635541487, 0.8748305496, 0.8396207585,
          0.5689166824, 0.6760675641, 0.6450598054, 0.382684506, 0.8330892555,
          0.8195628295, 0.3645267716, 0.6979159285, 0.1072737013, 0.7530872884,
          0.5518954763);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<90, int64_t>([=]() {
      double inputData_0(0.8618453702);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<91, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.3905544596, 0.6005704599);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<92, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.3584019452, 0.7262283051, 0.5805623974);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<93, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.8899768184, 0.1008102345, 0.2126069937,
                                    0.1348811057);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<94, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.2006782791, 0.8435082377, 0.8588866396,
                                    0.4843300278, 0.8573515157, 0.7547100882,
                                    0.7228941873, 0.6978255606);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<95, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.2501236681, 0.5391018089, 0.4391033845, 0.8598304381, 0.2390668305,
          0.2358870748, 0.6270894029, 0.2259214317, 0.1880429384, 0.5031385579,
          0.737328777, 0.5840365373, 0.7038031783, 0.3126068253, 0.3279702664,
          0.4429629134);
      return cl::sycl::isfinite(inputData_0);

    });

    test_function<96, int32_t>([=]() {
      float inputData_0(0.8926783074);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<97, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.6743346052, 0.8570031658);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<98, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.5302963646, 0.5436478813, 0.8920722684);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<99, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.2519906205, 0.7260723497, 0.7332110592,
                                   0.7757933021);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<100, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.7000421674, 0.2242664146, 0.6289021057,
                                   0.8389625477, 0.550628122, 0.3887532083,
                                   0.8596161189, 0.5492789204);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<101, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.4293091156, 0.5913067984, 0.7433000133, 0.2826416725, 0.1125536344,
          0.5232759177, 0.8530859361, 0.6442063701, 0.6047264001, 0.602252118,
          0.4975917698, 0.6847354158, 0.299355552, 0.8134034112, 0.3195781242,
          0.8559560106);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<102, int64_t>([=]() {
      double inputData_0(0.841197368);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<103, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.1623396192, 0.458543761);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<104, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.6952290279, 0.4597232572, 0.5071192199);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<105, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.7454591501, 0.6639937288, 0.8664033782,
                                    0.2315887954);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<106, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.8388474289, 0.842389002, 0.6077991509,
                                    0.8523126618, 0.3021484699, 0.8054298267,
                                    0.7187834344, 0.5877511998);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<107, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.1725033957, 0.124107483, 0.1087755964, 0.300446446, 0.709881928,
          0.4093000259, 0.7203573801, 0.6005139927, 0.4114095193, 0.804117303,
          0.1307337915, 0.4722503922, 0.7638818715, 0.2014507864, 0.6683900492,
          0.3624926735);
      return cl::sycl::isinf(inputData_0);

    });

    test_function<108, int32_t>([=]() {
      float inputData_0(0.1194410229);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<109, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.4789799447, 0.5173541912);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<110, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.1332690005, 0.5527354828, 0.3779470704);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<111, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.1035945658, 0.2526187045, 0.1886485378,
                                   0.5324975638);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<112, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.1344961304, 0.8425060561, 0.7760495779,
                                   0.856238115, 0.3518408262, 0.8242139108,
                                   0.887449938, 0.7117851474);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<113, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.3200660909, 0.6367114433, 0.576530523, 0.4233626417, 0.3448782832,
          0.1478785525, 0.2003059798, 0.2071649248, 0.4847142917, 0.6135147078,
          0.711254762, 0.137371008, 0.7590078981, 0.1347769786, 0.543957464,
          0.6953182798);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<114, int64_t>([=]() {
      double inputData_0(0.6049770974);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<115, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.8597429405, 0.3757586825);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<116, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.5687066842, 0.1662392502, 0.5478372703);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<117, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.7506390409, 0.2612836111, 0.3087716003,
                                    0.6603245122);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<118, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.3031055735, 0.3073963792, 0.8484122304,
                                    0.8988344247, 0.2241587446, 0.8201299098,
                                    0.5421811888, 0.1308809139);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<119, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.5684021722, 0.6132397205, 0.127036559, 0.7061535377, 0.7542401132,
          0.1573145937, 0.6187199521, 0.4652379847, 0.2909770299, 0.4669363053,
          0.2275117678, 0.3669327254, 0.6241658398, 0.4811884449, 0.5447360757,
          0.534754235);
      return cl::sycl::isnan(inputData_0);

    });

    test_function<120, int32_t>([=]() {
      float inputData_0(0.7564753921);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<121, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.3747062385, 0.7503696726);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<122, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.1639896697, 0.4421864367, 0.3818560932);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<123, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.461264511, 0.7668078564, 0.5099195204,
                                   0.889797317);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<124, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.7891685762, 0.1950773963, 0.3535132284,
                                   0.1181804012, 0.6870027371, 0.1153606435,
                                   0.8087508119, 0.2546742919);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<125, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.4310690322, 0.1496314448, 0.3490039098, 0.4116119916, 0.1417847788,
          0.7140405225, 0.6690797756, 0.3863068993, 0.7681540425, 0.1619374419,
          0.1432051208, 0.3839842355, 0.8214730657, 0.7051741615, 0.6378541428,
          0.5501885882);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<126, int64_t>([=]() {
      double inputData_0(0.743012431);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<127, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.4297813546, 0.1245508638);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<128, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.7419234292, 0.2523947474, 0.4101270799);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<129, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.3860874778, 0.1986925007, 0.3806274952,
                                    0.2416695023);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<130, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.5928110641, 0.6227474862, 0.1109172423,
                                    0.4651806819, 0.5432421252, 0.7973303955,
                                    0.4968250482, 0.1643603495);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<131, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.1413791033, 0.7896872664, 0.7325835275, 0.7867583449, 0.3097941341,
          0.6183978886, 0.1765744378, 0.7612584894, 0.3668903491, 0.8641177899,
          0.4771060524, 0.1264541576, 0.8272447021, 0.6004257191, 0.3296650288,
          0.129443115);
      return cl::sycl::isnormal(inputData_0);

    });

    test_function<132, int32_t>([=]() {
      float inputData_0(0.4013491174);
      float inputData_1(0.2254884478);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<133, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.5386242491, 0.2175069401);
      cl::sycl::float2 inputData_1(0.2396914231, 0.8366955886);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<134, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.6120960276, 0.2940651294, 0.8031170245);
      cl::sycl::float3 inputData_1(0.5997726639, 0.8564794721, 0.4863334355);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<135, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.8103206672, 0.6427550459, 0.1353348447,
                                   0.292232392);
      cl::sycl::float4 inputData_1(0.3252611239, 0.2360133467, 0.2905495605,
                                   0.2808321188);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<136, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.8026749994, 0.4703190348, 0.8012094651,
                                   0.2103983068, 0.5519347892, 0.1107741425,
                                   0.8442411279, 0.1045096915);
      cl::sycl::float8 inputData_1(0.4119261188, 0.741268767, 0.8999052474,
                                   0.1156077923, 0.7592683871, 0.5080703676,
                                   0.1305456164, 0.7216954167);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<137, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.1895219301, 0.5891793498, 0.7226601729, 0.6388727414, 0.4038994598,
          0.1211533095, 0.4490111742, 0.8309555869, 0.3663386924, 0.2983669753,
          0.2102646671, 0.508201964, 0.5266786182, 0.1584385939, 0.4262067888,
          0.6269451639);
      cl::sycl::float16 inputData_1(
          0.8728405481, 0.4452328974, 0.4488282695, 0.4769071781, 0.280026793,
          0.4158701136, 0.6162117808, 0.4176473672, 0.5651005987, 0.7684658304,
          0.8983740585, 0.8080317469, 0.3974373015, 0.1173817075, 0.5892836796,
          0.4796405666);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<138, int64_t>([=]() {
      double inputData_0(0.2896136896);
      double inputData_1(0.1322432793);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<139, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.3572561912, 0.7384570503);
      cl::sycl::double2 inputData_1(0.8712952055, 0.1853281113);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<140, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.8021115294, 0.138974137, 0.6707806551);
      cl::sycl::double3 inputData_1(0.1214365707, 0.4368397458, 0.7961846708);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<141, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.4144865181, 0.8396514541, 0.6705561129,
                                    0.5833474246);
      cl::sycl::double4 inputData_1(0.2291032384, 0.3723966269, 0.4288769314,
                                    0.5721638913);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<142, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.8968305282, 0.3269677982, 0.5028503127,
                                    0.8467583261, 0.376336635, 0.6028838298,
                                    0.712905231, 0.60421578);
      cl::sycl::double8 inputData_1(0.7027445439, 0.2565544002, 0.8658701495,
                                    0.2415182455, 0.5669449408, 0.3368340873,
                                    0.6075384202, 0.3328883323);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<143, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.4449706855, 0.6457780386, 0.3152550004, 0.682300706, 0.3775021382,
          0.2057248778, 0.5905029735, 0.2326064231, 0.4444619571, 0.4187179295,
          0.1609350779, 0.6686158699, 0.6446588521, 0.722236004, 0.5359305127,
          0.5431334206);
      cl::sycl::double16 inputData_1(
          0.2353864023, 0.2659711192, 0.2825995924, 0.520242823, 0.755186066,
          0.3855792934, 0.8054975904, 0.6887026148, 0.6731577146, 0.3681377034,
          0.1947819936, 0.8702323849, 0.7836885085, 0.4270943926, 0.7905745522,
          0.819373692);
      return cl::sycl::isordered(inputData_0, inputData_1);

    });

    test_function<144, int32_t>([=]() {
      float inputData_0(0.3739788987);
      float inputData_1(0.501249194);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<145, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.3654318722, 0.6561260113);
      cl::sycl::float2 inputData_1(0.8297338508, 0.8876352831);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<146, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.6950232599, 0.3441938831, 0.8043946321);
      cl::sycl::float3 inputData_1(0.8940957032, 0.377220931, 0.858969882);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<147, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.5092371244, 0.8717083538, 0.8966847921,
                                   0.7503536767);
      cl::sycl::float4 inputData_1(0.6467496394, 0.2232115754, 0.1039338266,
                                   0.5763766803);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<148, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.6635679244, 0.8484304361, 0.5136959202,
                                   0.6574772822, 0.6178847772, 0.2639361,
                                   0.6154400742, 0.8853769691);
      cl::sycl::float8 inputData_1(0.1889479653, 0.6508345946, 0.591444094,
                                   0.400683779, 0.7346782031, 0.1083886869,
                                   0.8139292977, 0.7538911624);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<149, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.4845638652, 0.1865113239, 0.4621028445, 0.5674023193, 0.3031067828,
          0.4892251719, 0.7205830111, 0.8381854365, 0.5493160221, 0.761793428,
          0.1623465704, 0.7850944371, 0.8366516524, 0.234401101, 0.7619898894,
          0.7796529363);
      cl::sycl::float16 inputData_1(
          0.8029270947, 0.5137116159, 0.5866033951, 0.2664665964, 0.6665052395,
          0.4240138427, 0.1169352686, 0.2074136908, 0.4105744253, 0.8081438448,
          0.5519538346, 0.8330056272, 0.8435870755, 0.1694359937, 0.570572331,
          0.3676224305);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<150, int64_t>([=]() {
      double inputData_0(0.5054360978);
      double inputData_1(0.464419842);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<151, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.4839546181, 0.1814446465);
      cl::sycl::double2 inputData_1(0.766528067, 0.4922239694);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<152, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.615990045, 0.4781430022, 0.2448147013);
      cl::sycl::double3 inputData_1(0.532800468, 0.2276317912, 0.7817434049);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<153, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.7652832205, 0.2149110225, 0.1550751624,
                                    0.1547935336);
      cl::sycl::double4 inputData_1(0.4145952199, 0.8624331499, 0.5449123328,
                                    0.3124212592);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<154, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.2837190617, 0.1886985558, 0.2128569684,
                                    0.7494906133, 0.2109067718, 0.7912492457,
                                    0.7583984593, 0.2094470417);
      cl::sycl::double8 inputData_1(0.546979816, 0.1056442144, 0.7896289075,
                                    0.5466216964, 0.7042723155, 0.4922762615,
                                    0.6523375994, 0.8449912993);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<155, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.5476366481, 0.7997643815, 0.3744363539, 0.1780260365, 0.1041157141,
          0.2813202232, 0.7708694695, 0.3491963701, 0.2796929181, 0.4965043387,
          0.857523298, 0.5071827586, 0.3726973301, 0.1620014328, 0.5589335467,
          0.2810055836);
      cl::sycl::double16 inputData_1(
          0.3939993022, 0.4049298934, 0.7065474698, 0.2853030758, 0.8487137806,
          0.6939104544, 0.4848956326, 0.804379593, 0.3873343842, 0.4074718989,
          0.2034953048, 0.7228448756, 0.4209541223, 0.5002024226, 0.4767749323,
          0.6249454541);
      return cl::sycl::isunordered(inputData_0, inputData_1);

    });

    test_function<156, int32_t>([=]() {
      float inputData_0(0.3991507449);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<157, cl::sycl::vec<int32_t, 2>>([=]() {
      cl::sycl::float2 inputData_0(0.8326890609, 0.4455378072);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<158, cl::sycl::vec<int32_t, 3>>([=]() {
      cl::sycl::float3 inputData_0(0.3873711823, 0.4207024441, 0.7130365772);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<159, cl::sycl::vec<int32_t, 4>>([=]() {
      cl::sycl::float4 inputData_0(0.894445272, 0.793211717, 0.4837819988,
                                   0.3330874748);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<160, cl::sycl::vec<int32_t, 8>>([=]() {
      cl::sycl::float8 inputData_0(0.4567896435, 0.3752124425, 0.2948256418,
                                   0.2495527323, 0.8647005876, 0.4994441523,
                                   0.1879798989, 0.4071252881);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<161, cl::sycl::vec<int32_t, 16>>([=]() {
      cl::sycl::float16 inputData_0(
          0.4109735338, 0.5108276216, 0.8840330597, 0.8813067973, 0.5527152886,
          0.5944732202, 0.6405032599, 0.5017777461, 0.4893424465, 0.3516191513,
          0.6471373916, 0.1735162227, 0.353716197, 0.8127828476, 0.2819025208,
          0.8740659024);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<162, int64_t>([=]() {
      double inputData_0(0.8873357776);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<163, cl::sycl::vec<int64_t, 2>>([=]() {
      cl::sycl::double2 inputData_0(0.5603061305, 0.1323487842);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<164, cl::sycl::vec<int64_t, 3>>([=]() {
      cl::sycl::double3 inputData_0(0.1747825579, 0.2602413102, 0.3614492546);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<165, cl::sycl::vec<int64_t, 4>>([=]() {
      cl::sycl::double4 inputData_0(0.1904865693, 0.7377686185, 0.391323656,
                                    0.2869869587);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<166, cl::sycl::vec<int64_t, 8>>([=]() {
      cl::sycl::double8 inputData_0(0.1349550963, 0.4061374875, 0.1036053844,
                                    0.1931931604, 0.583716408, 0.8479563291,
                                    0.2594927375, 0.6928489653);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<167, cl::sycl::vec<int64_t, 16>>([=]() {
      cl::sycl::double16 inputData_0(
          0.2581644168, 0.1011961551, 0.8172304369, 0.7768869902, 0.1534229728,
          0.2417082305, 0.2874407424, 0.8426570917, 0.4055432765, 0.7459054053,
          0.4486508263, 0.4049957334, 0.7122784438, 0.5926087973, 0.3154541554,
          0.5662484786);
      return cl::sycl::signbit(inputData_0);

    });

    test_function<168, int>([=]() {
      signed char inputData_0(52);
      return cl::sycl::any(inputData_0);

    });

    test_function<169, int>([=]() {
      short inputData_0(21435);
      return cl::sycl::any(inputData_0);

    });

    test_function<170, int>([=]() {
      int inputData_0(11611);
      return cl::sycl::any(inputData_0);

    });

    test_function<171, int>([=]() {
      long int inputData_0(604504068);
      return cl::sycl::any(inputData_0);

    });

    test_function<172, int>([=]() {
      long long int inputData_0(314163352589635146);
      return cl::sycl::any(inputData_0);

    });

    test_function<173, int>([=]() {
      cl::sycl::schar2 inputData_0(-64, -102);
      return cl::sycl::any(inputData_0);

    });

    test_function<174, int>([=]() {
      cl::sycl::schar3 inputData_0(-102, -122, -92);
      return cl::sycl::any(inputData_0);

    });

    test_function<175, int>([=]() {
      cl::sycl::schar4 inputData_0(69, -94, -75, 121);
      return cl::sycl::any(inputData_0);

    });

    test_function<176, int>([=]() {
      cl::sycl::schar8 inputData_0(-47, -21, 0, 15, 49, -7, 2, -14);
      return cl::sycl::any(inputData_0);

    });

    test_function<177, int>([=]() {
      cl::sycl::schar16 inputData_0(-83, 12, -96, -63, -108, 1, 81, 92, -12,
                                    -39, 70, -23, -14, 99, 13, 117);
      return cl::sycl::any(inputData_0);

    });

    test_function<178, int>([=]() {
      cl::sycl::short2 inputData_0(18039, -18090);
      return cl::sycl::any(inputData_0);

    });

    test_function<179, int>([=]() {
      cl::sycl::short3 inputData_0(-16970, -1493, -23528);
      return cl::sycl::any(inputData_0);

    });

    test_function<180, int>([=]() {
      cl::sycl::short4 inputData_0(-3873, -9177, -906, 28303);
      return cl::sycl::any(inputData_0);

    });

    test_function<181, int>([=]() {
      cl::sycl::short8 inputData_0(-23016, 21730, 12091, -20474, 28361, 8578,
                                   -24724, -28902);
      return cl::sycl::any(inputData_0);

    });

    test_function<182, int>([=]() {
      cl::sycl::short16 inputData_0(-10302, -10851, 7981, -22259, 26216, -7190,
                                    -20472, -256, -21790, -29680, 29367, 17836,
                                    -10554, 17028, 21614, -20471);
      return cl::sycl::any(inputData_0);

    });

    test_function<183, int>([=]() {
      cl::sycl::int2 inputData_0(-11269, 1258);
      return cl::sycl::any(inputData_0);

    });

    test_function<184, int>([=]() {
      cl::sycl::int3 inputData_0(2803, 23023, 5419);
      return cl::sycl::any(inputData_0);

    });

    test_function<185, int>([=]() {
      cl::sycl::int4 inputData_0(-22797, 12368, -430, 10695);
      return cl::sycl::any(inputData_0);

    });

    test_function<186, int>([=]() {
      cl::sycl::int8 inputData_0(28102, 6426, -30612, -803, 6813, -11949, 15145,
                                 -32735);
      return cl::sycl::any(inputData_0);

    });

    test_function<187, int>([=]() {
      cl::sycl::int16 inputData_0(-28688, -16853, 24520, 7207, 10540, -25622,
                                  30517, 1030, 27827, 8786, -13999, 11994, -431,
                                  -20176, -8347, 22494);
      return cl::sycl::any(inputData_0);

    });

    test_function<188, int>([=]() {
      cl::sycl::long2 inputData_0(934180686, -468934130);
      return cl::sycl::any(inputData_0);

    });

    test_function<189, int>([=]() {
      cl::sycl::long3 inputData_0(2104637493, -783715269, 2112476270);
      return cl::sycl::any(inputData_0);

    });

    test_function<190, int>([=]() {
      cl::sycl::long4 inputData_0(-1858332636, -1203560744, -1660877704,
                                  -1478455746);
      return cl::sycl::any(inputData_0);

    });

    test_function<191, int>([=]() {
      cl::sycl::long8 inputData_0(1874332780, 1935787150, 892297233,
                                  -1245454692, -567466922, -1947324768,
                                  -403133626, -1771380773);
      return cl::sycl::any(inputData_0);

    });

    test_function<192, int>([=]() {
      cl::sycl::long16 inputData_0(
          -376509262, 298979931, 1248639085, 175423174, 572856352, 1978641055,
          724834404, 132455348, 1745598447, -821919764, 188612849, -1863821342,
          450311967, 989909519, 1083564160, -1271497043);
      return cl::sycl::any(inputData_0);

    });

    test_function<193, int>([=]() {
      cl::sycl::longlong2 inputData_0(4744537713254111217,
                                      -6394019795975191926);
      return cl::sycl::any(inputData_0);

    });

    test_function<194, int>([=]() {
      cl::sycl::longlong3 inputData_0(8872975169495417721, 7884144868987929076,
                                      8035201109717797041);
      return cl::sycl::any(inputData_0);

    });

    test_function<195, int>([=]() {
      cl::sycl::longlong4 inputData_0(-6368089271617440225, 503855524135598155,
                                      -3330795772390640561,
                                      6865228204692103936);
      return cl::sycl::any(inputData_0);

    });

    test_function<196, int>([=]() {
      cl::sycl::longlong8 inputData_0(
          4826365131748427378, -8649269870630155448, -1295503639731006507,
          2385542965115448031, -2134346063461457468, -3843755390663350143,
          773307609841011770, 5893771085972119374);
      return cl::sycl::any(inputData_0);

    });

    test_function<197, int>([=]() {
      cl::sycl::longlong16 inputData_0(
          6695708426399163243, -6683842999557652826, 9133942184849621309,
          1129684827932159838, 381865431065705633, -271675892309835410,
          -2816434119176605274, 9141384724173490508, -1862518588466046016,
          -2796232116803647745, -5237618784459085996, -3171337782660449606,
          -3260126983878852045, 2060739005634728615, 5595793993634613144,
          3908031390915926584);
      return cl::sycl::any(inputData_0);

    });

    test_function<198, int>([=]() {
      signed char inputData_0(122);
      return cl::sycl::all(inputData_0);

    });

    test_function<199, int>([=]() {
      short inputData_0(832);
      return cl::sycl::all(inputData_0);

    });

    test_function<200, int>([=]() {
      int inputData_0(-10070);
      return cl::sycl::all(inputData_0);

    });

    test_function<201, int>([=]() {
      long int inputData_0(-218658526);
      return cl::sycl::all(inputData_0);

    });

    test_function<202, int>([=]() {
      long long int inputData_0(-1675409984900553983);
      return cl::sycl::all(inputData_0);

    });

    test_function<203, int>([=]() {
      cl::sycl::schar2 inputData_0(-108, 122);
      return cl::sycl::all(inputData_0);

    });

    test_function<204, int>([=]() {
      cl::sycl::schar3 inputData_0(127, -84, -67);
      return cl::sycl::all(inputData_0);

    });

    test_function<205, int>([=]() {
      cl::sycl::schar4 inputData_0(-17, 50, -120, 85);
      return cl::sycl::all(inputData_0);

    });

    test_function<206, int>([=]() {
      cl::sycl::schar8 inputData_0(35, -60, 94, 41, -47, 12, 122, -116);
      return cl::sycl::all(inputData_0);

    });

    test_function<207, int>([=]() {
      cl::sycl::schar16 inputData_0(53, 89, 49, -93, 24, 73, -21, 21, -64, -48,
                                    78, -3, -14, -97, -33, 5);
      return cl::sycl::all(inputData_0);

    });

    test_function<208, int>([=]() {
      cl::sycl::short2 inputData_0(-17629, 20180);
      return cl::sycl::all(inputData_0);

    });

    test_function<209, int>([=]() {
      cl::sycl::short3 inputData_0(-7622, -17139, -12564);
      return cl::sycl::all(inputData_0);

    });

    test_function<210, int>([=]() {
      cl::sycl::short4 inputData_0(21264, 26485, 30166, -31773);
      return cl::sycl::all(inputData_0);

    });

    test_function<211, int>([=]() {
      cl::sycl::short8 inputData_0(16639, 1670, -24605, -16612, -14308, -6278,
                                   -1919, 28625);
      return cl::sycl::all(inputData_0);

    });

    test_function<212, int>([=]() {
      cl::sycl::short16 inputData_0(-28944, 13708, 23206, -9352, -16436, -18265,
                                    -13053, -23246, 3386, -16358, -30983,
                                    -17523, 21012, -5416, 25135, 29072);
      return cl::sycl::all(inputData_0);

    });

    test_function<213, int>([=]() {
      cl::sycl::int2 inputData_0(-16820, 3930);
      return cl::sycl::all(inputData_0);

    });

    test_function<214, int>([=]() {
      cl::sycl::int3 inputData_0(24973, 5335, -21758);
      return cl::sycl::all(inputData_0);

    });

    test_function<215, int>([=]() {
      cl::sycl::int4 inputData_0(-16519, 31956, -13148, 24097);
      return cl::sycl::all(inputData_0);

    });

    test_function<216, int>([=]() {
      cl::sycl::int8 inputData_0(19333, 15858, 14545, 19004, 22767, -28681,
                                 -21771, 362);
      return cl::sycl::all(inputData_0);

    });

    test_function<217, int>([=]() {
      cl::sycl::int16 inputData_0(-18843, 2176, -447, -24460, -27135, -32005,
                                  21301, -27411, 30249, 31708, 16101, -3252,
                                  -14694, -5738, -10139, -6797);
      return cl::sycl::all(inputData_0);

    });

    test_function<218, int>([=]() {
      cl::sycl::long2 inputData_0(971503490, 1685887224);
      return cl::sycl::all(inputData_0);

    });

    test_function<219, int>([=]() {
      cl::sycl::long3 inputData_0(-1470102766, -1105221449, -1245983301);
      return cl::sycl::all(inputData_0);

    });

    test_function<220, int>([=]() {
      cl::sycl::long4 inputData_0(-1952724104, 1521279905, 48428089,
                                  -1859571466);
      return cl::sycl::all(inputData_0);

    });

    test_function<221, int>([=]() {
      cl::sycl::long8 inputData_0(-230831897, -212125034, 1193811740,
                                  1122693493, -1569857837, 544926620, 41602313,
                                  -2089536909);
      return cl::sycl::all(inputData_0);

    });

    test_function<222, int>([=]() {
      cl::sycl::long16 inputData_0(
          -1512963093, 716608523, -571119814, 1991513609, 7517748, 808670056,
          -1573600256, -88268199, 1005544061, 1432292891, -1290176269,
          -442782262, -113700631, -256098620, -105477222, -876601858);
      return cl::sycl::all(inputData_0);

    });

    test_function<223, int>([=]() {
      cl::sycl::longlong2 inputData_0(1901693478527224894, 1452717808480105357);
      return cl::sycl::all(inputData_0);

    });

    test_function<224, int>([=]() {
      cl::sycl::longlong3 inputData_0(-5740444319484428330, 178140559663917671,
                                      7693559934227690035);
      return cl::sycl::all(inputData_0);

    });

    test_function<225, int>([=]() {
      cl::sycl::longlong4 inputData_0(
          -1228120728670559697, -8886191716271714380, -4285653166305799562,
          7903514129575314986);
      return cl::sycl::all(inputData_0);

    });

    test_function<226, int>([=]() {
      cl::sycl::longlong8 inputData_0(
          6828070617691816698, 5941335046166813226, 5962162529846084867,
          5203235993424191407, -4195843876635996447, -2140707706070306314,
          -8224336294169008726, 1927327790084708333);
      return cl::sycl::all(inputData_0);

    });

    test_function<227, int>([=]() {
      cl::sycl::longlong16 inputData_0(
          -1488270987344422077, -7786362795307710688, 4697406186060579771,
          1992401486009680096, 3153327705921480287, 7816842250855621687,
          5304501071647141885, 5042495331198271230, -326926494046534051,
          4223675755277248195, 4876311171361168978, 2495550843274843422,
          -7844831880950853653, 1989073662182443295, 7163883812554042845,
          -6632705287178530904);
      return cl::sycl::all(inputData_0);

    });

    test_function<228, float>([=]() {
      float inputData_0(0.6623225116);
      float inputData_1(0.6151221429);
      float inputData_2(0.699603778);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<229, double>([=]() {
      double inputData_0(0.2800914767);
      double inputData_1(0.2673905814);
      double inputData_2(0.8319623942);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<230, cl::sycl::half>([=]() {
      cl::sycl::half inputData_0(0.8219989242);
      cl::sycl::half inputData_1(0.1377344048);
      cl::sycl::half inputData_2(0.7256129169);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<231, unsigned char>([=]() {
      unsigned char inputData_0(33);
      unsigned char inputData_1(187);
      unsigned char inputData_2(76);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<232, signed char>([=]() {
      signed char inputData_0(-25);
      signed char inputData_1(122);
      signed char inputData_2(85);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<233, unsigned short>([=]() {
      unsigned short inputData_0(35805);
      unsigned short inputData_1(48673);
      unsigned short inputData_2(29208);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<234, short>([=]() {
      short inputData_0(2778);
      short inputData_1(3969);
      short inputData_2(26340);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<235, unsigned int>([=]() {
      unsigned int inputData_0(42221);
      unsigned int inputData_1(38571);
      unsigned int inputData_2(11033);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<236, int>([=]() {
      int inputData_0(-26917);
      int inputData_1(-1993);
      int inputData_2(-15213);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<237, unsigned long int>([=]() {
      unsigned long int inputData_0(908961100);
      unsigned long int inputData_1(2940529376);
      unsigned long int inputData_2(3714774164);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<238, long int>([=]() {
      long int inputData_0(1555880033);
      long int inputData_1(100799928);
      long int inputData_2(1194728034);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<239, unsigned long long int>([=]() {
      unsigned long long int inputData_0(1154286504273872955);
      unsigned long long int inputData_1(3820427339003463635);
      unsigned long long int inputData_2(1556120435182349339);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<240, long long int>([=]() {
      long long int inputData_0(-1129739717919214768);
      long long int inputData_1(945997721878370511);
      long long int inputData_2(-3442326997472262741);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<241, cl::sycl::float2>([=]() {
      cl::sycl::float2 inputData_0(0.8034434204, 0.2145387599);
      cl::sycl::float2 inputData_1(0.1275572978, 0.6156290803);
      cl::sycl::float2 inputData_2(0.1927033881, 0.154491231);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<242, cl::sycl::float3>([=]() {
      cl::sycl::float3 inputData_0(0.3140895341, 0.3421663425, 0.8506508152);
      cl::sycl::float3 inputData_1(0.7954857929, 0.4815692032, 0.5731799115);
      cl::sycl::float3 inputData_2(0.3553426489, 0.635739612, 0.5978707103);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<243, cl::sycl::float4>([=]() {
      cl::sycl::float4 inputData_0(0.5099288186, 0.2858365224, 0.5017412248,
                                   0.6801627063);
      cl::sycl::float4 inputData_1(0.7253216146, 0.3927886387, 0.7339991779,
                                   0.1055160299);
      cl::sycl::float4 inputData_2(0.5594912176, 0.4802349216, 0.3943923082,
                                   0.7132680811);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<244, cl::sycl::float8>([=]() {
      cl::sycl::float8 inputData_0(0.2951155993, 0.1194318087, 0.5062467463,
                                   0.8796332064, 0.5559239945, 0.131798875,
                                   0.3926054012, 0.3727286951);
      cl::sycl::float8 inputData_1(0.4888361304, 0.7047783548, 0.8057982382,
                                   0.4352711737, 0.7489321509, 0.7156786098,
                                   0.6234806159, 0.4819321677);
      cl::sycl::float8 inputData_2(0.6802147203, 0.375408135, 0.7422388246,
                                   0.7607515859, 0.6805133591, 0.8376141372,
                                   0.7089642211, 0.484189579);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<245, cl::sycl::float16>([=]() {
      cl::sycl::float16 inputData_0(
          0.6267995775, 0.8425818513, 0.7664778792, 0.6562746661, 0.647960072,
          0.5642129331, 0.2704230303, 0.1218360883, 0.4633907139, 0.5984617567,
          0.7607116936, 0.6367033611, 0.5892566707, 0.3521984075, 0.3622089683,
          0.44395133);
      cl::sycl::float16 inputData_1(
          0.7155592851, 0.7920549728, 0.2152695269, 0.4436265313, 0.2897825561,
          0.4047858554, 0.292494495, 0.3039339659, 0.2934381934, 0.5697234086,
          0.8422164314, 0.3976519224, 0.4644280388, 0.5794604441, 0.5991857328,
          0.217170908);
      cl::sycl::float16 inputData_2(
          0.2948599586, 0.7752624915, 0.3053606185, 0.2383804115, 0.6637227876,
          0.577157057, 0.8149418945, 0.4575136213, 0.2638516527, 0.245217315,
          0.3863501406, 0.2992589456, 0.741720522, 0.5326545089, 0.8237928343,
          0.3349943423);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<246, cl::sycl::double2>([=]() {
      cl::sycl::double2 inputData_0(0.3187455863, 0.2616275689);
      cl::sycl::double2 inputData_1(0.7752398992, 0.3522597476);
      cl::sycl::double2 inputData_2(0.116589612, 0.25367223);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<247, cl::sycl::double3>([=]() {
      cl::sycl::double3 inputData_0(0.2576007164, 0.7650156329, 0.777521399);
      cl::sycl::double3 inputData_1(0.4666670744, 0.3006835908, 0.3302126747);
      cl::sycl::double3 inputData_2(0.4891527976, 0.7511373837, 0.7787978749);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<248, cl::sycl::double4>([=]() {
      cl::sycl::double4 inputData_0(0.6941862699, 0.6600210063, 0.6576565229,
                                    0.143562024);
      cl::sycl::double4 inputData_1(0.8317455418, 0.814040995, 0.3677155911,
                                    0.2277703331);
      cl::sycl::double4 inputData_2(0.3129016019, 0.6761619005, 0.4504987079,
                                    0.5383653039);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<249, cl::sycl::double8>([=]() {
      cl::sycl::double8 inputData_0(0.4318939081, 0.8329025595, 0.3999807371,
                                    0.8568389933, 0.1717350645, 0.1641418874,
                                    0.8789329099, 0.6025190001);
      cl::sycl::double8 inputData_1(0.5049047469, 0.1140238515, 0.6280877148,
                                    0.5104548355, 0.8645597859, 0.753348068,
                                    0.895698542, 0.8209032152);
      cl::sycl::double8 inputData_2(0.2879408182, 0.1329996156, 0.176380216,
                                    0.8464889391, 0.3496713703, 0.7526168739,
                                    0.7962484502, 0.6678272493);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<250, cl::sycl::double16>([=]() {
      cl::sycl::double16 inputData_0(
          0.7177340561, 0.5606144246, 0.8416257019, 0.58889322, 0.8667220048,
          0.8049556863, 0.3221787053, 0.2269967849, 0.24696298, 0.8609430923,
          0.2095237084, 0.8007417394, 0.3063033849, 0.7307843703, 0.2162149294,
          0.5929684238);
      cl::sycl::double16 inputData_1(
          0.2635491316, 0.2090811902, 0.7747317592, 0.425564581, 0.8618069302,
          0.2590999357, 0.5710306645, 0.3811589452, 0.6267924397, 0.6424824722,
          0.7042880941, 0.6930376784, 0.1652505809, 0.6761867446, 0.5943589202,
          0.6530050091);
      cl::sycl::double16 inputData_2(
          0.323414998, 0.5793013021, 0.2764591158, 0.8153963813, 0.7591297818,
          0.6922677551, 0.6403590024, 0.4345579632, 0.216379252, 0.5308407846,
          0.5026566644, 0.8596627725, 0.6358477066, 0.4653005798, 0.6313306503,
          0.4540713145);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<251, cl::sycl::half2>([=]() {
      cl::sycl::half2 inputData_0(0.8524970511, 0.6098706885);
      cl::sycl::half2 inputData_1(0.3202733497, 0.6659939956);
      cl::sycl::half2 inputData_2(0.1803065082, 0.4921148257);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<252, cl::sycl::half3>([=]() {
      cl::sycl::half3 inputData_0(0.3821002515, 0.5709059833, 0.3261790169);
      cl::sycl::half3 inputData_1(0.3285908732, 0.4139726404, 0.2401606105);
      cl::sycl::half3 inputData_2(0.1910361526, 0.5719973976, 0.6213613099);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<253, cl::sycl::half4>([=]() {
      cl::sycl::half4 inputData_0(0.5135263792, 0.2107834978, 0.7961961384,
                                  0.8322176365);
      cl::sycl::half4 inputData_1(0.2153443839, 0.3092781657, 0.6868464919,
                                  0.2560654328);
      cl::sycl::half4 inputData_2(0.7330376636, 0.6912518684, 0.3120484438,
                                  0.8793862422);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<254, cl::sycl::half8>([=]() {
      cl::sycl::half8 inputData_0(0.4009340198, 0.1410103163, 0.7664961151,
                                  0.4120157047, 0.6775458895, 0.5396621712,
                                  0.7177820129, 0.6699581119);
      cl::sycl::half8 inputData_1(0.5553855667, 0.6248506071, 0.3443136431,
                                  0.4549383135, 0.7091275607, 0.8908810521,
                                  0.2681076495, 0.2627573394);
      cl::sycl::half8 inputData_2(0.8918477481, 0.1780557843, 0.421691484,
                                  0.5053748469, 0.4732518479, 0.622757716,
                                  0.1258252931, 0.2864077284);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<255, cl::sycl::half16>([=]() {
      cl::sycl::half16 inputData_0(
          0.1761098659, 0.4023855591, 0.1557301932, 0.7628510806, 0.2700956734,
          0.2589694394, 0.4934290236, 0.253171974, 0.7874225384, 0.5105628975,
          0.5830356729, 0.5401895319, 0.5158357599, 0.5968139972, 0.5806056881,
          0.7917809882);
      cl::sycl::half16 inputData_1(
          0.8303623811, 0.6381238747, 0.4770708182, 0.6042414046, 0.7309435906,
          0.3806337258, 0.8842266605, 0.7307236505, 0.6616443792, 0.1277317761,
          0.2580777373, 0.2721940787, 0.6456854356, 0.7605388642, 0.214678885,
          0.4629930167);
      cl::sycl::half16 inputData_2(
          0.1731269157, 0.3208519985, 0.6191652852, 0.4137462874, 0.6904113634,
          0.8496263126, 0.8178179962, 0.3379678182, 0.6366844123, 0.5914261349,
          0.7613017946, 0.3964551307, 0.5970891157, 0.170685996, 0.7889701077,
          0.4274861629);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<256, cl::sycl::uchar2>([=]() {
      cl::sycl::uchar2 inputData_0(225, 219);
      cl::sycl::uchar2 inputData_1(124, 36);
      cl::sycl::uchar2 inputData_2(216, 94);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<257, cl::sycl::schar2>([=]() {
      cl::sycl::schar2 inputData_0(45, -99);
      cl::sycl::schar2 inputData_1(-46, 46);
      cl::sycl::schar2 inputData_2(-109, 103);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<258, cl::sycl::uchar3>([=]() {
      cl::sycl::uchar3 inputData_0(215, 51, 75);
      cl::sycl::uchar3 inputData_1(178, 166, 188);
      cl::sycl::uchar3 inputData_2(101, 36, 19);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<259, cl::sycl::schar3>([=]() {
      cl::sycl::schar3 inputData_0(33, -89, -80);
      cl::sycl::schar3 inputData_1(26, -39, 21);
      cl::sycl::schar3 inputData_2(96, -19, 61);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<260, cl::sycl::uchar4>([=]() {
      cl::sycl::uchar4 inputData_0(29, 40, 113, 217);
      cl::sycl::uchar4 inputData_1(108, 45, 9, 192);
      cl::sycl::uchar4 inputData_2(146, 174, 165, 25);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<261, cl::sycl::schar4>([=]() {
      cl::sycl::schar4 inputData_0(-2, 5, -121, 53);
      cl::sycl::schar4 inputData_1(77, -106, 82, 60);
      cl::sycl::schar4 inputData_2(-63, 15, -32, -68);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<262, cl::sycl::uchar8>([=]() {
      cl::sycl::uchar8 inputData_0(31, 210, 242, 199, 40, 127, 238, 62);
      cl::sycl::uchar8 inputData_1(57, 180, 172, 223, 221, 248, 217, 207);
      cl::sycl::uchar8 inputData_2(151, 45, 27, 24, 159, 147, 66, 107);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<263, cl::sycl::schar8>([=]() {
      cl::sycl::schar8 inputData_0(-118, -109, 83, -13, -40, -63, -24, 103);
      cl::sycl::schar8 inputData_1(123, 39, -74, -106, 56, -65, 5, -13);
      cl::sycl::schar8 inputData_2(121, -126, -64, 20, 12, -95, -44, 17);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<264, cl::sycl::uchar16>([=]() {
      cl::sycl::uchar16 inputData_0(136, 99, 51, 24, 166, 69, 205, 67, 198, 21,
                                    37, 6, 247, 114, 161, 197);
      cl::sycl::uchar16 inputData_1(130, 197, 119, 215, 190, 49, 214, 68, 214,
                                    181, 50, 225, 67, 103, 155, 68);
      cl::sycl::uchar16 inputData_2(223, 177, 53, 66, 180, 208, 168, 111, 82,
                                    184, 49, 105, 238, 246, 31, 91);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<265, cl::sycl::schar16>([=]() {
      cl::sycl::schar16 inputData_0(56, -8, -84, 86, -19, -17, -35, -101, -106,
                                    77, -95, -125, 77, 91, -30, -6);
      cl::sycl::schar16 inputData_1(-28, 56, -21, 44, -53, -85, -57, -97, 43,
                                    -42, 120, -128, -42, -42, 88, -66);
      cl::sycl::schar16 inputData_2(97, 18, -62, -85, -13, -125, -127, -44, -75,
                                    11, 0, -51, -105, 11, -55, -113);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<266, cl::sycl::ushort2>([=]() {
      cl::sycl::ushort2 inputData_0(60362, 7291);
      cl::sycl::ushort2 inputData_1(10783, 10851);
      cl::sycl::ushort2 inputData_2(30490, 59259);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<267, cl::sycl::short2>([=]() {
      cl::sycl::short2 inputData_0(12732, -3942);
      cl::sycl::short2 inputData_1(21741, -30982);
      cl::sycl::short2 inputData_2(-7962, 30197);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<268, cl::sycl::ushort3>([=]() {
      cl::sycl::ushort3 inputData_0(6582, 5430, 50366);
      cl::sycl::ushort3 inputData_1(13674, 47401, 15042);
      cl::sycl::ushort3 inputData_2(42279, 33074, 54849);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<269, cl::sycl::short3>([=]() {
      cl::sycl::short3 inputData_0(-20541, 9161, -32220);
      cl::sycl::short3 inputData_1(-5183, 29291, -17452);
      cl::sycl::short3 inputData_2(-10040, -12201, 8446);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<270, cl::sycl::ushort4>([=]() {
      cl::sycl::ushort4 inputData_0(50026, 30641, 48795, 59228);
      cl::sycl::ushort4 inputData_1(22582, 50816, 12222, 14564);
      cl::sycl::ushort4 inputData_2(30295, 42118, 24399, 40756);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<271, cl::sycl::short4>([=]() {
      cl::sycl::short4 inputData_0(28934, -14004, -16568, 31602);
      cl::sycl::short4 inputData_1(-2669, 1150, 26295, -19810);
      cl::sycl::short4 inputData_2(-7296, 13108, -11026, -24059);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<272, cl::sycl::ushort8>([=]() {
      cl::sycl::ushort8 inputData_0(32213, 58462, 51908, 57482, 24100, 12936,
                                    13371, 32135);
      cl::sycl::ushort8 inputData_1(8619, 5751, 34013, 44026, 36303, 40202,
                                    13622, 28054);
      cl::sycl::ushort8 inputData_2(35687, 47018, 22461, 59942, 12955, 19343,
                                    50105, 58167);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<273, cl::sycl::short8>([=]() {
      cl::sycl::short8 inputData_0(19786, -3351, 906, -14186, 31617, -27128,
                                   -7936, -5734);
      cl::sycl::short8 inputData_1(18454, 12816, 9199, -17551, -1003, -6519,
                                   -32725, -4265);
      cl::sycl::short8 inputData_2(-13859, -976, -6614, 21268, 27390, -6889,
                                   27485, 25831);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<274, cl::sycl::ushort16>([=]() {
      cl::sycl::ushort16 inputData_0(50284, 63561, 47302, 19790, 35891, 19926,
                                     53601, 38391, 50620, 22718, 27817, 62825,
                                     20339, 44676, 64895, 15141);
      cl::sycl::ushort16 inputData_1(1575, 22207, 11996, 58386, 57491, 50838,
                                     11611, 39580, 14996, 64855, 649, 34091,
                                     59767, 44503, 44248, 11764);
      cl::sycl::ushort16 inputData_2(26008, 20415, 34821, 10446, 10344, 58731,
                                     43316, 31037, 48038, 48084, 20005, 25636,
                                     38402, 17511, 49814, 4444);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<275, cl::sycl::short16>([=]() {
      cl::sycl::short16 inputData_0(17382, 13835, -13520, 28182, -2964, -12494,
                                    28015, -8021, -28309, -30118, 19298, -26776,
                                    -32305, -12529, -32692, 8174);
      cl::sycl::short16 inputData_1(-22556, 6157, -21677, 9788, 5603, -31443,
                                    15894, -23924, -26108, 13696, 7046, 10553,
                                    -19545, -26208, -837, -27506);
      cl::sycl::short16 inputData_2(-1732, 23566, 31928, 14885, -21755, -26092,
                                    23267, 15064, -5104, -24249, 8292, 30030,
                                    15100, 3476, -26591, 19083);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<276, cl::sycl::uint2>([=]() {
      cl::sycl::uint2 inputData_0(24781, 24363);
      cl::sycl::uint2 inputData_1(59959, 36811);
      cl::sycl::uint2 inputData_2(37477, 25715);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<277, cl::sycl::int2>([=]() {
      cl::sycl::int2 inputData_0(-12668, 4324);
      cl::sycl::int2 inputData_1(-26769, 6878);
      cl::sycl::int2 inputData_2(22925, 30583);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<278, cl::sycl::uint3>([=]() {
      cl::sycl::uint3 inputData_0(48181, 61759, 33122);
      cl::sycl::uint3 inputData_1(1373, 81, 16895);
      cl::sycl::uint3 inputData_2(44662, 13729, 59193);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<279, cl::sycl::int3>([=]() {
      cl::sycl::int3 inputData_0(-20565, 21524, 15378);
      cl::sycl::int3 inputData_1(10921, -14712, -15680);
      cl::sycl::int3 inputData_2(-23669, -30093, 27361);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<280, cl::sycl::uint4>([=]() {
      cl::sycl::uint4 inputData_0(23022, 48955, 13114, 41470);
      cl::sycl::uint4 inputData_1(32877, 32439, 41437, 21030);
      cl::sycl::uint4 inputData_2(41601, 40473, 45245, 23109);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<281, cl::sycl::int4>([=]() {
      cl::sycl::int4 inputData_0(-6594, -26009, -14757, 28088);
      cl::sycl::int4 inputData_1(-7350, -9378, 9029, -17592);
      cl::sycl::int4 inputData_2(-30557, 16856, 2430, 18243);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<282, cl::sycl::uint8>([=]() {
      cl::sycl::uint8 inputData_0(10436, 17166, 48792, 17543, 5214, 21949,
                                  27296, 50695);
      cl::sycl::uint8 inputData_1(17476, 42968, 63959, 53675, 51055, 11745,
                                  24202, 62303);
      cl::sycl::uint8 inputData_2(34326, 7432, 32599, 57124, 39990, 4846, 18688,
                                  45467);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<283, cl::sycl::int8>([=]() {
      cl::sycl::int8 inputData_0(-29975, -32428, 19584, -22369, -21497, -6257,
                                 5442, 20869);
      cl::sycl::int8 inputData_1(27186, -30699, 2824, 23440, -20351, 24624,
                                 10369, 6104);
      cl::sycl::int8 inputData_2(30260, 30385, -10051, -25099, 19135, -5323,
                                 -20576, 24211);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<284, cl::sycl::uint16>([=]() {
      cl::sycl::uint16 inputData_0(14945, 44611, 56500, 52784, 49033, 58251,
                                   59111, 25233, 39313, 43706, 61013, 40281,
                                   32396, 49649, 10600, 44481);
      cl::sycl::uint16 inputData_1(24987, 29146, 62962, 47231, 49080, 16560,
                                   45944, 10372, 37259, 19713, 64790, 12096,
                                   65278, 16403, 26078, 34184);
      cl::sycl::uint16 inputData_2(42262, 20999, 47729, 32144, 7862, 39665,
                                   16843, 47296, 24603, 33675, 32764, 42346,
                                   62081, 13303, 17842, 43000);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<285, cl::sycl::int16>([=]() {
      cl::sycl::int16 inputData_0(-6117, -30393, -31356, 3640, 29630, -14533,
                                  20409, -30847, 11712, -7208, 2197, 3230, 1567,
                                  3662, 15273, 26204);
      cl::sycl::int16 inputData_1(5122, 20066, 16696, 7023, -18990, -4560,
                                  -18403, -21935, 6969, 8446, 31657, -190,
                                  22676, 23600, 31597, -738);
      cl::sycl::int16 inputData_2(29862, -16229, -10411, 1635, 19693, -18084,
                                  -29825, 27680, -15480, 20522, 9840, 6236,
                                  -12667, -24266, -23616, -28584);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<286, cl::sycl::ulong2>([=]() {
      cl::sycl::ulong2 inputData_0(1460139878, 190144672);
      cl::sycl::ulong2 inputData_1(2102967055, 334892530);
      cl::sycl::ulong2 inputData_2(2248600513, 598226964);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<287, cl::sycl::long2>([=]() {
      cl::sycl::long2 inputData_0(-144079726, -1566026414);
      cl::sycl::long2 inputData_1(-1038137310, -848398965);
      cl::sycl::long2 inputData_2(1803752356, 1639576846);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<288, cl::sycl::ulong3>([=]() {
      cl::sycl::ulong3 inputData_0(3920602763, 183647946, 443457276);
      cl::sycl::ulong3 inputData_1(432411571, 2908361923, 1316148021);
      cl::sycl::ulong3 inputData_2(1324348223, 2491593035, 775212120);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<289, cl::sycl::long3>([=]() {
      cl::sycl::long3 inputData_0(-2116945771, 1092815119, -408203988);
      cl::sycl::long3 inputData_1(570667654, -1755229747, 1523812840);
      cl::sycl::long3 inputData_2(1905477839, -292318411, -30921860);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<290, cl::sycl::ulong4>([=]() {
      cl::sycl::ulong4 inputData_0(1618215620, 1843459385, 883418328,
                                   1821412180);
      cl::sycl::ulong4 inputData_1(3311348486, 1149488962, 3476148932,
                                   1997087370);
      cl::sycl::ulong4 inputData_2(1558556137, 3435540825, 1932190826,
                                   1056739813);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<291, cl::sycl::long4>([=]() {
      cl::sycl::long4 inputData_0(-166789249, -1599650698, -687801575,
                                  -1206482833);
      cl::sycl::long4 inputData_1(-1484560861, 157561266, 449008334, 984284266);
      cl::sycl::long4 inputData_2(-1399328414, 691220778, -1446650328,
                                  1868667659);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<292, cl::sycl::ulong8>([=]() {
      cl::sycl::ulong8 inputData_0(2845137718, 4290365187, 482975692,
                                   1383238040, 1407638409, 1431426555,
                                   3273077019, 6457318);
      cl::sycl::ulong8 inputData_1(2827610772, 1298906452, 801237605,
                                   3641666481, 683045295, 4117096487, 340845302,
                                   2249178652);
      cl::sycl::ulong8 inputData_2(1114998236, 3028605763, 218989292,
                                   3196388226, 1679631726, 2081503125,
                                   604785046, 604113153);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<293, cl::sycl::long8>([=]() {
      cl::sycl::long8 inputData_0(1809726979, -1498111820, -1633220774,
                                  -412062733, 1683959299, -939258878, 329475378,
                                  -1719403450);
      cl::sycl::long8 inputData_1(1517791063, 1716457161, 1482504186,
                                  -416780388, -1408323024, -203548010,
                                  -1001458536, -697002796);
      cl::sycl::long8 inputData_2(-1397661548, 1325062220, -1026955428,
                                  1698399813, 1787117285, 1395923735,
                                  -1779523226, 589662000);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<294, cl::sycl::ulong16>([=]() {
      cl::sycl::ulong16 inputData_0(
          3507740540, 3291302859, 2902262071, 1957824859, 3870106901,
          3585277250, 1425988272, 1822500121, 3387506367, 1298302350,
          3474093601, 601408651, 753151567, 971437469, 992812994, 539668093);
      cl::sycl::ulong16 inputData_1(
          3284151086, 1864907299, 1549276600, 965039971, 1384527492, 856121543,
          2649722353, 1350598485, 2132797019, 1475197298, 4184059636,
          3678786383, 3871481872, 4234756757, 3770684104, 3322958825);
      cl::sycl::ulong16 inputData_2(
          78279379, 2939377459, 3937793285, 1166462470, 2149555281, 3045692186,
          473080265, 1888554181, 471079585, 1955582256, 4220587120, 2411502515,
          2417216529, 3166752213, 1613995771, 1803157706);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<295, cl::sycl::long16>([=]() {
      cl::sycl::long16 inputData_0(
          464759385, -696255737, 2141066986, 2112228815, -1525432816, 953177815,
          1480446528, 2041401879, -78011943, -1547024608, -1786502223,
          1502543552, 1513418724, 1138220859, -1677657791, -768214179);
      cl::sycl::long16 inputData_1(
          -808176555, -1315144872, -1878143019, -1941677704, -1160698313,
          -1385819044, 1223198347, 1956099500, 2089345396, 1939282256,
          513688789, 2080822219, 1700722509, -1246391572, 943699521, 413764535);
      cl::sycl::long16 inputData_2(
          -1189839598, -2115904100, -232256105, -2101219155, -1488350920,
          -1283477662, 338906036, -1592078435, 1767072188, 1235933866,
          1075445224, 1645093417, 1189815388, -204420950, 101609485, -37960214);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<296, cl::sycl::ulonglong2>([=]() {
      cl::sycl::ulonglong2 inputData_0(5614478675811419992,
                                       9854525434385303847);
      cl::sycl::ulonglong2 inputData_1(12371267270486135964,
                                       13940587926734191681);
      cl::sycl::ulonglong2 inputData_2(12852369142513949182,
                                       7437117911648214072);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<297, cl::sycl::longlong2>([=]() {
      cl::sycl::longlong2 inputData_0(-5923446207033548997,
                                      -5663823650443488653);
      cl::sycl::longlong2 inputData_1(7964421331180610689, 2976651575190707136);
      cl::sycl::longlong2 inputData_2(4107851929354555921, 2694858929920906941);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<298, cl::sycl::ulonglong3>([=]() {
      cl::sycl::ulonglong3 inputData_0(1842290583553425451, 9007392714570676590,
                                       1760712777551806299);
      cl::sycl::ulonglong3 inputData_1(
          4901488883844701658, 12476919185565240290, 13919672678129354565);
      cl::sycl::ulonglong3 inputData_2(212472506281653757, 6335734343377933257,
                                       17085046512367800245);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<299, cl::sycl::longlong3>([=]() {
      cl::sycl::longlong3 inputData_0(
          -6937687870726127921, -3745836715562911313, -2861659391069830669);
      cl::sycl::longlong3 inputData_1(8820187737924884386, 6574523413830936751,
                                      3565453077147662821);
      cl::sycl::longlong3 inputData_2(
          -4209826600324267960, -8685800760501210112, -3678641332344031055);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<300, cl::sycl::ulonglong4>([=]() {
      cl::sycl::ulonglong4 inputData_0(
          11193720171011133174, 14766945319278874722, 11234279212385130795,
          2863359291736816810);
      cl::sycl::ulonglong4 inputData_1(491809102146078187, 1443568401830883925,
                                       14363250374435982521,
                                       6264155134698291909);
      cl::sycl::ulonglong4 inputData_2(15005551483001519192,
                                       4469699103009901614, 6940034926241187601,
                                       707935065428929861);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<301, cl::sycl::longlong4>([=]() {
      cl::sycl::longlong4 inputData_0(-1000285552930387889, 6377685396598954318,
                                      -484096626999798982,
                                      -3007805425429848298);
      cl::sycl::longlong4 inputData_1(-4038222604758305693,
                                      -6376142475596958612, 6734599963133477207,
                                      5974344536771142292);
      cl::sycl::longlong4 inputData_2(-1628540052371103334, 658410473141424927,
                                      5031107343582966336, 7307706640797356315);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<302, cl::sycl::ulonglong8>([=]() {
      cl::sycl::ulonglong8 inputData_0(
          9105321510911195274, 11968314124318722472, 3469950659910903277,
          9610443922174364690, 1616733509217579671, 14584969867007733458,
          12093120214537576631, 10713330234719415850);
      cl::sycl::ulonglong8 inputData_1(
          10587071144624274498, 1307614163566229840, 6495249432876600640,
          7387510113897893056, 10519696338046628584, 14992395646525053738,
          9922992563165164325, 14323354324466147785);
      cl::sycl::ulonglong8 inputData_2(
          2349447239874952920, 11026569299874013982, 1983246424531897578,
          754325045490668146, 15059158830561026738, 15536735626384710421,
          12367347149145088014, 12500717678609948995);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<303, cl::sycl::longlong8>([=]() {
      cl::sycl::longlong8 inputData_0(2402836616325711015, 6054350427055480842,
                                      1252673284783033200, -8565195991779819795,
                                      5428878726907830964, -4137434593948921913,
                                      7708436646018123515, 5393386920448161085);
      cl::sycl::longlong8 inputData_1(
          -5128669965753303270, -2081732815027667791, 2431925977071379163,
          -5022698121359405245, -1852481132716302257, 743237846026395550,
          7004478016981937430, -8703657761257869639);
      cl::sycl::longlong8 inputData_2(
          857564867949817023, -2463886296102106818, 278098380987799612,
          1648119586954539337, -146164745318050608, -3803196895702650869,
          -2597813231970562776, 7686054937232134613);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<304, cl::sycl::ulonglong16>([=]() {
      cl::sycl::ulonglong16 inputData_0(
          3675682536457897139, 6981222369085848534, 18150364089605510558,
          12341745267884205698, 13585512549640856183, 17860497813513228893,
          17993071245931323972, 10198322091159303983, 5436153355485386579,
          16592931127353477091, 12723414414402530627, 6609408456905934284,
          17172714250674677229, 8502233201350488756, 8893495303161804161,
          11246834165499268931);
      cl::sycl::ulonglong16 inputData_1(
          2600251143849194736, 1223665739736239611, 3274504110064015936,
          4266075606742351511, 1978242495908320316, 17758059977406060685,
          10441485321172949467, 9661819286797353969, 5459366989660810534,
          9421914835001737923, 14340559739188735588, 15999611237583742819,
          2763681370444263878, 4572525378034980144, 5284241348928793859,
          11367744882482195782);
      cl::sycl::ulonglong16 inputData_2(
          7452213431598478381, 13109144864323290217, 6277156198033780172,
          16645149972021173688, 15290346583827795886, 2348513035989650282,
          1786607514987547233, 13075129712994951792, 14071814416571710011,
          8184353276483140112, 13514099265305076166, 13479506808470339677,
          13246544245837966064, 321327854560953202, 5685305725934865075,
          11120535223202504483);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<305, cl::sycl::longlong16>([=]() {
      cl::sycl::longlong16 inputData_0(
          -3426390909553813242, -6578561522352555499, -6916861062150053587,
          8556019796317374209, 2859971957745143857, 4868104826303949855,
          -1237110810725217453, -3935575171683215254, -722388755601599847,
          8144347796640253683, -868746917987578052, 6738136275953933277,
          7575822078222621588, 3095604658147068761, -5051250615807043233,
          671144520336242266);
      cl::sycl::longlong16 inputData_1(
          -4726467498780019405, -7411648147712409296, -1223176566949336247,
          -923962507570061940, 994726611862116035, 4748864243855429115,
          7851847538441473923, 4389319718700022725, 8747234194534770176,
          -881512645895159431, -3485613252825280454, 6751978784943729854,
          3965009521548553385, 1638937954404377221, 3094915670741250406,
          7939694075171199362);
      cl::sycl::longlong16 inputData_2(
          7505604353625648484, 4441343215272496710, -4351497565691434331,
          4505862039811982045, -3563724643379653707, -7254511722334104463,
          -7541593321869190835, -804950893378196870, -2791477801145373217,
          3866780583916244119, 8636177755624033886, -4444562783032649000,
          9174663309623027690, -3090983686012804161, 7071653436359713533,
          -3323148917302465597);
      return cl::sycl::bitselect(inputData_0, inputData_1, inputData_2);

    });

    test_function<306, unsigned char>([=]() {
      unsigned char inputData_0(252);
      unsigned char inputData_1(89);
      signed char inputData_2(-123);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<307, signed char>([=]() {
      signed char inputData_0(-14);
      signed char inputData_1(31);
      signed char inputData_2(-37);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<308, unsigned short>([=]() {
      unsigned short inputData_0(40339);
      unsigned short inputData_1(45521);
      short inputData_2(-582);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<309, short>([=]() {
      short inputData_0(-3572);
      short inputData_1(-16780);
      short inputData_2(30587);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<310, unsigned int>([=]() {
      unsigned int inputData_0(7439);
      unsigned int inputData_1(41996);
      int inputData_2(-7863);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<311, int>([=]() {
      int inputData_0(19139);
      int inputData_1(9605);
      int inputData_2(6466);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<312, unsigned long int>([=]() {
      unsigned long int inputData_0(157618154);
      unsigned long int inputData_1(109438837);
      long int inputData_2(-2092996306);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<313, long int>([=]() {
      long int inputData_0(2007150984);
      long int inputData_1(-1236082666);
      long int inputData_2(-1391261874);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<314, unsigned long long int>([=]() {
      unsigned long long int inputData_0(4876076474467462821);
      unsigned long long int inputData_1(1077167640828336304);
      long long int inputData_2(-3071117159280673748);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<315, long long int>([=]() {
      long long int inputData_0(-2979645808336855771);
      long long int inputData_1(-4954304319926411411);
      long long int inputData_2(367674930569033221);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<316, cl::sycl::uchar2>([=]() {
      cl::sycl::uchar2 inputData_0(113, 181);
      cl::sycl::uchar2 inputData_1(173, 97);
      cl::sycl::schar2 inputData_2(-118, 120);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<317, cl::sycl::schar2>([=]() {
      cl::sycl::schar2 inputData_0(-77, 61);
      cl::sycl::schar2 inputData_1(15, -121);
      cl::sycl::schar2 inputData_2(59, 120);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<318, cl::sycl::uchar3>([=]() {
      cl::sycl::uchar3 inputData_0(38, 183, 146);
      cl::sycl::uchar3 inputData_1(138, 110, 114);
      cl::sycl::schar3 inputData_2(-84, -36, 44);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<319, cl::sycl::schar3>([=]() {
      cl::sycl::schar3 inputData_0(89, 60, 32);
      cl::sycl::schar3 inputData_1(-70, -127, 91);
      cl::sycl::schar3 inputData_2(68, 114, -94);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<320, cl::sycl::uchar4>([=]() {
      cl::sycl::uchar4 inputData_0(249, 84, 144, 4);
      cl::sycl::uchar4 inputData_1(203, 239, 242, 39);
      cl::sycl::schar4 inputData_2(-93, 12, -24, -124);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<321, cl::sycl::schar4>([=]() {
      cl::sycl::schar4 inputData_0(-55, 71, 67, -68);
      cl::sycl::schar4 inputData_1(107, -34, 69, -41);
      cl::sycl::schar4 inputData_2(124, 45, -34, -112);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<322, cl::sycl::uchar8>([=]() {
      cl::sycl::uchar8 inputData_0(48, 242, 147, 204, 54, 134, 153, 240);
      cl::sycl::uchar8 inputData_1(129, 236, 81, 149, 160, 24, 42, 65);
      cl::sycl::schar8 inputData_2(-61, -81, -21, -104, -90, 91, -83, -14);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<323, cl::sycl::schar8>([=]() {
      cl::sycl::schar8 inputData_0(-108, 75, -60, 41, 81, -23, -76, -88);
      cl::sycl::schar8 inputData_1(-11, 24, 40, 22, -66, -128, -94, -16);
      cl::sycl::schar8 inputData_2(120, 17, 122, 87, 126, -54, 48, -18);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<324, cl::sycl::uchar16>([=]() {
      cl::sycl::uchar16 inputData_0(222, 157, 103, 241, 215, 126, 108, 103, 103,
                                    79, 42, 212, 31, 159, 40, 222);
      cl::sycl::uchar16 inputData_1(58, 32, 4, 231, 201, 80, 221, 130, 56, 145,
                                    7, 178, 69, 95, 220, 159);
      cl::sycl::schar16 inputData_2(117, -70, -127, 90, 27, -94, -64, 114, 19,
                                    13, 69, 102, -29, 83, 94, 116);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<325, cl::sycl::schar16>([=]() {
      cl::sycl::schar16 inputData_0(-6, 100, 82, -8, 55, -94, 62, -27, 83, -40,
                                    76, -54, -1, -27, 127, -67);
      cl::sycl::schar16 inputData_1(87, 78, 114, 49, -1, 113, -119, 104, 61,
                                    -77, -54, 93, -102, -59, -90, -75);
      cl::sycl::schar16 inputData_2(-95, -40, 95, 23, -88, -102, 127, -64, -55,
                                    -81, -57, 113, -33, -23, -43, 24);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<326, cl::sycl::ushort2>([=]() {
      cl::sycl::ushort2 inputData_0(34381, 17082);
      cl::sycl::ushort2 inputData_1(7056, 36353);
      cl::sycl::short2 inputData_2(3760, -29061);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<327, cl::sycl::short2>([=]() {
      cl::sycl::short2 inputData_0(-10389, -4971);
      cl::sycl::short2 inputData_1(-1055, -11953);
      cl::sycl::short2 inputData_2(11102, 19743);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<328, cl::sycl::ushort3>([=]() {
      cl::sycl::ushort3 inputData_0(7819, 27646, 43753);
      cl::sycl::ushort3 inputData_1(38836, 57174, 57047);
      cl::sycl::short3 inputData_2(2981, 25421, 20089);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<329, cl::sycl::short3>([=]() {
      cl::sycl::short3 inputData_0(-22921, -32162, -7441);
      cl::sycl::short3 inputData_1(-8845, -30921, 20343);
      cl::sycl::short3 inputData_2(1071, 31193, 21855);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<330, cl::sycl::ushort4>([=]() {
      cl::sycl::ushort4 inputData_0(52941, 26150, 61828, 46745);
      cl::sycl::ushort4 inputData_1(6883, 36295, 13843, 42104);
      cl::sycl::short4 inputData_2(-6394, -30079, 26180, 27491);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<331, cl::sycl::short4>([=]() {
      cl::sycl::short4 inputData_0(8960, -19149, -5787, 12756);
      cl::sycl::short4 inputData_1(-25176, -19190, -26836, 28443);
      cl::sycl::short4 inputData_2(16706, -23762, -17325, 10306);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<332, cl::sycl::ushort8>([=]() {
      cl::sycl::ushort8 inputData_0(63812, 14115, 55590, 28523, 56195, 747,
                                    34229, 27437);
      cl::sycl::ushort8 inputData_1(51595, 62990, 6667, 49364, 23786, 57774,
                                    56616, 65365);
      cl::sycl::short8 inputData_2(611, -80, -18872, -29416, 26572, -25679,
                                   -4444, -26120);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<333, cl::sycl::short8>([=]() {
      cl::sycl::short8 inputData_0(-30315, -11284, -25264, -14221, 32295,
                                   -17005, 18866, 18935);
      cl::sycl::short8 inputData_1(-28515, -7107, -22992, -11491, -24219,
                                   -23730, 14050, -29217);
      cl::sycl::short8 inputData_2(-1533, 3392, -4961, -16548, 8073, -26392,
                                   -21321, -13412);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<334, cl::sycl::ushort16>([=]() {
      cl::sycl::ushort16 inputData_0(29437, 13229, 42554, 17161, 31391, 52749,
                                     43848, 39498, 54547, 26475, 28181, 25597,
                                     8577, 39096, 32124, 52297);
      cl::sycl::ushort16 inputData_1(39883, 22627, 56313, 18122, 20026, 56976,
                                     64220, 13875, 14728, 37966, 1600, 53415,
                                     27947, 47915, 1657, 52073);
      cl::sycl::short16 inputData_2(3171, -31331, -29606, -6095, 8074, -3078,
                                    9419, 26240, -7735, -5194, 19342, -29907,
                                    -10550, -28674, 10068, 11420);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<335, cl::sycl::short16>([=]() {
      cl::sycl::short16 inputData_0(10748, 7896, 16122, -31856, 8533, 8545,
                                    21906, 16670, -30513, 6033, -5776, -14443,
                                    -25671, -24800, -22578, -17098);
      cl::sycl::short16 inputData_1(-1110, 6864, -15735, -19392, 1445, -785,
                                    -27843, -17793, -6959, 28456, -2738, -16630,
                                    24494, 32649, -22582, 24403);
      cl::sycl::short16 inputData_2(-599, -15970, 20461, 3237, 3435, 12932,
                                    -7177, 13142, 15832, 20237, -29608, -12667,
                                    5439, 10892, 6317, 19853);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<336, cl::sycl::uint2>([=]() {
      cl::sycl::uint2 inputData_0(21763, 41941);
      cl::sycl::uint2 inputData_1(34928, 55207);
      cl::sycl::int2 inputData_2(-13181, 2272);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<337, cl::sycl::int2>([=]() {
      cl::sycl::int2 inputData_0(-17213, 27834);
      cl::sycl::int2 inputData_1(30339, -12864);
      cl::sycl::int2 inputData_2(23769, -4303);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<338, cl::sycl::uint3>([=]() {
      cl::sycl::uint3 inputData_0(19136, 24942, 48710);
      cl::sycl::uint3 inputData_1(64477, 54418, 55257);
      cl::sycl::int3 inputData_2(-12893, -19983, -16876);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<339, cl::sycl::int3>([=]() {
      cl::sycl::int3 inputData_0(29023, -22031, -18820);
      cl::sycl::int3 inputData_1(19245, 24054, -16726);
      cl::sycl::int3 inputData_2(1491, -8877, 23800);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<340, cl::sycl::uint4>([=]() {
      cl::sycl::uint4 inputData_0(38357, 31227, 44541, 64546);
      cl::sycl::uint4 inputData_1(41861, 52962, 53179, 17363);
      cl::sycl::int4 inputData_2(-7193, -19735, 6807, 18790);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<341, cl::sycl::int4>([=]() {
      cl::sycl::int4 inputData_0(-5509, -10936, -4787, 14519);
      cl::sycl::int4 inputData_1(-30106, 27817, 10563, 4755);
      cl::sycl::int4 inputData_2(-31839, -27119, -22874, 7628);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<342, cl::sycl::uint8>([=]() {
      cl::sycl::uint8 inputData_0(36648, 10760, 56321, 21770, 9158, 19128,
                                  49208, 31880);
      cl::sycl::uint8 inputData_1(41802, 17655, 40648, 16150, 868, 46993, 28953,
                                  19764);
      cl::sycl::int8 inputData_2(1710, -7798, 20157, 30826, 9971, -30247, 10082,
                                 -32164);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<343, cl::sycl::int8>([=]() {
      cl::sycl::int8 inputData_0(-6589, -26764, 23821, -31057, 11660, -2480,
                                 598, -15801);
      cl::sycl::int8 inputData_1(-8060, -4398, -3262, -760, 21949, 8576, 23612,
                                 1030);
      cl::sycl::int8 inputData_2(1545, 8437, 13893, 15377, 3875, -6564, -14597,
                                 15564);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<344, cl::sycl::uint16>([=]() {
      cl::sycl::uint16 inputData_0(3891, 62729, 64868, 58770, 58145, 62737,
                                   39384, 54770, 12155, 30082, 1501, 23224,
                                   28505, 11359, 46732, 11559);
      cl::sycl::uint16 inputData_1(10515, 49116, 59638, 60636, 56809, 13024,
                                   39662, 34861, 3355, 15073, 37507, 56285,
                                   49480, 53230, 29806, 62499);
      cl::sycl::int16 inputData_2(7299, -14483, -11075, -8231, -24201, 9385,
                                  2406, 21205, -26387, -22706, -10620, -18553,
                                  -16310, -9698, -9573, 500);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<345, cl::sycl::int16>([=]() {
      cl::sycl::int16 inputData_0(15708, 1292, 25639, -28629, 10242, 10569,
                                  28319, 4962, 13809, -2658, -27913, -26236,
                                  -2908, 8353, -7742, -20670);
      cl::sycl::int16 inputData_1(24108, 30020, -10210, -14522, -23521, 31052,
                                  4367, 3019, -12218, 19697, -20414, 10876,
                                  11144, 28563, 21845, -3638);
      cl::sycl::int16 inputData_2(14639, 461, -23826, 2612, 14376, 28004, 3658,
                                  -15389, 881, 12786, -14124, -24156, -20126,
                                  -13077, -19201, -13476);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<346, cl::sycl::ulong2>([=]() {
      cl::sycl::ulong2 inputData_0(181098157, 1874057969);
      cl::sycl::ulong2 inputData_1(2969346334, 44402806);
      cl::sycl::long2 inputData_2(-1432317259, -1704021461);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<347, cl::sycl::long2>([=]() {
      cl::sycl::long2 inputData_0(-347725497, 1608032472);
      cl::sycl::long2 inputData_1(-993016872, -422717840);
      cl::sycl::long2 inputData_2(-626429742, -1480969385);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<348, cl::sycl::ulong3>([=]() {
      cl::sycl::ulong3 inputData_0(391966730, 923350261, 180198623);
      cl::sycl::ulong3 inputData_1(1689471387, 3263102572, 1195988906);
      cl::sycl::long3 inputData_2(2101022758, 17592662, 1101123655);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<349, cl::sycl::long3>([=]() {
      cl::sycl::long3 inputData_0(292034610, 290966753, 1209404911);
      cl::sycl::long3 inputData_1(1221672103, -1265562360, -1457341413);
      cl::sycl::long3 inputData_2(-1356486450, -1153068863, -1810836321);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<350, cl::sycl::ulong4>([=]() {
      cl::sycl::ulong4 inputData_0(995230572, 1706615213, 2706464536,
                                   118489540);
      cl::sycl::ulong4 inputData_1(2639866473, 3049803897, 3828995209,
                                   2816707521);
      cl::sycl::long4 inputData_2(289331974, -867298545, -505854749, 486289557);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<351, cl::sycl::long4>([=]() {
      cl::sycl::long4 inputData_0(1300534294, -649730206, 712342190,
                                  -585178649);
      cl::sycl::long4 inputData_1(511188498, -1957225287, -1713127445,
                                  682477550);
      cl::sycl::long4 inputData_2(1535145478, -814919927, 1033278915,
                                  1168386240);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<352, cl::sycl::ulong8>([=]() {
      cl::sycl::ulong8 inputData_0(292346011, 2880213683, 1869004443, 861660783,
                                   877881999, 3528419601, 2907864503,
                                   1740918896);
      cl::sycl::ulong8 inputData_1(2030563422, 691502436, 4248532096,
                                   2254250030, 317672187, 1772963732,
                                   3770700630, 1374704510);
      cl::sycl::long8 inputData_2(999173701, -1803549998, 1149742942, 482516330,
                                  -191644237, -1319048746, -46036491,
                                  1268863597);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<353, cl::sycl::long8>([=]() {
      cl::sycl::long8 inputData_0(-1928348699, -1072047105, -894346639,
                                  -882962469, 721943469, 1844695096, 386834848,
                                  670398815);
      cl::sycl::long8 inputData_1(-29753982, -121220983, 949687854, 790690545,
                                  -572646819, -86635757, -1847227372,
                                  -823658074);
      cl::sycl::long8 inputData_2(1084580605, 2066560493, -157331157,
                                  -247085964, -1525871464, 813787407,
                                  1668667563, -1871039320);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<354, cl::sycl::ulong16>([=]() {
      cl::sycl::ulong16 inputData_0(
          2782284242, 898479666, 581738495, 2857079696, 1228607764, 348268261,
          3133712224, 3748041841, 2509966615, 3295711445, 2967523292,
          2310205790, 1957790681, 2385079311, 1193280837, 22283450);
      cl::sycl::ulong16 inputData_1(
          2756760028, 3868162829, 1511655817, 1877837313, 3012934781, 140718465,
          1067979309, 3309004670, 2800000546, 200392766, 1612411407, 2563820454,
          502739560, 4039456768, 746582767, 2602129601);
      cl::sycl::long16 inputData_2(
          -1859674528, -1067443682, -1522306066, 1125501119, 2059237253,
          1683313466, -1215303481, 604183111, 1750141297, 679263840,
          -1883486055, -218763537, -721973628, 627025218, -1882409669,
          682898003);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<355, cl::sycl::long16>([=]() {
      cl::sycl::long16 inputData_0(-1941562532, -1906211180, -1166658146,
                                   1560673388, 1150735480, -1013031047,
                                   -2105973617, -1025422087, 1860346723,
                                   239029924, -1803751669, 966662794, 553695888,
                                   1492484468, 2061632486, 1490005329);
      cl::sycl::long16 inputData_1(
          766894355, 1620828377, 518470617, -1239517711, 63497086, -863639504,
          -1082564091, -1351568533, -1858152829, -922266515, 1617874317,
          36687720, 128322660, -1671276309, 1964067468, -1629922566);
      cl::sycl::long16 inputData_2(
          1997926077, -1347093702, -1936020915, 1893534001, -2014206958,
          -979485219, 684404599, 312249055, 998754686, -792997433, 1084098048,
          -1910019299, -2005154784, -1144824487, 1405869258, -1666195404);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<356, cl::sycl::ulonglong2>([=]() {
      cl::sycl::ulonglong2 inputData_0(5252342526309844311,
                                       1967007845594755943);
      cl::sycl::ulonglong2 inputData_1(4885334461922444764,
                                       13690301365225509079);
      cl::sycl::longlong2 inputData_2(-2392493080550592924,
                                      5319821783262145138);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<357, cl::sycl::longlong2>([=]() {
      cl::sycl::longlong2 inputData_0(-2479042711149492922,
                                      -1684578106997018844);
      cl::sycl::longlong2 inputData_1(-9092682916986857969,
                                      -3035316762518680865);
      cl::sycl::longlong2 inputData_2(-7826587373544122002,
                                      -6239346703377534575);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<358, cl::sycl::ulonglong3>([=]() {
      cl::sycl::ulonglong3 inputData_0(
          11408936886273074474, 10205872575659147481, 3970293178983539208);
      cl::sycl::ulonglong3 inputData_1(
          10417224371883808560, 16852931399078002792, 11212170227681141696);
      cl::sycl::longlong3 inputData_2(
          -3401492404370155425, -2325155916257631181, 8028890750963152617);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<359, cl::sycl::longlong3>([=]() {
      cl::sycl::longlong3 inputData_0(7111683855587720610, 6245186911643482341,
                                      -5656768955098414739);
      cl::sycl::longlong3 inputData_1(2181533910560609294, -6404201247889953512,
                                      -707825770079179258);
      cl::sycl::longlong3 inputData_2(9054837898851010291, -6648744985807634603,
                                      -3372449125790101545);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<360, cl::sycl::ulonglong4>([=]() {
      cl::sycl::ulonglong4 inputData_0(5340739914656297578, 892070710090684546,
                                       304413307106749859, 1209891550593184061);
      cl::sycl::ulonglong4 inputData_1(3777357740691135813, 6568669949096135540,
                                       6355897041024134059,
                                       12175427240409065598);
      cl::sycl::longlong4 inputData_2(2672130944222368893, -2310993718620895406,
                                      -636369275076697966, 487143051565449969);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<361, cl::sycl::longlong4>([=]() {
      cl::sycl::longlong4 inputData_0(-7387946108066523037, 6455880100396127303,
                                      1338447961202028993, 7900304609180076268);
      cl::sycl::longlong4 inputData_1(
          -5949923981932267220, -5935522035796404649, -2058283072170314230,
          5783228525167548621);
      cl::sycl::longlong4 inputData_2(4800305378878319515, 2877063028339005801,
                                      -4292020211935348430,
                                      -8362106686109943368);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<362, cl::sycl::ulonglong8>([=]() {
      cl::sycl::ulonglong8 inputData_0(
          6038951568706148979, 17498226044009191261, 16195613821093794466,
          3351856050619695264, 4727203040922710104, 4873357820507718563,
          7362495640035833330, 12388454922194455698);
      cl::sycl::ulonglong8 inputData_1(
          6769191145649212996, 5842814282276543334, 18164345414568132624,
          12301845808154974529, 2139721294843214799, 11714127566737555471,
          3948678149261874450, 1579239721891290653);
      cl::sycl::longlong8 inputData_2(
          2538003778888336497, -6339491223564192431, 382897535521827961,
          4647551391970707975, -9167087125810747325, -3597491466391265714,
          5051422754014368770, -2765286766381587788);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<363, cl::sycl::longlong8>([=]() {
      cl::sycl::longlong8 inputData_0(6476920382197031327, 1745973364149647997,
                                      -1156267156085979641, 6593120384418657346,
                                      2321691229555625569, -102906573209831261,
                                      2441454804745986496, 642191472361984932);
      cl::sycl::longlong8 inputData_1(
          6991806753193977703, 855879051335909390, 768841023625776553,
          -9103640157063900393, 7368869092356013320, 1694846910002615464,
          6927104264402419862, -4383721416256702959);
      cl::sycl::longlong8 inputData_2(
          -8281847811780465075, -1724687112007494784, -5108477392237633680,
          4415975527771158823, 3999143939814372578, 6764806494813275364,
          3495872423460500686, 3949555688009817440);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<364, cl::sycl::ulonglong16>([=]() {
      cl::sycl::ulonglong16 inputData_0(
          535631140563426299, 5353971341089202945, 6949636667662414869,
          6809860130566970955, 16237172028809520104, 9848499910843195392,
          1403810803548674797, 4749531576854797602, 10314023011333671240,
          13402831421214608250, 14282167861754855926, 14831637345283145202,
          4502387482625820443, 6216401635437306842, 9100287251553859813,
          6718745807805121633);
      cl::sycl::ulonglong16 inputData_1(
          8473177668486805230, 5051539617270926337, 17621031642045735188,
          2808405368162044451, 13097959395841885702, 701000168615466892,
          9537506944785045466, 6194623912545468362, 15425956123077812968,
          13698124244900520875, 17766753583421348786, 6852678065697129259,
          4132845996457600010, 15126462742480736302, 17027907268368392692,
          16251969041466130832);
      cl::sycl::longlong16 inputData_2(
          -789281457594426376, -3388080327593942742, -406006915962195592,
          -364894343403670918, -5682311501014874966, 6411342240304996309,
          -7297228681660562486, 6416198317701919176, -6249189471560928834,
          -7026633213588303449, -8122975308845211433, 2190515698799270971,
          8806019739226450303, 2061981030633946894, -85817270651962637,
          -5539014242687470548);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<365, cl::sycl::longlong16>([=]() {
      cl::sycl::longlong16 inputData_0(
          -1960429194024051221, -4029753774059202194, -8345501965221607612,
          -6462470696954121054, 7569905542727400738, 3056429552866386775,
          1414830246784059626, -5404319387603178812, 3182287640757607316,
          -2703745715799571484, 6423432987865150329, 1873485052256179463,
          -4575180149293982554, 7968404149582321048, 118367609559436430,
          -5740587866941774181);
      cl::sycl::longlong16 inputData_1(
          -8314503286657623585, 5554560095369844018, 4099166727455024176,
          -2000426478042031578, -1136890972163834301, 1622170020156109988,
          3547243254670045742, -5825496060700037524, 8662059648393475351,
          8139661688822044172, 2568054155268562136, 6746100605139016622,
          -7679443014053563254, 93606903613299260, -6799007999699631171,
          2276949573757736815);
      cl::sycl::longlong16 inputData_2(
          6727336258328987855, 821919483711821032, 7810697122384526815,
          6285870239046645732, 750551865938276678, -8697867205347071821,
          4845695509770962049, -453112280787645603, 3918286654118418504,
          8142418414317423536, 7395236284209704556, -6468239847293608682,
          9220260410593186067, 4237274904319204478, 8880913382492960828,
          8145248554344514887);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<366, unsigned char>([=]() {
      unsigned char inputData_0(85);
      unsigned char inputData_1(212);
      unsigned char inputData_2(155);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<367, signed char>([=]() {
      signed char inputData_0(-122);
      signed char inputData_1(84);
      unsigned char inputData_2(249);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<368, unsigned short>([=]() {
      unsigned short inputData_0(48068);
      unsigned short inputData_1(47729);
      unsigned short inputData_2(26486);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<369, short>([=]() {
      short inputData_0(23287);
      short inputData_1(-31069);
      unsigned short inputData_2(23326);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<370, unsigned int>([=]() {
      unsigned int inputData_0(54317);
      unsigned int inputData_1(11184);
      unsigned int inputData_2(12467);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<371, int>([=]() {
      int inputData_0(-29314);
      int inputData_1(-30855);
      unsigned int inputData_2(58122);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<372, unsigned long int>([=]() {
      unsigned long int inputData_0(3308611891);
      unsigned long int inputData_1(2336054950);
      unsigned long int inputData_2(3628268617);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<373, long int>([=]() {
      long int inputData_0(-1495936749);
      long int inputData_1(2083478112);
      unsigned long int inputData_2(3101335756);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<374, unsigned long long int>([=]() {
      unsigned long long int inputData_0(18376238418080801045);
      unsigned long long int inputData_1(353490223435598187);
      unsigned long long int inputData_2(15378207618890199517);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<375, long long int>([=]() {
      long long int inputData_0(-3166257897225411659);
      long long int inputData_1(5043257559525820974);
      unsigned long long int inputData_2(8790930168778239009);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<376, cl::sycl::uchar2>([=]() {
      cl::sycl::uchar2 inputData_0(68, 8);
      cl::sycl::uchar2 inputData_1(93, 76);
      cl::sycl::uchar2 inputData_2(229, 144);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<377, cl::sycl::schar2>([=]() {
      cl::sycl::schar2 inputData_0(-69, -50);
      cl::sycl::schar2 inputData_1(6, 125);
      cl::sycl::uchar2 inputData_2(70, 96);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<378, cl::sycl::uchar3>([=]() {
      cl::sycl::uchar3 inputData_0(28, 19, 50);
      cl::sycl::uchar3 inputData_1(166, 175, 210);
      cl::sycl::uchar3 inputData_2(211, 70, 95);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<379, cl::sycl::schar3>([=]() {
      cl::sycl::schar3 inputData_0(113, 6, 37);
      cl::sycl::schar3 inputData_1(51, 108, -24);
      cl::sycl::uchar3 inputData_2(198, 113, 87);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<380, cl::sycl::uchar4>([=]() {
      cl::sycl::uchar4 inputData_0(188, 157, 79, 150);
      cl::sycl::uchar4 inputData_1(26, 234, 8, 18);
      cl::sycl::uchar4 inputData_2(178, 200, 239, 108);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<381, cl::sycl::schar4>([=]() {
      cl::sycl::schar4 inputData_0(112, 0, -105, -53);
      cl::sycl::schar4 inputData_1(-2, 97, -113, 10);
      cl::sycl::uchar4 inputData_2(140, 221, 94, 121);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<382, cl::sycl::uchar8>([=]() {
      cl::sycl::uchar8 inputData_0(170, 57, 162, 115, 61, 63, 151, 173);
      cl::sycl::uchar8 inputData_1(72, 98, 247, 88, 151, 116, 106, 59);
      cl::sycl::uchar8 inputData_2(95, 212, 116, 255, 244, 66, 33, 180);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<383, cl::sycl::schar8>([=]() {
      cl::sycl::schar8 inputData_0(-119, -78, -54, -68, -11, 115, -4, -46);
      cl::sycl::schar8 inputData_1(-82, 0, -128, -72, 9, 93, -33, -43);
      cl::sycl::uchar8 inputData_2(79, 221, 131, 230, 172, 65, 99, 21);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<384, cl::sycl::uchar16>([=]() {
      cl::sycl::uchar16 inputData_0(14, 237, 149, 210, 58, 143, 138, 32, 208,
                                    240, 235, 44, 149, 240, 253, 127);
      cl::sycl::uchar16 inputData_1(226, 220, 224, 173, 239, 174, 30, 53, 56,
                                    39, 132, 78, 75, 87, 34, 122);
      cl::sycl::uchar16 inputData_2(227, 131, 195, 99, 253, 89, 78, 173, 179,
                                    96, 104, 54, 157, 65, 133, 12);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<385, cl::sycl::schar16>([=]() {
      cl::sycl::schar16 inputData_0(-5, 93, -23, 24, -84, -125, -64, -83, 81,
                                    -59, 119, 53, 88, -102, -24, -44);
      cl::sycl::schar16 inputData_1(-76, 107, 72, -86, 26, 82, -30, 82, 8, 60,
                                    80, -104, 71, -13, 88, -94);
      cl::sycl::uchar16 inputData_2(77, 136, 167, 87, 173, 238, 85, 17, 92, 3,
                                    37, 170, 70, 32, 233, 41);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<386, cl::sycl::ushort2>([=]() {
      cl::sycl::ushort2 inputData_0(22357, 32852);
      cl::sycl::ushort2 inputData_1(56536, 49499);
      cl::sycl::ushort2 inputData_2(38666, 64138);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<387, cl::sycl::short2>([=]() {
      cl::sycl::short2 inputData_0(949, -19995);
      cl::sycl::short2 inputData_1(-741, -8329);
      cl::sycl::ushort2 inputData_2(25865, 32478);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<388, cl::sycl::ushort3>([=]() {
      cl::sycl::ushort3 inputData_0(53834, 35803, 46053);
      cl::sycl::ushort3 inputData_1(61570, 7924, 47695);
      cl::sycl::ushort3 inputData_2(51857, 40705, 47998);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<389, cl::sycl::short3>([=]() {
      cl::sycl::short3 inputData_0(-12054, -26373, -22902);
      cl::sycl::short3 inputData_1(21940, -329, -20979);
      cl::sycl::ushort3 inputData_2(30127, 34491, 31504);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<390, cl::sycl::ushort4>([=]() {
      cl::sycl::ushort4 inputData_0(1703, 12239, 39340, 22316);
      cl::sycl::ushort4 inputData_1(50886, 20294, 51676, 37024);
      cl::sycl::ushort4 inputData_2(21171, 19827, 26540, 4841);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<391, cl::sycl::short4>([=]() {
      cl::sycl::short4 inputData_0(18378, 27412, 9170, 5835);
      cl::sycl::short4 inputData_1(18704, -26597, 16176, 12205);
      cl::sycl::ushort4 inputData_2(10049, 37050, 52610, 15862);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<392, cl::sycl::ushort8>([=]() {
      cl::sycl::ushort8 inputData_0(5187, 6761, 20132, 25170, 54668, 31275,
                                    50050, 15450);
      cl::sycl::ushort8 inputData_1(45824, 63966, 27114, 56498, 3680, 5102,
                                    2561, 56353);
      cl::sycl::ushort8 inputData_2(54527, 48866, 57720, 19805, 50695, 57886,
                                    39374, 52354);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<393, cl::sycl::short8>([=]() {
      cl::sycl::short8 inputData_0(21886, 10156, 19622, 25896, -18693, -19203,
                                   7635, 24994);
      cl::sycl::short8 inputData_1(20769, 20765, -22586, -15468, 28562, 14728,
                                   14681, 22548);
      cl::sycl::ushort8 inputData_2(44502, 19205, 57850, 21694, 62012, 42434,
                                    12769, 56067);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<394, cl::sycl::ushort16>([=]() {
      cl::sycl::ushort16 inputData_0(52088, 57845, 42394, 9495, 4815, 8647,
                                     34658, 31681, 38337, 29177, 46892, 35338,
                                     42618, 56085, 5612, 55669);
      cl::sycl::ushort16 inputData_1(15582, 63560, 29975, 27307, 63808, 36474,
                                     2886, 58488, 18025, 26307, 9469, 65201,
                                     42062, 59266, 40619, 56576);
      cl::sycl::ushort16 inputData_2(57242, 62755, 49808, 62439, 43693, 48516,
                                     60669, 61774, 64129, 265, 38327, 62940,
                                     42538, 12583, 10084, 22034);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<395, cl::sycl::short16>([=]() {
      cl::sycl::short16 inputData_0(1816, -6008, -8573, -26995, 9430, -7327,
                                    711, -29356, 32445, -13500, 13818, 19788,
                                    -21643, -8302, -9654, -21524);
      cl::sycl::short16 inputData_1(10265, 16667, 9853, 15511, -26641, -5560,
                                    -135, 30697, -3030, 2571, 1635, -21081,
                                    -32767, 25631, 28273, -31626);
      cl::sycl::ushort16 inputData_2(10074, 57716, 37402, 5612, 34239, 52673,
                                     17030, 27888, 48347, 25719, 276, 18667,
                                     36200, 45084, 27233, 20810);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<396, cl::sycl::uint2>([=]() {
      cl::sycl::uint2 inputData_0(46184, 63912);
      cl::sycl::uint2 inputData_1(56809, 32343);
      cl::sycl::uint2 inputData_2(8370, 43291);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<397, cl::sycl::int2>([=]() {
      cl::sycl::int2 inputData_0(-18998, -17890);
      cl::sycl::int2 inputData_1(8389, -18142);
      cl::sycl::uint2 inputData_2(46674, 33046);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<398, cl::sycl::uint3>([=]() {
      cl::sycl::uint3 inputData_0(28388, 52671, 30303);
      cl::sycl::uint3 inputData_1(47242, 40601, 48779);
      cl::sycl::uint3 inputData_2(58473, 14007, 44427);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<399, cl::sycl::int3>([=]() {
      cl::sycl::int3 inputData_0(-32477, -18614, 15813);
      cl::sycl::int3 inputData_1(-12616, -18524, 3762);
      cl::sycl::uint3 inputData_2(58168, 35443, 13967);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<400, cl::sycl::uint4>([=]() {
      cl::sycl::uint4 inputData_0(19216, 8497, 21686, 3297);
      cl::sycl::uint4 inputData_1(42861, 34792, 7280, 3073);
      cl::sycl::uint4 inputData_2(9624, 56412, 63106, 19993);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<401, cl::sycl::int4>([=]() {
      cl::sycl::int4 inputData_0(-3860, -20274, 14870, 601);
      cl::sycl::int4 inputData_1(-31961, -17480, -32242, -4090);
      cl::sycl::uint4 inputData_2(57093, 57715, 39502, 62886);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<402, cl::sycl::uint8>([=]() {
      cl::sycl::uint8 inputData_0(59730, 33458, 25778, 20670, 10937, 36095,
                                  31053, 661);
      cl::sycl::uint8 inputData_1(2217, 14238, 17025, 29486, 59131, 60529,
                                  17679, 44468);
      cl::sycl::uint8 inputData_2(36444, 58995, 19825, 29107, 38986, 27491,
                                  27403, 13146);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<403, cl::sycl::int8>([=]() {
      cl::sycl::int8 inputData_0(15842, 20558, 9917, 4115, 24928, 23600, -31470,
                                 11739);
      cl::sycl::int8 inputData_1(-31417, -6089, -19602, -19334, -4217, -29769,
                                 14716, -564);
      cl::sycl::uint8 inputData_2(57194, 17905, 63399, 63110, 25091, 10113,
                                  8196, 4510);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<404, cl::sycl::uint16>([=]() {
      cl::sycl::uint16 inputData_0(2764, 56911, 23863, 30635, 28575, 52554,
                                   26286, 23400, 44475, 25148, 43103, 30023,
                                   37240, 26382, 16602, 61833);
      cl::sycl::uint16 inputData_1(38582, 41686, 18782, 41681, 36218, 38100,
                                   35331, 16874, 42890, 5291, 20381, 19897,
                                   28468, 179, 13412, 55967);
      cl::sycl::uint16 inputData_2(51901, 869, 39762, 33755, 6384, 48609, 23739,
                                   25048, 23975, 39866, 54637, 55209, 20484,
                                   58823, 3843, 40826);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<405, cl::sycl::int16>([=]() {
      cl::sycl::int16 inputData_0(13139, 29867, -5426, -5596, 8368, -10475,
                                  30714, -9973, -23142, -10519, -16085, 32670,
                                  25447, 15337, -14798, -14308);
      cl::sycl::int16 inputData_1(-22816, 5555, -18302, -25916, 26276, -15133,
                                  8258, -24289, -14114, -23696, -14677, -32152,
                                  -9058, 28761, 8252, 29437);
      cl::sycl::uint16 inputData_2(29574, 3901, 10063, 31929, 14936, 6006,
                                   17212, 30684, 27566, 49118, 61640, 35600,
                                   5961, 7694, 26433, 5628);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<406, cl::sycl::ulong2>([=]() {
      cl::sycl::ulong2 inputData_0(2333550994, 3047310971);
      cl::sycl::ulong2 inputData_1(1244341051, 2241133631);
      cl::sycl::ulong2 inputData_2(4205119455, 714994966);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<407, cl::sycl::long2>([=]() {
      cl::sycl::long2 inputData_0(1844265263, -366027361);
      cl::sycl::long2 inputData_1(1633581159, -541599736);
      cl::sycl::ulong2 inputData_2(3943437704, 3784948986);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<408, cl::sycl::ulong3>([=]() {
      cl::sycl::ulong3 inputData_0(3567706968, 1414465221, 1113582182);
      cl::sycl::ulong3 inputData_1(422501673, 3347764844, 1719422442);
      cl::sycl::ulong3 inputData_2(1996857271, 409460853, 297746013);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<409, cl::sycl::long3>([=]() {
      cl::sycl::long3 inputData_0(1467142939, 251923527, -1437688402);
      cl::sycl::long3 inputData_1(-1334005198, -600581576, -1005826367);
      cl::sycl::ulong3 inputData_2(586176953, 32637878, 4102461355);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<410, cl::sycl::ulong4>([=]() {
      cl::sycl::ulong4 inputData_0(1908242665, 2840216600, 2429051785,
                                   3013775828);
      cl::sycl::ulong4 inputData_1(3825697722, 1413181989, 2822023070,
                                   63791474);
      cl::sycl::ulong4 inputData_2(1274427301, 112308462, 1904843676,
                                   3680664099);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<411, cl::sycl::long4>([=]() {
      cl::sycl::long4 inputData_0(1688717357, 733984918, 1485834644,
                                  -116713064);
      cl::sycl::long4 inputData_1(-1794006047, -150459435, 937352685,
                                  -1180232120);
      cl::sycl::ulong4 inputData_2(549305970, 4229227983, 1917197634,
                                   2784542235);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<412, cl::sycl::ulong8>([=]() {
      cl::sycl::ulong8 inputData_0(4261793035, 2998732350, 1778682490,
                                   2320311913, 2131977071, 1937421802,
                                   2438176669, 1327630082);
      cl::sycl::ulong8 inputData_1(492891747, 4135707266, 3881728231,
                                   2953581901, 1895078082, 3823998514,
                                   1998657850, 2761057984);
      cl::sycl::ulong8 inputData_2(3974180497, 765639876, 3288474937,
                                   4010861429, 1766050362, 2409721141,
                                   1421474597, 3999001014);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<413, cl::sycl::long8>([=]() {
      cl::sycl::long8 inputData_0(-706362939, -1102567313, -463654299,
                                  -1731591765, -1463096466, -1731336290,
                                  908256260, -721197256);
      cl::sycl::long8 inputData_1(-1986989733, 819907171, -863434263,
                                  1333832712, 1794796357, 366353077, 1855778333,
                                  1817179814);
      cl::sycl::ulong8 inputData_2(1686837807, 3061153649, 85456392, 2717391608,
                                   4159831660, 2446263971, 1275643563,
                                   2318015377);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<414, cl::sycl::ulong16>([=]() {
      cl::sycl::ulong16 inputData_0(
          1640392905, 1268648194, 1928498422, 968082569, 2035482848, 1024281401,
          77478627, 95435306, 1819766788, 1282076287, 168406051, 3507085849,
          421321083, 1323687406, 3532020745, 3157302564);
      cl::sycl::ulong16 inputData_1(
          1840344680, 3293676011, 1114263725, 1811865810, 3485501072,
          1338365298, 3452501327, 1876564615, 3021781012, 769950773, 1952798601,
          575973891, 2419996357, 554898152, 4048273080, 3962673961);
      cl::sycl::ulong16 inputData_2(
          1189710691, 1360642499, 18404582, 4275473858, 2477213334, 1365657168,
          4277511489, 1144469811, 2798735471, 1835865276, 516000613, 1103752693,
          3029343654, 995207315, 1347897758, 772787226);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<415, cl::sycl::long16>([=]() {
      cl::sycl::long16 inputData_0(
          -1986867491, -1922417856, -1308069412, -1740631689, 803666017,
          1651750429, -20800734, -830804154, 19046539, 927231420, 565655565,
          -1460977445, 1172083271, 1713185974, 10675965, 834852785);
      cl::sycl::long16 inputData_1(
          350613437, -1434204914, -495388346, 1993029850, -1917014960,
          1135309459, 509814125, 295341910, -30831970, -1503030291, 471205850,
          1907580784, 33664926, 1498537308, 792516162, -1407253934);
      cl::sycl::ulong16 inputData_2(
          3526832634, 727160426, 4048785779, 688343731, 1767228764, 3549436506,
          1548948788, 2647418509, 2847997421, 1246879302, 1739857773,
          2512938221, 2150333814, 4028008679, 3109191598, 368471584);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<416, cl::sycl::ulonglong2>([=]() {
      cl::sycl::ulonglong2 inputData_0(9711287217147440757,
                                       18139286302892505057);
      cl::sycl::ulonglong2 inputData_1(16534629511586493462,
                                       9779744685413007092);
      cl::sycl::ulonglong2 inputData_2(15261085565452472947,
                                       16390976489580085941);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<417, cl::sycl::longlong2>([=]() {
      cl::sycl::longlong2 inputData_0(-5718436927091366274,
                                      -1167405202472583248);
      cl::sycl::longlong2 inputData_1(-8620924391894742563,
                                      -7669994914505239103);
      cl::sycl::ulonglong2 inputData_2(15817190920895345104,
                                       4224223549443284191);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<418, cl::sycl::ulonglong3>([=]() {
      cl::sycl::ulonglong3 inputData_0(9384491502753420658, 2914181991799944930,
                                       6909953592685391497);
      cl::sycl::ulonglong3 inputData_1(
          2087560166100013715, 13122386386595662706, 7431490832764492366);
      cl::sycl::ulonglong3 inputData_2(
          10839433048191909613, 3036387710260142550, 3811915761789035993);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<419, cl::sycl::longlong3>([=]() {
      cl::sycl::longlong3 inputData_0(8109127113937541786, 4830999043169551246,
                                      -4933424571013788276);
      cl::sycl::longlong3 inputData_1(-1650270008948093914, 4086623856306876900,
                                      -2319728851524907510);
      cl::sycl::ulonglong3 inputData_2(
          15952181068815715789, 3671345718453425830, 16576952922748288941);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<420, cl::sycl::ulonglong4>([=]() {
      cl::sycl::ulonglong4 inputData_0(
          15488962151089455650, 12260211789654429126, 5354681696830982711,
          8987935045764682679);
      cl::sycl::ulonglong4 inputData_1(
          17389241588255803177, 3855681394652589968, 17206988653162863604,
          9757701080813919288);
      cl::sycl::ulonglong4 inputData_2(13178053781895034878, 737942102044695844,
                                       15902709513341335749,
                                       10501469812158763397);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<421, cl::sycl::longlong4>([=]() {
      cl::sycl::longlong4 inputData_0(-414873709375639764, -6934042264038545549,
                                      -3416493614181215852,
                                      -8221982352080283759);
      cl::sycl::longlong4 inputData_1(-806084309616419928, -8104659222375904277,
                                      5034984467751655640,
                                      -7094757957412139857);
      cl::sycl::ulonglong4 inputData_2(
          10240287154909841683, 3546743206083849428, 15229548949161900320,
          8086725696709640391);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<422, cl::sycl::ulonglong8>([=]() {
      cl::sycl::ulonglong8 inputData_0(
          3634929323392232596, 1689031306168950562, 11032898962281069753,
          3203031734513967878, 1143318932358275668, 1822370144886656610,
          1756287686964748661, 7238993144350236671);
      cl::sycl::ulonglong8 inputData_1(
          12500709413293257078, 361880111385469616, 7439209439323749403,
          3427935543441145490, 11345497423110218406, 15335469564507099725,
          3989171118378431074, 12186203223067780827);
      cl::sycl::ulonglong8 inputData_2(
          1538253047446854157, 8677350587291163162, 163644991278145814,
          8088435866872393861, 17790350808853996124, 5958676935725329322,
          6939526125018890522, 5853405554573898500);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<423, cl::sycl::longlong8>([=]() {
      cl::sycl::longlong8 inputData_0(
          7550139070781451936, 3753794139578019553, -8351014949277132761,
          7516318730547798982, 930658028373216972, 8489870843052357160,
          1179240635168715263, -4548170082528691888);
      cl::sycl::longlong8 inputData_1(-6157940960807057822, 2255850574331503252,
                                      5733885402328181966, 6879944036748516048,
                                      3027253952784413822, 6362888681788492820,
                                      6207622491867021834, 6751726912924887230);
      cl::sycl::ulonglong8 inputData_2(
          4722988700396543506, 2182518066302094658, 10135873278062321034,
          2833276619641524375, 4878881776058623491, 12549733398608972900,
          4433369182805721454, 8934813756404509354);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<424, cl::sycl::ulonglong16>([=]() {
      cl::sycl::ulonglong16 inputData_0(
          2338051939099705745, 4015700744345169965, 14800776010341218106,
          5733753189557014345, 7918981978478404958, 3917126795683041099,
          14447111137891262837, 9692112995371138593, 1456582842819807533,
          15634469958809125199, 12246949510412759414, 3886010894890097760,
          17419084314404569387, 14419043872984479791, 13706042685455802778,
          9141469115742014266);
      cl::sycl::ulonglong16 inputData_1(
          9202439763011939773, 4700452261104212462, 17686859466233262159,
          9025531664007674979, 747286449310253212, 14647414633711327313,
          15683058166996929849, 14389831947839904832, 6824595264236516030,
          4792766778361686584, 11008424601263982003, 2587433137549260840,
          1684496779298150211, 6475756941814771807, 10783182591059990944,
          10405554179388717882);
      cl::sycl::ulonglong16 inputData_2(
          3576626628876701364, 3104710102153077383, 10108306142986757687,
          8729686557720921773, 3877511640500554229, 6988761473912216897,
          5862791050666643279, 7457577724137507552, 11201415929884370559,
          2001486124049448985, 6920596471117252717, 11803626738487447393,
          8410171585047451772, 17849933499441737548, 8931260648786732142,
          14062459244170249929);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<425, cl::sycl::longlong16>([=]() {
      cl::sycl::longlong16 inputData_0(
          -1059945454645392735, 1046254785884643388, 8136799943723313256,
          5426823128886825883, -4122862654348317781, -405393530970226792,
          -7606469385365242409, -5240911254903360578, 6388076381167383297,
          -1749479720814783400, 246846524758715237, 4882667093052142174,
          -5899735118324587434, -467733455909368613, -3579369905086872245,
          -232418754226997979);
      cl::sycl::longlong16 inputData_1(
          8111153673462738189, 7143435301936778922, 4472317307830269177,
          5412760809986203105, -1706084048692521775, 6925666089900052128,
          3472899352775071316, -5617930634428845861, -1121663167424290295,
          5286748050143238159, 938430790055794551, 4291021817022506195,
          -6552961994608077510, 2659639917616310813, -9110987948646730665,
          -2951236073013881258);
      cl::sycl::ulonglong16 inputData_2(
          9715192375963571752, 6621952419655491235, 17325464836523884134,
          9733492792550246822, 6987349518839958209, 11518170359572445037,
          2926444195303140826, 8434740346783524274, 11359821097672988926,
          13145348583078470371, 9114277308480309351, 4941236120714314232,
          130468282515600059, 18324660966326653667, 3140927507785848087,
          18231296752867989254);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<426, float>([=]() {
      float inputData_0(0.5932261792);
      float inputData_1(0.2424111937);
      int inputData_2(26496);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<427, cl::sycl::float2>([=]() {
      cl::sycl::float2 inputData_0(0.3367310655, 0.7521311615);
      cl::sycl::float2 inputData_1(0.6283858196, 0.3756758013);
      cl::sycl::int2 inputData_2(7026, -8137);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<428, cl::sycl::float3>([=]() {
      cl::sycl::float3 inputData_0(0.5808362883, 0.7262956134, 0.674558019);
      cl::sycl::float3 inputData_1(0.5654880489, 0.6755333708, 0.7428307123);
      cl::sycl::int3 inputData_2(-8020, -19322, 12526);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<429, cl::sycl::float4>([=]() {
      cl::sycl::float4 inputData_0(0.2668044656, 0.626562404, 0.6893982099,
                                   0.3698330373);
      cl::sycl::float4 inputData_1(0.2824916336, 0.6986421898, 0.8801160722,
                                   0.7029880387);
      cl::sycl::int4 inputData_2(24371, -1525, -16410, -25586);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<430, cl::sycl::float8>([=]() {
      cl::sycl::float8 inputData_0(0.2198747917, 0.3170872019, 0.7256868024,
                                   0.2029813455, 0.8988996853, 0.1335282742,
                                   0.555732743, 0.5602647883);
      cl::sycl::float8 inputData_1(0.5012982689, 0.2327858053, 0.471660798,
                                   0.7784800353, 0.3055008022, 0.3950002708,
                                   0.3445859006, 0.4122116025);
      cl::sycl::int8 inputData_2(7355, 1457, 29985, -2022, -12825, -31576,
                                 -13527, 21762);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<431, cl::sycl::float16>([=]() {
      cl::sycl::float16 inputData_0(
          0.8965548629, 0.558058778, 0.4859664797, 0.4089721601, 0.3653937562,
          0.380048374, 0.7472463973, 0.3759875197, 0.5610994843, 0.2141749949,
          0.8041387332, 0.3457217698, 0.4659693965, 0.3010855622, 0.2315336428,
          0.8513385207);
      cl::sycl::float16 inputData_1(
          0.6629788904, 0.8421072518, 0.1325005173, 0.5420365389, 0.5797117721,
          0.8713133733, 0.6731729771, 0.6067660284, 0.3145876825, 0.8165675976,
          0.5730442128, 0.1007841567, 0.6115655203, 0.7882502754, 0.179059855,
          0.253662551);
      cl::sycl::int16 inputData_2(2877, -30025, -7370, 5117, -16656, 22994,
                                  -26272, 19773, -19042, 21899, 10192, 16126,
                                  1281, 21058, -22150, 16634);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<432, float>([=]() {
      float inputData_0(0.1155345337);
      float inputData_1(0.3007874062);
      unsigned int inputData_2(5587);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<433, cl::sycl::float2>([=]() {
      cl::sycl::float2 inputData_0(0.4616625088, 0.3209976302);
      cl::sycl::float2 inputData_1(0.8059909306, 0.1325123991);
      cl::sycl::uint2 inputData_2(49947, 57478);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<434, cl::sycl::float3>([=]() {
      cl::sycl::float3 inputData_0(0.5957536879, 0.257775966, 0.1013830015);
      cl::sycl::float3 inputData_1(0.1688873243, 0.7617178292, 0.6140504622);
      cl::sycl::uint3 inputData_2(59365, 45659, 7085);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<435, cl::sycl::float4>([=]() {
      cl::sycl::float4 inputData_0(0.6138357058, 0.3525723224, 0.6010707091,
                                   0.2435667547);
      cl::sycl::float4 inputData_1(0.7372528209, 0.6637283533, 0.5199533797,
                                   0.5822108274);
      cl::sycl::uint4 inputData_2(28313, 33639, 33360, 21927);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<436, cl::sycl::float8>([=]() {
      cl::sycl::float8 inputData_0(0.31698595, 0.4054710057, 0.1874994669,
                                   0.1001312533, 0.7034611667, 0.7989876871,
                                   0.3757077384, 0.335538027);
      cl::sycl::float8 inputData_1(0.6921080831, 0.80877757, 0.1375644825,
                                   0.2460649966, 0.1790626004, 0.3871074707,
                                   0.765525907, 0.1278037831);
      cl::sycl::uint8 inputData_2(21737, 61624, 37515, 20768, 56851, 49428,
                                  3585, 57885);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<437, cl::sycl::float16>([=]() {
      cl::sycl::float16 inputData_0(
          0.8343769331, 0.4998840951, 0.6025312822, 0.4231378168, 0.37392086,
          0.6047563734, 0.7174629517, 0.458685804, 0.5344567021, 0.2092303301,
          0.3737184824, 0.455307519, 0.8137201155, 0.5888003121, 0.3851965932,
          0.7437406627);
      cl::sycl::float16 inputData_1(
          0.2915209553, 0.5647249199, 0.8555113541, 0.1103642325, 0.5624150647,
          0.7184078623, 0.5350449567, 0.2970558696, 0.7945912366, 0.5859282564,
          0.4333720206, 0.4528373494, 0.8074597703, 0.6394847053, 0.8918185688,
          0.637069825);
      cl::sycl::uint16 inputData_2(7060, 32321, 41701, 1032, 10204, 42280, 3421,
                                   46451, 8799, 28440, 52225, 41781, 13404,
                                   36694, 46895, 13674);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<438, double>([=]() {
      double inputData_0(0.1537709532);
      double inputData_1(0.2233067783);
      int64_t inputData_2(-9138498746682061787);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<439, cl::sycl::double2>([=]() {
      cl::sycl::double2 inputData_0(0.4208359765, 0.6102894006);
      cl::sycl::double2 inputData_1(0.6882903916, 0.5419797409);
      cl::sycl::vec<int64_t, 2> inputData_2(419976962342091453,
                                            5207972736070297835);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<440, cl::sycl::double3>([=]() {
      cl::sycl::double3 inputData_0(0.7139624468, 0.4514744741, 0.6839779846);
      cl::sycl::double3 inputData_1(0.239739868, 0.7891803038, 0.1593484413);
      cl::sycl::vec<int64_t, 3> inputData_2(
          -8657981951604255548, 6404171899310245229, 3453432959698714796);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<441, cl::sycl::double4>([=]() {
      cl::sycl::double4 inputData_0(0.2455672287, 0.8283440745, 0.6819500658,
                                    0.6918778445);
      cl::sycl::double4 inputData_1(0.3492993067, 0.8631127591, 0.4302416954,
                                    0.3137046153);
      cl::sycl::vec<int64_t, 4> inputData_2(
          -7602091990510848570, 2532265661746468441, 765552755340790887,
          4223067105197772286);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<442, cl::sycl::double8>([=]() {
      cl::sycl::double8 inputData_0(0.5548954334, 0.6076354295, 0.3845749844,
                                    0.8403174548, 0.5045324713, 0.295393783,
                                    0.8235369167, 0.283817916);
      cl::sycl::double8 inputData_1(0.2982278783, 0.8669644229, 0.3028616037,
                                    0.449685009, 0.3254715987, 0.1196784928,
                                    0.2082096732, 0.4979555679);
      cl::sycl::vec<int64_t, 8> inputData_2(
          3021073746256946019, 8365668785388396562, 2163160195834592023,
          6346036696457121316, -6430335724984072422, 4935733173647876295,
          6006377085245121765, 2860858764904870936);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<443, cl::sycl::double16>([=]() {
      cl::sycl::double16 inputData_0(
          0.3076092301, 0.5912437016, 0.8196633982, 0.7707769726, 0.4674007672,
          0.3243951031, 0.6498378352, 0.2811708304, 0.6689709509, 0.2651308074,
          0.7681085806, 0.1318599195, 0.7235776313, 0.7941675203, 0.5713489435,
          0.2779783861);
      cl::sycl::double16 inputData_1(
          0.8321976539, 0.7406091763, 0.7804858077, 0.6958705546, 0.5189024186,
          0.6678111558, 0.4944691734, 0.5313599765, 0.2735098066, 0.3855043732,
          0.4358263698, 0.544157251, 0.4422182781, 0.8240883337, 0.7276924355,
          0.7227012054);
      cl::sycl::vec<int64_t, 16> inputData_2(
          739003619022963064, 3205096189812979887, -3864067333681248996,
          6204460817800304690, 4015815434309301074, 5221033023924495355,
          -6344996857031376551, 6270533042100235717, 4811343654829442279,
          -5519266188191876238, -4460563145723451817, 5561679115011974292,
          -5524135164653883811, 2455029549144540090, -1671659355047248059,
          -1207164858762133741);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<444, double>([=]() {
      double inputData_0(0.4041763727);
      double inputData_1(0.4709585597);
      uint64_t inputData_2(15484530182295432397);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<445, cl::sycl::double2>([=]() {
      cl::sycl::double2 inputData_0(0.4540571183, 0.8784004026);
      cl::sycl::double2 inputData_1(0.6868436125, 0.6426777388);
      cl::sycl::vec<uint64_t, 2> inputData_2(9339997539943976553,
                                             17206033884371613203);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<446, cl::sycl::double3>([=]() {
      cl::sycl::double3 inputData_0(0.8137021216, 0.8896555682, 0.6308061246);
      cl::sycl::double3 inputData_1(0.3895143571, 0.2731266169, 0.5385431652);
      cl::sycl::vec<uint64_t, 3> inputData_2(
          13967350455710754075, 541956017163668881, 11763146403342588739);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<447, cl::sycl::double4>([=]() {
      cl::sycl::double4 inputData_0(0.708137913, 0.8407058114, 0.3750237046,
                                    0.6008359015);
      cl::sycl::double4 inputData_1(0.880637044, 0.1493121555, 0.7891606429,
                                    0.4616884787);
      cl::sycl::vec<uint64_t, 4> inputData_2(
          9910022833366301023, 10161210290266049339, 9242104745251367089,
          4066853654651998808);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<448, cl::sycl::double8>([=]() {
      cl::sycl::double8 inputData_0(0.1087327667, 0.8752161737, 0.7009484771,
                                    0.8207555721, 0.4622509175, 0.2892234021,
                                    0.4782519942, 0.8797025482);
      cl::sycl::double8 inputData_1(0.7968972547, 0.6835996945, 0.6430077704,
                                    0.4437665799, 0.558117589, 0.7077969347,
                                    0.2919984296, 0.1535755471);
      cl::sycl::vec<uint64_t, 8> inputData_2(
          6930784365021944444, 12203016513650444831, 4830200393534416674,
          4258499846346518634, 9164856728396586293, 6753452118836302757,
          9135920166168101045, 3732106541625972340);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });

    test_function<449, cl::sycl::double16>([=]() {
      cl::sycl::double16 inputData_0(
          0.8232387817, 0.7730964873, 0.3027032688, 0.5103149002, 0.8632813799,
          0.7830382454, 0.1844278335, 0.4263903199, 0.4016906949, 0.8035143907,
          0.8692976061, 0.717452856, 0.6140817381, 0.3121908954, 0.3117113616,
          0.4567102042);
      cl::sycl::double16 inputData_1(
          0.4269601131, 0.6646238445, 0.4343160704, 0.8088817047, 0.1818874651,
          0.219235682, 0.7112556033, 0.8047709259, 0.524538266, 0.5445612706,
          0.5644890333, 0.7483307344, 0.7736518861, 0.3542480122, 0.4199870221,
          0.6804832691);
      cl::sycl::vec<uint64_t, 16> inputData_2(
          5538879635751704538, 229925236969724759, 4574881327032281050,
          1372940293362904614, 777332635454250081, 3426130982608128238,
          10013045112806407663, 14717781279311161450, 17243517209281939547,
          12701780631040234682, 10810978961678835979, 141599674407354015,
          13162639964748822655, 9055452911015947770, 6780064219423435835,
          10883353381466244258);
      return cl::sycl::select(inputData_0, inputData_1, inputData_2);

    });
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}