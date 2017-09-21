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

  void run(util::logger &log) override { test_function<0, unsigned char>(
[=](){
unsigned char inputData_0(216);
return cl::sycl::abs(inputData_0);

});

test_function<1, unsigned char>(
[=](){
signed char inputData_0(66);
return cl::sycl::abs(inputData_0);

});

test_function<2, unsigned short>(
[=](){
unsigned short inputData_0(27562);
return cl::sycl::abs(inputData_0);

});

test_function<3, unsigned short>(
[=](){
short inputData_0(-15800);
return cl::sycl::abs(inputData_0);

});

test_function<4, unsigned int>(
[=](){
unsigned int inputData_0(33506);
return cl::sycl::abs(inputData_0);

});

test_function<5, unsigned int>(
[=](){
int inputData_0(-6231);
return cl::sycl::abs(inputData_0);

});

test_function<6, unsigned long int>(
[=](){
unsigned long int inputData_0(3366389306);
return cl::sycl::abs(inputData_0);

});

test_function<7, unsigned long int>(
[=](){
long int inputData_0(-844765410);
return cl::sycl::abs(inputData_0);

});

test_function<8, unsigned long long int>(
[=](){
unsigned long long int inputData_0(16751726038647824248);
return cl::sycl::abs(inputData_0);

});

test_function<9, unsigned long long int>(
[=](){
long long int inputData_0(-6654225565765916554);
return cl::sycl::abs(inputData_0);

});

test_function<10, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(35,24);
return cl::sycl::abs(inputData_0);

});

test_function<11, cl::sycl::uchar2>(
[=](){
cl::sycl::schar2 inputData_0(76,124);
return cl::sycl::abs(inputData_0);

});

test_function<12, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(136,180,154);
return cl::sycl::abs(inputData_0);

});

test_function<13, cl::sycl::uchar3>(
[=](){
cl::sycl::schar3 inputData_0(-91,-103,-110);
return cl::sycl::abs(inputData_0);

});

test_function<14, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(217,84,143,90);
return cl::sycl::abs(inputData_0);

});

test_function<15, cl::sycl::uchar4>(
[=](){
cl::sycl::schar4 inputData_0(-48,35,-76,13);
return cl::sycl::abs(inputData_0);

});

test_function<16, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(113,133,15,235,234,23,215,181);
return cl::sycl::abs(inputData_0);

});

test_function<17, cl::sycl::uchar8>(
[=](){
cl::sycl::schar8 inputData_0(72,32,28,83,-43,58,52,-112);
return cl::sycl::abs(inputData_0);

});

test_function<18, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(234,56,205,36,139,23,254,224,255,125,77,74,31,85,236,52);
return cl::sycl::abs(inputData_0);

});

test_function<19, cl::sycl::uchar16>(
[=](){
cl::sycl::schar16 inputData_0(76,12,-55,-105,76,-47,-67,-81,82,-120,123,-62,-111,45,-95,-90);
return cl::sycl::abs(inputData_0);

});

test_function<20, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(2532,5259);
return cl::sycl::abs(inputData_0);

});

test_function<21, cl::sycl::ushort2>(
[=](){
cl::sycl::short2 inputData_0(13062,21584);
return cl::sycl::abs(inputData_0);

});

test_function<22, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(44793,54910,34378);
return cl::sycl::abs(inputData_0);

});

test_function<23, cl::sycl::ushort3>(
[=](){
cl::sycl::short3 inputData_0(1428,-17335,-18665);
return cl::sycl::abs(inputData_0);

});

test_function<24, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(44530,54084,27487,18036);
return cl::sycl::abs(inputData_0);

});

test_function<25, cl::sycl::ushort4>(
[=](){
cl::sycl::short4 inputData_0(-482,9253,13121,31618);
return cl::sycl::abs(inputData_0);

});

test_function<26, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(23420,21254,7559,38474,21972,12476,1062,17762);
return cl::sycl::abs(inputData_0);

});

test_function<27, cl::sycl::ushort8>(
[=](){
cl::sycl::short8 inputData_0(13456,-8385,-21596,-4842,-28693,18554,23286,-18431);
return cl::sycl::abs(inputData_0);

});

test_function<28, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(53550,41563,61375,39463,4849,8155,12354,54400,7844,5998,54646,7605,39682,12752,63615,47062);
return cl::sycl::abs(inputData_0);

});

test_function<29, cl::sycl::ushort16>(
[=](){
cl::sycl::short16 inputData_0(-1361,14886,-28765,11753,2899,7903,22013,-28181,-28052,-13038,-4190,-28768,-2155,6323,13062,-7126);
return cl::sycl::abs(inputData_0);

});

test_function<30, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(17048,59270);
return cl::sycl::abs(inputData_0);

});

test_function<31, cl::sycl::uint2>(
[=](){
cl::sycl::int2 inputData_0(-1952,26365);
return cl::sycl::abs(inputData_0);

});

test_function<32, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(37340,45724,13330);
return cl::sycl::abs(inputData_0);

});

test_function<33, cl::sycl::uint3>(
[=](){
cl::sycl::int3 inputData_0(17520,18916,-22400);
return cl::sycl::abs(inputData_0);

});

test_function<34, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(10613,34699,7681,60386);
return cl::sycl::abs(inputData_0);

});

test_function<35, cl::sycl::uint4>(
[=](){
cl::sycl::int4 inputData_0(10853,-31903,11880,26220);
return cl::sycl::abs(inputData_0);

});

test_function<36, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(57331,60130,42528,25470,43097,10054,45273,30012);
return cl::sycl::abs(inputData_0);

});

test_function<37, cl::sycl::uint8>(
[=](){
cl::sycl::int8 inputData_0(-27586,15664,2904,-23932,17181,-1191,7217,11364);
return cl::sycl::abs(inputData_0);

});

test_function<38, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(38684,58454,55952,8673,20335,49052,54322,5290,38966,45782,10490,14620,29368,46553,44156,57313);
return cl::sycl::abs(inputData_0);

});

test_function<39, cl::sycl::uint16>(
[=](){
cl::sycl::int16 inputData_0(-30701,24358,4421,17837,13697,-21911,-28582,13206,-3516,25227,26741,6814,-32764,-30218,-11407,22133);
return cl::sycl::abs(inputData_0);

});

test_function<40, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(214134169,3533555164);
return cl::sycl::abs(inputData_0);

});

test_function<41, cl::sycl::ulong2>(
[=](){
cl::sycl::long2 inputData_0(2036574201,-364482042);
return cl::sycl::abs(inputData_0);

});

test_function<42, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2356056127,4136215732,4202933666);
return cl::sycl::abs(inputData_0);

});

test_function<43, cl::sycl::ulong3>(
[=](){
cl::sycl::long3 inputData_0(1449329815,-1586896993,-2084221844);
return cl::sycl::abs(inputData_0);

});

test_function<44, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(4078275605,1793084746,14572006,61380726);
return cl::sycl::abs(inputData_0);

});

test_function<45, cl::sycl::ulong4>(
[=](){
cl::sycl::long4 inputData_0(1094159328,2062635034,754891600,481302467);
return cl::sycl::abs(inputData_0);

});

test_function<46, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(818019171,2612884756,852621339,1298867016,2957256799,782633859,2042709979,3963265804);
return cl::sycl::abs(inputData_0);

});

test_function<47, cl::sycl::ulong8>(
[=](){
cl::sycl::long8 inputData_0(548205762,-2053657944,1777605838,1287736665,-1650214917,-1045872764,659565200,1362678100);
return cl::sycl::abs(inputData_0);

});

test_function<48, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(2769999083,494390875,663315336,3656046880,181666817,883628277,1115271505,1351683038,1575796862,2437255613,3648307521,3635088183,4128335062,2609607092,2123799798,2766575068);
return cl::sycl::abs(inputData_0);

});

test_function<49, cl::sycl::ulong16>(
[=](){
cl::sycl::long16 inputData_0(-177535212,-277280437,1595196945,-1381741146,-534384462,-897495744,-1552815580,-981943638,-697884647,-570365322,-1745012950,1200086515,-1994301631,-989465371,-1505746657,358420073);
return cl::sycl::abs(inputData_0);

});

test_function<50, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(2391453237470918543,17224722580768645308);
return cl::sycl::abs(inputData_0);

});

test_function<51, cl::sycl::ulonglong2>(
[=](){
cl::sycl::longlong2 inputData_0(4218046413955047081,-3703635780663906564);
return cl::sycl::abs(inputData_0);

});

test_function<52, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(8876062164656179221,14984818927713528932,8837586495630196759);
return cl::sycl::abs(inputData_0);

});

test_function<53, cl::sycl::ulonglong3>(
[=](){
cl::sycl::longlong3 inputData_0(-41790703095206580,-3654242442532693040,7741892703072918051);
return cl::sycl::abs(inputData_0);

});

test_function<54, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(10412954397149637280,1603423758059408124,1561877852935827171,4077813705490265038);
return cl::sycl::abs(inputData_0);

});

test_function<55, cl::sycl::ulonglong4>(
[=](){
cl::sycl::longlong4 inputData_0(-9078546442355537815,3307548583936611830,-7680057553915725515,-1268748169045012658);
return cl::sycl::abs(inputData_0);

});

test_function<56, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(2121604328392752619,3720485401245954389,7340948290457903085,11824312693564450701,11499978517417089636,3651902665220752110,2791626439830889648,18413227121734963827);
return cl::sycl::abs(inputData_0);

});

test_function<57, cl::sycl::ulonglong8>(
[=](){
cl::sycl::longlong8 inputData_0(-5187720598600905169,7360399133263130811,-3286750619130383154,6314935341336324578,5664172017953595340,5533023947575847293,-7999344092908148745,-8473843420724981631);
return cl::sycl::abs(inputData_0);

});

test_function<58, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(14769301725905091544,12102607460286780479,18272100793247263921,3801963095848298936,14721598824252572194,16330073370021340075,3356734141501636002,10519562691876581016,1210490908368889290,1148471473459482784,17257378073542104431,6161667275904974392,662862261255004151,2599236268562400480,6507491755420074617,11285180583623363310);
return cl::sycl::abs(inputData_0);

});

test_function<59, cl::sycl::ulonglong16>(
[=](){
cl::sycl::longlong16 inputData_0(-758486987975055276,-8850229868848215676,2165655670083957921,-3248262163868583783,-5626743606829171810,2251193521986836658,6440340456292187716,7596061498621704866,9146654488278716691,5204086687315424279,-821429141835144413,2435217606632834381,-8485858738607121864,7398461504990805906,-4986980663551169335,131095898465527538);
return cl::sycl::abs(inputData_0);

});

test_function<60, unsigned char>(
[=](){
unsigned char inputData_0(242);
unsigned char inputData_1(37);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<61, unsigned char>(
[=](){
signed char inputData_0(100);
signed char inputData_1(-20);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<62, unsigned short>(
[=](){
unsigned short inputData_0(5507);
unsigned short inputData_1(6865);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<63, unsigned short>(
[=](){
short inputData_0(-28645);
short inputData_1(-5560);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<64, unsigned int>(
[=](){
unsigned int inputData_0(10233);
unsigned int inputData_1(62557);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<65, unsigned int>(
[=](){
int inputData_0(19093);
int inputData_1(-4517);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<66, unsigned long int>(
[=](){
unsigned long int inputData_0(1791085152);
unsigned long int inputData_1(2132987165);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<67, unsigned long int>(
[=](){
long int inputData_0(1563615806);
long int inputData_1(-753770505);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<68, unsigned long long int>(
[=](){
unsigned long long int inputData_0(1448780409481033919);
unsigned long long int inputData_1(2239028887646444714);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<69, unsigned long long int>(
[=](){
long long int inputData_0(-8681452615460442845);
long long int inputData_1(-5941554947160112166);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<70, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(212,59);
cl::sycl::uchar2 inputData_1(93,152);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<71, cl::sycl::uchar2>(
[=](){
cl::sycl::schar2 inputData_0(-92,-128);
cl::sycl::schar2 inputData_1(40,59);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<72, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(230,191,75);
cl::sycl::uchar3 inputData_1(176,238,59);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<73, cl::sycl::uchar3>(
[=](){
cl::sycl::schar3 inputData_0(-92,-12,-6);
cl::sycl::schar3 inputData_1(53,-62,-121);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<74, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(53,85,245,75);
cl::sycl::uchar4 inputData_1(224,141,83,151);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<75, cl::sycl::uchar4>(
[=](){
cl::sycl::schar4 inputData_0(-108,8,-50,-32);
cl::sycl::schar4 inputData_1(-91,-96,-71,2);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<76, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(60,47,95,169,220,153,100,179);
cl::sycl::uchar8 inputData_1(33,76,106,237,151,76,90,63);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<77, cl::sycl::uchar8>(
[=](){
cl::sycl::schar8 inputData_0(33,35,7,-32,-126,118,102,-16);
cl::sycl::schar8 inputData_1(-33,115,-105,-81,-101,-100,27,-89);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<78, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(179,237,47,107,44,245,87,133,90,161,22,193,75,212,114,118);
cl::sycl::uchar16 inputData_1(55,29,250,77,155,108,192,23,127,245,139,103,161,5,69,171);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<79, cl::sycl::uchar16>(
[=](){
cl::sycl::schar16 inputData_0(-128,-27,99,53,-15,63,-38,95,44,-78,-107,-110,-61,8,-98,92);
cl::sycl::schar16 inputData_1(100,67,-111,92,-56,-94,5,-75,-102,116,11,61,103,-57,-15,17);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<80, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(9031,8080);
cl::sycl::ushort2 inputData_1(7898,26294);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<81, cl::sycl::ushort2>(
[=](){
cl::sycl::short2 inputData_0(-2101,3906);
cl::sycl::short2 inputData_1(-13176,8658);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<82, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(48650,14287,32037);
cl::sycl::ushort3 inputData_1(32870,32290,62811);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<83, cl::sycl::ushort3>(
[=](){
cl::sycl::short3 inputData_0(-3672,-23409,-294);
cl::sycl::short3 inputData_1(25384,-18631,-9477);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<84, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(25622,65069,55458,64153);
cl::sycl::ushort4 inputData_1(45016,5332,62213,60595);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<85, cl::sycl::ushort4>(
[=](){
cl::sycl::short4 inputData_0(-6874,-9092,-25156,-32523);
cl::sycl::short4 inputData_1(-15052,9199,-13597,26261);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<86, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(9219,37542,12530,28443,46946,25174,21672,42467);
cl::sycl::ushort8 inputData_1(45054,9700,61160,9666,20714,13688,12243,22883);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<87, cl::sycl::ushort8>(
[=](){
cl::sycl::short8 inputData_0(31685,-4766,-528,15059,19371,-3980,-19382,13762);
cl::sycl::short8 inputData_1(26595,-30588,8727,24314,-8939,15935,11632,-17494);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<88, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(19498,5677,56940,49249,50556,57865,26963,3535,45665,43622,42500,35860,48130,28220,29792,16695);
cl::sycl::ushort16 inputData_1(31172,22098,2767,3444,22924,18992,467,4173,28032,14559,39887,36553,14481,12638,39866,39738);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<89, cl::sycl::ushort16>(
[=](){
cl::sycl::short16 inputData_0(-27157,-11861,2352,25786,-16024,1441,-20308,-27518,24244,-10110,22104,-16301,15009,-13021,1107,-7559);
cl::sycl::short16 inputData_1(-1175,24377,-17042,-12727,3406,-32165,-308,-4051,27338,-5760,-421,-3926,-27158,-16965,21382,-22689);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<90, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(57950,13989);
cl::sycl::uint2 inputData_1(40157,53609);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<91, cl::sycl::uint2>(
[=](){
cl::sycl::int2 inputData_0(3867,25336);
cl::sycl::int2 inputData_1(-6916,30075);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<92, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(16366,14421,55080);
cl::sycl::uint3 inputData_1(18284,23075,65186);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<93, cl::sycl::uint3>(
[=](){
cl::sycl::int3 inputData_0(-25737,26189,7214);
cl::sycl::int3 inputData_1(18942,13832,18183);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<94, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(33687,30282,41573,17861);
cl::sycl::uint4 inputData_1(1085,40296,46655,11309);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<95, cl::sycl::uint4>(
[=](){
cl::sycl::int4 inputData_0(-5593,-18477,24889,32581);
cl::sycl::int4 inputData_1(-32346,2328,-4701,-29507);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<96, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(25211,17857,63645,37006,15044,46959,35824,18492);
cl::sycl::uint8 inputData_1(48488,59085,4245,20153,21450,24457,31474,38145);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<97, cl::sycl::uint8>(
[=](){
cl::sycl::int8 inputData_0(-23771,-31753,317,-8718,8835,20446,24696,-22594);
cl::sycl::int8 inputData_1(-21184,32193,-23883,-19235,27205,-214,17514,30756);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<98, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(15421,8659,15315,25205,39878,8686,32667,60153,40371,1704,39067,32060,20230,62824,36406,10684);
cl::sycl::uint16 inputData_1(57611,53034,48324,35784,46152,5162,9007,26335,12521,20736,19118,61851,13674,20654,48960,16358);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<99, cl::sycl::uint16>(
[=](){
cl::sycl::int16 inputData_0(23787,11038,10470,-17963,-10213,-22130,-31656,4822,-28955,8548,-9607,-583,-28750,-16944,-31953,29198);
cl::sycl::int16 inputData_1(-11360,21034,-10163,10746,-23865,27655,-3345,-23459,-12324,9769,14946,18696,-5988,-32137,25619,2208);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<100, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(3455732951,4211763861);
cl::sycl::ulong2 inputData_1(3027671438,3260249633);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<101, cl::sycl::ulong2>(
[=](){
cl::sycl::long2 inputData_0(278006653,-1620940147);
cl::sycl::long2 inputData_1(1949714828,-1411688672);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<102, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2149864228,2670799579,2843167600);
cl::sycl::ulong3 inputData_1(3607995735,3004765956,342083050);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<103, cl::sycl::ulong3>(
[=](){
cl::sycl::long3 inputData_0(-1127942278,1409847290,-1390393337);
cl::sycl::long3 inputData_1(1915375552,1960929305,1702211205);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<104, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(3190352089,3600658741,2411232357,3082896610);
cl::sycl::ulong4 inputData_1(1834712413,3713707219,445209715,2699523214);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<105, cl::sycl::ulong4>(
[=](){
cl::sycl::long4 inputData_0(902404932,493651123,-1975475352,-686119687);
cl::sycl::long4 inputData_1(1022270775,-530352736,540152989,1672798081);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<106, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(2131785282,1539166697,2884695204,1967596743,2179429579,697007703,3234016684,1450177512);
cl::sycl::ulong8 inputData_1(3450250031,4056256332,63496457,2277801660,1237928625,2013985471,150016024,2284034162);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<107, cl::sycl::ulong8>(
[=](){
cl::sycl::long8 inputData_0(223748200,1864401749,-1982982738,-450691241,-1632020692,-658895889,-1927785696,-974744137);
cl::sycl::long8 inputData_1(2128930465,-1056905611,774784434,870628664,1843069545,2125753141,1125630918,1127846896);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<108, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(2217324202,1658165567,3583710367,1076463846,498196434,4216472842,3456587262,4050588829,1042159186,2894223383,2288272789,3760862699,698151293,3728239356,657369642,3634472279);
cl::sycl::ulong16 inputData_1(3522217312,2512266810,220138098,670731166,1479964907,1250654164,1260247530,2125403769,1733699155,1851155071,730296768,3385894884,2442130241,1897898244,1466070589,40019390);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<109, cl::sycl::ulong16>(
[=](){
cl::sycl::long16 inputData_0(951871023,1760532990,705272410,699377784,-1041104809,481950118,-1845779087,1872940557,-950445282,1417959365,126828773,-1875668203,598675909,342252867,1896888688,567490425);
cl::sycl::long16 inputData_1(186499827,-495215112,-280729592,-810040407,-2088921267,1196282463,1373319926,-942999682,-1038562767,117103377,-781984225,-677471674,882927993,-293361735,-1538855685,-2122024936);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<110, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(10390777132318078694,16598644128534888829);
cl::sycl::ulonglong2 inputData_1(11638198480535248526,12914364876720733765);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<111, cl::sycl::ulonglong2>(
[=](){
cl::sycl::longlong2 inputData_0(-4160245213245673745,3781437979955860562);
cl::sycl::longlong2 inputData_1(-5750229798068698147,7813289869803017641);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<112, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(17347148906220082176,14268174517854179918,6046142301193147405);
cl::sycl::ulonglong3 inputData_1(555880709153672741,14066137294537942401,14062919789331473758);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<113, cl::sycl::ulonglong3>(
[=](){
cl::sycl::longlong3 inputData_0(-4088724930489580723,-8775092539860521286,-8456241111421836255);
cl::sycl::longlong3 inputData_1(6732116064997480239,-9140486961857043050,4008250935294199359);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<114, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(9972714885087273335,468353367211443932,16699235914035273397,5074378583052702229);
cl::sycl::ulonglong4 inputData_1(9435362333885754834,16797987557635453780,13347589067288214911,14997192293025084297);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<115, cl::sycl::ulonglong4>(
[=](){
cl::sycl::longlong4 inputData_0(1103052339032010494,-4324222002682111580,-4409426727892553041,3849332311397877936);
cl::sycl::longlong4 inputData_1(-7901787385806075056,5368347550963042760,410676197763804173,-6735344230186898826);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<116, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(10254915819075167300,6334294714703286479,14293737261494061195,10519702062901601958,15375542343377058090,2510758373317459331,419211583981273997,2307685914112377779);
cl::sycl::ulonglong8 inputData_1(814230185590057757,14697746330154854011,17097524511220116932,996242271928156584,13954366064675598242,14826857206910949345,3332704631293194322,14801746532887926358);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<117, cl::sycl::ulonglong8>(
[=](){
cl::sycl::longlong8 inputData_0(8287847947794918753,-2626643904507125293,-8740313220548074552,-5956695739073144445,-8971637587397460562,-432897400970155913,-7745927236513281104,8395984646525544585);
cl::sycl::longlong8 inputData_1(8635776214493857540,5621516313682960366,-8544459890759338721,-2994071560699083518,-6261594822631871704,7763671677455687967,3317407981693233183,-8408605963020785199);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<118, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(11726074370601495932,6494769771311649278,16202582374731683501,2545611678360062822,14218001649672028024,12671807785712840477,14786651111253964212,2746329469201956076,15201694010204834104,68193925253636064,14357566137717182916,14514804342926636132,487762091816969745,17017799113841457376,9412496533841042306,7385849424914252588);
cl::sycl::ulonglong16 inputData_1(4531358346474685209,18409252403323550811,9713796270078948833,14380984410236395135,4372194084512778926,11270343765150228103,14721817322134800825,11549546884222714052,7251565371637579792,11145232795552418475,3128729890230117158,7583385832576227457,6371888966530482848,994376247559640411,17659763286956845291,5461022253646718636);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<119, cl::sycl::ulonglong16>(
[=](){
cl::sycl::longlong16 inputData_0(-8458523588043371475,-1268889587780232844,-1650117868028288235,4203568155874468958,-6748507112272496204,2086856491418815797,-8647357350552854969,-1874236962863162213,3335606132075940176,994588964287997166,-6401606744886027599,-5012912027038200187,3397753989424338734,6236689174872179510,6700193008496600199,-6302889203374811197);
cl::sycl::longlong16 inputData_1(7603145021478615614,-3592644384422372181,-2831092554128573063,8571011061898045323,-6499780063726305767,315806406440203342,-7172371498354501559,-2290076067581669414,-8558900361797511516,7238716801474544041,4103725007253717916,7798025344670047232,-6124293515392510566,-5384913680877992397,3388096160060677106,7741346773982991447);
return cl::sycl::abs_diff(inputData_0,inputData_1);

});

test_function<120, unsigned char>(
[=](){
unsigned char inputData_0(43);
unsigned char inputData_1(112);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<121, signed char>(
[=](){
signed char inputData_0(-115);
signed char inputData_1(-32);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<122, unsigned short>(
[=](){
unsigned short inputData_0(52582);
unsigned short inputData_1(42208);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<123, short>(
[=](){
short inputData_0(-6214);
short inputData_1(-17492);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<124, unsigned int>(
[=](){
unsigned int inputData_0(57240);
unsigned int inputData_1(2562);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<125, int>(
[=](){
int inputData_0(7864);
int inputData_1(-15160);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<126, unsigned long int>(
[=](){
unsigned long int inputData_0(2179696879);
unsigned long int inputData_1(2365634484);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<127, long int>(
[=](){
long int inputData_0(-454876465);
long int inputData_1(-1007998971);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<128, unsigned long long int>(
[=](){
unsigned long long int inputData_0(9979699346122358719);
unsigned long long int inputData_1(8428703835312456394);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<129, long long int>(
[=](){
long long int inputData_0(-7959919231219204139);
long long int inputData_1(1035607987960716113);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<130, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(67,58);
cl::sycl::uchar2 inputData_1(28,36);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<131, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(79,-93);
cl::sycl::schar2 inputData_1(93,82);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<132, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(35,143,1);
cl::sycl::uchar3 inputData_1(220,142,193);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<133, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-3,48,110);
cl::sycl::schar3 inputData_1(15,95,-41);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<134, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(24,1,58,214);
cl::sycl::uchar4 inputData_1(79,57,126,242);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<135, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(2,-41,-109,18);
cl::sycl::schar4 inputData_1(-71,-34,-31,66);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<136, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(59,239,190,123,225,91,98,33);
cl::sycl::uchar8 inputData_1(199,102,128,120,167,95,234,110);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<137, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-37,-26,68,126,93,-6,-54,-14);
cl::sycl::schar8 inputData_1(-40,-66,-81,116,-1,-100,-30,-29);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<138, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(131,250,250,144,158,172,128,124,80,175,23,81,228,58,247,251);
cl::sycl::uchar16 inputData_1(147,10,23,51,83,28,204,93,59,11,97,1,29,154,239,51);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<139, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(61,-78,-128,101,88,-111,-83,-69,109,-31,78,-17,-31,67,29,-60);
cl::sycl::schar16 inputData_1(21,52,83,45,36,24,-105,113,54,-59,49,30,40,-31,18,40);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<140, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(13215,33293);
cl::sycl::ushort2 inputData_1(7886,6916);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<141, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(26939,-24606);
cl::sycl::short2 inputData_1(25773,-1980);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<142, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(29812,22270,27277);
cl::sycl::ushort3 inputData_1(24722,37026,21993);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<143, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(21101,-17462,-16485);
cl::sycl::short3 inputData_1(-1275,28513,-31201);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<144, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(47409,393,26532,50083);
cl::sycl::ushort4 inputData_1(29234,28146,16594,31135);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<145, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-17809,-14188,10046,6517);
cl::sycl::short4 inputData_1(28150,30727,1466,-27030);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<146, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(19654,33934,44116,62009,10165,2404,57018,52767);
cl::sycl::ushort8 inputData_1(50184,30710,44419,26966,12586,25617,51579,52550);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<147, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(30220,25406,11933,1370,14675,-20762,27727,13931);
cl::sycl::short8 inputData_1(6192,-4323,8751,7712,26139,4635,-18785,-3842);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<148, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(15923,59306,55281,36426,12870,2853,8792,29046,44184,14679,44860,56488,49626,27887,42318,64773);
cl::sycl::ushort16 inputData_1(58026,22160,44921,10696,36527,23365,28714,28763,43465,55443,30708,9606,49424,49259,62511,25824);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<149, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-2368,2660,25698,13383,-31374,-19181,23192,5601,24504,-5807,-18975,-32497,32509,-23831,9369,-675);
cl::sycl::short16 inputData_1(-7855,2438,-27638,30804,-476,-31766,-5286,16855,-12316,16057,17521,-17097,30669,-30941,23829,828);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<150, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(10051,16934);
cl::sycl::uint2 inputData_1(38896,18248);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<151, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(22178,-18381);
cl::sycl::int2 inputData_1(-7599,446);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<152, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(22267,54011,17293);
cl::sycl::uint3 inputData_1(5831,10144,41087);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<153, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(4165,-28620,32312);
cl::sycl::int3 inputData_1(-1348,-11834,15018);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<154, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(1591,28458,43543,63054);
cl::sycl::uint4 inputData_1(49914,58009,7792,28165);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<155, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(-30685,-14943,-7583,-10236);
cl::sycl::int4 inputData_1(-8275,19862,-20347,21266);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<156, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(35515,22200,36191,10579,32470,1438,56555,21730);
cl::sycl::uint8 inputData_1(22547,65218,40203,27371,51816,4434,37388,34124);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<157, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(23673,5649,-966,1325,18474,-10006,3787,13591);
cl::sycl::int8 inputData_1(32476,12693,30269,-6617,7129,16075,-9935,-15128);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<158, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(63755,22841,65529,55854,14160,54278,64462,18141,43545,50435,5457,53695,20208,46293,62268,2300);
cl::sycl::uint16 inputData_1(40089,19163,7514,46652,64162,33600,22698,29431,27172,34858,26815,5267,64187,65320,11412,15796);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<159, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-4132,13024,-30714,21987,9072,-15120,24305,10565,-11999,3135,31407,-29594,13661,22899,12603,-23592);
cl::sycl::int16 inputData_1(6366,18740,-5335,5402,-16157,-12272,20222,-689,-3355,-24715,-8227,1357,-17629,20180,-7622,-17139);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<160, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(1324127651,3541043909);
cl::sycl::ulong2 inputData_1(3883266660,4124447854);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<161, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-2082225572,1090466987);
cl::sycl::long2 inputData_1(109453101,-1612502192);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<162, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(1058853647,1209852904,1736097453);
cl::sycl::ulong3 inputData_1(2021732470,4023477501,250633061);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<163, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(898375324,1520874096,-612892186);
cl::sycl::long3 inputData_1(-1077103932,-1196970919,-855389223);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<164, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(624050117,2369438911,1075457681,117044453);
cl::sycl::ulong4 inputData_1(999152987,3524588055,1792591453,3794759318);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<165, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(1905314629,-1102310536,257579716,1636669914);
cl::sycl::long4 inputData_1(349697685,-1425927913,-1082532560,2094332696);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<166, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(1285856793,3726755931,3414551990,3186800032,3100718382,3392946268,3639588294,267861805);
cl::sycl::ulong8 inputData_1(720737708,2171231916,912635564,2290153968,2118202019,544479227,369200337,50048380);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<167, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(1396019548,-1796405856,1982407991,2078041991,1055258188,-213102187,-962981217,-376013985);
cl::sycl::long8 inputData_1(-664460363,-445408984,971503490,1685887224,-1470102766,-1105221449,-1245983301,-1952724104);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<168, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3668763553,2195911737,287912182,1916651751,1935358614,3341295388,3270177141,577625811,2692410268,2189085961,57946739,634520555,2864092171,1576363834,4138997257,2155001396);
cl::sycl::ulong16 inputData_1(2956153704,573883392,2059215449,3153027709,3579776539,857307379,1704701386,2033783017,1891385028,2042006426,1270881790,3473429142,3921639914,1498946715,2739588147,1635119468);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<169, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(338237228,839833372,6512361,749824282,1104432562,1474443527,-1336551347,-1218117592,61724485,41476580,1321670931,74661016,1718212273,1192295101,27126202,1401560020);
cl::sycl::long16 inputData_1(-103697859,-679828216,-285944151,-188085155,646529808,-1923470631,985732262,2011044456,-176877194,-1852108896,-1283094061,-1704182725,-1046453057,1262321406,-2142976358,1604511014);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<170, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(16526953890399529314,12198608531049168334);
cl::sycl::ulonglong2 inputData_1(14974629799128039005,304190700782305251);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<171, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(4396577698859929641,-2679978504330305598);
cl::sycl::longlong2 inputData_1(3679961806870609170,-6047015462126960179);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<172, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(9808011925563911411,13620499546731731608,16986717937138135627);
cl::sycl::ulonglong3 inputData_1(5889461196677055272,3298537824502812307,9327291191188177219);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<173, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(6663150088683632042,-9172665125081230532,7947451021091121763);
cl::sycl::longlong3 inputData_1(-7651374683935116938,2197477831083790121,-8089587425981705874);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<174, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(2548874532399439797,5061556933626756694,6098218510375975693,11915899133192568866);
cl::sycl::ulonglong4 inputData_1(16565955815049669513,14135466598253586910,15471753475314362445,17776066572661253186);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<175, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-4386123456723341152,5567522174886505360,-1695220004905800080,8812064445580209722);
cl::sycl::longlong4 inputData_1(-8724428156174602432,5899240147589096258,3742902301900646432,-5070745218438222443);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<176, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(16648161758647391980,12974506595540806722,10078294683037783495,15436778794830095478,10005510225456334145,11884156620039161998,17626725718671713490,5309126677102060806);
cl::sycl::ulonglong8 inputData_1(3903958147381982797,15905825953656893306,16163107059329810075,5250994918250571049,9358084239121134715,18356839726355595352,6128366466548468463,15136223477436466392);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<177, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-6541939528487516695,4631257581288911742,-2342426152996631650,-902717968939644246,-273339777440617400,487844631416130465,-3402905680649915804,5101366137824342794);
cl::sycl::longlong8 inputData_1(344050551803474659,-5177547607889861629,211042099542250029,6996928898972763049,-6347631611816246301,3030011222082339580,228942993199780136,4154269119641794843);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<178, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(18026033119682517158,8767620357188406293,17977117790105315280,6616776943073577786,6747021192269836416,4966995886761963354,17468593734288080344,16364833253605716111,17008224115240513881,13628060451863419216,12826820527219870946,13045352984303423472,11281490756338211714,10669434587005065195,7220819004086411946,7027883300452486410);
cl::sycl::ulonglong16 inputData_1(3899087770380477988,5940848543127217170,10152938595878904904,8243702881203665561,12761990640811555377,9976333723244384176,3879155147839708130,799247084062442854,8454767150339842950,9250557093561761786,8973252617875105312,14073181917715558913,8886073372677988522,13395827100518484289,372567664949992613,13468253966582177719);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<179, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-4100279880776325889,7539391720998540172,8322781044539572112,-4466339874564037015,-3624986281729467627,2143705999784218665,-7776453594687606282,-8906170166732466944,2923634659318216214,4451145813760646820,6760380579590843294,5975125977023894564,-1508990644371248973,61258526013730872,-800114164106923829,7032208179170024283);
cl::sycl::longlong16 inputData_1(4316504648466340696,3827561036167196728,-2770687542566931051,8652171496735677993,-8015541207633976423,7660417150971873291,-1428607934230101226,4409967846432044022,2878859056664404698,2152738464488893550,-1552086582505965100,-5470443303952613423,-7605124871975851294,-1805671417932824134,3200300144402124455,2830600308476468868);
return cl::sycl::add_sat(inputData_0,inputData_1);

});

test_function<180, unsigned char>(
[=](){
unsigned char inputData_0(8);
unsigned char inputData_1(59);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<181, signed char>(
[=](){
signed char inputData_0(-104);
signed char inputData_1(-32);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<182, unsigned short>(
[=](){
unsigned short inputData_0(4565);
unsigned short inputData_1(54300);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<183, short>(
[=](){
short inputData_0(-18834);
short inputData_1(-19746);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<184, unsigned int>(
[=](){
unsigned int inputData_0(32229);
unsigned int inputData_1(12547);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<185, int>(
[=](){
int inputData_0(23545);
int inputData_1(865);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<186, unsigned long int>(
[=](){
unsigned long int inputData_0(2593278022);
unsigned long int inputData_1(2363249554);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<187, long int>(
[=](){
long int inputData_0(85017588);
long int inputData_1(519766189);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<188, unsigned long long int>(
[=](){
unsigned long long int inputData_0(16841009959454530865);
unsigned long long int inputData_1(14765704615684152210);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<189, long long int>(
[=](){
long long int inputData_0(-8690692653066033616);
long long int inputData_1(-1096834522272122263);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<190, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(132,54);
cl::sycl::uchar2 inputData_1(1,60);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<191, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-8,26);
cl::sycl::schar2 inputData_1(85,-54);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<192, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(84,184,170);
cl::sycl::uchar3 inputData_1(182,224,22);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<193, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-97,-1,28);
cl::sycl::schar3 inputData_1(39,-70,-94);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<194, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(235,61,4,72);
cl::sycl::uchar4 inputData_1(132,162,189,37);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<195, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(2,-47,57,-36);
cl::sycl::schar4 inputData_1(79,-79,126,5);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<196, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(108,185,96,9,112,73,169,134);
cl::sycl::uchar8 inputData_1(212,125,39,38,146,67,54,241);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<197, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-93,106,9,111,86,-52,-8,-107);
cl::sycl::schar8 inputData_1(-35,109,-103,-66,-118,92,46,22);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<198, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(119,66,149,179,203,41,160,173,148,186,132,244,166,160,3,36);
cl::sycl::uchar16 inputData_1(153,196,36,163,39,195,210,158,17,71,69,119,199,148,253,181);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<199, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-92,122,-113,-43,35,-29,-123,-53,-67,70,23,-92,95,-74,-47,95);
cl::sycl::schar16 inputData_1(67,-21,4,122,53,55,39,124,108,-52,-15,34,-68,37,103,-50);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<200, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(24112,29487);
cl::sycl::ushort2 inputData_1(25317,42160);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<201, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-29363,18136);
cl::sycl::short2 inputData_1(-14068,7995);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<202, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(27778,40061,37286);
cl::sycl::ushort3 inputData_1(33918,10506,493);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<203, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-25758,-7544,-15919);
cl::sycl::short3 inputData_1(-997,-1912,1008);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<204, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(8697,32598,62291,11271);
cl::sycl::ushort4 inputData_1(1018,22209,46432,56410);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<205, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-25608,-30724,-12443,7997);
cl::sycl::short4 inputData_1(27551,-11114,18308,-24412);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<206, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(42006,16381,49884,59777,28928,45037,23202,55644);
cl::sycl::ushort8 inputData_1(26878,38278,64648,36545,29676,6295,62234,34399);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<207, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(13159,10130,-17207,8983,-26436,-29014,22342,6604);
cl::sycl::short8 inputData_1(-13018,1758,3794,11650,-32760,-23302,-26663,16586);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<208, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(29644,13002,24542,43873,30226,35773,61442,26235,6766,7021,47495,20478,7539,50998,58249,6708);
cl::sycl::ushort16 inputData_1(40403,48528,16111,54831,44827,29135,10860,16897,54360,10997,46178,37417,36747,1070,8041,20303);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<209, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(8541,-7771,-15959,-7613,-2418,6198,3915,-8650,-4885,20069,5833,30833,6715,-14203,859,-1776);
cl::sycl::short16 inputData_1(23129,15564,24942,14275,-16170,-14889,-22033,24840,26219,-11563,-31247,-1687,18899,13300,11508,-31391);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<210, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(6679,47789);
cl::sycl::uint2 inputData_1(53644,11941);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<211, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(20734,29575);
cl::sycl::int2 inputData_1(6660,3389);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<212, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(2157,27128,30605);
cl::sycl::uint3 inputData_1(62485,28892,780);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<213, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(4397,-28304,32244);
cl::sycl::int3 inputData_1(10465,14358,12680);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<214, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(61669,26538,18280,5158);
cl::sycl::uint4 inputData_1(1495,31272,48739,48446);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<215, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(-32580,7645,21743,24048);
cl::sycl::int4 inputData_1(17670,-5317,13366,13207);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<216, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(4207,2548,22684,42162,24967,42270,49906,50553);
cl::sycl::uint8 inputData_1(18558,63683,36293,41150,41629,44128,10093,44209);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<217, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-4487,30716,14034,31063,32182,21889,5648,6597);
cl::sycl::int8 inputData_1(-1987,-8542,-5326,27098,9588,-21632,-30327,-3988);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<218, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(28892,4313,14860,21718,24689,40953,10223,53856,32578,4530,6526,61811,2096,42531,11730,42873);
cl::sycl::uint16 inputData_1(64722,60175,28656,28279,18968,28862,62778,2481,31380,58702,6986,7649,53396,18535,52293,19779);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<219, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-30651,20960,-11022,-2536,-28846,9884,21558,-18367,28390,8755,3158,-19278,24646,-4985,-29973,27903);
cl::sycl::int16 inputData_1(-8596,-13462,-28967,-30600,-29944,24573,24109,-1984,-27444,-23642,29328,3978,19371,-27733,-28488,18161);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<220, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(497395046,3524885891);
cl::sycl::ulong2 inputData_1(4013429295,1858995266);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<221, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-1653000369,934680347);
cl::sycl::long2 inputData_1(-233276129,6061367);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<222, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(3813873509,2318665283,593550235);
cl::sycl::ulong3 inputData_1(1599924087,3649268237,2196643204);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<223, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1800678188,28103998,-1996751337);
cl::sycl::long3 inputData_1(1629110385,-1081838493,984312409);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<224, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(4271958892,3849857355,2270444893,1392201681);
cl::sycl::ulong4 inputData_1(3001098386,2347677740,3903275785,1040134962);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<225, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(830883522,-1492563917,-407625989,1191383742);
cl::sycl::long4 inputData_1(-582612553,-929326562,-631779856,-542651648);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<226, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(34261504,614708776,3180750741,2975498598,2802390524,587781349,497494906,2102609975);
cl::sycl::ulong8 inputData_1(2078352661,540304936,4148177832,3510466724,3746143233,739559069,2819802315,3492799767);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<227, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-757998042,2084548500,-1739140752,1259409135,-1526383212,-1068101046,-1333507980,-1650385549);
cl::sycl::long8 inputData_1(733978230,-1103981877,1285281676,1157256547,1964610696,1179792317,2104666380,246212427);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<228, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(137398196,1425965479,1692560481,4150294456,1808738929,1160376600,3434884261,3950435304,200466903,4213242536,3103890498,4276722506,2785533069,33435256,2913382952,973799252);
cl::sycl::ulong16 inputData_1(4245421375,4003270170,3696928472,3043674117,1088317323,3979890782,3533104752,2722701795,734236143,855760571,2434899795,2548202850,1752395145,3494035926,2913608129,1108340128);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<229, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-2018445031,458649093,-732730203,1674332684,-1141582670,208878458,-1297385699,2004585601,613920657,1086984731,1678715142,-198076292,200748565,-216298254,511436734,-709726754);
cl::sycl::long16 inputData_1(1382823147,-578885917,1165581145,-2075763027,-585468731,-86353285,1722503559,955055765,1004181691,586883641,1721201655,966022876,564679292,1807562796,1134709811,-2083967307);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<230, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(17173648249724983033,13805084026624090662);
cl::sycl::ulonglong2 inputData_1(13974952335958567671,3960067404699552991);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<231, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(4590667492108938863,9150982493808582577);
cl::sycl::longlong2 inputData_1(-5969925616045973965,-1453973716872065769);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<232, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(13434757987392136193,8561084104778502648,11164903233765563556);
cl::sycl::ulonglong3 inputData_1(430059479896934059,9262533303620198988,14043591656762522841);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<233, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-8682575095140194079,5714325845598862559,618468847950874749);
cl::sycl::longlong3 inputData_1(1441870525316150716,1976855429344771736,-3544324006665334837);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<234, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(4043614681864225798,11600737179889962657,17229086678755937909,15606371623992444148);
cl::sycl::ulonglong4 inputData_1(15593404978740728598,4866301620481992488,10978849092973681898,16495318348285152417);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<235, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-6647114684770018466,8003877951023644485,-8406706793781201596,-537342235313128702);
cl::sycl::longlong4 inputData_1(434293643597839817,2377294888907061257,-7730096002385885144,7747057397815037250);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<236, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(788761962294084473,16266310996131225335,12491319364077650743,12126305378014253880,10701310681854639930,14213643497461042252,13916977253536423975,1684717630197685819);
cl::sycl::ulonglong8 inputData_1(16359807999137033080,14055701190564898001,14222133528741095738,8577424980647130374,8298696385012051704,2993342935056814576,14818426759447378713,2847231726433459066);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<237, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(4227468742937456195,-6213315837560047681,9203606235350110126,-2234987666233253940,-3177611046512425080,-9195637977614447984,2759407866489727620,-5782082621228921850);
cl::sycl::longlong8 inputData_1(1065792203954023356,8459422744212764106,-7272288628862821566,4505010938071250152,-666626635451259535,6178393477528094550,7232550209803645620,-7384781497986521011);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<238, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(16640797124466339732,14094537260075512977,7446613425104017472,16517943718826695044,1580378041953113504,14136038191752696753,13569651237027905840,1674873267173915989,18405587527213334206,14921118401507801200,6440538898175673472,1202365860600946633,14105321547785993204,1836043231790905906,4144815238265042061,2471497472548419230);
cl::sycl::ulonglong16 inputData_1(12180724091157774174,14271999497668861068,16912693401628596291,5751375596851732958,2023271498034900525,6232976420206372980,14870452880029622720,1013863610927085894,14128839763395640131,13502336616409146930,14476968954764297079,16527919576982204351,10495504358334299406,7653630420143173808,3710877465338683210,11734454403006155696);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<239, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(7065622404610085714,7132451685551252574,436409468047966715,-3097542789676511573,-8960095031153342045,4661980478311002624,-418312560978448114,9093196938022196737,115173593218321014,6544130668007660692,-277502678877130538,8058666892106400500,8644296205382983668,292993319022956425,1683830327431002049,1161109143129109623);
cl::sycl::longlong16 inputData_1(-2599120274105954325,-7456132816782155386,-6906204785922232333,1978118326588712058,1900882742487897058,-6634526309178454287,9078338622737741078,7451846181073878784,4007987304831441970,-3234420566406009884,852965680201741677,-7556837088253141875,-8074026917106099554,5576980404600827085,-3168955320043544157,1992711209317671418);
return cl::sycl::hadd(inputData_0,inputData_1);

});

test_function<240, unsigned char>(
[=](){
unsigned char inputData_0(76);
unsigned char inputData_1(98);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<241, signed char>(
[=](){
signed char inputData_0(54);
signed char inputData_1(-40);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<242, unsigned short>(
[=](){
unsigned short inputData_0(11076);
unsigned short inputData_1(5884);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<243, short>(
[=](){
short inputData_0(-14675);
short inputData_1(-2740);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<244, unsigned int>(
[=](){
unsigned int inputData_0(51894);
unsigned int inputData_1(40656);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<245, int>(
[=](){
int inputData_0(-22303);
int inputData_1(-23512);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<246, unsigned long int>(
[=](){
unsigned long int inputData_0(2395770015);
unsigned long int inputData_1(2436788362);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<247, long int>(
[=](){
long int inputData_0(773601367);
long int inputData_1(-196834117);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<248, unsigned long long int>(
[=](){
unsigned long long int inputData_0(12788825114002438629);
unsigned long long int inputData_1(5013545436530507848);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<249, long long int>(
[=](){
long long int inputData_0(-8685800760501210112);
long long int inputData_1(-3678641332344031055);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<250, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(76,39);
cl::sycl::uchar2 inputData_1(204,1);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<251, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-45,-89);
cl::sycl::schar2 inputData_1(78,-52);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<252, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(45,147,207);
cl::sycl::uchar3 inputData_1(20,221,131);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<253, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-13,80,8);
cl::sycl::schar3 inputData_1(71,104,82);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<254, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(86,177,143,251);
cl::sycl::uchar4 inputData_1(43,21,62,51);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<255, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(125,55,88,113);
cl::sycl::schar4 inputData_1(-121,91,-44,-38);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<256, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(64,232,191,242,79,114,233,57);
cl::sycl::uchar8 inputData_1(193,190,3,209,186,253,204,207);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<257, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-74,120,-39,77,-83,55,-77,120);
cl::sycl::schar8 inputData_1(-98,60,-89,79,53,31,-86,-25);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<258, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(170,232,19,47,99,20,173,63,17,82,115,142,155,25,43,229);
cl::sycl::uchar16 inputData_1(196,150,12,245,198,54,15,18,166,152,43,193,51,74,189,145);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<259, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-90,88,78,-128,124,-43,-49,11,-71,-51,18,-38,-44,-57,68,82);
cl::sycl::schar16 inputData_1(-52,-90,-119,-125,36,104,-26,-18,82,17,-53,-127,48,-12,-111,70);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<260, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(43833,52271);
cl::sycl::ushort2 inputData_1(8346,25143);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<261, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-21283,-18358);
cl::sycl::short2 inputData_1(3145,-30815);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<262, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(2679,26566,18545);
cl::sycl::ushort3 inputData_1(55197,1723,20895);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<263, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-5246,-19596,910);
cl::sycl::short3 inputData_1(-9420,-16643,-30655);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<264, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(32272,63845,3125,54277);
cl::sycl::ushort4 inputData_1(22170,47706,37218,13917);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<265, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-19659,-13523,-11565,15263);
cl::sycl::short4 inputData_1(19287,-20939,14455,-14700);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<266, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(61947,52277,7242,37280,51947,60153,10636,24903);
cl::sycl::ushort8 inputData_1(14547,49920,8436,41407,57911,45137,23766,3015);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<267, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(7656,-17845,852,-903,4703,-27641,-25491,24884);
cl::sycl::short8 inputData_1(-12889,-11853,-24170,30364,32491,7803,-20992,-5498);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<268, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(865,51681,9025,33756,46877,43716,17698,27312,61336,3930,16329,17449,22100,6883,55007,32248);
cl::sycl::ushort16 inputData_1(54470,3809,17358,7061,18656,60074,15243,64178,13058,44895,7974,64483,43568,63584,22697,25603);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<269, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(24726,11078,2271,12118,-16126,13710,732,15497,26136,-25915,31156,11409,-9639,-23108,-24779,15182);
cl::sycl::short16 inputData_1(19544,16961,-21367,26181,-16575,27291,27676,5260,-17004,-1468,-20137,-4819,28241,28075,32512,19111);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<270, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(9204,53550);
cl::sycl::uint2 inputData_1(23018,8498);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<271, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(28439,-2562);
cl::sycl::int2 inputData_1(17552,-11870);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<272, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(39956,31727,18573);
cl::sycl::uint3 inputData_1(26913,52427,30381);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<273, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(-21135,17228,24175);
cl::sycl::int3 inputData_1(-17612,8242,-4894);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<274, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(41087,38238,19353,35467);
cl::sycl::uint4 inputData_1(2040,3213,64630,27027);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<275, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(-8666,25371,-16686,10056);
cl::sycl::int4 inputData_1(-25665,-25123,6897,-13373);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<276, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(31811,16135,50947,61040,39245,12248,22390,49860);
cl::sycl::uint8 inputData_1(56842,21937,6740,20611,58427,6697,48825,36826);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<277, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(27755,15480,-1474,7551,27676,-428,-10804,7618);
cl::sycl::int8 inputData_1(-19086,8774,-6293,800,-10482,32185,26481,-15706);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<278, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(51297,45932,4351,54322,52361,2506,6347,61240,18416,49993,16660,1172,48011,41820,53453,47888);
cl::sycl::uint16 inputData_1(64041,59627,47061,13195,25808,41187,35287,26247,39508,35723,57948,20595,28447,6076,8194,43018);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<279, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(3691,30397,16634,-9080,32522,-27338,-18153,-4396,9453,12854,-13982,-28511,-14744,28934,-18756,14763);
cl::sycl::int16 inputData_1(-3087,-22424,-32596,4709,11781,-29513,-8533,-29210,-18060,-17946,-16033,4312,-8223,10135,-12223,10125);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<280, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(3000714639,4240807805);
cl::sycl::ulong2 inputData_1(2648809776,3554967098);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<281, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(1183702881,-1100466492);
cl::sycl::long2 inputData_1(1035337478,1988579175);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<282, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(421824836,453307041,2579005265);
cl::sycl::ulong3 inputData_1(393157586,3596984119,3451767155);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<283, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-284792978,646045070,-448943328);
cl::sycl::long3 inputData_1(1013646643,1127263572,103370580);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<284, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(3788203678,1369142646,2242516326,544315507);
cl::sycl::ulong4 inputData_1(1286297297,1262294485,1725997235,2240712184);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<285, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-1931480562,303441458,-1084993635,-461039256);
cl::sycl::long4 inputData_1(243524757,-826573865,-1627913702,-798860494);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<286, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(3026527775,1098177662,2989852464,1147634030,1918461123,3934864562,3505688617,1385950006);
cl::sycl::ulong8 inputData_1(327432470,1871408315,1464101687,29303801,465713338,4145647135,3070659370,3576537746);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<287, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-1694552271,-1303942606,743600501,-2125945891,1848604064,-2069829751,1068314382,-1674307184);
cl::sycl::long8 inputData_1(712732029,-2061708817,1034080775,-969181509,1760755888,-853056199,1811688104,-1285041922);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<288, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3196586412,2591638990,3898375183,1317739338,63976824,3687839378,3331996371,4042573939,3326084873,458411018,2978437442,659569932,1960066408,3495390667,1195506451,3937367221);
cl::sycl::ulong16 inputData_1(4069315991,3290914567,3047788645,2367278686,1951372654,1112653231,403665442,3991435733,4283626405,2374180897,3331006312,1546915486,3702383530,1546075937,1648807445,1465497893);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<289, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(2023674089,1646497615,-1957148975,18417737,-776420163,-987271400,516673428,-773730910,2084604730,-648351526,-2062391591,-219721958,525443653,-606070084,496174153,835818888);
cl::sycl::long16 inputData_1(-38124608,-234091259,-1099654997,2004611527,-1659935525,604768929,-515248338,1254327724,629525787,423782842,-1989865494,-2038044811,-2092996306,2007150984,-1236082666,-1391261874);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<290, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(4876076474467462821,1077167640828336304);
cl::sycl::ulonglong2 inputData_1(6152254877574102060,6243726228517920037);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<291, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-4954304319926411411,367674930569033221);
cl::sycl::longlong2 inputData_1(-3344521199602150062,1132969457862016251);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<292, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(17938471886663976954,10552432307379182945,9110647362613253086);
cl::sycl::ulonglong3 inputData_1(8240251180303784565,707681147920187075,12460153240251656160);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<293, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(2328931457784129776,-2009501848332892428,1818407256578008218);
cl::sycl::longlong3 inputData_1(1176746920400417153,8023643233624430188,1463813296212513612);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<294, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(2574232582415427505,3443699025029222059,345473573094270311,14097805748641853767);
cl::sycl::ulonglong4 inputData_1(6776293176061072299,12642154660071491673,1193772795553740682,10609774161120270752);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<295, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-6450585256436126163,449542792965510903,-7165632177508080043,2225886367413581452);
cl::sycl::longlong4 inputData_1(-6170516452805680370,-590962781295818201,-5812145645077145344,-2575407490058889258);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<296, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(2761748489056888739,14592226291769643551,3684294845150090993,14118355821457161258,2517620593949819486,18372604868161308821,7948242216563438614,18128467207692781603);
cl::sycl::ulonglong8 inputData_1(7471775104979890109,3247575376865149383,15317624685883277616,7193474164671074434,2319524617734281372,526967317995835463,2534821634445486228,6895469376418281296);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<297, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(7568901037126551358,-2085447339623892742,991100802628661004,-6732313266850319835,1831027669099485236,6048870722322618698,-4422224842671197727,-811767918913909641);
cl::sycl::longlong8 inputData_1(-1659393416582698477,-1377277978120463027,-4237325397586240285,4686991295583004761,-6818257243890445272,-8116755531760120357,1379362412148046577,-5797373539554721102);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<298, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(6845488625651825879,4806515048239704955,11540361347448277479,1043670373433672908,13440411361803448880,14780781187477365527,7844506927681497539,12315380359521400581,744000052706884087,14955039485873861267,14949693016126860692,10927689924657139919,2049766891607129586,10216202761325329369,7423697626007103233,16961638043461117261);
cl::sycl::ulonglong16 inputData_1(6538511077175478224,11910655488361270019,3821914222487231834,5041984786515517355,3157183742817384752,15647341096256040523,5174801231548593645,14522726616439230951,9400571439449284659,15936201280921823349,10443541120931101470,4896097593591962272,1843196846010657164,1197213782499123058,8857741283649488826,5988998840631188564);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<299, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-4832786055182336090,-596338693940234170,954997602796917618,6806256270889185772,-3091038703533367999,-3774863134384125973,-909089762537811569,2754682149550882725,5272701672752752814,5624348597933573773,-2752860978853208483,-2018195699516296577,-181071171718284104,1214361625352062213,-3586483226125464098,-5317683087940360543);
cl::sycl::longlong16 inputData_1(-8772988854172008481,-8578586984255621456,4596042871456218777,-3675284781881694731,2651453157726787084,1029394366390611333,-1461917475948098352,4361771010283693971,-2969472176642190741,2405368604994844473,-8588470426736737925,1493440533149883755,-4065308480524433385,-5890473239985416581,-4955636425917327495,-1842130988469459530);
return cl::sycl::rhadd(inputData_0,inputData_1);

});

test_function<300, unsigned char>(
[=](){
unsigned char inputData_0(13);
unsigned char inputData_1(135);
unsigned char inputData_2(217);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<301, signed char>(
[=](){
signed char inputData_0(95);
signed char inputData_1(-95);
signed char inputData_2(48);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<302, unsigned short>(
[=](){
unsigned short inputData_0(43210);
unsigned short inputData_1(4348);
unsigned short inputData_2(14960);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<303, short>(
[=](){
short inputData_0(17379);
short inputData_1(-25007);
short inputData_2(-4326);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<304, unsigned int>(
[=](){
unsigned int inputData_0(44079);
unsigned int inputData_1(27548);
unsigned int inputData_2(39882);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<305, int>(
[=](){
int inputData_0(-20188);
int inputData_1(-30348);
int inputData_2(32450);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<306, unsigned long int>(
[=](){
unsigned long int inputData_0(2852636396);
unsigned long int inputData_1(666456726);
unsigned long int inputData_2(2118369463);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<307, long int>(
[=](){
long int inputData_0(312989881);
long int inputData_1(-1551168799);
long int inputData_2(1618475849);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<308, unsigned long long int>(
[=](){
unsigned long long int inputData_0(10754361548339533735);
unsigned long long int inputData_1(14811759013427178439);
unsigned long long int inputData_2(9831454927694726328);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<309, long long int>(
[=](){
long long int inputData_0(-8039856540277583640);
long long int inputData_1(-1211164806587860157);
long long int inputData_2(5312796262037214555);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<310, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(82,81);
unsigned char inputData_1(215);
unsigned char inputData_2(53);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<311, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-104,-52);
signed char inputData_1(-102);
signed char inputData_2(-24);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<312, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(136,80,107);
unsigned char inputData_1(118);
unsigned char inputData_2(217);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<313, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-58,112,-28);
signed char inputData_1(-112);
signed char inputData_2(115);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<314, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(112,24,42,30);
unsigned char inputData_1(134);
unsigned char inputData_2(68);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<315, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-126,-28,-122,-87);
signed char inputData_1(-113);
signed char inputData_2(-20);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<316, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(229,205,97,68,121,243,151,232);
unsigned char inputData_1(9);
unsigned char inputData_2(246);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<317, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(124,-68,102,57,-20,1,111,108);
signed char inputData_1(-65);
signed char inputData_2(-109);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<318, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(37,49,34,229,7,70,186,5,119,157,197,42,42,220,133,90);
unsigned char inputData_1(250);
unsigned char inputData_2(80);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<319, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(115,96,56,-69,124,-113,92,117,-86,-73,-100,10,-104,-73,69,89);
signed char inputData_1(89);
signed char inputData_2(108);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<320, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(21083,15134);
unsigned short inputData_1(39148);
unsigned short inputData_2(20882);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<321, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(11416,-14481);
short inputData_1(-27092);
short inputData_2(13529);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<322, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(52446,38364,52578);
unsigned short inputData_1(7041);
unsigned short inputData_2(50904);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<323, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-1440,-16374,26487);
short inputData_1(-30332);
short inputData_2(1760);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<324, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(55817,28195,46726,46163);
unsigned short inputData_1(50554);
unsigned short inputData_2(47973);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<325, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-13061,29510,5159,-29186);
short inputData_1(-31211);
short inputData_2(17613);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<326, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(43818,24219,16891,53202,47504,18491,34379,65154);
unsigned short inputData_1(64202);
unsigned short inputData_2(4372);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<327, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-3869,-25321,15264,31931,-21991,-5581,2557,14707);
short inputData_1(8292);
short inputData_2(-16128);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<328, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(57314,712,52710,61053,61705,27559,65138,7281,54137,27559,51789,48966,47317,54800,9608,48158);
unsigned short inputData_1(26765);
unsigned short inputData_2(36789);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<329, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-30542,-4867,-14242,-27283,8823,-3703,27469,27125,6255,3950,19393,10430,-29880,-18350,15581,8469);
short inputData_1(-6006);
short inputData_2(3527);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<330, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(2232,12896);
unsigned int inputData_1(18595);
unsigned int inputData_2(13501);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<331, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-29154,-8753);
int inputData_1(15108);
int inputData_2(-7597);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<332, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(2411,40934,54565);
unsigned int inputData_1(29371);
unsigned int inputData_2(210);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<333, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(-4706,3308,29268);
int inputData_1(12634);
int inputData_2(11533);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<334, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(24053,15107,50170,50408);
unsigned int inputData_1(32794);
unsigned int inputData_2(37054);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<335, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(14671,-11301,10778,30937);
int inputData_1(24560);
int inputData_2(-8147);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<336, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(21587,12797,46875,15498,38404,2391,32192,16183);
unsigned int inputData_1(18413);
unsigned int inputData_2(56937);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<337, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-3200,-29906,32445,-6823,-5063,16430,-23230,-13881);
int inputData_1(20161);
int inputData_2(-12551);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<338, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(2917,13371,31863,7396,44153,45183,5850,30002,14671,38125,52080,23472,30216,12508,18952,6495);
unsigned int inputData_1(192);
unsigned int inputData_2(56208);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<339, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(6732,24347,-309,-739,3799,14986,1259,28959,-22654,-32482,-2963,23567,8259,12805,29728,-21025);
int inputData_1(4352);
int inputData_2(30435);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<340, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(1461277518,2659364518);
unsigned long int inputData_1(3063090895);
unsigned long int inputData_2(2909869273);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<341, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-344119551,-1470443512);
long int inputData_1(2057691753);
long int inputData_2(164181673);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<342, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(3626399669,1871234621,3345979570);
unsigned long int inputData_1(8299220);
unsigned long int inputData_2(4166044655);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<343, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1852038705,-896803753,-142172843);
long int inputData_1(232918827);
long int inputData_2(1873572399);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<344, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(2659967554,2800255145,904335096,3956399103);
unsigned long int inputData_1(1797403214);
unsigned long int inputData_2(1510684654);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<345, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(626450180,1195158326,-78593188,-596488894);
long int inputData_1(-1043216044);
long int inputData_2(281000614);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<346, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(2520887108,4218383669,2257228073,2816841250,761131095,18231675,2399281746,3134723192);
unsigned long int inputData_1(2892186998);
unsigned long int inputData_2(3848880753);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<347, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(1933772334,1316306769,-73686200,-1368816448,1398677428,1684861628,-280027573,-851328691);
long int inputData_1(-1942521258);
long int inputData_2(1767942625);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<348, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(199152135,1817761496,4157826644,393443617,4159898412,2017781331,901422499,1148472030,78536229,2036418131,202583778,81407816,4081709188,1879992920,4250572287,681075772);
unsigned long int inputData_1(3494932294);
unsigned long int inputData_2(1027097492);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<349, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-1207285187,169844245,-2070559935,1295433934,-49122902,-1496442508,-1838785599,1609742741,412604545,1705385857,588675367,1954132763,-422701736,1157684008,1221916640,-24183544);
long int inputData_1(1288424317);
long int inputData_2(-1716190639);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<350, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(1476317651987639642,17712455675059692404);
unsigned long long int inputData_1(4573794758334161878);
unsigned long long int inputData_2(6524838978857775973);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<351, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-3961104468146649509,551141732206992099);
long long int inputData_1(7744231535285759666);
long long int inputData_2(-8315146472140911081);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<352, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(10311204080635939104,401844106793043007,15261534603138370245);
unsigned long long int inputData_1(12900976726398471144);
unsigned long long int inputData_2(4719782662794121651);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<353, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-1414304872956938442,819648462677032460,-8227693977783456823);
long long int inputData_1(5518153361733522638);
long long int inputData_2(-6680958230332420849);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<354, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(4067560946759918460,7824598371097796672,12195760828438154102,1637908118830746376);
unsigned long long int inputData_1(14292516322330385393);
unsigned long long int inputData_2(11549331509198080468);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<355, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(98085153692545501,7270389040638278262,-3447228564060397206,4227263268276240338);
long long int inputData_1(-2046796474921289554);
long long int inputData_2(-4814311144938577149);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<356, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(16387234527955820640,14610388082105247065,14756224963136479266,3827637233883241576,897476985477865115,12001010840014799136,15638039734661429633,856807443760831880);
unsigned long long int inputData_1(3027826642960495845);
unsigned long long int inputData_2(7659925061724097590);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<357, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(8622309914985202449,4405781761925400366,3006490009874206666,2392583270999247510,-6974016487695407261,3619046406097395206,3043878066063524674,2058518226454962101);
long long int inputData_1(-830826380702155190);
long long int inputData_2(-4845837082388961804);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<358, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(11037309151329172400,6852381771223833220,12816637683368049396,12512583657936592674,11386538656313140526,3039382910423838383,12786744133737440537,8405701122269367953,5795263297260703983,427446008879625077,17542666995668222938,16988305267879619731,7559045753083575848,16870337503368055361,7125591737138093908,10572378113127610387);
unsigned long long int inputData_1(8349402439775996455);
unsigned long long int inputData_2(16247905585377121250);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<359, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-848126441122814029,2452416892776816104,-8474495252769709676,7013460994046935850,-1393843330241024421,-2171934354244195724,5141801258543213422,5265292132158759114,-2148059434510031330,5920772965001052580,7097842653163021064,-1435650567683594978,6508573966071956299,4779818410092020972,-4405861626416422371,7853547409020252752);
long long int inputData_1(-5260507914301276293);
long long int inputData_2(-8989362580617832286);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<360, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(220,107);
cl::sycl::uchar2 inputData_1(150,252);
cl::sycl::uchar2 inputData_2(239,161);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<361, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(59,36);
cl::sycl::schar2 inputData_1(-21,-99);
cl::sycl::schar2 inputData_2(-75,-52);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<362, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(54,238,52);
cl::sycl::uchar3 inputData_1(60,178,248);
cl::sycl::uchar3 inputData_2(163,75,208);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<363, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-88,38,-2);
cl::sycl::schar3 inputData_1(64,-4,31);
cl::sycl::schar3 inputData_2(-128,87,57);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<364, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(50,242,253,160);
cl::sycl::uchar4 inputData_1(214,253,78,47);
cl::sycl::uchar4 inputData_2(82,247,75,115);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<365, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(89,-83,-12,-61);
cl::sycl::schar4 inputData_1(-21,117,-17,-113);
cl::sycl::schar4 inputData_2(100,91,-41,-116);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<366, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(126,164,112,17,23,191,136,166);
cl::sycl::uchar8 inputData_1(75,111,87,167,116,28,173,249);
cl::sycl::uchar8 inputData_2(105,8,138,163,90,230,243,120);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<367, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(1,-20,47,-113,-50,76,-79,9);
cl::sycl::schar8 inputData_1(-73,-99,-25,12,-61,-42,-13,58);
cl::sycl::schar8 inputData_2(-90,22,25,121,27,-24,-5,29);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<368, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(136,172,75,63,250,241,189,251,254,69,88,235,183,38,201,49);
cl::sycl::uchar16 inputData_1(52,157,53,97,41,176,157,232,24,252,78,7,89,242,74,126);
cl::sycl::uchar16 inputData_2(70,173,95,236,27,160,252,94,136,198,37,121,218,225,134,204);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<369, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(8,46,79,-109,-105,-117,101,-72,1,-37,-29,-39,94,-101,61,0);
cl::sycl::schar16 inputData_1(21,-8,70,11,-23,-23,-20,13,-66,122,92,-42,8,38,-2,-84);
cl::sycl::schar16 inputData_2(36,42,22,5,-8,122,-103,98,-128,66,-11,8,-115,116,-82,83);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<370, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(65268,58129);
cl::sycl::ushort2 inputData_1(41727,9977);
cl::sycl::ushort2 inputData_2(38828,44303);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<371, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(28122,12197);
cl::sycl::short2 inputData_1(-22172,-30278);
cl::sycl::short2 inputData_2(20155,-9677);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<372, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(22007,22917,60486);
cl::sycl::ushort3 inputData_1(54804,51830,31819);
cl::sycl::ushort3 inputData_2(64509,37904,58221);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<373, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(30640,6662,6099);
cl::sycl::short3 inputData_1(30352,-30784,2522);
cl::sycl::short3 inputData_2(-25004,20892,28549);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<374, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(36750,19594,59059,6731);
cl::sycl::ushort4 inputData_1(51709,2725,50880,50033);
cl::sycl::ushort4 inputData_2(51841,43440,11938,53911);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<375, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-18086,21549,8314,22436);
cl::sycl::short4 inputData_1(2494,-31879,27727,4412);
cl::sycl::short4 inputData_2(-3517,-2805,4327,18478);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<376, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(20731,28088,12486,31471,50916,2794,55545,10452);
cl::sycl::ushort8 inputData_1(10709,13882,48543,44989,55562,12387,59259,10566);
cl::sycl::ushort8 inputData_2(10523,27584,21980,49797,44274,3909,7195,34415);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<377, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-25946,4057,14334,7782,-32150,28575,-16094,-2763);
cl::sycl::short8 inputData_1(1618,31285,-12506,10331,22178,4231,-18164,-14543);
cl::sycl::short8 inputData_2(-24199,15205,-305,-11185,27409,-19679,-24459,7900);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<378, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(25803,24848,38611,39920,1257,6448,3118,2221,13849,59661,5019,12667,37794,24813,43631,13568);
cl::sycl::ushort16 inputData_1(45177,4901,23162,28002,40250,1150,2106,7543,39423,28814,45430,29332,489,16105,61077,54570);
cl::sycl::ushort16 inputData_2(49247,20726,47283,18126,2171,25623,45400,15965,64772,20701,40938,6900,5799,52501,39182,6959);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<379, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-7107,8395,19921,-4040,-2719,24633,5763,17791,3177,12602,20114,21872,-20697,13413,14003,-4381);
cl::sycl::short16 inputData_1(-12868,-19213,28917,12172,27706,9123,-12470,18497,23966,20694,2891,-51,20939,13538,-27283,16984);
cl::sycl::short16 inputData_2(20836,332,-18211,-3101,-197,21326,-287,23900,-4963,-21129,-5908,19616,7180,27079,13714,12886);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<380, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(60517,47113);
cl::sycl::uint2 inputData_1(35686,56810);
cl::sycl::uint2 inputData_2(60334,60720);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<381, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-17936,-21);
cl::sycl::int2 inputData_1(9085,-29500);
cl::sycl::int2 inputData_2(-5671,6862);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<382, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(55099,2753,21050);
cl::sycl::uint3 inputData_1(20663,10692,21313);
cl::sycl::uint3 inputData_2(64970,32154,4793);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<383, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(18723,31840,27627);
cl::sycl::int3 inputData_1(7211,-6394,21040);
cl::sycl::int3 inputData_2(4797,-21078,10772);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<384, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(31158,45793,13499,19013);
cl::sycl::uint4 inputData_1(19965,19027,46102,23003);
cl::sycl::uint4 inputData_2(43610,1002,1638,30688);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<385, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(27310,-29687,-27149,32756);
cl::sycl::int4 inputData_1(-12385,-27922,-28671,-27519);
cl::sycl::int4 inputData_2(-18092,5771,-30506,22291);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<386, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(61705,21866,54395,39869,1564,54304,63905,48068);
cl::sycl::uint8 inputData_1(47729,26486,56055,1699,23326,54317,11184,12467);
cl::sycl::uint8 inputData_2(3454,1913,58122,50485,35645,55362,9941,64559);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<387, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(14554,20887,-15879,-31513,7362,16378,28051,-23001);
cl::sycl::int8 inputData_1(-18079,7411,20634,-13305,-7123,12887,-27657,-1537);
cl::sycl::int8 inputData_2(-15204,-30569,-8794,-13253,25861,4280,-17473,-12645);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<388, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(34379,64787,17952,24761,7341,5045,12919,42560,44869,53953,54179,17969,24347,61944,34354,42342);
cl::sycl::uint16 inputData_1(46014,60479,26878,50798,29135,22417,48299,40238,20307,38514,6903,59971,2068,4672,45615,51445);
cl::sycl::uint16 inputData_2(61198,27702,61660,32928,5967,19370,32472,57750,3910,35535,36039,56735,24313,31064,43553,14844);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<389, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(8911,-3311,-17013,-16385,6079,11666,-14187,-7466,30532,-10123,5997,-2821,-5587,-17627,-8280,21613);
cl::sycl::int16 inputData_1(-3018,32544,29726,-15760,-24255,13523,-30400,-19962,-13607,-17182,-2704,29646,-849,-11734,-20959,216);
cl::sycl::int16 inputData_2(-32547,-18293,2498,23975,-8322,-10965,-12454,23829,917,26145,11407,-15951,-7381,-27247,-28991,28132);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<390, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(2515517150,3528633028);
cl::sycl::ulong2 inputData_1(978294332,2406839776);
cl::sycl::ulong2 inputData_2(2316455703,547668562);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<391, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(1344382387,1891554281);
cl::sycl::long2 inputData_1(1810873077,-1406998619);
cl::sycl::long2 inputData_2(361195407,1888636577);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<392, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(4253821556,2133986931,3805508041);
cl::sycl::ulong3 inputData_1(3700931313,3772587559,2914756246);
cl::sycl::ulong3 inputData_2(4026205596,2924311979,514595833);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<393, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1244996998,-1199896508,-1481949912);
cl::sycl::long3 inputData_1(68207055,-832958770,-880347258);
cl::sycl::long3 inputData_2(-673510266,-1567575538,-93505991);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<394, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(3815308389,2199199943,3276130637,1671463336);
cl::sycl::ulong4 inputData_1(4257729286,1495554740,1318331022,2914171748);
cl::sycl::ulong4 inputData_2(3004272813,1618298030,1756278871,906166012);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<395, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(494310433,-1050004910,94852183,-1931817355);
cl::sycl::long4 inputData_1(-78990830,1567655905,-375666605,418272189);
cl::sycl::long4 inputData_2(-1399458176,-2081850125,-1065063812,-1391767742);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<396, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(3513886593,1171520042,4150222016,3042820457,3639215564,440717100,1747551127,1420417176);
cl::sycl::ulong8 inputData_1(883054134,3956446522,3364595239,705263355,2588177134,3536448764,1652480970,3538661006);
cl::sycl::ulong8 inputData_2(2297372223,3165034244,3490355698,418892279,3341440649,1944000755,3626780707,586465273);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<397, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-843108033,149128871,657426256,-683098056,769995309,1852567762,-707035464,-1857087524);
cl::sycl::long8 inputData_1(-587213553,-2085045341,-1515151804,706792680,-959899236,-1601643399,1769971271,-1452268501);
cl::sycl::long8 inputData_2(-682232651,5554503,1557663998,1096522535,386576961,2055870843,62253205,-1310341583);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<398, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(2098931034,1601673765,1695095663,2128523654,3528100526,2346447340,3018176489,4035081703,519364304,3125781454,3398532589,2667657275,3145605544,1357537954,419118904,646627572);
cl::sycl::ulong16 inputData_1(3585369959,2125965553,772609185,1974411813,2260453271,2064648036,111611335,802150566,2578188417,1462523371,3334922863,1329998142,3386666922,2426431914,1387526946,1299405396);
cl::sycl::ulong16 inputData_2(1739364188,317260758,3351947631,3944006547,2748512394,2529947307,3373270413,404450847,3207645251,2947375872,658579537,2428119947,3447851548,1039542688,339936369,443105845);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<399, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-828050771,-497928439,1435300234,-97839679,1132653823,-1134922520,855684095,2044630890,-370532487,1555170506,-1906248318,-1813094657,-1979630966,1545712626,1426038052,1055041889);
cl::sycl::long16 inputData_1(1635277757,-849502254,1174879080,1646179737,432987483,1283636791,1434373767,665621200,1285955114,1697146129,-1225055622,-1258441692,500421518,1638014028,1361180031,1360907670);
cl::sycl::long16 inputData_2(-1480173310,-1013693705,1871840030,965236735,962153740,1477749098,769027370,-888840364,1643819117,-725719332,1916563545,633486814,-1310617260,1526979264,1266173112,1643454425);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<400, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(7814929444389930860,8917599515875480596);
cl::sycl::ulonglong2 inputData_1(853604149634380066,13199113710869942388);
cl::sycl::ulonglong2 inputData_2(15786580356518699360,16606434184487563674);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<401, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-8837077464624131930,4796461231805481308);
cl::sycl::longlong2 inputData_1(4432771798649641254,8492897352699159650);
cl::sycl::longlong2 inputData_2(-8460490489728457520,-6384959945230813398);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<402, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(7532514326246716264,10871259958083792236,960662814034229324);
cl::sycl::ulonglong3 inputData_1(896836209865488248,13112988558484322466,16304346391986873250);
cl::sycl::ulonglong3 inputData_2(5248831533401925046,12112914233259723046,13589569247962235541);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<403, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-6960558930188593130,300240478965166939,7541942147166240526);
cl::sycl::longlong3 inputData_1(-8901688572348483865,1304422233955429333,5602881110421486546);
cl::sycl::longlong3 inputData_2(-3113613489029636911,3593476065960867224,8766335793588326464);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<404, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(315764929878123986,2356107912217978492,600835652323060335,4187819519580465148);
cl::sycl::ulonglong4 inputData_1(9572315119917331241,13137602885627245712,5447288147301254761,3942872248307419332);
cl::sycl::ulonglong4 inputData_2(15391256930102804077,3984111340106673452,3062931294393798754,4009533840290885087);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<405, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(753174434673626587,-6841664270597414565,-6831641080188711581,7984530644920579544);
cl::sycl::longlong4 inputData_1(2841009618616940743,-7195418537603057241,-8358279987275554168,-8662400855160833608);
cl::sycl::longlong4 inputData_2(-6182475859606630604,-1151072033394959196,7436127015854546086,-6144667439143523032);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<406, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(2673542476119096985,5478588664343267255,4792126146322098330,17037629156338271671,10258244198970885445,8414693337414730569,8193045932578085904,10021394555192669292);
cl::sycl::ulonglong8 inputData_1(7713425307433877358,13555414065478671209,12527701864006788195,13230694884820102659,6076952543705585237,16612014131930501161,16098707878869404422,17763950310410626173);
cl::sycl::ulonglong8 inputData_2(10689076464877522367,2307080752767447904,3190324509745749276,9773744089919342697,3739534178741037210,9287321749164907559,10860017079024857840,3206456903267178696);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<407, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(721605163782003560,-1196532425101042757,-41089136065291630,6530014153908698044,-5477822485875554435,-3551032117207268709,4283279725175722138,6316797575704895353);
cl::sycl::longlong8 inputData_1(-8141546242555671921,-7188940932839984581,2355627415399278526,-2807053350172177690,-3458450478065859804,-4527306905885460616,2529372096525675709,-6421881040454926876);
cl::sycl::longlong8 inputData_2(-1915946597125781872,-7294601199126928953,8484546045569985143,5486172374407524762,-8351657102381706535,-2549594990784380678,2532104491055354904,-5232332687804872130);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<408, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(1690670408138638693,9049808900563183626,10020715039761696034,14287299927050104842,7440376657212188217,4535602269992679546,16936935947898726838,6075081971840492439,4007084203427490417,14378540525225245696,6250569596688682471,10305375405891912274,8440868033191124409,663434468254306005,6787521136197801911,3016049637094471604);
cl::sycl::ulonglong16 inputData_1(11828944316025276817,1518174842538301240,4154313681642246416,8234301178706895050,9156771822866502707,5702127834600418272,16671895808726061260,7831769495936701778,6105186887668555422,12697418499285051833,4487881546686563490,3124912696086835835,2939420514521923712,5514950089746156805,7244913245676711098,10274381366522380133);
cl::sycl::ulonglong16 inputData_2(5478847449343995466,10186464579231030158,5448802577613429581,12856765919949655285,8742332363755983787,14894760826216729397,409891565869653049,2634786279832121354,723298498607804878,14606552507204887050,5685194131137556773,7904220291878938702,6671826523299873260,9533560277878998677,14828380287046469539,5289048920404875160);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<409, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(1170433216267552187,7796183007936582248,-7266497618922337419,-9144324971474218428,-8393681760848337583,2797105288128688407,7316612975394886833,-5626384357044249176,6042484589233949709,-8256721722340309901,6920096023059132867,3451719348933953645,7880288123122595346,-3568276588813372132,2429472104793242131,-848757799801076476);
cl::sycl::longlong16 inputData_1(2189634983682758255,3922289691928989514,-6455465957874601161,6104346165593453046,2074901888251677371,3008683742952008406,-5410532939050918335,1569615415188489090,4130504246057297799,1702418124453553839,-8563752116915180475,9201665228800287679,-7908103265853070855,6035951394717657223,3062691822059488439,-8595980328883714071);
cl::sycl::longlong16 inputData_2(-4498551282280395993,3070246089223615658,-1384901797527767997,-2743683065585732778,8054375902316169083,-4612056994618730962,-3586391072852901655,-5214762161642801783,7618217168450062267,5902022110087297453,64640941445807944,1460122401414828860,5112336328891871169,454884427698699508,-2059132575126689144,-7902327996518324944);
return cl::sycl::clamp(inputData_0,inputData_1,inputData_2);

});

test_function<410, unsigned char>(
[=](){
unsigned char inputData_0(240);
return cl::sycl::clz(inputData_0);

});

test_function<411, signed char>(
[=](){
signed char inputData_0(54);
return cl::sycl::clz(inputData_0);

});

test_function<412, unsigned short>(
[=](){
unsigned short inputData_0(9427);
return cl::sycl::clz(inputData_0);

});

test_function<413, short>(
[=](){
short inputData_0(-17528);
return cl::sycl::clz(inputData_0);

});

test_function<414, unsigned int>(
[=](){
unsigned int inputData_0(40235);
return cl::sycl::clz(inputData_0);

});

test_function<415, int>(
[=](){
int inputData_0(-13166);
return cl::sycl::clz(inputData_0);

});

test_function<416, unsigned long int>(
[=](){
unsigned long int inputData_0(676646487);
return cl::sycl::clz(inputData_0);

});

test_function<417, long int>(
[=](){
long int inputData_0(1439625719);
return cl::sycl::clz(inputData_0);

});

test_function<418, unsigned long long int>(
[=](){
unsigned long long int inputData_0(7323431383950491871);
return cl::sycl::clz(inputData_0);

});

test_function<419, long long int>(
[=](){
long long int inputData_0(3751875236586475376);
return cl::sycl::clz(inputData_0);

});

test_function<420, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(105,160);
return cl::sycl::clz(inputData_0);

});

test_function<421, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-93,121);
return cl::sycl::clz(inputData_0);

});

test_function<422, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(192,210,52);
return cl::sycl::clz(inputData_0);

});

test_function<423, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(86,-47,-41);
return cl::sycl::clz(inputData_0);

});

test_function<424, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(170,223,174,74);
return cl::sycl::clz(inputData_0);

});

test_function<425, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(83,-94,113,52);
return cl::sycl::clz(inputData_0);

});

test_function<426, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(143,53,117,108,135,195,210,123);
return cl::sycl::clz(inputData_0);

});

test_function<427, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(29,80,54,25,-25,-113,84,-77);
return cl::sycl::clz(inputData_0);

});

test_function<428, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(145,54,0,103,138,66,12,197,123,13,35,179,252,240,246,116);
return cl::sycl::clz(inputData_0);

});

test_function<429, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(40,-107,22,60,101,-107,-23,-59,-99,-68,76,14,12,-69,83,-4);
return cl::sycl::clz(inputData_0);

});

test_function<430, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(54692,28729);
return cl::sycl::clz(inputData_0);

});

test_function<431, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-32655,-17912);
return cl::sycl::clz(inputData_0);

});

test_function<432, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(39905,10934,31753);
return cl::sycl::clz(inputData_0);

});

test_function<433, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(6428,-7166,-30897);
return cl::sycl::clz(inputData_0);

});

test_function<434, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(4061,53055,32442,6239);
return cl::sycl::clz(inputData_0);

});

test_function<435, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-19333,-11713,-29150,7011);
return cl::sycl::clz(inputData_0);

});

test_function<436, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(32818,50979,31231,35684,52686,54201,21388,6401);
return cl::sycl::clz(inputData_0);

});

test_function<437, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(26053,17753,-31483,-28834,23830,-6339,-32412,4320);
return cl::sycl::clz(inputData_0);

});

test_function<438, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(65292,64446,16006,48377,12155,53714,40307,53531,45051,54482,9009,17714,12243,8108,14551,5464);
return cl::sycl::clz(inputData_0);

});

test_function<439, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-3324,-10145,-32187,-19956,13259,-29923,-31550,23367,27200,-30347,-15738,-9280,5306,8338,30436,-15048);
return cl::sycl::clz(inputData_0);

});

test_function<440, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(26319,24654);
return cl::sycl::clz(inputData_0);

});

test_function<441, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(22188,-20111);
return cl::sycl::clz(inputData_0);

});

test_function<442, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(59591,43389,32060);
return cl::sycl::clz(inputData_0);

});

test_function<443, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(14386,-19876,-29881);
return cl::sycl::clz(inputData_0);

});

test_function<444, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(56718,23374,57241,40855);
return cl::sycl::clz(inputData_0);

});

test_function<445, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(5380,25074,-12856,-7655);
return cl::sycl::clz(inputData_0);

});

test_function<446, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(55364,59471,51497,60225,36074,11356,55589,57889);
return cl::sycl::clz(inputData_0);

});

test_function<447, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-12833,10932,-31536,27924,-10019,4189,12066,21196);
return cl::sycl::clz(inputData_0);

});

test_function<448, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(46332,21432,60991,14609,48949,33233,16609,33604,20638,34135,13667,17328,16031,56465,25659,56159);
return cl::sycl::clz(inputData_0);

});

test_function<449, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-7946,-11767,10754,-6255,20886,9645,13094,30790,22605,-10379,-2661,23986,25741,2548,2994,-27311);
return cl::sycl::clz(inputData_0);

});

test_function<450, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(913776293,508157064);
return cl::sycl::clz(inputData_0);

});

test_function<451, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-750297711,-2079104022);
return cl::sycl::clz(inputData_0);

});

test_function<452, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2180666814,2560132,1544887151);
return cl::sycl::clz(inputData_0);

});

test_function<453, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1708831661,-1103092916,704910663);
return cl::sycl::clz(inputData_0);

});

test_function<454, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(1135953185,24798835,3545829227,2315086578);
return cl::sycl::clz(inputData_0);

});

test_function<455, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(1643220085,-994050821,-773930403,-2015010498);
return cl::sycl::clz(inputData_0);

});

test_function<456, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(2180151192,3842984179,7670878,2480438924,1032224206,2950046293,3974841210,2080298441);
return cl::sycl::clz(inputData_0);

});

test_function<457, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(1618182532,-490890848,-1212505465,-332042760,681186562,1298590530,760155977,-1838922134);
return cl::sycl::clz(inputData_0);

});

test_function<458, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3416596438,569173325,494179107,912027139,748233081,829633950,861175998,2299095883,3694975714,2256620903,2823482110,3359940989,4252475829,1178076344,4046052822,839839024);
return cl::sycl::clz(inputData_0);

});

test_function<459, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-1319431000,1270802970,-1808346522,1723266106,1731347443,868138809,844464475,-87805401,-1843651918,-662199302,1119815616,376636955,-833883787,-1407897028,-1242701248,146763540);
return cl::sycl::clz(inputData_0);

});

test_function<460, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(18409907339769700410,10986132924762172292);
return cl::sycl::clz(inputData_0);

});

test_function<461, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-6268505754172183554,-5385832651777731690);
return cl::sycl::clz(inputData_0);

});

test_function<462, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(7695624744564290802,11977248871099101110,18201791146376351393);
return cl::sycl::clz(inputData_0);

});

test_function<463, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-8313237273939867123,7172732515707774874,6031085334397726883);
return cl::sycl::clz(inputData_0);

});

test_function<464, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(2765951045886541153,6582880859993161715,16763436161232827892,18017682408831915929);
return cl::sycl::clz(inputData_0);

});

test_function<465, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(6269497593760040939,-4551763356473886346,-7177032789668342833,-7513490132026821918);
return cl::sycl::clz(inputData_0);

});

test_function<466, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(4084659196655883638,8171397277994493107,11534767847915298884,3451662409392481056,2410910887382018566,610882019085969114,5798698163846563711,7438869151173053374);
return cl::sycl::clz(inputData_0);

});

test_function<467, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(8021772195792429608,-6276009974136570878,3728212652834831600,2023535567507639721,492933162248042078,6252131448650403201,-3097356672602010150,-2101394885152365386);
return cl::sycl::clz(inputData_0);

});

test_function<468, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(9177109458591463557,13673321815606971773,6422820025576378830,10273355758116198533,4918282033492548371,9068045863241334619,4631740666923428400,17438190897541942106,8779732678684317164,10047080011673488384,1327363250480860437,3468743398866647678,10022150020956234292,242530645794605571,7418779728479894288,12715359195998159240);
return cl::sycl::clz(inputData_0);

});

test_function<469, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-6948274344944940843,3611976600120033825,6568839967371913238,7292156305970807261,8948208336854682813,5816632915193593250,-2155692759421526581,4902487093660640073,-3582386745529773939,8575472860129814595,7588373725569111107,5768291894656280975,8737088562208980009,-3109430310047515353,-4053777425312774276,-6710962502628414196);
return cl::sycl::clz(inputData_0);

});

test_function<470, unsigned char>(
[=](){
unsigned char inputData_0(203);
unsigned char inputData_1(244);
unsigned char inputData_2(223);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<471, signed char>(
[=](){
signed char inputData_0(-43);
signed char inputData_1(-109);
signed char inputData_2(23);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<472, unsigned short>(
[=](){
unsigned short inputData_0(17600);
unsigned short inputData_1(9487);
unsigned short inputData_2(51469);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<473, short>(
[=](){
short inputData_0(5287);
short inputData_1(9448);
short inputData_2(6155);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<474, unsigned int>(
[=](){
unsigned int inputData_0(28670);
unsigned int inputData_1(22283);
unsigned int inputData_2(48911);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<475, int>(
[=](){
int inputData_0(12251);
int inputData_1(-2437);
int inputData_2(-25116);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<476, unsigned long int>(
[=](){
unsigned long int inputData_0(22696110);
unsigned long int inputData_1(1119058517);
unsigned long int inputData_2(488283819);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<477, long int>(
[=](){
long int inputData_0(-1829159025);
long int inputData_1(1886415466);
long int inputData_2(1674070448);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<478, unsigned long long int>(
[=](){
unsigned long long int inputData_0(16713925574075610959);
unsigned long long int inputData_1(3312923839815916076);
unsigned long long int inputData_2(7472387468473804365);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<479, long long int>(
[=](){
long long int inputData_0(-8607786456263907667);
long long int inputData_1(2021040750296286924);
long long int inputData_2(9128767526512939808);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<480, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(74,157);
cl::sycl::uchar2 inputData_1(167,80);
cl::sycl::uchar2 inputData_2(76,25);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<481, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(114,-2);
cl::sycl::schar2 inputData_1(51,73);
cl::sycl::schar2 inputData_2(-60,-79);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<482, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(95,156,135);
cl::sycl::uchar3 inputData_1(29,251,253);
cl::sycl::uchar3 inputData_2(217,43,107);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<483, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(78,125,29);
cl::sycl::schar3 inputData_1(-83,103,-53);
cl::sycl::schar3 inputData_2(80,41,-40);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<484, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(155,96,153,200);
cl::sycl::uchar4 inputData_1(183,148,184,205);
cl::sycl::uchar4 inputData_2(96,52,176,53);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<485, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(40,60,-42,-70);
cl::sycl::schar4 inputData_1(63,121,64,95);
cl::sycl::schar4 inputData_2(-6,-65,-100,-90);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<486, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(69,200,32,255,10,145,147,128);
cl::sycl::uchar8 inputData_1(42,118,217,65,94,78,99,156);
cl::sycl::uchar8 inputData_2(133,245,120,77,4,75,213,254);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<487, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(18,-5,-30,-44,-39,79,-40,19);
cl::sycl::schar8 inputData_1(-92,97,-50,-11,-64,-86,112,52);
cl::sycl::schar8 inputData_2(109,-118,13,25,118,55,34,-60);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<488, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(229,151,0,163,220,25,49,139,10,99,147,62,217,25,205,53);
cl::sycl::uchar16 inputData_1(213,167,190,133,210,41,192,4,64,21,115,70,225,10,195,224);
cl::sycl::uchar16 inputData_2(158,50,0,22,211,164,231,178,27,164,80,160,45,203,180,134);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<489, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(26,-18,3,2,-43,-59,-31,-101,-128,65,95,-40,-53,61,98,-116);
cl::sycl::schar16 inputData_1(-82,-103,-37,84,-120,-44,112,18,-47,94,65,-114,98,107,-1,32);
cl::sycl::schar16 inputData_2(-25,-41,33,69,-14,11,-94,-41,-15,100,28,-37,77,-67,20,113);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<490, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(849,37881);
cl::sycl::ushort2 inputData_1(50659,35638);
cl::sycl::ushort2 inputData_2(16142,56900);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<491, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(7039,-5459);
cl::sycl::short2 inputData_1(-3864,25187);
cl::sycl::short2 inputData_2(11426,32097);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<492, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(43996,7060,32321);
cl::sycl::ushort3 inputData_1(41701,1032,10204);
cl::sycl::ushort3 inputData_2(42280,3421,46451);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<493, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-23969,-4328,19457);
cl::sycl::short3 inputData_1(9013,-19364,3926);
cl::sycl::short3 inputData_2(14127,-19094,-28364);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<494, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(10101,51113,13052,51135);
cl::sycl::ushort4 inputData_1(62652,46306,52873,51434);
cl::sycl::ushort4 inputData_2(47995,45503,608,30271);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<495, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-6530,18502,17527,-3976);
cl::sycl::short4 inputData_1(15071,-21321,23689,-27907);
cl::sycl::short4 inputData_2(-2334,-11517,22108,-7443);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<496, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(7707,45037,11924,59665,47673,48486,20422,62514);
cl::sycl::ushort8 inputData_1(27053,17506,42837,54030,61425,27247,48440,5759);
cl::sycl::ushort8 inputData_2(60753,603,35487,19443,25567,28942,32508,52970);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<497, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-1530,349,-30999,-9003,21002,-12593,24811,-4531);
cl::sycl::short8 inputData_1(-14563,-24466,-16906,7346,-19368,31066,24039,-328);
cl::sycl::short8 inputData_2(29720,-29183,10364,-1856,-29861,12943,7685,12267);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<498, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(14953,5058,15826,27793,43220,47460,19938,54106,41655,14025,63857,59575,52211,37441,19237,9654);
cl::sycl::ushort16 inputData_1(16716,35857,27318,16080,44409,33283,8907,18021,23595,17309,61742,14349,53141,1194,39548,4925);
cl::sycl::ushort16 inputData_2(17642,64810,1990,24827,28852,24554,46443,50736,47523,17068,35393,55490,22229,19040,48081,217);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<499, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(14267,-8155,20647,25246,-102,-1324,-8779,19372,-22415,-6889,-16709,-25820,17093,25100,18586,-19609);
cl::sycl::short16 inputData_1(-14732,6360,-15848,-12941,1264,19759,6486,30569,-10842,18932,4523,-19626,-25127,-141,-19491,-6925);
cl::sycl::short16 inputData_2(-29952,-4289,-7850,-2380,20414,9414,-11336,-31352,-23439,301,24980,-2665,-5220,414,-24212,19705);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<500, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(32156,4814);
cl::sycl::uint2 inputData_1(59830,4486);
cl::sycl::uint2 inputData_2(1869,168);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<501, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(2964,-11486);
cl::sycl::int2 inputData_1(-28478,-29801);
cl::sycl::int2 inputData_2(-1953,-28136);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<502, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(13095,48647,20261);
cl::sycl::uint3 inputData_1(1925,62790,61175);
cl::sycl::uint3 inputData_2(41791,49818,60678);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<503, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(-10239,8260,31181);
cl::sycl::int3 inputData_1(-28729,23688,-3139);
cl::sycl::int3 inputData_2(27342,-8086,-29938);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<504, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(33800,60953,36099,15808);
cl::sycl::uint4 inputData_1(44515,36301,36264,15660);
cl::sycl::uint4 inputData_2(14448,715,63505,49229);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<505, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(26276,-3093,-17267,-1782);
cl::sycl::int4 inputData_1(31105,24321,15040,11715);
cl::sycl::int4 inputData_2(-4607,4760,17022,-17040);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<506, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(4388,48118,31686,43353,18905,46428,56629,39570);
cl::sycl::uint8 inputData_1(354,53423,14730,30608,5849,25997,58536,32560);
cl::sycl::uint8 inputData_2(46808,29561,32457,18013,44356,13259,59247,55140);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<507, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-16163,844,29760,23186,-25852,-6031,-8054,24863);
cl::sycl::int8 inputData_1(30252,17813,9345,-15386,-15425,-3547,-5984,13485);
cl::sycl::int8 inputData_2(-5381,25303,-26060,-23001,17306,24966,2010,3650);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<508, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(38050,53111,55185,20827,26213,47553,45330,34466,19678,45702,50711,44520,60026,1971,16253,13239);
cl::sycl::uint16 inputData_1(13555,2761,42361,38474,12172,36041,18292,52288,45461,32328,45125,57812,39425,23539,11542,31353);
cl::sycl::uint16 inputData_2(503,17440,34482,35840,11574,25733,31094,55649,13309,24087,60247,23757,35801,34134,59917,62032);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<509, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-14169,6801,15637,-29692,16536,-7238,9815,-6980,-20206,10194,-29003,4643,1668,-31350,11379,-216);
cl::sycl::int16 inputData_1(-16609,2347,9957,7815,13718,-18188,27318,-205,-14627,6353,-27041,-31605,-16044,22912,-25842,2070);
cl::sycl::int16 inputData_2(16195,5034,-17712,-604,-5747,-25007,-9608,32120,-29687,-22695,23003,23238,-25607,22748,23904,-1015);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<510, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(748807178,2128422045);
cl::sycl::ulong2 inputData_1(105798545,3308099254);
cl::sycl::ulong2 inputData_2(829305481,1417236725);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<511, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(385457146,660241663);
cl::sycl::long2 inputData_1(-1120352187,-494041104);
cl::sycl::long2 inputData_2(386996521,-996134397);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<512, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(3689394092,2242805016,2793913951);
cl::sycl::ulong3 inputData_1(3670641547,348359607,3563388540);
cl::sycl::ulong3 inputData_2(1134811659,893718002,2932728447);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<513, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-295577419,1075836395,-1752389792);
cl::sycl::long3 inputData_1(542791198,-92668682,1034877690);
cl::sycl::long3 inputData_2(-2037603961,-1833801222,650223497);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<514, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(4269435221,484855570,1082847085,225703240);
cl::sycl::ulong4 inputData_1(3500735832,457224054,3079866290,1618783142);
cl::sycl::ulong4 inputData_2(2296225438,1605863165,898448964,3648959214);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<515, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-1848574639,-531122761,1220215267,16204389);
cl::sycl::long4 inputData_1(80884763,-1663012946,-1245106811,991920919);
cl::sycl::long4 inputData_2(-804166949,1021597829,424216544,846012054);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<516, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(1286668045,1785008442,3654110081,825485489,316513499,3439122447,3704403806,769974621);
cl::sycl::ulong8 inputData_1(3506067959,987506830,3402158361,2962034583,1983588832,1688067395,3737337992,921994524);
cl::sycl::ulong8 inputData_2(2515819020,807175997,4003876130,433012931,4024087852,3851126135,4156579159,3097991174);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<517, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(1912202345,-2083690557,-589648935,120713607,284962651,-2021572212,1587289441,-1555543901);
cl::sycl::long8 inputData_1(-1556624490,-1822192306,1698501707,-1220090594,523919903,1717402085,827822518,797772136);
cl::sycl::long8 inputData_2(1358588955,-1798244453,664983637,-358401782,931221833,649810591,-798908386,292140465);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<518, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3285669187,3887201557,3817201799,816719165,1743965581,3974401988,2565223003,330799479,1028984199,1256864989,3055569911,1870260265,1760337441,3400458164,268147532,1682296840);
cl::sycl::ulong16 inputData_1(4222371581,3464539968,3431597388,438009312,484968562,87076585,189677504,4133626348,4051765617,4113181011,2414495512,614877263,405674649,3619467202,2460334209,4158412557);
cl::sycl::ulong16 inputData_2(880961223,1014548154,3355559520,310914915,2828914967,935773240,2812166937,3395645702,3882295534,3679564427,2313966154,3024448253,1496830935,2863544534,2621361088,1867898956);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<519, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-519451022,-70586003,1873934858,1388138564,-2116447290,1294660102,-419636628,2053856862,-1587777481,1498941400,-42357928,-1000004203,1823243956,264841649,1225240385,-357613735);
cl::sycl::long16 inputData_1(-753661018,1619456672,-366174198,-1383739741,661618855,1046759882,1766772876,752876810,-787693787,-1309329352,374559111,1373662988,-1679551444,1159614329,1522682962,264061245);
cl::sycl::long16 inputData_2(552647251,2089510311,1040415448,-860563040,1290189413,376440521,-592457398,1809546384,-702483745,390450093,2021785719,1761945390,1945763205,-649529822,-1085285493,754378987);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<520, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(17397799632664249971,15842323508147268329);
cl::sycl::ulonglong2 inputData_1(5090832492430164358,17664775623582173437);
cl::sycl::ulonglong2 inputData_2(17120905256745950479,10966832294949853788);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<521, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(1606874907324335352,1764822756129142574);
cl::sycl::longlong2 inputData_1(-2770456341611478884,517882489041225024);
cl::sycl::longlong2 inputData_2(3935083643952505766,4772946069719647414);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<522, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(3951418850442615581,7058606334648159059,4891436557801768555);
cl::sycl::ulonglong3 inputData_1(2590372623787819745,9215991286341455523,11330756606656574758);
cl::sycl::ulonglong3 inputData_2(5425979960586265259,5086493682508538453,15545765730080129630);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<523, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(2616748293810640568,6078600949529587753,-2622896157337911814);
cl::sycl::longlong3 inputData_1(-443023224137264200,6508933678534787198,-7287075042772548373);
cl::sycl::longlong3 inputData_2(-4327286965939140047,1901171325406450967,-1696024802555083063);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<524, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(16974007917975599522,16396967969153049054,14655927199898082492,763800515287045458);
cl::sycl::ulonglong4 inputData_1(5423535396631898216,10237740953240718479,366058115243998212,7856932800104073549);
cl::sycl::ulonglong4 inputData_2(4433175100495660949,18100080384128362423,10698962816943377617,13113489391902298570);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<525, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-6964679750363453674,-8956982488153363590,5659281047000798162,-53387696081508781);
cl::sycl::longlong4 inputData_1(-3822585497427195916,5134265942973316761,2823664242948701865,-6398862084625102126);
cl::sycl::longlong4 inputData_2(3440002306845840713,-4806596733538819993,706855459660641989,-2428660473736064942);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<526, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(9199509459017727624,1596854839872237857,576037200103066790,16587031082125195581,17370136797391085115,10043003752178861221,782211457595267160,14244070545266442583);
cl::sycl::ulonglong8 inputData_1(16627514735581744337,242471857457581721,7522908765223900538,14548802431247597136,8088217337398917359,4999799325466977333,11783252768510694643,1541918123170824666);
cl::sycl::ulonglong8 inputData_2(1557379042467956536,7683364490341454544,9096577060075276654,9262361102782303256,12819320287414490487,2394358569805610673,5975154663674090052,4642863326056557212);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<527, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-5005371522379335557,6580123821301132758,-4232180102161750511,-4181945305666637328,-6954915022104949171,-746941851966695908,3063729280701517372,6228162330059191815);
cl::sycl::longlong8 inputData_1(-3525461837442392934,-6787151078684817145,-2259913282246909302,7562860096104773008,7294318352450357081,3724644535594728144,-4073928744247314860,-5562972821211868169);
cl::sycl::longlong8 inputData_2(-8810895789843984403,-3265797433426775151,725198938424018978,-289651351279947334,1224610189423098165,1426057886339142518,5669835859503864021,-8513298443290805784);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<528, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(3813818952457627720,81729900428308299,10885782669787604610,14453989552988920440,15193152198831254686,1942010306184489190,11414533272627427377,13045496647100920523,18431872780290863276,12338206238552764154,7276685490656036667,14728652038954225062,5022414939497693426,12212955813256974010,14110995935767751235,1129620042199225567);
cl::sycl::ulonglong16 inputData_1(17353920802658033117,4377918705338039506,12193439393624681543,14311754661028444628,2447205280015603508,14873370604127548197,18359337834924413651,15666806566962992277,1323781531333792156,1998542969264937400,9734645340206172167,13297395525279124035,7005577454136836077,4189439269614241209,14904505573775963507,12323096989750013481);
cl::sycl::ulonglong16 inputData_2(7503663079252949649,5203970800084646651,16137775577139804112,11003404816343400191,10955951922126739093,11324025932957950357,14000938281283383898,15214358342911821339,7447350377396839002,12974256413076455180,74261396121880212,5582818877006443085,13128644522477188744,66473813535743744,4579432035020296887,16287744375796475764);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<529, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(8516432817959597578,6863458582466674174,181751368048131857,-5825024515365058809,-5593308710998075745,-411934463145159596,-8006590410575447101,5180389551335611236,-525902789912180816,-500199000006991659,-3026578232198057201,8413078366011506958,-5905873565531037955,221784650375517818,6226979292040944342,-437227021543563778);
cl::sycl::longlong16 inputData_1(3504108466571622594,7917349887227935143,1415792258331630011,-4139994045746675298,-8159550142815409214,-1333474082067942210,-3056145307610680424,-8756002056170154401,1575169208035704349,-6528214395685606607,-990210122904719518,1070076535339657644,2818337540861996715,-5304963035182748073,-8775394405221005407,6324103227366641308);
cl::sycl::longlong16 inputData_2(1967293239239540322,-5288557044563055599,-1602057815750136065,3255979881320386577,-4258107363816831381,2389679790864720841,-7267233747633097702,8862281469436104176,2692960121279948235,-2071259569616059410,3738437427815048372,6123455601215384966,-2310353580266655405,-2562699052205830640,4804481850007187602,-4454571762509004334);
return cl::sycl::mad_hi(inputData_0,inputData_1,inputData_2);

});

test_function<530, unsigned char>(
[=](){
unsigned char inputData_0(249);
unsigned char inputData_1(158);
unsigned char inputData_2(113);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<531, signed char>(
[=](){
signed char inputData_0(-67);
signed char inputData_1(-103);
signed char inputData_2(51);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<532, unsigned short>(
[=](){
unsigned short inputData_0(255);
unsigned short inputData_1(40894);
unsigned short inputData_2(23345);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<533, short>(
[=](){
short inputData_0(15850);
short inputData_1(15723);
short inputData_2(-17287);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<534, unsigned int>(
[=](){
unsigned int inputData_0(5449);
unsigned int inputData_1(48239);
unsigned int inputData_2(31481);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<535, int>(
[=](){
int inputData_0(-27425);
int inputData_1(5538);
int inputData_2(-30832);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<536, unsigned long int>(
[=](){
unsigned long int inputData_0(3696878762);
unsigned long int inputData_1(4233769571);
unsigned long int inputData_2(436501567);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<537, long int>(
[=](){
long int inputData_0(-364535473);
long int inputData_1(1773059738);
long int inputData_2(1879714331);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<538, unsigned long long int>(
[=](){
unsigned long long int inputData_0(10650800261779975938);
unsigned long long int inputData_1(14943644942800605309);
unsigned long long int inputData_2(11719718536270356582);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<539, long long int>(
[=](){
long long int inputData_0(889757660254432791);
long long int inputData_1(7198297301961535591);
long long int inputData_2(-723370553787456712);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<540, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(206,24);
cl::sycl::uchar2 inputData_1(214,40);
cl::sycl::uchar2 inputData_2(140,229);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<541, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(117,58);
cl::sycl::schar2 inputData_1(80,-81);
cl::sycl::schar2 inputData_2(88,-9);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<542, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(16,221,208);
cl::sycl::uchar3 inputData_1(7,98,159);
cl::sycl::uchar3 inputData_2(242,197,222);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<543, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(21,-107,18);
cl::sycl::schar3 inputData_1(89,-16,58);
cl::sycl::schar3 inputData_2(-120,-117,-33);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<544, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(31,106,157,26);
cl::sycl::uchar4 inputData_1(49,222,45,2);
cl::sycl::uchar4 inputData_2(106,34,160,186);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<545, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-93,114,47,-25);
cl::sycl::schar4 inputData_1(-97,8,53,24);
cl::sycl::schar4 inputData_2(-99,87,-107,-118);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<546, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(167,83,48,29,32,57,218,175);
cl::sycl::uchar8 inputData_1(139,25,33,169,150,211,39,45);
cl::sycl::uchar8 inputData_2(124,213,34,239,176,177,88,217);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<547, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-42,-60,-23,37,-47,-124,40,-7);
cl::sycl::schar8 inputData_1(-53,73,46,-79,-124,-3,-50,117);
cl::sycl::schar8 inputData_2(64,-126,-34,78,94,37,-67,-79);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<548, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(199,239,45,65,25,43,219,147,221,105,106,59,143,212,20,23);
cl::sycl::uchar16 inputData_1(109,78,127,253,105,103,101,8,72,134,231,77,125,157,32,8);
cl::sycl::uchar16 inputData_2(180,100,0,59,25,98,70,199,3,73,142,18,182,151,211,57);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<549, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-68,71,-107,98,-10,-123,65,65,95,76,-20,-31,-43,70,-94,36);
cl::sycl::schar16 inputData_1(-127,67,-78,82,-17,67,39,125,5,88,72,26,103,45,52,-31);
cl::sycl::schar16 inputData_2(-102,-24,-112,-47,-50,-86,-96,-121,32,37,118,-105,107,-70,-17,92);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<550, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(37608,55709);
cl::sycl::ushort2 inputData_1(51563,42142);
cl::sycl::ushort2 inputData_2(30992,60367);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<551, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(22345,10621);
cl::sycl::short2 inputData_1(9429,15772);
cl::sycl::short2 inputData_2(-29587,27567);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<552, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(7053,11094,45589);
cl::sycl::ushort3 inputData_1(50665,17180,46002);
cl::sycl::ushort3 inputData_2(20390,5606,29837);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<553, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(8714,-13802,-22023);
cl::sycl::short3 inputData_1(24160,22660,9077);
cl::sycl::short3 inputData_2(22298,-24474,-22829);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<554, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(45766,25448,12702,31890);
cl::sycl::ushort4 inputData_1(59463,40918,1073,46058);
cl::sycl::ushort4 inputData_2(25273,56082,55850,16646);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<555, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-11345,14013,20327,3994);
cl::sycl::short4 inputData_1(-19196,15501,10764,31318);
cl::sycl::short4 inputData_2(-30639,-16458,9574,21729);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<556, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(49515,39681,4326,5021,19992,25449,37861,52425);
cl::sycl::ushort8 inputData_1(61693,25863,53360,22684,62656,14047,55629,49406);
cl::sycl::ushort8 inputData_2(50876,52166,2343,12344,8964,17621,28718,12155);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<557, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-1471,-15011,-15198,15893,-5632,-28176,10060,3962);
cl::sycl::short8 inputData_1(650,-7838,7366,-26346,27528,-13655,-18334,25468);
cl::sycl::short8 inputData_2(5941,-20156,-64,8179,-19878,-4406,3064,16456);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<558, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(16737,31361,40368,9087,64229,44766,16204,42232,4933,61931,64334,26563,30715,2238,46602,14624);
cl::sycl::ushort16 inputData_1(40899,55241,57649,49838,45928,357,42209,30966,53588,14169,50989,45120,46021,25176,48505,23110);
cl::sycl::ushort16 inputData_2(32049,54185,2315,32627,16155,59872,13179,30691,19551,7511,47847,64598,43647,36433,3988,43858);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<559, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-30127,-23766,9784,23553,684,18017,1574,4857,5161,-27986,30041,-9300,16290,19322,-31471,22213);
cl::sycl::short16 inputData_1(-7979,5518,22395,-10966,-4719,-2871,-10381,-6710,-1059,15216,-4396,3573,-16028,-20183,30745,-22710);
cl::sycl::short16 inputData_2(-8825,8784,28671,12207,17420,26666,-13149,-29621,-3740,-9727,6397,-9273,8875,-18504,-9760,-21524);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<560, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(12605,25515);
cl::sycl::uint2 inputData_1(21316,37708);
cl::sycl::uint2 inputData_2(55394,11956);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<561, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-7523,26539);
cl::sycl::int2 inputData_1(21266,-32645);
cl::sycl::int2 inputData_2(13445,15952);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<562, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(51525,6198,30761);
cl::sycl::uint3 inputData_1(61799,32775,14756);
cl::sycl::uint3 inputData_2(30283,48547,60138);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<563, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(23688,-9548,3433);
cl::sycl::int3 inputData_1(-10156,19100,17965);
cl::sycl::int3 inputData_2(-28323,-12477,10356);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<564, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(8654,13339,49668,22285);
cl::sycl::uint4 inputData_1(18647,30108,61594,10539);
cl::sycl::uint4 inputData_2(8067,56810,46551,53689);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<565, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(31514,16643,-22661,-6128);
cl::sycl::int4 inputData_1(-30055,10641,5426,-7850);
cl::sycl::int4 inputData_2(-21112,23584,-9551,7595);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<566, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(42089,46776,59050,49292,24528,40212,55734,34313);
cl::sycl::uint8 inputData_1(53750,17069,51463,42991,1097,2272,38757,45174);
cl::sycl::uint8 inputData_2(60063,62629,8669,7563,16113,52584,8454,16930);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<567, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(1928,13296,15670,-20330,-12502,27516,246,9816);
cl::sycl::int8 inputData_1(-17348,15323,-13010,6472,2480,3292,29371,-9501);
cl::sycl::int8 inputData_2(-32727,-24369,20005,10832,-26633,-15153,-4410,-687);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<568, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(41866,26842,921,24410,30070,18657,21772,26109,903,23943,60815,59982,21286,18998,40465,13095);
cl::sycl::uint16 inputData_1(14588,39675,12524,16983,518,31083,56218,22053,47999,32410,9215,4937,34029,38078,45064,48177);
cl::sycl::uint16 inputData_2(21252,5530,55712,5672,54875,26630,16599,8842,12168,15233,63385,2513,6576,26816,40222,51592);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<569, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(18309,-28431,14740,26121,-21902,10652,10826,7569,-20714,-4659,-13969,19509,31785,-864,-23790,-28360);
cl::sycl::int16 inputData_1(-25183,-3963,26851,-2628,29255,30182,5889,3392,19301,-21605,21393,-32239,-4148,25555,9924,-17801);
cl::sycl::int16 inputData_2(-31459,-1721,10262,9077,-9844,8510,15238,11814,18188,9503,-18209,-29979,2127,7638,-26670,26884);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<570, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(1903299018,584261139);
cl::sycl::ulong2 inputData_1(2298470285,1949492567);
cl::sycl::ulong2 inputData_2(3432459413,346366868);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<571, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-1607227057,-1517829022);
cl::sycl::long2 inputData_1(1833300315,-305465432);
cl::sycl::long2 inputData_2(-145518399,1417482321);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<572, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(1409369456,1784765025,4198898160);
cl::sycl::ulong3 inputData_1(1209728824,3066951282,2851026487);
cl::sycl::ulong3 inputData_2(3428163368,3552104210,3088745495);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<573, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(787681664,275012964,2142471355);
cl::sycl::long3 inputData_1(-1061461355,-1531482793,1736131593);
cl::sycl::long3 inputData_2(-221362066,433635279,-860901056);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<574, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(2119592643,566498661,499473473,3458783687);
cl::sycl::ulong4 inputData_1(1763974305,848218060,3054553894,331023824);
cl::sycl::ulong4 inputData_2(918813129,4198246321,2214587283,2336925465);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<575, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-1255428013,1006649708,1332845973,-1542450788);
cl::sycl::long4 inputData_1(1013268444,-1793773964,2074401795,1806101555);
cl::sycl::long4 inputData_2(-293380622,-1124125383,-807254606,-2100575485);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<576, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(4197396633,3949492814,2910967666,2125119275,2637590765,247187297,814190128,519744337);
cl::sycl::ulong8 inputData_1(196782938,641670635,1456931430,122300703,3027569755,904563112,628066206,438230500);
cl::sycl::ulong8 inputData_2(1524126883,3866336218,3437522080,3715704576,2761503262,2572414274,1176521015,3529764008);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<577, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-274624910,1604789135,-799626076,-543893149,1978736595,66161510,-1250896911,584693509);
cl::sycl::long8 inputData_1(-1610375687,1573208434,759355644,1449283561,-1535889724,-909148002,1018334067,-1891956762);
cl::sycl::long8 inputData_2(1432080230,902433413,-508477642,616372984,-1020707895,243282407,-1998996572,-1574720484);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<578, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(848009877,1622873920,2485261274,1965625698,1829999235,4115191670,3759872817,2725649121,1590858349,4145288583,521498275,2396175150,560269700,2350274157,1959701847,1166306481);
cl::sycl::ulong16 inputData_1(3031047642,261480794,2559083754,1744215379,1435475430,1526055595,2335518925,2032183750,1959300953,3993473908,1583799763,1716449548,813259588,1701492135,1021539014,1324495033);
cl::sycl::ulong16 inputData_2(59493313,1843552249,2369767150,442463447,1469605658,198399988,1240685939,902057958,3637557447,384474452,1177131358,2729120113,3876159776,3314989883,331918525,254389540);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<579, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-811926164,1901205878,-1715736531,-1150342450,1363803769,-1948372391,-1405328592,550027669,1140863465,510123634,1150772180,-1345672280,-1139662784,2091848346,202530827,-201681377);
cl::sycl::long16 inputData_1(-2013778493,-1889142254,614883807,-736930525,1574946384,-538047121,-470303380,-1052857528,1418353388,-1292213346,275898688,-19773660,695427980,2129375304,1556283508,1566486107);
cl::sycl::long16 inputData_2(-861094032,1544360168,1595510358,472619716,-492565584,1898093151,1841644151,-1847286563,374931391,2105667489,740511514,-539599025,-889317401,-1696350642,196089157,804527827);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<580, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(9803045892795411485,13463290347678080892);
cl::sycl::ulonglong2 inputData_1(9135447540944775650,14351382948363205940);
cl::sycl::ulonglong2 inputData_2(15648284259204576075,11917329246209858558);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<581, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-6217412366067920253,5419071859579456299);
cl::sycl::longlong2 inputData_1(-3593229135939262337,883937386076956845);
cl::sycl::longlong2 inputData_2(-8770307236790776488,-2621947518428428654);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<582, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(12542545809193668677,6787476832093168376,10283806793608038636);
cl::sycl::ulonglong3 inputData_1(12427291474475961736,6166437035524956721,7683906828522398626);
cl::sycl::ulonglong3 inputData_2(13283575298479148931,11830551631449487744,8300913593596267003);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<583, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(1152105081885725616,8383539663869980295,-3013159033463244495);
cl::sycl::longlong3 inputData_1(9169239286331099647,8545168655265359544,69290337040907021);
cl::sycl::longlong3 inputData_2(-5670250901301018333,216462155376518854,-7910909987096217335);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<584, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(12386342548671650253,14289960240492411428,287906265695528597,2821646371500263025);
cl::sycl::ulonglong4 inputData_1(3451402331245540584,10469930441534118773,15376272376926689927,2671043876090500466);
cl::sycl::ulonglong4 inputData_2(11748823045433441068,8109363163283697612,810261391194415691,11619613451054474580);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<585, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-4713774672458165250,7161321293740295698,-7560042360032818022,1712118953348815386);
cl::sycl::longlong4 inputData_1(-5256951628950351348,3094294642897896981,4183324171724765944,1726930751531248453);
cl::sycl::longlong4 inputData_2(-614349234816759997,-7793620271163345724,5480991826433743823,-3977840325478979484);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<586, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(14091915926521315135,3693691162441803632,12994247426963817705,18341887438669034092,4327503941346759678,8488960384636063728,15931537829781600128,2725368065761006117);
cl::sycl::ulonglong8 inputData_1(1069781885704079300,11875452334202558628,6105416202160848489,3496791056624502379,3542422128480001641,14482253940621648246,6579570319242269242,17102829815642407733);
cl::sycl::ulonglong8 inputData_2(16832424368974571393,12006952529952514976,9117722926106042669,16467585196181911857,4216918968657150479,1028309947267809211,13349111996345818587,3339325255041669210);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<587, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(3002837817109371705,-6132505093056073745,-2677806413031023542,-3906932152445696896,-5966911996430888011,487233493241732294,8234534527416862935,8302379558520488989);
cl::sycl::longlong8 inputData_1(3895748400226584336,-3171989754828069475,6135091761884568657,3449810579449494485,-5153085649597103327,2036067225828737775,-2456339276147680058,-2321401317481120691);
cl::sycl::longlong8 inputData_2(5847800471474896191,6421268696360310080,426131359031594004,3388848179800138438,9095634920776267157,3909069092545608647,-6551917618131929798,-5283018165188606431);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<588, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(17838460598664281573,13448355111522325971,7206236042890793252,14178128506388101237,16478061486284511185,16215323412800735050,9373560689937531034,13577569244813527949,9573621305105439365,11809786552468337740,9592863561556506282,1231003202843148633,11285504874430107791,13744629803932158854,5464375308136557674,10477929060546265823);
cl::sycl::ulonglong16 inputData_1(7801246701368901752,2998855284163357670,11432623147423611595,16597580083629064501,2638242829732094260,8972038242192626167,14635974727194535250,13797739791804678595,9248059610121372112,11179235691871884884,3819103611572950050,4799134789419553541,12636102217660611250,14828218969763049318,4821904175955452786,3131023191964893967);
cl::sycl::ulonglong16 inputData_2(2856974678209179309,7031716499283170746,5892774107813669157,12084293772899672465,6853964848652473575,7243561636033790102,4324866046062969785,2229530337678953974,3207250972839195978,4294852343716686613,4520828951662597842,15874093648060166770,12107774080714160306,13866174015567796255,1845003062891666711,156650448309392492);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<589, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(4711072418277000515,-8205098172692021203,-7385016145788992368,5953521028589173909,-5219240995491769312,8710496141913755416,-6685846261491268433,4193173269411595542,-8540195959022520771,-4715465363106336895,-1020086937442724783,4496316677230042947,1321442475247578017,-7374746170855359764,-3206370806055241163,-2175226063524462053);
cl::sycl::longlong16 inputData_1(-9126728881985856159,-8235441378758843293,-3529617622861997052,-4696495345590499183,-2446014787831249326,3966377959819902357,-8707315735766590681,4940281453308003965,-4008494233289413829,-1007875458987895243,8007184939842565626,7006363475270750393,-3126435375497361798,-2666957213164527889,3425215156535282625,5057359883753713949);
cl::sycl::longlong16 inputData_2(-5792361016316836568,1155364222481085809,7552404711758320408,-9123476257323872288,-924920183965907175,1921314238201973170,3462681782260196063,7822120358287768333,-3130033938219713817,-3165995450630991604,-7647706888277832178,-8427901934971949821,4207763935319579681,1564279736903158695,3722632463806041635,939009161285897285);
return cl::sycl::mad_sat(inputData_0,inputData_1,inputData_2);

});

test_function<590, unsigned char>(
[=](){
unsigned char inputData_0(28);
unsigned char inputData_1(97);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<591, signed char>(
[=](){
signed char inputData_0(-27);
signed char inputData_1(-20);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<592, unsigned short>(
[=](){
unsigned short inputData_0(37999);
unsigned short inputData_1(64377);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<593, short>(
[=](){
short inputData_0(-12005);
short inputData_1(-7644);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<594, unsigned int>(
[=](){
unsigned int inputData_0(26391);
unsigned int inputData_1(6146);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<595, int>(
[=](){
int inputData_0(21840);
int inputData_1(-11936);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<596, unsigned long int>(
[=](){
unsigned long int inputData_0(2105241206);
unsigned long int inputData_1(1620480941);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<597, long int>(
[=](){
long int inputData_0(744603980);
long int inputData_1(1640570480);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<598, unsigned long long int>(
[=](){
unsigned long long int inputData_0(8017735608989312941);
unsigned long long int inputData_1(17017760777836849134);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<599, long long int>(
[=](){
long long int inputData_0(4396991617956302040);
long long int inputData_1(-7841751116480327549);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<600, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(46,121);
cl::sycl::uchar2 inputData_1(44,224);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<601, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(60,-29);
cl::sycl::schar2 inputData_1(89,-89);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<602, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(162,12,51);
cl::sycl::uchar3 inputData_1(17,75,119);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<603, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(2,-16,-76);
cl::sycl::schar3 inputData_1(-25,-32,-96);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<604, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(132,97,47,125);
cl::sycl::uchar4 inputData_1(99,64,104,138);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<605, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-5,-83,93,1);
cl::sycl::schar4 inputData_1(44,-16,-20,-51);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<606, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(35,137,255,7,194,132,151,130);
cl::sycl::uchar8 inputData_1(59,183,119,139,151,145,11,198);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<607, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(55,-88,-72,45,-18,126,-119,28);
cl::sycl::schar8 inputData_1(74,-49,115,-66,-47,-25,-110,37);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<608, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(178,200,22,83,151,136,167,119,222,236,199,12,35,45,228,0);
cl::sycl::uchar16 inputData_1(101,246,240,209,136,88,187,155,176,144,128,97,221,15,158,160);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<609, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(21,-82,-19,-6,39,-2,-105,-122,20,63,-87,-43,70,-10,-96,72);
cl::sycl::schar16 inputData_1(-22,-91,70,122,-47,59,41,0,46,-6,12,-92,127,-89,-25,-99);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<610, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(23016,14564);
cl::sycl::ushort2 inputData_1(24445,6791);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<611, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(13351,-8732);
cl::sycl::short2 inputData_1(29622,31426);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<612, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(2947,51959,59525);
cl::sycl::ushort3 inputData_1(55482,17888,39010);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<613, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(26314,25660,23710);
cl::sycl::short3 inputData_1(24988,14846,28421);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<614, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(60457,56431,22740,7520);
cl::sycl::ushort4 inputData_1(7594,42764,4739,52406);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<615, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(30893,12139,26868,19230);
cl::sycl::short4 inputData_1(21964,-22786,-17750,-21276);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<616, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(33850,19896,33291,2070,26523,28041,60534,52006);
cl::sycl::ushort8 inputData_1(15047,22850,60411,15423,32361,21871,36495,18592);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<617, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-13838,-31439,-27119,-15332,25061,-13919,6510,11402);
cl::sycl::short8 inputData_1(30491,8641,-6445,-4405,6179,-28884,-3663,-20753);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<618, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(16355,17219,52675,44480,41344,61186,16570,54849,30912,62655,43524,40521,49905,22169,57575,56485);
cl::sycl::ushort16 inputData_1(51462,46094,1069,1169,54635,15040,40537,9578,11678,31869,22426,39757,23292,31091,19897,47549);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<619, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-30515,-31115,21,-18516,10260,747,-24156,-5740,20657,-6884,10271,-22750,-20363,6567,-20015,8001);
cl::sycl::short16 inputData_1(22571,2362,-31819,17019,-14801,-5992,-13904,-22294,9626,15392,-29973,-8241,15589,22110,31145,-31971);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<620, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(7319,2247);
cl::sycl::uint2 inputData_1(3803,62385);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<621, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-18146,-27281);
cl::sycl::int2 inputData_1(-15074,7826);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<622, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(59760,57924,58345);
cl::sycl::uint3 inputData_1(50277,51819,13928);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<623, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(-1520,-23482,-10804);
cl::sycl::int3 inputData_1(-4154,12202,9314);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<624, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(29875,21269,46919,45170);
cl::sycl::uint4 inputData_1(33377,852,59073,38382);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<625, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(-20345,-7578,-24295,-30344);
cl::sycl::int4 inputData_1(17744,-6176,-2452,11836);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<626, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(26880,29071,25369,26686,47377,1320,14454,42281);
cl::sycl::uint8 inputData_1(42577,57273,4743,4546,4589,61854,5161,12083);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<627, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-23297,-17009,11297,-11526,12074,13885,28446,-16150);
cl::sycl::int8 inputData_1(8442,252,-25424,-24595,-16555,-12462,-20733,-1765);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<628, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(13814,32494,43657,52915,24995,207,63966,23935,61979,18377,53675,6624,1225,11932,37514,11275);
cl::sycl::uint16 inputData_1(23455,18463,2543,54935,47824,59652,29826,33270,3724,5654,35468,14452,8690,38509,1698,43153);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<629, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-21561,-25062,-8347,22493,19977,8745,3423,-10772,-17485,-30051,24599,21260,15813,-13245,-1788,17386);
cl::sycl::int16 inputData_1(24977,29793,-9182,-5925,16940,-21267,-19481,-18511,27017,23121,20959,-26240,25326,29614,-31274,32629);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<630, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(854403739,1086399958);
cl::sycl::ulong2 inputData_1(1336220263,2035575314);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<631, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(1021559978,-932645340);
cl::sycl::long2 inputData_1(1360920002,-473595144);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<632, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(47186480,3285803003,2108352903);
cl::sycl::ulong3 inputData_1(3292794625,172240981,1700097666);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<633, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(331065907,1787670639,20329898);
cl::sycl::long3 inputData_1(1862026485,310245854,-160339253);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<634, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(1360490118,3240722049,3317667156,88488422);
cl::sycl::ulong4 inputData_1(1424248536,1343201365,976787357,2360913166);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<635, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-695986230,-1473887420,1074450386,-1495288845);
cl::sycl::long4 inputData_1(-708110010,1686151820,1188307645,-639679970);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<636, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(1837053496,1572670612,3005294715,4205448331,3318881258,6740672,168375743,1620600990);
cl::sycl::ulong8 inputData_1(4061066811,2493898800,1645868406,3597958580,2212220052,3207089753,2364063365,1034645031);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<637, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(818405641,-1272863466,-1638292769,485347737,-2099230321,-153511879,-1011813666,-445697816);
cl::sycl::long8 inputData_1(-387583635,-311455893,-1750545093,129040257,-1011297848,1772829625,-1747105101,1501181901);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<638, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(2656005582,1500158679,2307034065,2330513225,2407316605,3796070765,2417678631,288279695,3157174268,360006237,3360263241,2131774949,716953374,1353946121,2897471050,3593260092);
cl::sycl::ulong16 inputData_1(2677949001,3310684179,818958023,2543933708,651883751,3934623971,4199428343,21114722,2948219251,349641794,769400175,3012391640,1124314070,197920479,3555736915,539505453);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<639, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-1590845022,1651818362,2026242899,965838742,-840748408,510967295,379108815,1346157493,559743220,640094470,-1784444517,272161283,948640620,-1729773048,-1515979567,-1281643283);
cl::sycl::long16 inputData_1(-1093419098,1154317650,-1691206828,-165699246,-1261849613,479683590,259034708,631708994,2135080563,-447073006,-348692486,-1764634613,118084374,-1864278951,-469267904,1588913701);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<640, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(12468597746779790248,13147085652951328083);
cl::sycl::ulonglong2 inputData_1(16323777530882724477,10208381428899765719);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<641, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(5942014344065967904,2110055110692087642);
cl::sycl::longlong2 inputData_1(-8891583831702132702,5140915917852636677);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<642, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(17115823200992285526,17597079749647273276,11702371035478862647);
cl::sycl::ulonglong3 inputData_1(9075349355605540689,4524206972996433935,14783391933833574761);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<643, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-6942348289015392260,1564747043990986415,-846037542567447318);
cl::sycl::longlong3 inputData_1(-9108392639297938305,2863819061942317752,-3363770353162118888);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<644, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(400714765537140314,4470653785876674056,12887681199118776903,13324556520635933786);
cl::sycl::ulonglong4 inputData_1(17758946747958512386,7661948509944250450,7651479830766558708,11273660034399006185);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<645, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(4247107677314852375,995128517586657629,-3828010376789560025,5452884378543115236);
cl::sycl::longlong4 inputData_1(7057358270272516195,-3902301165068474416,-713026423170945380,-223181843767057365);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<646, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(6118235610760871452,522419655484921039,3790621188259409554,14813484194683743619,981914750651756264,17215776490458178434,9012144252004968653,9779985105179805541);
cl::sycl::ulonglong8 inputData_1(10811421368440754068,7168240605296549297,13269448464325447352,8049771724530869296,7304831583163985838,16697997628021422146,225974417586792645,9874429167475045712);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<647, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(1377879701202798531,-4292751098415757860,425264189122661171,-4919766654444634125,632005405867204627,5639659581127268702,-3573559221983688413,-6339566113874382379);
cl::sycl::longlong8 inputData_1(-5410051862440953799,-7492675789087121142,3844784353547819233,-6629090027030497146,4864617399949551685,-4866626897147611391,4416682397540770724,-8370703220039358204);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<648, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(2244186219031136235,15203066436101918393,3313569033064317336,6364501911920672699,3810535983379518498,7541280827179047531,14658889032914669508,14103472413595627351,10635495539446220738,15187972160159156709,11032802141531792406,207821664198544532,3749392985228060404,17777350898038366717,16456336060968384731,14054412195194065045);
cl::sycl::ulonglong16 inputData_1(16526147817635889895,12959875791796370238,4372730743478856789,17632514864627046026,10594002827160958128,12988615431184813480,14431544081471388802,2105707190759050073,3743109913320612839,6869438654965485374,4542620511622554158,9800470196440641653,17055570120713787989,16347011384561510538,18152994152258658940,4085474423144373826);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<649, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-2453683066108957504,-6182938557309636023,-2725944267904151405,4871296022396301042,9132573758595587977,4759642612650232637,6629492090462137859,6562856901167671545,2949796156779876989,3467220366931838845,-2129390430094987279,3728507044488429167,-3851142138651081539,-2226105800957407541,-1591180550175869937,6383159097059173938);
cl::sycl::longlong16 inputData_1(-3841332354323024486,-8021434980648108417,-2008947154245573671,8599134316652542450,456653899046869741,5664549386434603281,8730785498601801024,968359513110278119,751953478761599492,-6887858596096812647,1136930331520672661,-1843328073044355795,7407599417999575867,5569176013311302421,-4399886390869919133,-4408515889430542751);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<650, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(184,70);
unsigned char inputData_1(235);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<651, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(41,97);
signed char inputData_1(-103);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<652, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(17,115,198);
unsigned char inputData_1(51);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<653, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-9,-40,71);
signed char inputData_1(-4);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<654, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(238,197,5,71);
unsigned char inputData_1(182);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<655, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(102,-119,4,-28);
signed char inputData_1(-124);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<656, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(85,89,99,200,69,229,255,125);
unsigned char inputData_1(202);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<657, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(19,-82,29,-127,100,-63,-85,127);
signed char inputData_1(69);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<658, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(146,245,221,198,42,203,29,73,68,221,185,53,33,147,175,56);
unsigned char inputData_1(240);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<659, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-61,85,-14,-48,91,120,-6,12,-50,30,-64,42,105,8,102,73);
signed char inputData_1(-113);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<660, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(57509,48553);
unsigned short inputData_1(30450);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<661, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-1362,18378);
short inputData_1(-10276);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<662, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(18976,37696,1053);
unsigned short inputData_1(27332);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<663, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-29652,-23841,18685);
short inputData_1(8122);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<664, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(13361,56743,35673,16654);
unsigned short inputData_1(31617);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<665, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-9897,7816,32616,10417);
short inputData_1(-5581);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<666, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(36633,35893,44835,45796,35343,30730,37829,16877);
unsigned short inputData_1(26801);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<667, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-25020,-4129,9654,-2616,2624,23699,5263,-31564);
short inputData_1(18712);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<668, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(19895,25223,63369,55372,20464,15238,26201,14129,49352,4498,48,28744,197,42184,45960,3724);
unsigned short inputData_1(60737);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<669, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-14876,32158,25933,-3993,-7821,2848,9164,-25138,-31823,30132,-5488,-18976,20115,-9196,18746,27916);
short inputData_1(4507);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<670, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(33543,48044);
unsigned int inputData_1(3209);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<671, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(25630,-18166);
int inputData_1(-18997);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<672, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(21085,37019,31658);
unsigned int inputData_1(53944);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<673, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(13798,-586,-16362);
int inputData_1(26031);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<674, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(19793,37545,45652,34863);
unsigned int inputData_1(48793);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<675, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(-26693,-23849,-16193,15148);
int inputData_1(-1825);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<676, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(5772,11804,54619,56017,21687,14207,36185,16794);
unsigned int inputData_1(25024);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<677, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(1456,15515,28196,-2134,30732,-30567,10876,-11570);
int inputData_1(-23511);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<678, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(35002,29862,13231,45438,1448,10409,64734,7413,36146,24729,51367,2270,14633,4041,16882,31765);
unsigned int inputData_1(1265);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<679, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(860,-25568,8098,5653,-22526,-17395,-28656,30575,28382,-29912,9913,-5052,8626,22891,-20458,14851);
int inputData_1(31235);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<680, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(2270522601,2825242384);
unsigned long int inputData_1(2143342813);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<681, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(1936258400,-14078118);
long int inputData_1(112408492);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<682, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2402261129,1797631039,845510515);
unsigned long int inputData_1(1244637182);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<683, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(1156215923,-1808401893,-478605911);
long int inputData_1(-2067693606);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<684, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(4082436244,219156259,70937059,3664454500);
unsigned long int inputData_1(2687008000);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<685, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-876530651,-1941217805,-1190807856,962239460);
long int inputData_1(1288316110);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<686, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(2963799836,248595065,3443733963,3543371800,1308819982,1064273502,3141359920,1519616829);
unsigned long int inputData_1(3838326572);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<687, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(510157866,-93957598,603098951,285154942,1459357634,-185550751,1878271846,-1030416010);
long int inputData_1(-1725610260);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<688, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(4050763444,1509174974,2229830156,1250953170,1610148675,2624376926,2408902741,2263044893,2456333121,2699880669,383335157,3599922362,949600820,2137640320,1801175891,883989031);
unsigned long int inputData_1(4138834228);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<689, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-960464853,-1955521558,-1630336096,-606898059,-1432059372,62494013,2039552486,-743678306,2106625385,-2010241465,96337831,1044507231,1856608828,731227219,1679496847,-1663543823);
long int inputData_1(-149346104);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<690, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(6030011924616659483,7726935153801348771);
unsigned long long int inputData_1(17072325136355225821);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<691, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(964242795805883771,339279068383428050);
long long int inputData_1(8667175614363069542);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<692, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(9940746158765064761,9909944749190745217,15609536369878310944);
unsigned long long int inputData_1(11697312221255777221);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<693, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(6578315919309791277,-7509784589495887867,-8939330447636796160);
long long int inputData_1(-8174766646577935173);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<694, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(10051861490009499805,14526783277888494921,564953466617381525,12644586876657621645);
unsigned long long int inputData_1(12243041327365439481);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<695, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(4549124173483381138,472260180122822900,5899641571339928322,-8118144873192339086);
long long int inputData_1(-4013201791719724301);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<696, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(3484650872374265707,7476000316375908610,835116077728381650,6875181418707911599,10704564742902791524,11003963521907700706,14397665476607022821,6810872937307139722);
unsigned long long int inputData_1(15411982528685741106);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<697, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-7037695720001131734,5058859556495652560,2448145253716099154,1691270267435473681,8019892295947433492,-9164296128692382053,-5571913634645982058,-8656941929575141248);
long long int inputData_1(6085680923176401739);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<698, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(14287491954883794555,14059431811823795500,13021472105079021895,2891967634408747796,9039980857401480920,12802097201369809095,4839661986636477882,7360924624361400915,5426324847023217149,3652083689004442933,1188600467159121619,3880447393804999774,15583954618231163068,17505392692861160256,3956458765504181432,16027593438431688895);
unsigned long long int inputData_1(7117854679084158396);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<699, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(276283122103876638,-1313105894550710633,-2671982496051822701,980800207068142507,-2490693922141924390,5753827057225219566,2212148462388409172,-7877750674235820981,1937959108982315153,358625517944648682,5079054714239480900,102461148878812800,-1748257259445312003,-4870571745396284270,4289803683847600289,-4679202206547702406);
long long int inputData_1(-2645013570760775639);
return cl::sycl::max(inputData_0,inputData_1);

});

test_function<700, unsigned char>(
[=](){
unsigned char inputData_0(36);
unsigned char inputData_1(43);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<701, signed char>(
[=](){
signed char inputData_0(54);
signed char inputData_1(-40);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<702, unsigned short>(
[=](){
unsigned short inputData_0(51276);
unsigned short inputData_1(25954);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<703, short>(
[=](){
short inputData_0(-3929);
short inputData_1(-24444);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<704, unsigned int>(
[=](){
unsigned int inputData_0(57762);
unsigned int inputData_1(29958);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<705, int>(
[=](){
int inputData_0(-10707);
int inputData_1(-20590);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<706, unsigned long int>(
[=](){
unsigned long int inputData_0(3746732801);
unsigned long int inputData_1(4284278040);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<707, long int>(
[=](){
long int inputData_0(-1983804690);
long int inputData_1(-1027522937);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<708, unsigned long long int>(
[=](){
unsigned long long int inputData_0(18231387884711122740);
unsigned long long int inputData_1(11048583921944439289);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<709, long long int>(
[=](){
long long int inputData_0(7855546933705226737);
long long int inputData_1(8645632278630056000);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<710, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(157,242);
cl::sycl::uchar2 inputData_1(145,48);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<711, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-56,56);
cl::sycl::schar2 inputData_1(14,-29);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<712, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(12,226,166);
cl::sycl::uchar3 inputData_1(164,226,197);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<713, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(94,102,-68);
cl::sycl::schar3 inputData_1(-106,86,-40);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<714, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(81,154,140,117);
cl::sycl::uchar4 inputData_1(237,14,142,81);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<715, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-5,91,-10,41);
cl::sycl::schar4 inputData_1(-111,86,84,119);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<716, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(183,38,140,34,153,86,73,99);
cl::sycl::uchar8 inputData_1(122,238,213,199,147,234,115,44);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<717, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-22,83,-24,-93,27,64,-123,37);
cl::sycl::schar8 inputData_1(-92,80,46,116,66,3,-81,-70);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<718, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(194,255,173,82,254,130,153,49,94,147,139,98,0,12,86,246);
cl::sycl::uchar16 inputData_1(225,225,110,105,42,226,104,157,172,119,58,175,194,216,76,52);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<719, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-100,-77,69,-20,99,60,-72,10,122,2,116,-124,0,-20,-13,44);
cl::sycl::schar16 inputData_1(3,-17,73,9,44,-126,-55,-48,61,65,98,-66,93,-63,-85,35);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<720, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(14924,57718);
cl::sycl::ushort2 inputData_1(55412,19054);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<721, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(31295,-2);
cl::sycl::short2 inputData_1(-3464,-29427);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<722, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(24231,3167,64721);
cl::sycl::ushort3 inputData_1(13403,49777,55127);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<723, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-30865,-13159,12825);
cl::sycl::short3 inputData_1(-16566,6567,11672);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<724, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(42608,10305,653,31859);
cl::sycl::ushort4 inputData_1(59010,9141,19481,61577);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<725, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(32090,-26620,30792,-30129);
cl::sycl::short4 inputData_1(6741,30281,-11215,-15996);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<726, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(31409,11632,11885,45523,54496,48687,4870,23050);
cl::sycl::ushort8 inputData_1(61388,19783,60530,19727,23548,25180,3436,10738);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<727, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-8145,-14131,-29150,-10944,-4459,26136,9117,28942);
cl::sycl::short8 inputData_1(-21969,18373,-16863,32289,27972,-32702,-5839,17974);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<728, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(22205,15889,46794,55865,62745,37203,60862,63833,57674,56598,10239,36425,45793,41431,56756,1127);
cl::sycl::ushort16 inputData_1(20132,6307,4467,35619,38480,10637,65251,39701,37899,39182,41,29759,7384,10581,2598,14902);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<729, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(2387,428,-22102,32761,-829,-27473,12656,24867,6457,31533,-1722,-25926,9670,24782,31460,23810);
cl::sycl::short16 inputData_1(2656,2376,16216,28216,1681,31767,-8247,415,-12237,3494,26078,21751,32169,-9675,-16883,-3639);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<730, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(34004,46101);
cl::sycl::uint2 inputData_1(40433,2184);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<731, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(9699,-21882);
cl::sycl::int2 inputData_1(18481,-5809);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<732, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(45308,4167,14896);
cl::sycl::uint3 inputData_1(43830,46573,11925);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<733, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(11638,-30119,32148);
cl::sycl::int3 inputData_1(10083,-14927,3496);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<734, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(23856,12693,54808,48115);
cl::sycl::uint4 inputData_1(29236,13709,7379,23107);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<735, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(-19571,-21675,17346,-9484);
cl::sycl::int4 inputData_1(30535,22535,-12501,-13669);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<736, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(51359,57062,42624,37821,41901,42435,57079,40348);
cl::sycl::uint8 inputData_1(27683,29730,3640,27594,26074,12322,2208,45366);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<737, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(30571,8490,-14328,5534,17177,21183,-3548,-12862);
cl::sycl::int8 inputData_1(11655,4619,13196,2804,21891,15771,31116,10839);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<738, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(8996,30016,44808,16078,20433,52336,46289,2372,56374,23490,63004,36113,42628,58727,41823,58280);
cl::sycl::uint16 inputData_1(964,6350,60475,56737,12922,3924,17194,50655,9173,49771,8454,59765,23380,39304,29321,23085);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<739, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-8409,10403,-17395,261,-31978,23622,-18595,7678,-26316,26144,-30230,19719,-27791,-14025,-10404,-9907);
cl::sycl::int16 inputData_1(-27440,-26574,-21877,3297,24432,-15113,10238,26191,15866,-24710,23257,-4152,20084,30090,-25455,31670);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<740, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(3044704179,2130651508);
cl::sycl::ulong2 inputData_1(3281311097,722897645);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<741, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-1750369368,426503945);
cl::sycl::long2 inputData_1(-2074252763,-719534026);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<742, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2331470520,1436103400,2473920111);
cl::sycl::ulong3 inputData_1(847286707,1929197906,847130063);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<743, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1843305886,-946260505,1082456918);
cl::sycl::long3 inputData_1(1258339439,1275779371,-1076062878);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<744, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(3742614035,1398436239,3430577380,1440391953);
cl::sycl::ulong4 inputData_1(2829002354,820223279,596984074,916605716);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<745, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-1674813178,-521347442,1688492324,-118569409);
cl::sycl::long4 inputData_1(-1031231043,855656928,-2023354106,1742999364);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<746, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(1044692888,2634875018,2845533063,3605354598,2748273454,1248728367,2148757624,1798127815);
cl::sycl::ulong8 inputData_1(3809940029,3840775625,1015328251,1612643522,681356824,299832251,1666788874,345550425);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<747, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-2112202210,843788832,1479806084,2034504888,1486121708,-906184179,1324592637,-1306719440);
cl::sycl::long8 inputData_1(-221004925,-1755475750,605128817,-1449670397,-611369465,281376084,-832457060,-634700570);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<748, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(2904650079,164522124,4208274757,3249265545,1960133366,699595341,2696490376,363406387,2464361513,545351428,3609843365,3471429004,188840845,1328994642,3859312440,408163500);
cl::sycl::ulong16 inputData_1(1353482353,2265655545,4171639360,1240157381,147867160,2187401312,2405279645,3348205085,1496494089,2447362767,2066099264,337661072,3901320489,3431780177,2679461266,219562536);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<749, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-45196754,-1413617029,1941329366,-1085865568,2146407087,1817945883,1339020314,-1945221251,1253913226,684264782,1836462564,-1640010461,1437404901,1375264221,1768268052,-362637020);
cl::sycl::long16 inputData_1(1723788865,-386885476,-385897273,605370661,-586472347,1702429442,574954753,-1587664813,-1526868269,-890276633,200278392,-1345909352,-1065377379,-1339036096,-2138253395,-1688093102);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<750, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(6268883853216694642,3790717068598245436);
cl::sycl::ulonglong2 inputData_1(405015804914994910,13105222705017975129);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<751, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(1357451280316456006,5451403007673187497);
cl::sycl::longlong2 inputData_1(-4703564933881002591,5807357614341219335);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<752, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(6797131796986143202,981316359301845845,14496709247726327464);
cl::sycl::ulonglong3 inputData_1(10001913942136340419,4351275503161273613,17526978967780960587);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<753, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-1502372149972787128,2385162706923369233,4433576325229907538);
cl::sycl::longlong3 inputData_1(-5418832981530151647,-5945082934085039800,3748882049949363296);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<754, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(14899685492490646182,3879045203823422196,17444458732503727056,14350267731920044483);
cl::sycl::ulonglong4 inputData_1(17503738134648924924,2318355510526402115,3413949367809290599,4310295685823870033);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<755, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-3124211301003218832,-2576089265210113715,-1119021538122268455,4310150711178599900);
cl::sycl::longlong4 inputData_1(3047124020296623403,-6850411367655631344,-2649999796137840104,1218990308508750231);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<756, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(18214149612958153603,9269393461480096081,13219337206312609380,10576612768572046135,4977149204350592058,2803882945904516045,4461441845434826723,940044167242435268);
cl::sycl::ulonglong8 inputData_1(10580076595818635244,2006974865053677584,16770615335571922915,15544031941156564798,12747274676923103697,4517598041860894315,15552565430834678449,5596425933822577446);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<757, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-9109242410175509875,1461687612191360029,-2294462100364967730,-7306851882852864759,-2940867003115485920,-1099689745969191911,-4727012592043751655,-6800281024392740413);
cl::sycl::longlong8 inputData_1(3704200267168152163,29031561337112080,4699074678102708785,-9159760759188542593,1976293484776001105,372433554856332756,6735630720248593222,8936644421964692191);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<758, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(4615128810213463707,2130516737630595880,3835771129875751971,13878929857529970326,2156015068728200620,990693062828453044,9002845597027597381,18358405366300998444,1839328494316336625,12308192652175028510,2346465208482157898,5994154096327471788,10444687246336612470,7404576437573070709,16072151055185863836,158475663202317968);
cl::sycl::ulonglong16 inputData_1(11792891715428916398,13910451890135149387,12246567275124900991,593732582666569690,11146244783984480496,15674671297395530880,2223896740064510186,3088005371859510086,1219656153578420176,16513057827862949715,1509233941291776612,3564298116826607530,505009691302535333,2028014492845951021,10318674352377229634,3246091437551430435);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<759, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-7473803953076790296,-3130282413480703444,4490383706590242835,8982972088267205210,1705423502351357621,-3238772515234775001,2275354173170345006,5137202272294673293,4653273717614880181,3748689997021804161,1003338823472284467,6342544130348938424,6530087007252285679,6732005531767316113,-4373914070722924516,8651483042774723275);
cl::sycl::longlong16 inputData_1(6573476741553850421,483859794685332977,-1731839422749930006,-2641901984517854311,-2780774069385194848,-8182566199895577706,-8738442485462144878,-746865531293472737,3802536795035930972,2353041125604821211,595357043253175360,-3067617756344675268,-3023419555557401864,1069522497524391687,8944029729682222816,3821913848879998833);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<760, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(58,225);
unsigned char inputData_1(253);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<761, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(1,-71);
signed char inputData_1(-66);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<762, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(213,186,31);
unsigned char inputData_1(140);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<763, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(24,3,9);
signed char inputData_1(32);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<764, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(125,74,174,204);
unsigned char inputData_1(88);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<765, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(48,65,116,-10);
signed char inputData_1(-44);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<766, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(234,187,238,37,152,77,43,188);
unsigned char inputData_1(51);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<767, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-21,-44,70,-90,28,-125,25,-121);
signed char inputData_1(-127);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<768, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(139,196,61,214,23,7,42,94,131,73,172,108,2,202,22,81);
unsigned char inputData_1(39);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<769, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(67,81,77,119,-97,107,48,-98,122,-100,-102,43,-67,-100,110,98);
signed char inputData_1(-80);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<770, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(20427,40537);
unsigned short inputData_1(37002);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<771, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(24273,5096);
short inputData_1(7670);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<772, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(60684,12234,1035);
unsigned short inputData_1(21734);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<773, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(27916,10889,-2483);
short inputData_1(-28461);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<774, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(29920,2247,12726,13895);
unsigned short inputData_1(31232);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<775, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(18225,-25262,13736,631);
short inputData_1(12099);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<776, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(805,61697,6063,25963,9223,44591,19172,16091);
unsigned short inputData_1(7826);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<777, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(20788,-19219,-15728,19641,-7168,12724,18607,24572);
short inputData_1(2246);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<778, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(14667,53594,38641,36538,10159,40807,39240,23525,48602,29586,15224,44778,59536,53880,22117,50450);
unsigned short inputData_1(52313);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<779, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-31815,-22868,-8042,12757,-9021,-14167,3045,-25851,23028,29386,16085,2387,7053,16348,16886,-5332);
short inputData_1(-23922);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<780, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(16115,1361);
unsigned int inputData_1(34831);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<781, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(18031,-24126);
int inputData_1(14303);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<782, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(28358,33894,17795);
unsigned int inputData_1(3138);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<783, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(28065,24473,-1678);
int inputData_1(-29490);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<784, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(37038,26317,35163,27977);
unsigned int inputData_1(3499);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<785, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(24189,-27715,-19553,15755);
int inputData_1(4658);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<786, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(45320,18206,29646,43313,43899,53480,57823,60240);
unsigned int inputData_1(51938);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<787, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-10561,-14848,25792,-26781,-6755,496,-1588,4643);
int inputData_1(27833);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<788, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(4874,7996,17153,64652,39374,51273,50907,36378,53771,37413,36099,2792,21543,33839,11472,3819);
unsigned int inputData_1(64411);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<789, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-17736,2814,2968,19719,32179,-892,-26216,-26303,-19676,-26875,-24854,25864,31533,-23196,-1924,16456);
int inputData_1(20019);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<790, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(569656475,2598696528);
unsigned long int inputData_1(1118155248);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<791, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(2053634596,806867666);
long int inputData_1(-1331906854);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<792, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(3174371074,178727092,164969833);
unsigned long int inputData_1(1198805887);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<793, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(976649583,2127357511,-195027312);
long int inputData_1(527956582);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<794, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(3554762891,3158120268,919092717,3585176304);
unsigned long int inputData_1(2581202340);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<795, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-1835679617,365907836,-1150008332,-318901623);
long int inputData_1(-1259156791);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<796, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(637002373,1179317323,459256415,618365387,4172813941,3700135708,4292172601,2974321755);
unsigned long int inputData_1(2359331341);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<797, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-1665212578,1254766232,-1261460387,-587685524,-1346159579,-435770980,-1042855362,513368663);
long int inputData_1(-1507831007);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<798, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3702495563,4047853687,2899287488,1117378543,1379396698,2573642328,3789960005,3085794269,2635165894,3684646741,3403025826,3010204001,805770496,2141371539,3624402411,1688246159);
unsigned long int inputData_1(280742875);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<799, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-1763581821,866091897,964467963,-1137623581,-1124810035,1425088614,1267162043,-1332609929,-161612196,-1332288448,-1165073164,1504003845,-2093393188,2005704588,-1444160415,252105131);
long int inputData_1(-298990078);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<800, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(12305406650644095510,7837209700042672044);
unsigned long long int inputData_1(18374614153964629990);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<801, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-2941585190664931706,767421739489625707);
long long int inputData_1(-4840571083046024395);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<802, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(1409247271564457512,3518218445908490662,2530701690857095545);
unsigned long long int inputData_1(18234272785901690398);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<803, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-4461667293896449961,8815029282955352405,5674446404303876133);
long long int inputData_1(857412657384744675);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<804, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(6667062351029357137,15267892593650533360,7161771848156464297,1910545865826409950);
unsigned long long int inputData_1(10004741086816233888);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<805, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(1258445920752889866,-2390153014354710049,3299665779324934927,-588451064540606925);
long long int inputData_1(5985061499668803925);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<806, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(10172126869256740043,18257538339317029388,7682580058056814604,3731694865289163063,5166954087284931005,13836687104536869389,13343328582144066592,3355366965646389442);
unsigned long long int inputData_1(8639489509512893912);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<807, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-717359627785360355,812265192883048352,-2871308815982062140,-7326037071147064467,1577136840612077723,-6278159960255962215,-4664735820447523375,-1785294638634931418);
long long int inputData_1(2621133010238055693);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<808, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(13978958077933191589,9240641578823976302,3875929349077674487,17364438767421604,3778404932820072543,9499927762410159600,7500317457222378673,16692426928892729278,4992114155115405718,5710569042734411359,9476502086727481798,17210828403595459862,11459569660535400789,6267097122755649720,2686178886414640533,17668237307442045943);
unsigned long long int inputData_1(10657534362335691193);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<809, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(7863166018889752267,5459664016426211504,-3527094427989633666,-243923744120871375,-4049300536172373802,-3803178006111609879,5411325386551658620,3993844743844990725,-5704621396473569180,-7903507833401104525,7991081259170460528,2349131577564448211,-5007140869488436422,1994533530642120454,-7664044467472151353,-8432913918482352751);
long long int inputData_1(5020647587615099227);
return cl::sycl::min(inputData_0,inputData_1);

});

test_function<810, unsigned char>(
[=](){
unsigned char inputData_0(38);
unsigned char inputData_1(68);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<811, signed char>(
[=](){
signed char inputData_0(-91);
signed char inputData_1(8);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<812, unsigned short>(
[=](){
unsigned short inputData_0(3121);
unsigned short inputData_1(11797);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<813, short>(
[=](){
short inputData_0(-14661);
short inputData_1(31972);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<814, unsigned int>(
[=](){
unsigned int inputData_0(8338);
unsigned int inputData_1(49710);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<815, int>(
[=](){
int inputData_0(19810);
int inputData_1(-31915);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<816, unsigned long int>(
[=](){
unsigned long int inputData_0(3241216942);
unsigned long int inputData_1(2491169279);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<817, long int>(
[=](){
long int inputData_0(-1791833375);
long int inputData_1(303564742);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<818, unsigned long long int>(
[=](){
unsigned long long int inputData_0(11396417126072631807);
unsigned long long int inputData_1(17389030327506987798);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<819, long long int>(
[=](){
long long int inputData_0(8208033156010020131);
long long int inputData_1(2165820894694198600);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<820, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(223,175);
cl::sycl::uchar2 inputData_1(8,69);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<821, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(88,109);
cl::sycl::schar2 inputData_1(-53,-114);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<822, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(79,95,159);
cl::sycl::uchar3 inputData_1(236,49,63);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<823, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(118,-60,20);
cl::sycl::schar3 inputData_1(-112,-25,53);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<824, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(67,183,141,81);
cl::sycl::uchar4 inputData_1(119,202,88,13);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<825, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-95,65,-62,69);
cl::sycl::schar4 inputData_1(-59,-61,75,-31);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<826, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(36,58,254,167,132,169,13,50);
cl::sycl::uchar8 inputData_1(104,212,8,57,200,160,31,85);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<827, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(18,-91,-111,125,91,-57,66,48);
cl::sycl::schar8 inputData_1(-127,33,-77,-102,57,29,99,17);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<828, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(2,27,228,203,21,59,113,11,129,119,215,255,128,226,220,149);
cl::sycl::uchar16 inputData_1(159,122,243,160,185,21,69,87,22,235,232,40,162,203,198,11);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<829, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(46,-82,116,1,-72,6,8,-78,-76,118,124,-92,-89,0,-75,7);
cl::sycl::schar16 inputData_1(6,-52,-66,113,127,-39,-103,-102,10,-109,33,27,52,4,90,-69);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<830, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(6372,2478);
cl::sycl::ushort2 inputData_1(45039,52097);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<831, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-5831,-10220);
cl::sycl::short2 inputData_1(22475,-31249);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<832, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(8052,3390,22546);
cl::sycl::ushort3 inputData_1(58792,39473,24257);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<833, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-26080,19741,-13148);
cl::sycl::short3 inputData_1(4455,19758,32309);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<834, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(3384,41048,8734,20137);
cl::sycl::ushort4 inputData_1(3441,7855,15229,36956);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<835, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-29000,23109,-28391,-24899);
cl::sycl::short4 inputData_1(12378,-29129,-5415,-14776);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<836, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(25444,28582,55672,57921,24423,21501,25318,51512);
cl::sycl::ushort8 inputData_1(44587,14089,9277,54792,54932,28284,31824,56467);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<837, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(-427,25,9150,21936,24483,20926,22920,-24399);
cl::sycl::short8 inputData_1(1122,21353,3077,-4014,-8984,-5273,31198,-17054);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<838, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(65363,20890,41681,9683,33066,21019,58341,62759,7601,8734,37329,36855,44939,3911,49539,11941);
cl::sycl::ushort16 inputData_1(54981,45730,2803,24473,575,18250,45880,3160,37297,24996,57551,5228,17269,30672,55866,10980);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<839, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(-16988,-1225,20428,11242,-2351,21966,-20835,4487,19287,-19688,-9298,-8418,29475,-20208,-15225,-26339);
cl::sycl::short16 inputData_1(16931,29187,-27320,25486,1736,-5390,-6904,-25687,3107,26559,-32269,26295,16162,28079,-973,17915);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<840, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(48806,52509);
cl::sycl::uint2 inputData_1(40822,50345);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<841, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-29945,8188);
cl::sycl::int2 inputData_1(-2028,-6848);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<842, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(18284,45348,56466);
cl::sycl::uint3 inputData_1(57726,56472,30626);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<843, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(-8330,-26720,-20108);
cl::sycl::int3 inputData_1(30482,-22441,-24849);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<844, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(44268,6195,42095,44129);
cl::sycl::uint4 inputData_1(22885,8113,63829,19482);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<845, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(74,-30040,14742,2169);
cl::sycl::int4 inputData_1(13582,31669,-23367,-10163);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<846, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(34295,27637,181,42165,10984,30528,2571,42381);
cl::sycl::uint8 inputData_1(46565,43515,11219,9259,56443,14658,36093,4543);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<847, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(13209,-1609,1544,17205,5054,-27277,-8040,5097);
cl::sycl::int8 inputData_1(-20736,-23265,5321,-7614,18882,-32235,4645,-2946);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<848, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(54519,57802,50138,40808,5840,35066,51391,4072,5600,21169,23495,16674,4296,42079,65481,40157);
cl::sycl::uint16 inputData_1(37830,46383,59847,28028,47424,16726,36027,45908,37458,37552,33289,16271,14373,54620,52320,36208);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<849, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-4044,-9771,-6029,-31545,-25493,2070,690,-14548,-22951,-13538,16010,-32195,-23775,4226,-19080,-30263);
cl::sycl::int16 inputData_1(-29021,-30122,-29935,9200,-3686,-16073,-25805,30364,-23917,-8442,30997,-30608,-9003,3776,18334,-2311);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<850, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(2920253629,2980439398);
cl::sycl::ulong2 inputData_1(450180132,3841100692);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<851, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-1771588631,-857907331);
cl::sycl::long2 inputData_1(98635515,-299105778);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<852, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2446436079,830971561,1648027600);
cl::sycl::ulong3 inputData_1(2560743275,3356301631,2919735793);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<853, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(54299427,-1143885746,-318179571);
cl::sycl::long3 inputData_1(1752548848,443689897,-980851440);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<854, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(1198300308,992817882,20593897,825641654);
cl::sycl::ulong4 inputData_1(1345567690,2130852473,1074012486,3378009938);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<855, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(1999634439,789436509,-1393256723,1859941745);
cl::sycl::long4 inputData_1(-807830879,-1754601579,1579261712,-2068055577);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<856, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(1546080269,4198752742,311538668,769873842,2757136964,3646888912,3710166713,4267145798);
cl::sycl::ulong8 inputData_1(1197798267,2229728365,3291001941,2081708365,3047475100,870966456,348259481,216941081);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<857, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(261174954,379034410,-2084133350,862908751,1647857452,508428585,463668199,2044212597);
cl::sycl::long8 inputData_1(-116501130,1749438579,451787749,-1194679405,1444766803,-580606244,243789445,-1769680997);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<858, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3177938154,2374106609,1463680172,1789425714,4201899150,1554406720,4202303170,300396345,1925949588,3482535944,852401975,3817799163,2276591004,2060654621,2630079441,2081162448);
cl::sycl::ulong16 inputData_1(2771714441,2236992631,882710189,314310131,4078132787,936857454,3062631805,3931792577,963218947,3912637750,3681030379,1408176424,2493965972,1736799690,862507406,4191728004);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<859, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(-1757908097,2132929094,1012762801,-1249584402,1605308793,-1147228153,-1446958799,1350737517,-1801878265,-1355424274,-1745654905,542492522,546530600,1364360496,-2019873419,-1575903913);
cl::sycl::long16 inputData_1(1124738328,-2064563963,-1828782817,1576241885,469376897,-1138998133,-186352520,988373613,2109847893,-44556600,-1348760708,958384378,-833040433,1087713710,-27894853,497388166);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<860, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(3308542510219103759,8065904412981150239);
cl::sycl::ulonglong2 inputData_1(16383800180750582459,10211594884003496584);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<861, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-1948091594406220456,-8917793984063991922);
cl::sycl::longlong2 inputData_1(-3080268384079298306,2595047809877136752);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<862, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(15611747966581177524,14217594360148775993,17106104337232720065);
cl::sycl::ulonglong3 inputData_1(5739554131296671090,8739589566613977257,13265577921961918882);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<863, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-6231705875289097910,7953960981588984520,-1822069871011289507);
cl::sycl::longlong3 inputData_1(-8434726512543648068,-7021181456114547455,8847049234315222498);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<864, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(4850233139747390293,18010727160362291974,9739021573103583859,12242855757575329660);
cl::sycl::ulonglong4 inputData_1(1355089631503920585,9755479329342800381,4255767255535241824,7029779293561396427);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<865, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(1568069008163869264,-6868694915640816640,4033275925283583254,-2430453221775978351);
cl::sycl::longlong4 inputData_1(-796750901428094790,-6965801814771113050,6015106472472057662,-8221817628408129436);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<866, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(6417068587657431568,10256319434061354357,6496505640936137506,8796294075657756238,17543882270638673705,15622243515256478755,14210476668273392024,1004924250950341276);
cl::sycl::ulonglong8 inputData_1(10487575028137855144,14812471476064574705,4231119861351793897,848231318705584633,10588400928051192303,2741826899172337509,17966756939326115109,16821123088219142408);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<867, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(6991475635010189197,1632348011959619093,1821796015403309063,-1416939291386858079,7970651890719855886,9045724733367668070,-7480196386101585946,-460349099036482245);
cl::sycl::longlong8 inputData_1(5273066820224022305,2759847460825297937,-2875597563220546711,-961518629938700770,-8397193777849614077,1706718285042125198,-6542188145057499427,8668169138647235898);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<868, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(14900252457887091871,6297296543784675141,2910197111000401356,5854666015920231835,11866416725508364303,11582519937734501956,10386749468282902636,12671069492306496853,9852012966942379926,5990480180382154280,17486502901011025827,12739125299602097444,6658664326625614971,11091186750319354821,15097291893883246002,17926443134535253804);
cl::sycl::ulonglong16 inputData_1(3918832983842708814,4451977837123516815,12717291011473710220,2092955602628214515,9580514608465145246,1480800075788398369,13307696663057158226,11224461816969067533,341960678566377646,11434436509726511822,12594612615527849937,5302713298162499443,16014564819206292382,1428642333311310433,9497852730563482496,7908712878552505164);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<869, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-7241089214916566205,-3084053651413701243,38745159108576102,3469579907839020929,8313106788433045551,7464332074794288058,4476327826723413543,-2284360043037160155,529610303350083821,952088879924111581,-3504388514408982532,-5777724039782203506,-5081926747748672605,413641858847294940,254905224844685314,4342383262545927627);
cl::sycl::longlong16 inputData_1(1448680994806677507,5560392220698501379,1576730766034253860,-7865100824908340704,4431688766751928155,-6284826857600213287,-7376909625881104409,1748863625388239413,8499039451256013990,-6898684969770192334,-7976183763642075668,3577765447883124082,-5862155433798190116,775600363278828697,-8680821435646577876,8194510618409494476);
return cl::sycl::mul_hi(inputData_0,inputData_1);

});

test_function<870, unsigned char>(
[=](){
unsigned char inputData_0(243);
unsigned char inputData_1(75);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<871, signed char>(
[=](){
signed char inputData_0(-11);
signed char inputData_1(-29);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<872, unsigned short>(
[=](){
unsigned short inputData_0(5455);
unsigned short inputData_1(35990);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<873, short>(
[=](){
short inputData_0(20568);
short inputData_1(-17374);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<874, unsigned int>(
[=](){
unsigned int inputData_0(34776);
unsigned int inputData_1(39383);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<875, int>(
[=](){
int inputData_0(-25406);
int inputData_1(-5363);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<876, unsigned long int>(
[=](){
unsigned long int inputData_0(445511131);
unsigned long int inputData_1(1510967055);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<877, long int>(
[=](){
long int inputData_0(469243468);
long int inputData_1(671366991);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<878, unsigned long long int>(
[=](){
unsigned long long int inputData_0(16610156506204279221);
unsigned long long int inputData_1(5578009465720273728);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<879, long long int>(
[=](){
long long int inputData_0(-2427150282570574314);
long long int inputData_1(8861070222600726019);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<880, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(4,71);
cl::sycl::uchar2 inputData_1(130,158);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<881, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-20,119);
cl::sycl::schar2 inputData_1(112,-53);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<882, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(56,93,90);
cl::sycl::uchar3 inputData_1(118,63,44);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<883, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(-54,-11,-90);
cl::sycl::schar3 inputData_1(48,66,-58);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<884, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(39,147,71,87);
cl::sycl::uchar4 inputData_1(115,115,213,26);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<885, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-43,-126,124,118);
cl::sycl::schar4 inputData_1(-7,-115,-26,-125);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<886, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(13,23,73,12,81,219,240,13);
cl::sycl::uchar8 inputData_1(183,61,99,236,92,141,176,199);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<887, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(8,-115,-60,43,-38,6,52,-12);
cl::sycl::schar8 inputData_1(-68,60,-21,61,-26,-57,88,79);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<888, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(116,95,115,152,60,159,111,116,102,61,2,80,245,165,39,167);
cl::sycl::uchar16 inputData_1(107,48,113,220,246,122,179,193,167,53,9,213,77,32,169,135);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<889, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(13,2,-73,23,-122,-6,-128,14,-68,81,102,-119,40,106,46,-112);
cl::sycl::schar16 inputData_1(36,-112,-89,38,46,-120,20,96,33,36,-73,36,-90,85,-67,81);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<890, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(1583,38997);
cl::sycl::ushort2 inputData_1(21821,19783);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<891, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(7177,-16765);
cl::sycl::short2 inputData_1(31371,-21231);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<892, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(35836,44457,20245);
cl::sycl::ushort3 inputData_1(60173,51116,14235);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<893, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(-3918,3921,-31145);
cl::sycl::short3 inputData_1(17842,15692,-12782);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<894, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(51791,27330,28511,56703);
cl::sycl::ushort4 inputData_1(20062,51163,47773,13208);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<895, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-13560,-22402,7382,-16290);
cl::sycl::short4 inputData_1(27690,-19458,-16680,16836);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<896, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(39849,28178,107,52137,21872,34946,20916,12485);
cl::sycl::ushort8 inputData_1(58602,32583,46948,54103,26388,42150,5972,3251);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<897, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(6881,31500,28949,30225,5433,-3302,14192,7566);
cl::sycl::short8 inputData_1(23788,26871,11019,-12593,-242,15672,14877,-27406);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<898, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(2954,13818,12890,27550,3179,45274,19952,8618,45978,51958,41270,61955,50494,21529,47782,22626);
cl::sycl::ushort16 inputData_1(63921,4981,35828,42724,44811,16080,28518,2371,650,65419,5658,27652,46238,32986,26696,55584);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<899, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(22831,-20851,24785,-30020,-26018,-26110,-10507,-11204,-32525,-18884,2726,-8911,24675,-15749,-25523,-32030);
cl::sycl::short16 inputData_1(-2986,17387,-15453,-27700,-14082,-24599,26958,-31970,-27699,22726,-21697,12555,-11008,-1467,31635,10739);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<900, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(16421,37400);
cl::sycl::uint2 inputData_1(42762,57045);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<901, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-30406,-24056);
cl::sycl::int2 inputData_1(30399,-9759);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<902, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(50878,20417,42070);
cl::sycl::uint3 inputData_1(7049,47224,53300);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<903, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(-22096,-22969,-457);
cl::sycl::int3 inputData_1(6063,15680,-15469);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<904, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(18685,39149,54721,27602);
cl::sycl::uint4 inputData_1(11598,49577,38938,53186);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<905, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(28118,-26894,19307,8116);
cl::sycl::int4 inputData_1(20606,-30860,20971,29179);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<906, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(15452,21102,47475,1021,22009,1189,36104,26915);
cl::sycl::uint8 inputData_1(36479,1032,34325,54389,59012,14088,4051,53028);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<907, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-1697,-14089,-29415,28611,-25394,4718,19792,29975);
cl::sycl::int8 inputData_1(-30185,-24685,1752,30690,-2602,-12372,-30165,-7840);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<908, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(22978,24947,24834,16151,30769,39041,18729,48560,44885,26981,11227,7950,41550,3465,63269,44900);
cl::sycl::uint16 inputData_1(40865,19913,38498,15519,38346,25792,59124,63179,50214,53711,51598,64191,7152,51772,59425,33301);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<909, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(24392,19702,-30929,19904,-1214,7136,-10957,6251,5952,-23084,28,-20842,10730,-29409,-9360,6051);
cl::sycl::int16 inputData_1(30046,-26313,-22643,-6842,-12277,13821,21357,-24302,11245,19429,27112,-10287,-15122,16155,-6651,4279);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<910, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(2992419448,1306260889);
cl::sycl::ulong2 inputData_1(3219306287,4009511013);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<911, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(1006440750,-752006309);
cl::sycl::long2 inputData_1(-115509401,1878640518);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<912, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(2448314181,473376504,2157493024);
cl::sycl::ulong3 inputData_1(1551414407,3893921352,2368211059);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<913, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1346249560,-1989596020,1839285893);
cl::sycl::long3 inputData_1(-1480911083,668692949,-389359686);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<914, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(1566692751,3088367810,1848188070,798325439);
cl::sycl::ulong4 inputData_1(1234359507,1675307631,2501784444,3513538161);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<915, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(-202421671,1909037989,-702695528,-507994930);
cl::sycl::long4 inputData_1(1287068983,-562452576,2144032029,282396177);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<916, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(2784067162,2296547470,1452828465,2958484215,1497786408,377044898,596901459,1512296675);
cl::sycl::ulong8 inputData_1(2766674806,1190480765,1464706299,576842828,2996399973,174758894,4281339293,1197928903);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<917, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(1553310842,831090412,491004422,86916933,191530697,1377796404,-1460491282,-1078768300);
cl::sycl::long8 inputData_1(-1744944677,759369227,-1324431279,1049678630,1370687791,-531753599,1029895388,-2020521854);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<918, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3599637063,3902468982,167518277,1946171953,2277695824,1700062304,348313477,2797481212,3105197782,4020523111,651619368,3940826186,3516053988,273257429,2536462661,4222822499);
cl::sycl::ulong16 inputData_1(2834053275,41860123,3386472059,974536456,2758275591,1151096085,4089391516,1780356501,725628024,1199609897,3554213448,1088079654,4142301217,2533270544,3078196687,420470908);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<919, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(1738855136,1639834866,2010181692,1975351861,1143097691,-670248140,19333571,2027310862,-1985245453,1168192785,428343659,695440797,1437432663,-1157500023,1780769989,-6617980);
cl::sycl::long16 inputData_1(-2089605442,1317742054,-1154711266,1440186332,-1227834419,-1571380981,263134552,-2042663421,-1319608387,1204311283,341022881,1886680249,1614083002,-790952655,2098590156,-878897826);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<920, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(11498929722846901931,16255214514380178076);
cl::sycl::ulonglong2 inputData_1(8267550236668864105,18139880413531215031);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<921, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-7115960909674978219,2966530241170508434);
cl::sycl::longlong2 inputData_1(1447319243836234435,-4416281442050715490);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<922, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(12526982631739979928,8161464446852326256,2269691279273640439);
cl::sycl::ulonglong3 inputData_1(17553157843368116462,5037189480168474524,4467358407774780360);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<923, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-8969861920273732027,-2538835367686980840,-4519794941039259725);
cl::sycl::longlong3 inputData_1(-2516180701222442539,-8829389765726796430,6143009911618112987);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<924, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(18283531363065051887,928732061569357741,16537620239285436305,11576227763592331435);
cl::sycl::ulonglong4 inputData_1(1126361640815649949,16647157803585562527,8196815716377542504,8910578492213303826);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<925, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-427339502622003136,2795778435054728741,457152406074797604,2021272439769534946);
cl::sycl::longlong4 inputData_1(75674428689235304,367748269964645465,-3430342144035004535,5695910456318571277);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<926, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(9025720334703296327,9536581804877091046,5349912252360917454,1080977330192676866,4465317289208241733,6216864683134167356,12177925127143386154,7389823523950904854);
cl::sycl::ulonglong8 inputData_1(16822581141096869784,17206647451899290169,14681846990910496829,17558942144706011720,13482019067520397396,3790734191792229767,860057572211428839,10942145038815447270);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<927, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-7567290084368630740,-4721447112943671354,-2270724134457925111,6135202093159820663,-8025883748882063692,6281623335186587955,-8729698758218461958,-6272262712426053099);
cl::sycl::longlong8 inputData_1(-2690397758306150300,8975889654632733290,1160574112289355379,-2079680075884366402,-9076637664528897399,1975773453719773151,2357207984293193471,-1029404012261165235);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<928, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(6544717914325572482,16740795654463019609,5927123450380969852,13801466915722674583,12453989912994872610,3792531908275082429,11036221764865558237,5620320019641437153,18079623932226151319,11064543197530445257,1614779060390898653,6257728064950668352,14171528265253128149,1115524997771089068,14717854909176823874,5443017967794373079);
cl::sycl::ulonglong16 inputData_1(4830122917384904729,2897304515511531323,13347011861545134952,3501130942780397422,4124455329323452878,6053229605287054670,3056645984943759683,188619293840137555,2209162690649906798,125148925631782856,17173293503230635480,3839633217459427237,15792366977448254327,14829902564948313462,8240881677597158805,3977748282679854333);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<929, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-9217018917573564766,5219769005340958364,-1850246075476575095,-2732865358657207446,-3969260794279827446,3924055775414067184,4560460957785948899,3939575870039926487,-2043025512731142393,-5407951105985898459,115188613763832557,1144667412499943367,166497995773907294,2314522398829901995,2971975924819977651,8804276746840264033);
cl::sycl::longlong16 inputData_1(6170357895228332433,3214099324831776778,2427535955754674419,-2465060651120362978,-2792393172768276934,-2214132045913038536,-1394186978553251494,-7423133754761814417,6869062823806853945,-110960051546242458,7472038486000543560,7441649404534819740,5549873886465187750,-1904382212977216845,-6846158767385603413,-5594663223874500243);
return cl::sycl::rotate(inputData_0,inputData_1);

});

test_function<930, unsigned char>(
[=](){
unsigned char inputData_0(252);
unsigned char inputData_1(138);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<931, signed char>(
[=](){
signed char inputData_0(-116);
signed char inputData_1(67);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<932, unsigned short>(
[=](){
unsigned short inputData_0(31969);
unsigned short inputData_1(14486);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<933, short>(
[=](){
short inputData_0(27530);
short inputData_1(-14404);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<934, unsigned int>(
[=](){
unsigned int inputData_0(5787);
unsigned int inputData_1(41206);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<935, int>(
[=](){
int inputData_0(-15502);
int inputData_1(-1447);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<936, unsigned long int>(
[=](){
unsigned long int inputData_0(361168314);
unsigned long int inputData_1(3526826048);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<937, long int>(
[=](){
long int inputData_0(-1807494862);
long int inputData_1(-636639599);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<938, unsigned long long int>(
[=](){
unsigned long long int inputData_0(6248848330002813889);
unsigned long long int inputData_1(13883551214866157341);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<939, long long int>(
[=](){
long long int inputData_0(6816000491185777551);
long long int inputData_1(8997172856495338034);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<940, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(108,157);
cl::sycl::uchar2 inputData_1(147,51);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<941, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(-122,-94);
cl::sycl::schar2 inputData_1(-98,-66);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<942, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(222,135,179);
cl::sycl::uchar3 inputData_1(17,242,7);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<943, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(100,-123,10);
cl::sycl::schar3 inputData_1(8,-126,-54);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<944, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(220,73,62,132);
cl::sycl::uchar4 inputData_1(53,232,57,215);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<945, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(-51,30,-98,-29);
cl::sycl::schar4 inputData_1(36,-102,56,-76);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<946, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(158,48,226,188,71,201,230,189);
cl::sycl::uchar8 inputData_1(156,209,150,255,114,7,210,119);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<947, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(104,-89,113,-33,62,-10,-118,101);
cl::sycl::schar8 inputData_1(55,29,8,123,-58,-48,88,45);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<948, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(198,126,27,55,78,79,48,114,9,89,227,99,253,255,216,6);
cl::sycl::uchar16 inputData_1(153,182,40,128,17,137,22,217,21,247,33,136,15,68,215,40);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<949, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-68,108,81,-65,-94,106,-40,-44,60,-4,72,-68,102,2,-85,-75);
cl::sycl::schar16 inputData_1(-125,42,85,-113,98,70,62,-80,59,-34,-49,100,85,-125,103,3);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<950, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(15116,26214);
cl::sycl::ushort2 inputData_1(51561,58170);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<951, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-32566,5303);
cl::sycl::short2 inputData_1(24582,-21625);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<952, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(35396,10883,56492);
cl::sycl::ushort3 inputData_1(48097,4899,62659);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<953, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(9991,20002,21256);
cl::sycl::short3 inputData_1(-7924,-22506,24955);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<954, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(14655,58687,10052,60172);
cl::sycl::ushort4 inputData_1(24331,31024,48645,3553);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<955, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(-30107,3844,-28111,10873);
cl::sycl::short4 inputData_1(-31403,-25560,-4422,14525);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<956, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(38918,27592,54383,46365,10971,5711,4546,30009);
cl::sycl::ushort8 inputData_1(58895,55724,25936,24457,3131,16420,58265,15259);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<957, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(19425,971,4432,30888,-29254,-31233,-4453,28827);
cl::sycl::short8 inputData_1(32703,-7039,19536,-11408,-14812,31565,-13653,1121);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<958, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(19298,53054,32754,17419,7609,49165,14670,2205,54240,20562,25610,19986,29414,15497,64271,12613);
cl::sycl::ushort16 inputData_1(52997,51,27877,31870,34123,24865,24853,1372,55923,63599,54230,2982,60016,27605,14009,64073);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<959, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(12945,-9356,4174,-15047,-17619,-21394,-12701,-9846,-30703,536,-31971,-2656,-24136,-24453,26605,-5591);
cl::sycl::short16 inputData_1(31616,8547,-21348,-24285,-5235,29059,17350,-19138,-22398,-24938,-21944,-5316,-12001,14887,17911,-1230);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<960, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(14636,50393);
cl::sycl::uint2 inputData_1(21339,10635);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<961, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(-7529,20929);
cl::sycl::int2 inputData_1(-11111,6245);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<962, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(13998,59424,37756);
cl::sycl::uint3 inputData_1(3525,54234,25296);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<963, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(28333,17063,18887);
cl::sycl::int3 inputData_1(-11342,9697,-7855);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<964, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(33020,55483,218,9519);
cl::sycl::uint4 inputData_1(4961,28400,43504,50010);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<965, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(2130,-24422,-19859,13627);
cl::sycl::int4 inputData_1(31512,-3006,-1017,-15076);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<966, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(57116,21894,41166,5856,56962,11285,25149,11993);
cl::sycl::uint8 inputData_1(59481,10508,17431,24075,46281,50905,57025,14186);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<967, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-28832,-9633,-14119,13883,-26334,4661,-18689,-14680);
cl::sycl::int8 inputData_1(-1986,26278,1058,-30071,-16326,-25849,30294,-10202);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<968, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(55721,42349,59226,25518,57342,14720,40833,64265,1587,6284,24095,14827,60671,36612,13176,9255);
cl::sycl::uint16 inputData_1(5052,42531,56417,2680,50987,27385,2158,12061,52068,47716,26305,45140,49098,21238,32723,62233);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<969, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(23024,14624,6573,-1963,17611,-20104,-13616,12683,-31180,-28903,18699,-17327,14019,11231,-32466,23606);
cl::sycl::int16 inputData_1(17215,-32431,-31345,23701,26076,-3074,-22690,-21483,-4320,12306,3783,16440,-28575,17345,-14857,20216);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<970, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(3529433171,1199480946);
cl::sycl::ulong2 inputData_1(3546382022,33217906);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<971, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(1964241587,1227599348);
cl::sycl::long2 inputData_1(62042818,1416271963);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<972, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(3330768684,1699117237,2298347741);
cl::sycl::ulong3 inputData_1(2698226991,571673490,3656670812);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<973, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-1898700416,-1917986099,1834272005);
cl::sycl::long3 inputData_1(2095018028,520212366,-949525808);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<974, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(22573017,4293326220,771770776,2744789152);
cl::sycl::ulong4 inputData_1(2221414839,584227534,4105964568,1245610974);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<975, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(94881992,1059530506,1882556980,1248893411);
cl::sycl::long4 inputData_1(-89340871,194250939,-184848228,-91204152);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<976, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(1135853308,1498016796,1112847137,1532137035,3506602127,1339604144,1192412341,1016716847);
cl::sycl::ulong8 inputData_1(4150658148,302914504,2204916979,526493861,2051614945,924206765,1176832712,2586866346);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<977, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(-485207042,-1774387710,-653690115,1900146127,-106146647,1441798606,-971192015,2137779334);
cl::sycl::long8 inputData_1(2025114360,2068174876,2010845661,55738640,-1362495130,2093954346,-1895913018,1393660532);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<978, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(3264531070,254348060,3459843646,4248051139,4186326961,3093977568,2001256456,2782540783,4290355854,4102180274,4052406911,2991981837,2961955321,792581191,3354769383,1213856200);
cl::sycl::ulong16 inputData_1(529754382,596379721,1302760503,3629383596,374668868,1688392334,377923822,741742672,2120113954,2529538331,223052562,1536543658,3334217231,2264190943,1548496154,2292328017);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<979, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(685111656,1796580496,-1597412982,249487023,538505063,829481042,15586420,-1042310601,-1422066969,1977200669,940495703,-1015046085,-176816996,-94359633,2090376529,-1791415446);
cl::sycl::long16 inputData_1(1744407473,-1247169738,1099619072,879390694,1088193489,-1663377864,-2077486127,1386149429,860154344,355205337,1332314902,1898025244,213859294,2059073871,1928418619,2030830018);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<980, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(14344367004985669682,2743994203643984507);
cl::sycl::ulonglong2 inputData_1(10776937082906044567,7516329712179988539);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<981, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(-7159872700836010832,5826425473817904008);
cl::sycl::longlong2 inputData_1(4261663968815079703,-292452894649887271);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<982, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(11222425818613743874,5447008542984376601,4123258102727736619);
cl::sycl::ulonglong3 inputData_1(16686763508692590756,15289442731589818025,7381355607131584568);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<983, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(1098312925230429811,6132851225868399719,23486687246825213);
cl::sycl::longlong3 inputData_1(-7268416832687293858,565461625162996179,2497963196786797941);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<984, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(6144755103719283574,10103572156870402580,6462303961462751537,6115354305919030393);
cl::sycl::ulonglong4 inputData_1(5767372598679623112,12068865009390176199,16206042516748898875,9517543568360469502);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<985, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(-6950587705008266022,2134778377705841587,-2386481911069376956,-4429131933325285741);
cl::sycl::longlong4 inputData_1(-3922918353527958865,5714929294457420181,5128500891908319331,4127790410641699568);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<986, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(8396627131205287482,1649266809832242893,12732638580535658764,16433392303342286089,14922184019333233122,15091156480148462040,6704216841355880468,17532900995618891049);
cl::sycl::ulonglong8 inputData_1(9322739527960522323,11811772144882228584,5422985548299627594,11651019047273457424,17555488830906840554,17995735475875902912,2110077837628093822,6264774342504128133);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<987, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(-5704528567458785542,5117955088002258789,-495232474267128927,7331608981515653929,-5460179492698796358,7563295258755798046,302653557884331502,987948503164750507);
cl::sycl::longlong8 inputData_1(2983940327713855090,7140130715418745293,-3308924074352619952,3269407821091302554,5328715723291683935,8314279326306232603,-7560737689577327256,-531535257632879503);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<988, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(7376015781805483925,11500575955770821811,3259142694401703279,9444163180756125990,4066145087641776643,4783507012054409614,14331487933165272192,17125426745449059140,8553865757421836244,1007071521129015618,2431314767595414544,14893254246050134037,7750288725366364037,5511301122910177694,9962672203762070166,2922957982937631870);
cl::sycl::ulonglong16 inputData_1(12396350582362030253,912397168503709048,17860206915709451187,7869510866190022188,13470541106357948606,8869941278459981297,11467028318482349566,17972831152076882778,16625407941327871653,748609394888949166,7905103705125594141,8725289399237154261,2340011829073956700,253071895971870289,3950739217252726851,93676813472964522);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<989, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(2866612206717327643,-5960616991765439335,-7200913678492934188,9124042710827146316,-360204377116323207,9029710451713450576,-4910786202855588058,4480775162229666622,8095228744210327718,4261519928238454412,4746686167178172184,-3191595940566870881,-5033686727953582535,-1604240814445592036,-9136817902400645539,2832011991046335576);
cl::sycl::longlong16 inputData_1(-1994413515239274539,1712432583012829036,3421813443698477053,6212707020776355990,6932037793572286207,-4552244333158854916,2331023398000497511,-410332954567063663,-2217242603168223679,-4844151057280221365,-7800329904624552217,-7649563825805135085,-6239176862359381451,-757736174504279540,-9159789388181555711,949818789826703137);
return cl::sycl::sub_sat(inputData_0,inputData_1);

});

test_function<990, uint16_t>(
[=](){
uint8_t inputData_0(8);
uint8_t inputData_1(81);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<991, cl::sycl::vec<uint16_t, 2>>(
[=](){
cl::sycl::vec<uint8_t, 2> inputData_0(53,29);
cl::sycl::vec<uint8_t, 2> inputData_1(208,50);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<992, cl::sycl::vec<uint16_t, 3>>(
[=](){
cl::sycl::vec<uint8_t, 3> inputData_0(244,222,28);
cl::sycl::vec<uint8_t, 3> inputData_1(228,19,45);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<993, cl::sycl::vec<uint16_t, 4>>(
[=](){
cl::sycl::vec<uint8_t, 4> inputData_0(51,51,228,28);
cl::sycl::vec<uint8_t, 4> inputData_1(22,77,177,237);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<994, cl::sycl::vec<uint16_t, 8>>(
[=](){
cl::sycl::vec<uint8_t, 8> inputData_0(173,199,90,165,89,181,99,16);
cl::sycl::vec<uint8_t, 8> inputData_1(210,98,44,252,25,82,31,168);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<995, cl::sycl::vec<uint16_t, 16>>(
[=](){
cl::sycl::vec<uint8_t, 16> inputData_0(50,170,73,76,56,23,156,179,4,162,89,230,1,107,152,160);
cl::sycl::vec<uint8_t, 16> inputData_1(251,88,119,112,201,62,17,21,184,230,224,213,83,166,82,62);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<996, int16_t>(
[=](){
int8_t inputData_0(-10);
uint8_t inputData_1(69);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<997, cl::sycl::vec<int16_t, 2>>(
[=](){
cl::sycl::vec<int8_t, 2> inputData_0(14,-48);
cl::sycl::vec<uint8_t, 2> inputData_1(167,154);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<998, cl::sycl::vec<int16_t, 3>>(
[=](){
cl::sycl::vec<int8_t, 3> inputData_0(13,-59,76);
cl::sycl::vec<uint8_t, 3> inputData_1(80,158,223);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<999, cl::sycl::vec<int16_t, 4>>(
[=](){
cl::sycl::vec<int8_t, 4> inputData_0(89,-77,-127,-44);
cl::sycl::vec<uint8_t, 4> inputData_1(185,13,1,82);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1000, cl::sycl::vec<int16_t, 8>>(
[=](){
cl::sycl::vec<int8_t, 8> inputData_0(-3,36,123,-110,15,-60,-93,-76);
cl::sycl::vec<uint8_t, 8> inputData_1(40,200,235,68,32,104,23,9);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1001, cl::sycl::vec<int16_t, 16>>(
[=](){
cl::sycl::vec<int8_t, 16> inputData_0(87,2,-16,-25,48,18,1,-72,1,52,-43,-125,120,43,109,-86);
cl::sycl::vec<uint8_t, 16> inputData_1(226,68,192,228,228,209,20,243,143,69,122,222,191,144,200,236);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1002, uint32_t>(
[=](){
uint16_t inputData_0(28096);
uint16_t inputData_1(3925);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1003, cl::sycl::vec<uint32_t, 2>>(
[=](){
cl::sycl::vec<uint16_t, 2> inputData_0(51970,15102);
cl::sycl::vec<uint16_t, 2> inputData_1(44301,56459);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1004, cl::sycl::vec<uint32_t, 3>>(
[=](){
cl::sycl::vec<uint16_t, 3> inputData_0(59025,3126,48717);
cl::sycl::vec<uint16_t, 3> inputData_1(45598,24040,39607);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1005, cl::sycl::vec<uint32_t, 4>>(
[=](){
cl::sycl::vec<uint16_t, 4> inputData_0(11080,13166,57771,16954);
cl::sycl::vec<uint16_t, 4> inputData_1(48231,45015,47384,18340);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1006, cl::sycl::vec<uint32_t, 8>>(
[=](){
cl::sycl::vec<uint16_t, 8> inputData_0(35403,3807,39042,7477,62332,46789,12810,32762);
cl::sycl::vec<uint16_t, 8> inputData_1(29832,4344,39542,13014,22515,5568,8957,42986);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1007, cl::sycl::vec<uint32_t, 16>>(
[=](){
cl::sycl::vec<uint16_t, 16> inputData_0(29983,914,44961,48390,14085,49370,32917,14238,6857,47195,11940,45989,56935,39920,57406,22171);
cl::sycl::vec<uint16_t, 16> inputData_1(55786,40047,64822,30451,40984,27572,46037,12746,7143,55733,1468,22829,60220,37395,10810,29251);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1008, int32_t>(
[=](){
int16_t inputData_0(25191);
uint16_t inputData_1(29641);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1009, cl::sycl::vec<int32_t, 2>>(
[=](){
cl::sycl::vec<int16_t, 2> inputData_0(7625,-5965);
cl::sycl::vec<uint16_t, 2> inputData_1(53355,45357);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1010, cl::sycl::vec<int32_t, 3>>(
[=](){
cl::sycl::vec<int16_t, 3> inputData_0(-14539,-13335,4978);
cl::sycl::vec<uint16_t, 3> inputData_1(21583,37594,49964);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1011, cl::sycl::vec<int32_t, 4>>(
[=](){
cl::sycl::vec<int16_t, 4> inputData_0(31230,24391,569,-10062);
cl::sycl::vec<uint16_t, 4> inputData_1(12452,28226,23118,18331);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1012, cl::sycl::vec<int32_t, 8>>(
[=](){
cl::sycl::vec<int16_t, 8> inputData_0(12569,21236,24589,31147,26081,-19350,-3321,31463);
cl::sycl::vec<uint16_t, 8> inputData_1(34722,38299,13527,37663,20711,3751,20228,62626);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1013, cl::sycl::vec<int32_t, 16>>(
[=](){
cl::sycl::vec<int16_t, 16> inputData_0(23655,10934,0,17381,-9219,-15945,-2173,-1322,-4554,-12209,13769,19987,28244,-21150,-23817,-15712);
cl::sycl::vec<uint16_t, 16> inputData_1(12042,5154,17233,63039,62749,24494,15683,15162,5475,62191,2749,52577,22786,22451,37972,7129);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1014, uint64_t>(
[=](){
uint32_t inputData_0(3016924077);
uint32_t inputData_1(3163620853);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1015, cl::sycl::vec<uint64_t, 2>>(
[=](){
cl::sycl::vec<uint32_t, 2> inputData_0(91308477,2009757797);
cl::sycl::vec<uint32_t, 2> inputData_1(3745072437,3267173591);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1016, cl::sycl::vec<uint64_t, 3>>(
[=](){
cl::sycl::vec<uint32_t, 3> inputData_0(3116117870,2572752253,3299830510);
cl::sycl::vec<uint32_t, 3> inputData_1(1826509560,1508862045,3592566061);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1017, cl::sycl::vec<uint64_t, 4>>(
[=](){
cl::sycl::vec<uint32_t, 4> inputData_0(552499095,719404171,1815662145,2863952871);
cl::sycl::vec<uint32_t, 4> inputData_1(3431711519,136162367,817615924,56807620);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1018, cl::sycl::vec<uint64_t, 8>>(
[=](){
cl::sycl::vec<uint32_t, 8> inputData_0(3135283524,3871690470,468936563,4050519798,4116050982,2366573231,3301009240,894337639);
cl::sycl::vec<uint32_t, 8> inputData_1(2648442075,2690566963,3844456674,940267670,403122085,3372859491,3614329198,3100373372);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1019, cl::sycl::vec<uint64_t, 16>>(
[=](){
cl::sycl::vec<uint32_t, 16> inputData_0(153277236,303998603,792363832,1094285352,2571787917,2355953941,466451577,3201925114,4094377727,2866743372,2901138391,3231425752,429407499,3595381278,3792537045,476864746);
cl::sycl::vec<uint32_t, 16> inputData_1(3889596491,1992989229,1238533093,3513553364,2895517288,3622074329,2629583873,3916344366,2176325889,944200610,403132827,2098366168,3352681075,2768488409,1473360047,2979048431);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1020, int64_t>(
[=](){
int32_t inputData_0(-438845539);
uint32_t inputData_1(2236060836);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1021, cl::sycl::vec<int64_t, 2>>(
[=](){
cl::sycl::vec<int32_t, 2> inputData_0(-1168893735,-177124293);
cl::sycl::vec<uint32_t, 2> inputData_1(1614410963,1216479415);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1022, cl::sycl::vec<int64_t, 3>>(
[=](){
cl::sycl::vec<int32_t, 3> inputData_0(823854683,738087154,-547333435);
cl::sycl::vec<uint32_t, 3> inputData_1(641875871,1634700895,3760003847);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1023, cl::sycl::vec<int64_t, 4>>(
[=](){
cl::sycl::vec<int32_t, 4> inputData_0(-1916099621,-1382062350,744391208,-1168266035);
cl::sycl::vec<uint32_t, 4> inputData_1(3650738284,627338241,1431372485,3274569325);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1024, cl::sycl::vec<int64_t, 8>>(
[=](){
cl::sycl::vec<int32_t, 8> inputData_0(237812084,-1326651765,-598611224,2141545189,-1062261597,-1892345134,-1380042957,1157447299);
cl::sycl::vec<uint32_t, 8> inputData_1(3608035589,1584804330,1499900514,1907688704,644022515,2434111791,1458174007,2390898039);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1025, cl::sycl::vec<int64_t, 16>>(
[=](){
cl::sycl::vec<int32_t, 16> inputData_0(-1112243749,-272062683,-53735795,-608294004,-732830275,816141125,74510971,-1631997047,-604466554,878866883,-1311491815,-1616223490,-1251373408,717249624,-2054786735,-1054664168);
cl::sycl::vec<uint32_t, 16> inputData_1(3682250773,3001592401,4109349495,1847304029,520917437,3220604856,3468149359,3839774040,2938135170,436120909,13239339,2013249051,3361280046,694693114,528627241,4001444806);
return cl::sycl::upsample(inputData_0,inputData_1);

});

test_function<1026, unsigned char>(
[=](){
unsigned char inputData_0(189);
return cl::sycl::popcount(inputData_0);

});

test_function<1027, signed char>(
[=](){
signed char inputData_0(-19);
return cl::sycl::popcount(inputData_0);

});

test_function<1028, unsigned short>(
[=](){
unsigned short inputData_0(29046);
return cl::sycl::popcount(inputData_0);

});

test_function<1029, short>(
[=](){
short inputData_0(26523);
return cl::sycl::popcount(inputData_0);

});

test_function<1030, unsigned int>(
[=](){
unsigned int inputData_0(48776);
return cl::sycl::popcount(inputData_0);

});

test_function<1031, int>(
[=](){
int inputData_0(-30699);
return cl::sycl::popcount(inputData_0);

});

test_function<1032, unsigned long int>(
[=](){
unsigned long int inputData_0(2414737549);
return cl::sycl::popcount(inputData_0);

});

test_function<1033, long int>(
[=](){
long int inputData_0(462474958);
return cl::sycl::popcount(inputData_0);

});

test_function<1034, unsigned long long int>(
[=](){
unsigned long long int inputData_0(10646959087213951916);
return cl::sycl::popcount(inputData_0);

});

test_function<1035, long long int>(
[=](){
long long int inputData_0(-2095475258484596882);
return cl::sycl::popcount(inputData_0);

});

test_function<1036, cl::sycl::uchar2>(
[=](){
cl::sycl::uchar2 inputData_0(78,5);
return cl::sycl::popcount(inputData_0);

});

test_function<1037, cl::sycl::schar2>(
[=](){
cl::sycl::schar2 inputData_0(56,91);
return cl::sycl::popcount(inputData_0);

});

test_function<1038, cl::sycl::uchar3>(
[=](){
cl::sycl::uchar3 inputData_0(205,66,189);
return cl::sycl::popcount(inputData_0);

});

test_function<1039, cl::sycl::schar3>(
[=](){
cl::sycl::schar3 inputData_0(106,-52,37);
return cl::sycl::popcount(inputData_0);

});

test_function<1040, cl::sycl::uchar4>(
[=](){
cl::sycl::uchar4 inputData_0(191,40,71,194);
return cl::sycl::popcount(inputData_0);

});

test_function<1041, cl::sycl::schar4>(
[=](){
cl::sycl::schar4 inputData_0(34,-82,-116,-87);
return cl::sycl::popcount(inputData_0);

});

test_function<1042, cl::sycl::uchar8>(
[=](){
cl::sycl::uchar8 inputData_0(50,190,82,208,66,64,221,3);
return cl::sycl::popcount(inputData_0);

});

test_function<1043, cl::sycl::schar8>(
[=](){
cl::sycl::schar8 inputData_0(-53,106,-4,44,26,119,58,35);
return cl::sycl::popcount(inputData_0);

});

test_function<1044, cl::sycl::uchar16>(
[=](){
cl::sycl::uchar16 inputData_0(77,3,154,77,194,214,192,193,243,166,119,235,132,71,35,37);
return cl::sycl::popcount(inputData_0);

});

test_function<1045, cl::sycl::schar16>(
[=](){
cl::sycl::schar16 inputData_0(-103,-83,-85,22,-124,120,-61,32,112,3,110,39,45,124,-77,-63);
return cl::sycl::popcount(inputData_0);

});

test_function<1046, cl::sycl::ushort2>(
[=](){
cl::sycl::ushort2 inputData_0(37501,37126);
return cl::sycl::popcount(inputData_0);

});

test_function<1047, cl::sycl::short2>(
[=](){
cl::sycl::short2 inputData_0(-26388,-19708);
return cl::sycl::popcount(inputData_0);

});

test_function<1048, cl::sycl::ushort3>(
[=](){
cl::sycl::ushort3 inputData_0(24044,34125,6607);
return cl::sycl::popcount(inputData_0);

});

test_function<1049, cl::sycl::short3>(
[=](){
cl::sycl::short3 inputData_0(32698,29124,19261);
return cl::sycl::popcount(inputData_0);

});

test_function<1050, cl::sycl::ushort4>(
[=](){
cl::sycl::ushort4 inputData_0(10931,50058,55365,50349);
return cl::sycl::popcount(inputData_0);

});

test_function<1051, cl::sycl::short4>(
[=](){
cl::sycl::short4 inputData_0(14765,-16277,7314,15354);
return cl::sycl::popcount(inputData_0);

});

test_function<1052, cl::sycl::ushort8>(
[=](){
cl::sycl::ushort8 inputData_0(23673,24210,5482,63434,33804,2085,39444,60526);
return cl::sycl::popcount(inputData_0);

});

test_function<1053, cl::sycl::short8>(
[=](){
cl::sycl::short8 inputData_0(23738,-1945,-30183,-6709,26368,-14512,-17656,-7120);
return cl::sycl::popcount(inputData_0);

});

test_function<1054, cl::sycl::ushort16>(
[=](){
cl::sycl::ushort16 inputData_0(47465,36345,42222,10448,46743,46216,56789,24295,65224,63208,886,51839,44883,5496,63708,4246);
return cl::sycl::popcount(inputData_0);

});

test_function<1055, cl::sycl::short16>(
[=](){
cl::sycl::short16 inputData_0(10210,-26643,-25514,-1019,-5527,5644,21582,-11306,4931,-7119,-21210,9783,12022,28700,-16676,27103);
return cl::sycl::popcount(inputData_0);

});

test_function<1056, cl::sycl::uint2>(
[=](){
cl::sycl::uint2 inputData_0(6710,40938);
return cl::sycl::popcount(inputData_0);

});

test_function<1057, cl::sycl::int2>(
[=](){
cl::sycl::int2 inputData_0(27630,124);
return cl::sycl::popcount(inputData_0);

});

test_function<1058, cl::sycl::uint3>(
[=](){
cl::sycl::uint3 inputData_0(1727,44920,65382);
return cl::sycl::popcount(inputData_0);

});

test_function<1059, cl::sycl::int3>(
[=](){
cl::sycl::int3 inputData_0(18148,11243,23723);
return cl::sycl::popcount(inputData_0);

});

test_function<1060, cl::sycl::uint4>(
[=](){
cl::sycl::uint4 inputData_0(54398,965,50302,27393);
return cl::sycl::popcount(inputData_0);

});

test_function<1061, cl::sycl::int4>(
[=](){
cl::sycl::int4 inputData_0(15487,20590,19904,24213);
return cl::sycl::popcount(inputData_0);

});

test_function<1062, cl::sycl::uint8>(
[=](){
cl::sycl::uint8 inputData_0(35234,34266,52242,12378,27003,23093,35285,25155);
return cl::sycl::popcount(inputData_0);

});

test_function<1063, cl::sycl::int8>(
[=](){
cl::sycl::int8 inputData_0(-17906,28665,13205,-4796,25892,30248,4231,-13412);
return cl::sycl::popcount(inputData_0);

});

test_function<1064, cl::sycl::uint16>(
[=](){
cl::sycl::uint16 inputData_0(42613,29723,63973,58905,1731,374,11480,20724,37865,13672,7972,2739,12873,9296,63064,37044);
return cl::sycl::popcount(inputData_0);

});

test_function<1065, cl::sycl::int16>(
[=](){
cl::sycl::int16 inputData_0(-24267,32260,10926,5025,16252,17907,-12113,27734,-23565,4897,-15409,32737,-14872,27057,-13500,-31653);
return cl::sycl::popcount(inputData_0);

});

test_function<1066, cl::sycl::ulong2>(
[=](){
cl::sycl::ulong2 inputData_0(3058408564,1286456899);
return cl::sycl::popcount(inputData_0);

});

test_function<1067, cl::sycl::long2>(
[=](){
cl::sycl::long2 inputData_0(-1391256400,154024523);
return cl::sycl::popcount(inputData_0);

});

test_function<1068, cl::sycl::ulong3>(
[=](){
cl::sycl::ulong3 inputData_0(1240770930,2726012510,1411142647);
return cl::sycl::popcount(inputData_0);

});

test_function<1069, cl::sycl::long3>(
[=](){
cl::sycl::long3 inputData_0(-434669700,1541493045,-145163706);
return cl::sycl::popcount(inputData_0);

});

test_function<1070, cl::sycl::ulong4>(
[=](){
cl::sycl::ulong4 inputData_0(2321980323,7096729,3157719558,2592603653);
return cl::sycl::popcount(inputData_0);

});

test_function<1071, cl::sycl::long4>(
[=](){
cl::sycl::long4 inputData_0(501193852,790567219,1286277586,1605631100);
return cl::sycl::popcount(inputData_0);

});

test_function<1072, cl::sycl::ulong8>(
[=](){
cl::sycl::ulong8 inputData_0(4290949162,2844127303,2413762787,591880138,924787726,4220201331,2534546740,3625057653);
return cl::sycl::popcount(inputData_0);

});

test_function<1073, cl::sycl::long8>(
[=](){
cl::sycl::long8 inputData_0(1789387375,-50761114,-121502658,-783445661,755178584,1895956674,1408586449,-1676866212);
return cl::sycl::popcount(inputData_0);

});

test_function<1074, cl::sycl::ulong16>(
[=](){
cl::sycl::ulong16 inputData_0(2761987029,1639571076,1897371619,434150247,4204473758,1942954804,1662886485,3315593734,3569341247,2829198579,1794271233,1064791045,2761398244,1491875797,2920737541,497898611);
return cl::sycl::popcount(inputData_0);

});

test_function<1075, cl::sycl::long16>(
[=](){
cl::sycl::long16 inputData_0(1632406155,1635748515,998843061,-740883036,-773195810,382506173,-878806832,816766663,-237248713,-933908691,-2083370406,1533947001,-17795233,-2104744629,-114980359,1083706460);
return cl::sycl::popcount(inputData_0);

});

test_function<1076, cl::sycl::ulonglong2>(
[=](){
cl::sycl::ulonglong2 inputData_0(2573030504657235788,1034219332254387118);
return cl::sycl::popcount(inputData_0);

});

test_function<1077, cl::sycl::longlong2>(
[=](){
cl::sycl::longlong2 inputData_0(7939947282321535315,-9054464436562107779);
return cl::sycl::popcount(inputData_0);

});

test_function<1078, cl::sycl::ulonglong3>(
[=](){
cl::sycl::ulonglong3 inputData_0(9250565762229463908,14387255527647581557,4557936642225326121);
return cl::sycl::popcount(inputData_0);

});

test_function<1079, cl::sycl::longlong3>(
[=](){
cl::sycl::longlong3 inputData_0(-3726650224654699924,-341802350471260150,8556780092362182220);
return cl::sycl::popcount(inputData_0);

});

test_function<1080, cl::sycl::ulonglong4>(
[=](){
cl::sycl::ulonglong4 inputData_0(3709304798055834886,11572662135339520918,17611045436704556441,15478845355543170607);
return cl::sycl::popcount(inputData_0);

});

test_function<1081, cl::sycl::longlong4>(
[=](){
cl::sycl::longlong4 inputData_0(6253665832466705610,-8549984973543038247,-1509198829595568988,8025777256218301221);
return cl::sycl::popcount(inputData_0);

});

test_function<1082, cl::sycl::ulonglong8>(
[=](){
cl::sycl::ulonglong8 inputData_0(5707281028830857896,10617481859245160504,15090360294315659096,12380351801970922952,14530638219015063788,16513003092710500279,10261410132195036614,14449045607568794504);
return cl::sycl::popcount(inputData_0);

});

test_function<1083, cl::sycl::longlong8>(
[=](){
cl::sycl::longlong8 inputData_0(3767348294041240168,1372227923812650917,4578757046678967324,-4845163142037692660,8761274671148254989,4489429364952801219,3478787942907586655,-561136020652235476);
return cl::sycl::popcount(inputData_0);

});

test_function<1084, cl::sycl::ulonglong16>(
[=](){
cl::sycl::ulonglong16 inputData_0(16850957764931755528,15423038549697372610,2045635962032141604,10951018514055794414,17556475180207458906,18328179732467300909,5842122184751639566,16178182292728173151,6167242793101850774,11975828153051320699,1844412145027201774,9661792340257128016,2847044411464805750,386452442422959506,10343985869872462321,11502162298082493359);
return cl::sycl::popcount(inputData_0);

});

test_function<1085, cl::sycl::longlong16>(
[=](){
cl::sycl::longlong16 inputData_0(-8806084975772242850,-8551478917140449352,9040126462084258888,1003131367261311679,-4993262571757575266,4023848234243909995,-7733708326298838426,1766876880836987739,-5622568411814138255,-463381350770984388,-1668784923247138256,-6185614667568431579,-902235307448825256,-2497696703287543303,-1599865190629377461,284148064277687784);
return cl::sycl::popcount(inputData_0);

});

test_function<1086, uint32_t>(
[=](){
uint32_t inputData_0(3841091602);
uint32_t inputData_1(3892259637);
uint32_t inputData_2(94631705);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1087, int32_t>(
[=](){
int32_t inputData_0(1196896160);
int32_t inputData_1(-1632613204);
int32_t inputData_2(-792852885);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1088, cl::sycl::vec<uint32_t, 2>>(
[=](){
cl::sycl::vec<uint32_t, 2> inputData_0(474924340,4203935496);
cl::sycl::vec<uint32_t, 2> inputData_1(216475236,2486659060);
cl::sycl::vec<uint32_t, 2> inputData_2(2626505995,3214794760);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1089, cl::sycl::vec<int32_t, 2>>(
[=](){
cl::sycl::vec<int32_t, 2> inputData_0(-1453928542,-456905377);
cl::sycl::vec<int32_t, 2> inputData_1(949239796,-1279250360);
cl::sycl::vec<int32_t, 2> inputData_2(1565911817,-35135168);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1090, cl::sycl::vec<uint32_t, 3>>(
[=](){
cl::sycl::vec<uint32_t, 3> inputData_0(4142935865,2185638156,2142482149);
cl::sycl::vec<uint32_t, 3> inputData_1(3961540636,3488485382,1600164163);
cl::sycl::vec<uint32_t, 3> inputData_2(4261056153,2880838072,795933825);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1091, cl::sycl::vec<int32_t, 3>>(
[=](){
cl::sycl::vec<int32_t, 3> inputData_0(-1955305562,-1045091882,1780340099);
cl::sycl::vec<int32_t, 3> inputData_1(-591371081,-1744843883,2039839722);
cl::sycl::vec<int32_t, 3> inputData_2(856800507,-1430038499,-577956797);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1092, cl::sycl::vec<uint32_t, 4>>(
[=](){
cl::sycl::vec<uint32_t, 4> inputData_0(3294933338,899386977,2119706101,3926560657);
cl::sycl::vec<uint32_t, 4> inputData_1(358546148,304293351,3831006818,4176593733);
cl::sycl::vec<uint32_t, 4> inputData_2(4073349991,1558411909,1542801009,942349032);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1093, cl::sycl::vec<int32_t, 4>>(
[=](){
cl::sycl::vec<int32_t, 4> inputData_0(1474574961,1828224540,845239765,2120714630);
cl::sycl::vec<int32_t, 4> inputData_1(1075759556,-1905256297,449214759,-1758270150);
cl::sycl::vec<int32_t, 4> inputData_2(166299101,1081384883,1714798025,224723272);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1094, cl::sycl::vec<uint32_t, 8>>(
[=](){
cl::sycl::vec<uint32_t, 8> inputData_0(790590994,3548300035,1633561517,1715340570,1523085810,237770357,3637964881,2586289540);
cl::sycl::vec<uint32_t, 8> inputData_1(1471519219,1368862577,1138822463,1254624560,4274888254,2546681055,1308051143,2733914257);
cl::sycl::vec<uint32_t, 8> inputData_2(675927130,577728313,1062356461,4020750975,1699236291,3080449329,684012770,1460502255);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1095, cl::sycl::vec<int32_t, 8>>(
[=](){
cl::sycl::vec<int32_t, 8> inputData_0(1125775169,-724564797,-901042868,329386045,1513428333,654798563,-874840804,-1986827266);
cl::sycl::vec<int32_t, 8> inputData_1(-1244130030,1361058703,1813066674,876863886,304428743,-1663765420,126302404,-130800346);
cl::sycl::vec<int32_t, 8> inputData_2(194592775,628045300,-1358309254,41393085,-138705906,1454341088,237997699,-378372580);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1096, cl::sycl::vec<uint32_t, 16>>(
[=](){
cl::sycl::vec<uint32_t, 16> inputData_0(111747872,211382849,3079003179,3148205133,1187284914,2521885158,3340686091,2362224530,1264187893,1850493062,1301933472,976568229,4140642757,4181342952,3001066799,3043529894);
cl::sycl::vec<uint32_t, 16> inputData_1(1044332760,2589573469,658206042,296742099,2831157640,3558290464,391744221,619154959,3008401843,723429508,2860311127,3285254636,115049383,2196069015,4033735106,2467326923);
cl::sycl::vec<uint32_t, 16> inputData_2(1729444605,3845216068,1367687021,2222984832,185363130,1193532481,2654889248,1871717,2443123656,52896146,2090234545,61185865,3284892308,2254527036,4250094198,151404224);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1097, cl::sycl::vec<int32_t, 16>>(
[=](){
cl::sycl::vec<int32_t, 16> inputData_0(2104345059,1018166806,1523835449,1949380525,-1537029850,71972010,-1095332677,-1772402222,243594618,1286492744,1684292213,-1171516686,1389677539,-1843587151,1447572684,-928680755);
cl::sycl::vec<int32_t, 16> inputData_1(-546512021,1272716864,-405058366,1688398697,-357288391,709319588,-937400364,-414149408,-874102186,40206901,2031100463,-1467619925,1051178635,-1424408667,-903084971,1454691513);
cl::sycl::vec<int32_t, 16> inputData_2(-1594247766,1364361211,1939638305,-108984683,-1868084084,2046479963,920089516,-1803779665,1812394768,222528069,-2143524176,-2062527475,1096054864,-1444013875,-703813553,-1832887658);
return cl::sycl::mad24(inputData_0,inputData_1,inputData_2);

});

test_function<1098, uint32_t>(
[=](){
uint32_t inputData_0(3104696864);
uint32_t inputData_1(1965255554);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1099, int32_t>(
[=](){
int32_t inputData_0(1539492422);
int32_t inputData_1(293964841);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1100, cl::sycl::vec<uint32_t, 2>>(
[=](){
cl::sycl::vec<uint32_t, 2> inputData_0(1062488033,1889797221);
cl::sycl::vec<uint32_t, 2> inputData_1(3964381509,1403520900);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1101, cl::sycl::vec<int32_t, 2>>(
[=](){
cl::sycl::vec<int32_t, 2> inputData_0(1844786058,1111248999);
cl::sycl::vec<int32_t, 2> inputData_1(1173418751,-1478821410);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1102, cl::sycl::vec<uint32_t, 3>>(
[=](){
cl::sycl::vec<uint32_t, 3> inputData_0(3254023934,1228482806,812810786);
cl::sycl::vec<uint32_t, 3> inputData_1(2244367687,242573322,2338039747);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1103, cl::sycl::vec<int32_t, 3>>(
[=](){
cl::sycl::vec<int32_t, 3> inputData_0(827396054,1098017492,1596455914);
cl::sycl::vec<int32_t, 3> inputData_1(788945712,169160808,207607840);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1104, cl::sycl::vec<uint32_t, 4>>(
[=](){
cl::sycl::vec<uint32_t, 4> inputData_0(3168127986,1834767826,3666245864,529753038);
cl::sycl::vec<uint32_t, 4> inputData_1(1901163918,3515791196,1457141226,3865988604);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1105, cl::sycl::vec<int32_t, 4>>(
[=](){
cl::sycl::vec<int32_t, 4> inputData_0(-1599016291,-434639933,1197317373,-903391182);
cl::sycl::vec<int32_t, 4> inputData_1(-1582576145,-1808791950,-331829973,1715460640);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1106, cl::sycl::vec<uint32_t, 8>>(
[=](){
cl::sycl::vec<uint32_t, 8> inputData_0(44010409,362609746,3630364450,3648210746,529188763,3285544590,2749884264,1579463606);
cl::sycl::vec<uint32_t, 8> inputData_1(2736003834,1327662421,1029606922,1100218681,3147857059,503028970,1101456237,3543658849);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1107, cl::sycl::vec<int32_t, 8>>(
[=](){
cl::sycl::vec<int32_t, 8> inputData_0(737661300,-920449703,-972156755,1171017012,1523703416,-1522402408,1333974974,1868230248);
cl::sycl::vec<int32_t, 8> inputData_1(-291311171,99390449,-583675560,-2056757397,481774045,1943090764,-811944007,-1047571072);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1108, cl::sycl::vec<uint32_t, 16>>(
[=](){
cl::sycl::vec<uint32_t, 16> inputData_0(1008373181,3417111787,1439110778,384866028,3053219119,2932718598,2902180576,3874948157,1808801212,3971310824,2200363186,3753301437,2162498955,814092083,1049348818,2367988816);
cl::sycl::vec<uint32_t, 16> inputData_1(352428787,416102563,3926319299,2608587913,798862838,2914334060,2356384448,922914457,590299055,2177603436,3656464086,2802383993,2475806113,3604168231,3808265509,3542773751);
return cl::sycl::mul24(inputData_0,inputData_1);

});

test_function<1109, cl::sycl::vec<int32_t, 16>>(
[=](){
cl::sycl::vec<int32_t, 16> inputData_0(1149908560,940512293,-2012914287,330324138,-577122656,1255274109,-1502618531,-1746260803,303872758,1709808505,-777776123,-1449961259,-1862211550,400925002,-157199849,-2129210512);
cl::sycl::vec<int32_t, 16> inputData_1(1457634942,480566473,431459541,129591276,-1345924387,2104448848,878814346,-600982548,952210865,65903454,573587500,-226053550,1689057998,-1830487950,-878000418,-68387347);
return cl::sycl::mul24(inputData_0,inputData_1);

});

 }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}