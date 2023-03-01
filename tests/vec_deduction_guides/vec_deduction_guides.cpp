#include "../common/common.h"

namespace vec_deduction_guides {
using namespace sycl;

template <typename T>
class check_vec_deduction {
public:
  operator()(const std::string& type) {
    typeName = type;
  }

private:
  std::string typeName;

};

TEST_CASE("vec deduction guides", "[test_vector_deduction]") {
  for_all_types<check_vec_deduction>(deduction::vector_types);
  for_all_types<check_vec_deduction>(deduction::scalar_types);
}

} // deduction_guides