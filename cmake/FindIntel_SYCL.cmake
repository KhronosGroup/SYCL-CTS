message(WARNING
        "Intel_SYCL is deprecated and will be removed, use DPCPP instead")

set(DPCPP_INSTALL_DIR ${INTEL_SYCL_ROOT})
set(DPCPP_FLAGS ${INTEL_SYCL_FLAGS})
if(DEFINED INTEL_SYCL_TRIPLE)
    set(DPCPP_TARGET_TRIPLES ${INTEL_SYCL_TRIPLE})
endif()

include(FindDPCPP)
