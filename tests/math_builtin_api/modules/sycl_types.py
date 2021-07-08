"""Represents an argument or return type. The type may be generic."""
class argtype:
    def __init__(self, name, var_type, base_type, dim, child_types, unsigned=False):
        self.name = name # Type name. This works also as an identifier for the type.
        self.var_type = var_type # Variable type (scalar,vector or NULL). Generic types should have a NULL variable type.
        self.base_type = base_type # The base type (e.g. the base type of sycl::double3 is double). Generic types should have a NULL base type.
        self.dim = dim # Dimensionality of the type. Generic types should have this as zero.
        self.child_types = child_types # Contains the list of types which this type can be decomposed to. Basic types should accept an empty list.
        self.unsigned = unsigned # A flag indicating whether this type is an unsigned one (e.g. int vs uint).
    def __eq__(self, other):
        if isinstance(other, argtype):
            return (self.name == other.name)
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash(name)

def create_basic_types():
    type_dic = {}

    # Basic Types

    t_float_0 = argtype("float", "scalar", "float", 1, [])
    type_dic["float"] = t_float_0

    t_float_2 = argtype("sycl::float2", "vector", "float", 2, [])
    type_dic["sycl::float2"] = t_float_2

    t_float_3 = argtype("sycl::float3", "vector", "float", 3, [])
    type_dic["sycl::float3"] = t_float_3

    t_float_4 = argtype("sycl::float4", "vector", "float", 4, [])
    type_dic["sycl::float4"] = t_float_4

    t_float_8 = argtype("sycl::float8", "vector", "float", 8, [])
    type_dic["sycl::float8"] = t_float_8

    t_float_16 = argtype("sycl::float16", "vector", "float", 16, [])
    type_dic["sycl::float16"] = t_float_16


    t_double_0 = argtype("double", "scalar", "double", 1, [])
    type_dic["double"] = t_double_0

    t_double_2 = argtype("sycl::double2", "vector", "double", 2, [])
    type_dic["sycl::double2"] = t_double_2

    t_double_3 = argtype("sycl::double3", "vector", "double", 3, [])
    type_dic["sycl::double3"] = t_double_3

    t_double_4 = argtype("sycl::double4", "vector", "double", 4, [])
    type_dic["sycl::double4"] = t_double_4

    t_double_8 = argtype("sycl::double8", "vector", "double", 8, [])
    type_dic["sycl::double8"] = t_double_8

    t_double_16 = argtype("sycl::double16", "vector", "double", 16, [])
    type_dic["sycl::double16"] = t_double_16


    t_half_0 = argtype("sycl::half", "scalar", "sycl::half", 1, [])
    type_dic["sycl::half"] = t_half_0

    t_half_2 = argtype("sycl::half2", "vector", "sycl::half", 2, [])
    type_dic["sycl::half2"] = t_half_2

    t_half_3 = argtype("sycl::half3", "vector", "sycl::half", 3, [])
    type_dic["sycl::half3"] = t_half_3

    t_half_4 = argtype("sycl::half4", "vector", "sycl::half", 4, [])
    type_dic["sycl::half4"] = t_half_4

    t_half_8 = argtype("sycl::half8", "vector", "sycl::half", 8, [])
    type_dic["sycl::half8"] = t_half_8

    t_half_16 = argtype("sycl::half16", "vector", "sycl::half", 16, [])
    type_dic["sycl::half16"] = t_half_16


    t_char_0 = argtype("char", "scalar", "char", 1, [])
    type_dic["char"] = t_char_0

    t_char_2 = argtype("sycl::char2", "vector", "char", 2, [])
    type_dic["sycl::char2"] = t_char_2

    t_char_3 = argtype("sycl::char3", "vector", "char", 3, [])
    type_dic["sycl::char3"] = t_char_3

    t_char_4 = argtype("sycl::char4", "vector", "char", 4, [])
    type_dic["sycl::char4"] = t_char_4

    t_char_8 = argtype("sycl::char8", "vector", "char", 8, [])
    type_dic["sycl::char8"] = t_char_8

    t_char_16 = argtype("sycl::char16", "vector", "char", 16, [])
    type_dic["sycl::char16"] = t_char_16


    t_schar_0 = argtype("signed char", "scalar", "char", 1, [])
    type_dic["signed char"] = t_schar_0

    t_schar_2 = argtype("sycl::schar2", "vector", "char", 2, [])
    type_dic["sycl::schar2"] = t_schar_2

    t_schar_3 = argtype("sycl::schar3", "vector", "char", 3, [])
    type_dic["sycl::schar3"] = t_schar_3

    t_schar_4 = argtype("sycl::schar4", "vector", "char", 4, [])
    type_dic["sycl::schar4"] = t_schar_4

    t_schar_8 = argtype("sycl::schar8", "vector", "char", 8, [])
    type_dic["sycl::schar8"] = t_schar_8

    t_schar_16 = argtype("sycl::schar16", "vector", "char", 16, [])
    type_dic["sycl::schar16"] = t_schar_16


    t_uchar_0 = argtype("unsigned char", "scalar", "char", 1, [], True)
    type_dic["unsigned char"] = t_uchar_0

    t_uchar_2 = argtype("sycl::uchar2", "vector", "char", 2, [], True)
    type_dic["sycl::uchar2"] = t_uchar_2

    t_uchar_3 = argtype("sycl::uchar3", "vector", "char", 3, [], True)
    type_dic["sycl::uchar3"] = t_uchar_3

    t_uchar_4 = argtype("sycl::uchar4", "vector", "char", 4, [], True)
    type_dic["sycl::uchar4"] = t_uchar_4

    t_uchar_8 = argtype("sycl::uchar8", "vector", "char", 8, [], True)
    type_dic["sycl::uchar8"] = t_uchar_8

    t_uchar_16 = argtype("sycl::uchar16", "vector", "char", 16, [], True)
    type_dic["sycl::uchar16"] = t_uchar_16


    t_short_0 = argtype("short", "scalar", "short", 1, [])
    type_dic["short"] = t_short_0

    t_short_2 = argtype("sycl::short2", "vector", "short", 2, [])
    type_dic["sycl::short2"] = t_short_2

    t_short_3 = argtype("sycl::short3", "vector", "short", 3, [])
    type_dic["sycl::short3"] = t_short_3

    t_short_4 = argtype("sycl::short4", "vector", "short", 4, [])
    type_dic["sycl::short4"] = t_short_4

    t_short_8 = argtype("sycl::short8", "vector", "short", 8, [])
    type_dic["sycl::short8"] = t_short_8

    t_short_16 = argtype("sycl::short16", "vector", "short", 16, [])
    type_dic["sycl::short16"] = t_short_16


    t_ushort_0 = argtype("unsigned short", "scalar", "short", 1, [], True)
    type_dic["unsigned short"] = t_ushort_0

    t_ushort_2 = argtype("sycl::ushort2", "vector", "short", 2, [], True)
    type_dic["sycl::ushort2"] = t_ushort_2

    t_ushort_3 = argtype("sycl::ushort3", "vector", "short", 3, [], True)
    type_dic["sycl::ushort3"] = t_ushort_3

    t_ushort_4 = argtype("sycl::ushort4", "vector", "short", 4, [], True)
    type_dic["sycl::ushort4"] = t_ushort_4

    t_ushort_8 = argtype("sycl::ushort8", "vector", "short", 8, [], True)
    type_dic["sycl::ushort8"] = t_ushort_8

    t_ushort_16 = argtype("sycl::ushort16", "vector", "short", 16, [], True)
    type_dic["sycl::ushort16"] = t_ushort_16


    t_int_0 = argtype("int", "scalar", "int", 1, [])
    type_dic["int"] = t_int_0

    t_int_2 = argtype("sycl::int2", "vector", "int", 2, [])
    type_dic["sycl::int2"] = t_int_2

    t_int_3 = argtype("sycl::int3", "vector", "int", 3, [])
    type_dic["sycl::int3"] = t_int_3

    t_int_4 = argtype("sycl::int4", "vector", "int", 4, [])
    type_dic["sycl::int4"] = t_int_4

    t_int_8 = argtype("sycl::int8", "vector", "int", 8, [])
    type_dic["sycl::int8"] = t_int_8

    t_int_16 = argtype("sycl::int16", "vector", "int", 16, [])
    type_dic["sycl::int16"] = t_int_16


    t_uint_0 = argtype("unsigned int", "scalar", "int", 1, [], True)
    type_dic["unsigned int"] = t_uint_0

    t_uint_2 = argtype("sycl::uint2", "vector", "int", 2, [], True)
    type_dic["sycl::uint2"] = t_uint_2

    t_uint_3 = argtype("sycl::uint3", "vector", "int", 3, [], True)
    type_dic["sycl::uint3"] = t_uint_3

    t_uint_4 = argtype("sycl::uint4", "vector", "int", 4, [], True)
    type_dic["sycl::uint4"] = t_uint_4

    t_uint_8 = argtype("sycl::uint8", "vector", "int", 8, [], True)
    type_dic["sycl::uint8"] = t_uint_8

    t_uint_16 = argtype("sycl::uint16", "vector", "int", 16, [], True)
    type_dic["sycl::uint16"] = t_uint_16


    t_long_0 = argtype("long int", "scalar", "long int", 1, [])
    type_dic["long int"] = t_long_0

    t_long_2 = argtype("sycl::long2", "vector", "long int", 2, [])
    type_dic["sycl::long2"] = t_long_2

    t_long_3 = argtype("sycl::long3", "vector", "long int", 3, [])
    type_dic["sycl::long3"] = t_long_3

    t_long_4 = argtype("sycl::long4", "vector", "long int", 4, [])
    type_dic["sycl::long4"] = t_long_4

    t_long_8 = argtype("sycl::long8", "vector", "long int", 8, [])
    type_dic["sycl::long8"] = t_long_8

    t_long_16 = argtype("sycl::long16", "vector", "long int", 16, [])
    type_dic["sycl::long16"] = t_long_16


    t_ulong_0 = argtype("unsigned long int", "scalar", "long int", 1, [], True)
    type_dic["unsigned long int"] = t_ulong_0

    t_ulong_2 = argtype("sycl::ulong2", "vector", "long int", 2, [], True)
    type_dic["sycl::ulong2"] = t_ulong_2

    t_ulong_3 = argtype("sycl::ulong3", "vector", "long int", 3, [], True)
    type_dic["sycl::ulong3"] = t_ulong_3

    t_ulong_4 = argtype("sycl::ulong4", "vector", "long int", 4, [], True)
    type_dic["sycl::ulong4"] = t_ulong_4

    t_ulong_8 = argtype("sycl::ulong8", "vector", "long int", 8, [], True)
    type_dic["sycl::ulong8"] = t_ulong_8

    t_ulong_16 = argtype("sycl::ulong16", "vector", "long int", 16, [], True)
    type_dic["sycl::ulong16"] = t_ulong_16


    t_longlong_0 = argtype("long long int", "scalar", "long long int", 1, [])
    type_dic["long long int"] = t_longlong_0

    t_longlong_2 = argtype("sycl::longlong2", "vector", "long long int", 2, [])
    type_dic["sycl::longlong2"] = t_longlong_2

    t_longlong_3 = argtype("sycl::longlong3", "vector", "long long int", 3, [])
    type_dic["sycl::longlong3"] = t_longlong_3

    t_longlong_4 = argtype("sycl::longlong4", "vector", "long long int", 4, [])
    type_dic["sycl::longlong4"] = t_longlong_4

    t_longlong_8 = argtype("sycl::longlong8", "vector", "long long int", 8, [])
    type_dic["sycl::longlong8"] = t_longlong_8

    t_longlong_16 = argtype("sycl::longlong16", "vector", "long long int", 16, [])
    type_dic["sycl::longlong16"] = t_longlong_16


    t_ulonglong_0 = argtype("unsigned long long int", "scalar", "long long int", 1, [], True)
    type_dic["unsigned long long int"] = t_ulonglong_0

    t_ulonglong_2 = argtype("sycl::ulonglong2", "vector", "long long int", 2, [], True)
    type_dic["sycl::ulonglong2"] = t_ulonglong_2

    t_ulonglong_3 = argtype("sycl::ulonglong3", "vector", "long long int", 3, [], True)
    type_dic["sycl::ulonglong3"] = t_ulonglong_3

    t_ulonglong_4 = argtype("sycl::ulonglong4", "vector", "long long int", 4, [], True)
    type_dic["sycl::ulonglong4"] = t_ulonglong_4

    t_ulonglong_8 = argtype("sycl::ulonglong8", "vector", "long long int", 8, [], True)
    type_dic["sycl::ulonglong8"] = t_ulonglong_8

    t_ulonglong_16 = argtype("sycl::ulonglong16", "vector", "long long int", 16, [], True)
    type_dic["sycl::ulonglong16"] = t_ulonglong_16


    # Fixed size basic types

    t_int8t_0 = argtype("int8_t", "scalar", "int8_t", 1, [])
    type_dic["int8_t"] = t_int8t_0

    t_int8t_2 = argtype("sycl::vec<int8_t, 2>", "vector", "int8_t", 2, [])
    type_dic["sycl::vec<int8_t, 2>"] = t_int8t_2

    t_int8t_3 = argtype("sycl::vec<int8_t, 3>", "vector", "int8_t", 3, [])
    type_dic["sycl::vec<int8_t, 3>"] = t_int8t_3

    t_int8t_4 = argtype("sycl::vec<int8_t, 4>", "vector", "int8_t", 4, [])
    type_dic["sycl::vec<int8_t, 4>"] = t_int8t_4

    t_int8t_8 = argtype("sycl::vec<int8_t, 8>", "vector", "int8_t", 8, [])
    type_dic["sycl::vec<int8_t, 8>"] = t_int8t_8

    t_int8t_16 = argtype("sycl::vec<int8_t, 16>", "vector", "int8_t", 16, [])
    type_dic["sycl::vec<int8_t, 16>"] = t_int8t_16


    t_uint8t_0 = argtype("uint8_t", "scalar", "int8_t", 1, [], True)
    type_dic["uint8_t"] = t_uint8t_0

    t_uint8t_2 = argtype("sycl::vec<uint8_t, 2>", "vector", "int8_t", 2, [], True)
    type_dic["sycl::vec<uint8_t, 2>"] = t_uint8t_2

    t_uint8t_3 = argtype("sycl::vec<uint8_t, 3>", "vector", "int8_t", 3, [], True)
    type_dic["sycl::vec<uint8_t, 3>"] = t_uint8t_3

    t_uint8t_4 = argtype("sycl::vec<uint8_t, 4>", "vector", "int8_t", 4, [], True)
    type_dic["sycl::vec<uint8_t, 4>"] = t_uint8t_4

    t_uint8t_8 = argtype("sycl::vec<uint8_t, 8>", "vector", "int8_t", 8, [], True)
    type_dic["sycl::vec<uint8_t, 8>"] = t_uint8t_8

    t_uint8t_16 = argtype("sycl::vec<uint8_t, 16>", "vector", "int8_t", 16, [], True)
    type_dic["sycl::vec<uint8_t, 16>"] = t_uint8t_16


    t_int16t_0 = argtype("int16_t", "scalar", "int16_t", 1, [])
    type_dic["int16_t"] = t_int16t_0

    t_int16t_2 = argtype("sycl::vec<int16_t, 2>", "vector", "int16_t", 2, [])
    type_dic["sycl::vec<int16_t, 2>"] = t_int16t_2

    t_int16t_3 = argtype("sycl::vec<int16_t, 3>", "vector", "int16_t", 3, [])
    type_dic["sycl::vec<int16_t, 3>"] = t_int16t_3

    t_int16t_4 = argtype("sycl::vec<int16_t, 4>", "vector", "int16_t", 4, [])
    type_dic["sycl::vec<int16_t, 4>"] = t_int16t_4

    t_int16t_8 = argtype("sycl::vec<int16_t, 8>", "vector", "int16_t", 8, [])
    type_dic["sycl::vec<int16_t, 8>"] = t_int16t_8

    t_int16t_16 = argtype("sycl::vec<int16_t, 16>", "vector", "int16_t", 16, [])
    type_dic["sycl::vec<int16_t, 16>"] = t_int16t_16


    t_uint16t_0 = argtype("uint16_t", "scalar", "int16_t", 1, [], True)
    type_dic["uint16_t"] = t_uint16t_0

    t_uint16t_2 = argtype("sycl::vec<uint16_t, 2>", "vector", "int16_t", 2, [], True)
    type_dic["sycl::vec<uint16_t, 2>"] = t_uint16t_2

    t_uint16t_3 = argtype("sycl::vec<uint16_t, 3>", "vector", "int16_t", 3, [], True)
    type_dic["sycl::vec<uint16_t, 3>"] = t_uint16t_3

    t_uint16t_4 = argtype("sycl::vec<uint16_t, 4>", "vector", "int16_t", 4, [], True)
    type_dic["sycl::vec<uint16_t, 4>"] = t_uint16t_4

    t_uint16t_8 = argtype("sycl::vec<uint16_t, 8>", "vector", "int16_t", 8, [], True)
    type_dic["sycl::vec<uint16_t, 8>"] = t_uint16t_8

    t_uint16t_16 = argtype("sycl::vec<uint16_t, 16>", "vector", "int16_t", 16, [], True)
    type_dic["sycl::vec<uint16_t, 16>"] = t_uint16t_16


    t_int32t_0 = argtype("int32_t", "scalar", "int32_t", 1, [])
    type_dic["int32_t"] = t_int32t_0

    t_int32t_2 = argtype("sycl::vec<int32_t, 2>", "vector", "int32_t", 2, [])
    type_dic["sycl::vec<int32_t, 2>"] = t_int32t_2

    t_int32t_3 = argtype("sycl::vec<int32_t, 3>", "vector", "int32_t", 3, [])
    type_dic["sycl::vec<int32_t, 3>"] = t_int32t_3

    t_int32t_4 = argtype("sycl::vec<int32_t, 4>", "vector", "int32_t", 4, [])
    type_dic["sycl::vec<int32_t, 4>"] = t_int32t_4

    t_int32t_8 = argtype("sycl::vec<int32_t, 8>", "vector", "int32_t", 8, [])
    type_dic["sycl::vec<int32_t, 8>"] = t_int32t_8

    t_int32t_16 = argtype("sycl::vec<int32_t, 16>", "vector", "int32_t", 16, [])
    type_dic["sycl::vec<int32_t, 16>"] = t_int32t_16


    t_uint32t_0 = argtype("uint32_t", "scalar", "int32_t", 1, [], True)
    type_dic["uint32_t"] = t_uint32t_0

    t_uint32t_2 = argtype("sycl::vec<uint32_t, 2>", "vector", "int32_t", 2, [], True)
    type_dic["sycl::vec<uint32_t, 2>"] = t_uint32t_2

    t_uint32t_3 = argtype("sycl::vec<uint32_t, 3>", "vector", "int32_t", 3, [], True)
    type_dic["sycl::vec<uint32_t, 3>"] = t_uint32t_3

    t_uint32t_4 = argtype("sycl::vec<uint32_t, 4>", "vector", "int32_t", 4, [], True)
    type_dic["sycl::vec<uint32_t, 4>"] = t_uint32t_4

    t_uint32t_8 = argtype("sycl::vec<uint32_t, 8>", "vector", "int32_t", 8, [], True)
    type_dic["sycl::vec<uint32_t, 8>"] = t_uint32t_8

    t_uint32t_16 = argtype("sycl::vec<uint32_t, 16>", "vector", "int32_t", 16, [], True)
    type_dic["sycl::vec<uint32_t, 16>"] = t_uint32t_16


    t_int64t_0 = argtype("int64_t", "scalar", "int64_t", 1, [])
    type_dic["int64_t"] = t_int64t_0

    t_int64t_2 = argtype("sycl::vec<int64_t, 2>", "vector", "int64_t", 2, [])
    type_dic["sycl::vec<int64_t, 2>"] = t_int64t_2

    t_int64t_3 = argtype("sycl::vec<int64_t, 3>", "vector", "int64_t", 3, [])
    type_dic["sycl::vec<int64_t, 3>"] = t_int64t_3

    t_int64t_4 = argtype("sycl::vec<int64_t, 4>", "vector", "int64_t", 4, [])
    type_dic["sycl::vec<int64_t, 4>"] = t_int64t_4

    t_int64t_8 = argtype("sycl::vec<int64_t, 8>", "vector", "int64_t", 8, [])
    type_dic["sycl::vec<int64_t, 8>"] = t_int64t_8

    t_int64t_16 = argtype("sycl::vec<int64_t, 16>", "vector", "int64_t", 16, [])
    type_dic["sycl::vec<int64_t, 16>"] = t_int64t_16


    t_uint64t_0 = argtype("uint64_t", "scalar", "int64_t", 1, [], True)
    type_dic["uint64_t"] = t_uint64t_0

    t_uint64t_2 = argtype("sycl::vec<uint64_t, 2>", "vector", "int64_t", 2, [], True)
    type_dic["sycl::vec<uint64_t, 2>"] = t_uint64t_2

    t_uint64t_3 = argtype("sycl::vec<uint64_t, 3>", "vector", "int64_t", 3, [], True)
    type_dic["sycl::vec<uint64_t, 3>"] = t_uint64t_3

    t_uint64t_4 = argtype("sycl::vec<uint64_t, 4>", "vector", "int64_t", 4, [], True)
    type_dic["sycl::vec<uint64_t, 4>"] = t_uint64t_4

    t_uint64t_8 = argtype("sycl::vec<uint64_t, 8>", "vector", "int64_t", 8, [], True)
    type_dic["sycl::vec<uint64_t, 8>"] = t_uint64t_8

    t_uint64t_16 = argtype("sycl::vec<uint64_t, 16>", "vector", "int64_t", 16, [], True)
    type_dic["sycl::vec<uint64_t, 16>"] = t_uint64t_16

    return type_dic


    # Generic Types
def create_types():
    type_dic = create_basic_types()

    t_float_n = argtype("floatn", "NULL", "NULL", 0, ["sycl::float2","sycl::float3","sycl::float4","sycl::float8","sycl::float16"])
    type_dic["floatn"] = t_float_n

    t_gen_float_f = argtype("genfloatf", "NULL", "NULL", 0, ["float", "floatn"])
    type_dic["genfloatf"] = t_gen_float_f

    t_double_n = argtype("doublen", "NULL", "NULL", 0, ["sycl::double2","sycl::double3","sycl::double4","sycl::double8","sycl::double16"])
    type_dic["doublen"] = t_double_n

    t_gen_float_d = argtype("genfloatd", "NULL", "NULL", 0, ["double","doublen"])
    type_dic["genfloatd"] = t_gen_float_d

    t_half_n = argtype("halfn", "NULL", "NULL", 0, ["sycl::half2","sycl::half3","sycl::half4","sycl::half8","sycl::half16"])
    type_dic["halfn"] = t_half_n

    t_gen_float_h = argtype("genfloath", "NULL", "NULL", 0, ["sycl::half","halfn"])
    type_dic["genfloath"] = t_gen_float_h

    t_gen_float = argtype("genfloat", "NULL", "NULL", 0, ["genfloatf","genfloatd","genfloath"])
    type_dic["genfloat"] = t_gen_float

    t_sgen_float = argtype("sgenfloat", "NULL", "NULL", 0, ["float","double","sycl::half"])
    type_dic["sgenfloat"] = t_sgen_float

    t_gen_geofloat = argtype("gengeofloat", "NULL", "NULL", 0, ["float","sycl::float2","sycl::float3","sycl::float4"])
    type_dic["gengeofloat"] = t_gen_geofloat

    t_gen_geodouble = argtype("gengeodouble", "NULL", "NULL", 0, ["double","sycl::double2","sycl::double3","sycl::double4"])
    type_dic["gengeodouble"] = t_gen_geodouble

    t_char_n = argtype("charn", "NULL", "NULL", 0, ["sycl::char2","sycl::char3","sycl::char4","sycl::char8","sycl::char16"])
    type_dic["charn"] = t_char_n

    t_schar_n = argtype("scharn", "NULL", "NULL", 0, ["sycl::schar2","sycl::schar3","sycl::schar4","sycl::schar8","sycl::schar16"])
    type_dic["scharn"] = t_schar_n

    t_uchar_n = argtype("ucharn", "NULL", "NULL", 0, ["sycl::uchar2","sycl::uchar3","sycl::uchar4","sycl::uchar8","sycl::uchar16"])
    type_dic["ucharn"] = t_uchar_n

    t_igen_char = argtype("igenchar", "NULL", "NULL", 0, ["signed char","scharn"])
    type_dic["igenchar"] = t_igen_char

    t_ugen_char = argtype("ugenchar", "NULL", "NULL", 0, ["unsigned char","ucharn"])
    type_dic["ugenchar"] = t_ugen_char

    t_gen_char = argtype("genchar", "NULL", "NULL", 0, ["char","charn","igenchar","ugenchar"])
    type_dic["genchar"] = t_gen_char

    t_short_n = argtype("shortn", "NULL", "NULL", 0, ["sycl::short2","sycl::short3","sycl::short4","sycl::short8","sycl::short16"])
    type_dic["shortn"] = t_short_n

    t_gen_short = argtype("genshort", "NULL", "NULL", 0, ["short","shortn"])
    type_dic["genshort"] = t_gen_short

    t_ushort_n = argtype("ushortn", "NULL", "NULL", 0, ["sycl::ushort2","sycl::ushort3","sycl::ushort4","sycl::ushort8","sycl::ushort16"])
    type_dic["ushortn"] = t_ushort_n

    t_ugen_short = argtype("ugenshort", "NULL", "NULL", 0, ["unsigned short","ushortn"])
    type_dic["ugenshort"] = t_ugen_short

    t_uint_n = argtype("uintn", "NULL", "NULL", 0, ["sycl::uint2","sycl::uint3","sycl::uint4","sycl::uint8","sycl::uint16"])
    type_dic["uintn"] = t_uint_n

    t_ugen_int = argtype("ugenint", "NULL", "NULL", 0, ["unsigned int","uintn"])
    type_dic["ugenint"] = t_ugen_int

    t_int_n = argtype("intn", "NULL", "NULL", 0, ["sycl::int2","sycl::int3","sycl::int4","sycl::int8","sycl::int16"])
    type_dic["intn"] = t_int_n

    t_gen_int = argtype("genint", "NULL", "NULL", 0, ["int","intn"])
    type_dic["genint"] = t_gen_int

    t_ulong_n = argtype("ulongn", "NULL", "NULL", 0, ["sycl::ulong2","sycl::ulong3","sycl::ulong4","sycl::ulong8","sycl::ulong16"])
    type_dic["ulongn"] = t_ulong_n

    t_ugen_long = argtype("ugenlong", "NULL", "NULL", 0, ["unsigned long int", "ulongn"])
    type_dic["ugenlong"] = t_ugen_long

    t_long_n = argtype("longn", "NULL", "NULL", 0, ["sycl::long2","sycl::long3","sycl::long4","sycl::long8","sycl::long16"])
    type_dic["longn"] = t_long_n

    t_gen_long = argtype("genlong", "NULL", "NULL", 0, ["long int", "longn"])
    type_dic["genlong"] = t_gen_long

    t_ulonglong_n = argtype("ulonglongn", "NULL", "NULL", 0, ["sycl::ulonglong2","sycl::ulonglong3","sycl::ulonglong4","sycl::ulonglong8","sycl::ulonglong16"])
    type_dic["ulonglongn"] = t_ulonglong_n

    t_ugen_longlong = argtype("ugenlonglong", "NULL", "NULL", 0, ["unsigned long long int", "ulonglongn"])
    type_dic["ugenlonglong"] = t_ugen_longlong

    t_longlong_n = argtype("longlongn", "NULL", "NULL", 0, ["sycl::longlong2","sycl::longlong3","sycl::longlong4","sycl::longlong8","sycl::longlong16"])
    type_dic["longlongn"] = t_longlong_n

    t_gen_longlong = argtype("genlonglong", "NULL", "NULL", 0, ["long long int", "longlongn"])
    type_dic["genlonglong"] = t_gen_longlong

    t_igen_long_integer = argtype("igenlonginteger", "NULL", "NULL", 0, ["genlong", "genlonglong"])
    type_dic["igenlonginteger"] = t_igen_long_integer

    t_ugen_long_integer = argtype("ugenlonginteger", "NULL", "NULL", 0, ["ugenlong", "ugenlonglong"])
    type_dic["ugenlonginteger"] = t_ugen_long_integer

    t_gen_integer = argtype("geninteger", "NULL", "NULL", 0, ["genchar","genshort","ugenshort","genint","ugenint","igenlonginteger","ugenlonginteger"])
    type_dic["geninteger"] = t_gen_integer

    t_igen_integer = argtype("igeninteger", "NULL", "NULL", 0, ["igenchar","genshort","genint","igenlonginteger"])
    type_dic["igeninteger"] = t_igen_integer

    t_ugen_integer = argtype("ugeninteger", "NULL", "NULL", 0, ["ugenchar","ugenshort","ugenint","ugenlonginteger"])
    type_dic["ugeninteger"] = t_ugen_integer

    t_sgen_integer = argtype("sgeninteger", "NULL", "NULL", 0, ["char","signed char","unsigned char", "short","unsigned short","int","unsigned int","long int","unsigned long int","long long int","unsigned long long int"])
    type_dic["sgeninteger"] = t_sgen_integer

    t_gen_type = argtype("gentype", "NULL", "NULL", 0, ["genfloat", "geninteger"])
    type_dic["gentype"] = t_gen_type


    # Fixed size generic types

    t_igen_integer_8bit = argtype("igeninteger8bit", "NULL", "NULL", 0, ["int8_t","sycl::vec<int8_t, 2>","sycl::vec<int8_t, 3>","sycl::vec<int8_t, 4>","sycl::vec<int8_t, 8>","sycl::vec<int8_t, 16>"])
    type_dic["igeninteger8bit"] = t_igen_integer_8bit

    t_ugen_integer_8bit = argtype("ugeninteger8bit", "NULL", "NULL", 0, ["uint8_t","sycl::vec<uint8_t, 2>","sycl::vec<uint8_t, 3>","sycl::vec<uint8_t, 4>","sycl::vec<uint8_t, 8>","sycl::vec<uint8_t, 16>"])
    type_dic["ugeninteger8bit"] = t_ugen_integer_8bit

    t_gen_integer_8bit = argtype("geninteger8bit", "NULL", "NULL", 0, ["igeninteger8bit","ugeninteger8bit"])
    type_dic["geninteger8bit"] = t_gen_integer_8bit

    t_igen_integer_16bit = argtype("igeninteger16bit", "NULL", "NULL", 0, ["int16_t","sycl::vec<int16_t, 2>","sycl::vec<int16_t, 3>","sycl::vec<int16_t, 4>","sycl::vec<int16_t, 8>","sycl::vec<int16_t, 16>"])
    type_dic["igeninteger16bit"] = t_igen_integer_16bit

    t_ugen_integer_16bit = argtype("ugeninteger16bit", "NULL", "NULL", 0, ["uint16_t","sycl::vec<uint16_t, 2>","sycl::vec<uint16_t, 3>","sycl::vec<uint16_t, 4>","sycl::vec<uint16_t, 8>","sycl::vec<uint16_t, 16>"])
    type_dic["ugeninteger16bit"] = t_ugen_integer_16bit

    t_gen_integer_16bit = argtype("geninteger16bit", "NULL", "NULL", 0, ["igeninteger16bit","ugeninteger16bit"])
    type_dic["geninteger16bit"] = t_gen_integer_16bit

    t_igen_integer_32bit = argtype("igeninteger32bit", "NULL", "NULL", 0, ["int32_t","sycl::vec<int32_t, 2>","sycl::vec<int32_t, 3>","sycl::vec<int32_t, 4>","sycl::vec<int32_t, 8>","sycl::vec<int32_t, 16>"])
    type_dic["igeninteger32bit"] = t_igen_integer_32bit

    t_ugen_integer_32bit = argtype("ugeninteger32bit", "NULL", "NULL", 0, ["uint32_t","sycl::vec<uint32_t, 2>","sycl::vec<uint32_t, 3>","sycl::vec<uint32_t, 4>","sycl::vec<uint32_t, 8>","sycl::vec<uint32_t, 16>"])
    type_dic["ugeninteger32bit"] = t_ugen_integer_32bit

    t_gen_integer_32bit = argtype("geninteger32bit", "NULL", "NULL", 0, ["igeninteger32bit","ugeninteger32bit"])
    type_dic["geninteger32bit"] = t_gen_integer_32bit

    t_igen_integer_64bit = argtype("igeninteger64bit", "NULL", "NULL", 0, ["int64_t","sycl::vec<int64_t, 2>","sycl::vec<int64_t, 3>","sycl::vec<int64_t, 4>","sycl::vec<int64_t, 8>","sycl::vec<int64_t, 16>"])
    type_dic["igeninteger64bit"] = t_igen_integer_64bit

    t_ugen_integer_64bit = argtype("ugeninteger64bit", "NULL", "NULL", 0, ["uint64_t","sycl::vec<uint64_t, 2>","sycl::vec<uint64_t, 3>","sycl::vec<uint64_t, 4>","sycl::vec<uint64_t, 8>","sycl::vec<uint64_t, 16>"])
    type_dic["ugeninteger64bit"] = t_ugen_integer_64bit

    t_gen_integer_64bit = argtype("geninteger64bit", "NULL", "NULL", 0, ["igeninteger64bit","ugeninteger64bit"])
    type_dic["geninteger64bit"] = t_gen_integer_64bit


    return type_dic
