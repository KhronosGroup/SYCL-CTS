"""Represents an argument or return type. The type may be generic."""
class argtype:
    def __init__(self, name, var_type, base_type, dim, child_types, unsigned=False):
        self.name = name # Type name. This works also as an identifier for the type.
        self.var_type = var_type # Variable type (scalar,vector or NULL). Generic types should have a NULL variable type.
        self.base_type = base_type # The base type (e.g. the base type of cl::sycl::double3 is double). Generic types should have a NULL base type.
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

    t_float_2 = argtype("cl::sycl::float2", "vector", "float", 2, [])
    type_dic["cl::sycl::float2"] = t_float_2

    t_float_3 = argtype("cl::sycl::float3", "vector", "float", 3, [])
    type_dic["cl::sycl::float3"] = t_float_3

    t_float_4 = argtype("cl::sycl::float4", "vector", "float", 4, [])
    type_dic["cl::sycl::float4"] = t_float_4

    t_float_8 = argtype("cl::sycl::float8", "vector", "float", 8, [])
    type_dic["cl::sycl::float8"] = t_float_8

    t_float_16 = argtype("cl::sycl::float16", "vector", "float", 16, [])
    type_dic["cl::sycl::float16"] = t_float_16
    
    
    t_double_0 = argtype("double", "scalar", "double", 1, [])
    type_dic["double"] = t_double_0

    t_double_2 = argtype("cl::sycl::double2", "vector", "double", 2, [])
    type_dic["cl::sycl::double2"] = t_double_2

    t_double_3 = argtype("cl::sycl::double3", "vector", "double", 3, [])
    type_dic["cl::sycl::double3"] = t_double_3

    t_double_4 = argtype("cl::sycl::double4", "vector", "double", 4, [])
    type_dic["cl::sycl::double4"] = t_double_4

    t_double_8 = argtype("cl::sycl::double8", "vector", "double", 8, [])
    type_dic["cl::sycl::double8"] = t_double_8

    t_double_16 = argtype("cl::sycl::double16", "vector", "double", 16, [])
    type_dic["cl::sycl::double16"] = t_double_16

    
    t_half_0 = argtype("cl::sycl::half", "scalar", "cl::sycl::half", 1, [])
    type_dic["cl::sycl::half"] = t_half_0

    t_half_2 = argtype("cl::sycl::half2", "vector", "cl::sycl::half", 2, [])
    type_dic["cl::sycl::half2"] = t_half_2

    t_half_3 = argtype("cl::sycl::half3", "vector", "cl::sycl::half", 3, [])
    type_dic["cl::sycl::half3"] = t_half_3

    t_half_4 = argtype("cl::sycl::half4", "vector", "cl::sycl::half", 4, [])
    type_dic["cl::sycl::half4"] = t_half_4

    t_half_8 = argtype("cl::sycl::half8", "vector", "cl::sycl::half", 8, [])
    type_dic["cl::sycl::half8"] = t_half_8

    t_half_16 = argtype("cl::sycl::half16", "vector", "cl::sycl::half", 16, [])
    type_dic["cl::sycl::half16"] = t_half_16
    

    t_char_0 = argtype("char", "scalar", "char", 1, [])
    type_dic["char"] = t_char_0

    t_char_2 = argtype("cl::sycl::char2", "vector", "char", 2, [])
    type_dic["cl::sycl::char2"] = t_char_2
    
    t_char_3 = argtype("cl::sycl::char3", "vector", "char", 3, [])
    type_dic["cl::sycl::char3"] = t_char_3

    t_char_4 = argtype("cl::sycl::char4", "vector", "char", 4, [])
    type_dic["cl::sycl::char4"] = t_char_4

    t_char_8 = argtype("cl::sycl::char8", "vector", "char", 8, [])
    type_dic["cl::sycl::char8"] = t_char_8

    t_char_16 = argtype("cl::sycl::char16", "vector", "char", 16, [])
    type_dic["cl::sycl::char16"] = t_char_16


    t_schar_0 = argtype("signed char", "scalar", "char", 1, [])
    type_dic["signed char"] = t_schar_0

    t_schar_2 = argtype("cl::sycl::schar2", "vector", "char", 2, [])
    type_dic["cl::sycl::schar2"] = t_schar_2
    
    t_schar_3 = argtype("cl::sycl::schar3", "vector", "char", 3, [])
    type_dic["cl::sycl::schar3"] = t_schar_3

    t_schar_4 = argtype("cl::sycl::schar4", "vector", "char", 4, [])
    type_dic["cl::sycl::schar4"] = t_schar_4

    t_schar_8 = argtype("cl::sycl::schar8", "vector", "char", 8, [])
    type_dic["cl::sycl::schar8"] = t_schar_8

    t_schar_16 = argtype("cl::sycl::schar16", "vector", "char", 16, [])
    type_dic["cl::sycl::schar16"] = t_schar_16


    t_uchar_0 = argtype("unsigned char", "scalar", "char", 1, [], True)
    type_dic["unsigned char"] = t_uchar_0
    
    t_uchar_2 = argtype("cl::sycl::uchar2", "vector", "char", 2, [], True)
    type_dic["cl::sycl::uchar2"] = t_uchar_2
    
    t_uchar_3 = argtype("cl::sycl::uchar3", "vector", "char", 3, [], True)
    type_dic["cl::sycl::uchar3"] = t_uchar_3

    t_uchar_4 = argtype("cl::sycl::uchar4", "vector", "char", 4, [], True)
    type_dic["cl::sycl::uchar4"] = t_uchar_4

    t_uchar_8 = argtype("cl::sycl::uchar8", "vector", "char", 8, [], True)
    type_dic["cl::sycl::uchar8"] = t_uchar_8

    t_uchar_16 = argtype("cl::sycl::uchar16", "vector", "char", 16, [], True)
    type_dic["cl::sycl::uchar16"] = t_uchar_16
    
    
    t_short_0 = argtype("short", "scalar", "short", 1, [])
    type_dic["short"] = t_short_0

    t_short_2 = argtype("cl::sycl::short2", "vector", "short", 2, [])
    type_dic["cl::sycl::short2"] = t_short_2

    t_short_3 = argtype("cl::sycl::short3", "vector", "short", 3, [])
    type_dic["cl::sycl::short3"] = t_short_3

    t_short_4 = argtype("cl::sycl::short4", "vector", "short", 4, [])
    type_dic["cl::sycl::short4"] = t_short_4

    t_short_8 = argtype("cl::sycl::short8", "vector", "short", 8, [])
    type_dic["cl::sycl::short8"] = t_short_8

    t_short_16 = argtype("cl::sycl::short16", "vector", "short", 16, [])
    type_dic["cl::sycl::short16"] = t_short_16

    
    t_ushort_0 = argtype("unsigned short", "scalar", "short", 1, [], True)
    type_dic["unsigned short"] = t_ushort_0
    
    t_ushort_2 = argtype("cl::sycl::ushort2", "vector", "short", 2, [], True)
    type_dic["cl::sycl::ushort2"] = t_ushort_2

    t_ushort_3 = argtype("cl::sycl::ushort3", "vector", "short", 3, [], True)
    type_dic["cl::sycl::ushort3"] = t_ushort_3

    t_ushort_4 = argtype("cl::sycl::ushort4", "vector", "short", 4, [], True)
    type_dic["cl::sycl::ushort4"] = t_ushort_4

    t_ushort_8 = argtype("cl::sycl::ushort8", "vector", "short", 8, [], True)
    type_dic["cl::sycl::ushort8"] = t_ushort_8

    t_ushort_16 = argtype("cl::sycl::ushort16", "vector", "short", 16, [], True)
    type_dic["cl::sycl::ushort16"] = t_ushort_16


    t_int_0 = argtype("int", "scalar", "int", 1, [])
    type_dic["int"] = t_int_0

    t_int_2 = argtype("cl::sycl::int2", "vector", "int", 2, [])
    type_dic["cl::sycl::int2"] = t_int_2

    t_int_3 = argtype("cl::sycl::int3", "vector", "int", 3, [])
    type_dic["cl::sycl::int3"] = t_int_3

    t_int_4 = argtype("cl::sycl::int4", "vector", "int", 4, [])
    type_dic["cl::sycl::int4"] = t_int_4

    t_int_8 = argtype("cl::sycl::int8", "vector", "int", 8, [])
    type_dic["cl::sycl::int8"] = t_int_8

    t_int_16 = argtype("cl::sycl::int16", "vector", "int", 16, [])
    type_dic["cl::sycl::int16"] = t_int_16


    t_uint_0 = argtype("unsigned int", "scalar", "int", 1, [], True)
    type_dic["unsigned int"] = t_uint_0

    t_uint_2 = argtype("cl::sycl::uint2", "vector", "int", 2, [], True)
    type_dic["cl::sycl::uint2"] = t_uint_2

    t_uint_3 = argtype("cl::sycl::uint3", "vector", "int", 3, [], True)
    type_dic["cl::sycl::uint3"] = t_uint_3

    t_uint_4 = argtype("cl::sycl::uint4", "vector", "int", 4, [], True)
    type_dic["cl::sycl::uint4"] = t_uint_4

    t_uint_8 = argtype("cl::sycl::uint8", "vector", "int", 8, [], True)
    type_dic["cl::sycl::uint8"] = t_uint_8

    t_uint_16 = argtype("cl::sycl::uint16", "vector", "int", 16, [], True)
    type_dic["cl::sycl::uint16"] = t_uint_16


    t_long_0 = argtype("long int", "scalar", "long int", 1, [])
    type_dic["long int"] = t_long_0

    t_long_2 = argtype("cl::sycl::long2", "vector", "long int", 2, [])
    type_dic["cl::sycl::long2"] = t_long_2

    t_long_3 = argtype("cl::sycl::long3", "vector", "long int", 3, [])
    type_dic["cl::sycl::long3"] = t_long_3

    t_long_4 = argtype("cl::sycl::long4", "vector", "long int", 4, [])
    type_dic["cl::sycl::long4"] = t_long_4

    t_long_8 = argtype("cl::sycl::long8", "vector", "long int", 8, [])
    type_dic["cl::sycl::long8"] = t_long_8

    t_long_16 = argtype("cl::sycl::long16", "vector", "long int", 16, [])
    type_dic["cl::sycl::long16"] = t_long_16


    t_ulong_0 = argtype("unsigned long int", "scalar", "long int", 1, [], True)
    type_dic["unsigned long int"] = t_ulong_0

    t_ulong_2 = argtype("cl::sycl::ulong2", "vector", "long int", 2, [], True)
    type_dic["cl::sycl::ulong2"] = t_ulong_2

    t_ulong_3 = argtype("cl::sycl::ulong3", "vector", "long int", 3, [], True)
    type_dic["cl::sycl::ulong3"] = t_ulong_3

    t_ulong_4 = argtype("cl::sycl::ulong4", "vector", "long int", 4, [], True)
    type_dic["cl::sycl::ulong4"] = t_ulong_4

    t_ulong_8 = argtype("cl::sycl::ulong8", "vector", "long int", 8, [], True)
    type_dic["cl::sycl::ulong8"] = t_ulong_8

    t_ulong_16 = argtype("cl::sycl::ulong16", "vector", "long int", 16, [], True)
    type_dic["cl::sycl::ulong16"] = t_ulong_16


    t_longlong_0 = argtype("long long int", "scalar", "long long int", 1, [])
    type_dic["long long int"] = t_longlong_0

    t_longlong_2 = argtype("cl::sycl::longlong2", "vector", "long long int", 2, [])
    type_dic["cl::sycl::longlong2"] = t_longlong_2

    t_longlong_3 = argtype("cl::sycl::longlong3", "vector", "long long int", 3, [])
    type_dic["cl::sycl::longlong3"] = t_longlong_3

    t_longlong_4 = argtype("cl::sycl::longlong4", "vector", "long long int", 4, [])
    type_dic["cl::sycl::longlong4"] = t_longlong_4

    t_longlong_8 = argtype("cl::sycl::longlong8", "vector", "long long int", 8, [])
    type_dic["cl::sycl::longlong8"] = t_longlong_8

    t_longlong_16 = argtype("cl::sycl::longlong16", "vector", "long long int", 16, [])
    type_dic["cl::sycl::longlong16"] = t_longlong_16

    
    t_ulonglong_0 = argtype("unsigned long long int", "scalar", "long long int", 1, [], True)
    type_dic["unsigned long long int"] = t_ulonglong_0

    t_ulonglong_2 = argtype("cl::sycl::ulonglong2", "vector", "long long int", 2, [], True)
    type_dic["cl::sycl::ulonglong2"] = t_ulonglong_2

    t_ulonglong_3 = argtype("cl::sycl::ulonglong3", "vector", "long long int", 3, [], True)
    type_dic["cl::sycl::ulonglong3"] = t_ulonglong_3

    t_ulonglong_4 = argtype("cl::sycl::ulonglong4", "vector", "long long int", 4, [], True)
    type_dic["cl::sycl::ulonglong4"] = t_ulonglong_4

    t_ulonglong_8 = argtype("cl::sycl::ulonglong8", "vector", "long long int", 8, [], True)
    type_dic["cl::sycl::ulonglong8"] = t_ulonglong_8

    t_ulonglong_16 = argtype("cl::sycl::ulonglong16", "vector", "long long int", 16, [], True)
    type_dic["cl::sycl::ulonglong16"] = t_ulonglong_16


    # Fixed size basic types

    t_int8t_0 = argtype("int8_t", "scalar", "int8_t", 1, [])
    type_dic["int8_t"] = t_int8t_0

    t_int8t_2 = argtype("cl::sycl::vec<int8_t, 2>", "vector", "int8_t", 2, [])
    type_dic["cl::sycl::vec<int8_t, 2>"] = t_int8t_2

    t_int8t_3 = argtype("cl::sycl::vec<int8_t, 3>", "vector", "int8_t", 3, [])
    type_dic["cl::sycl::vec<int8_t, 3>"] = t_int8t_3

    t_int8t_4 = argtype("cl::sycl::vec<int8_t, 4>", "vector", "int8_t", 4, [])
    type_dic["cl::sycl::vec<int8_t, 4>"] = t_int8t_4

    t_int8t_8 = argtype("cl::sycl::vec<int8_t, 8>", "vector", "int8_t", 8, [])
    type_dic["cl::sycl::vec<int8_t, 8>"] = t_int8t_8

    t_int8t_16 = argtype("cl::sycl::vec<int8_t, 16>", "vector", "int8_t", 16, [])
    type_dic["cl::sycl::vec<int8_t, 16>"] = t_int8t_16


    t_uint8t_0 = argtype("uint8_t", "scalar", "int8_t", 1, [], True)
    type_dic["uint8_t"] = t_uint8t_0

    t_uint8t_2 = argtype("cl::sycl::vec<uint8_t, 2>", "vector", "int8_t", 2, [], True)
    type_dic["cl::sycl::vec<uint8_t, 2>"] = t_uint8t_2

    t_uint8t_3 = argtype("cl::sycl::vec<uint8_t, 3>", "vector", "int8_t", 3, [], True)
    type_dic["cl::sycl::vec<uint8_t, 3>"] = t_uint8t_3

    t_uint8t_4 = argtype("cl::sycl::vec<uint8_t, 4>", "vector", "int8_t", 4, [], True)
    type_dic["cl::sycl::vec<uint8_t, 4>"] = t_uint8t_4

    t_uint8t_8 = argtype("cl::sycl::vec<uint8_t, 8>", "vector", "int8_t", 8, [], True)
    type_dic["cl::sycl::vec<uint8_t, 8>"] = t_uint8t_8

    t_uint8t_16 = argtype("cl::sycl::vec<uint8_t, 16>", "vector", "int8_t", 16, [], True)
    type_dic["cl::sycl::vec<uint8_t, 16>"] = t_uint8t_16


    t_int16t_0 = argtype("int16_t", "scalar", "int16_t", 1, [])
    type_dic["int16_t"] = t_int16t_0

    t_int16t_2 = argtype("cl::sycl::vec<int16_t, 2>", "vector", "int16_t", 2, [])
    type_dic["cl::sycl::vec<int16_t, 2>"] = t_int16t_2

    t_int16t_3 = argtype("cl::sycl::vec<int16_t, 3>", "vector", "int16_t", 3, [])
    type_dic["cl::sycl::vec<int16_t, 3>"] = t_int16t_3

    t_int16t_4 = argtype("cl::sycl::vec<int16_t, 4>", "vector", "int16_t", 4, [])
    type_dic["cl::sycl::vec<int16_t, 4>"] = t_int16t_4

    t_int16t_8 = argtype("cl::sycl::vec<int16_t, 8>", "vector", "int16_t", 8, [])
    type_dic["cl::sycl::vec<int16_t, 8>"] = t_int16t_8

    t_int16t_16 = argtype("cl::sycl::vec<int16_t, 16>", "vector", "int16_t", 16, [])
    type_dic["cl::sycl::vec<int16_t, 16>"] = t_int16t_16


    t_uint16t_0 = argtype("uint16_t", "scalar", "int16_t", 1, [], True)
    type_dic["uint16_t"] = t_uint16t_0

    t_uint16t_2 = argtype("cl::sycl::vec<uint16_t, 2>", "vector", "int16_t", 2, [], True)
    type_dic["cl::sycl::vec<uint16_t, 2>"] = t_uint16t_2

    t_uint16t_3 = argtype("cl::sycl::vec<uint16_t, 3>", "vector", "int16_t", 3, [], True)
    type_dic["cl::sycl::vec<uint16_t, 3>"] = t_uint16t_3

    t_uint16t_4 = argtype("cl::sycl::vec<uint16_t, 4>", "vector", "int16_t", 4, [], True)
    type_dic["cl::sycl::vec<uint16_t, 4>"] = t_uint16t_4

    t_uint16t_8 = argtype("cl::sycl::vec<uint16_t, 8>", "vector", "int16_t", 8, [], True)
    type_dic["cl::sycl::vec<uint16_t, 8>"] = t_uint16t_8

    t_uint16t_16 = argtype("cl::sycl::vec<uint16_t, 16>", "vector", "int16_t", 16, [], True)
    type_dic["cl::sycl::vec<uint16_t, 16>"] = t_uint16t_16


    t_int32t_0 = argtype("int32_t", "scalar", "int32_t", 1, [])
    type_dic["int32_t"] = t_int32t_0

    t_int32t_2 = argtype("cl::sycl::vec<int32_t, 2>", "vector", "int32_t", 2, [])
    type_dic["cl::sycl::vec<int32_t, 2>"] = t_int32t_2

    t_int32t_3 = argtype("cl::sycl::vec<int32_t, 3>", "vector", "int32_t", 3, [])
    type_dic["cl::sycl::vec<int32_t, 3>"] = t_int32t_3

    t_int32t_4 = argtype("cl::sycl::vec<int32_t, 4>", "vector", "int32_t", 4, [])
    type_dic["cl::sycl::vec<int32_t, 4>"] = t_int32t_4

    t_int32t_8 = argtype("cl::sycl::vec<int32_t, 8>", "vector", "int32_t", 8, [])
    type_dic["cl::sycl::vec<int32_t, 8>"] = t_int32t_8

    t_int32t_16 = argtype("cl::sycl::vec<int32_t, 16>", "vector", "int32_t", 16, [])
    type_dic["cl::sycl::vec<int32_t, 16>"] = t_int32t_16


    t_uint32t_0 = argtype("uint32_t", "scalar", "int32_t", 1, [], True)
    type_dic["uint32_t"] = t_uint32t_0

    t_uint32t_2 = argtype("cl::sycl::vec<uint32_t, 2>", "vector", "int32_t", 2, [], True)
    type_dic["cl::sycl::vec<uint32_t, 2>"] = t_uint32t_2

    t_uint32t_3 = argtype("cl::sycl::vec<uint32_t, 3>", "vector", "int32_t", 3, [], True)
    type_dic["cl::sycl::vec<uint32_t, 3>"] = t_uint32t_3

    t_uint32t_4 = argtype("cl::sycl::vec<uint32_t, 4>", "vector", "int32_t", 4, [], True)
    type_dic["cl::sycl::vec<uint32_t, 4>"] = t_uint32t_4

    t_uint32t_8 = argtype("cl::sycl::vec<uint32_t, 8>", "vector", "int32_t", 8, [], True)
    type_dic["cl::sycl::vec<uint32_t, 8>"] = t_uint32t_8

    t_uint32t_16 = argtype("cl::sycl::vec<uint32_t, 16>", "vector", "int32_t", 16, [], True)
    type_dic["cl::sycl::vec<uint32_t, 16>"] = t_uint32t_16


    t_int64t_0 = argtype("int64_t", "scalar", "int64_t", 1, [])
    type_dic["int64_t"] = t_int64t_0

    t_int64t_2 = argtype("cl::sycl::vec<int64_t, 2>", "vector", "int64_t", 2, [])
    type_dic["cl::sycl::vec<int64_t, 2>"] = t_int64t_2

    t_int64t_3 = argtype("cl::sycl::vec<int64_t, 3>", "vector", "int64_t", 3, [])
    type_dic["cl::sycl::vec<int64_t, 3>"] = t_int64t_3

    t_int64t_4 = argtype("cl::sycl::vec<int64_t, 4>", "vector", "int64_t", 4, [])
    type_dic["cl::sycl::vec<int64_t, 4>"] = t_int64t_4

    t_int64t_8 = argtype("cl::sycl::vec<int64_t, 8>", "vector", "int64_t", 8, [])
    type_dic["cl::sycl::vec<int64_t, 8>"] = t_int64t_8

    t_int64t_16 = argtype("cl::sycl::vec<int64_t, 16>", "vector", "int64_t", 16, [])
    type_dic["cl::sycl::vec<int64_t, 16>"] = t_int64t_16


    t_uint64t_0 = argtype("uint64_t", "scalar", "int64_t", 1, [], True)
    type_dic["uint64_t"] = t_uint64t_0

    t_uint64t_2 = argtype("cl::sycl::vec<uint64_t, 2>", "vector", "int64_t", 2, [], True)
    type_dic["cl::sycl::vec<uint64_t, 2>"] = t_uint64t_2

    t_uint64t_3 = argtype("cl::sycl::vec<uint64_t, 3>", "vector", "int64_t", 3, [], True)
    type_dic["cl::sycl::vec<uint64_t, 3>"] = t_uint64t_3

    t_uint64t_4 = argtype("cl::sycl::vec<uint64_t, 4>", "vector", "int64_t", 4, [], True)
    type_dic["cl::sycl::vec<uint64_t, 4>"] = t_uint64t_4

    t_uint64t_8 = argtype("cl::sycl::vec<uint64_t, 8>", "vector", "int64_t", 8, [], True)
    type_dic["cl::sycl::vec<uint64_t, 8>"] = t_uint64t_8

    t_uint64t_16 = argtype("cl::sycl::vec<uint64_t, 16>", "vector", "int64_t", 16, [], True)
    type_dic["cl::sycl::vec<uint64_t, 16>"] = t_uint64t_16

    return type_dic
    

    # Generic Types
def create_types():
    type_dic = create_basic_types()
    
    t_float_n = argtype("floatn", "NULL", "NULL", 0, ["cl::sycl::float2","cl::sycl::float3","cl::sycl::float4","cl::sycl::float8","cl::sycl::float16"])
    type_dic["floatn"] = t_float_n
    
    t_gen_float_f = argtype("genfloatf", "NULL", "NULL", 0, ["float", "floatn"])
    type_dic["genfloatf"] = t_gen_float_f

    t_double_n = argtype("doublen", "NULL", "NULL", 0, ["cl::sycl::double2","cl::sycl::double3","cl::sycl::double4","cl::sycl::double8","cl::sycl::double16"])
    type_dic["doublen"] = t_double_n
    
    t_gen_float_d = argtype("genfloatd", "NULL", "NULL", 0, ["double","doublen"])
    type_dic["genfloatd"] = t_gen_float_d
    
    t_half_n = argtype("halfn", "NULL", "NULL", 0, ["cl::sycl::half2","cl::sycl::half3","cl::sycl::half4","cl::sycl::half8","cl::sycl::half16"])
    type_dic["halfn"] = t_half_n

    t_gen_float_h = argtype("genfloath", "NULL", "NULL", 0, ["cl::sycl::half","halfn"])
    type_dic["genfloath"] = t_gen_float_h

    t_gen_float = argtype("genfloat", "NULL", "NULL", 0, ["genfloatf","genfloatd","genfloath"])
    type_dic["genfloat"] = t_gen_float

    t_sgen_float = argtype("sgenfloat", "NULL", "NULL", 0, ["float","double","cl::sycl::half"])
    type_dic["sgenfloat"] = t_sgen_float

    t_gen_geofloat = argtype("gengeofloat", "NULL", "NULL", 0, ["float","cl::sycl::float2","cl::sycl::float3","cl::sycl::float4"])
    type_dic["gengeofloat"] = t_gen_geofloat

    t_gen_geodouble = argtype("gengeodouble", "NULL", "NULL", 0, ["double","cl::sycl::double2","cl::sycl::double3","cl::sycl::double4"])
    type_dic["gengeodouble"] = t_gen_geodouble

    t_char_n = argtype("charn", "NULL", "NULL", 0, ["cl::sycl::char2","cl::sycl::char3","cl::sycl::char4","cl::sycl::char8","cl::sycl::char16"])
    type_dic["charn"] = t_char_n

    t_schar_n = argtype("scharn", "NULL", "NULL", 0, ["cl::sycl::schar2","cl::sycl::schar3","cl::sycl::schar4","cl::sycl::schar8","cl::sycl::schar16"])
    type_dic["scharn"] = t_schar_n

    t_uchar_n = argtype("ucharn", "NULL", "NULL", 0, ["cl::sycl::uchar2","cl::sycl::uchar3","cl::sycl::uchar4","cl::sycl::uchar8","cl::sycl::uchar16"])
    type_dic["ucharn"] = t_uchar_n

    t_igen_char = argtype("igenchar", "NULL", "NULL", 0, ["signed char","scharn"])
    type_dic["igenchar"] = t_igen_char

    t_ugen_char = argtype("ugenchar", "NULL", "NULL", 0, ["unsigned char","ucharn"])
    type_dic["ugenchar"] = t_ugen_char

    t_gen_char = argtype("genchar", "NULL", "NULL", 0, ["char","charn","igenchar","ugenchar"])
    type_dic["genchar"] = t_gen_char

    t_short_n = argtype("shortn", "NULL", "NULL", 0, ["cl::sycl::short2","cl::sycl::short3","cl::sycl::short4","cl::sycl::short8","cl::sycl::short16"])
    type_dic["shortn"] = t_short_n 
    
    t_gen_short = argtype("genshort", "NULL", "NULL", 0, ["short","shortn"])
    type_dic["genshort"] = t_gen_short
    
    t_ushort_n = argtype("ushortn", "NULL", "NULL", 0, ["cl::sycl::ushort2","cl::sycl::ushort3","cl::sycl::ushort4","cl::sycl::ushort8","cl::sycl::ushort16"])
    type_dic["ushortn"] = t_ushort_n 
    
    t_ugen_short = argtype("ugenshort", "NULL", "NULL", 0, ["unsigned short","ushortn"])
    type_dic["ugenshort"] = t_ugen_short

    t_uint_n = argtype("uintn", "NULL", "NULL", 0, ["cl::sycl::uint2","cl::sycl::uint3","cl::sycl::uint4","cl::sycl::uint8","cl::sycl::uint16"])
    type_dic["uintn"] = t_uint_n

    t_ugen_int = argtype("ugenint", "NULL", "NULL", 0, ["unsigned int","uintn"])
    type_dic["ugenint"] = t_ugen_int

    t_int_n = argtype("intn", "NULL", "NULL", 0, ["cl::sycl::int2","cl::sycl::int3","cl::sycl::int4","cl::sycl::int8","cl::sycl::int16"])
    type_dic["intn"] = t_int_n

    t_gen_int = argtype("genint", "NULL", "NULL", 0, ["int","intn"])
    type_dic["genint"] = t_gen_int

    t_ulong_n = argtype("ulongn", "NULL", "NULL", 0, ["cl::sycl::ulong2","cl::sycl::ulong3","cl::sycl::ulong4","cl::sycl::ulong8","cl::sycl::ulong16"])
    type_dic["ulongn"] = t_ulong_n

    t_ugen_long = argtype("ugenlong", "NULL", "NULL", 0, ["unsigned long int", "ulongn"])
    type_dic["ugenlong"] = t_ugen_long

    t_long_n = argtype("longn", "NULL", "NULL", 0, ["cl::sycl::long2","cl::sycl::long3","cl::sycl::long4","cl::sycl::long8","cl::sycl::long16"])
    type_dic["longn"] = t_long_n

    t_gen_long = argtype("genlong", "NULL", "NULL", 0, ["long int", "longn"])
    type_dic["genlong"] = t_gen_long

    t_ulonglong_n = argtype("ulonglongn", "NULL", "NULL", 0, ["cl::sycl::ulonglong2","cl::sycl::ulonglong3","cl::sycl::ulonglong4","cl::sycl::ulonglong8","cl::sycl::ulonglong16"])
    type_dic["ulonglongn"] = t_ulonglong_n

    t_ugen_longlong = argtype("ugenlonglong", "NULL", "NULL", 0, ["unsigned long long int", "ulonglongn"])
    type_dic["ugenlonglong"] = t_ugen_longlong

    t_longlong_n = argtype("longlongn", "NULL", "NULL", 0, ["cl::sycl::longlong2","cl::sycl::longlong3","cl::sycl::longlong4","cl::sycl::longlong8","cl::sycl::longlong16"])
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

    t_igen_integer_8bit = argtype("igeninteger8bit", "NULL", "NULL", 0, ["int8_t","cl::sycl::vec<int8_t, 2>","cl::sycl::vec<int8_t, 3>","cl::sycl::vec<int8_t, 4>","cl::sycl::vec<int8_t, 8>","cl::sycl::vec<int8_t, 16>"])
    type_dic["igeninteger8bit"] = t_igen_integer_8bit

    t_ugen_integer_8bit = argtype("ugeninteger8bit", "NULL", "NULL", 0, ["uint8_t","cl::sycl::vec<uint8_t, 2>","cl::sycl::vec<uint8_t, 3>","cl::sycl::vec<uint8_t, 4>","cl::sycl::vec<uint8_t, 8>","cl::sycl::vec<uint8_t, 16>"])
    type_dic["ugeninteger8bit"] = t_ugen_integer_8bit

    t_gen_integer_8bit = argtype("geninteger8bit", "NULL", "NULL", 0, ["igeninteger8bit","ugeninteger8bit"])
    type_dic["geninteger8bit"] = t_gen_integer_8bit

    t_igen_integer_16bit = argtype("igeninteger16bit", "NULL", "NULL", 0, ["int16_t","cl::sycl::vec<int16_t, 2>","cl::sycl::vec<int16_t, 3>","cl::sycl::vec<int16_t, 4>","cl::sycl::vec<int16_t, 8>","cl::sycl::vec<int16_t, 16>"])
    type_dic["igeninteger16bit"] = t_igen_integer_16bit

    t_ugen_integer_16bit = argtype("ugeninteger16bit", "NULL", "NULL", 0, ["uint16_t","cl::sycl::vec<uint16_t, 2>","cl::sycl::vec<uint16_t, 3>","cl::sycl::vec<uint16_t, 4>","cl::sycl::vec<uint16_t, 8>","cl::sycl::vec<uint16_t, 16>"])
    type_dic["ugeninteger16bit"] = t_ugen_integer_16bit

    t_gen_integer_16bit = argtype("geninteger16bit", "NULL", "NULL", 0, ["igeninteger16bit","ugeninteger16bit"])
    type_dic["geninteger16bit"] = t_gen_integer_16bit

    t_igen_integer_32bit = argtype("igeninteger32bit", "NULL", "NULL", 0, ["int32_t","cl::sycl::vec<int32_t, 2>","cl::sycl::vec<int32_t, 3>","cl::sycl::vec<int32_t, 4>","cl::sycl::vec<int32_t, 8>","cl::sycl::vec<int32_t, 16>"])
    type_dic["igeninteger32bit"] = t_igen_integer_32bit

    t_ugen_integer_32bit = argtype("ugeninteger32bit", "NULL", "NULL", 0, ["uint32_t","cl::sycl::vec<uint32_t, 2>","cl::sycl::vec<uint32_t, 3>","cl::sycl::vec<uint32_t, 4>","cl::sycl::vec<uint32_t, 8>","cl::sycl::vec<uint32_t, 16>"])
    type_dic["ugeninteger32bit"] = t_ugen_integer_32bit

    t_gen_integer_32bit = argtype("geninteger32bit", "NULL", "NULL", 0, ["igeninteger32bit","ugeninteger32bit"])
    type_dic["geninteger32bit"] = t_gen_integer_32bit

    t_igen_integer_64bit = argtype("igeninteger64bit", "NULL", "NULL", 0, ["int64_t","cl::sycl::vec<int64_t, 2>","cl::sycl::vec<int64_t, 3>","cl::sycl::vec<int64_t, 4>","cl::sycl::vec<int64_t, 8>","cl::sycl::vec<int64_t, 16>"])
    type_dic["igeninteger64bit"] = t_igen_integer_64bit

    t_ugen_integer_64bit = argtype("ugeninteger64bit", "NULL", "NULL", 0, ["uint64_t","cl::sycl::vec<uint64_t, 2>","cl::sycl::vec<uint64_t, 3>","cl::sycl::vec<uint64_t, 4>","cl::sycl::vec<uint64_t, 8>","cl::sycl::vec<uint64_t, 16>"])
    type_dic["ugeninteger64bit"] = t_ugen_integer_64bit

    t_gen_integer_64bit = argtype("geninteger64bit", "NULL", "NULL", 0, ["igeninteger64bit","ugeninteger64bit"])
    type_dic["geninteger64bit"] = t_gen_integer_64bit


    return type_dic
