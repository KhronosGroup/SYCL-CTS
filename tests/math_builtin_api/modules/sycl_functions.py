"""Represents a function signature."""
class funsig:
    def __init__(self, namespace, ret_type, name, arg_types=[], pntr_indx=[]):
        self.namespace = namespace # Namespace of function.
        self.ret_type = ret_type # Function return type.
        self.name = name # Function name.
        self.arg_types = arg_types # List containing the function argument types.
        self.pntr_indx = pntr_indx # List containing the indexes of the arguments which are pointers.
    def __eq__(self, other):
        if isinstance(other, funsig):
            return ((self.namespace == other.namespace) and
                    (self.ret_type == other.ret_type) and
                    (self.name == other.name) and
                    (self.arg_types == other.arg_types) and
                    (self.pntr_indx == other.pntr_indx))
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash((self.namespace, self.ret_type, self.name, str(self.arg_types), str(self.pntr_indx)))

def create_integer_signatures():
    sig_list = []

    f_abs = funsig("cl::sycl", "ugeninteger", "abs", ["geninteger"])
    sig_list.append(f_abs)
    
    f_abs_diff = funsig("cl::sycl", "ugeninteger", "abs_diff", ["geninteger", "geninteger"])
    sig_list.append(f_abs_diff)
    
    f_add_sat = funsig("cl::sycl", "geninteger", "add_sat", ["geninteger", "geninteger"])
    sig_list.append(f_add_sat)
    
    f_hadd = funsig("cl::sycl", "geninteger", "hadd", ["geninteger", "geninteger"])
    sig_list.append(f_hadd)

    f_rhadd = funsig("cl::sycl", "geninteger", "rhadd", ["geninteger", "geninteger"])
    sig_list.append(f_rhadd)

    f_clamp = funsig("cl::sycl", "geninteger", "clamp", ["geninteger", "sgeninteger", "sgeninteger"])
    sig_list.append(f_clamp)
    
    f_clamp_2 = funsig("cl::sycl", "geninteger", "clamp", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_clamp_2)

    f_clz = funsig("cl::sycl", "geninteger", "clz", ["geninteger"])
    sig_list.append(f_clz)

    f_mad_hi = funsig("cl::sycl", "geninteger", "mad_hi", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_mad_hi)

    f_mad_sat = funsig("cl::sycl", "geninteger", "mad_sat", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_mad_sat)

    f_max = funsig("cl::sycl", "geninteger", "max", ["geninteger", "geninteger"])
    sig_list.append(f_max)

    f_max_2 = funsig("cl::sycl", "geninteger", "max", ["geninteger", "sgeninteger"])
    sig_list.append(f_max_2)

    f_min = funsig("cl::sycl", "geninteger", "min", ["geninteger", "geninteger"])
    sig_list.append(f_min)

    f_min_2 = funsig("cl::sycl", "geninteger", "min", ["geninteger", "sgeninteger"])
    sig_list.append(f_min_2)

    f_mul_hi = funsig("cl::sycl", "geninteger", "mul_hi", ["geninteger", "geninteger"])
    sig_list.append(f_mul_hi)

    f_rotate = funsig("cl::sycl", "geninteger", "rotate", ["geninteger", "geninteger"])
    sig_list.append(f_rotate)

    f_sub_sat = funsig("cl::sycl", "geninteger", "sub_sat", ["geninteger", "geninteger"])
    sig_list.append(f_sub_sat)

    f_upsample = funsig("cl::sycl", "ugeninteger16bit", "upsample", ["ugeninteger8bit", "ugeninteger8bit"])
    sig_list.append(f_upsample)

    f_upsample_2 = funsig("cl::sycl", "igeninteger16bit", "upsample", ["igeninteger8bit", "ugeninteger8bit"])
    sig_list.append(f_upsample_2)

    f_upsample_3 = funsig("cl::sycl", "ugeninteger32bit", "upsample", ["ugeninteger16bit", "ugeninteger16bit"])
    sig_list.append(f_upsample_3)

    f_upsample_4 = funsig("cl::sycl", "igeninteger32bit", "upsample", ["igeninteger16bit", "ugeninteger16bit"])
    sig_list.append(f_upsample_4)
    
    f_upsample_5 = funsig("cl::sycl", "ugeninteger64bit", "upsample", ["ugeninteger32bit", "ugeninteger32bit"])
    sig_list.append(f_upsample_5)

    f_upsample_6 = funsig("cl::sycl", "igeninteger64bit", "upsample", ["igeninteger32bit", "ugeninteger32bit"])
    sig_list.append(f_upsample_6)

    f_popcount = funsig("cl::sycl", "geninteger", "popcount", ["geninteger"])
    sig_list.append(f_popcount)
    
    f_mad24 = funsig("cl::sycl", "geninteger32bit", "mad24", ["geninteger32bit","geninteger32bit","geninteger32bit"])
    sig_list.append(f_mad24)
    
    f_mul24 = funsig("cl::sycl", "geninteger32bit", "mul24", ["geninteger32bit","geninteger32bit"])
    sig_list.append(f_mul24)
    
    return sig_list

def create_common_signatures():
    sig_list = []

    f_clamp = funsig("cl::sycl", "genfloat", "clamp", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_clamp)
    
    f_clamp_2 = funsig("cl::sycl", "genfloatf", "clamp", ["genfloatf", "float", "float"])
    sig_list.append(f_clamp_2)

    f_clamp_3 = funsig("cl::sycl", "genfloatd", "clamp", ["genfloatd", "double", "double"])
    sig_list.append(f_clamp_3)
    
    f_degrees = funsig("cl::sycl", "genfloat", "degrees", ["genfloat"])
    sig_list.append(f_degrees)

    f_max = funsig("cl::sycl", "genfloat", "max", ["genfloat", "genfloat"])
    sig_list.append(f_max)
    
    f_max_2 = funsig("cl::sycl", "genfloatf", "max", ["genfloatf", "float"])
    sig_list.append(f_max_2)

    f_max_3 = funsig("cl::sycl", "genfloatd", "max", ["genfloatd", "double"])
    sig_list.append(f_max_3)

    f_min = funsig("cl::sycl", "genfloat", "min", ["genfloat", "genfloat"])
    sig_list.append(f_min)
    
    f_min_2 = funsig("cl::sycl", "genfloatf", "min", ["genfloatf", "float"])
    sig_list.append(f_min_2)

    f_min_3 = funsig("cl::sycl", "genfloatd", "min", ["genfloatd", "double"])
    sig_list.append(f_min_3)
    
    f_mix = funsig("cl::sycl", "genfloat", "mix", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_mix)
    
    f_mix_2 = funsig("cl::sycl", "genfloatf", "mix", ["genfloatf", "genfloatf", "float"])
    sig_list.append(f_mix_2)

    f_mix_3 = funsig("cl::sycl", "genfloatd", "mix", ["genfloatd", "genfloatd", "double"])
    sig_list.append(f_mix_3)

    f_radians = funsig("cl::sycl", "genfloat", "radians", ["genfloat"])
    sig_list.append(f_radians)

    f_step = funsig("cl::sycl", "genfloat", "step", ["genfloat", "genfloat"])
    sig_list.append(f_step)
    
    f_step_2 = funsig("cl::sycl", "genfloatf", "step", ["float", "genfloatf"])
    sig_list.append(f_step_2)

    f_step_3 = funsig("cl::sycl", "genfloatd", "step", ["double", "genfloatd"])
    sig_list.append(f_step_3)

    f_smoothstep = funsig("cl::sycl", "genfloat", "smoothstep", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_step)
    
    f_smoothstep_2 = funsig("cl::sycl", "genfloatf", "smoothstep", ["float", "float", "genfloatf"])
    sig_list.append(f_smoothstep_2)

    f_smoothstep_3 = funsig("cl::sycl", "genfloatd", "smoothstep", ["double", "double", "genfloatd"])
    sig_list.append(f_smoothstep_3)

    f_sign = funsig("cl::sycl", "genfloat", "sign", ["genfloat"])
    sig_list.append(f_sign)

    return sig_list

def create_geometric_signatures():
    sig_list = []

    f_cross = funsig("cl::sycl", "cl::sycl::float3", "cross", ["cl::sycl::float3", "cl::sycl::float3"])
    sig_list.append(f_cross)

    f_cross_2 = funsig("cl::sycl", "cl::sycl::float4", "cross", ["cl::sycl::float4", "cl::sycl::float4"])
    sig_list.append(f_cross_2)

    f_cross_3 = funsig("cl::sycl", "cl::sycl::double3", "cross", ["cl::sycl::double3", "cl::sycl::double3"])
    sig_list.append(f_cross_3)

    f_cross_4 = funsig("cl::sycl", "cl::sycl::double4", "cross", ["cl::sycl::double4", "cl::sycl::double4"])
    sig_list.append(f_cross_4)

    f_dot = funsig("cl::sycl", "float", "dot", ["gengeofloat", "gengeofloat"])
    sig_list.append(f_dot)

    f_dot_2 = funsig("cl::sycl", "double", "dot", ["gengeodouble", "gengeodouble"])
    sig_list.append(f_dot_2)

    f_distance = funsig("cl::sycl", "float", "distance", ["gengeofloat", "gengeofloat"])
    sig_list.append(f_distance)

    f_distance_2 = funsig("cl::sycl", "double", "distance", ["gengeodouble", "gengeodouble"])
    sig_list.append(f_distance_2)

    f_length = funsig("cl::sycl", "float", "length", ["gengeofloat"])
    sig_list.append(f_length)

    f_length_2 = funsig("cl::sycl", "double", "length", ["gengeodouble"])
    sig_list.append(f_length_2)

    f_normalize = funsig("cl::sycl", "gengeofloat", "normalize", ["gengeofloat"])
    sig_list.append(f_length)

    f_normalize_2 = funsig("cl::sycl", "gengeodouble", "normalize", ["gengeodouble"])
    sig_list.append(f_normalize_2)

    f_fast_distance = funsig("cl::sycl", "float", "fast_distance", ["gengeofloat", "gengeofloat"])
    sig_list.append(f_fast_distance)

    f_fast_length = funsig("cl::sycl", "float", "fast_length", ["gengeofloat"])
    sig_list.append(f_fast_length)

    f_fast_normalize = funsig("cl::sycl", "gengeofloat", "fast_normalize", ["gengeofloat"])
    sig_list.append(f_fast_normalize)

    return sig_list

def create_relational_signatures():
    sig_list = []

    f_isequal = funsig("cl::sycl", "igeninteger32bit", "isequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_isequal)

    f_isequal_2 = funsig("cl::sycl", "igeninteger64bit", "isequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_isequal_2)

    f_isnotequal = funsig("cl::sycl", "igeninteger32bit", "isnotequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_isnotequal)

    f_isnotequal_2 = funsig("cl::sycl", "igeninteger64bit", "isnotequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_isnotequal_2)

    f_isgreater = funsig("cl::sycl", "igeninteger32bit", "isgreater", ["genfloatf", "genfloatf"])
    sig_list.append(f_isgreater)

    f_isgreater_2 = funsig("cl::sycl", "igeninteger64bit", "isgreater", ["genfloatd", "genfloatd"])
    sig_list.append(f_isgreater_2)

    f_isgreaterequal = funsig("cl::sycl", "igeninteger32bit", "isgreaterequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_isgreaterequal)

    f_isgreaterequal_2 = funsig("cl::sycl", "igeninteger64bit", "isgreaterequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_isgreaterequal_2)

    f_isless = funsig("cl::sycl", "igeninteger32bit", "isless", ["genfloatf", "genfloatf"])
    sig_list.append(f_isless)

    f_isless_2 = funsig("cl::sycl", "igeninteger64bit", "isless", ["genfloatd", "genfloatd"])
    sig_list.append(f_isless_2)

    f_islessequal = funsig("cl::sycl", "igeninteger32bit", "islessequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_islessequal)

    f_islessequal_2 = funsig("cl::sycl", "igeninteger64bit", "islessequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_islessequal_2)

    f_islessgreater = funsig("cl::sycl", "igeninteger32bit", "islessgreater", ["genfloatf", "genfloatf"])
    sig_list.append(f_islessgreater)

    f_islessgreater_2 = funsig("cl::sycl", "igeninteger64bit", "islessgreater", ["genfloatd", "genfloatd"])
    sig_list.append(f_islessgreater_2)

    f_isfinite = funsig("cl::sycl", "igeninteger32bit", "isfinite", ["genfloatf"])
    sig_list.append(f_isfinite)

    f_isfinite_2 = funsig("cl::sycl", "igeninteger64bit", "isfinite", ["genfloatd"])
    sig_list.append(f_isfinite_2)

    f_isinf = funsig("cl::sycl", "igeninteger32bit", "isinf", ["genfloatf"])
    sig_list.append(f_isinf)

    f_isinf_2 = funsig("cl::sycl", "igeninteger64bit", "isinf", ["genfloatd"])
    sig_list.append(f_isinf_2)

    f_isnan = funsig("cl::sycl", "igeninteger32bit", "isnan", ["genfloatf"])
    sig_list.append(f_isnan)

    f_isnan_2 = funsig("cl::sycl", "igeninteger64bit", "isnan", ["genfloatd"])
    sig_list.append(f_isnan_2)

    f_isnormal = funsig("cl::sycl", "igeninteger32bit", "isnormal", ["genfloatf"])
    sig_list.append(f_isnormal)

    f_isnormal_2 = funsig("cl::sycl", "igeninteger64bit", "isnormal", ["genfloatd"])
    sig_list.append(f_isnormal_2)

    f_isordered = funsig("cl::sycl", "igeninteger32bit", "isordered", ["genfloatf", "genfloatf"])
    sig_list.append(f_isordered)

    f_isordered_2 = funsig("cl::sycl", "igeninteger64bit", "isordered", ["genfloatd", "genfloatd"])
    sig_list.append(f_isordered_2)

    f_isunordered = funsig("cl::sycl", "igeninteger32bit", "isunordered", ["genfloatf", "genfloatf"])
    sig_list.append(f_isunordered)

    f_isunordered_2 = funsig("cl::sycl", "igeninteger64bit", "isunordered", ["genfloatd", "genfloatd"])
    sig_list.append(f_isunordered_2)

    f_signbit = funsig("cl::sycl", "igeninteger32bit", "signbit", ["genfloatf"])
    sig_list.append(f_signbit)

    f_signbit_2 = funsig("cl::sycl", "igeninteger64bit", "signbit", ["genfloatd"])
    sig_list.append(f_signbit_2)

    f_any = funsig("cl::sycl", "int", "any", ["igeninteger"])
    sig_list.append(f_any)

    f_all = funsig("cl::sycl", "int", "all", ["igeninteger"])
    sig_list.append(f_all)

    f_bitselect = funsig("cl::sycl", "gentype", "bitselect", ["gentype", "gentype","gentype"])
    sig_list.append(f_bitselect)

    f_select = funsig("cl::sycl", "geninteger", "select", ["geninteger", "geninteger","igeninteger"])
    sig_list.append(f_select)

    f_select_2 = funsig("cl::sycl", "geninteger", "select", ["geninteger", "geninteger","ugeninteger"])
    sig_list.append(f_select_2)

    f_select_3 = funsig("cl::sycl", "genfloatf", "select", ["genfloatf", "genfloatf","genint"])
    sig_list.append(f_select_3)

    f_select_4 = funsig("cl::sycl", "genfloatf", "select", ["genfloatf", "genfloatf","ugenint"])
    sig_list.append(f_select_4)

    f_select_5 = funsig("cl::sycl", "genfloatd", "select", ["genfloatd", "genfloatd","igeninteger64bit"])
    sig_list.append(f_select_5)

    f_select_6 = funsig("cl::sycl", "genfloatd", "select", ["genfloatd", "genfloatd","ugeninteger64bit"])
    sig_list.append(f_select_6)

    return sig_list

def create_float_signatures():
    sig_list = []

    f_acos = funsig("cl::sycl", "genfloat", "acos", ["genfloat"])
    sig_list.append(f_acos)

    f_acosh = funsig("cl::sycl", "genfloat", "acosh", ["genfloat"])
    sig_list.append(f_acosh)

    f_acospi = funsig("cl::sycl", "genfloat", "acospi", ["genfloat"])
    sig_list.append(f_acospi)

    f_asin = funsig("cl::sycl", "genfloat", "asin", ["genfloat"])
    sig_list.append(f_asin)

    f_asinh = funsig("cl::sycl", "genfloat", "asinh", ["genfloat"])
    sig_list.append(f_asinh)

    f_asinpi = funsig("cl::sycl", "genfloat", "asinpi", ["genfloat"])
    sig_list.append(f_asinpi)

    f_atan = funsig("cl::sycl", "genfloat", "atan", ["genfloat"])
    sig_list.append(f_atan)

    f_atan2 = funsig("cl::sycl", "genfloat", "atan2", ["genfloat", "genfloat"])
    sig_list.append(f_atan2)

    f_atanh = funsig("cl::sycl", "genfloat", "atanh", ["genfloat"])
    sig_list.append(f_atanh)

    f_atanpi = funsig("cl::sycl", "genfloat", "atanpi", ["genfloat"])
    sig_list.append(f_atanpi)

    f_atan2pi = funsig("cl::sycl", "genfloat", "atan2pi", ["genfloat", "genfloat"])
    sig_list.append(f_atan2pi)

    f_cbrt = funsig("cl::sycl", "genfloat", "cbrt", ["genfloat"])
    sig_list.append(f_cbrt)

    f_ceil = funsig("cl::sycl", "genfloat", "ceil", ["genfloat"])
    sig_list.append(f_ceil)

    f_copysign = funsig("cl::sycl", "genfloat", "copysign", ["genfloat", "genfloat"])
    sig_list.append(f_copysign)

    f_cos = funsig("cl::sycl", "genfloat", "cos", ["genfloat"])
    sig_list.append(f_cos)

    f_cosh = funsig("cl::sycl", "genfloat", "cosh", ["genfloat"])
    sig_list.append(f_cosh)

    f_cospi = funsig("cl::sycl", "genfloat", "cospi", ["genfloat"])
    sig_list.append(f_cospi)

    f_erfc = funsig("cl::sycl", "genfloat", "erfc", ["genfloat"])
    sig_list.append(f_erfc)

    f_erf = funsig("cl::sycl", "genfloat", "erf", ["genfloat"])
    sig_list.append(f_erf)

    f_exp = funsig("cl::sycl", "genfloat", "exp", ["genfloat"])
    sig_list.append(f_exp)

    f_exp2 = funsig("cl::sycl", "genfloat", "exp2", ["genfloat"])
    sig_list.append(f_exp2)

    f_exp10 = funsig("cl::sycl", "genfloat", "exp10", ["genfloat"])
    sig_list.append(f_exp10)

    f_expm1 = funsig("cl::sycl", "genfloat", "expm1", ["genfloat"])
    sig_list.append(f_expm1)

    f_fabs = funsig("cl::sycl", "genfloat", "fabs", ["genfloat"])
    sig_list.append(f_fabs)

    f_fdim = funsig("cl::sycl", "genfloat", "fdim", ["genfloat", "genfloat"])
    sig_list.append(f_fdim)

    f_floor = funsig("cl::sycl", "genfloat", "floor", ["genfloat"])
    sig_list.append(f_floor)

    f_fma = funsig("cl::sycl", "genfloat", "fma", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_fma)

    f_fmax = funsig("cl::sycl", "genfloat", "fmax", ["genfloat", "genfloat"])
    sig_list.append(f_fmax)

    f_fmax_2 = funsig("cl::sycl", "genfloat", "fmax", ["genfloat", "sgenfloat"])
    sig_list.append(f_fmax_2)

    f_fmin = funsig("cl::sycl", "genfloat", "fmin", ["genfloat", "genfloat"])
    sig_list.append(f_fmin)

    f_fmin_2 = funsig("cl::sycl", "genfloat", "fmin", ["genfloat", "sgenfloat"])
    sig_list.append(f_fmin_2)

    f_fmod = funsig("cl::sycl", "genfloat", "fmod", ["genfloat", "genfloat"])
    sig_list.append(f_fmod)

    f_fract = funsig("cl::sycl", "genfloat", "fract", ["genfloat", "genfloat"], [2])
    sig_list.append(f_fract)

    f_frexp = funsig("cl::sycl", "genfloat", "frexp", ["genfloat", "genint"], [2])
    sig_list.append(f_frexp)

    f_hypot = funsig("cl::sycl", "genfloat", "hypot", ["genfloat", "genfloat"])
    sig_list.append(f_hypot)

    f_ilogb = funsig("cl::sycl", "genint", "ilogb", ["genfloat"])
    sig_list.append(f_ilogb)

    f_ldexp = funsig("cl::sycl", "genfloat", "ldexp", ["genfloat", "genint"])
    sig_list.append(f_ldexp)

    f_ldexp_2 = funsig("cl::sycl", "genfloat", "ldexp", ["genfloat", "int"])
    sig_list.append(f_ldexp_2)

    f_lgamma = funsig("cl::sycl", "genfloat", "lgamma", ["genfloat"])
    sig_list.append(f_lgamma)

    f_lgamma_r = funsig("cl::sycl", "genfloat", "lgamma_r", ["genfloat", "genint"],[2])
    sig_list.append(f_lgamma_r)

    f_log = funsig("cl::sycl", "genfloat", "log", ["genfloat"])
    sig_list.append(f_log)

    f_log2 = funsig("cl::sycl", "genfloat", "log2", ["genfloat"])
    sig_list.append(f_log2)

    f_log10 = funsig("cl::sycl", "genfloat", "log10", ["genfloat"])
    sig_list.append(f_log10)

    f_log1p = funsig("cl::sycl", "genfloat", "log1p", ["genfloat"])
    sig_list.append(f_log1p)

    f_logb = funsig("cl::sycl", "genfloat", "logb", ["genfloat"])
    sig_list.append(f_logb)
    
    f_mad = funsig("cl::sycl", "genfloat", "mad", ["genfloat","genfloat","genfloat"])
    sig_list.append(f_mad)
    
    f_maxmag = funsig("cl::sycl", "genfloat", "maxmag", ["genfloat","genfloat"])
    sig_list.append(f_maxmag)

    f_minmag = funsig("cl::sycl", "genfloat", "minmag", ["genfloat","genfloat"])
    sig_list.append(f_minmag)

    f_modf = funsig("cl::sycl", "genfloat", "modf", ["genfloat", "genfloat"],[2])
    sig_list.append(f_modf)

    f_nan = funsig("cl::sycl", "genfloatf", "nan", ["ugenint"])
    sig_list.append(f_nan)

    f_nan_2 = funsig("cl::sycl", "genfloatd", "nan", ["ugenlonginteger"])
    sig_list.append(f_nan_2)

    f_nextafter = funsig("cl::sycl", "genfloat", "nextafter", ["genfloat", "genfloat"])
    sig_list.append(f_nextafter)

    f_pow = funsig("cl::sycl", "genfloat", "pow", ["genfloat", "genfloat"])
    sig_list.append(f_pow)

    f_pown = funsig("cl::sycl", "genfloat", "pown", ["genfloat", "genint"])
    sig_list.append(f_pown)

    f_powr = funsig("cl::sycl", "genfloat", "powr", ["genfloat", "genfloat"])
    sig_list.append(f_powr)

    f_remainder = funsig("cl::sycl", "genfloat", "remainder", ["genfloat", "genfloat"])
    sig_list.append(f_remainder)

    f_remquo = funsig("cl::sycl", "genfloat", "remquo", ["genfloat", "genfloat", "genint"],[3])
    sig_list.append(f_remquo)

    f_rint = funsig("cl::sycl", "genfloat", "rint", ["genfloat"])
    sig_list.append(f_rint)

    f_rootn = funsig("cl::sycl", "genfloat", "rootn", ["genfloat", "genint"])
    sig_list.append(f_rootn)

    f_round = funsig("cl::sycl", "genfloat", "round", ["genfloat"])
    sig_list.append(f_round)

    f_rsqrt = funsig("cl::sycl", "genfloat", "rsqrt", ["genfloat"])
    sig_list.append(f_rsqrt)

    f_sin = funsig("cl::sycl", "genfloat", "sin", ["genfloat"])
    sig_list.append(f_sin)

    f_sincos = funsig("cl::sycl", "genfloat", "sincos", ["genfloat", "genfloat"],[2])
    sig_list.append(f_sincos)

    f_sinh = funsig("cl::sycl", "genfloat", "sinh", ["genfloat"])
    sig_list.append(f_sinh)

    f_sinpi = funsig("cl::sycl", "genfloat", "sinpi", ["genfloat"])
    sig_list.append(f_sinpi)

    f_sqrt = funsig("cl::sycl", "genfloat", "sqrt", ["genfloat"])
    sig_list.append(f_sqrt)

    f_tan = funsig("cl::sycl", "genfloat", "tan", ["genfloat"])
    sig_list.append(f_tan)

    f_tanh = funsig("cl::sycl", "genfloat", "tanh", ["genfloat"])
    sig_list.append(f_tanh)

    f_tanpi = funsig("cl::sycl", "genfloat", "tanpi", ["genfloat"])
    sig_list.append(f_tanpi)

    f_tgamma = funsig("cl::sycl", "genfloat", "tgamma", ["genfloat"])
    sig_list.append(f_tgamma)

    f_trunc = funsig("cl::sycl", "genfloat", "trunc", ["genfloat"])
    sig_list.append(f_trunc)
    
    return sig_list

def create_native_signatures():
    sig_list = []

    f_cos = funsig("cl::sycl::native", "genfloatf", "cos", ["genfloatf"])
    sig_list.append(f_cos)

    f_divide = funsig("cl::sycl::native", "genfloatf", "divide", ["genfloatf", "genfloatf"])
    sig_list.append(f_divide)

    f_exp = funsig("cl::sycl::native", "genfloatf", "exp", ["genfloatf"])
    sig_list.append(f_exp)

    f_exp2 = funsig("cl::sycl::native", "genfloatf", "exp2", ["genfloatf"])
    sig_list.append(f_exp2)

    f_exp10 = funsig("cl::sycl::native", "genfloatf", "exp10", ["genfloatf"])
    sig_list.append(f_exp10)

    f_log = funsig("cl::sycl::native", "genfloatf", "log", ["genfloatf"])
    sig_list.append(f_log)

    f_log2 = funsig("cl::sycl::native", "genfloatf", "log2", ["genfloatf"])
    sig_list.append(f_log2)

    f_log10 = funsig("cl::sycl::native", "genfloatf", "log10", ["genfloatf"])
    sig_list.append(f_log10)

    f_powr = funsig("cl::sycl::native", "genfloatf", "powr", ["genfloatf", "genfloatf"])
    sig_list.append(f_powr)

    f_recip = funsig("cl::sycl::native", "genfloatf", "recip", ["genfloatf"])
    sig_list.append(f_recip)

    f_rsqrt = funsig("cl::sycl::native", "genfloatf", "rsqrt", ["genfloatf"])
    sig_list.append(f_rsqrt)

    f_sin = funsig("cl::sycl::native", "genfloatf", "sin", ["genfloatf"])
    sig_list.append(f_sin)

    f_sqrt = funsig("cl::sycl::native", "genfloatf", "sqrt", ["genfloatf"])
    sig_list.append(f_sqrt)

    f_tan = funsig("cl::sycl::native", "genfloatf", "tan", ["genfloatf"])
    sig_list.append(f_tan)

    return sig_list

def create_half_signatures():
    sig_list = []

    f_cos = funsig("cl::sycl::half_precision", "genfloatf", "cos", ["genfloatf"])
    sig_list.append(f_cos)
    
    f_divide = funsig("cl::sycl::half_precision", "genfloatf", "divide", ["genfloatf", "genfloatf"])
    sig_list.append(f_divide)

    f_exp = funsig("cl::sycl::half_precision", "genfloatf", "exp", ["genfloatf"])
    sig_list.append(f_exp)

    f_exp2 = funsig("cl::sycl::half_precision", "genfloatf", "exp2", ["genfloatf"])
    sig_list.append(f_exp2)

    f_exp10 = funsig("cl::sycl::half_precision", "genfloatf", "exp10", ["genfloatf"])
    sig_list.append(f_exp10)

    f_log = funsig("cl::sycl::half_precision", "genfloatf", "log", ["genfloatf"])
    sig_list.append(f_log)

    f_log2 = funsig("cl::sycl::half_precision", "genfloatf", "log2", ["genfloatf"])
    sig_list.append(f_log2)

    f_log10 = funsig("cl::sycl::half_precision", "genfloatf", "log10", ["genfloatf"])
    sig_list.append(f_log10)

    f_powr = funsig("cl::sycl::half_precision", "genfloatf", "powr", ["genfloatf", "genfloatf"])
    sig_list.append(f_powr)

    f_recip = funsig("cl::sycl::half_precision", "genfloatf", "recip", ["genfloatf"])
    sig_list.append(f_recip)

    f_rsqrt = funsig("cl::sycl::half_precision", "genfloatf", "rsqrt", ["genfloatf"])
    sig_list.append(f_rsqrt)

    f_sin = funsig("cl::sycl::half_precision", "genfloatf", "sin", ["genfloatf"])
    sig_list.append(f_sin)

    f_sqrt = funsig("cl::sycl::half_precision", "genfloatf", "sqrt", ["genfloatf"])
    sig_list.append(f_sqrt)

    f_tan = funsig("cl::sycl::half_precision", "genfloatf", "tan", ["genfloatf"])
    sig_list.append(f_tan)

    return sig_list
