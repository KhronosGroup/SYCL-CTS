"""Represents a function signature."""
class funsig:
    def __init__(self, namespace, ret_type, name, arg_types=[], accuracy="", comment="", pntr_indx=[]):
        self.namespace = namespace # Namespace of function.
        self.ret_type = ret_type # Function return type.
        self.name = name # Function name.
        self.arg_types = arg_types # List containing the function argument types.
        self.accuracy = accuracy # The function maximum relative error defined as ulp.
        self.comment = comment # The comment for function maximum relative error.
        self.pntr_indx = pntr_indx # List containing the indexes of the arguments which are pointers.
    def __eq__(self, other):
        if isinstance(other, funsig):
            return ((self.namespace == other.namespace) and
                    (self.ret_type == other.ret_type) and
                    (self.name == other.name) and
                    (self.arg_types == other.arg_types) and
                    (self.accuracy == other.accuracy) and
                    (self.comment == other.comment) and
                    (self.pntr_indx == other.pntr_indx))
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash((self.namespace, self.ret_type, self.name, str(self.arg_types), self.accuracy, self.comment, str(self.pntr_indx)))

def create_integer_signatures():
    sig_list = []

    f_abs = funsig("sycl", "ugeninteger", "abs", ["geninteger"])
    sig_list.append(f_abs)

    f_abs_diff = funsig("sycl", "ugeninteger", "abs_diff", ["geninteger", "geninteger"])
    sig_list.append(f_abs_diff)

    f_add_sat = funsig("sycl", "geninteger", "add_sat", ["geninteger", "geninteger"])
    sig_list.append(f_add_sat)

    f_hadd = funsig("sycl", "geninteger", "hadd", ["geninteger", "geninteger"])
    sig_list.append(f_hadd)

    f_rhadd = funsig("sycl", "geninteger", "rhadd", ["geninteger", "geninteger"])
    sig_list.append(f_rhadd)

    f_clamp = funsig("sycl", "geninteger", "clamp", ["geninteger", "sgeninteger", "sgeninteger"])
    sig_list.append(f_clamp)

    f_clamp_2 = funsig("sycl", "geninteger", "clamp", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_clamp_2)

    f_clz = funsig("sycl", "geninteger", "clz", ["geninteger"])
    sig_list.append(f_clz)

    f_mad_hi = funsig("sycl", "geninteger", "mad_hi", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_mad_hi)

    f_mad_sat = funsig("sycl", "geninteger", "mad_sat", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_mad_sat)

    f_max = funsig("sycl", "geninteger", "max", ["geninteger", "geninteger"])
    sig_list.append(f_max)

    f_max_2 = funsig("sycl", "geninteger", "max", ["geninteger", "sgeninteger"])
    sig_list.append(f_max_2)

    f_min = funsig("sycl", "geninteger", "min", ["geninteger", "geninteger"])
    sig_list.append(f_min)

    f_min_2 = funsig("sycl", "geninteger", "min", ["geninteger", "sgeninteger"])
    sig_list.append(f_min_2)

    f_mul_hi = funsig("sycl", "geninteger", "mul_hi", ["geninteger", "geninteger"])
    sig_list.append(f_mul_hi)

    f_rotate = funsig("sycl", "geninteger", "rotate", ["geninteger", "geninteger"])
    sig_list.append(f_rotate)

    f_sub_sat = funsig("sycl", "geninteger", "sub_sat", ["geninteger", "geninteger"])
    sig_list.append(f_sub_sat)

    f_upsample = funsig("sycl", "ugeninteger16bit", "upsample", ["ugeninteger8bit", "ugeninteger8bit"])
    sig_list.append(f_upsample)

    f_upsample_2 = funsig("sycl", "igeninteger16bit", "upsample", ["igeninteger8bit", "ugeninteger8bit"])
    sig_list.append(f_upsample_2)

    f_upsample_3 = funsig("sycl", "ugeninteger32bit", "upsample", ["ugeninteger16bit", "ugeninteger16bit"])
    sig_list.append(f_upsample_3)

    f_upsample_4 = funsig("sycl", "igeninteger32bit", "upsample", ["igeninteger16bit", "ugeninteger16bit"])
    sig_list.append(f_upsample_4)

    f_upsample_5 = funsig("sycl", "ugeninteger64bit", "upsample", ["ugeninteger32bit", "ugeninteger32bit"])
    sig_list.append(f_upsample_5)

    f_upsample_6 = funsig("sycl", "igeninteger64bit", "upsample", ["igeninteger32bit", "ugeninteger32bit"])
    sig_list.append(f_upsample_6)

    f_popcount = funsig("sycl", "geninteger", "popcount", ["geninteger"])
    sig_list.append(f_popcount)

    f_mad24 = funsig("sycl", "geninteger32bit", "mad24", ["geninteger32bit","geninteger32bit","geninteger32bit"])
    sig_list.append(f_mad24)

    f_mul24 = funsig("sycl", "geninteger32bit", "mul24", ["geninteger32bit","geninteger32bit"])
    sig_list.append(f_mul24)

    return sig_list

def create_common_signatures():
    sig_list = []

    f_clamp = funsig("sycl", "genfloat", "clamp", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_clamp)

    f_clamp_2 = funsig("sycl", "genfloatf", "clamp", ["genfloatf", "float", "float"])
    sig_list.append(f_clamp_2)

    f_clamp_3 = funsig("sycl", "genfloatd", "clamp", ["genfloatd", "double", "double"])
    sig_list.append(f_clamp_3)

    f_degrees = funsig("sycl", "genfloat", "degrees", ["genfloat"], "3")
    sig_list.append(f_degrees)

    f_max = funsig("sycl", "genfloat", "max", ["genfloat", "genfloat"])
    sig_list.append(f_max)

    f_max_2 = funsig("sycl", "genfloatf", "max", ["genfloatf", "float"])
    sig_list.append(f_max_2)

    f_max_3 = funsig("sycl", "genfloatd", "max", ["genfloatd", "double"])
    sig_list.append(f_max_3)

    f_min = funsig("sycl", "genfloat", "min", ["genfloat", "genfloat"])
    sig_list.append(f_min)

    f_min_2 = funsig("sycl", "genfloatf", "min", ["genfloatf", "float"])
    sig_list.append(f_min_2)

    f_min_3 = funsig("sycl", "genfloatd", "min", ["genfloatd", "double"])
    sig_list.append(f_min_3)

    f_mix = funsig("sycl", "genfloat", "mix", ["genfloat", "genfloat", "genfloat"], "1")
    sig_list.append(f_mix)

    f_mix_2 = funsig("sycl", "genfloatf", "mix", ["genfloatf", "genfloatf", "float"], "1")
    sig_list.append(f_mix_2)

    f_mix_3 = funsig("sycl", "genfloatd", "mix", ["genfloatd", "genfloatd", "double"], "1")
    sig_list.append(f_mix_3)

    f_radians = funsig("sycl", "genfloat", "radians", ["genfloat"], "3")
    sig_list.append(f_radians)

    f_step = funsig("sycl", "genfloat", "step", ["genfloat", "genfloat"])
    sig_list.append(f_step)

    f_step_2 = funsig("sycl", "genfloatf", "step", ["float", "genfloatf"])
    sig_list.append(f_step_2)

    f_step_3 = funsig("sycl", "genfloatd", "step", ["double", "genfloatd"])
    sig_list.append(f_step_3)

    f_smoothstep = funsig("sycl", "genfloat", "smoothstep", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_step)

    f_smoothstep_2 = funsig("sycl", "genfloatf", "smoothstep", ["float", "float", "genfloatf"])
    sig_list.append(f_smoothstep_2)

    f_smoothstep_3 = funsig("sycl", "genfloatd", "smoothstep", ["double", "double", "genfloatd"])
    sig_list.append(f_smoothstep_3)

    f_sign = funsig("sycl", "genfloat", "sign", ["genfloat"])
    sig_list.append(f_sign)

    return sig_list

def create_geometric_signatures():
    sig_list = []

    f_cross = funsig("sycl", "sycl::float3", "cross", ["sycl::float3", "sycl::float3"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross)

    f_cross_2 = funsig("sycl", "sycl::float4", "cross", ["sycl::float4", "sycl::float4"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_2)

    f_cross_3 = funsig("sycl", "sycl::double3", "cross", ["sycl::double3", "sycl::double3"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_3)

    f_cross_4 = funsig("sycl", "sycl::double4", "cross", ["sycl::double4", "sycl::double4"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_4)

    f_dot = funsig("sycl", "float", "dot", ["gengeofloat", "gengeofloat"], "2*vecSize - 1",
            "cumulative error for multiplications and additions 'vecSize + vecSize-1'")
    sig_list.append(f_dot)

    f_dot_2 = funsig("sycl", "double", "dot", ["gengeodouble", "gengeodouble"], "2*vecSize - 1",
            "cumulative error for multiplications and additions 'vecSize + vecSize-1'")
    sig_list.append(f_dot_2)

    f_distance = funsig("sycl", "float", "distance", ["gengeofloat", "gengeofloat"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'")
    sig_list.append(f_distance)

    f_distance_2 = funsig("sycl", "double", "distance", ["gengeodouble", "gengeodouble"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'")
    sig_list.append(f_distance_2)

    f_length = funsig("sycl", "float", "length", ["gengeofloat"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'")
    sig_list.append(f_length)

    f_length_2 = funsig("sycl", "double", "length", ["gengeodouble"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'")
    sig_list.append(f_length_2)

    f_normalize = funsig("sycl", "gengeofloat", "normalize", ["gengeofloat"], "2*vecSize + 1",
            "cumulative error for multiplications, additions and rsqrt 'vecSize + vecSize-1 + 2'")
    sig_list.append(f_length)

    f_normalize_2 = funsig("sycl", "gengeodouble", "normalize", ["gengeodouble"], "2*vecSize + 1",
            "cumulative error for multiplications, additions and rsqrt 'vecSize + vecSize-1 + 2'")
    sig_list.append(f_normalize_2)

    f_fast_distance = funsig("sycl", "float", "fast_distance", ["gengeofloat", "gengeofloat"], "8192")
    sig_list.append(f_fast_distance)

    f_fast_length = funsig("sycl", "float", "fast_length", ["gengeofloat"], "8192")
    sig_list.append(f_fast_length)

    f_fast_normalize = funsig("sycl", "gengeofloat", "fast_normalize", ["gengeofloat"], "8192")
    sig_list.append(f_fast_normalize)

    return sig_list

def create_relational_signatures():
    sig_list = []

    f_isequal = funsig("sycl", "igeninteger32bit", "isequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_isequal)

    f_isequal_2 = funsig("sycl", "igeninteger64bit", "isequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_isequal_2)

    f_isnotequal = funsig("sycl", "igeninteger32bit", "isnotequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_isnotequal)

    f_isnotequal_2 = funsig("sycl", "igeninteger64bit", "isnotequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_isnotequal_2)

    f_isgreater = funsig("sycl", "igeninteger32bit", "isgreater", ["genfloatf", "genfloatf"])
    sig_list.append(f_isgreater)

    f_isgreater_2 = funsig("sycl", "igeninteger64bit", "isgreater", ["genfloatd", "genfloatd"])
    sig_list.append(f_isgreater_2)

    f_isgreaterequal = funsig("sycl", "igeninteger32bit", "isgreaterequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_isgreaterequal)

    f_isgreaterequal_2 = funsig("sycl", "igeninteger64bit", "isgreaterequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_isgreaterequal_2)

    f_isless = funsig("sycl", "igeninteger32bit", "isless", ["genfloatf", "genfloatf"])
    sig_list.append(f_isless)

    f_isless_2 = funsig("sycl", "igeninteger64bit", "isless", ["genfloatd", "genfloatd"])
    sig_list.append(f_isless_2)

    f_islessequal = funsig("sycl", "igeninteger32bit", "islessequal", ["genfloatf", "genfloatf"])
    sig_list.append(f_islessequal)

    f_islessequal_2 = funsig("sycl", "igeninteger64bit", "islessequal", ["genfloatd", "genfloatd"])
    sig_list.append(f_islessequal_2)

    f_islessgreater = funsig("sycl", "igeninteger32bit", "islessgreater", ["genfloatf", "genfloatf"])
    sig_list.append(f_islessgreater)

    f_islessgreater_2 = funsig("sycl", "igeninteger64bit", "islessgreater", ["genfloatd", "genfloatd"])
    sig_list.append(f_islessgreater_2)

    f_isfinite = funsig("sycl", "igeninteger32bit", "isfinite", ["genfloatf"])
    sig_list.append(f_isfinite)

    f_isfinite_2 = funsig("sycl", "igeninteger64bit", "isfinite", ["genfloatd"])
    sig_list.append(f_isfinite_2)

    f_isinf = funsig("sycl", "igeninteger32bit", "isinf", ["genfloatf"])
    sig_list.append(f_isinf)

    f_isinf_2 = funsig("sycl", "igeninteger64bit", "isinf", ["genfloatd"])
    sig_list.append(f_isinf_2)

    f_isnan = funsig("sycl", "igeninteger32bit", "isnan", ["genfloatf"])
    sig_list.append(f_isnan)

    f_isnan_2 = funsig("sycl", "igeninteger64bit", "isnan", ["genfloatd"])
    sig_list.append(f_isnan_2)

    f_isnormal = funsig("sycl", "igeninteger32bit", "isnormal", ["genfloatf"])
    sig_list.append(f_isnormal)

    f_isnormal_2 = funsig("sycl", "igeninteger64bit", "isnormal", ["genfloatd"])
    sig_list.append(f_isnormal_2)

    f_isordered = funsig("sycl", "igeninteger32bit", "isordered", ["genfloatf", "genfloatf"])
    sig_list.append(f_isordered)

    f_isordered_2 = funsig("sycl", "igeninteger64bit", "isordered", ["genfloatd", "genfloatd"])
    sig_list.append(f_isordered_2)

    f_isunordered = funsig("sycl", "igeninteger32bit", "isunordered", ["genfloatf", "genfloatf"])
    sig_list.append(f_isunordered)

    f_isunordered_2 = funsig("sycl", "igeninteger64bit", "isunordered", ["genfloatd", "genfloatd"])
    sig_list.append(f_isunordered_2)

    f_signbit = funsig("sycl", "igeninteger32bit", "signbit", ["genfloatf"])
    sig_list.append(f_signbit)

    f_signbit_2 = funsig("sycl", "igeninteger64bit", "signbit", ["genfloatd"])
    sig_list.append(f_signbit_2)

    f_any = funsig("sycl", "int", "any", ["igeninteger"])
    sig_list.append(f_any)

    f_all = funsig("sycl", "int", "all", ["igeninteger"])
    sig_list.append(f_all)

    f_bitselect = funsig("sycl", "gentype", "bitselect", ["gentype", "gentype","gentype"])
    sig_list.append(f_bitselect)

    f_select = funsig("sycl", "geninteger", "select", ["geninteger", "geninteger","igeninteger"])
    sig_list.append(f_select)

    f_select_2 = funsig("sycl", "geninteger", "select", ["geninteger", "geninteger","ugeninteger"])
    sig_list.append(f_select_2)

    f_select_3 = funsig("sycl", "genfloatf", "select", ["genfloatf", "genfloatf","genint"])
    sig_list.append(f_select_3)

    f_select_4 = funsig("sycl", "genfloatf", "select", ["genfloatf", "genfloatf","ugenint"])
    sig_list.append(f_select_4)

    f_select_5 = funsig("sycl", "genfloatd", "select", ["genfloatd", "genfloatd","igeninteger64bit"])
    sig_list.append(f_select_5)

    f_select_6 = funsig("sycl", "genfloatd", "select", ["genfloatd", "genfloatd","ugeninteger64bit"])
    sig_list.append(f_select_6)

    return sig_list

def create_float_signatures():
    sig_list = []

    f_acos = funsig("sycl", "genfloat", "acos", ["genfloat"], "4")
    sig_list.append(f_acos)

    f_acosh = funsig("sycl", "genfloat", "acosh", ["genfloat"], "4")
    sig_list.append(f_acosh)

    f_acospi = funsig("sycl", "genfloat", "acospi", ["genfloat"], "5")
    sig_list.append(f_acospi)

    f_asin = funsig("sycl", "genfloat", "asin", ["genfloat"], "4")
    sig_list.append(f_asin)

    f_asinh = funsig("sycl", "genfloat", "asinh", ["genfloat"], "4")
    sig_list.append(f_asinh)

    f_asinpi = funsig("sycl", "genfloat", "asinpi", ["genfloat"], "5")
    sig_list.append(f_asinpi)

    f_atan = funsig("sycl", "genfloat", "atan", ["genfloat"], "5")
    sig_list.append(f_atan)

    f_atan2 = funsig("sycl", "genfloat", "atan2", ["genfloat", "genfloat"], "6")
    sig_list.append(f_atan2)

    f_atanh = funsig("sycl", "genfloat", "atanh", ["genfloat"], "5")
    sig_list.append(f_atanh)

    f_atanpi = funsig("sycl", "genfloat", "atanpi", ["genfloat"], "5")
    sig_list.append(f_atanpi)

    f_atan2pi = funsig("sycl", "genfloat", "atan2pi", ["genfloat", "genfloat"], "6")
    sig_list.append(f_atan2pi)

    f_cbrt = funsig("sycl", "genfloat", "cbrt", ["genfloat"], "2")
    sig_list.append(f_cbrt)

    f_ceil = funsig("sycl", "genfloat", "ceil", ["genfloat"], "0")
    sig_list.append(f_ceil)

    f_copysign = funsig("sycl", "genfloat", "copysign", ["genfloat", "genfloat"], "0")
    sig_list.append(f_copysign)

    f_cos = funsig("sycl", "genfloat", "cos", ["genfloat"], "4")
    sig_list.append(f_cos)

    f_cosh = funsig("sycl", "genfloat", "cosh", ["genfloat"], "4")
    sig_list.append(f_cosh)

    f_cospi = funsig("sycl", "genfloat", "cospi", ["genfloat"], "4")
    sig_list.append(f_cospi)

    f_erfc = funsig("sycl", "genfloat", "erfc", ["genfloat"], "16")
    sig_list.append(f_erfc)

    f_erf = funsig("sycl", "genfloat", "erf", ["genfloat"], "16")
    sig_list.append(f_erf)

    f_exp = funsig("sycl", "genfloat", "exp", ["genfloat"], "3")
    sig_list.append(f_exp)

    f_exp2 = funsig("sycl", "genfloat", "exp2", ["genfloat"], "3")
    sig_list.append(f_exp2)

    f_exp10 = funsig("sycl", "genfloat", "exp10", ["genfloat"], "3")
    sig_list.append(f_exp10)

    f_expm1 = funsig("sycl", "genfloat", "expm1", ["genfloat"], "3")
    sig_list.append(f_expm1)

    f_fabs = funsig("sycl", "genfloat", "fabs", ["genfloat"], "0")
    sig_list.append(f_fabs)

    f_fdim = funsig("sycl", "genfloat", "fdim", ["genfloat", "genfloat"], "0")
    sig_list.append(f_fdim)

    f_floor = funsig("sycl", "genfloat", "floor", ["genfloat"], "0")
    sig_list.append(f_floor)

    f_fma = funsig("sycl", "genfloat", "fma", ["genfloat", "genfloat", "genfloat"], "0")
    sig_list.append(f_fma)

    f_fmax = funsig("sycl", "genfloat", "fmax", ["genfloat", "genfloat"], "0")
    sig_list.append(f_fmax)

    f_fmax_2 = funsig("sycl", "genfloat", "fmax", ["genfloat", "sgenfloat"], "0")
    sig_list.append(f_fmax_2)

    f_fmin = funsig("sycl", "genfloat", "fmin", ["genfloat", "genfloat"], "0")
    sig_list.append(f_fmin)

    f_fmin_2 = funsig("sycl", "genfloat", "fmin", ["genfloat", "sgenfloat"], "0")
    sig_list.append(f_fmin_2)

    f_fmod = funsig("sycl", "genfloat", "fmod", ["genfloat", "genfloat"], "0")
    sig_list.append(f_fmod)

    f_fract = funsig("sycl", "genfloat", "fract", ["genfloat", "genfloat"], "0", "", [2])
    sig_list.append(f_fract)

    f_frexp = funsig("sycl", "genfloat", "frexp", ["genfloat", "genint"], "0", "", [2])
    sig_list.append(f_frexp)

    f_hypot = funsig("sycl", "genfloat", "hypot", ["genfloat", "genfloat"], "4")
    sig_list.append(f_hypot)

    f_ilogb = funsig("sycl", "genint", "ilogb", ["genfloat"], "0")
    sig_list.append(f_ilogb)

    f_ldexp = funsig("sycl", "genfloat", "ldexp", ["genfloat", "genint"], "0")
    sig_list.append(f_ldexp)

    f_ldexp_2 = funsig("sycl", "genfloat", "ldexp", ["genfloat", "int"], "0")
    sig_list.append(f_ldexp_2)

    f_lgamma = funsig("sycl", "genfloat", "lgamma", ["genfloat"], "-1")
    sig_list.append(f_lgamma)

    f_lgamma_r = funsig("sycl", "genfloat", "lgamma_r", ["genfloat", "genint"], "-1", "", [2])
    sig_list.append(f_lgamma_r)

    f_log = funsig("sycl", "genfloat", "log", ["genfloat"], "3")
    sig_list.append(f_log)

    f_log2 = funsig("sycl", "genfloat", "log2", ["genfloat"], "3")
    sig_list.append(f_log2)

    f_log10 = funsig("sycl", "genfloat", "log10", ["genfloat"], "3")
    sig_list.append(f_log10)

    f_log1p = funsig("sycl", "genfloat", "log1p", ["genfloat"], "2")
    sig_list.append(f_log1p)

    f_logb = funsig("sycl", "genfloat", "logb", ["genfloat"], "0")
    sig_list.append(f_logb)

    f_mad = funsig("sycl", "genfloat", "mad", ["genfloat","genfloat","genfloat"], "-1")
    sig_list.append(f_mad)

    f_maxmag = funsig("sycl", "genfloat", "maxmag", ["genfloat","genfloat"], "0")
    sig_list.append(f_maxmag)

    f_minmag = funsig("sycl", "genfloat", "minmag", ["genfloat","genfloat"], "0")
    sig_list.append(f_minmag)

    f_modf = funsig("sycl", "genfloat", "modf", ["genfloat", "genfloat"], "0", "", [2])
    sig_list.append(f_modf)

    f_nan = funsig("sycl", "genfloatf", "nan", ["ugenint"], "0")
    sig_list.append(f_nan)

    f_nan_2 = funsig("sycl", "genfloatd", "nan", ["ugenlonginteger"], "0")
    sig_list.append(f_nan_2)

    f_nextafter = funsig("sycl", "genfloat", "nextafter", ["genfloat", "genfloat"], "0")
    sig_list.append(f_nextafter)

    f_pow = funsig("sycl", "genfloat", "pow", ["genfloat", "genfloat"], "16")
    sig_list.append(f_pow)

    f_pown = funsig("sycl", "genfloat", "pown", ["genfloat", "genint"], "16")
    sig_list.append(f_pown)

    f_powr = funsig("sycl", "genfloat", "powr", ["genfloat", "genfloat"], "16")
    sig_list.append(f_powr)

    f_remainder = funsig("sycl", "genfloat", "remainder", ["genfloat", "genfloat"], "0")
    sig_list.append(f_remainder)

    f_remquo = funsig("sycl", "genfloat", "remquo", ["genfloat", "genfloat", "genint"], "0", "", [3])
    sig_list.append(f_remquo)

    f_rint = funsig("sycl", "genfloat", "rint", ["genfloat"], "0")
    sig_list.append(f_rint)

    f_rootn = funsig("sycl", "genfloat", "rootn", ["genfloat", "genint"], "16")
    sig_list.append(f_rootn)

    f_round = funsig("sycl", "genfloat", "round", ["genfloat"], "0")
    sig_list.append(f_round)

    f_rsqrt = funsig("sycl", "genfloat", "rsqrt", ["genfloat"], "2")
    sig_list.append(f_rsqrt)

    f_sin = funsig("sycl", "genfloat", "sin", ["genfloat"], "4")
    sig_list.append(f_sin)

    f_sincos = funsig("sycl", "genfloat", "sincos", ["genfloat", "genfloat"], "4", "",[2])
    sig_list.append(f_sincos)

    f_sinh = funsig("sycl", "genfloat", "sinh", ["genfloat"], "4")
    sig_list.append(f_sinh)

    f_sinpi = funsig("sycl", "genfloat", "sinpi", ["genfloat"], "4")
    sig_list.append(f_sinpi)

    f_sqrt = funsig("sycl", "genfloat", "sqrt", ["genfloat"], "3")
    sig_list.append(f_sqrt)

    f_tan = funsig("sycl", "genfloat", "tan", ["genfloat"], "5")
    sig_list.append(f_tan)

    f_tanh = funsig("sycl", "genfloat", "tanh", ["genfloat"], "5")
    sig_list.append(f_tanh)

    f_tanpi = funsig("sycl", "genfloat", "tanpi", ["genfloat"], "6")
    sig_list.append(f_tanpi)

    f_tgamma = funsig("sycl", "genfloat", "tgamma", ["genfloat"], "16")
    sig_list.append(f_tgamma)

    f_trunc = funsig("sycl", "genfloat", "trunc", ["genfloat"], "0")
    sig_list.append(f_trunc)

    return sig_list

def create_native_signatures():
    sig_list = []

    f_cos = funsig("sycl::native", "genfloatf", "cos", ["genfloatf"], "-1")
    sig_list.append(f_cos)

    f_divide = funsig("sycl::native", "genfloatf", "divide", ["genfloatf", "genfloatf"], "-1")
    sig_list.append(f_divide)

    f_exp = funsig("sycl::native", "genfloatf", "exp", ["genfloatf"], "-1")
    sig_list.append(f_exp)

    f_exp2 = funsig("sycl::native", "genfloatf", "exp2", ["genfloatf"], "-1")
    sig_list.append(f_exp2)

    f_exp10 = funsig("sycl::native", "genfloatf", "exp10", ["genfloatf"], "-1")
    sig_list.append(f_exp10)

    f_log = funsig("sycl::native", "genfloatf", "log", ["genfloatf"], "-1")
    sig_list.append(f_log)

    f_log2 = funsig("sycl::native", "genfloatf", "log2", ["genfloatf"], "-1")
    sig_list.append(f_log2)

    f_log10 = funsig("sycl::native", "genfloatf", "log10", ["genfloatf"], "-1")
    sig_list.append(f_log10)

    f_powr = funsig("sycl::native", "genfloatf", "powr", ["genfloatf", "genfloatf"], "-1")
    sig_list.append(f_powr)

    f_recip = funsig("sycl::native", "genfloatf", "recip", ["genfloatf"], "-1")
    sig_list.append(f_recip)

    f_rsqrt = funsig("sycl::native", "genfloatf", "rsqrt", ["genfloatf"], "-1")
    sig_list.append(f_rsqrt)

    f_sin = funsig("sycl::native", "genfloatf", "sin", ["genfloatf"], "-1")
    sig_list.append(f_sin)

    f_sqrt = funsig("sycl::native", "genfloatf", "sqrt", ["genfloatf"], "-1")
    sig_list.append(f_sqrt)

    f_tan = funsig("sycl::native", "genfloatf", "tan", ["genfloatf"], "-1")
    sig_list.append(f_tan)

    return sig_list

def create_half_signatures():
    sig_list = []

    f_cos = funsig("sycl::half_precision", "genfloatf", "cos", ["genfloatf"], "8192")
    sig_list.append(f_cos)

    f_divide = funsig("sycl::half_precision", "genfloatf", "divide", ["genfloatf", "genfloatf"], "8192")
    sig_list.append(f_divide)

    f_exp = funsig("sycl::half_precision", "genfloatf", "exp", ["genfloatf"], "8192")
    sig_list.append(f_exp)

    f_exp2 = funsig("sycl::half_precision", "genfloatf", "exp2", ["genfloatf"], "8192")
    sig_list.append(f_exp2)

    f_exp10 = funsig("sycl::half_precision", "genfloatf", "exp10", ["genfloatf"], "8192")
    sig_list.append(f_exp10)

    f_log = funsig("sycl::half_precision", "genfloatf", "log", ["genfloatf"], "8192")
    sig_list.append(f_log)

    f_log2 = funsig("sycl::half_precision", "genfloatf", "log2", ["genfloatf"], "8192")
    sig_list.append(f_log2)

    f_log10 = funsig("sycl::half_precision", "genfloatf", "log10", ["genfloatf"], "8192")
    sig_list.append(f_log10)

    f_powr = funsig("sycl::half_precision", "genfloatf", "powr", ["genfloatf", "genfloatf"], "8192")
    sig_list.append(f_powr)

    f_recip = funsig("sycl::half_precision", "genfloatf", "recip", ["genfloatf"], "8192")
    sig_list.append(f_recip)

    f_rsqrt = funsig("sycl::half_precision", "genfloatf", "rsqrt", ["genfloatf"], "8192")
    sig_list.append(f_rsqrt)

    f_sin = funsig("sycl::half_precision", "genfloatf", "sin", ["genfloatf"], "8192")
    sig_list.append(f_sin)

    f_sqrt = funsig("sycl::half_precision", "genfloatf", "sqrt", ["genfloatf"], "8192")
    sig_list.append(f_sqrt)

    f_tan = funsig("sycl::half_precision", "genfloatf", "tan", ["genfloatf"], "8192")
    sig_list.append(f_tan)

    return sig_list
