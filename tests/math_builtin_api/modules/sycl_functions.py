"""Represents a function signature."""
class funsig:
    def __init__(self, namespace, ret_type, name, arg_types=[], accuracy="", comment="", pntr_indx=[], mutations=[]):
        self.namespace = namespace # Namespace of function.
        self.ret_type = ret_type # Function return type.
        self.name = name # Function name.
        self.arg_types = arg_types # List containing the function argument types.
        self.accuracy = accuracy # The function maximum relative error defined as ulp.
        self.comment = comment # The comment for function maximum relative error.
        self.pntr_indx = pntr_indx # List containing the indexes of the arguments which are pointers.
        self.mutations = mutations # List containing triples: [first type category, second type category, mutation]
        # The type categories refer to the return type category or to an argument type category of the function.
        # The type categories are used to pick actual types. The mutation refers to what is allowed to differ between the actual two types.
        # Types have var_type, base_type and dim, see sycl_types.py
        # Mutation is one of "dim" (meaning that base_type must be the same but the dim and var_type may differ, e.g. float and vec<float, 2>),
        # "base_type" (meaning that the base_type may differ, e.g. char and int, but dim and var_type should be the same),
        # or "base_type_but_same_sizeof" (meaning that the base_type may differ but should keep the same byte size, and dim and var_type should be the same,
        # e.g. int32_t and float are OK, but not int64_t and marray<float, 2>)
    def __eq__(self, other):
        if isinstance(other, funsig):
            return ((self.namespace == other.namespace) and
                    (self.ret_type == other.ret_type) and
                    (self.name == other.name) and
                    (self.arg_types == other.arg_types) and
                    (self.accuracy == other.accuracy) and
                    (self.comment == other.comment) and
                    (self.pntr_indx == other.pntr_indx) and
                    (self.mutations == other.mutations))
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash((self.namespace, self.ret_type, self.name, str(self.arg_types), self.accuracy, self.comment, str(self.pntr_indx), str(self.mutations)))

def create_integer_signatures():
    sig_list = []

    f_abs = funsig("sycl", "geninteger", "abs", ["geninteger"])
    sig_list.append(f_abs)

    f_abs_diff = funsig("sycl", "geninteger", "abs_diff", ["geninteger", "geninteger"], "0")
    sig_list.append(f_abs_diff)

    f_add_sat = funsig("sycl", "geninteger", "add_sat", ["geninteger", "geninteger"])
    sig_list.append(f_add_sat)

    f_hadd = funsig("sycl", "geninteger", "hadd", ["geninteger", "geninteger"])
    sig_list.append(f_hadd)

    f_rhadd = funsig("sycl", "geninteger", "rhadd", ["geninteger", "geninteger"])
    sig_list.append(f_rhadd)

    f_clamp = funsig("sycl", "geninteger", "clamp", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_clamp)

    f_clamp_2 = funsig("sycl", "geninteger", "clamp", ["geninteger", "sgeninteger", "sgeninteger"], "0", "", [], [["geninteger", "sgeninteger", "dim"]])
    sig_list.append(f_clamp_2)

    f_clz = funsig("sycl", "geninteger", "clz", ["geninteger"])
    sig_list.append(f_clz)

    f_ctz = funsig("sycl", "geninteger", "ctz", ["geninteger"])
    sig_list.append(f_ctz)

    f_mad_hi = funsig("sycl", "geninteger", "mad_hi", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_mad_hi)

    f_mad_sat = funsig("sycl", "geninteger", "mad_sat", ["geninteger", "geninteger", "geninteger"])
    sig_list.append(f_mad_sat)

    f_max = funsig("sycl", "geninteger", "max", ["geninteger", "geninteger"])
    sig_list.append(f_max)

    f_max_2 = funsig("sycl", "geninteger", "max", ["geninteger", "sgeninteger"], "0", "", [], [["geninteger", "sgeninteger", "dim"]])
    sig_list.append(f_max_2)

    f_min = funsig("sycl", "geninteger", "min", ["geninteger", "geninteger"])
    sig_list.append(f_min)

    f_min_2 = funsig("sycl", "geninteger", "min", ["geninteger", "sgeninteger"], "0", "", [], [["geninteger", "sgeninteger", "dim"]])
    sig_list.append(f_min_2)

    f_mul_hi = funsig("sycl", "geninteger", "mul_hi", ["geninteger", "geninteger"])
    sig_list.append(f_mul_hi)

    f_rotate = funsig("sycl", "geninteger", "rotate", ["geninteger", "geninteger"])
    sig_list.append(f_rotate)

    f_sub_sat = funsig("sycl", "geninteger", "sub_sat", ["geninteger", "geninteger"])
    sig_list.append(f_sub_sat)

    f_upsample = funsig("sycl", "ugeninteger16bit", "upsample", ["ugeninteger8bit", "ugeninteger8bit"], "0", "", [], [["ugeninteger16bit", "ugeninteger8bit", "base_type"]])
    sig_list.append(f_upsample)

    f_upsample_2 = funsig("sycl", "igeninteger16bit", "upsample", ["igeninteger8bit", "ugeninteger8bit"], "0", "", [], [["igeninteger16bit", "igeninteger8bit", "base_type"], ["igeninteger16bit", "ugeninteger8bit", "base_type"]])
    sig_list.append(f_upsample_2)

    f_upsample_3 = funsig("sycl", "ugeninteger32bit", "upsample", ["ugeninteger16bit", "ugeninteger16bit"], "0", "", [], [["ugeninteger32bit", "ugeninteger16bit", "base_type"]])
    sig_list.append(f_upsample_3)

    f_upsample_4 = funsig("sycl", "igeninteger32bit", "upsample", ["igeninteger16bit", "ugeninteger16bit"], "0", "", [], [["igeninteger32bit", "igeninteger16bit", "base_type"], ["igeninteger32bit", "ugeninteger16bit", "base_type"]])
    sig_list.append(f_upsample_4)

    f_upsample_5 = funsig("sycl", "ugeninteger64bit", "upsample", ["ugeninteger32bit", "ugeninteger32bit"], "0", "", [], [["ugeninteger64bit", "ugeninteger32bit", "base_type"]])
    sig_list.append(f_upsample_5)

    f_upsample_6 = funsig("sycl", "igeninteger64bit", "upsample", ["igeninteger32bit", "ugeninteger32bit"], "0", "", [], [["igeninteger64bit", "igeninteger32bit", "base_type"], ["igeninteger64bit", "ugeninteger32bit", "base_type"]])
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

    f_clamp_4 = funsig("sycl", "genfloath", "clamp", ["genfloath", "sycl::half", "sycl::half"])
    sig_list.append(f_clamp_4)

    f_degrees = funsig("sycl", "genfloat", "degrees", ["genfloat"], "3")
    sig_list.append(f_degrees)

    f_max = funsig("sycl", "genfloat", "max", ["genfloat", "genfloat"])
    sig_list.append(f_max)

    f_max_2 = funsig("sycl", "genfloatf", "max", ["genfloatf", "float"])
    sig_list.append(f_max_2)

    f_max_3 = funsig("sycl", "genfloatd", "max", ["genfloatd", "double"])
    sig_list.append(f_max_3)

    f_max_4 = funsig("sycl", "genfloath", "max", ["genfloath", "sycl::half"])
    sig_list.append(f_max_4)

    f_min = funsig("sycl", "genfloat", "min", ["genfloat", "genfloat"])
    sig_list.append(f_min)

    f_min_2 = funsig("sycl", "genfloatf", "min", ["genfloatf", "float"])
    sig_list.append(f_min_2)

    f_min_3 = funsig("sycl", "genfloatd", "min", ["genfloatd", "double"])
    sig_list.append(f_min_3)

    f_min_4 = funsig("sycl", "genfloath", "min", ["genfloath", "sycl::half"])
    sig_list.append(f_min_4)

    f_mix = funsig("sycl", "genfloat", "mix", ["genfloat", "genfloat", "genfloat"], "1")
    sig_list.append(f_mix)

    f_mix_2 = funsig("sycl", "genfloatf", "mix", ["genfloatf", "genfloatf", "float"], "1")
    sig_list.append(f_mix_2)

    f_mix_3 = funsig("sycl", "genfloatd", "mix", ["genfloatd", "genfloatd", "double"], "1")
    sig_list.append(f_mix_3)

    f_mix_4 = funsig("sycl", "genfloath", "mix", ["genfloath", "genfloath", "sycl::half"], "1")
    sig_list.append(f_mix_4)

    f_radians = funsig("sycl", "genfloat", "radians", ["genfloat"], "3")
    sig_list.append(f_radians)

    f_step = funsig("sycl", "genfloat", "step", ["genfloat", "genfloat"])
    sig_list.append(f_step)

    f_step_2 = funsig("sycl", "genfloatf", "step", ["float", "genfloatf"])
    sig_list.append(f_step_2)

    f_step_3 = funsig("sycl", "genfloatd", "step", ["double", "genfloatd"])
    sig_list.append(f_step_3)

    f_step_4 = funsig("sycl", "genfloath", "step", ["sycl::half", "genfloath"])
    sig_list.append(f_step_4)

    f_smoothstep = funsig("sycl", "genfloat", "smoothstep", ["genfloat", "genfloat", "genfloat"])
    sig_list.append(f_smoothstep)

    f_smoothstep_2 = funsig("sycl", "genfloatf", "smoothstep", ["float", "float", "genfloatf"])
    sig_list.append(f_smoothstep_2)

    f_smoothstep_3 = funsig("sycl", "genfloatd", "smoothstep", ["double", "double", "genfloatd"])
    sig_list.append(f_smoothstep_3)

    f_smoothstep_4 = funsig("sycl", "genfloath", "smoothstep", ["sycl::half", "sycl::half", "genfloath"])
    sig_list.append(f_smoothstep_4)

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

    f_cross_5 = funsig("sycl", "sycl::marray<float, 3>", "cross", ["sycl::marray<float, 3>", "sycl::marray<float, 3>"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_5)

    f_cross_6 = funsig("sycl", "sycl::marray<float, 4>", "cross", ["sycl::marray<float, 4>", "sycl::marray<float, 4>"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_6)

    f_cross_7 = funsig("sycl", "sycl::marray<double, 3>", "cross", ["sycl::marray<double, 3>", "sycl::marray<double, 3>"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_7)

    f_cross_8 = funsig("sycl", "sycl::marray<double, 4>", "cross", ["sycl::marray<double, 4>", "sycl::marray<double, 4>"], "3",
            "cumulative error for multiplications and substruction for each component")
    sig_list.append(f_cross_8)


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

    f_isequal = funsig("sycl", "vigeninteger", "isequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isequal)

    f_isequal_2 = funsig("sycl", "bool", "isequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isequal_2)

    f_isequal_3 = funsig("sycl", "mbooln", "isequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isequal_3)


    f_isnotequal = funsig("sycl", "vigeninteger", "isnotequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isnotequal)

    f_isnotequal_2 = funsig("sycl", "bool", "isnotequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isnotequal_2)

    f_isnotequal_3 = funsig("sycl", "mbooln", "isnotequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isnotequal_3)


    f_isgreater = funsig("sycl", "vigeninteger", "isgreater", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isgreater)

    f_isgreater_2 = funsig("sycl", "bool", "isgreater", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isgreater_2)

    f_isgreater_3 = funsig("sycl", "mbooln", "isgreater", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isgreater_3)


    f_isgreaterequal = funsig("sycl", "vigeninteger", "isgreaterequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isgreaterequal)

    f_isgreaterequal_2 = funsig("sycl", "bool", "isgreaterequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isgreaterequal_2)

    f_isgreaterequal_3 = funsig("sycl", "mbooln", "isgreaterequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isgreaterequal_3)


    f_isless = funsig("sycl", "vigeninteger", "isless", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isless)

    f_isless_2 = funsig("sycl", "bool", "isless", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isless_2)

    f_isless_3 = funsig("sycl", "mbooln", "isless", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isless_3)


    f_islessequal = funsig("sycl", "vigeninteger", "islessequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_islessequal)

    f_islessequal_2 = funsig("sycl", "bool", "islessequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_islessequal_2)

    f_islessequal_3 = funsig("sycl", "mbooln", "islessequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_islessequal_3)


    f_islessgreater = funsig("sycl", "vigeninteger", "islessgreater", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_islessgreater)

    f_islessgreater_2 = funsig("sycl", "bool", "islessgreater", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_islessgreater_2)

    f_islessgreater_3 = funsig("sycl", "mbooln", "islessgreater", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_islessgreater_3)


    f_isfinite = funsig("sycl", "vigeninteger", "isfinite", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isfinite)

    f_isfinite_2 = funsig("sycl", "bool", "isfinite", ["sgenfloat"])
    sig_list.append(f_isfinite_2)

    f_isfinite_3 = funsig("sycl", "mbooln", "isfinite", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isfinite_3)


    f_isinf = funsig("sycl", "vigeninteger", "isinf", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isinf)

    f_isinf_2 = funsig("sycl", "bool", "isinf", ["sgenfloat"])
    sig_list.append(f_isinf_2)

    f_isinf_3 = funsig("sycl", "mbooln", "isinf", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isinf_3)


    f_isnan = funsig("sycl", "vigeninteger", "isnan", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isnan)

    f_isnan_2 = funsig("sycl", "bool", "isnan", ["sgenfloat"])
    sig_list.append(f_isnan_2)

    f_isnan_3 = funsig("sycl", "mbooln", "isnan", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isnan_3)


    f_isnormal = funsig("sycl", "vigeninteger", "isnormal", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isnormal)

    f_isnormal_2 = funsig("sycl", "bool", "isnormal", ["sgenfloat"])
    sig_list.append(f_isnormal_2)

    f_isnormal_3 = funsig("sycl", "mbooln", "isnormal", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isnormal_3)


    f_isordered = funsig("sycl", "vigeninteger", "isordered", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isordered)

    f_isordered_2 = funsig("sycl", "bool", "isordered", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isordered_2)

    f_isordered_3 = funsig("sycl", "mbooln", "isordered", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isordered_3)


    f_isunordered = funsig("sycl", "vigeninteger", "isunordered", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_isunordered)

    f_isunordered_2 = funsig("sycl", "bool", "isunordered", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isunordered_2)

    f_isunordered_3 = funsig("sycl", "mbooln", "isunordered", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_isunordered_3)


    f_signbit = funsig("sycl", "vigeninteger", "signbit", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_signbit)

    f_signbit_2 = funsig("sycl", "bool", "signbit", ["sgenfloat"])
    sig_list.append(f_signbit_2)

    f_signbit_3 = funsig("sycl", "mbooln", "signbit", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]])
    sig_list.append(f_signbit_3)


    f_any = funsig("sycl", "int", "any", ["vigeninteger"])
    sig_list.append(f_any)

    f_any_2 = funsig("sycl", "bool", "any", ["sigeninteger"])
    sig_list.append(f_any_2)

    f_any_3 = funsig("sycl", "bool", "any", ["migeninteger"])
    sig_list.append(f_any_3)


    f_all = funsig("sycl", "int", "all", ["vigeninteger"])
    sig_list.append(f_all)

    f_all_2 = funsig("sycl", "bool", "all", ["sigeninteger"])
    sig_list.append(f_all_2)

    f_all_3 = funsig("sycl", "bool", "all", ["migeninteger"])
    sig_list.append(f_all_3)


    f_bitselect = funsig("sycl", "gentype", "bitselect", ["gentype", "gentype", "gentype"])
    sig_list.append(f_bitselect)


    f_select = funsig("sycl", "vgentype", "select", ["vgentype", "vgentype", "vigeninteger"], "0", "", [], [["vgentype", "vigeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_select)

    f_select_2 = funsig("sycl", "vgentype", "select", ["vgentype", "vgentype", "vugeninteger"], "0", "", [], [["vgentype", "vugeninteger", "base_type_but_same_sizeof"]])
    sig_list.append(f_select_2)

    f_select_3 = funsig("sycl", "sgentype", "select", ["sgentype", "sgentype", "bool"])
    sig_list.append(f_select_3)

    f_select_4 = funsig("sycl", "mgentype", "select", ["mgentype", "mgentype", "mbooln"], "0", "", [], [["mgentype", "mbooln", "base_type"]])
    sig_list.append(f_select_4)


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

    f_fmax_2 = funsig("sycl", "genfloat", "fmax", ["genfloat", "sgenfloat"], "0", "", [], [["genfloat", "sgenfloat", "dim"]])
    sig_list.append(f_fmax_2)

    f_fmin = funsig("sycl", "genfloat", "fmin", ["genfloat", "genfloat"], "0")
    sig_list.append(f_fmin)

    f_fmin_2 = funsig("sycl", "genfloat", "fmin", ["genfloat", "sgenfloat"], "0", "", [], [["genfloat", "sgenfloat", "dim"]])
    sig_list.append(f_fmin_2)

    f_fmod = funsig("sycl", "genfloat", "fmod", ["genfloat", "genfloat"], "0")
    sig_list.append(f_fmod)

    f_fract = funsig("sycl", "genfloat", "fract", ["genfloat", "genfloat"], "0", "", [2])
    sig_list.append(f_fract)

    f_frexp = funsig("sycl", "genfloat", "frexp", ["genfloat", "genint"], "0", "", [2], [["genfloat", "genint", "base_type"]])
    sig_list.append(f_frexp)

    f_hypot = funsig("sycl", "genfloat", "hypot", ["genfloat", "genfloat"], "4")
    sig_list.append(f_hypot)

    f_ilogb = funsig("sycl", "genint", "ilogb", ["genfloat"], "0", "", [], [["genfloat", "genint", "base_type"]])
    sig_list.append(f_ilogb)

    f_ldexp = funsig("sycl", "genfloat", "ldexp", ["genfloat", "genint"], "0", "", [], [["genfloat", "genint", "base_type"]])
    sig_list.append(f_ldexp)

    f_ldexp_2 = funsig("sycl", "genfloat", "ldexp", ["genfloat", "int"], "0")
    sig_list.append(f_ldexp_2)

    f_lgamma = funsig("sycl", "genfloat", "lgamma", ["genfloat"], "-1")
    sig_list.append(f_lgamma)

    f_lgamma_r = funsig("sycl", "genfloat", "lgamma_r", ["genfloat", "genint"], "-1", "", [2], [["genfloat", "genint", "base_type"]])
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

    f_nan = funsig("sycl", "genfloatf", "nan", ["ugenint"], "0", "", [], [["genfloatf", "ugenint", "base_type"]])
    sig_list.append(f_nan)

    f_nan_2 = funsig("sycl", "genfloatd", "nan", ["ugenlonginteger"], "0", "", [], [["genfloatd", "ugenlonginteger", "base_type"]])
    sig_list.append(f_nan_2)

    f_nextafter = funsig("sycl", "genfloat", "nextafter", ["genfloat", "genfloat"], "0")
    sig_list.append(f_nextafter)

    f_pow = funsig("sycl", "genfloat", "pow", ["genfloat", "genfloat"], "16")
    sig_list.append(f_pow)

    f_pown = funsig("sycl", "genfloat", "pown", ["genfloat", "genint"], "16", "", [], [["genfloat", "genint", "base_type"]])
    sig_list.append(f_pown)

    f_powr = funsig("sycl", "genfloat", "powr", ["genfloat", "genfloat"], "16")
    sig_list.append(f_powr)

    f_remainder = funsig("sycl", "genfloat", "remainder", ["genfloat", "genfloat"], "0")
    sig_list.append(f_remainder)

    f_remquo = funsig("sycl", "genfloat", "remquo", ["genfloat", "genfloat", "genint"], "0", "", [3], [["genfloat", "genint", "base_type"]])
    sig_list.append(f_remquo)

    f_rint = funsig("sycl", "genfloat", "rint", ["genfloat"], "0")
    sig_list.append(f_rint)

    f_rootn = funsig("sycl", "genfloat", "rootn", ["genfloat", "genint"], "16", "", [], [["genfloat", "genint", "base_type"]])
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
