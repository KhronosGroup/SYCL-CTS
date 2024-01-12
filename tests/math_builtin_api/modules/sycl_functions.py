"""Represents a function signature."""
class funsig:
    def __init__(self, namespace, ret_type, name, arg_types=[], accuracy="", comment="", pntr_indx=[], mutations=[], template_arg_map=[]):
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
        self.template_arg_map = template_arg_map # List of indices mapping template arugments to function argument types.
        # An empty list signifies a non-templated function.
    def __eq__(self, other):
        if isinstance(other, funsig):
            return ((self.namespace == other.namespace) and
                    (self.ret_type == other.ret_type) and
                    (self.name == other.name) and
                    (self.arg_types == other.arg_types) and
                    (self.accuracy == other.accuracy) and
                    (self.comment == other.comment) and
                    (self.pntr_indx == other.pntr_indx) and
                    (self.mutations == other.mutations) and
                    (self.template_arg_map == other.template_arg_map))
        else:
            return False
    def __ne__(self, other):
        return (not self.__eq__(other))
    def __hash__(self):
        return hash((self.namespace, self.ret_type, self.name, str(self.arg_types), self.accuracy, self.comment, str(self.pntr_indx), str(self.mutations), str(self.template_arg_map)))

def create_integer_signatures():
    sig_list = []

    f_abs = funsig("sycl", "geninteger", "abs", ["geninteger"], template_arg_map=[0])
    sig_list.append(f_abs)

    f_abs_diff = funsig("sycl", "geninteger", "abs_diff", ["geninteger", "geninteger"], "0", template_arg_map=[0,1])
    sig_list.append(f_abs_diff)

    f_add_sat = funsig("sycl", "geninteger", "add_sat", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_add_sat)

    f_hadd = funsig("sycl", "geninteger", "hadd", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_hadd)

    f_rhadd = funsig("sycl", "geninteger", "rhadd", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_rhadd)

    f_clamp = funsig("sycl", "geninteger", "clamp", ["geninteger", "geninteger", "geninteger"], template_arg_map=[0,1,2])
    sig_list.append(f_clamp)

    f_clamp_2 = funsig("sycl", "geninteger", "clamp", ["geninteger", "sgeninteger", "sgeninteger"], "0", "", [], [["geninteger", "sgeninteger", "dim"]], template_arg_map=[0])
    sig_list.append(f_clamp_2)

    f_clz = funsig("sycl", "geninteger", "clz", ["geninteger"], template_arg_map=[0])
    sig_list.append(f_clz)

    f_ctz = funsig("sycl", "geninteger", "ctz", ["geninteger"], template_arg_map=[0])
    sig_list.append(f_ctz)

    f_mad_hi = funsig("sycl", "geninteger", "mad_hi", ["geninteger", "geninteger", "geninteger"], template_arg_map=[0,1,2])
    sig_list.append(f_mad_hi)

    f_mad_sat = funsig("sycl", "geninteger", "mad_sat", ["geninteger", "geninteger", "geninteger"], template_arg_map=[0,1,2])
    sig_list.append(f_mad_sat)

    f_max = funsig("sycl", "geninteger", "max", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_max)

    f_max_2 = funsig("sycl", "geninteger", "max", ["geninteger", "sgeninteger"], "0", "", [], [["geninteger", "sgeninteger", "dim"]], template_arg_map=[0])
    sig_list.append(f_max_2)

    f_min = funsig("sycl", "geninteger", "min", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_min)

    f_min_2 = funsig("sycl", "geninteger", "min", ["geninteger", "sgeninteger"], "0", "", [], [["geninteger", "sgeninteger", "dim"]], template_arg_map=[0])
    sig_list.append(f_min_2)

    f_mul_hi = funsig("sycl", "geninteger", "mul_hi", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_mul_hi)

    f_rotate = funsig("sycl", "geninteger", "rotate", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_rotate)

    f_sub_sat = funsig("sycl", "geninteger", "sub_sat", ["geninteger", "geninteger"], template_arg_map=[0,1])
    sig_list.append(f_sub_sat)

    f_upsample = funsig("sycl", "ugeninteger16bit", "upsample", ["ugeninteger8bit", "ugeninteger8bit"], "0", "", [], [["ugeninteger16bit", "ugeninteger8bit", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_upsample)

    f_upsample_2 = funsig("sycl", "igeninteger16bit", "upsample", ["igeninteger8bit", "ugeninteger8bit"], "0", "", [], [["igeninteger16bit", "igeninteger8bit", "base_type"], ["igeninteger16bit", "ugeninteger8bit", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_upsample_2)

    f_upsample_3 = funsig("sycl", "ugeninteger32bit", "upsample", ["ugeninteger16bit", "ugeninteger16bit"], "0", "", [], [["ugeninteger32bit", "ugeninteger16bit", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_upsample_3)

    f_upsample_4 = funsig("sycl", "igeninteger32bit", "upsample", ["igeninteger16bit", "ugeninteger16bit"], "0", "", [], [["igeninteger32bit", "igeninteger16bit", "base_type"], ["igeninteger32bit", "ugeninteger16bit", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_upsample_4)

    f_upsample_5 = funsig("sycl", "ugeninteger64bit", "upsample", ["ugeninteger32bit", "ugeninteger32bit"], "0", "", [], [["ugeninteger64bit", "ugeninteger32bit", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_upsample_5)

    f_upsample_6 = funsig("sycl", "igeninteger64bit", "upsample", ["igeninteger32bit", "ugeninteger32bit"], "0", "", [], [["igeninteger64bit", "igeninteger32bit", "base_type"], ["igeninteger64bit", "ugeninteger32bit", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_upsample_6)

    f_popcount = funsig("sycl", "geninteger", "popcount", ["geninteger"], template_arg_map=[0])
    sig_list.append(f_popcount)

    f_mad24 = funsig("sycl", "geninteger32bit", "mad24", ["geninteger32bit","geninteger32bit","geninteger32bit"], template_arg_map=[0,1,2])
    sig_list.append(f_mad24)

    f_mul24 = funsig("sycl", "geninteger32bit", "mul24", ["geninteger32bit","geninteger32bit"], template_arg_map=[0,1])
    sig_list.append(f_mul24)

    return sig_list

def create_common_signatures():
    sig_list = []

    f_clamp = funsig("sycl", "genfloat", "clamp", ["genfloat", "genfloat", "genfloat"], template_arg_map=[0,1,2])
    sig_list.append(f_clamp)

    f_clamp_2 = funsig("sycl", "genfloatf", "clamp", ["genfloatf", "float", "float"], template_arg_map=[0])
    sig_list.append(f_clamp_2)

    f_clamp_3 = funsig("sycl", "genfloatd", "clamp", ["genfloatd", "double", "double"], template_arg_map=[0])
    sig_list.append(f_clamp_3)

    f_clamp_4 = funsig("sycl", "genfloath", "clamp", ["genfloath", "sycl::half", "sycl::half"], template_arg_map=[0])
    sig_list.append(f_clamp_4)

    f_degrees = funsig("sycl", "genfloat", "degrees", ["genfloat"], "3", template_arg_map=[0])
    sig_list.append(f_degrees)

    f_max = funsig("sycl", "genfloat", "max", ["genfloat", "genfloat"], template_arg_map=[0,1])
    sig_list.append(f_max)

    f_max_2 = funsig("sycl", "genfloatf", "max", ["genfloatf", "float"], template_arg_map=[0])
    sig_list.append(f_max_2)

    f_max_3 = funsig("sycl", "genfloatd", "max", ["genfloatd", "double"], template_arg_map=[0])
    sig_list.append(f_max_3)

    f_max_4 = funsig("sycl", "genfloath", "max", ["genfloath", "sycl::half"], template_arg_map=[0])
    sig_list.append(f_max_4)

    f_min = funsig("sycl", "genfloat", "min", ["genfloat", "genfloat"], template_arg_map=[0,1])
    sig_list.append(f_min)

    f_min_2 = funsig("sycl", "genfloatf", "min", ["genfloatf", "float"], template_arg_map=[0])
    sig_list.append(f_min_2)

    f_min_3 = funsig("sycl", "genfloatd", "min", ["genfloatd", "double"], template_arg_map=[0])
    sig_list.append(f_min_3)

    f_min_4 = funsig("sycl", "genfloath", "min", ["genfloath", "sycl::half"], template_arg_map=[0])
    sig_list.append(f_min_4)

    f_mix = funsig("sycl", "genfloat", "mix", ["genfloat", "genfloat", "genfloat"], "1", template_arg_map=[0,1,2])
    sig_list.append(f_mix)

    f_mix_2 = funsig("sycl", "genfloatf", "mix", ["genfloatf", "genfloatf", "float"], "1", template_arg_map=[0,1])
    sig_list.append(f_mix_2)

    f_mix_3 = funsig("sycl", "genfloatd", "mix", ["genfloatd", "genfloatd", "double"], "1", template_arg_map=[0,1])
    sig_list.append(f_mix_3)

    f_mix_4 = funsig("sycl", "genfloath", "mix", ["genfloath", "genfloath", "sycl::half"], "1", template_arg_map=[0,1])
    sig_list.append(f_mix_4)

    f_radians = funsig("sycl", "genfloat", "radians", ["genfloat"], "3", template_arg_map=[0])
    sig_list.append(f_radians)

    f_step = funsig("sycl", "genfloat", "step", ["genfloat", "genfloat"], template_arg_map=[0,1])
    sig_list.append(f_step)

    f_step_2 = funsig("sycl", "genfloatf", "step", ["float", "genfloatf"], template_arg_map=[1])
    sig_list.append(f_step_2)

    f_step_3 = funsig("sycl", "genfloatd", "step", ["double", "genfloatd"], template_arg_map=[1])
    sig_list.append(f_step_3)

    f_step_4 = funsig("sycl", "genfloath", "step", ["sycl::half", "genfloath"], template_arg_map=[1])
    sig_list.append(f_step_4)

    f_smoothstep = funsig("sycl", "genfloat", "smoothstep", ["genfloat", "genfloat", "genfloat"], template_arg_map=[0,1,2])
    sig_list.append(f_smoothstep)

    f_smoothstep_2 = funsig("sycl", "genfloatf", "smoothstep", ["float", "float", "genfloatf"], template_arg_map=[2])
    sig_list.append(f_smoothstep_2)

    f_smoothstep_3 = funsig("sycl", "genfloatd", "smoothstep", ["double", "double", "genfloatd"], template_arg_map=[2])
    sig_list.append(f_smoothstep_3)

    f_smoothstep_4 = funsig("sycl", "genfloath", "smoothstep", ["sycl::half", "sycl::half", "genfloath"], template_arg_map=[2])
    sig_list.append(f_smoothstep_4)

    f_sign = funsig("sycl", "genfloat", "sign", ["genfloat"], template_arg_map=[0])
    sig_list.append(f_sign)

    return sig_list

def create_geometric_signatures():
    sig_list = []

    f_cross = funsig("sycl", "sycl::float3", "cross", ["sycl::float3", "sycl::float3"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross)

    f_cross_2 = funsig("sycl", "sycl::float4", "cross", ["sycl::float4", "sycl::float4"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_2)

    f_cross_3 = funsig("sycl", "sycl::double3", "cross", ["sycl::double3", "sycl::double3"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_3)

    f_cross_4 = funsig("sycl", "sycl::double4", "cross", ["sycl::double4", "sycl::double4"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_4)

    f_cross_5 = funsig("sycl", "sycl::marray<float, 3>", "cross", ["sycl::marray<float, 3>", "sycl::marray<float, 3>"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_5)

    f_cross_6 = funsig("sycl", "sycl::marray<float, 4>", "cross", ["sycl::marray<float, 4>", "sycl::marray<float, 4>"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_6)

    f_cross_7 = funsig("sycl", "sycl::marray<double, 3>", "cross", ["sycl::marray<double, 3>", "sycl::marray<double, 3>"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_7)

    f_cross_8 = funsig("sycl", "sycl::marray<double, 4>", "cross", ["sycl::marray<double, 4>", "sycl::marray<double, 4>"], "3",
            "cumulative error for multiplications and substruction for each component", template_arg_map=[0,1])
    sig_list.append(f_cross_8)


    f_dot = funsig("sycl", "float", "dot", ["gengeofloat", "gengeofloat"], "2*vecSize - 1",
            "cumulative error for multiplications and additions 'vecSize + vecSize-1'", template_arg_map=[0,1])
    sig_list.append(f_dot)

    f_dot_2 = funsig("sycl", "double", "dot", ["gengeodouble", "gengeodouble"], "2*vecSize - 1",
            "cumulative error for multiplications and additions 'vecSize + vecSize-1'", template_arg_map=[0,1])
    sig_list.append(f_dot_2)

    f_distance = funsig("sycl", "float", "distance", ["gengeofloat", "gengeofloat"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'", template_arg_map=[0,1])
    sig_list.append(f_distance)

    f_distance_2 = funsig("sycl", "double", "distance", ["gengeodouble", "gengeodouble"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'", template_arg_map=[0,1])
    sig_list.append(f_distance_2)

    f_length = funsig("sycl", "float", "length", ["gengeofloat"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'", template_arg_map=[0])
    sig_list.append(f_length)

    f_length_2 = funsig("sycl", "double", "length", ["gengeodouble"], "2*vecSize + 2",
            "cumulative error for multiplications, additions and sqrt 'vecSize + vecSize-1 + 3'", template_arg_map=[0])
    sig_list.append(f_length_2)

    f_normalize = funsig("sycl", "gengeofloat", "normalize", ["gengeofloat"], "2*vecSize + 1",
            "cumulative error for multiplications, additions and rsqrt 'vecSize + vecSize-1 + 2'", template_arg_map=[0])
    sig_list.append(f_length)

    f_normalize_2 = funsig("sycl", "gengeodouble", "normalize", ["gengeodouble"], "2*vecSize + 1",
            "cumulative error for multiplications, additions and rsqrt 'vecSize + vecSize-1 + 2'", template_arg_map=[0])
    sig_list.append(f_normalize_2)

    f_fast_distance = funsig("sycl", "float", "fast_distance", ["gengeofloat", "gengeofloat"], "8192", template_arg_map=[0,1])
    sig_list.append(f_fast_distance)

    f_fast_length = funsig("sycl", "float", "fast_length", ["gengeofloat"], "8192", template_arg_map=[0])
    sig_list.append(f_fast_length)

    f_fast_normalize = funsig("sycl", "gengeofloat", "fast_normalize", ["gengeofloat"], "8192", template_arg_map=[0])
    sig_list.append(f_fast_normalize)

    return sig_list

def create_relational_signatures():
    sig_list = []

    f_isequal = funsig("sycl", "vigeninteger", "isequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isequal)

    f_isequal_2 = funsig("sycl", "bool", "isequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isequal_2)

    f_isequal_3 = funsig("sycl", "mbooln", "isequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isequal_3)


    f_isnotequal = funsig("sycl", "vigeninteger", "isnotequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isnotequal)

    f_isnotequal_2 = funsig("sycl", "bool", "isnotequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isnotequal_2)

    f_isnotequal_3 = funsig("sycl", "mbooln", "isnotequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isnotequal_3)


    f_isgreater = funsig("sycl", "vigeninteger", "isgreater", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isgreater)

    f_isgreater_2 = funsig("sycl", "bool", "isgreater", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isgreater_2)

    f_isgreater_3 = funsig("sycl", "mbooln", "isgreater", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isgreater_3)


    f_isgreaterequal = funsig("sycl", "vigeninteger", "isgreaterequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isgreaterequal)

    f_isgreaterequal_2 = funsig("sycl", "bool", "isgreaterequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isgreaterequal_2)

    f_isgreaterequal_3 = funsig("sycl", "mbooln", "isgreaterequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isgreaterequal_3)


    f_isless = funsig("sycl", "vigeninteger", "isless", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isless)

    f_isless_2 = funsig("sycl", "bool", "isless", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isless_2)

    f_isless_3 = funsig("sycl", "mbooln", "isless", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isless_3)


    f_islessequal = funsig("sycl", "vigeninteger", "islessequal", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_islessequal)

    f_islessequal_2 = funsig("sycl", "bool", "islessequal", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_islessequal_2)

    f_islessequal_3 = funsig("sycl", "mbooln", "islessequal", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_islessequal_3)


    f_islessgreater = funsig("sycl", "vigeninteger", "islessgreater", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_islessgreater)

    f_islessgreater_2 = funsig("sycl", "bool", "islessgreater", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_islessgreater_2)

    f_islessgreater_3 = funsig("sycl", "mbooln", "islessgreater", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_islessgreater_3)


    f_isfinite = funsig("sycl", "vigeninteger", "isfinite", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0])
    sig_list.append(f_isfinite)

    f_isfinite_2 = funsig("sycl", "bool", "isfinite", ["sgenfloat"])
    sig_list.append(f_isfinite_2)

    f_isfinite_3 = funsig("sycl", "mbooln", "isfinite", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0])
    sig_list.append(f_isfinite_3)


    f_isinf = funsig("sycl", "vigeninteger", "isinf", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0])
    sig_list.append(f_isinf)

    f_isinf_2 = funsig("sycl", "bool", "isinf", ["sgenfloat"])
    sig_list.append(f_isinf_2)

    f_isinf_3 = funsig("sycl", "mbooln", "isinf", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0])
    sig_list.append(f_isinf_3)


    f_isnan = funsig("sycl", "vigeninteger", "isnan", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0])
    sig_list.append(f_isnan)

    f_isnan_2 = funsig("sycl", "bool", "isnan", ["sgenfloat"])
    sig_list.append(f_isnan_2)

    f_isnan_3 = funsig("sycl", "mbooln", "isnan", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0])
    sig_list.append(f_isnan_3)


    f_isnormal = funsig("sycl", "vigeninteger", "isnormal", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0])
    sig_list.append(f_isnormal)

    f_isnormal_2 = funsig("sycl", "bool", "isnormal", ["sgenfloat"])
    sig_list.append(f_isnormal_2)

    f_isnormal_3 = funsig("sycl", "mbooln", "isnormal", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0])
    sig_list.append(f_isnormal_3)


    f_isordered = funsig("sycl", "vigeninteger", "isordered", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isordered)

    f_isordered_2 = funsig("sycl", "bool", "isordered", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isordered_2)

    f_isordered_3 = funsig("sycl", "mbooln", "isordered", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isordered_3)


    f_isunordered = funsig("sycl", "vigeninteger", "isunordered", ["vgenfloat", "vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1])
    sig_list.append(f_isunordered)

    f_isunordered_2 = funsig("sycl", "bool", "isunordered", ["sgenfloat", "sgenfloat"])
    sig_list.append(f_isunordered_2)

    f_isunordered_3 = funsig("sycl", "mbooln", "isunordered", ["mgenfloat", "mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_isunordered_3)


    f_signbit = funsig("sycl", "vigeninteger", "signbit", ["vgenfloat"], "0", "", [], [["vgenfloat", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0])
    sig_list.append(f_signbit)

    f_signbit_2 = funsig("sycl", "bool", "signbit", ["sgenfloat"])
    sig_list.append(f_signbit_2)

    f_signbit_3 = funsig("sycl", "mbooln", "signbit", ["mgenfloat"], "0", "", [], [["mbooln", "mgenfloat", "base_type"]], template_arg_map=[0])
    sig_list.append(f_signbit_3)


    f_any = funsig("sycl", "int", "any", ["vigeninteger"], template_arg_map=[0])
    sig_list.append(f_any)

    f_any_2 = funsig("sycl", "bool", "any", ["sigeninteger"], template_arg_map=[0])
    sig_list.append(f_any_2)

    f_any_3 = funsig("sycl", "bool", "any", ["migeninteger"], template_arg_map=[0])
    sig_list.append(f_any_3)


    f_all = funsig("sycl", "int", "all", ["vigeninteger"], template_arg_map=[0])
    sig_list.append(f_all)

    f_all_2 = funsig("sycl", "bool", "all", ["sigeninteger"], template_arg_map=[0])
    sig_list.append(f_all_2)

    f_all_3 = funsig("sycl", "bool", "all", ["migeninteger"], template_arg_map=[0])
    sig_list.append(f_all_3)


    f_bitselect = funsig("sycl", "gentype", "bitselect", ["gentype", "gentype", "gentype"], template_arg_map=[0,1,2])
    sig_list.append(f_bitselect)


    f_select = funsig("sycl", "vgentype", "select", ["vgentype", "vgentype", "vigeninteger"], "0", "", [], [["vgentype", "vigeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1,2])
    sig_list.append(f_select)

    f_select_2 = funsig("sycl", "vgentype", "select", ["vgentype", "vgentype", "vugeninteger"], "0", "", [], [["vgentype", "vugeninteger", "base_type_but_same_sizeof"]], template_arg_map=[0,1,2])
    sig_list.append(f_select_2)

    f_select_3 = funsig("sycl", "sgentype", "select", ["sgentype", "sgentype", "bool"], template_arg_map=[0])
    sig_list.append(f_select_3)

    f_select_4 = funsig("sycl", "mgentype", "select", ["mgentype", "mgentype", "mbooln"], "0", "", [], [["mgentype", "mbooln", "base_type"]], template_arg_map=[0,1,2])
    sig_list.append(f_select_4)


    return sig_list

def create_float_signatures():
    sig_list = []

    f_acos_1 = funsig("sycl", "sgenfloat", "acos", ["sgenfloat"], "4")
    sig_list.append(f_acos_1)

    f_acos_2 = funsig("sycl", "vgenfloat", "acos", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_acos_2)

    f_acos_3 = funsig("sycl", "mgenfloat", "acos", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_acos_3)


    f_acosh_1 = funsig("sycl", "sgenfloat", "acosh", ["sgenfloat"], "4")
    sig_list.append(f_acosh_1)

    f_acosh_2 = funsig("sycl", "vgenfloat", "acosh", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_acosh_2)

    f_acosh_3 = funsig("sycl", "mgenfloat", "acosh", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_acosh_3)


    f_acospi_1 = funsig("sycl", "sgenfloat", "acospi", ["sgenfloat"], "5")
    sig_list.append(f_acospi_1)

    f_acospi_2 = funsig("sycl", "vgenfloat", "acospi", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_acospi_2)

    f_acospi_3 = funsig("sycl", "mgenfloat", "acospi", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_acospi_3)


    f_asin_1 = funsig("sycl", "sgenfloat", "asin", ["sgenfloat"], "4")
    sig_list.append(f_asin_1)

    f_asin_2 = funsig("sycl", "vgenfloat", "asin", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_asin_2)

    f_asin_3 = funsig("sycl", "mgenfloat", "asin", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_asin_3)


    f_asinh_1 = funsig("sycl", "sgenfloat", "asinh", ["sgenfloat"], "4")
    sig_list.append(f_asinh_1)

    f_asinh_2 = funsig("sycl", "vgenfloat", "asinh", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_asinh_2)

    f_asinh_3 = funsig("sycl", "mgenfloat", "asinh", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_asinh_3)


    f_asinpi_1 = funsig("sycl", "sgenfloat", "asinpi", ["sgenfloat"], "5")
    sig_list.append(f_asinpi_1)

    f_asinpi_2 = funsig("sycl", "vgenfloat", "asinpi", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_asinpi_2)

    f_asinpi_3 = funsig("sycl", "mgenfloat", "asinpi", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_asinpi_3)


    f_atan_1 = funsig("sycl", "sgenfloat", "atan", ["sgenfloat"], "5")
    sig_list.append(f_atan_1)

    f_atan_2 = funsig("sycl", "vgenfloat", "atan", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_atan_2)

    f_atan_3 = funsig("sycl", "mgenfloat", "atan", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_atan_3)


    f_atan2_1 = funsig("sycl", "sgenfloat", "atan2", ["sgenfloat", "sgenfloat"], "6")
    sig_list.append(f_atan2_1)

    f_atan2_2 = funsig("sycl", "vgenfloat", "atan2", ["vgenfloat", "vgenfloat"], "6", template_arg_map=[0,1])
    sig_list.append(f_atan2_2)

    f_atan2_3 = funsig("sycl", "mgenfloat", "atan2", ["mgenfloat", "mgenfloat"], "6", template_arg_map=[0,1])
    sig_list.append(f_atan2_3)


    f_atanh_1 = funsig("sycl", "sgenfloat", "atanh", ["sgenfloat"], "5")
    sig_list.append(f_atanh_1)

    f_atanh_2 = funsig("sycl", "vgenfloat", "atanh", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_atanh_2)

    f_atanh_3 = funsig("sycl", "mgenfloat", "atanh", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_atanh_3)


    f_atanpi_1 = funsig("sycl", "sgenfloat", "atanpi", ["sgenfloat"], "5")
    sig_list.append(f_atanpi_1)

    f_atanpi_2 = funsig("sycl", "vgenfloat", "atanpi", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_atanpi_2)

    f_atanpi_3 = funsig("sycl", "mgenfloat", "atanpi", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_atanpi_3)


    f_atan2pi_1 = funsig("sycl", "sgenfloat", "atan2pi", ["sgenfloat", "sgenfloat"], "6")
    sig_list.append(f_atan2pi_1)

    f_atan2pi_2 = funsig("sycl", "vgenfloat", "atan2pi", ["vgenfloat", "vgenfloat"], "6", template_arg_map=[0,1])
    sig_list.append(f_atan2pi_2)

    f_atan2pi_3 = funsig("sycl", "mgenfloat", "atan2pi", ["mgenfloat", "mgenfloat"], "6", template_arg_map=[0,1])
    sig_list.append(f_atan2pi_3)


    f_cbrt_1 = funsig("sycl", "sgenfloat", "cbrt", ["sgenfloat"], "2")
    sig_list.append(f_cbrt_1)

    f_cbrt_2 = funsig("sycl", "vgenfloat", "cbrt", ["vgenfloat"], "2", template_arg_map=[0])
    sig_list.append(f_cbrt_2)

    f_cbrt_3 = funsig("sycl", "mgenfloat", "cbrt", ["mgenfloat"], "2", template_arg_map=[0])
    sig_list.append(f_cbrt_3)


    f_ceil_1 = funsig("sycl", "sgenfloat", "ceil", ["sgenfloat"], "0")
    sig_list.append(f_ceil_1)

    f_ceil_2 = funsig("sycl", "vgenfloat", "ceil", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_ceil_2)

    f_ceil_3 = funsig("sycl", "mgenfloat", "ceil", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_ceil_3)


    f_copysign_1 = funsig("sycl", "sgenfloat", "copysign", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_copysign_1)

    f_copysign_2 = funsig("sycl", "vgenfloat", "copysign", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_copysign_2)

    f_copysign_3 = funsig("sycl", "mgenfloat", "copysign", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_copysign_3)


    f_cos_1 = funsig("sycl", "sgenfloat", "cos", ["sgenfloat"], "4")
    sig_list.append(f_cos_1)

    f_cos_2 = funsig("sycl", "vgenfloat", "cos", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_cos_2)

    f_cos_3 = funsig("sycl", "mgenfloat", "cos", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_cos_3)


    f_cosh_1 = funsig("sycl", "sgenfloat", "cosh", ["sgenfloat"], "4")
    sig_list.append(f_cosh_1)

    f_cosh_2 = funsig("sycl", "vgenfloat", "cosh", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_cosh_2)

    f_cosh_3 = funsig("sycl", "mgenfloat", "cosh", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_cosh_3)


    f_cospi_1 = funsig("sycl", "sgenfloat", "cospi", ["sgenfloat"], "4")
    sig_list.append(f_cospi_1)

    f_cospi_2 = funsig("sycl", "vgenfloat", "cospi", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_cospi_2)

    f_cospi_3 = funsig("sycl", "mgenfloat", "cospi", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_cospi_3)


    f_erfc_1 = funsig("sycl", "sgenfloat", "erfc", ["sgenfloat"], "16")
    sig_list.append(f_erfc_1)

    f_erfc_2 = funsig("sycl", "vgenfloat", "erfc", ["vgenfloat"], "16", template_arg_map=[0])
    sig_list.append(f_erfc_2)

    f_erfc_3 = funsig("sycl", "mgenfloat", "erfc", ["mgenfloat"], "16", template_arg_map=[0])
    sig_list.append(f_erfc_3)


    f_erf_1 = funsig("sycl", "sgenfloat", "erf", ["sgenfloat"], "16")
    sig_list.append(f_erf_1)

    f_erf_2 = funsig("sycl", "vgenfloat", "erf", ["vgenfloat"], "16", template_arg_map=[0])
    sig_list.append(f_erf_2)

    f_erf_3 = funsig("sycl", "mgenfloat", "erf", ["mgenfloat"], "16", template_arg_map=[0])
    sig_list.append(f_erf_3)


    f_exp_1 = funsig("sycl", "sgenfloat", "exp", ["sgenfloat"], "3")
    sig_list.append(f_exp_1)

    f_exp_2 = funsig("sycl", "vgenfloat", "exp", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_exp_2)

    f_exp_3 = funsig("sycl", "mgenfloat", "exp", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_exp_3)


    f_exp2_1 = funsig("sycl", "sgenfloat", "exp2", ["sgenfloat"], "3")
    sig_list.append(f_exp2_1)

    f_exp2_2 = funsig("sycl", "vgenfloat", "exp2", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_exp2_2)

    f_exp2_3 = funsig("sycl", "mgenfloat", "exp2", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_exp2_3)


    f_exp10_1 = funsig("sycl", "sgenfloat", "exp10", ["sgenfloat"], "3")
    sig_list.append(f_exp10_1)

    f_exp10_2 = funsig("sycl", "vgenfloat", "exp10", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_exp10_2)

    f_exp10_3 = funsig("sycl", "mgenfloat", "exp10", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_exp10_3)


    f_expm1_1 = funsig("sycl", "sgenfloat", "expm1", ["sgenfloat"], "3")
    sig_list.append(f_expm1_1)

    f_expm1_2 = funsig("sycl", "vgenfloat", "expm1", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_expm1_2)

    f_expm1_3 = funsig("sycl", "mgenfloat", "expm1", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_expm1_3)


    f_fabs_1 = funsig("sycl", "sgenfloat", "fabs", ["sgenfloat"], "0")
    sig_list.append(f_fabs_1)

    f_fabs_2 = funsig("sycl", "vgenfloat", "fabs", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_fabs_2)

    f_fabs_3 = funsig("sycl", "mgenfloat", "fabs", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_fabs_3)


    f_fdim_1 = funsig("sycl", "sgenfloat", "fdim", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_fdim_1)

    f_fdim_2 = funsig("sycl", "vgenfloat", "fdim", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fdim_2)

    f_fdim_3 = funsig("sycl", "mgenfloat", "fdim", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fdim_3)


    f_floor_1 = funsig("sycl", "sgenfloat", "floor", ["sgenfloat"], "0")
    sig_list.append(f_floor_1)

    f_floor_2 = funsig("sycl", "vgenfloat", "floor", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_floor_2)

    f_floor_3 = funsig("sycl", "mgenfloat", "floor", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_floor_3)


    f_fma_1 = funsig("sycl", "sgenfloat", "fma", ["sgenfloat", "sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_fma_1)

    f_fma_2 = funsig("sycl", "vgenfloat", "fma", ["vgenfloat", "vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1,2])
    sig_list.append(f_fma_2)

    f_fma_3 = funsig("sycl", "mgenfloat", "fma", ["mgenfloat", "mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1,2])
    sig_list.append(f_fma_3)


    f_fmax_1 = funsig("sycl", "sgenfloat", "fmax", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_fmax_1)

    f_fmax_2 = funsig("sycl", "vgenfloat", "fmax", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fmax_2)

    f_fmax_3 = funsig("sycl", "mgenfloat", "fmax", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fmax_3)

    f_fmax_4 = funsig("sycl", "vgenfloat", "fmax", ["vgenfloat", "sgenfloat"], "0", "", [], [["vgenfloat", "sgenfloat", "dim"]], template_arg_map=[0])
    sig_list.append(f_fmax_4)

    f_fmax_5 = funsig("sycl", "mgenfloat", "fmax", ["mgenfloat", "sgenfloat"], "0", "", [], [["mgenfloat", "sgenfloat", "dim"]], template_arg_map=[0])
    sig_list.append(f_fmax_5)


    f_fmin_1 = funsig("sycl", "sgenfloat", "fmin", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_fmin_1)

    f_fmin_2 = funsig("sycl", "vgenfloat", "fmin", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fmin_2)

    f_fmin_3 = funsig("sycl", "mgenfloat", "fmin", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fmin_3)

    f_fmin_4 = funsig("sycl", "vgenfloat", "fmin", ["vgenfloat", "sgenfloat"], "0", "", [], [["vgenfloat", "sgenfloat", "dim"]], template_arg_map=[0])
    sig_list.append(f_fmin_4)

    f_fmin_5 = funsig("sycl", "mgenfloat", "fmin", ["mgenfloat", "sgenfloat"], "0", "", [], [["mgenfloat", "sgenfloat", "dim"]], template_arg_map=[0])
    sig_list.append(f_fmin_5)


    f_fmod_1 = funsig("sycl", "sgenfloat", "fmod", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_fmod_1)

    f_fmod_2 = funsig("sycl", "vgenfloat", "fmod", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fmod_2)

    f_fmod_3 = funsig("sycl", "mgenfloat", "fmod", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_fmod_3)


    f_fract_1 = funsig("sycl", "sgenfloat", "fract", ["sgenfloat", "sgenfloat"], "0", "", [2], template_arg_map=[1])
    sig_list.append(f_fract_1)

    f_fract_2 = funsig("sycl", "vgenfloat", "fract", ["vgenfloat", "vgenfloat"], "0", "", [2], template_arg_map=[0,1])
    sig_list.append(f_fract_2)

    f_fract_3 = funsig("sycl", "mgenfloat", "fract", ["mgenfloat", "mgenfloat"], "0", "", [2], template_arg_map=[0,1])
    sig_list.append(f_fract_3)


    f_frexp_1 = funsig("sycl", "sgenfloat", "frexp", ["sgenfloat", "genint"], "0", "", [2], [["sgenfloat", "genint", "base_type"]], template_arg_map=[1])
    sig_list.append(f_frexp_1)

    f_frexp_2 = funsig("sycl", "vgenfloat", "frexp", ["vgenfloat", "genint"], "0", "", [2], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_frexp_2)

    f_frexp_3 = funsig("sycl", "mgenfloat", "frexp", ["mgenfloat", "genint"], "0", "", [2], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_frexp_3)


    f_hypot_1 = funsig("sycl", "sgenfloat", "hypot", ["sgenfloat", "sgenfloat"], "4")
    sig_list.append(f_hypot_1)

    f_hypot_2 = funsig("sycl", "vgenfloat", "hypot", ["vgenfloat", "vgenfloat"], "4", template_arg_map=[0,1])
    sig_list.append(f_hypot_2)

    f_hypot_3 = funsig("sycl", "mgenfloat", "hypot", ["mgenfloat", "mgenfloat"], "4", template_arg_map=[0,1])
    sig_list.append(f_hypot_3)


    f_ilogb_1 = funsig("sycl", "genint", "ilogb", ["sgenfloat"], "0", "", [], [["sgenfloat", "genint", "base_type"]])
    sig_list.append(f_ilogb_1)

    f_ilogb_2 = funsig("sycl", "genint", "ilogb", ["vgenfloat"], "0", "", [], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0])
    sig_list.append(f_ilogb_2)

    f_ilogb_3 = funsig("sycl", "genint", "ilogb", ["mgenfloat"], "0", "", [], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0])
    sig_list.append(f_ilogb_3)


    f_ldexp_1 = funsig("sycl", "sgenfloat", "ldexp", ["sgenfloat", "genint"], "0", "", [], [["sgenfloat", "genint", "base_type"]])
    sig_list.append(f_ldexp_1)

    f_ldexp_2 = funsig("sycl", "vgenfloat", "ldexp", ["vgenfloat", "genint"], "0", "", [], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0])
    sig_list.append(f_ldexp_2)

    f_ldexp_3 = funsig("sycl", "mgenfloat", "ldexp", ["mgenfloat", "genint"], "0", "", [], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0])
    sig_list.append(f_ldexp_3)

    f_ldexp_4 = funsig("sycl", "vgenfloat", "ldexp", ["vgenfloat", "int"], "0", template_arg_map=[0])
    sig_list.append(f_ldexp_4)

    f_ldexp_5 = funsig("sycl", "mgenfloat", "ldexp", ["mgenfloat", "int"], "0", template_arg_map=[0])
    sig_list.append(f_ldexp_5)


    f_lgamma_1 = funsig("sycl", "sgenfloat", "lgamma", ["sgenfloat"], "-1")
    sig_list.append(f_lgamma_1)

    f_lgamma_2 = funsig("sycl", "vgenfloat", "lgamma", ["vgenfloat"], "-1", template_arg_map=[0])
    sig_list.append(f_lgamma_2)

    f_lgamma_3 = funsig("sycl", "mgenfloat", "lgamma", ["mgenfloat"], "-1", template_arg_map=[0])
    sig_list.append(f_lgamma_3)


    f_lgamma_r_1 = funsig("sycl", "sgenfloat", "lgamma_r", ["sgenfloat", "genint"], "-1", "", [2], [["sgenfloat", "genint", "base_type"]], template_arg_map=[1])
    sig_list.append(f_lgamma_r_1)

    f_lgamma_r_2 = funsig("sycl", "vgenfloat", "lgamma_r", ["vgenfloat", "genint"], "-1", "", [2], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_lgamma_r_2)

    f_lgamma_r_3 = funsig("sycl", "mgenfloat", "lgamma_r", ["mgenfloat", "genint"], "-1", "", [2], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_lgamma_r_3)


    f_log_1 = funsig("sycl", "sgenfloat", "log", ["sgenfloat"], "3")
    sig_list.append(f_log_1)

    f_log_2 = funsig("sycl", "vgenfloat", "log", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_log_2)

    f_log_3 = funsig("sycl", "mgenfloat", "log", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_log_3)


    f_log2_1 = funsig("sycl", "sgenfloat", "log2", ["sgenfloat"], "3")
    sig_list.append(f_log2_1)

    f_log2_2 = funsig("sycl", "vgenfloat", "log2", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_log2_2)

    f_log2_3 = funsig("sycl", "mgenfloat", "log2", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_log2_3)


    f_log10_1 = funsig("sycl", "sgenfloat", "log10", ["sgenfloat"], "3")
    sig_list.append(f_log10_1)

    f_log10_2 = funsig("sycl", "vgenfloat", "log10", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_log10_2)

    f_log10_3 = funsig("sycl", "mgenfloat", "log10", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_log10_3)


    f_log1p_1 = funsig("sycl", "sgenfloat", "log1p", ["sgenfloat"], "2")
    sig_list.append(f_log1p_1)

    f_log1p_2 = funsig("sycl", "vgenfloat", "log1p", ["vgenfloat"], "2", template_arg_map=[0])
    sig_list.append(f_log1p_2)

    f_log1p_3 = funsig("sycl", "mgenfloat", "log1p", ["mgenfloat"], "2", template_arg_map=[0])
    sig_list.append(f_log1p_3)


    f_logb_1 = funsig("sycl", "sgenfloat", "logb", ["sgenfloat"], "0")
    sig_list.append(f_logb_1)

    f_logb_2 = funsig("sycl", "vgenfloat", "logb", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_logb_2)

    f_logb_3 = funsig("sycl", "mgenfloat", "logb", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_logb_3)


    f_mad_1 = funsig("sycl", "sgenfloat", "mad", ["sgenfloat","sgenfloat","sgenfloat"], "-1")
    sig_list.append(f_mad_1)

    f_mad_2 = funsig("sycl", "vgenfloat", "mad", ["vgenfloat","vgenfloat","vgenfloat"], "-1", template_arg_map=[0,1,2])
    sig_list.append(f_mad_2)

    f_mad_3 = funsig("sycl", "mgenfloat", "mad", ["mgenfloat","mgenfloat","mgenfloat"], "-1", template_arg_map=[0,1,2])
    sig_list.append(f_mad_3)


    f_maxmag_1 = funsig("sycl", "sgenfloat", "maxmag", ["sgenfloat","sgenfloat"], "0")
    sig_list.append(f_maxmag_1)

    f_maxmag_2 = funsig("sycl", "vgenfloat", "maxmag", ["vgenfloat","vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_maxmag_2)

    f_maxmag_3 = funsig("sycl", "mgenfloat", "maxmag", ["mgenfloat","mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_maxmag_3)


    f_minmag_1 = funsig("sycl", "sgenfloat", "minmag", ["sgenfloat","sgenfloat"], "0")
    sig_list.append(f_minmag_1)

    f_minmag_2 = funsig("sycl", "vgenfloat", "minmag", ["vgenfloat","vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_minmag_2)

    f_minmag_3 = funsig("sycl", "mgenfloat", "minmag", ["mgenfloat","mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_minmag_3)


    f_modf_1 = funsig("sycl", "sgenfloat", "modf", ["sgenfloat", "sgenfloat"], "0", "", [2], template_arg_map=[1])
    sig_list.append(f_modf_1)

    f_modf_2 = funsig("sycl", "vgenfloat", "modf", ["vgenfloat", "vgenfloat"], "0", "", [2], template_arg_map=[0,1])
    sig_list.append(f_modf_2)

    f_modf_3 = funsig("sycl", "mgenfloat", "modf", ["mgenfloat", "mgenfloat"], "0", "", [2], template_arg_map=[0,1])
    sig_list.append(f_modf_3)


    f_nan_1 = funsig("sycl", "float", "nan", ["unsigned"])
    sig_list.append(f_nan_1)

    f_nan_2 = funsig("sycl", "vgenfloatf", "nan", ["ugenint"], "0", "", [], [["vgenfloatf", "ugenint", "base_type"]], template_arg_map=[0])
    sig_list.append(f_nan_2)

    f_nan_3 = funsig("sycl", "mgenfloatf", "nan", ["ugenint"], "0", "", [], [["mgenfloatf", "ugenint", "base_type"]], template_arg_map=[0])
    sig_list.append(f_nan_3)

    f_nan_4 = funsig("sycl", "double", "nan", ["unsigned long"], "0")
    sig_list.append(f_nan_4)

    f_nan_5 = funsig("sycl", "vgenfloatd", "nan", ["ugenlonginteger"], "0", "", [], [["vgenfloatd", "ugenlonginteger", "base_type"]], template_arg_map=[0])
    sig_list.append(f_nan_5)

    f_nan_6 = funsig("sycl", "mgenfloatd", "nan", ["ugenlonginteger"], "0", "", [], [["mgenfloatd", "ugenlonginteger", "base_type"]], template_arg_map=[0])
    sig_list.append(f_nan_6)

    f_nan_7 = funsig("sycl", "sycl::half", "nan", ["unsigned short"], "0")
    sig_list.append(f_nan_7)

    f_nan_8 = funsig("sycl", "vgenfloath", "nan", ["ugenshort"], "0", "", [], [["vgenfloath", "ugenshort", "base_type"]], template_arg_map=[0])
    sig_list.append(f_nan_8)

    f_nan_9 = funsig("sycl", "mgenfloath", "nan", ["ugenshort"], "0", "", [], [["mgenfloath", "ugenshort", "base_type"]], template_arg_map=[0])
    sig_list.append(f_nan_9)


    f_nextafter_1 = funsig("sycl", "sgenfloat", "nextafter", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_nextafter_1)

    f_nextafter_2 = funsig("sycl", "vgenfloat", "nextafter", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_nextafter_2)

    f_nextafter_3 = funsig("sycl", "mgenfloat", "nextafter", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_nextafter_3)


    f_pow_1 = funsig("sycl", "sgenfloat", "pow", ["sgenfloat", "sgenfloat"], "16")
    sig_list.append(f_pow_1)

    f_pow_2 = funsig("sycl", "vgenfloat", "pow", ["vgenfloat", "vgenfloat"], "16", template_arg_map=[0,1])
    sig_list.append(f_pow_2)

    f_pow_3 = funsig("sycl", "mgenfloat", "pow", ["mgenfloat", "mgenfloat"], "16", template_arg_map=[0,1])
    sig_list.append(f_pow_3)


    f_pown_1 = funsig("sycl", "sgenfloat", "pown", ["sgenfloat", "genint"], "16", "", [], [["sgenfloat", "genint", "base_type"]])
    sig_list.append(f_pown_1)

    f_pown_2 = funsig("sycl", "vgenfloat", "pown", ["vgenfloat", "genint"], "16", "", [], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_pown_2)

    f_pown_3 = funsig("sycl", "mgenfloat", "pown", ["mgenfloat", "genint"], "16", "", [], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_pown_3)


    f_powr_1 = funsig("sycl", "sgenfloat", "powr", ["sgenfloat", "sgenfloat"], "16")
    sig_list.append(f_powr_1)

    f_powr_2 = funsig("sycl", "vgenfloat", "powr", ["vgenfloat", "vgenfloat"], "16", template_arg_map=[0,1])
    sig_list.append(f_powr_2)

    f_powr_3 = funsig("sycl", "mgenfloat", "powr", ["mgenfloat", "mgenfloat"], "16", template_arg_map=[0,1])
    sig_list.append(f_powr_3)


    f_remainder_1 = funsig("sycl", "sgenfloat", "remainder", ["sgenfloat", "sgenfloat"], "0")
    sig_list.append(f_remainder_1)

    f_remainder_2 = funsig("sycl", "vgenfloat", "remainder", ["vgenfloat", "vgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_remainder_2)

    f_remainder_3 = funsig("sycl", "mgenfloat", "remainder", ["mgenfloat", "mgenfloat"], "0", template_arg_map=[0,1])
    sig_list.append(f_remainder_3)


    f_remquo_1 = funsig("sycl", "sgenfloat", "remquo", ["sgenfloat", "sgenfloat", "genint"], "0", "", [3], [["sgenfloat", "genint", "base_type"]], template_arg_map=[2])
    sig_list.append(f_remquo_1)

    f_remquo_2 = funsig("sycl", "vgenfloat", "remquo", ["vgenfloat", "vgenfloat", "genint"], "0", "", [3], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0,1,2])
    sig_list.append(f_remquo_2)

    f_remquo_3 = funsig("sycl", "mgenfloat", "remquo", ["mgenfloat", "mgenfloat", "genint"], "0", "", [3], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0,1,2])
    sig_list.append(f_remquo_3)


    f_rint_1 = funsig("sycl", "sgenfloat", "rint", ["sgenfloat"], "0")
    sig_list.append(f_rint_1)

    f_rint_2 = funsig("sycl", "vgenfloat", "rint", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_rint_2)

    f_rint_3 = funsig("sycl", "mgenfloat", "rint", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_rint_3)


    f_rootn_1 = funsig("sycl", "sgenfloat", "rootn", ["sgenfloat", "genint"], "16", "", [], [["sgenfloat", "genint", "base_type"]])
    sig_list.append(f_rootn_1)

    f_rootn_2 = funsig("sycl", "vgenfloat", "rootn", ["vgenfloat", "genint"], "16", "", [], [["vgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_rootn_2)

    f_rootn_3 = funsig("sycl", "mgenfloat", "rootn", ["mgenfloat", "genint"], "16", "", [], [["mgenfloat", "genint", "base_type"]], template_arg_map=[0,1])
    sig_list.append(f_rootn_3)


    f_round_1 = funsig("sycl", "sgenfloat", "round", ["sgenfloat"], "0")
    sig_list.append(f_round_1)

    f_round_2 = funsig("sycl", "vgenfloat", "round", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_round_2)

    f_round_3 = funsig("sycl", "mgenfloat", "round", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_round_3)


    f_rsqrt_1 = funsig("sycl", "sgenfloat", "rsqrt", ["sgenfloat"], "2")
    sig_list.append(f_rsqrt_1)

    f_rsqrt_2 = funsig("sycl", "vgenfloat", "rsqrt", ["vgenfloat"], "2", template_arg_map=[0])
    sig_list.append(f_rsqrt_2)

    f_rsqrt_3 = funsig("sycl", "mgenfloat", "rsqrt", ["mgenfloat"], "2", template_arg_map=[0])
    sig_list.append(f_rsqrt_3)


    f_sin_1 = funsig("sycl", "sgenfloat", "sin", ["sgenfloat"], "4")
    sig_list.append(f_sin_1)

    f_sin_2 = funsig("sycl", "vgenfloat", "sin", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_sin_2)

    f_sin_3 = funsig("sycl", "mgenfloat", "sin", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_sin_3)


    f_sincos_1 = funsig("sycl", "sgenfloat", "sincos", ["sgenfloat", "sgenfloat"], "4", "",[2], template_arg_map=[1])
    sig_list.append(f_sincos_1)

    f_sincos_2 = funsig("sycl", "vgenfloat", "sincos", ["vgenfloat", "vgenfloat"], "4", "",[2], template_arg_map=[0,1])
    sig_list.append(f_sincos_2)

    f_sincos_3 = funsig("sycl", "mgenfloat", "sincos", ["mgenfloat", "mgenfloat"], "4", "",[2], template_arg_map=[0,1])
    sig_list.append(f_sincos_3)


    f_sinh_1 = funsig("sycl", "sgenfloat", "sinh", ["sgenfloat"], "4")
    sig_list.append(f_sinh_1)

    f_sinh_2 = funsig("sycl", "vgenfloat", "sinh", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_sinh_2)

    f_sinh_3 = funsig("sycl", "mgenfloat", "sinh", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_sinh_3)


    f_sinpi_1 = funsig("sycl", "sgenfloat", "sinpi", ["sgenfloat"], "4")
    sig_list.append(f_sinpi_1)

    f_sinpi_2 = funsig("sycl", "vgenfloat", "sinpi", ["vgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_sinpi_2)

    f_sinpi_3 = funsig("sycl", "mgenfloat", "sinpi", ["mgenfloat"], "4", template_arg_map=[0])
    sig_list.append(f_sinpi_3)


    f_sqrt_1 = funsig("sycl", "sgenfloat", "sqrt", ["sgenfloat"], "3")
    sig_list.append(f_sqrt_1)

    f_sqrt_2 = funsig("sycl", "vgenfloat", "sqrt", ["vgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_sqrt_2)

    f_sqrt_3 = funsig("sycl", "mgenfloat", "sqrt", ["mgenfloat"], "3", template_arg_map=[0])
    sig_list.append(f_sqrt_3)


    f_tan_1 = funsig("sycl", "sgenfloat", "tan", ["sgenfloat"], "5")
    sig_list.append(f_tan_1)

    f_tan_2 = funsig("sycl", "vgenfloat", "tan", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_tan_2)

    f_tan_3 = funsig("sycl", "mgenfloat", "tan", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_tan_3)


    f_tanh_1 = funsig("sycl", "sgenfloat", "tanh", ["sgenfloat"], "5")
    sig_list.append(f_tanh_1)

    f_tanh_2 = funsig("sycl", "vgenfloat", "tanh", ["vgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_tanh_2)

    f_tanh_3 = funsig("sycl", "mgenfloat", "tanh", ["mgenfloat"], "5", template_arg_map=[0])
    sig_list.append(f_tanh_3)


    f_tanpi_1 = funsig("sycl", "sgenfloat", "tanpi", ["sgenfloat"], "6")
    sig_list.append(f_tanpi_1)

    f_tanpi_2 = funsig("sycl", "vgenfloat", "tanpi", ["vgenfloat"], "6", template_arg_map=[0])
    sig_list.append(f_tanpi_2)

    f_tanpi_3 = funsig("sycl", "mgenfloat", "tanpi", ["mgenfloat"], "6", template_arg_map=[0])
    sig_list.append(f_tanpi_3)


    f_tgamma_1 = funsig("sycl", "sgenfloat", "tgamma", ["sgenfloat"], "16")
    sig_list.append(f_tgamma_1)

    f_tgamma_2 = funsig("sycl", "vgenfloat", "tgamma", ["vgenfloat"], "16", template_arg_map=[0])
    sig_list.append(f_tgamma_2)

    f_tgamma_3 = funsig("sycl", "mgenfloat", "tgamma", ["mgenfloat"], "16", template_arg_map=[0])
    sig_list.append(f_tgamma_3)


    f_trunc_1 = funsig("sycl", "sgenfloat", "trunc", ["sgenfloat"], "0")
    sig_list.append(f_trunc_1)

    f_trunc_2 = funsig("sycl", "vgenfloat", "trunc", ["vgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_trunc_2)

    f_trunc_3 = funsig("sycl", "mgenfloat", "trunc", ["mgenfloat"], "0", template_arg_map=[0])
    sig_list.append(f_trunc_3)

    return sig_list

def create_native_signatures():
    sig_list = []

    f_cos_1 = funsig("sycl::native", "float", "cos", ["float"], "-1")
    sig_list.append(f_cos_1)

    f_cos_2 = funsig("sycl::native", "vgenfloatf", "cos", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_cos_2)

    f_cos_3 = funsig("sycl::native", "mgenfloatf", "cos", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_cos_3)


    f_divide_1 = funsig("sycl::native", "float", "divide", ["float", "float"], "-1")
    sig_list.append(f_divide_1)

    f_divide_2 = funsig("sycl::native", "vgenfloatf", "divide", ["vgenfloatf", "vgenfloatf"], "-1", template_arg_map=[0,1])
    sig_list.append(f_divide_2)

    f_divide_3 = funsig("sycl::native", "mgenfloatf", "divide", ["mgenfloatf", "mgenfloatf"], "-1", template_arg_map=[0,1])
    sig_list.append(f_divide_3)


    f_exp_1 = funsig("sycl::native", "float", "exp", ["float"], "-1")
    sig_list.append(f_exp_1)

    f_exp_2 = funsig("sycl::native", "vgenfloatf", "exp", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_exp_2)

    f_exp_3 = funsig("sycl::native", "mgenfloatf", "exp", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_exp_3)


    f_exp2_1 = funsig("sycl::native", "float", "exp2", ["float"], "-1")
    sig_list.append(f_exp2_1)

    f_exp2_2 = funsig("sycl::native", "vgenfloatf", "exp2", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_exp2_2)

    f_exp2_3 = funsig("sycl::native", "mgenfloatf", "exp2", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_exp2_3)


    f_exp10_1 = funsig("sycl::native", "float", "exp10", ["float"], "-1")
    sig_list.append(f_exp10_1)

    f_exp10_2 = funsig("sycl::native", "vgenfloatf", "exp10", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_exp10_2)

    f_exp10_3 = funsig("sycl::native", "mgenfloatf", "exp10", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_exp10_3)


    f_log_1 = funsig("sycl::native", "float", "log", ["float"], "-1")
    sig_list.append(f_log_1)

    f_log_2 = funsig("sycl::native", "vgenfloatf", "log", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_log_2)

    f_log_3 = funsig("sycl::native", "mgenfloatf", "log", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_log_3)


    f_log2_1 = funsig("sycl::native", "float", "log2", ["float"], "-1")
    sig_list.append(f_log2_1)

    f_log2_2 = funsig("sycl::native", "vgenfloatf", "log2", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_log2_2)

    f_log2_3 = funsig("sycl::native", "mgenfloatf", "log2", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_log2_3)


    f_log10_1 = funsig("sycl::native", "float", "log10", ["float"], "-1")
    sig_list.append(f_log10_1)

    f_log10_2 = funsig("sycl::native", "vgenfloatf", "log10", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_log10_2)

    f_log10_3 = funsig("sycl::native", "mgenfloatf", "log10", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_log10_3)


    f_powr_1 = funsig("sycl::native", "float", "powr", ["float", "float"], "-1")
    sig_list.append(f_powr_1)

    f_powr_2 = funsig("sycl::native", "vgenfloatf", "powr", ["vgenfloatf", "vgenfloatf"], "-1", template_arg_map=[0,1])
    sig_list.append(f_powr_2)

    f_powr_3 = funsig("sycl::native", "mgenfloatf", "powr", ["mgenfloatf", "mgenfloatf"], "-1", template_arg_map=[0,1])
    sig_list.append(f_powr_3)


    f_recip_1 = funsig("sycl::native", "float", "recip", ["float"], "-1")
    sig_list.append(f_recip_1)

    f_recip_2 = funsig("sycl::native", "vgenfloatf", "recip", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_recip_2)

    f_recip_3 = funsig("sycl::native", "mgenfloatf", "recip", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_recip_3)


    f_rsqrt_1 = funsig("sycl::native", "float", "rsqrt", ["float"], "-1")
    sig_list.append(f_rsqrt_1)

    f_rsqrt_2 = funsig("sycl::native", "vgenfloatf", "rsqrt", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_rsqrt_2)

    f_rsqrt_3 = funsig("sycl::native", "mgenfloatf", "rsqrt", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_rsqrt_3)


    f_sin_1 = funsig("sycl::native", "float", "sin", ["float"], "-1")
    sig_list.append(f_sin_1)

    f_sin_2 = funsig("sycl::native", "vgenfloatf", "sin", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_sin_2)

    f_sin_3 = funsig("sycl::native", "mgenfloatf", "sin", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_sin_3)


    f_sqrt_1 = funsig("sycl::native", "float", "sqrt", ["float"], "-1")
    sig_list.append(f_sqrt_1)

    f_sqrt_2 = funsig("sycl::native", "vgenfloatf", "sqrt", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_sqrt_2)

    f_sqrt_3 = funsig("sycl::native", "mgenfloatf", "sqrt", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_sqrt_3)


    f_tan_1 = funsig("sycl::native", "float", "tan", ["float"], "-1")
    sig_list.append(f_tan_1)

    f_tan_2 = funsig("sycl::native", "vgenfloatf", "tan", ["vgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_tan_2)

    f_tan_3 = funsig("sycl::native", "mgenfloatf", "tan", ["mgenfloatf"], "-1", template_arg_map=[0])
    sig_list.append(f_tan_3)

    return sig_list

def create_half_signatures():
    sig_list = []

    f_cos_1 = funsig("sycl::half_precision", "float", "cos", ["float"], "8192")
    sig_list.append(f_cos_1)

    f_cos_2 = funsig("sycl::half_precision", "vgenfloatf", "cos", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_cos_2)

    f_cos_3 = funsig("sycl::half_precision", "mgenfloatf", "cos", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_cos_3)


    f_divide_1 = funsig("sycl::half_precision", "float", "divide", ["float", "float"], "8192")
    sig_list.append(f_divide_1)

    f_divide_2 = funsig("sycl::half_precision", "vgenfloatf", "divide", ["vgenfloatf", "vgenfloatf"], "8192", template_arg_map=[0,1])
    sig_list.append(f_divide_2)

    f_divide_3 = funsig("sycl::half_precision", "mgenfloatf", "divide", ["mgenfloatf", "mgenfloatf"], "8192", template_arg_map=[0,1])
    sig_list.append(f_divide_3)


    f_exp_1 = funsig("sycl::half_precision", "float", "exp", ["float"], "8192")
    sig_list.append(f_exp_1)

    f_exp_2 = funsig("sycl::half_precision", "vgenfloatf", "exp", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_exp_2)

    f_exp_3 = funsig("sycl::half_precision", "mgenfloatf", "exp", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_exp_3)


    f_exp2_1 = funsig("sycl::half_precision", "float", "exp2", ["float"], "8192")
    sig_list.append(f_exp2_1)

    f_exp2_2 = funsig("sycl::half_precision", "vgenfloatf", "exp2", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_exp2_2)

    f_exp2_3 = funsig("sycl::half_precision", "mgenfloatf", "exp2", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_exp2_3)


    f_exp10_1 = funsig("sycl::half_precision", "float", "exp10", ["float"], "8192")
    sig_list.append(f_exp10_1)

    f_exp10_2 = funsig("sycl::half_precision", "vgenfloatf", "exp10", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_exp10_2)

    f_exp10_3 = funsig("sycl::half_precision", "mgenfloatf", "exp10", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_exp10_3)


    f_log_1 = funsig("sycl::half_precision", "float", "log", ["float"], "8192")
    sig_list.append(f_log_1)

    f_log_2 = funsig("sycl::half_precision", "vgenfloatf", "log", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_log_2)

    f_log_3 = funsig("sycl::half_precision", "mgenfloatf", "log", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_log_3)


    f_log2_1 = funsig("sycl::half_precision", "float", "log2", ["float"], "8192")
    sig_list.append(f_log2_1)

    f_log2_2 = funsig("sycl::half_precision", "vgenfloatf", "log2", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_log2_2)

    f_log2_3 = funsig("sycl::half_precision", "mgenfloatf", "log2", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_log2_3)


    f_log10_1 = funsig("sycl::half_precision", "float", "log10", ["float"], "8192")
    sig_list.append(f_log10_1)

    f_log10_2 = funsig("sycl::half_precision", "vgenfloatf", "log10", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_log10_2)

    f_log10_3 = funsig("sycl::half_precision", "mgenfloatf", "log10", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_log10_3)


    f_powr_1 = funsig("sycl::half_precision", "float", "powr", ["float", "float"], "8192")
    sig_list.append(f_powr_1)

    f_powr_2 = funsig("sycl::half_precision", "vgenfloatf", "powr", ["vgenfloatf", "vgenfloatf"], "8192", template_arg_map=[0,1])
    sig_list.append(f_powr_2)

    f_powr_3 = funsig("sycl::half_precision", "mgenfloatf", "powr", ["mgenfloatf", "mgenfloatf"], "8192", template_arg_map=[0,1])
    sig_list.append(f_powr_3)


    f_recip_1 = funsig("sycl::half_precision", "float", "recip", ["float"], "8192")
    sig_list.append(f_recip_1)

    f_recip_2 = funsig("sycl::half_precision", "vgenfloatf", "recip", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_recip_2)

    f_recip_3 = funsig("sycl::half_precision", "mgenfloatf", "recip", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_recip_3)


    f_rsqrt_1 = funsig("sycl::half_precision", "float", "rsqrt", ["float"], "8192")
    sig_list.append(f_rsqrt_1)

    f_rsqrt_2 = funsig("sycl::half_precision", "vgenfloatf", "rsqrt", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_rsqrt_2)

    f_rsqrt_3 = funsig("sycl::half_precision", "mgenfloatf", "rsqrt", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_rsqrt_3)


    f_sin_1 = funsig("sycl::half_precision", "float", "sin", ["float"], "8192")
    sig_list.append(f_sin_1)

    f_sin_2 = funsig("sycl::half_precision", "vgenfloatf", "sin", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_sin_2)

    f_sin_3 = funsig("sycl::half_precision", "mgenfloatf", "sin", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_sin_3)


    f_sqrt_1 = funsig("sycl::half_precision", "float", "sqrt", ["float"], "8192")
    sig_list.append(f_sqrt_1)

    f_sqrt_2 = funsig("sycl::half_precision", "vgenfloatf", "sqrt", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_sqrt_2)

    f_sqrt_3 = funsig("sycl::half_precision", "mgenfloatf", "sqrt", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_sqrt_3)


    f_tan_1 = funsig("sycl::half_precision", "float", "tan", ["float"], "8192")
    sig_list.append(f_tan_1)

    f_tan_2 = funsig("sycl::half_precision", "vgenfloatf", "tan", ["vgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_tan_2)

    f_tan_3 = funsig("sycl::half_precision", "mgenfloatf", "tan", ["mgenfloatf"], "8192", template_arg_map=[0])
    sig_list.append(f_tan_3)

    return sig_list
