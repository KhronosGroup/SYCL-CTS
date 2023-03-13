from . import sycl_functions
from . import sycl_types
import random
from string import Template
import re
import itertools

test_case_templates = { "private" : ("""
{
  test_function<$TEST_ID, $RETURN_TYPE>(
      [=]{
        $FUNCTION_CALL
      });
}
"""),

                    "local" : ("""
{
  $DECL
  test_function_multi_ptr_local<$TEST_ID, $RETURN_TYPE>(
      [=]($ACCESSOR acc){
        $FUNCTION_CALL"
      }, $DATA);
}
"""),

                    "global" : ("""
{
  $DECL
  test_function_multi_ptr_global<$TEST_ID, $RETURN_TYPE>(
      [=]($ACCESSOR acc){
        $FUNCTION_CALL
      }, $DATA);
}
""")
}

test_case_templates_check = {
    "no_ptr" : ("""
{
  $REFERENCE
  check_function<$TEST_ID, $RETURN_TYPE>(log,
      [=]{
        $FUNCTION_CALL
      }, ref$ACCURACY$COMMENT);
}
"""),

    "private" : ("""
{
  $PTR_REF
  check_function_multi_ptr_private<$TEST_ID, $RETURN_TYPE>(log,
      [=]{
        $FUNCTION_PRIVATE_CALL
      }, ref, refPtr$ACCURACY$COMMENT);
}
"""),

    "local" : ("""
{
  $PTR_REF
  check_function_multi_ptr_local<$TEST_ID, $RETURN_TYPE>(log,
      [=]($ACCESSOR acc){
        $FUNCTION_CALL
      }, $DATA, ref, refPtr$ACCURACY$COMMENT);
}
"""),

    "global" : ("""
{
  $PTR_REF
  check_function_multi_ptr_global<$TEST_ID, $RETURN_TYPE>(log,
      [=]($ACCESSOR acc){
        $FUNCTION_CALL
      }, $DATA, ref, refPtr$ACCURACY$COMMENT);
}
""")
}

def generate_value(base_type, dim):
    val = ""
    for i in range(dim):
        if base_type == "bool":
            val += "true,"
        if base_type == "float" or base_type == "double" or base_type == "sycl::half":
            # 10 digits of precision for floats, doubles and half.
            val += str(round(random.uniform(0.1, 0.9), 10))
            if base_type == "double":
                val += ","
            else:
                val += "f,"
        # random 8 bit integer
        if base_type == "char":
            val += str(random.randint(0, 127)) + ","
        if base_type == "signed char" or base_type == "int8_t":
            val += str(random.randint(-128, 127)) + ","
        if base_type == "unsigned char" or base_type == "uint8_t":
            val += str(random.randint(0, 255)) + ","
        # random 16 bit integer
        if base_type == "int" or base_type == "short" or base_type == "int16_t":
            val += str(random.randint(-32768, 32767)) + ","
        if base_type == "unsigned" or base_type == "unsigned short" or base_type == "uint16_t":
            val += str(random.randint(0, 65535)) + ","
        # random 32 bit integer
        if base_type == "long" or base_type == "int32_t":
            val += str(random.randint(-2147483648, 2147483647)) + ","
        if base_type == "unsigned long" or base_type == "uint32_t":
            val += str(random.randint(0, 4294967295)) + "U" + ","
        # random 64 bit integer
        if base_type == "long long" or base_type == "int64_t":
            val += str(random.randint(-9223372036854775808, 9223372036854775807)) + "LL" + ","
        if base_type == "unsigned long long" or base_type == "uint64_t":
            val += str(random.randint(0, 18446744073709551615)) + "LLU" + ","
    return val[:-1]

def generate_multi_ptr(var_name, var_type, memory):
    decl = ""
    if memory == "global":
        source_name = "multiPtrSourceData"
        decl = var_type.name + " " + source_name + "(" + generate_value(var_type.base_type, var_type.dim) + ");\n"
        decl += "sycl::multi_ptr<" + var_type.name + ", sycl::access::address_space::global_space> " + var_name + "(acc);\n"
    if memory == "local":
        source_name = "multiPtrSourceData"
        decl = var_type.name + " " + source_name + "(" + generate_value(var_type.base_type, var_type.dim) + ");\n"
        decl += "sycl::multi_ptr<" + var_type.name + ", sycl::access::address_space::local_space> " + var_name + "(acc);\n"
    if memory == "private":
        source_name = "multiPtrSourceData"
        decl = var_type.name + " " + source_name + "(" + generate_value(var_type.base_type, var_type.dim) + ");\n"
        decl += "sycl::multi_ptr<" + var_type.name + ", sycl::access::address_space::private_space> " + var_name + "(&" + source_name + ");\n"
    return decl

def generate_variable(var_name, var_type, var_index):
    return var_type.name + " " + var_name + "(" + generate_value(var_type.base_type, var_type.dim) + ");\n"

def generate_arguments(sig, memory):
    arg_src = ""
    arg_names = []
    arg_index = 0
    for arg in sig.arg_types:
        # Create argument name.
        arg_name = "inputData_" + str(arg_index)
        arg_names.append(arg_name)

        # Identify whether argument is a pointer.
        is_pointer = False
        # Value 0 in pntr_indx is reserved for the return type.
        if (arg_index + 1) in sig.pntr_indx:
            is_pointer = True

        current_arg = ""
        if is_pointer:
            current_arg = generate_multi_ptr(arg_name, arg, memory)
        else:
            current_arg = generate_variable(arg_name, arg, arg_index)

        arg_src += current_arg + "        "
        arg_index += 1
    return (arg_names, arg_src)

function_call_template = Template("""
        ${arg_src}
        static_assert(std::is_same_v<decltype(${namespace}::${func_name}(${arg_names})), ${ret_type}>,
            "Error: Wrong return type of ${namespace}::${func_name}(${arg_types}), not ${ret_type}");
        return ${namespace}::${func_name}(${arg_names});
""")
def generate_function_call(sig, arg_names, arg_src):
    fc = function_call_template.substitute(
        arg_src=arg_src,
        namespace=sig.namespace,
        func_name=sig.name,
        arg_names=", ".join(arg_names),
        ret_type=sig.ret_type.name,
        arg_types=", ".join([a.name for a in sig.arg_types]))
    return fc

function_private_call_template = Template("""
        ${arg_src}
        ${ret_type} res = ${namespace}::${func_name}(${arg_names});
        return privatePtrCheck<${ret_type}, ${arg_type}>(res, multiPtrSourceData);
""")
def generate_function_private_call(sig, arg_names, arg_src, types):
    fc = function_private_call_template.substitute(
        arg_src=arg_src,
        namespace=sig.namespace,
        func_name=sig.name,
        arg_names=",".join(arg_names),
        ret_type=sig.ret_type.name,
        arg_type=sig.arg_types[-1].name)
    return fc

reference_template = Template("""
        ${arg_src}
        sycl_cts::resultRef<${ret_type}> ref = reference::${func_name}(${arg_names});
""")
def generate_reference(sig, arg_names, arg_src):
    fc = reference_template.substitute(
        arg_src=arg_src,
        func_name=sig.name,
        ret_type=sig.ret_type.name,
        arg_names=",".join(arg_names))
    return fc

reference_ptr_template = Template("""
        ${arg_src}
        ${arg_type} refPtr = multiPtrSourceData;
        sycl_cts::resultRef<${ret_type}> ref = reference::${func_name}(${arg_names}, &refPtr);
""")
def generate_reference_ptr(types, sig, arg_names, arg_src):
    fc = reference_ptr_template.substitute(
        arg_src=re.sub(r'^sycl::multi_ptr.*\n?', '', arg_src, flags=re.MULTILINE),
        func_name=sig.name,
        ret_type=sig.ret_type.name,
        arg_names=",".join(arg_names[:-1]),
        arg_type=sig.arg_types[-1].name)
    return fc

def generate_test_case(test_id, types, sig, memory, check):
    testCaseSource = test_case_templates_check[memory] if check else test_case_templates[memory]
    testCaseId = str(test_id)
    (arg_names, arg_src) = generate_arguments(sig, memory)
    testCaseSource = testCaseSource.replace("$REFERENCE", generate_reference(sig, arg_names, arg_src))
    testCaseSource = testCaseSource.replace("$PTR_REF", generate_reference_ptr(types, sig, arg_names, arg_src))
    testCaseSource = testCaseSource.replace("$TEST_ID", testCaseId)
    testCaseSource = testCaseSource.replace("$FUNCTION_PRIVATE_CALL", generate_function_private_call(sig, arg_names, arg_src, types))
    testCaseSource = testCaseSource.replace("$RETURN_TYPE", sig.ret_type.name)
    if sig.accuracy:##If the signature contains an accuracy value
        accuracy = sig.accuracy
        # if accuracy depends on vecSize
        if "vecSize" in accuracy:
            vecSize = str(sig.arg_types[0].dim)
            accuracy = accuracy.replace("vecSize", vecSize)
        testCaseSource = testCaseSource.replace("$ACCURACY", ", " + accuracy)
    else:
        testCaseSource = testCaseSource.replace("$ACCURACY", "")
    if sig.comment:##If the signature contains comment for accuracy
        testCaseSource = testCaseSource.replace("$COMMENT", ', "' + sig.comment +'"')
    else:
        testCaseSource = testCaseSource.replace("$COMMENT", "")

    if memory != "private" and memory !="no_ptr":
        # We rely on the fact that all SYCL math builtins have at most one arguments as pointer.
        pointerType = sig.arg_types[sig.pntr_indx[0] - 1]
        sourcePtrDataName = "multiPtrSourceData"
        sourcePtrData =  generate_variable(sourcePtrDataName, pointerType, 0)
        testCaseSource = testCaseSource.replace("$DECL", sourcePtrData)
        testCaseSource = testCaseSource.replace("$DATA", sourcePtrDataName)
        accessorType = ""
        if memory == "local":
            accessorType = "sycl::accessor<" + pointerType.name + ", 1, sycl::access_mode::read_write, sycl::target::local>"
        if memory == "global":
            accessorType = "sycl::accessor<" + pointerType.name + ", 1, sycl::access_mode::read_write, sycl::target::device>"
        testCaseSource = testCaseSource.replace("$ACCESSOR", accessorType)
    testCaseSource = testCaseSource.replace("$FUNCTION_CALL", generate_function_call(sig, arg_names, arg_src))
    return testCaseSource

def generate_test_cases(test_id, types, sig_list, check):
    random.seed(0)
    test_source = ""
    for sig in sig_list:
        if sig.pntr_indx:#If the signature contains a pointer argument.
            test_source += generate_test_case(test_id, types, sig, "private", check)
            test_id += 1
            test_source += generate_test_case(test_id, types, sig, "local", check)
            test_id += 1
            test_source += generate_test_case(test_id, types, sig, "global", check)
            test_id += 1
        else:
            if check:
                test_source += generate_test_case(test_id, types, sig, "no_ptr", check)
                test_id += 1
            else:
                test_source += generate_test_case(test_id, types, sig, "private", check)
                test_id += 1
    return test_source

# Lists of the types with equal sizes
chars = ["char", "signed char", "unsigned char"]
shorts = ["short", "unsigned short"]
ints = ["int", "unsigned"]
longs = ["long", "unsigned long"]
longlongs = ["long long", "unsigned long long"]
bit8s = ["int8_t", "uint8_t"]
bit16s = ["int16_t", "uint16_t", "sycl::half"]
bit32s = ["int32_t", "uint32_t", "float"]
bit64s = ["int64_t", "uint64_t", "double"]

# Checks if type pair cannot be connected by a mutation 
def bad_mutation(type1, type2, mutation):
    if mutation == "dim":
        return not (type1.base_type == type2.base_type)
    if mutation == "base_type":
        return not ((type1.var_type == type2.var_type) and (type1.dim == type2.dim))
    if mutation == "base_type_but_same_sizeof":
        return not ((type1.var_type == type2.var_type) and (type1.dim == type2.dim) and
                    ((type1.base_type in chars     and type2.base_type in chars)     or
                     (type1.base_type in shorts    and type2.base_type in shorts)    or
                     (type1.base_type in ints      and type2.base_type in ints)      or
                     (type1.base_type in longs     and type2.base_type in longs)     or
                     (type1.base_type in longlongs and type2.base_type in longlongs) or
                     (type1.base_type in bit8s     and type2.base_type in bit8s)     or
                     (type1.base_type in bit16s    and type2.base_type in bit16s)    or
                     (type1.base_type in bit32s    and type2.base_type in bit32s)    or
                     (type1.base_type in bit64s    and type2.base_type in bit64s)))
    print("Unknown mutation: " + mutation)
    return True


def expand_signature(types, signature):
    """
    Produces all possible overloads of a function signature.
    We produce the dict of typelists matched_typelists where each line (with the same index in typelists)
    contains possible set of individual types for function arguments and return type.
    """
    print("signature: " + str(signature.ret_type) + " " + signature.name + " " + str(signature.arg_types))

    exp_sig = []

    # We construct dict of all typelists in the function
    arg_list = signature.arg_types.copy()
    arg_list.append(signature.ret_type)

    used_typelists = {}
    for arg in arg_list:
        if len(types[arg]) != 1:
            used_typelists[arg] = list(types[arg].keys())

    # We match types from used_typelists and individual types by lines in matched_typelists
    matched_typelists = {}
    common_size = 0

    if len(used_typelists) <= 1:
        # Return value and all arguments that are lists are of the same type
        common_size = 1
        for typelist in used_typelists.values():
            common_size = len(typelist)

        for arg in arg_list:
            # All fixed args are changed to lists of the common size
            if len(types[arg]) == 1:
                matched_typelists[arg] = [list(types[arg])[0] for i in range(common_size)]
            else:
                matched_typelists[arg] = list(types[arg])
    else:
        # Some typelists are different, use matching rules
        typelist = []

        # Making Cartesian product of all the types
        for element in itertools.product(*list(used_typelists.values())):
            # and filter it out by matching rules
            match = True
            for mutation in signature.mutations:
                base_index = list(used_typelists.keys()).index(mutation[0])
                derived_index = list(used_typelists.keys()).index(mutation[1])

                if bad_mutation(element[base_index], element[derived_index], mutation[2]):
                    # argument or return types do not match with function limitations
                    match = False
                    break

            if match:
                typelist.append(list(element))

        common_size = len(typelist)
        if common_size > 0:
            for arg in arg_list:
                # All fixed args are changed to lists of the common size.
                if len(types[arg]) == 1:
                    matched_typelists[arg] = [list(types[arg])[0] for i in range(common_size)]
                else:
                    index = list(used_typelists.keys()).index(arg)
                    matched_typelists[arg] = list([typelist[i][index] for i in range(common_size)])
        else:
            print("No matching for " + signature.name + " " + str(signature.arg_types) + " => " + str(signature.ret_type))

    # Construct function signatures
    for i in range(common_size):
        new_sig = sycl_functions.funsig(signature.namespace, matched_typelists[signature.ret_type][i], 
                                        signature.name, [matched_typelists[signature.arg_types[j]][i] 
                                                         for j in range(len(signature.arg_types))],
                                        signature.accuracy, signature.comment, signature.pntr_indx[:])
        exp_sig.append(new_sig)

    return exp_sig

def get_unique_signatures(signatures):
    uniq_sig = []

    for sig in signatures:
        if sig not in uniq_sig:
            uniq_sig.append(sig)

    return uniq_sig

def expand_signatures(types, signatures):
    ex_sig_list = []

    for sig in signatures:
        ex_sig_list.extend(expand_signature(types, sig))

    return get_unique_signatures(ex_sig_list)

# Expands a generic type (e.g. floatn) to the collection of its basic types.
# Uses recursion.

def expand_type(run, types, current):
    # If this is a basic type, stop.
    if types[current].dim > 0:
        # To skip non-supported types by the lists in run
        if (types[current].var_type in run.var_types) and (types[current].base_type in run.base_types) and (types[current].dim in run.dimensions):
            return {types[current] : types[current]}
        else:
            return {}

    base_types = {}
    for ct in types[current].child_types:
        base_types.update(expand_type(run, types, ct))

    return base_types

def expand_types(run, types):
    ex_types = {}

    for tp in types:
        ex_types[tp] = expand_type(run, types, tp)

    return ex_types
