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
  $DECL
  $PTR_REF
  check_function_multi_ptr_local<$TEST_ID, $RETURN_TYPE>(log,
      [=]($ACCESSOR acc){
        $FUNCTION_CALL
      }, $DATA, ref, refPtr$ACCURACY$COMMENT);
}
"""),

    "global" : ("""
{
  $DECL
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
    #print(base_type, dim)
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


def get_base_type(var_type):
    return var_type[1]

def get_dim(var_type):
    return var_type[2]

def get_name(var_type):
    return var_type[3]


def generate_multi_ptr(var_name, var_type, memory):
    decl = ""
    if memory == "global":
        decl = "sycl::multi_ptr<" + get_name(var_type) + ", sycl::access::address_space::global_space> " + var_name + "(acc);\n"
    if memory == "local":
        decl = "sycl::multi_ptr<" + get_name(var_type) + ", sycl::access::address_space::local_space> " + var_name + "(acc);\n"
    if memory == "private":
        source_name = "multiPtrSourceData"
        decl = get_name(var_type) + " " + source_name + "(" + generate_value(get_base_type(var_type), get_dim(var_type)) + ");\n"
        decl += "sycl::multi_ptr<" + get_name(var_type) + ", sycl::access::address_space::private_space> " + var_name + "(&" + source_name + ");\n"
    return decl

def generate_variable(var_name, var_type, var_index):
    #print(var_name, str(var_type))
    return get_name(var_type) + " " + var_name + "(" + generate_value(get_base_type(var_type), get_dim(var_type)) + ");\n"

def extract_type(type_dict):
    # At this point, it is guaranteed that type_dict is a dictionary with one entry.
    for bt in list(type_dict.keys()):
        return type_dict[bt]

def generate_arguments(sig, memory):
    arg_src = ""
    arg_names = []
    arg_index = 0
    for arg in sig.arg_types:
        # Get argument type.
        arg_type = arg
        #arg_type = extract_type(types[arg])
        #print(str(arg_type))

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
            current_arg = generate_multi_ptr(arg_name, arg_type, memory)
        else:
            current_arg = generate_variable(arg_name, arg_type, arg_index)

        arg_src += current_arg
        arg_index += 1
    return (arg_names, arg_src)

function_call_template = Template("""
        ${arg_src}
        return ${namespace}::${func_name}(${arg_names});
""")
def generate_function_call(sig, arg_names, arg_src):
    fc = function_call_template.substitute(
        arg_src=arg_src,
        namespace=sig.namespace,
        func_name=sig.name,
        arg_names=",".join(arg_names))
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
        ret_type=get_name(sig.ret_type),
        arg_type=get_name(sig.arg_types[-1]))
    return fc

reference_template = Template("""
        ${arg_src}
        sycl_cts::resultRef<${ret_type}> ref = reference::${func_name}(${arg_names});
""")
def generate_reference(sig, arg_names, arg_src):
    fc = reference_template.substitute(
        arg_src=arg_src,
        func_name=sig.name,
        ret_type=get_name(sig.ret_type),
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
        ret_type=get_name(sig.ret_type),
        arg_names=",".join(arg_names[:-1]),
        arg_type=get_name(sig.arg_types[-1]))
    return fc

def generate_test_case(test_id, types, sig, memory, check):
    #print("sig: " + str(sig.ret_type) + " " + sig.name + " " + str(sig.arg_types))
    testCaseSource = test_case_templates_check[memory] if check else test_case_templates[memory]
    testCaseId = str(test_id)
    (arg_names, arg_src) = generate_arguments(sig, memory)
    testCaseSource = testCaseSource.replace("$REFERENCE", generate_reference(sig, arg_names, arg_src))
    #print(str(testCaseSource))
    testCaseSource = testCaseSource.replace("$PTR_REF", generate_reference_ptr(types, sig, arg_names, arg_src))
    testCaseSource = testCaseSource.replace("$TEST_ID", testCaseId)
    testCaseSource = testCaseSource.replace("$FUNCTION_PRIVATE_CALL", generate_function_private_call(sig, arg_names, arg_src, types))
    testCaseSource = testCaseSource.replace("$RETURN_TYPE", get_name(sig.ret_type))
    if sig.accuracy:##If the signature contains an accuracy value
        accuracy = sig.accuracy
        # if accuracy depends on vecSize
        if "vecSize" in accuracy:
            vecSize = str(get_dim(sig.arg_types[0]))
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
            accessorType = "sycl::accessor<" + get_name(pointerType) + ", 1, sycl::access_mode::read_write, sycl::target::local>"
        if memory == "global":
            accessorType = "sycl::accessor<" + get_name(pointerType) + ", 1, sycl::access_mode::read_write, sycl::target::device>"
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
chars = itertools.product(["char", "signed char", "unsigned char"], ["char", "signed char", "unsigned char"])
shorts = itertools.product(["short", "unsigned short"], ["short", "unsigned short"])
ints = itertools.product(["int", "unsigned"], ["int", "unsigned"])
longs = itertools.product(["long", "unsigned long"], ["long", "unsigned long"])
longlongs = itertools.product(["long long", "unsigned long long"], ["long long", "unsigned long long"])
bit8s = itertools.product(["int8_t", "uint8_t"], ["int8_t", "uint8_t"])
bit16s = itertools.product(["int16_t", "uint16_t", "sycl::half"], ["int16_t", "uint16_t", "sycl::half"])
bit32s = itertools.product(["int32_t", "uint32_t", "float"], ["int32_t", "uint32_t", "float"])
bit64s = itertools.product(["int64_t", "uint64_t", "double"], ["int64_t", "uint64_t", "double"])

# Checks if type pair cannot be connected by a mutation 
def bad_mutation(type1, type2, mutation):
    if mutation == "dim":
        return not (get_base_type(type1) == get_base_type(type2))
    if mutation == "base_type":
        return not ((type1[0] == type2[0]) and (get_dim(type1) == get_dim(type2)))
    if mutation == "base_type_but_same_sizeof":
        return not ((type1[0] == type2[0]) and (get_dim(type1) == get_dim(type2)) and 
                    (([get_base_type(type1), get_base_type(type2)] in chars) or
                     ([get_base_type(type1), get_base_type(type2)] in shorts) or
                     ([get_base_type(type1), get_base_type(type2)] in ints) or
                     ([get_base_type(type1), get_base_type(type2)] in longs) or
                     ([get_base_type(type1), get_base_type(type2)] in longlongs) or
                     ([get_base_type(type1), get_base_type(type2)] in bit8s) or
                     ([get_base_type(type1), get_base_type(type2)] in bit16s) or
                     ([get_base_type(type1), get_base_type(type2)] in bit32s) or
                     ([get_base_type(type1), get_base_type(type2)] in bit64s)))
    print("Unknown mutation: " + mutation)
    return True


# Produces all possible overloads of a function signature.

def expand_signature(runner, types, signature):
    print("signature: " + str(signature.ret_type) + " " + signature.name + " " + str(signature.arg_types))

    exp_sig = []

    # we construct dict of all types/typelists in the function
    # and then will match them by lines in matched_typelists
    used_typelists = {}
    if len(types[signature.ret_type]) > 1:
        used_typelists[signature.ret_type] = list(types[signature.ret_type].keys())
    for arg in signature.arg_types:
        if len(types[arg]) > 1:
            used_typelists[arg] = list(types[arg].keys())

    #print(str(types[signature.ret_type]))
    #print("ut: " + str(used_typelists))

    arg_list = signature.arg_types.copy()
    arg_list.append(signature.ret_type)
    matched_typelists = {}
    common_size = 0

    # Return value and all arguments that are lists are of the same type.
    if len(used_typelists) <= 1:
        common_size = 1
        for typelist in used_typelists.values():
            common_size = len(typelist)

        for arg in arg_list:
            #print(str(list(types[arg])))
            # All fixed args are changed to lists of the common size.
            if len(types[arg]) == 1:
                matched_typelists[arg] = [list(types[arg])[0] for i in range(common_size)]
            else:
                matched_typelists[arg] = list(types[arg])
        #print("mt: " + str(matched_typelists))
    # Some typelists are different, use matching rules
    else:
        typelist = []

        # Making Cartesian product of all the types
        #print(str(list(used_typelists.values())))
        #print(str(itertools.product(*list(used_typelists.values()))))
        for element in itertools.product(*list(used_typelists.values())):
            # and filter it out by matching rules
            match = True
            for mutation in signature.mutations:
                #print(str(mutation))
                base_index = list(used_typelists.keys()).index(mutation[0])
                derived_index = list(used_typelists.keys()).index(mutation[1])
                #print(str(element))

                if bad_mutation(element[base_index], element[derived_index], mutation[2]):
                    match = False
                    break

            if match:
                typelist.append(list(element))
                #print(str(list(element)))

        common_size = len(typelist)
        #print(common_size)
        if common_size > 0:
            for arg in arg_list:
                # All fixed args are changed to lists of the common size.
                if len(types[arg]) == 1:
                    matched_typelists[arg] = [list(types[arg])[0] for i in range(common_size)]
                else:
                    index = list(used_typelists.keys()).index(arg)
                    matched_typelists[arg] = list([typelist[i][index] for i in range(common_size)])

    #print(signature.name + " " + str(arg_list) + " " + str(common_size))
    #print(str(matched_typelists))
    #print(str(matched_typelists[signature.ret_type]))
    #print(str(matched_typelists[signature.arg_types[0]]))
    # Construct function signatures
    for i in range(common_size):
        #print("ret " + str(matched_typelists[signature.ret_type][i]))
        #print(str(signature.arg_types))
        #print(str([matched_typelists[signature.arg_types[j]][i] for j in range(len(signature.arg_types))]))
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

def expand_signatures(runner, types, signatures):
    ex_sig_list = []

    for sig in signatures:
        ex_sig_list.extend(expand_signature(runner, types, sig))

    return get_unique_signatures(ex_sig_list)

# Expands a generic type (e.g. floatn) to the collection of its basic types.
# Uses recursion.

def expand_type(types, current):
    # If this is a basic type, stop.
    if types[current].dim > 0:
        # Name should be in the key too or we'll lose char or signed char - they both have the same var_type, base_type, dim, unsigned
        return {(types[current].var_type, types[current].base_type, types[current].dim, types[current].name) : types[current]}

    base_types = {}
    for ct in types[current].child_types:
        base_types.update(expand_type(types, ct))

    return base_types

def expand_types(types):
    ex_types = {}

    for tp in types:
        ex_types[tp] = expand_type(types, tp)

    return ex_types
