from . import sycl_functions
from . import sycl_types
import random
from string import Template
import re

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

def generate_value(base_type, dim, unsigned):
    val = ""
    for i in range(dim):
        if base_type == "float" or base_type == "double" or base_type == "sycl::half":
            # 10 digits of precision for floats, doubles and half.
            val += str(round(random.uniform(0.1, 0.9), 10)) + ","
        # random 8 bit integer
        if base_type == "char" or base_type == "int8_t":
            if unsigned:
                val += str(random.randint(0, 255)) + ","
            else:
                val += str(random.randint(-128, 127)) + ","
        # random 16 bit integer
        if base_type == "int" or base_type == "short" or base_type == "int16_t":
            if unsigned:
                val += str(random.randint(0, 65535)) + ","
            else:
                val += str(random.randint(-32768, 32767)) + ","
        # random 32 bit integer
        if base_type == "long int" or base_type == "int32_t":
            if unsigned:
                val += str(random.randint(0, 4294967295)) + "U" + ","
            else:
                val += str(random.randint(-2147483648, 2147483647)) + ","
        # random 64 bit integer
        if base_type == "long long int" or base_type == "int64_t":
            if unsigned:
                val += str(random.randint(0, 18446744073709551615)) + "LLU" + ","
            else:
                val += str(random.randint(-9223372036854775808,
                                          9223372036854775807)) + "LL" + ","
    return val[:-1]

def generate_multi_ptr(var_name, var_type, memory):
    decl = ""
    if memory == "global":
        decl = "sycl::multi_ptr<" + var_type.name + ", sycl::access::address_space::global_space> " + var_name + "(acc);\n"
    if memory == "local":
        decl = "sycl::multi_ptr<" + var_type.name + ", sycl::access::address_space::local_space> " + var_name + "(acc);\n"
    if memory == "private":
        source_name = "multiPtrSourceData"
        decl = var_type.name + " " + source_name + "(" + generate_value(var_type.base_type, var_type.dim, var_type.unsigned) + ");\n"
        decl += "sycl::multi_ptr<" + var_type.name + ", sycl::access::address_space::private_space> " + var_name + "(&" + source_name + ");\n"
    return decl

def generate_variable(var_name, var_type, var_index):
    return var_type.name + " " + var_name + "(" + generate_value(var_type.base_type, var_type.dim, var_type.unsigned) + ");\n"

def extract_type(type_dict):
    # At this point, it is guaranteed that type_dict is a dictionary with one entry.
    for bt in list(type_dict.keys()):
        return type_dict[bt]

def generate_arguments(types, sig, memory):
    arg_src = ""
    arg_names = []
    arg_index = 0
    for arg in sig.arg_types:
        # Get argument type.
        arg_type = extract_type(types[arg])

        # Create argument name.
        arg_name = "inputData_" + str(arg_index)
        arg_names.append(arg_name)

        # Identify whether aegument is a pointer.
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
        ret_type=sig.ret_type,
        arg_type=extract_type(types[sig.arg_types[-1]]).name)
    return fc

reference_template = Template("""
        ${arg_src}
        sycl_cts::resultRef<${ret_type}> ref = reference::${func_name}(${arg_names});
""")
def generate_reference(sig, arg_names, arg_src):
    fc = reference_template.substitute(
        arg_src=arg_src,
        func_name=sig.name,
        ret_type=sig.ret_type,
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
        ret_type=sig.ret_type,
        arg_names=",".join(arg_names[:-1]),
        arg_type=extract_type(types[sig.arg_types[-1]]).name)
    return fc

def generate_test_case(test_id, types, sig, memory, check):
    testCaseSource = test_case_templates_check[memory] if check else test_case_templates[memory]
    testCaseId = str(test_id)
    (arg_names, arg_src) = generate_arguments(types, sig, memory)
    testCaseSource = testCaseSource.replace("$REFERENCE", generate_reference(sig, arg_names, arg_src))
    testCaseSource = testCaseSource.replace("$PTR_REF", generate_reference_ptr(types, sig, arg_names, arg_src))
    testCaseSource = testCaseSource.replace("$TEST_ID", testCaseId)
    testCaseSource = testCaseSource.replace("$FUNCTION_PRIVATE_CALL", generate_function_private_call(sig, arg_names, arg_src, types))
    testCaseSource = testCaseSource.replace("$RETURN_TYPE", sig.ret_type)
    if sig.accuracy:##If the signature contains an accuracy value
        accuracy = sig.accuracy
        # if accuracy depends on vecSize
        if "vecSize" in accuracy:
            vecSize = str(extract_type(types[sig.arg_types[0]]).dim)
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
        sourcePtrData =  generate_variable(sourcePtrDataName, extract_type(types[pointerType]), 0)
        testCaseSource = testCaseSource.replace("$DECL", sourcePtrData)
        testCaseSource = testCaseSource.replace("$DATA", sourcePtrDataName)
        accessorType = ""
        if memory == "local":
            accessorType = "sycl::accessor<" + pointerType + ", 1, sycl::access::mode::read_write, sycl::access::target::local>"
        if memory == "global":
            accessorType = "sycl::accessor<" + pointerType + ", 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>"
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

# Given the current combination of:
# -- variable type(scalar/vector)
# -- base type(e.g. float or int)
# -- dimension(1,2,3,4,8,16)
# -- unsigned flag(e.g. intn vs uintn)
# We attempt to find a combination that is part of the current generic type (e.g. floatn)

def attempt_match(runner, var_type, base_type, dim, unsigned, current_type):
    ct_part_keys = []
    for t in current_type.keys():
        # Matching by all keys except name
        ct_part_keys.append(t[:-1])
    # Change sign.
    if (var_type, base_type, dim, not unsigned) in ct_part_keys:
        return (var_type, base_type, dim, not unsigned)
    # Change type and sign, same dimensionality.
    for new_sign in [unsigned, not unsigned]:
        for new_type in runner.base_types:
            if (var_type, new_type, dim, new_sign) in ct_part_keys:
                return (var_type, new_type, dim, new_sign)
    # Change to base type (for scalars).
    for new_sign in [unsigned, not unsigned]:
        if ("scalar", base_type, 1, new_sign) in ct_part_keys:
            return ("scalar", base_type, 1, new_sign)
    # See if any other scalar type match.
    for new_sign in [unsigned, not unsigned]:
        for new_type in runner.base_types:
            if ("scalar", new_type, 1, new_sign) in ct_part_keys:
                return ("scalar", new_type, 1, new_sign)
    return None

# Produces all possible overloads of a function signature.

def expand_signature(runner, types, signature):
    current_types = [types[signature.ret_type]]

    # to control cases when some arg types are base type of ret type
    sgeninteger = False
    exp_sig = []
    # to control cases when arg types are base type of ret type
    sgeninteger = False
    for arg in signature.arg_types:
        current_types.extend([types[arg]])
        if arg == "sgeninteger":
            sgeninteger = True

    name_key_index = 4

    # Iterate over all basic types
    # Try to match all function types (return type and argument types).
    all_types = sycl_types.create_basic_types()
    for name in all_types.keys():
        match = False
        nomatch = []
        index = 0
        for ct in current_types:
            ct_keys = ct.keys()
            name_keys = []
            for t in ct_keys:
                # Matching by name
                name_keys.append(t[name_key_index])
            if name in name_keys:
                match = True
            else:
                nomatch.append(index)
            index += 1
        if match:
            # Return value and all arguments are of the same type.
            if len(nomatch) == 0:
                new_sig = sycl_functions.funsig(signature.namespace, name, signature.name, [
                                                name for i in range(len(signature.arg_types))], signature.accuracy,
                                                signature.comment, signature.pntr_indx[:])
                exp_sig.append(new_sig)
            else:
                function_types = []
                function_types_extra = []
                all_matched = True
                extra = False
                for ct in current_types:
                    ct_keys = ct.keys()
                    name_keys = []
                    for t in ct_keys:
                        name_keys.append(t[name_key_index])
                    # Current function type already matches - no need for mutation.
                    if name in name_keys:
                        function_types.append(name)
                        function_types_extra.append(name)
                    else:
                        # Return value and all arguments are of the same base type.
                        # Types are selected by scalar/vector, size, dim and unsigned/signed
                        # but these parameters are the same for signed char and char.
                        # It works well when we need to select corresponding signed type for unsigned char
                        # but in cases of functions such as clamp where we have parameters like (vec<T,N>, T..)
                        # we shouldn't test it with paramenters (scharN, char..) or (charN, signed char) and we need special case for it
                        if sgeninteger:
                            part_keys = ('scalar', all_types[name].base_type, 1, all_types[name].unsigned)
                            arg_types = [ct[key] for key in ct.keys() if key[:-1] == part_keys]
                            if len(arg_types) > 0:
                                if len(arg_types) > 1 and 'schar' in name:
                                    function_types.append(arg_types[1].name)
                                else:
                                    function_types.append(arg_types[0].name)
                                continue
                        var_type = all_types[name].var_type
                        base_type = all_types[name].base_type
                        dim = all_types[name].dim
                        unsigned = all_types[name].unsigned

                        # Get var_type, base_type, dim, unsigned of suitable type
                        mutation = attempt_match(runner, var_type, base_type, dim, unsigned, ct)
                        if mutation:
                            # Find suitable type by all key members except name
                            new_types = [ct[key] for key in ct.keys() if key[:-1] == mutation]
                            function_types.append(new_types[0].name)
                            if len(new_types) > 1:
                                function_types_extra.append(new_types[1].name)
                                extra = True
                        else:
                            all_matched = False
                if all_matched:
                    new_sig = sycl_functions.funsig(
                        signature.namespace, function_types[0], signature.name, function_types[1:], signature.accuracy,
                        signature.comment, signature.pntr_indx[:])
                    exp_sig.append(new_sig)
                    if extra:
                        new_sig = sycl_functions.funsig(
                        signature.namespace, function_types_extra[0], signature.name, function_types_extra[1:],
                        signature.accuracy, signature.comment, signature.pntr_indx[:])
                        exp_sig.append(new_sig)
                else:
                    print("[WARNING] Unable to fully match function " + signature.name + " for: " +
                            var_type + ", " + base_type + ", " + str(dim) + ", " + str(unsigned))
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
        return {(types[current].var_type, types[current].base_type, types[current].dim, types[current].unsigned, types[current].name) : types[current]}

    base_types = {}
    for ct in types[current].child_types:
        base_types.update(expand_type(types, ct))

    return base_types

def expand_types(types):
    ex_types = {}

    for tp in types:
        ex_types[tp] = expand_type(types, tp)

    return ex_types
