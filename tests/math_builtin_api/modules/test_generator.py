import sycl_functions
import random

test_case_template = ("test_function<$TEST_ID, $RETURN_TYPE>(\n"
                      "[=](){\n"
                      "$FUNCTION_CALL\n"
                      "});\n\n")


def generate_value(base_type, dim, unsigned):
    val = ""
    for i in range(dim):
        if base_type == "float" or base_type == "double" or base_type == "cl::sycl::half":
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
                val += str(random.randint(0, 4294967295)) + ","
            else:
                val += str(random.randint(-2147483648, 2147483647)) + ","
        # random 64 bit integer
        if base_type == "long long int" or base_type == "int64_t":
            if unsigned:
                val += str(random.randint(0, 18446744073709551615)) + ","
            else:
                val += str(random.randint(-9223372036854775808,
                                          9223372036854775807)) + ","
    return val[:-1]


def generate_argument(arg_name, arg, is_pointer):
    arg_decl = ""
    arg_del = ""
    # At this point, it is guaranteed that arg is a dictionary with one entry.
    for rr in arg.keys():
        if is_pointer is False:
            arg_decl = arg[rr].name + " " + arg_name + "(" + generate_value(
                arg[rr].base_type, arg[rr].dim, arg[rr].unsigned) + ");\n"
        else:
            arg_decl = arg[rr].name + " * " + arg_name + " = new " + arg[rr].name + \
                "(" + generate_value(arg[rr].base_type,
                                     arg[rr].dim, arg[rr].unsigned) + ");\n"
            arg_del = "delete " + arg_name + ";\n"
    return (arg_decl, arg_del)


def generate_arguments(types, sig):
    arg_src = ""
    arg_src_deletes = ""
    arg_names = []
    arg_index = 0
    for arg in sig.arg_types:
        arg_name = "inputData_" + str(arg_index)
        arg_names.append(arg_name)
        is_pointer = False
        # Value 0 in pntr_indx is reserved for the return type.
        if (arg_index + 1) in sig.pntr_indx:
            is_pointer = True
        (current_arg, current_arg_del) = generate_argument(
            arg_name, types[arg], is_pointer)
        arg_src += current_arg
        arg_src_deletes += current_arg_del
        arg_index += 1
    return (arg_names, arg_src, arg_src_deletes)


def generate_function_call(types, sig):
    (arg_names, arg_src, arg_src_deletes) = generate_arguments(types, sig)
    fc = arg_src
    fc += "return " + sig.namespace + "::" + sig.name + "("
    for arg_n in arg_names:
        fc += arg_n + ","
    fc = fc[:-1] + ");\n"
    fc += arg_src_deletes
    return fc


def generate_test_case(test_id, types, sig):
    testCaseSource = test_case_template
    testCaseId = str(test_id)
    testCaseSource = testCaseSource.replace("$TEST_ID", testCaseId)
    testCaseSource = testCaseSource.replace("$RETURN_TYPE", sig.ret_type)
    testCaseSource = testCaseSource.replace(
        "$FUNCTION_CALL", generate_function_call(types, sig))
    return testCaseSource


def generate_test_cases(types, sig_list):
    random.seed(0)
    test_source = ""
    test_id = 0
    for sig in sig_list:
        test_source += generate_test_case(test_id, types, sig)
        test_id += 1
    return test_source

# Given the current combination of:
# -- variable type(scalar/vector)
# -- base type(e.g. float or int)
# -- dimension(1,2,3,4,8,16)
# -- unsigned flag(e.g. intn vs uintn)
# We attempt to find a combination that is part of the current generic type (e.g. floatn)


def attempt_match(runner, var_type, base_type, dim, unsigned, current_type):
    # Change sign.
    if (var_type, base_type, dim, not unsigned) in current_type.keys():
        return (var_type, base_type, dim, not unsigned)
    # Change type and sign, same dimensionality.
    for new_sign in [unsigned, not unsigned]:
        for new_type in runner.base_types:
            if (var_type, new_type, dim, new_sign) in current_type.keys():
                return (var_type, new_type, dim, new_sign)
    # Change to base type (for scalars).
    for new_sign in [unsigned, not unsigned]:
        if ("scalar", base_type, 1, new_sign) in current_type.keys():
            return ("scalar", base_type, 1, new_sign)
    # See if any other scalar type match.
    for new_sign in [unsigned, not unsigned]:
        for new_type in runner.base_types:
            if ("scalar", new_type, 1, new_sign) in current_type.keys():
                return ("scalar", new_type, 1, new_sign)
    return None

# Produces all possible overloads of a function signature.


def expand_signature(runner, types, signature):
    current_types = [types[signature.ret_type]]

    exp_sig = []
    for arg in signature.arg_types:
        current_types.extend([types[arg]])

    # Itterate over all possible combinations of namespace, base type, dimension and sign flag.
    # Try to match all function types (return type and argument types).
    for var_type in runner.var_types:
        for base_type in runner.base_types:
            for dim in runner.dimensions:
                for unsigned in runner.unsigned:
                    match = False
                    nomatch = []
                    index = 0
                    for ct in current_types:
                        if (var_type, base_type, dim, unsigned) in ct.keys():
                            match = True
                        else:
                            nomatch.append(index)
                        index += 1
                    if match is True:
                        # Return value and all arguments are of the same type.
                        if len(nomatch) == 0:
                            common_type = current_types[0][(
                                var_type, base_type, dim, unsigned)]
                            new_sig = sycl_functions.funsig(signature.namespace, common_type.name, signature.name, [
                                                            common_type.name for i in range(len(signature.arg_types))], signature.pntr_indx[:])
                            exp_sig.append(new_sig)
                        else:
                            function_types = []
                            all_matched = True
                            for ct in current_types:
                                # Current function type already matches - no need for mutation.
                                if (var_type, base_type, dim, unsigned) in ct.keys():
                                    function_types.append(
                                        ct[(var_type, base_type, dim, unsigned)].name)
                                else:
                                    mutation = attempt_match(
                                        runner, var_type, base_type, dim, unsigned, ct)
                                    if mutation is not None:
                                        function_types.append(
                                            ct[mutation].name)
                                    else:
                                        all_matched = False
                            if all_matched is True:
                                new_sig = sycl_functions.funsig(
                                    signature.namespace, function_types[0], signature.name, function_types[1:], signature.pntr_indx[:])
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
        return {(types[current].var_type, types[current].base_type, types[current].dim, types[current].unsigned): types[current]}

    base_types = {}
    for ct in types[current].child_types:
        base_types.update(expand_type(types, ct))

    return base_types


def expand_types(types):
    ex_types = {}

    for tp in types:
        ex_types[tp] = expand_type(types, tp)

    return ex_types
