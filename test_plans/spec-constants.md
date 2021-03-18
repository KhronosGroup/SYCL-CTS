# Test plan for specialization constants

This is a test plan for the APIs described in SYCL 2020 section 4.9.5.
"Specialization constants".

## Testing scope

### Device coverage

All of the tests described below are performed only on the default device that
is selected on the CTS command line.

### Specialization constant types

All of tests described below are performed using each of the following
types as the underlying type of a `specialization_id` variable.

* `char`
* `signed char`
* `unsigned char`
* `short int`
* `unsigned short int`
* `int`
* `unsigned int`
* `long int`
* `unsigned long int`
* `long long int`
* `unsigned long long int`
* `float`
* `bool`
* `std::byte`
* `std::int8_t`
* `std::int16_t`
* `std::int32_t`
* `std::int64_t`
* `std::uint8_t`
* `std::uint16_t`
* `std::uint32_t`
* `std::uint64_t`
* `std::size_t`
* `vec<T, int dim>` where `T` is each of the scalar types listed above except
   for `bool` and `dim` is `1`, `2`, `3`, `4`, `8`, and `16`.
* `marray<T, size_t dim>` where `T` is each of the scalar types listed above
  and `dim` is `2`, `5`, and `10`.
* A user-defined struct with several scalar fields.

In addition, if the device has `aspect::fp64`, the following types are tested:

* `double`
* `vec<double, int dim>` where `dim` is `1`, `2`, `3`, `4`, `8`, and `16`.
* `marray<double, size_t dim>` where `dim` is `2`, `5`, and `10`.

In addition, if the device has `aspect::fp16`, the following types are tested:

* `sycl::half`
* `vec<sycl::half, int dim>` where `dim` is `1`, `2`, `3`, `4`, `8`, and `16`.
* `marray<sycl::half, size_t dim>` where `dim` is `2`, `5`, and `10`.


## Tests

### Basic tests

All of the following basic tests have these initial steps:

* Declare a `specialization_id` variable in the global namespace for the
  tested type.  The variable's default value has some non-zero value.
* Create a `queue` from the tested device and call `queue::submit()`.

#### Read a spec constant from a handler without writing its value

* Call `handler::get_specialization_constant()` and make sure we get the
  default value.
* No kernel is submitted.

#### Write and read a spec constant from a handler

* Set the value of the spec constant via
  `handler::set_specialization_constant()`.
* Call `handler::get_specialization_constant()` and make sure we get the same
  value back.
* No kernel is submitted.

#### Write the value twice from a handler and make sure we read the second value

* Set the value of the spec constant via
  `handler::set_specialization_constant()`.
* Set the value again to a different value.
* Call `handler::get_specialization_constant()` and make sure we get the second
  value back.
* No kernel is submitted.

#### Read a spec constant from a kernel without writing its value

* Submit a kernel via `handler::single_task()`.
* Read the value of the spec constant via
  `kernel_handler::get_specialization_constant()` and make sure we get the
   default value back.

#### Write the value from a handler and read it from a kernel

* Set the value of the spec constant via
  `handler::set_specialization_constant()`.
* Submit a kernel via `handler::single_task()`.
* Read the value of the spec constant via
  `kernel_handler::get_specialization_constant()` and make sure we get the same
   value back.

#### Write the value twice from a handler and read it from a kernel

* Set the value of the spec constant via
  `handler::set_specialization_constant()`.
* Set the value again to a different value.
* Submit a kernel via `handler::single_task()`.
* Read the value of the spec constant via
  `kernel_handler::get_specialization_constant()` and make sure we get the second
   value back.

### Multiple spec constants

* Declare several `specialization_id` variables of the tested type in the
  global namespace.  All the default values are different and none are zero.
* Create a `queue` from the tested device and call `queue::submit()`.
* Set the values of some of the spec constants via
  `handler::set_specialization_constant()` but do not set the values for all of
  them.
* Submit a kernel via `handler::single_task()`.
* Read the values of the spec constant via
  `kernel_handler::get_specialization_constant()` and make sure we get the
  expected value from each (either the value we set or the default value).

### Two command groups that read the same spec constant, both set value

* Declare a `specialization_id` variable in the global namespace for the
  tested type.  The variable's default value has some non-zero value.
* Declare a single object whose type is a class with `operator(handler &)`.
  This object will be the command group handler for our test.
* Create a `queue` from the tested device and call `queue::submit()` twice.
  Each call passes the same command group handler object described above.
* Each command group handler sets the value of the spec constant to a
  different value.
* Each command group handler submits a kernel via `handler::single_task()`.
* Each kernel reads the spec constant via
  `kernel_handler::get_specialization_constant()`.
* Verify that the value read in the kernel is the same as the value set in
  the command group handler which launched that kernel instance.
* The two kernels should run in parallel for this test.

### Two command groups that read the same spec constant, only one sets value

Same test as above except only one of the command group handlers sets the value
of the spec constant.  The kernel instance that is launched from the handler
that does not set a value should read the spec constant's default value.

### Spec constant defined in various ways

Do the following test for a `specialization_id` variable defined in the
following ways:

* Defined in a non-global namespace.
* Defined in the global namespace as `const`.
* Defined in the global namespace as `constexpr`.
* Defined in the global namespace as `inline`.
* Defined in the global namespace as `inline const`.
* Defined in the global namespace as `inline constexpr`.
* A static member variable of a struct in the global namespace.
* A static member variable of a struct in a non-global namespace.
* A static member variable declared `const` of a struct in the global namespace.
* A static member variable declared `constexpr` of a struct in the global
  namespace.
* A static member variable declared `inline` of a struct in the global
  namespace.
* A static member variable declared `inline const` of a struct in the global
  namespace.
* A static member variable declared `inline constexpr` of a struct in the global
  namespace.

The test that is performed is:

* Create a `queue` from the tested device and call `queue::submit()`.
* Set the value of the spec constant via
  `handler::set_specialization_constant()`.
* Submit a kernel via `handler::single_task()`.
* Read the value of the spec constant via
  `kernel_handler::get_specialization_constant()` and make sure we get the same
   value back.

### Spec constant defined in another translation unit

* In one translation unit:
  - Define a `specialization_id` variable in the global namespace for the
    tested type.  The variable's default value has some non-zero value.
* In a second translation unit:
  - Declare an external reference to the `specialization_id` variable.
  - Create a `queue` from the tested device and call `queue::submit()`.
  - Set the value of the spec constant via
    `handler::set_specialization_constant()`.
  - Submit a kernel via `handler::single_task()`.
  - Read the value of the spec constant via
    `kernel_handler::get_specialization_constant()` and make sure we get the
    same value back.

### Spec constant defined and set in another translation unit

* This test runs only if the implementation defines `SYCL_EXTERNAL`.
* In one translation unit:
  - Define a `specialization_id` variable in the global namespace for the
    tested type.  The variable's default value has some non-zero value.
  - Create a `queue` from the tested device and call `queue::submit()`.
  - Set the value of the spec constant via
    `handler::set_specialization_constant()`.
  - Submit a kernel via `handler::single_task()`.  The kernel is declared
    `SYCL_EXTERNAL`.
* In a second translation unit:
  - Declare an external reference to the `specialization_id` variable.
  - Define the kernel which is declared `SYCL_EXTERNAL` above.
  - Read the value of the spec constant via
    `kernel_handler::get_specialization_constant()` and make sure we get the
    value we set in the first translation unit.
