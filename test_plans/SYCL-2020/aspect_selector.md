# Test plan for the aspect selector

This is the test plan for the `aspect_selector` feature of SYCL 2020.

## Testing scope

### Device coverage

All tests construct a test device (i.e. `testDevice`),
representing the device where conformance will be passed on.
Each test will construct an aspect selector object,
which will try to select another device,
this one based on the requested aspects (i.e. `selectedDevice`).
`selectedDevice` can be any device in the system,
not necessarily the same as `testDevice`.

### Function overloads

Each test that tests aspects `{asp1, ..., aspN}` must test all three overloads of the `aspect_selector` function:

* Pass a vector of aspects: `aspect_selector(std::vector{asp1, ..., aspN})`
* Pass aspects as function arguments: `aspect_selector(asp1, ..., aspN)`
* Pass aspects as template parameters: `aspect_selector<asp1, ..., aspN>()`

### Combinations of aspects

All aspects listed in the `enum class aspect` will be tested,
let's identify these aspects as `{asp1, ..., aspN}`.

The CTS will cover these combinations:

1. All combinations where no aspect is passed in.
2. All combinations where a single aspect is used.
3. All combinations of two aspects,
including duplicates (e.g. `aspect_selector(aspX, aspX)`),
and reversed order (e.g. `aspect_selector(aspX, aspY)` and `aspect_selector(aspY, aspX)`).

The CTS does not have to test all combinations of 3 or more aspects.

The CTS will test all the different numbers of aspects at least once, up to `N`,
where the aspects are not duplicated,
i.e. `aspect_selector(asp1), aspect_selector(asp1, asp2), ..., aspect_selector(asp1, asp2, ..., aspN)`.

Additional testing will include 100 combinations
of randomly selected aspect combinations,
allowing aspect duplicates,
with a random arity.

> Note: it is recommended a Python generator is written to cover all required combinations.

## Tests

### No aspects

1. `auto selector1 = aspect_selector()`
2. `auto selector2 = aspect_selector<>()`

In the test, both `selector1` and `selector2` are used
to construct a SYCL device, `dev1` and `dev2` respectively.
Both `dev1` and `dev2` must compare equal to the device obtained by `default_selector`
(let's call this device `defaultDevice`).

### Requested aspects using a device

Assume requested aspects `{asp1, ..., aspN}` for each test.

These steps are performed for each combination:

1. Construct `testDevice`, which represents the device where conformance will be passed on.

    * Also retrieve `testPlatform` SYCL platform from `testDevice` and construct `testQueue` SYCL queue from the device.
2. Test using a vector of aspects

    1. Construct `selectorVector` object obtained by calling `aspect_selector(std::vector{asp1, ..., aspN})`.
    2. Try/catch block

        1. Construct SYCL device (`selectedDevice`) using `selectorVector`.
        2. If exception is throw:

            * If `testDevice` contains all the requested aspects,
              but no device was found using the aspect selector,
              this is a failed test.
            * Otherwise there is most likely no device in the system with the requested aspects, skip test.
        3. If test passed so far, there should be no exceptions anymore with further testing.
        4. The same checks for `testPlatform` (SYCL platform from aspect selector object)
        and `testQueue` (SYCL queue from aspect selector object).

            * Return `selectedPlatform` and `selectedQueue`, respectively.
    3. Check that `selectedDevice` contains all requested aspects.

        * `selectedDevice.has(asp1), ..., selectedDevice.has(aspN)`.
        * The same checks for `selectedPlatform` and `selectedQueue`.
3. Test using function arguments

    1. Construct `selectorArgs` object obtained by calling `aspect_selector(asp1, ..., aspN)`.
    2. Construct `selectedDevice`, `selectedPlatform`, and `selectedQueue` using `selectorArgs`.
    3. Check that `selectedDevice`, `selectedPlatform`, and `selectedQueue` contain all requested aspects.
4. Test using template parameters

    1. Construct `selectorTemplate` object obtained by calling `aspect_selector<asp1, ..., aspN>()`.
    2. Construct `selectedDevice`, `selectedPlatform`, and `selectedQueue` using `selectorArgs`.
    3. Check that `selectedDevice`, `selectedPlatform`, and `selectedQueue` contain all requested aspects.
