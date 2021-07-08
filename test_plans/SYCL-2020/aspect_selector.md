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
The random combinations are selected at test generation time,
with each test generation producing the same combinations
and each run testing the same random combinations.

For each of the combinations described so far,
add new combinations which also use forbidden aspects `{noasp1, ..., noaspM}` -
`M` is independent of `N`, can be smaller, same size, or larger.
These forbidden aspects will be passed to the `denyList` argument.
Use the same rules as above for generating aspects,
following these extra restrictions:

* `noaspY` is not a member of the set `{asp1, ..., aspN}`
    for each `noaspY` from the set of forbidden aspects.

> Note: it is recommended a Python generator is written to cover all required combinations.

## Tests

### No aspects

1. `auto selector1 = aspect_selector()`
1. `auto selector2 = aspect_selector({})`
1. `auto selector3 = aspect_selector({}, {})`
1. `auto selector4 = aspect_selector<>()`

In the test, all constructed selectors `selectorX` are used
to construct a SYCL device, `devX` for each selector respectively.
All devices `devX` must compare equal to the device obtained by `default_selector`
(let's call this device `defaultDevice`).

### Requested aspects using a device

Assume requested aspects `{asp1, ..., aspN}` for each test,
and forbidden aspects `denyList = {noasp1, ..., noaspM}`.

These steps are performed before testing combinations:

1. Iterate over all devices to find out if there's one that contains all requested aspects
    but doesn't contain any forbidden ones,
    we call this `suitableDevice`.

    * If a device like this cannot be found, skip test.

1. Iterate over all platforms to find out if there's one that contains all requested aspects
    but doesn't contain any forbidden ones,
    we call this `suitablePlatform`.

    * If a platform like this cannot be found,
      continue testing but ignore further tests for the platform.

These steps are performed for each combination:

1. Test using a vector of aspects

    1. Construct `selectorVector` object obtained by calling `aspect_selector(std::vector{asp1, ..., aspN})`.
    1. Construct SYCL device (`selectedDevice`) using `selectorVector`.
    1. Construct SYCL queue (`selectedQueue`) using `selectorVector`.
    1. Check that `selectedDevice` contains all requested aspects
        and none of the forbidden ones.

        * `selectedDevice.has(asp1), ..., selectedDevice.has(aspN)`.
        * `!selectedDevice.has(noasp1), ..., !selectedDevice.has(noaspN)`.

            * If `denyList` is not empty.
        * The same checks for `selectedQueue.get_device()`.
    1. If `suitablePlatform` was found, construct SYCL platform (`selectedPlatform`) using `selectorVector`.
    1. Check that `selectedPlatform` contains all requested aspects.
1. If `denyList` is not empty, no further testing is required

    * Already covered by other combinations.
1. Test using function arguments

    1. Construct `selectorArgs` object obtained by calling `aspect_selector(asp1, ..., aspN)`.
    1. Construct `selectedDevice`, `selectedPlatform`, and `selectedQueue` using `selectorArgs`.
    1. Check that `selectedDevice`, `selectedPlatform`, and `selectedQueue.get_device()` contain all requested aspects.
1. Test using template parameters

    1. Construct `selectorTemplate` object obtained by calling `aspect_selector<asp1, ..., aspN>()`.
    1. Construct `selectedDevice`, `selectedPlatform`, and `selectedQueue` using `selectorArgs`.
    1. Check that `selectedDevice`, `selectedPlatform`, and `selectedQueue.get_device()` contain all requested aspects.
