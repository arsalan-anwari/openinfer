Operations Catalog
==================

This section is generated from `ops.json` to stay in sync with the runtime.

Summary
-------
- total ops: 47

Alphabetical Index
------------------

.. list-table::
   :header-rows: 1

   * - Op
     - Category
     - Capabilities
     - Devices
   * - :doc:`operations/abs`
     - arithmetic
     - inplace, accumulate
     - cpu, vulkan
   * - :doc:`operations/add`
     - arithmetic
     - broadcast, inplace, accumulate
     - cpu, vulkan
   * - :doc:`operations/and`
     - bitwise
     - inplace
     - cpu, vulkan
   * - :doc:`operations/argmax_axis`
     - reduction
     - -
     - cpu, vulkan
   * - :doc:`operations/argmin_axis`
     - reduction
     - -
     - cpu, vulkan
   * - :doc:`operations/cast`
     - casting
     - -
     - cpu, vulkan
   * - :doc:`operations/ceil`
     - rounding
     - inplace
     - cpu, vulkan
   * - :doc:`operations/clamp`
     - rounding
     - inplace
     - cpu, vulkan
   * - :doc:`operations/div`
     - arithmetic
     - broadcast, inplace
     - cpu, vulkan
   * - :doc:`operations/eq`
     - comparison
     - -
     - cpu, vulkan
   * - :doc:`operations/fill`
     - mutation
     - inplace
     - cpu, vulkan
   * - :doc:`operations/filter`
     - filter
     - -
     - cpu, vulkan
   * - :doc:`operations/floor`
     - rounding
     - inplace
     - cpu, vulkan
   * - :doc:`operations/floor_div`
     - arithmetic
     - broadcast, inplace
     - cpu, vulkan
   * - :doc:`operations/fma`
     - arithmetic
     - inplace
     - cpu, vulkan
   * - :doc:`operations/ge`
     - comparison
     - -
     - cpu, vulkan
   * - :doc:`operations/gt`
     - comparison
     - -
     - cpu, vulkan
   * - :doc:`operations/is_finite`
     - filter
     - -
     - cpu, vulkan
   * - :doc:`operations/is_inf`
     - filter
     - -
     - cpu, vulkan
   * - :doc:`operations/is_nan`
     - filter
     - -
     - cpu, vulkan
   * - :doc:`operations/is_neg`
     - filter
     - -
     - cpu, vulkan
   * - :doc:`operations/le`
     - comparison
     - -
     - cpu, vulkan
   * - :doc:`operations/lt`
     - comparison
     - -
     - cpu, vulkan
   * - :doc:`operations/matmul`
     - numerical
     - broadcast, inplace, accumulate
     - cpu, vulkan
   * - :doc:`operations/max`
     - statistics
     - inplace
     - cpu, vulkan
   * - :doc:`operations/max_axis`
     - reduction
     - -
     - cpu, vulkan
   * - :doc:`operations/mean_axis`
     - reduction
     - accumulate
     - cpu, vulkan
   * - :doc:`operations/min`
     - statistics
     - inplace
     - cpu, vulkan
   * - :doc:`operations/min_axis`
     - reduction
     - -
     - cpu, vulkan
   * - :doc:`operations/mul`
     - arithmetic
     - broadcast, inplace, accumulate
     - cpu, vulkan
   * - :doc:`operations/ne`
     - comparison
     - -
     - cpu, vulkan
   * - :doc:`operations/neg`
     - arithmetic
     - inplace
     - cpu, vulkan
   * - :doc:`operations/not`
     - bitwise
     - inplace
     - cpu, vulkan
   * - :doc:`operations/or`
     - bitwise
     - inplace
     - cpu, vulkan
   * - :doc:`operations/popcount`
     - bitwise
     - -
     - cpu, vulkan
   * - :doc:`operations/prod_axis`
     - reduction
     - accumulate
     - cpu, vulkan
   * - :doc:`operations/recip`
     - arithmetic
     - inplace
     - cpu, vulkan
   * - :doc:`operations/relu`
     - numerical
     - inplace
     - cpu, vulkan
   * - :doc:`operations/rem`
     - arithmetic
     - broadcast, inplace
     - cpu, vulkan
   * - :doc:`operations/round`
     - rounding
     - inplace
     - cpu, vulkan
   * - :doc:`operations/shl`
     - bitwise
     - inplace
     - cpu, vulkan
   * - :doc:`operations/shr`
     - bitwise
     - inplace
     - cpu, vulkan
   * - :doc:`operations/sign`
     - statistics
     - inplace
     - cpu, vulkan
   * - :doc:`operations/sub`
     - arithmetic
     - broadcast, inplace, accumulate
     - cpu, vulkan
   * - :doc:`operations/sum_axis`
     - reduction
     - accumulate
     - cpu, vulkan
   * - :doc:`operations/trunc`
     - rounding
     - inplace
     - cpu, vulkan
   * - :doc:`operations/xor`
     - bitwise
     - inplace
     - cpu, vulkan

By Capability
-------------

- broadcast: ops that allow shape broadcasting
- inplace: ops that can write into an existing output buffer
- accumulate: ops that support accumulation modes

Device Matrix
-------------

.. list-table::
   :header-rows: 1

   * - Op
     - CPU
     - Vulkan
   * - :doc:`operations/abs`
     - yes
     - yes
   * - :doc:`operations/add`
     - yes
     - yes
   * - :doc:`operations/and`
     - yes
     - yes
   * - :doc:`operations/argmax_axis`
     - yes
     - yes
   * - :doc:`operations/argmin_axis`
     - yes
     - yes
   * - :doc:`operations/cast`
     - yes
     - yes
   * - :doc:`operations/ceil`
     - yes
     - yes
   * - :doc:`operations/clamp`
     - yes
     - yes
   * - :doc:`operations/div`
     - yes
     - yes
   * - :doc:`operations/eq`
     - yes
     - yes
   * - :doc:`operations/fill`
     - yes
     - yes
   * - :doc:`operations/filter`
     - yes
     - yes
   * - :doc:`operations/floor`
     - yes
     - yes
   * - :doc:`operations/floor_div`
     - yes
     - yes
   * - :doc:`operations/fma`
     - yes
     - yes
   * - :doc:`operations/ge`
     - yes
     - yes
   * - :doc:`operations/gt`
     - yes
     - yes
   * - :doc:`operations/is_finite`
     - yes
     - yes
   * - :doc:`operations/is_inf`
     - yes
     - yes
   * - :doc:`operations/is_nan`
     - yes
     - yes
   * - :doc:`operations/is_neg`
     - yes
     - yes
   * - :doc:`operations/le`
     - yes
     - yes
   * - :doc:`operations/lt`
     - yes
     - yes
   * - :doc:`operations/matmul`
     - yes
     - yes
   * - :doc:`operations/max`
     - yes
     - yes
   * - :doc:`operations/max_axis`
     - yes
     - yes
   * - :doc:`operations/mean_axis`
     - yes
     - yes
   * - :doc:`operations/min`
     - yes
     - yes
   * - :doc:`operations/min_axis`
     - yes
     - yes
   * - :doc:`operations/mul`
     - yes
     - yes
   * - :doc:`operations/ne`
     - yes
     - yes
   * - :doc:`operations/neg`
     - yes
     - yes
   * - :doc:`operations/not`
     - yes
     - yes
   * - :doc:`operations/or`
     - yes
     - yes
   * - :doc:`operations/popcount`
     - yes
     - yes
   * - :doc:`operations/prod_axis`
     - yes
     - yes
   * - :doc:`operations/recip`
     - yes
     - yes
   * - :doc:`operations/relu`
     - yes
     - yes
   * - :doc:`operations/rem`
     - yes
     - yes
   * - :doc:`operations/round`
     - yes
     - yes
   * - :doc:`operations/shl`
     - yes
     - yes
   * - :doc:`operations/shr`
     - yes
     - yes
   * - :doc:`operations/sign`
     - yes
     - yes
   * - :doc:`operations/sub`
     - yes
     - yes
   * - :doc:`operations/sum_axis`
     - yes
     - yes
   * - :doc:`operations/trunc`
     - yes
     - yes
   * - :doc:`operations/xor`
     - yes
     - yes

By Device
---------

Ops can declare CPU and/or Vulkan support in `ops.json`.

By Category
-----------

.. toctree::
   :maxdepth: 1
   :glob:

   categories/*

By Operation
------------

.. toctree::
   :maxdepth: 1
   :glob:

   operations/*

