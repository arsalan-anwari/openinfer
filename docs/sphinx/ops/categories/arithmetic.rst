Arithmetic Operations
=====================

Operations in the `arithmetic` category.

- total ops: 10
- cpu support: 10
- vulkan support: 10

.. list-table::
   :header-rows: 1

   * - Op
     - Inputs
     - Outputs
     - Capabilities
   * - :doc:`../operations/abs`
     - 1
     - 1
     - inplace, accumulate
   * - :doc:`../operations/add`
     - 2
     - 1
     - broadcast, inplace, accumulate
   * - :doc:`../operations/div`
     - 2
     - 1
     - broadcast, inplace
   * - :doc:`../operations/floor_div`
     - 2
     - 1
     - broadcast, inplace
   * - :doc:`../operations/fma`
     - 3
     - 1
     - inplace
   * - :doc:`../operations/mul`
     - 2
     - 1
     - broadcast, inplace, accumulate
   * - :doc:`../operations/neg`
     - 1
     - 1
     - inplace
   * - :doc:`../operations/recip`
     - 1
     - 1
     - inplace
   * - :doc:`../operations/rem`
     - 2
     - 1
     - broadcast, inplace
   * - :doc:`../operations/sub`
     - 2
     - 1
     - broadcast, inplace, accumulate
