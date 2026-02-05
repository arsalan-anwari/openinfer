Export .oinf Files
==================

Walkthrough for exporting `.oinf` files using the Python tooling.

Outline
-------

- Define a dataclass
- Encode to `.oinf`
- Verify output

Define a dataclass
------------------

.. code-block:: python

   from dataclasses import dataclass
   from oinf_encoder import TensorSpec, SizeVar

   @dataclass
   class Model:
       B: SizeVar
       w: TensorSpec

Encode
------

.. code-block:: bash

   python openinfer-oinf/dataclass_to_oinf.py --input my_model:Model --output model.oinf

Verify
------

.. code-block:: bash

   python openinfer-oinf/verify_oinf.py model.oinf
