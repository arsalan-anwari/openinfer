Workflow
========

How to run the Python tooling and tests.

Topics
------

- Encoding dataclasses to `.oinf`
- Verifying and inspecting `.oinf` files

Encode
------

.. code-block:: bash

   python openinfer-oinf/dataclass_to_oinf.py --input module:MyModel --output model.oinf

Verify
------

.. code-block:: bash

   python openinfer-oinf/verify_oinf.py model.oinf
