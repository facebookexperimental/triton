============
Installation
============

For supported platform/OS and supported hardware, review the `Compatibility <https://github.com/triton-lang/triton?tab=readme-ov-file#compatibility>`_ section on Github.

--------------------
Binary Distributions
--------------------

You can install the latest stable release of Triton from pip:

.. code-block:: bash

      pip install triton

Binary wheels are available for CPython 3.10-3.14.

-----------
From Source
-----------

++++++++++++++
Python Package
++++++++++++++

You can install the Python package from source by running the following commands:

.. code-block:: bash

      git clone https://github.com/triton-lang/triton.git
      cd triton

      pip install -r python/requirements.txt # build-time dependencies
      pip install -e .

Note that, if llvm is not present on your system, the setup.py script will download the official LLVM static libraries and link against that.

For building with a custom LLVM, review the `Building with a custom LLVM <https://github.com/triton-lang/triton?tab=readme-ov-file#building-with-a-custom-llvm>`_ section on Github.

You can then test your installation by running the tests:

.. code-block:: bash

      # One-time setup
      make dev-install

      # To run all tests (requires a GPU)
      make test

      # Or, to run tests without a GPU
      make test-nogpu

----------------------------------------
uTLX: standalone TLX for upstream Triton
----------------------------------------

`triton-utlx <https://pypi.org/project/triton-utlx/>`_ (uTLX) distributes TLX -- a low-level, warp-aware extension of the Triton DSL, with intrinsics for asynchronous copies, warp-group MMA, barriers, and shared/tensor-memory buffers -- as a Triton plugin, so it does not require switching to the `FBTriton <https://github.com/facebookexperimental/triton>`_ fork:

.. code-block:: bash

      pip install torch
      pip install triton-utlx

      export TRITON_PLUGIN_PATHS=$(python -c \
        "import utlx_plugin, os; print(os.path.join(os.path.dirname(utlx_plugin.__file__), 'libutlx.so'))")

``TRITON_PLUGIN_PATHS`` is a colon-separated list of plugin shared libraries, and nothing sets it for you. Plugins additionally load only into a Triton built with ``TRITON_EXT_ENABLED``, which exposes the ``libtriton`` symbols they link against; a Triton built without it warns and skips the plugin rather than failing outright.

The ``triton`` that ships with a PyTorch release has ``TRITON_EXT_ENABLED`` on by default. To run against a Triton you build yourself instead, turn it on at build time:

.. code-block:: bash

      TRITON_EXT_ENABLED=ON pip install -e . --no-build-isolation
