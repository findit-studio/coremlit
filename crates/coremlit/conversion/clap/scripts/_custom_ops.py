"""Custom coremltools torch-op shims for CLAP conversion.

`new_ones`: coremltools 9.0 ships `new_zeros` but not `new_ones`. transformers
5.14's `masking_utils` builds the bidirectional attention mask with
`q_idx.new_ones((), dtype=torch.bool)` (a scalar all-True). We register the exact
sibling of the stock `new_zeros` op (fill-with-1 of the traced-constant shape).
Importing this module registers the op globally for any subsequent ct.convert.
"""
import numpy as np

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import Var
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import (
    register_torch_op,
    _TORCH_OPS_REGISTRY,
)

if "new_ones" not in _TORCH_OPS_REGISTRY.name_to_func_mapping:

    @register_torch_op
    def new_ones(context, node):
        # aten::new_ones(self, size, dtype, layout, device, pin_memory). Both CLAP
        # call sites are `new_ones((), dtype=torch.bool)` -> a scalar True.
        inputs = _get_inputs(context, node)
        shape = inputs[1]
        if isinstance(shape, (list, tuple)):
            if len(shape) == 0:  # new_ones(()) -> bool scalar True
                context.add(mb.const(val=True, name=node.name))
                return
            shape = mb.concat(values=list(shape), axis=0)
        elif isinstance(shape, Var):
            # A constant, possibly-empty shape tensor. Empty () -> scalar True.
            if shape.val is not None and int(np.prod(np.shape(shape.val))) == 0:
                context.add(mb.const(val=True, name=node.name))
                return
            shape = mb.cast(x=shape, dtype="int32")  # fill wants an int32 shape
        context.add(mb.fill(shape=shape, value=1.0, name=node.name))
