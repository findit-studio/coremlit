"""Builds `two_functions.mlpackage`: the committed two-function ML Program that
`Model::load_function`'s hermetic laws compile and load (`src/model/tests.rs`).

Each function takes one input named `x` but at its own shape, and names its
output differently, so a loaded function's description says which function
Core ML handed back, and a prediction says it again:

    function            input          output         computes
    double              x fp32 [2]     y fp32 [2]     2 * x
    negate (default)    x fp32 [3]     z fp32 [3]     -x

The default function is `negate`, the SECOND one added. A load that dropped
the function name would get `negate` for `double`; a load that fell back to
the first function instead of the declared default would get `double` for no
name. Both are visible in the description.

Toolchain: python 3.11, coremltools 9.0, numpy 1.26.4 (the conversion
recipes' coremltools). No torch: the two programs are written with the MIL
builder. Usage:

    python -I two_functions.py <out-dir>

writes `<out-dir>/two_functions.mlpackage` and refuses to overwrite one.

Both programs' constants are scalars, which the converter writes inline, so
the package has no weights. coremltools still writes an empty `weights`
directory and a `Manifest.json` item for it. Git keeps no empty directory, and
Core ML refuses a package whose manifest names an item that is not there
("Item does not exist for identifier"), so this script removes both.

The package is not byte-reproducible: `save_multifunction` writes fresh item
identifiers into `Manifest.json` on every run. The function names, the default
and each function's I/O, which are what the laws read, are reproducible.
"""

import json
import pathlib
import sys
import tempfile

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

COREMLTOOLS = "9.0"


def single_function_package(directory, name, size, factor, output):
    """One single-function ML Program, `output = factor * x`, saved as `name`."""

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(size,), dtype=types.fp32)],
        opset_version=ct.target.macOS15,
    )
    def program(x):
        return mb.mul(x=x, y=np.float32(factor), name=output)

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT32,
    )
    path = pathlib.Path(directory) / f"{name}.mlpackage"
    model.save(str(path))
    return path


def drop_empty_weights(package):
    """Removes the empty `weights` directory and its `Manifest.json` item."""
    weights = package / "Data" / "com.apple.CoreML" / "weights"
    if any(weights.iterdir()):
        sys.exit(f"{weights} is not empty; the programs were meant to carry no weights")
    weights.rmdir()
    manifest_path = package / "Manifest.json"
    manifest = json.loads(manifest_path.read_text())
    items = manifest["itemInfoEntries"]
    dropped = [key for key, item in items.items() if item["path"] == "com.apple.CoreML/weights"]
    if len(dropped) != 1:
        sys.exit(f"expected one `weights` item in {manifest_path}, found {len(dropped)}")
    del items[dropped[0]]
    manifest_path.write_text(json.dumps(manifest, indent=4))


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python -I two_functions.py <out-dir>")
    if ct.__version__ != COREMLTOOLS:
        sys.exit(f"coremltools {ct.__version__} found; this fixture is built with {COREMLTOOLS}")
    out = pathlib.Path(sys.argv[1]) / "two_functions.mlpackage"
    if out.exists():
        sys.exit(f"{out} exists; remove it first")

    with tempfile.TemporaryDirectory() as scratch:
        double = single_function_package(scratch, "double", 2, 2.0, "y")
        negate = single_function_package(scratch, "negate", 3, -1.0, "z")
        descriptor = ct.utils.MultiFunctionDescriptor()
        descriptor.add_function(str(double), src_function_name="main", target_function_name="double")
        descriptor.add_function(str(negate), src_function_name="main", target_function_name="negate")
        descriptor.default_function_name = "negate"
        ct.utils.save_multifunction(descriptor, str(out))
    drop_empty_weights(out)
    print(out)


if __name__ == "__main__":
    main()
