"""SCDX container, scene graph and conventional raster codecs.

The photo coder in scripts/research/local_probability/ uses this package for its
SCDX floor. The name dates from the project's start as a screenshot codec.
"""

__version__ = "0.0.1"

# Dependency rule:
#   core -> (stdlib, numpy) only.  No ml, no format, no render.
# Checked by the tests in tests/.
