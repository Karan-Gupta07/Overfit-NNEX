"""Format core: scene graph, geometry, colour.

Dependency rule: this package may
import only the stdlib and numpy. No ml, no render, no format. That is what
lets the decoder be ported without dragging PyTorch along, and what makes a
classical-only ablation possible.
"""
