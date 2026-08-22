"""
Parametric CAD for the desk-sorting arm.

Every module in this package derives its dimensions from
:mod:`src.geometry` -- ``DEFAULT_ARM`` for kinematic lengths and
``DEFAULT_HARDWARE`` for off-the-shelf component envelopes. No module here
declares a physical dimension of its own; changing a number in
``src/geometry.py`` regenerates every part consistently.

See ``cad/README.md`` for the parametric approach and STL regeneration.
"""
