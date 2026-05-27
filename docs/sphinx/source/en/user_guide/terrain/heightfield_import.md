# Heightfield Import

The procedural generator at `unilab.terrains.heightfield_terrains` lets you import a real-world or scanned heightfield as a sim asset. Pattern:

1. Resample the heightfield to UniLab's grid resolution.
2. Express it as a NumPy array of shape `(rows, cols)`.
3. Pass to the terrain config under `conf/terrain/heightfield/...`.

See {py:mod}`unilab.terrains.heightfield_terrains` for the import API.
