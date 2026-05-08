"""Cold-path scene composition and terrain materialization.

Ported from mjlab (https://github.com/mjlab/mjlab) for unilab issues #197 / #270.
The composer takes a base robot XML plus a TerrainGeneratorCfg and writes a
final scene.xml + assets directory that unilab's existing model_file path can
load. All materialization happens at env init; step/reset never touches assets.
"""

from unilab.scene.composer import compose_and_materialize
from unilab.scene.spec_export import export_spec, non_default_option_fields
from unilab.scene.spec_xml import fix_spec_xml, strip_buffer_textures

__all__ = [
    "compose_and_materialize",
    "export_spec",
    "fix_spec_xml",
    "non_default_option_fields",
    "strip_buffer_textures",
]
