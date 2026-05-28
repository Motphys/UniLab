"""Sphinx configuration for UniLab documentation."""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — make the UniLab source tree importable for autodoc.
# Layout: docs live in-repo at <repo>/docs/sphinx/source/, so the package
# is at ../../../src. CI additionally runs `pip install -e .`, which makes
# the explicit sys.path push redundant but harmless.
# ---------------------------------------------------------------------------
_here = os.path.abspath(os.path.dirname(__file__))
_candidate_siblings = [
    os.path.abspath(os.path.join(_here, "..", "..", "..", "src")),
]
for _p in _candidate_siblings:
    if os.path.isdir(os.path.join(_p, "unilab")):
        sys.path.insert(0, _p)
        break

# Probe whether unilab is importable. If not (heavy deps missing), we
# downgrade the build: skip autodoc / autosummary so a preview build still
# produces a usable site for the prose pages. CI installs UniLab properly
# and gets the full API reference.
_UNILAB_AVAILABLE = False
_UNILAB_VERSION = "0.1.0"
try:
    import unilab  # type: ignore

    _UNILAB_AVAILABLE = True
    _UNILAB_VERSION = getattr(unilab, "__version__", "0.1.0")
except Exception as exc:  # pragma: no cover — diagnostic only
    warnings.warn(
        f"UniLab is not importable in this environment ({exc!r}); "
        "API reference will be skipped for this build.",
        stacklevel=1,
    )

# Honour an explicit opt-out for ultra-fast prose-only previews.
if os.environ.get("UNILAB_DOCS_SKIP_AUTODOC") == "1":
    _UNILAB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project info
# ---------------------------------------------------------------------------
project = "UniLab"
author = "UniLab Sim Authors"
copyright = f"{datetime.now().year}, {author}"
release = _UNILAB_VERSION
version = release

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.video",
    "sphinxcontrib.mermaid",
    "sphinx_sitemap",
]
# Only enable autodoc / autosummary when UniLab is importable. Otherwise
# autosummary's recursive import probe blows up the whole build.
if _UNILAB_AVAILABLE:
    extensions.extend(
        [
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx_autodoc_typehints",
        ]
    )

# MyST settings — closer to GitHub-flavored Markdown.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "linkify",
    "substitution",
    "tasklist",
    "fieldlist",
    "attrs_inline",
]
myst_heading_anchors = 4
myst_linkify_fuzzy_links = False

# Substitutions that vary with build mode (full vs prose-only).
if _UNILAB_AVAILABLE:
    _api_ref_blurb = (
        "Class / function reference auto-generated from `unilab` — typed "
        "signatures and source links for every public symbol."
    )
    _api_ref_label = "API Reference"
    _api_ref_button = "[Browse the API →](api_reference/index.html){.sd-btn .sd-btn-primary}"
else:
    _api_ref_blurb = (
        "API reference will publish once UniLab source is available to the "
        "build environment. Browse the typed source tree on GitHub in the "
        "meantime."
    )
    _api_ref_label = "`unilab/` on GitHub"
    _api_ref_button = (
        "[View source on GitHub →]"
        "(https://github.com/unilabsim/UniLab/tree/main/src/unilab)"
        "{.sd-btn .sd-btn-primary}"
    )

myst_substitutions = {
    "api_ref_blurb": _api_ref_blurb,
    "api_ref_label": _api_ref_label,
    "api_ref_button": _api_ref_button,
}

# Autodoc / autosummary -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "separated"
autosummary_generate = _UNILAB_AVAILABLE
autosummary_imported_members = False
typehints_fully_qualified = False
always_document_param_types = True

# Heavy / optional deps that should not block doc builds.
# These are mocked so `autodoc` can still import unilab modules in CI even
# when the deps are missing (e.g. mlx on non-macOS runners).
autodoc_mock_imports = [
    "mlx",
    "mlx.core",
    "mlx.nn",
    "motrixsim",
    "mxpython",
    "wandb",
    "viser",
    "onnxruntime",
    "rsl_rl",
    "tensorboard",
    "mediapy",
]

# Napoleon (Google / NumPy docstring styles)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "hydra": ("https://hydra.cc/docs/", None),
}

# General -------------------------------------------------------------------
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# When autodoc is disabled, skip the entire API reference tree — its pages
# only contain autosummary directives that would otherwise fail.
if not _UNILAB_AVAILABLE:
    exclude_patterns.append("api_reference/**")
language = (
    "en"  # Sphinx search-index language only; both /en/ and /zh_CN/ trees are built in one pass
)
nitpicky = False  # flip to True once API ref stabilizes

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = "furo"
html_title = f"UniLab {release} documentation"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_favicon = None  # add _static/favicon.ico when ready
# html_logo = "_static/images/logo.png"
html_baseurl = "https://unilabsim.github.io/UniLab-doc/"
sitemap_url_scheme = "{link}"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/unilabsim/UniLab/",
    "source_branch": "main",
    "source_directory": "docs/sphinx/source/",
    "top_of_page_buttons": ["view", "edit"],
    "light_css_variables": {
        "color-brand-primary": "#2563eb",
        "color-brand-content": "#1d4ed8",
        "font-stack": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
        "font-stack--monospace": "'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#93c5fd",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/unilabsim/UniLab",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 '
                "3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82"
                "-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15"
                "-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28"
                "-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-"
                "1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
                "1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
                "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21"
                '.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# ---------------------------------------------------------------------------
# Copy button — strip the prompt characters when copying.
# ---------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |# "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

# ---------------------------------------------------------------------------
# Suppress noisy warnings while the doc is still bootstrapping.
# ---------------------------------------------------------------------------
suppress_warnings = ["myst.header"]
# When api_reference is excluded (no UniLab source), the hidden toctree on
# index.md still references those docs — silence the resulting noise.
if not _UNILAB_AVAILABLE:
    suppress_warnings.append("toc.excluded")


_LANGUAGE_DOC_ROOTS = ("en", "zh_CN")

# Explicit cross-language mapping for pages whose paths don't mirror 1:1.
# zh_CN still carries legacy numbered paths (01-getting-started, A-getting-started/,
# etc.) from the pre-migration content. Until those are renamed, this table keeps
# the language switcher landing on the closest equivalent page instead of
# bouncing to the language index. Forward direction only — reverse map is
# computed below.
_LANGUAGE_PATH_FORWARD: dict[str, str] = {
    "en/user_guide/getting_started/quickstart": "zh_CN/user_guide/01-getting-started",
    "en/user_guide/getting_started/installation": "zh_CN/user_guide/A-getting-started/01-install",
    "en/user_guide/getting_started/training": "zh_CN/user_guide/03-training",
    "en/user_guide/getting_started/configuration_overrides": "zh_CN/user_guide/B-training/03-hydra-overrides",
    "en/user_guide/getting_started/evaluation_and_playback": "zh_CN/user_guide/B-training/02-playback-and-resume",
    "en/user_guide/backends/index": "zh_CN/user_guide/02-simulation-backends",
    "en/user_guide/backends/choosing_a_backend": "zh_CN/user_guide/E-reference/01-backend-support-matrix",
    "en/user_guide/algorithms/overview": "zh_CN/user_guide/04-algorithms",
    "en/user_guide/algorithms/ppo": "zh_CN/user_guide/C-algorithms/01-ppo-torch",
    "en/user_guide/algorithms/mlx_ppo": "zh_CN/user_guide/C-algorithms/02-mlx-ppo",
    "en/user_guide/algorithms/appo": "zh_CN/user_guide/C-algorithms/03-appo",
    "en/user_guide/algorithms/fast_sac": "zh_CN/user_guide/C-algorithms/04-sac",
    "en/user_guide/algorithms/fast_td3": "zh_CN/user_guide/C-algorithms/05-td3",
    "en/user_guide/algorithms/flash_sac": "zh_CN/user_guide/C-algorithms/06-flashsac",
    "en/user_guide/tasks/g1_motion_tracking": "zh_CN/user_guide/D-tasks/02-g1-motion-tracking",
    "en/user_guide/tasks/go2_arm_manip_loco": "zh_CN/user_guide/D-tasks/06-go2-arm-manip-loco",
    "en/user_guide/domain_randomization/index": "zh_CN/user_guide/05-domain-randomization",
    "en/developer_guide/architecture/development_standard": "zh_CN/developer_guide/development-standard",
    "en/developer_guide/architecture/scene_composition": "zh_CN/developer_guide/scene-composition-design",
    "en/developer_guide/contracts/domain_randomization": "zh_CN/developer_guide/domain-randomization-contract",
    "en/developer_guide/contributing": "zh_CN/developer_guide/CONTRIBUTING",
    "en/developer_guide/contributing_workflow": "zh_CN/developer_guide/collaboration",
    "en/agents/index": "zh_CN/agents/01-agent-quick-reference",
}
# Keyed by (current_pagename, target_language) → target_pagename.
_LANGUAGE_PATH_MAP: dict[tuple[str, str], str] = {}
for _en_page, _zh_page in _LANGUAGE_PATH_FORWARD.items():
    _LANGUAGE_PATH_MAP[(_en_page, "zh_CN")] = _zh_page
    _LANGUAGE_PATH_MAP[(_zh_page, "en")] = _en_page


def _page_language(pagename: str) -> str:
    root = pagename.split("/", 1)[0]
    if root in _LANGUAGE_DOC_ROOTS:
        return root
    return "shared"


def _language_target(app, pagename: str, language_code: str) -> str:
    found_docs = app.env.found_docs
    current_language = _page_language(pagename)

    # Same language: stay on the same page.
    if current_language == language_code:
        return pagename

    # Try explicit map for known legacy mismatches.
    mapped = _LANGUAGE_PATH_MAP.get((pagename, language_code))
    if mapped and mapped in found_docs:
        return mapped

    # Try direct 1:1 mirror.
    if current_language in _LANGUAGE_DOC_ROOTS:
        _, _, rest = pagename.partition("/")
        candidate = f"{language_code}/{rest}" if rest else f"{language_code}/index"
        if candidate in found_docs:
            return candidate

    return f"{language_code}/index"


def _inject_language_switcher(app, pagename, templatename, context, doctree):
    if doctree is None or pagename == "index":
        return

    context["current_language"] = _page_language(pagename)
    context["language_switcher_targets"] = {
        language_code: _language_target(app, pagename, language_code)
        for language_code in _LANGUAGE_DOC_ROOTS
    }
    switcher = app.builder.templates.render("language_switcher.html", context)
    context["body"] = f"{switcher}\n{context.get('body', '')}"


# Expose `_UNILAB_AVAILABLE` as a Sphinx tag so `.. only:: api_ref` blocks
# in prose can be conditionally rendered.
def setup(app):
    if _UNILAB_AVAILABLE:
        app.tags.add("api_ref")
    else:
        app.tags.add("prose_only")
    app.connect("html-page-context", _inject_language_switcher)
    return {"parallel_read_safe": True}
