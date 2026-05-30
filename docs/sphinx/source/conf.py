"""Sphinx configuration for UniLab documentation."""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime
from importlib import machinery, util
from types import ModuleType, SimpleNamespace
from typing import Any

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


class _MlxDocDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"mlx.core.{self.name}"


def _make_doc_module(name: str, *, package: bool = False) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_mlx_doc_stubs() -> None:
    # Sphinx's generic mock objects are callable and can look like wrapper
    # loops to sphinx-autodoc-typehints. MLX dtype defaults need simpler stubs.
    if "mlx" in sys.modules or util.find_spec("mlx") is not None:
        return

    mlx_module = _make_doc_module("mlx", package=True)
    core_module = _make_doc_module("mlx.core")
    nn_module = _make_doc_module("mlx.nn")
    optimizers_module = _make_doc_module("mlx.optimizers")
    utils_module = _make_doc_module("mlx.utils")

    class Dtype:
        pass

    class array:
        pass

    class Module:
        pass

    class Linear:
        pass

    class Adam:
        pass

    for cls, module_name in (
        (Dtype, "mlx.core"),
        (array, "mlx.core"),
        (Module, "mlx.nn"),
        (Linear, "mlx.nn"),
        (Adam, "mlx.optimizers"),
    ):
        cls.__module__ = module_name
        cls.__qualname__ = cls.__name__

    def _identity(*args: Any, **kwargs: Any) -> Any:
        return args[0] if args else None

    core_module.Dtype = Dtype
    core_module.array = array
    core_module.float32 = _MlxDocDtype("float32")
    core_module.int32 = _MlxDocDtype("int32")
    nn_module.Module = Module
    nn_module.Linear = Linear
    nn_module.init = SimpleNamespace(orthogonal=lambda *args, **kwargs: _identity)
    nn_module.value_and_grad = lambda *args, **kwargs: _identity
    nn_module.softplus = _identity
    optimizers_module.Adam = Adam
    utils_module.tree_flatten = lambda tree: []
    utils_module.tree_map = lambda fn, tree: tree

    mlx_module.core = core_module
    mlx_module.nn = nn_module
    mlx_module.optimizers = optimizers_module
    mlx_module.utils = utils_module
    sys.modules.update(
        {
            "mlx": mlx_module,
            "mlx.core": core_module,
            "mlx.nn": nn_module,
            "mlx.optimizers": optimizers_module,
            "mlx.utils": utils_module,
        }
    )


_install_mlx_doc_stubs()

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
# when the deps are missing. MLX uses the lightweight docs stubs above because
# Sphinx's generic mocks interact poorly with dtype defaults.
autodoc_mock_imports = [
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

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/lang_switcher.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
    ],
}

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
_LANGUAGE_ROOT_INDEX = {
    "en": "en/0-index",
    "zh_CN": "zh_CN/0-index",
}

# Explicit cross-language mapping for pages whose paths don't mirror 1:1.
# The language roots use numbered paths but are not a strict 1:1 mirror. This
# table keeps the language switcher landing on the closest equivalent page
# instead of bouncing to the language index. Forward direction only — reverse
# map is computed below.
_LANGUAGE_PATH_FORWARD: dict[str, str] = {}
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
        candidate = f"{language_code}/{rest}" if rest else _LANGUAGE_ROOT_INDEX[language_code]
        if candidate in found_docs:
            return candidate

    return _LANGUAGE_ROOT_INDEX[language_code]


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


def _sidebar_navigation_language(pagename: str) -> str:
    page_language = _page_language(pagename)
    if page_language in _LANGUAGE_DOC_ROOTS:
        return page_language
    return "en"


def _filter_sidebar_navigation_tree(
    pagename: str,
    context: dict[str, Any],
) -> str | None:
    # Root index.md keeps both language roots and shared resources in the build
    # graph. The sidebar should expose only the active language sections; shared
    # pages stay reachable through language-local wrapper pages.
    navigation_tree = context.get("furo_navigation_tree")
    pathto = context.get("pathto")
    if not navigation_tree or pathto is None:
        return None

    from bs4 import BeautifulSoup

    sidebar_language = _sidebar_navigation_language(pagename)
    language_root_href = pathto(_LANGUAGE_ROOT_INDEX[sidebar_language])

    soup = BeautifulSoup(navigation_tree, "html.parser")
    root_list = soup.find("ul")
    if root_list is None:
        return None

    active_language_item = None
    for item in list(root_list.find_all("li", recursive=False)):
        link = item.find("a", recursive=False)
        href = link.get("href", "") if link else ""
        if href != language_root_href:
            item.decompose()
        else:
            active_language_item = item

    if active_language_item is None:
        return str(soup)

    language_children = active_language_item.find("ul", recursive=False)
    if language_children is None:
        return str(soup)

    root_list.clear()
    for child in list(language_children.find_all("li", recursive=False)):
        root_list.append(child.extract())

    # Furo's generated classes encode the visual nesting depth. After removing
    # the language index wrapper, decrement those classes so section roots render
    # as top-level sidebar entries.
    for node in root_list.find_all(True):
        classes = node.get("class")
        if not classes:
            continue

        normalized_classes = []
        for class_name in classes:
            prefix = "toctree-l"
            if class_name.startswith(prefix):
                level = class_name[len(prefix) :]
                if level.isdigit() and int(level) > 1:
                    normalized_classes.append(f"{prefix}{int(level) - 1}")
                    continue
            normalized_classes.append(class_name)

        node["class"] = normalized_classes

    return str(soup)


def _inject_language_sidebar_navigation(
    app: Any,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    if doctree is None:
        return

    navigation_tree = _filter_sidebar_navigation_tree(pagename, context)
    if navigation_tree is not None:
        context["furo_navigation_tree"] = navigation_tree


# Expose `_UNILAB_AVAILABLE` as a Sphinx tag so `.. only:: api_ref` blocks
# in prose can be conditionally rendered.
def setup(app):
    if _UNILAB_AVAILABLE:
        app.tags.add("api_ref")
    else:
        app.tags.add("prose_only")
    app.connect("html-page-context", _inject_language_switcher)
    app.connect("html-page-context", _inject_language_sidebar_navigation, priority=700)
    return {"parallel_read_safe": True}
