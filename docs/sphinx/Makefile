# Minimal makefile for Sphinx documentation
#
# Default (`make html`, `make live`) is lenient: warnings stay warnings so
# the build completes even while many doc stubs are unfilled. Use
# `make strict` to promote warnings to errors — same as CI's release path.
SPHINXOPTS    ?= -n
STRICT_OPTS   ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/api_reference/_autosummary"

live:
	sphinx-autobuild --watch ../../src "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

strict:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(STRICT_OPTS)

.PHONY: help clean live strict Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
