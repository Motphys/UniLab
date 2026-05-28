---
sd_hide_title: true
---

# UniLab Documentation

::::{div} landing-hero

:::{div} landing-hero-text

# UniLab Documentation

### Robot learning infrastructure documentation.

Pick a language to continue:

```{raw} html
<div class="landing-language-picker">
  <label class="landing-language-picker__label" for="landing-language-select">Documentation language</label>
  <select class="landing-language-picker__select" id="landing-language-select" aria-label="Documentation language">
    <option value="" selected disabled>Choose a language</option>
    <option value="en/index.html">English</option>
    <option value="zh_CN/index.html">简体中文</option>
  </select>
</div>
<script>
(function () {
  var select = document.getElementById("landing-language-select");
  if (!select) {
    return;
  }
  select.addEventListener("change", function () {
    if (this.value) {
      window.location.href = this.value;
    }
  });
}());
</script>
```

:::

::::

```{toctree}
:hidden:
:caption: Languages

en/index
zh_CN/index
```

```{toctree}
:hidden:
:caption: Shared

adr/README
api_reference/index
glossary
changelog
```
