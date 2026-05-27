#compdef uv
# Zsh completion for UniLab commands launched through `uv run`.

_unilab_uv_complete() {
    if (( ${#words[@]} < 2 )) || [[ ${words[1]} != "uv" || ${words[2]} != "run" ]]; then
        return 0
    fi

    local output
    output="$(
        uv run --no-sync unilab-complete --cword "$((CURRENT - 1))" -- "${words[@]}" 2>/dev/null \
            || uv run --no-sync python -m unilab.tools.completion --cword "$((CURRENT - 1))" -- "${words[@]}" 2>/dev/null
    )" || return 0

    [[ -n "$output" ]] || return 0

    local -a candidates
    candidates=("${(@f)output}")
    compadd -- "${candidates[@]}"
}

compdef _unilab_uv_complete uv
