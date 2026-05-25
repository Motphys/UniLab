# Bash completion for UniLab commands launched through `uv run`.

_unilab_uv_complete() {
    COMPREPLY=()

    if [[ ${#COMP_WORDS[@]} -lt 2 || ${COMP_WORDS[0]} != "uv" || ${COMP_WORDS[1]} != "run" ]]; then
        return 0
    fi

    local candidates
    if ! mapfile -t candidates < <(
        uv run --no-sync unilab-complete --cword "$COMP_CWORD" -- "${COMP_WORDS[@]}" 2>/dev/null \
            || uv run --no-sync python -m unilab.tools.completion --cword "$COMP_CWORD" -- "${COMP_WORDS[@]}" 2>/dev/null
    ); then
        return 0
    fi

    COMPREPLY=("${candidates[@]}")
}

complete -F _unilab_uv_complete uv
