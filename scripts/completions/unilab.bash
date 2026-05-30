# Bash completion for UniLab commands launched through `uv run`.

_unilab_uv_complete() {
    COMPREPLY=()

    if [[ ${#COMP_WORDS[@]} -lt 2 || ${COMP_WORDS[0]} != "uv" || ${COMP_WORDS[1]} != "run" ]]; then
        return 0
    fi

    local script_dir repo_root
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    repo_root="$(cd -- "$script_dir/../.." && pwd -P)"

    local candidates
    if ! mapfile -t candidates < <(
        uv run --no-sync unilab-complete --cword "$COMP_CWORD" -- "${COMP_WORDS[@]}" 2>/dev/null \
            || PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m unilab.tools.completion --cword "$COMP_CWORD" -- "${COMP_WORDS[@]}" 2>/dev/null
    ); then
        return 0
    fi

    if [[ ${#candidates[@]} -eq 0 ]]; then
        if [[ $COMP_CWORD -le 2 || ${COMP_WORDS[2]} != "train" && ${COMP_WORDS[2]} != "eval" ]]; then
            compopt -o default -o bashdefault 2>/dev/null || true
            return 1
        fi
        return 0
    fi

    COMPREPLY=("${candidates[@]}")
}

complete -o default -o bashdefault -F _unilab_uv_complete uv
