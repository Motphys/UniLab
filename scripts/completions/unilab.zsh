#compdef uv
# Zsh completion for UniLab commands launched through `uv run`.

_unilab_uv_complete() {
    if (( ${#words[@]} < 2 )) || [[ ${words[1]} != "uv" || ${words[2]} != "run" ]]; then
        return 0
    fi

    local script_path repo_root
    script_path="${${(%):-%x}:A}"
    repo_root="${script_path:h:h:h}"

    local output
    output="$(
        uv run --no-sync unilab-complete --cword "$((CURRENT - 1))" -- "${words[@]}" 2>/dev/null \
            || PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m unilab.tools.completion --cword "$((CURRENT - 1))" -- "${words[@]}" 2>/dev/null
    )" || return 0

    if [[ -z "$output" ]]; then
        if (( CURRENT <= 3 )) || [[ ${words[3]} != "train" && ${words[3]} != "eval" ]]; then
            _files
        fi
        return 0
    fi

    local -a candidates
    candidates=("${(@f)output}")
    compadd -- "${candidates[@]}"
}

compdef _unilab_uv_complete uv
