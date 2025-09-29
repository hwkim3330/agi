#!/bin/bash

# AGI Trinity - Non-Interactive CLI Wrapper
# Automatically bypasses permission prompts and interactive confirmations

set -euo pipefail

# Default timeout for commands
TIMEOUT=${AGI_TIMEOUT:-180}

# Command to execute
CMD=("$@")

if [ ${#CMD[@]} -eq 0 ]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 1
fi

# Function to run command with various non-interactive strategies
run_noninteractive() {
    local cmd=("$@")

    # Strategy 1: Set environment variables for non-interactive mode
    export DEBIAN_FRONTEND=noninteractive
    export CI=true
    export BATCH_MODE=1
    export AUTO_APPROVE=yes
    export SKIP_PROMPTS=1
    export NO_INTERACTION=1

    # Strategy 2: Add common non-interactive flags based on command
    local enhanced_cmd=()

    case "${cmd[0]}" in
        "claude-code")
            enhanced_cmd+=("${cmd[@]}" "--dangerously-skip-permissions")
            ;;
        "gemini")
            enhanced_cmd+=("${cmd[@]}" "--quiet" "--no-interactive")
            ;;
        "openai")
            enhanced_cmd+=("${cmd[@]}")
            ;;
        "docker")
            enhanced_cmd+=("${cmd[@]}" "--force" "--quiet")
            ;;
        "npm"|"yarn")
            enhanced_cmd+=("${cmd[@]}" "--yes" "--silent")
            ;;
        "pip"|"pip3")
            enhanced_cmd+=("${cmd[@]}" "--quiet" "--disable-pip-version-check")
            ;;
        *)
            enhanced_cmd+=("${cmd[@]}")
            ;;
    esac

    # Strategy 3: Use expect-style automation with yes piping
    # This pipes 'yes' responses to handle any remaining prompts
    timeout "${TIMEOUT}s" bash -c "
        echo 'yes
        y
        Y
        1

        ' | ${enhanced_cmd[*]} 2>&1
    " || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "[nonint.sh] Command timed out after ${TIMEOUT}s" >&2
            exit $exit_code
        fi
        # Don't exit on other errors, let the calling code handle it
        exit $exit_code
    }
}

# Execute with error handling
echo "[nonint.sh] Executing non-interactively: ${CMD[*]}"
run_noninteractive "${CMD[@]}"