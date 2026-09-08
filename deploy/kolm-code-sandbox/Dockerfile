# kolm code sandbox container image (spec section 11.2).
#
# The Cloudflare sandbox base image already carries a POSIX userland; git is
# installed defensively because the ladder cannot run without it and the base
# image contents are not contractual.
FROM docker.io/cloudflare/sandbox:0.12.9

RUN command -v git >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*)

# Candidate worktrees live under /workspace/<runId>/<candidateId>.
RUN install -d -m 0755 /workspace

# `git worktree add` writes an absolute path into .git; keep the layout stable.
WORKDIR /workspace

# Deterministic, non-interactive git for every command the worker issues.
ENV GIT_TERMINAL_PROMPT=0 \
    GIT_CONFIG_COUNT=3 \
    GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0=* \
    GIT_CONFIG_KEY_1=core.autocrlf GIT_CONFIG_VALUE_1=false \
    GIT_CONFIG_KEY_2=advice.detachedHead GIT_CONFIG_VALUE_2=false
