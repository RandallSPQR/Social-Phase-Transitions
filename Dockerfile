# syntax=docker/dockerfile:1.7
# Pinned environment for the tier-1 replication (no API keys, no GPU).
#
#   docker build -t spt-replication .
#   docker run --rm -v "$PWD/replication/out:/repo/replication/out" spt-replication                     # fast mode
#   docker run --rm -v "$PWD/replication/out:/repo/replication/out" spt-replication \
#       python replication/replicate.py --mode full                                                      # full mode
#
# Behind a TLS-intercepting proxy, pass its CA certificate as a build secret (used by pip only):
#   docker build --secret id=ca,src=/path/to/proxy-ca.crt -t spt-replication .
# (the standard HTTP_PROXY/HTTPS_PROXY build arguments are honoured and are not stored in the image).

# python:3.13-slim (Debian trixie), pinned by digest
FROM python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f

# No system packages are installed: ffmpeg (for the MP4/GIF illustrations) comes pinned from PyPI (imageio-ffmpeg,
# in the lock file) and replication/replicate.py puts it on PATH.

WORKDIR /repo
COPY replication/requirements.lock replication/requirements.lock
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir --require-hashes -r replication/requirements.lock

# Code and committed results. The response cache and simulation units are fetched (and SHA-256-verified) at run
# time from the Hugging Face dataset, so the image stays small and the check covers the download.
COPY . /repo
# The repository is not a git checkout inside the image; nothing in the replication needs git.
ENV MPLBACKEND=Agg PYTHONHASHSEED=0 SPT_OFFLINE=1 NUMBA_CACHE_DIR=/tmp/numba
CMD ["python", "replication/replicate.py", "--mode", "fast"]
