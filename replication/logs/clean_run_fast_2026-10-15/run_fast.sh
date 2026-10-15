#!/usr/bin/env bash
# Clean-room run: image built from 'git archive' of commit 38b926f; data fetched from HF inside.
set -o pipefail
echo "commit 82a2a781dc3c67055e1f6ca88eae17928f418b3b"; echo "image spt-replication:clean"; date -u
docker run --rm --name spt-clean-fast --network host -e HF_TOKEN -e HTTPS_PROXY -e HTTP_PROXY   -e SSL_CERT_FILE=/ca.crt -e REQUESTS_CA_BUNDLE=/ca.crt -v /root/.ccr/ca-bundle.crt:/ca.crt:ro   -v <scratch>/clean/out:/repo/replication/out spt-replication:clean   bash -c 'nproc; free -g | head -2; python replication/replicate.py --mode fast'
echo "exit $?"; date -u
