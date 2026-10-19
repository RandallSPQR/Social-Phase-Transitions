#!/usr/bin/env bash
# Clean-room FULL run: image built from 'git archive' of commit 9e37858; data fetched from HF inside.
set -o pipefail
for i in 1 2 3 4 5 6; do
  docker build -q --network host --build-arg HTTPS_PROXY=http://127.0.0.1:41609 --build-arg HTTP_PROXY=http://127.0.0.1:41609 --secret id=ca,src=/root/.ccr/ca-bundle.crt -t spt-replication:full <scratch>/cleanfull/src && break
  echo "build failed (attempt $i)"; sleep $((i*90))
done
echo "commit 9e378582497f16d6ba503df0eb5f0be28685230e"; date -u
docker run --rm --name spt-clean-full --network host -e HF_TOKEN -e HTTPS_PROXY -e HTTP_PROXY \
  -e SSL_CERT_FILE=/ca.crt -e REQUESTS_CA_BUNDLE=/ca.crt -v /root/.ccr/ca-bundle.crt:/ca.crt:ro \
  -v <scratch>/cleanfull/out:/repo/replication/out spt-replication:full python replication/replicate.py --mode full
echo "exit $?"; date -u
