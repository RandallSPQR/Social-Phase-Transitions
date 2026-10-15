#!/usr/bin/env bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  if docker build -q --network host --build-arg HTTPS_PROXY=http://127.0.0.1:41609 --build-arg HTTP_PROXY=http://127.0.0.1:41609 --secret id=ca,src=/root/.ccr/ca-bundle.crt -t spt-replication:clean <scratch>/clean/src; then
    echo "build ok (attempt $i)"; exec <scratch>/clean/run_fast.sh
  fi
  echo "build failed (attempt $i); retrying in $((i*90)) s"; sleep $((i*90))
done
echo "build failed 10 times"; exit 1
