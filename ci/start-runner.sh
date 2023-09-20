#!/usr/bin/env bash

GH_OWNER=${GH_OWNER}
GH_REPOSITORY=${GH_REPOSITORY}
GH_TOKEN=${GH_TOKEN}

HOSTNAME=$(hostname | cut -d "." -f 1)
RUNNER_SUFFIX=$(cat /dev/urandom | tr -dc 'a-z0-9' | fold -w 7 | head -n 1)
RUNNER_NAME="minion-${HOSTNAME}-${RUNNER_SUFFIX}"

echo ${RUNNER_NAME}

REG_TOKEN=$(curl -sX POST -H "Accept: application/vnd.github+json" -H "Authorization: token ${GH_TOKEN}" https://api.github.com/repos/${GH_OWNER}/${GH_REPOSITORY}/actions/runners/registration-token | jq .token --raw-output)

cd /home/ga/actions-runner

./config.sh --unattended --url https://github.com/${GH_OWNER}/${GH_REPOSITORY} --token ${REG_TOKEN} --name ${RUNNER_NAME}

cleanup() {
    echo "Removing runner..."
    ./config.sh remove --unattended --token ${REG_TOKEN}
}

export PATH=/home/ga/miniconda3/condabin:/home/ga/miniconda3/bin:$PATH

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!