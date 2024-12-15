set export

ACCOUNT := "noelmni"
SERVICE := "deep-fcd"
IMAGE := ACCOUNT + "/" + SERVICE
TAG := "latest"
UID := "2551"
GID := "618"
CASE_ID := "PLO_JUL"
TMPDIR := "/host/hamlet/local_raid/data/ravnoor/sandbox"
PRED_DIR := "/host/hamlet/local_raid/data/ravnoor/sandbox/pytests"
BRAIN_MASKING := "1"
PREPROCESS := "1"

# build Docker image
build:
  docker build -t {{ACCOUNT}}/{{SERVICE}}:{{TAG}} .

# build Docker image without cache
clean-build:
  docker build -t {{ACCOUNT}}/{{SERVICE}}:{{TAG}} . --no-cache

create-locked-pyenv: 
  micromamba run -n base conda-lock -f app/environment.yml -p linux-64 --lockfile app/conda-lock.yml

install-pyenv: create-locked-pyenv
  micromamba create --prefix ${MAMBA_ROOT_PREFIX}/envs/fcd -f app/conda-lock.yml --yes

# run test pipeline
test-pipeline:
    ./app/inference.py {{CASE_ID}} T1.nii.gz FLAIR.nii.gz {{TMPDIR}} cuda0 {{BRAIN_MASKING}} {{PREPROCESS}}

# run scalene profiling
scalene-profiling:
    python3 -m scalene --cpu --gpu --memory ./app/inference.py {{CASE_ID}} T1.nii.gz FLAIR.nii.gz {{TMPDIR}} cuda0 1 1

# run memray profiling
memray-profiling:
    python3 -m memray run ./app/inference.py {{CASE_ID}} t1_brain.nii.gz t2_brain.nii.gz {{TMPDIR}} cuda0 0 0

# run memray profiling on CPU
memray-profiling-cpu:
    python3 -m memray run ./app/inference.py {{CASE_ID}} t1_brain.nii.gz t2_brain.nii.gz {{TMPDIR}} cpu 0 0

# test preprocessing
test-preprocess:
    ./app/preprocess.sh {{CASE_ID}} t1.nii.gz flair.nii.gz {{TMPDIR}} {{BRAIN_MASKING}} {{PREPROCESS}}

# test pipeline in Docker
test-pipeline-docker:
    docker run --rm -it --init \
    --gpus=all \
    --user="{{UID}}:{{GID}}" \
    --volume="{{TMPDIR}}:/tmp" \
    {{ACCOUNT}}/{{SERVICE}}:{{TAG}} \
    /app/inference.py {{CASE_ID}} T1.nii.gz FLAIR.nii.gz /tmp cuda0 {{BRAIN_MASKING}} {{PREPROCESS}}

# test pipeline in Docker with CI testing
test-pipeline-docker_ci:
    docker run --rm -it --init \
    --gpus=all \
    --user="{{UID}}:{{GID}}" \
    --volume="{{TMPDIR}}:/tmp" \
    --env CI_TESTING=1 \
    --env CI_TESTING_GT=/tmp/{{CASE_ID}}/label_final_MD.nii.gz \
    {{ACCOUNT}}/{{SERVICE}}:{{TAG}} \
    /app/inference.py {{CASE_ID}} T1.nii.gz FLAIR.nii.gz /tmp cuda0 {{BRAIN_MASKING}} {{PREPROCESS}}

# test pipeline in Docker for testing
test-pipeline-docker_testing:
    docker run --rm -it --init \
    --gpus=all \
    --user="{{UID}}:{{GID}}" \
    --volume="{{PRED_DIR}}:/tmp" \
    --env CI_TESTING=1 \
    --env CI_TESTING_PATIENT_ID={{CASE_ID}} \
    --env CI_TESTING_PRED_DIR=/tmp \
    {{ACCOUNT}}/{{SERVICE}}:{{TAG}} \
    bash /tests/run_tests.sh

# test reporting
test-reporting:
    ./app/utils/reporting.py {{CASE_ID}} {{TMPDIR}}/

# install Jupyter kernel
install-jupyter-kernel:
    python -m ipykernel install --user --name deepFCD

# clean up temporary files
clean:
    rm -rf {{TMPDIR}}/{{CASE_ID}}/{tmp,native,transforms}
    rm -f {{TMPDIR}}/{{CASE_ID}}/{*_final,*denseCrf3d*,*_native,*_maskpred}.nii.gz

# clean up Docker container
docker-clean:
    docker run --rm -it --init \
    --volume="{{TMPDIR}}:/tmp" \
    busybox:latest \
    rm -rf /tmp/{{CASE_ID}}/{tmp,native,transforms,noel_deepFCD_dropoutMC} && \
    rm -f /tmp/{{CASE_ID}}/{*_final,*denseCrf3d*,*_native,*_maskpred}.nii.gz

# prune Docker images
prune:
    docker image prune

# build runner
runner-build:
    docker-compose -f ci/runner.docker-compose.yml build

# show runner processes
runner-ps:
    docker-compose -f ci/runner.docker-compose.yml ps

# start runner
runner-up:
    docker-compose -f ci/runner.docker-compose.yml up --remove-orphans -d

# stop runner
runner-down:
    docker-compose -f ci/runner.docker-compose.yml down

# show runner logs
runner-logs:
    docker-compose -f ci/runner.docker-compose.yml logs -f

# scale runner
runner-scale:
    docker-compose -f ci/runner.docker-compose.yml up --scale runner=1 -d

# access runner bash
runner-bash:
    docker-compose -f ci/runner.docker-compose.yml exec -it runner bash
