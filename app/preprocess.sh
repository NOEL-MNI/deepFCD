#!/bin/bash

# preprocess.sh ${ID} ${T1_FNAME} ${T2_FNAME} ${BASEDIR} ${PREPROCESSING} ${USE_GPU}


echo "*************************************************************************"
echo "run stage:$0 $*"
echo "on: `hostname`"
echo "at: `date`"
echo "*************************************************************************"
echo ""
echo ""


ID=$1                           # args.id = sys.argv[1]
T1_FNAME=$2                     # args.t1_fname = sys.argv[2]
T2_FNAME=$3                     # args.t2_fname = sys.argv[3]
BASEDIR=$4                      # args.dir = sys.argv[4]
PREPROCESSING=$5                # args.preprocess = True # co-register T1 and T2 contrasts before brain extraction
USE_GPU=$6                      # args.use_gpu = sys.argv[6]

BRAIN_MASKING=cpu

OUTDIR=${BASEDIR}/${ID}/        # args.outdir = os.path.join(args.dir, args.id)

PWD=$(dirname "$0")

# conditional switching between conda and micromamba
if [ -n "${MAMBA_EXE}" ] && command -v micromamba &> /dev/null; then
    echo "Using micromamba python environment"
    eval "$(micromamba shell hook --shell bash)"
    CONDA_CMD="micromamba"
    # micromamba activate preprocess
elif [ -n "${CONDA_EXE}" ] && command -v conda &> /dev/null; then
    echo "Using conda python environment"
    eval "$(conda shell.bash hook)"
    CONDA_CMD="conda"
    # conda activate preprocess
else
    echo "Error: Neither conda nor micromamba found"
    exit 1
fi

if [ ${PREPROCESSING} -eq 1 ] && [ ${USE_GPU} -eq 0 ]; then
  ${CONDA_CMD} run -n preprocess python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR} --preprocess
elif [ ${PREPROCESSING} -eq 0 ] && [ ${USE_GPU} -eq 1 ]; then	
  ${CONDA_CMD} run -n preprocess python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR} --use_gpu
elif [ ${PREPROCESSING} -eq 1 ] && [ ${USE_GPU} -eq 1 ]; then
  ${CONDA_CMD} run -n preprocess python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR} --preprocess --use_gpu
else
  ${CONDA_CMD} run -n preprocess python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR}
fi

# deactivate environment (works for both conda and micromamba)
# if command -v micromamba &> /dev/null; then
#     micromamba deactivate
# else
#     conda deactivate
# fi