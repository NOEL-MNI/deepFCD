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
T1=${OUTDIR}/${T1_FNAME}        # args.t1 = os.path.join(args.outdir, args.t1_fname)
T2=${OUTDIR}/${T2_FNAME}        # args.t2 = os.path.join(args.outdir, args.t2_fname)

PWD=$(dirname "$0")

eval "$(conda shell.bash hook)"
conda activate preprocess
# echo $CONDA_PREFIX

if [ ${PREPROCESSING} -eq 1 ] && [ ${USE_GPU} -eq 0 ]; then
	python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR} --preprocess
elif [ ${PREPROCESSING} -eq 0 ] && [ ${USE_GPU} -eq 1 ]; then	
	python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR} --use_gpu
elif [ ${PREPROCESSING} -eq 1 ] && [ ${USE_GPU} -eq 1 ]; then
	python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR} --preprocess --use_gpu
else
    python3 $PWD/preprocess.py -i ${ID} -t1 ${T1_FNAME} -t2 ${T2_FNAME} -d ${BASEDIR}
fi

conda deactivate