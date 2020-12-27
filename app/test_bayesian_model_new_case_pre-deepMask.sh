#!/bin/bash
# source /data/noel/noel7/ravnoor/noelsoft/path_minc2.sh

export PATH=/host/hamlet/local_raid/data/ravnoor/01_Projects/12_deepMask/src/noel_CIVET_masker:$PATH

host=`hostname`
echo $host

USE_GPU=1

ID=$1
INDIR=$2
OUTDIR=$2
# OUTDIR=/host/silius/local_raid/ravnoor/sandbox
# CIVET_MASK=$3
# OUTDIR=$4

MOD_IDX=( t1 flair )

DATADIR=${OUTDIR}
# DATADIR=${OUTDIR}/${ID}
# for MOD in "${MOD_IDX[@]}"
# do
# 	if [ ! -d ${OUTDIR}/${ID} ]; then
# 		mkdir ${OUTDIR}
# 		mkdir ${OUTDIR}/${ID}
# 	fi
# 	# DATADIR=${OUTDIR}/${ID}
# 	XMOD=`echo "${MOD}" | tr '[:lower:]' '[:upper:]'`
# 	# minccalc -clobber -expr "if(A[0]==0){out=0;}else{out=A[1];}" ${CIVET_MASK} ${INDIR}/${ID}_${MOD}_final.mnc.gz ${DATADIR}/${XMOD}.mnc
#
# 	# /data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert -i ${DATADIR}/${XMOD}.mnc -o ${DATADIR}/${XMOD}.nii
# 	/data/noel/noel2/local/brainvisa-Mandriva-2008.0-x86_64-4.1.0-2011_05_16/bin/AimsFileConvert -i ${INDIR}/${ID}_${MOD}_final.mnc.gz -o ${DATADIR}/${XMOD}.nii
# 	gzip ${DATADIR}/${XMOD}.nii
# 	rm -f ${DATADIR}/*.minf
#
# done

# conda activate torch3
# deepMask-run_postCIVET.py ${ID} T1.nii.gz FLAIR.nii.gz ${DATADIR} ${DATADIR}
# conda deactivate

# fsl5.0-fslmaths ${DATADIR}/T1.nii.gz -mul ${DATADIR}/vnet.masker.20180316_0441/${ID}_denseCrf3dSegmMap.nii.gz ${DATADIR}/T1_stripped.nii.gz
# fsl5.0-fslmaths ${DATADIR}/FLAIR.nii.gz -mul ${DATADIR}/vnet.masker.20180316_0441/${ID}_denseCrf3dSegmMap.nii.gz ${DATADIR}/FLAIR_stripped.nii.gz
fsl5.0-fslmaths ${DATADIR}/T1.nii.gz -mul ${DATADIR}/${ID}_denseCrf3dSegmMap.nii.gz ${DATADIR}/T1_stripped.nii.gz
fsl5.0-fslmaths ${DATADIR}/FLAIR.nii.gz -mul ${DATADIR}/${ID}_denseCrf3dSegmMap.nii.gz ${DATADIR}/FLAIR_stripped.nii.gz

# conda activate keras-theano-py3
# python deepFCD-run_postCIVET.py ${ID} ${DATADIR} T1_stripped.nii.gz FLAIR_stripped.nii.gz ${USE_GPU} 2>&1 | tee -a logs/${ID}_deepFCD.log
# python test_bayesian_model_new_case.py
