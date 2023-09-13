import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import psutil
import torch
from mo_dots import to_data

import deepMask.app.vnet as vnet
from deepMask.app.utils.data import *
from deepMask.app.utils.deepmask import *
from deepMask.app.utils.image_processing import noelImageProcessor


def preprocess_image(id_, t1_fname, t2_fname, indir_,outdir_, preprocess, use_gpu):
    # set up parameters
    outdir = os.path.join(outdir_, id_,'preproc')
    os.makedirs(outdir, exist_ok=True)
    
    # tmpdir = os.path.join(outdir, id_, "tmp")

    # os.makedirs(tmpdir,exist_ok=True)

    t1 = os.path.join(indir_, id_, "anat", t1_fname)
    t2 = os.path.join(indir_, id_, "anat", t2_fname)
    args = to_data({})  # this is really dumb but the code needs it...
    args.seed = 666

    cwd = os.path.dirname(__file__)

    # trained weights based on manually corrected masks from
    # 153 patients with cortical malformations
    args.inference = os.path.join(
        cwd, "deepMask/app/weights", "vnet_masker_model_best.pth.tar"
    )
    # resize all input images to this resolution matching training data
    args.resize = (160, 160, 160)
    args.cuda = torch.cuda.is_available() and use_gpu
    torch.manual_seed(args.seed)
    args.device_ids = list(range(torch.cuda.device_count()))
    # args.tmpdir = tmpdir
    args.outdir = outdir
    
    mem_size = psutil.virtual_memory().available // (
        1024 * 1024 * 1024
    )  # available RAM in GB
    # mem_size = 32
    if mem_size < 64 and not use_gpu:
        os.environ["BRAIN_MASKING"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        model = None
    else:
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            print("build vnet, using GPU")
        else:
            print("build vnet, using CPU")
        model = vnet.build_model(args)

    template = os.path.join(
        cwd, "deepMask/app/template", "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
    )

    # MRI pre-processing configuration
    output_suffix = "_brain_final.nii.gz"

    noelImageProcessor(
        id=id_,
        t1=t1,
        t2=t2,
        output_suffix=output_suffix,
        output_dir=outdir,
        template=template,
        usen3=True,
        args=args,
        model=model,
        preprocess=preprocess,
    ).pipeline()
        

if __name__ == "__main__":
    # configuration
    # parse command line arguments

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i", "--id", dest="id", default="FCD_123", help="Alphanumeric patient code"
    )
    parser.add_argument(
        "-t1",
        "--t1_fname",
        dest="t1_fname",
        default="t1.nii.gz",
        help="T1-weighted image",
    )
    parser.add_argument(
        "-t2",
        "--t2_fname",
        dest="t2_fname",
        default="t2.nii.gz",
        help="T2-weighted image",
    )
    parser.add_argument(
        "-i",
        "--indir",
        dest="indir",
        default="data/",
        help="Directory containing the input images",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        dest="outdir",
        default="data/",
        help="Directory containing the input images",
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        dest="preprocess",
        action="store_true",
        help="Co-register and perform non-uniformity correction of input images",
    )
    parser.add_argument(
        "-g",
        "--use_gpu",
        dest="use_gpu",
        action="store_true",
        help="Compute using GPU, defaults to using CPU",
    )
    args = parser.parse_args()

    preprocess_image(
        id_= args.id,
        t1_fname_= args.t1_fname,
        t2_fname_= args.t2_fname,
        indir_= args.indir,
        outdir_= args.outdir,
        preprocess_= args.preprocess,
        use_gpu_= args.use_gpu,
    )
