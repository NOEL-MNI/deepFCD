import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import psutil
import torch
from mo_dots import to_data

from deepMask import vnet
from deepMask.utils.image_processing import noelImageProcessor
import deepMask


def preprocess_image(id_, t1_fname, t2_fname, indir_, outdir_, preprocess, use_gpu):
    # set up parameters
    # Parse subject and session from id_ to create proper BIDS structure
    if "_ses-" in id_:
        # Split subject and session: sub-PX034_ses-02 -> sub-PX034, ses-02
        subject_part, session_part = id_.split("_ses-", 1)
        outdir = os.path.join(outdir_, subject_part, f"ses-{session_part}", "anat")
    else:
        # No session: sub-PX034 -> sub-PX034/anat
        outdir = os.path.join(outdir_, id_, "anat")
    os.makedirs(outdir, exist_ok=True)

    # tmpdir = os.path.join(outdir, id_, "tmp")

    # os.makedirs(tmpdir,exist_ok=True)
    if not os.path.isabs(t1_fname):
        t1 = os.path.join(indir_, id_, "anat", t1_fname)
    else:
        t1 = t1_fname

    if not os.path.isabs(t2_fname):
        t2 = os.path.join(indir_, id_, "anat", t2_fname)
    else:
        t2 = t2_fname

    # Check if input files exist
    if not os.path.exists(t1):
        raise FileNotFoundError(f"T1 file not found: {t1}")
    if not os.path.exists(t2):
        raise FileNotFoundError(f"T2/FLAIR file not found: {t2}")

    args = to_data({})  # this is really dumb but the code needs it...
    args.seed = 666

    # locate package assets inside installed deepMask package
    pkg_dir = os.path.dirname(deepMask.__file__)
    # trained weights based on manually corrected masks
    args.inference = os.path.join(pkg_dir, "weights", "vnet_masker_model_best.pth.tar")
    # resize all input images to this resolution matching training data
    args.resize = (160, 160, 160)
    args.cuda = torch.cuda.is_available() and use_gpu
    torch.manual_seed(args.seed)
    args.device_ids = list(range(torch.cuda.device_count()))
    # args.tmpdir = tmpdir
    args.outdir = outdir
    # temporary working directory expected by deepMask image processor
    args.tmpdir = os.path.join(outdir, "tmp")
    os.makedirs(args.tmpdir, exist_ok=True)

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
        pkg_dir, "template", "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
    )

    # MRI pre-processing configuration
    output_suffix = "_brain.nii.gz"

    try:
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
    except ValueError as e:
        if "images do not occupy same physical space" in str(e):
            print(
                f"Warning: T1 and FLAIR images for {id_} are not in the same physical space."
            )
            print(
                "Attempting to process with T1 only (skipping FLAIR brain masking)..."
            )

            # Try processing with T1 only by setting t2 to None or empty
            try:
                noelImageProcessor(
                    id=id_,
                    t1=t1,
                    t2=None,  # Skip FLAIR processing
                    output_suffix=output_suffix,
                    output_dir=outdir,
                    template=template,
                    usen3=True,
                    args=args,
                    model=model,
                    preprocess=preprocess,
                ).pipeline()
                print(f"Successfully processed {id_} with T1 only.")
            except Exception as e2:
                print(f"Error processing {id_} even with T1 only: {e2}")
                raise e2
        else:
            # Re-raise the original error if it's not the spatial mismatch issue
            raise e


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
        id_=args.id,
        t1_fname=args.t1_fname,
        t2_fname=args.t2_fname,
        indir_=args.indir,
        outdir_=args.outdir,
        preprocess=args.preprocess,
        use_gpu=args.use_gpu,
    )
