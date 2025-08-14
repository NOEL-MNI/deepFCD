import os
import multiprocessing
import logging

def set_theano_flags(device):
    """
    Set THEANO_FLAGS environment variable based on the device argument.
    """
    if device.startswith("cuda1"):
        os.environ["THEANO_FLAGS"] = (
            "mode=FAST_RUN,device=cuda1,floatX=float32,dnn.enabled=False"
        )
    elif device.startswith("cpu"):
        cores = str(multiprocessing.cpu_count() // 2)
        var = os.getenv("OMP_NUM_THREADS", cores)
        try:
            logging.info("# of threads initialized: {}".format(int(var)))
        except ValueError:
            raise TypeError(
                "The environment variable OMP_NUM_THREADS"
                " should be a number, got '%s'." % var
            )
        os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,openmp=True,floatX=float32"
    else:
        os.environ["THEANO_FLAGS"] = (
            "mode=FAST_RUN,device=cuda0,floatX=float32,dnn.enabled=False"
        )