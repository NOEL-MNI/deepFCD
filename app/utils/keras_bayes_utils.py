import numpy as np
import sys, time, collections

def batch_shuffle(index_array, batch_size):
    """Shuffles an array in a batch-wise fashion.
    Useful for shuffling HDF5 arrays
    (where one cannot access arbitrary indices).
    # Arguments
        index_array: array of indices to be shuffled.
        batch_size: integer.
    # Returns
        The `index_array` array, shuffled in a batch-wise fashion.
    """
    batch_count = int(len(index_array) / batch_size)
    # to reshape we need to be cleanly divisible by batch size
    # we stash extra items and reappend them after shuffling
    last_batch = index_array[batch_count * batch_size:]
    index_array = index_array[:batch_count * batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.flatten()
    return np.append(index_array, last_batch)


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).
    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.
    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]


def check_num_samples(ins,
                      batch_size=None,
                      steps=None,
                      steps_name='steps'):
    """Checks the number of samples provided for training and evaluation.
    The number of samples is not defined when running with `steps`,
    in which case the number of samples is set to `None`.
    # Arguments
        ins: List of tensors to be fed to the Keras function.
        batch_size: Integer batch size or `None` if not defined.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.
        steps_name: The public API's parameter name for `steps`.
    # Raises
        ValueError: when `steps` is `None` and the attribute `ins.shape`
        does not exist. Also raises ValueError when `steps` is not `None`
        and `batch_size` is not `None` because they are mutually
        exclusive.
    # Returns
        When steps is `None`, returns the number of samples to be
        processed based on the size of the first dimension of the
        first input numpy array. When steps is not `None` and
        `batch_size` is `None`, returns `None`.
    # Raises
        ValueError: In case of invalid arguments.
    """
    if steps is not None and batch_size is not None:
        raise ValueError(
            'If ' + steps_name + ' is set, the `batch_size` must be None.')

    if not ins or any(K.is_tensor(x) for x in ins):
        if steps is None:
            raise ValueError(
                'If your data is in the form of symbolic tensors, '
                'you should specify the `' + steps_name + '` argument '
                '(instead of the `batch_size` argument, '
                'because symbolic tensors are expected to produce '
                'batches of input data).')
        return None

    if hasattr(ins[0], 'shape'):
        return int(ins[0].shape[0])
    return None  # Edge case where ins == [static_learning_phase]


class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)


def unpack_singleton(x):
    """Gets the first element if the iterable has only one value.
    Otherwise return the iterable.
    # Argument:
        x: A list or tuple.
    # Returns:
        The same iterable or the first element.
    """
    if len(x) == 1:
        return x[0]
    return x


def to_list(x):
    """Normalizes a list/tensor into a list.
    If a tensor is passed, we return
    a list of size 1 containing the tensor.
    # Arguments
        x: target object to be normalized.
    # Returns
        A list.
    """
    if isinstance(x, list):
        return x
    return [x]


def slice_arrays(arrays, start=None, stop=None):
    """Slices an array or list of arrays.
    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list
    Can also work on list/array of indices: `_slice_arrays(x, indices)`
    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.
    # Returns
        A slice of the array(s).
    """
    if arrays is None:
        return [None]
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]

def slice_X(X, start=None, stop=None):
    if type(X) == list:
        if hasattr(start, '__len__'):
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            return X[start]
        else:
            return X[start:stop]

# def make_batches(size, batch_size):
#     nb_batch = int(np.ceil(size/float(batch_size)))
#     return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def numpy_minibatch(numpy_array, batch_size=1, min_batch_size=1):
    """
    Creates a minibatch generator over a numpy array. :func:`minibatch` delegates to this generator
    when the input is a numpy.ndarray.

    Parameters
    ----------
    numpy_array : numpy.ndarray
        A numpy array.
    batch_size : int, optional
        The number of examples to pull from the array as a batch. Default is 1.
    min_batch_size : int, optional
        The minimum number of examples to pull from the iterable. Default is 1.

    Yields
    ------
    numpy array
        A numpy array of the minibatch. It will yield over the first dimension of the input.
    """
    numpy_array = np.asarray(numpy_array)
    assert 0 < min_batch_size <= batch_size, \
        "batch_size (%d) has to be larger than min_batch_size (%d) and they both have to be greater than zero!" % \
        (batch_size, min_batch_size)
    # go through the first dimension of the input array.
    for i in iter(range((numpy_array.shape[0] // batch_size) + 1)):
        idx = i * batch_size
        data = numpy_array[idx:(idx + batch_size)]
        if data.shape[0] >= min_batch_size:
            yield data
