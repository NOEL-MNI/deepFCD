import ants
import numpy as np

def dilate_labels(label, dilated_label_fname):
  """
    Apply morphological operations to an image

    ANTsR function: `morphology`

    Arguments
    ---------
    input : ANTsImage
        input image

    operation : string
        operation to apply
            "close" Morpholgical closing
            "dilate" Morpholgical dilation
            "erode" Morpholgical erosion
            "open" Morpholgical opening

    radius : scalar
        radius of structuring element

    mtype : string
        type of morphology
            "binary" Binary operation on a single value
            "grayscale" Grayscale operations

    value : scalar
        value to operation on (type='binary' only)

    shape : string
        shape of the structuring element ( type='binary' only )
            "ball" spherical structuring element
            "box" box shaped structuring element
            "cross" cross shaped structuring element
            "annulus" annulus shaped structuring element
            "polygon" polygon structuring element

    radius_is_parametric : boolean
        used parametric radius boolean (shape='ball' and shape='annulus' only)

    thickness : scalar
        thickness (shape='annulus' only)

    lines : integer
        number of lines in polygon (shape='polygon' only)

    include_center : boolean
        include center of annulus boolean (shape='annulus' only)

    Returns
    -------
    ANTsImage

    Example
    -------
    >>> import ants
    >>> fi = ants.image_read( ants.get_ants_data('r16') , 2 )
    >>> mask = ants.get_mask( fi )
    >>> dilated_ball = ants.morphology( mask, operation='dilate', radius=3, mtype='binary', shape='ball')
    >>> eroded_box = ants.morphology( mask, operation='erode', radius=3, mtype='binary', shape='box')
    >>> opened_annulus = ants.morphology( mask, operation='open', radius=5, mtype='binary', shape='annulus', thickness=2)
  """
  label = ants.image_read(label)
  ants.morphology(label, operation='dilate', radius=30, mtype='binary', shape='ball').to_filename(dilated_label_fname)


def compare_images(predicted_image, ground_truth_image, metric_type='correlation'):
  """
    Measure similarity between two images.
    NOTE: Similarity is actually returned as distance (i.e. dissimilarity)
    per ITK/ANTs convention. E.g. using Correlation metric, the similarity
    of an image with itself returns -1.
  """
  # predicted_image = ants.image_read(predicted_image)
  # ground_truth_image = ants.image_read(ground_truth_image)
  if metric_type == 'correlation':
    metric = ants.image_similarity(predicted_image, ground_truth_image, metric_type='ANTSNeighborhoodCorrelation')
    metric = np.abs(metric)
  else:
    metric = ants.label_overlap_measures(predicted_image, ground_truth_image).TotalOrTargetOverlap[1]

  return metric