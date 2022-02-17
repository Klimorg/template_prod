import io
from itertools import product
from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from loguru import logger
from PIL import Image
from skimage.measure import label, regionprops


def get_segmentator() -> rt.InferenceSession:

    providers = ["CPUExecutionProvider"]  # ['CUDAExecutionProvider']
    session = rt.InferenceSession("./models/model_ML_v2.onnx", providers=providers)
    batch_size = session.get_inputs()[0].shape[0]
    img_size_h = session.get_inputs()[0].shape[2]
    channels = session.get_inputs()[0].shape[3]

    logger.info(
        f"Model loaded. Batch_size: {batch_size}, size: {img_size_h}, channels: {channels}.",
    )

    return session


def preprocess(image: Image.Image, stride: int) -> npt.NDArray:

    cropped_images = []

    image = image.resize((1024, 1024))
    width, height = image.size

    grid = list(
        product(
            range(0, height - height % stride, stride),
            range(0, width - width % stride, stride),
        ),
    )
    for idy, idx in grid:
        box = (idx, idy, idx + stride, idy + stride)

        cropped_image = image.crop(box)
        cropped_image = np.asarray(cropped_image).astype("float32") / 255

        cropped_images.append(cropped_image)

    return np.array(cropped_images)


def postprocess(
    inference: List[npt.NDArray[np.float32]],
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray]:

    # compute mask
    predicted_masks = np.argmax(inference[0], axis=-1)

    predicted_masks = (
        predicted_masks.reshape((4, 4, 256, 256))
        .transpose(0, 2, 1, 3)
        .reshape((4 * 256, 4 * 256))
    )

    mask = predicted_masks.astype(np.float64) / 3  # normalize the data to 0 - 1
    mask = 255 * mask  # Now scale by 255
    mask = mask.astype(np.uint8)

    # compute probabilities
    probabilities = np.max(inference[0], axis=-1)

    probabilities = (
        probabilities.reshape((4, 4, 256, 256))
        .transpose(0, 2, 1, 3)
        .reshape((4 * 256, 4 * 256))
    )

    return mask, probabilities


def compute_segmentation_score(
    labeled_mask: np.ndarray,
    probabilities: np.ndarray,
    num_segment: int,
) -> float:
    """Compute the segmentation score of a given segmented zone.

    Given a `labeled_mask` (ie a numpy array with only integer values),
    and the corresponding `probabilities` (a numpy array of the same size where to each
    element of the first correspond a probability, a float in [0,1].), isolate
    the zone of the `labeled_mask` with only the value `num_segment` and compute
    the associated mean probability (`mean_score_segment`) of this segment.

    Args:
        probabilities (np.ndarray): A numpy array of size (H,W) with float values in [0,1].
        labeled_mask (np.ndarray): A numpy array of size (H,W) with integer values.
        num_segment (int): An integer corersponding to the zone we want to isolate.

    Returns:
        float: The mean probability of the isolated zone.
    """

    segment = labeled_mask == num_segment
    score_segment = probabilities * segment
    area_segment = np.sum(score_segment > 0)
    mean_score_segment = np.sum(score_segment) / area_segment

    return mean_score_segment


def draw_detection(
    image_path: str,
    mask: np.ndarray,
    probabilities: np.ndarray,
    img_size_w: int = 1024,
    img_size_h: int = 1024,
    min_probability: float = 0.50,
):
    """Given an image, draw the zone infered from the mask and the probabilities.

    Given an image_path and the corresponding mask and probabilities infered from
    the model, resize the image and draw a rectangle around each detected zone
    which have a mean probability of at least `min_probability` (ie 0.5 here).

    ```python
    mask = mask.astype(np.float64) / 3  # normalize the data to 0 - 1
    mask = 255 * mask  # Now scale by 255
    mask = mask.astype(np.uint8)
    ```
    Transforms the mask in 8bits integer values in [0,255].

    ```python
    _, threshold_levure = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY)
    threshold_moisissure = threshold2 - threshold_levure

    labeled_levure, num_levures = label(threshold_levure, return_num=True)
    props_levure = regionprops(labeled_levure)

    labeled_moisissure, num_moisissures = label(
        threshold_moisissure, return_num=True
    )
    props_moisissure = regionprops(labeled_moisissure)
    ```
    Creates multiples masks where each connected component of infered levures or
    moisissures zones get a different label, which is used to compute the
    mean_probability of this zone.

    We then loop on these zones, compute the mean probability of these ones, and
    if `mean probability > min_probability`, we draw it, otherwise we discard it as a
    false positive.

    Args:
        image_path (str): Path from where we have to load the image.
        mask (np.ndarray): A numpy array of size (H,W) with integer values.
        probabilities (np.ndarray): A numpy array of size (H,W) with float values in [0,1].
        img_size_w (int, optional): Width to which resize the loaded image. Defaults to 1024.
        img_size_h (int, optional): Height to which resize the loaded image. Defaults to 1024.
    """

    # draw detection results in resize image
    img = cv2.imread(image_path)
    img = cv2.resize(
        img,
        (img_size_w, img_size_h),
        interpolation=cv2.INTER_LINEAR,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_name = image_path.split("/")[-1]

    true_levures = 0
    true_moisissures = 0
    mean_scores_levure = []
    mean_scores_moisissure = []

    _, threshold_levure = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY)
    threshold_moisissure = threshold2 - threshold_levure

    labeled_levure, num_levures = label(threshold_levure, return_num=True)
    props_levure = regionprops(labeled_levure)

    labeled_moisissure, num_moisissures = label(threshold_moisissure, return_num=True)
    props_moisissure = regionprops(labeled_moisissure)

    if num_levures != 0:
        for prop, idx_segment in zip(props_levure, range(num_levures)):
            mean_score = compute_segmentation_score(
                labeled_levure,
                probabilities,
                idx_segment + 1,
            )
            if mean_score > min_probability:
                true_levures += 1
                mean_scores_levure.append(mean_score)
                img = cv2.rectangle(
                    img,
                    (prop.bbox[1], prop.bbox[0]),
                    (prop.bbox[3], prop.bbox[2]),
                    (0, 255, 0),
                    2,
                )
    if num_moisissures != 0:
        for prop, idx_segment in zip(props_moisissure, range(num_moisissures)):
            mean_score = compute_segmentation_score(
                labeled_moisissure,
                probabilities,
                idx_segment + 1,
            )
            if mean_score > min_probability:
                true_moisissures += 1
                mean_scores_moisissure.append(mean_score)
                img = cv2.rectangle(
                    img,
                    (prop.bbox[1], prop.bbox[0]),
                    (prop.bbox[3], prop.bbox[2]),
                    (0, 0, 255),
                    2,
                )

    logger.info(f"{true_levures} levures trouvées.")
    levure_confidence = float(np.mean(mean_scores_levure)) if true_levures != 0 else 1
    logger.info(
        "Probabilité moyenne pour les levures : " f"{levure_confidence * 100:.2f}",
    )
    logger.info(f"{true_moisissures} moisissures trouvées.")
    moisissure_confidence = (
        float(np.mean(mean_scores_moisissure)) if true_moisissures != 0 else 1
    )
    logger.info(
        "Probabilité moyenne pour les moisissures : "
        f"{moisissure_confidence * 100:.2f}",
    )
    report = {
        "img_name": img_name,
        "model": "ML",
        "levures": true_levures,
        "mean_score_l": levure_confidence,
        "moisissures": true_moisissures,
        "mean_score_m": moisissure_confidence,
    }
    logger.info(f"{report}")

    new_img_path = image_path.replace(".jpeg", ".predicted.jpeg")
    cv2.imwrite(new_img_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return (
        true_levures,
        levure_confidence,
        true_moisissures,
        moisissure_confidence,
    )


def get_segments(session, binary_image):

    input_image = Image.open(binary_image.file).convert("RGB")

    preprocessed_image = preprocess(input_image, 256)
    logger.info(
        f"Preprocessing : {preprocessed_image.shape}, {type(preprocessed_image)}",
    )

    predicted_masks = session.run(["activation"], {"input": preprocessed_image})
    logger.info(f"Prediction1 : {type(predicted_masks)}")
    logger.info(f"Prediction2 : {predicted_masks[0].shape}, {type(predicted_masks[0])}")

    postprocessed_masks, postprocessed_probabilities = postprocess(predicted_masks)

    # return Image.fromarray(postprocessed_masks).resize((1024, 1024))

    return postprocessed_masks
