from itertools import product
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as rt
from loguru import logger
from PIL import Image
from skimage.measure import label, regionprops


class InferenceEngine:
    def __init__(
        self,
        providers: str,
        model: str,
        num_classes: int,
        inference_img_dims: Tuple[int, int],
    ):
        self.providers = (
            ["CPUExecutionProvider"]
            if providers == "cpu"
            else ["CUDAExecutionProvider"]
        )
        self.model = Path(model)
        self.num_classes = num_classes
        self.inference_img_dims = inference_img_dims

        self.engine = rt.InferenceSession(model, providers=providers)

    def preprocess(self, image: Image.Image, stride: int = 256) -> npt.NDArray:

        cropped_images = []

        width, height = self.inference_img_dims[0], self.inference_img_dims[1]
        image = image.resize((width, height))

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
        self,
        inference: List[npt.NDArray[np.float32]],
    ) -> Tuple[npt.NDArray[np.uint8], npt.NDArray]:

        # compute mask
        predicted_masks = np.argmax(inference[0], axis=-1)

        # TODO : switch to einsum
        predicted_masks = (
            predicted_masks.reshape((4, 4, 256, 256))
            .transpose(0, 2, 1, 3)
            .reshape((4 * 256, 4 * 256))
        )

        # mask = predicted_masks.astype(np.float64) / 3  # normalize the data to 0 - 1
        # mask = 255 * mask  # Now scale by 255
        # mask = mask.astype(np.uint8)

        # compute probabilities
        probabilities = np.max(inference[0], axis=-1)

        probabilities = (
            probabilities.reshape((4, 4, 256, 256))
            .transpose(0, 2, 1, 3)
            .reshape((4 * 256, 4 * 256))
        )

        return predicted_masks, probabilities

    def split_by_value(self, inference: npt.NDArray, num_classes: int) -> npt.NDArray:
        height, width = inference.shape[0], inference.shape[1]
        split_mask = np.zeros((height, width))
        split_mask = np.expand_dims(split_mask, axis=-1)
        for idx in range(1, num_classes):
            mask = inference[:, :] == idx
            mask = np.expand_dims(mask, axis=-1)
            split_mask = np.concatenate((split_mask, mask), axis=-1)

        return split_mask

    def compute_segmentation_score(
        self,
        labeled_mask: npt.NDArray,
        probabilities: npt.NDArray,
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
            num_segment (int): An integer corresponding to the zone we want to isolate.

        Returns:
            float: The mean probability of the isolated zone.
        """

        segment = labeled_mask == num_segment
        score_segment = probabilities * segment
        area_segment = np.sum(score_segment > 0)

        return np.sum(score_segment) / area_segment

    def infer(self, image):
        input_image = Image.open(image.file).convert("RGB")

        preprocessed_image = self.preprocess(input_image, 256)

        predicted_masks = self.engine.run(["activation"], {"input": preprocessed_image})

        postprocessed_masks, postprocessed_probabilities = self.postprocess(
            predicted_masks,
        )

        # return Image.fromarray(postprocessed_masks).resize((1024, 1024))

        return postprocessed_masks


# def get_segmentator() -> rt.InferenceSession:

#     providers = ["CPUExecutionProvider"]  # ['CUDAExecutionProvider']
#     session = rt.InferenceSession("./models/model_ML_v2.onnx", providers=providers)
#     batch_size = session.get_inputs()[0].shape[0]
#     img_size_h = session.get_inputs()[0].shape[2]
#     channels = session.get_inputs()[0].shape[3]

#     logger.info(
#         f"Model loaded. Batch_size: {batch_size}, size: {img_size_h}, channels: {channels}.",
#     )

#     return session


def draw_detection(
    image_path: str,
    mask: npt.NDArray,
    probabilities: npt.NDArray,
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
    img = Image.open(image_path)
    img = img.resize((img_size_w, img_size_h), resample=Image.BILINEAR)
    img = np.asarray(img)

    img_name = image_path.split("/")[-1]

    true_levures = 0
    true_moisissures = 0
    mean_scores_levure = []
    mean_scores_moisissure = []

    split_mask = split_by_value(img=mask, num_classes=4)

    # _, threshold_levure = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    # _, threshold2 = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY)
    # threshold_moisissure = threshold2 - threshold_levure

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

    # def draw_image(self, image, response):
    #     """
    # 	Draws on image and saves it.
    # 	:param image: image of type pillow image
    # 	:param response: inference response
    # 	:return:
    # 	"""
    #     draw = ImageDraw.Draw(image)
    #     for bbox in response['bounding-boxes']:
    #         draw.rectangle([bbox['coordinates']['left'], bbox['coordinates']['top'], bbox['coordinates']['right'],
    #                         bbox['coordinates']['bottom']], outline="red")
    #         left = bbox['coordinates']['left']
    #         top = bbox['coordinates']['top']
    #         conf = "{0:.2f}".format(bbox['confidence'])
    #         draw.text((int(left), int(top) - 20), str(conf) + "% " + str(bbox['ObjectClassName']), 'red', self.font)
    #     image.save('/main/result.jpg', 'PNG')

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
    pass
