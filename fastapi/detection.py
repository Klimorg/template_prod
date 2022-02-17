import io
import os

import cv2
import numpy as np
import onnxruntime as rt
from loguru import logger
from PIL import Image


def cxcywh2xywh(x):
    # Convert nx4 boxes from [cx, cy, w, h] to [x1, y1, w, h] where xy1=top-left
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 2]  # w
    y[:, 3] = x[:, 3]  # h
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


def get_detector() -> rt.InferenceSession:

    providers = ["CPUExecutionProvider"]
    session = rt.InferenceSession("./models/model_GA_v2.onnx", providers=providers)
    batch_size = session.get_inputs()[0].shape[0]
    img_size_h = session.get_inputs()[0].shape[2]
    img_size_w = session.get_inputs()[0].shape[3]

    logger.info(
        f"Model loaded. Batch_size: {batch_size}, height: {img_size_h}, width: {img_size_w}."
    )

    return session, img_size_h, img_size_w


def preprocess(input_image: np.ndarray, img_size_w: int, img_size_h: int) -> np.ndarray:

    cv2_img = np.ascontiguousarray(input_image)

    image_src = Image.fromarray(cv2_img)

    logger.info(f"Loaded Image Infos. Size: {image_src.size}, mode: {image_src.mode}")

    resized = image_src.resize((img_size_w, img_size_h))

    logger.info(f"After resizing: {resized.size}, {resized.mode}")

    img_in = np.transpose(resized, (2, 0, 1)).astype(np.float32)  # HWC -> CHW
    img_in = np.expand_dims(img_in, axis=0)  # CHW -> NCHW N=1
    img_in /= 255.0

    return img_in


def postprocess(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    nms_thres=0.3,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    # Settings

    min_wh, max_wh = 2, 1024  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    output = [np.zeros((0, 6))] * prediction.shape[0]

    for idx, pred in enumerate(prediction):  # image index, image inference

        # take alls predictions for the given index with a good enough rediction score
        pred = pred[xc[idx]]  # confidence
        if labels and len(labels[idx]):
            l = labels[idx]

            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            pred = np.concatenate((pred, v), 0)

        # If none remain process next image
        if not pred.shape[0]:
            continue

        # Compute conf
        pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (xmin, ymin, w, h)
        box = cxcywh2xywh(pred[:, :4])

        conf, j = pred[:, 5:].max(1, keepdims=True), np.array(
            [[float(0)]] * len(pred[:, 5:].max(1))
        )
        pred = np.concatenate((box, conf, j), 1)[conf.reshape(-1) > conf_thres]
        # Filter by class
        if classes is not None:
            pred = pred[(pred[:, 5:6] == np.array(classes)).any(1)]
        # Check shape
        n = pred.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            pred = pred[pred[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
        # Batched NMS
        c = pred[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = pred[:, :4] + c, pred[:, 4]  # boxes (offset by class), scores

        i = cv2.dnn.NMSBoxes(boxes.tolist(), scores, iou_thres, nms_thres)

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[idx] = pred[i]
    return output


def draw_detection(
    input_image: str, img_size_w: int, img_size_h: int, filtered_predictions: np.ndarray
):

    # draw detection results in resize image
    input_image = np.asarray(input_image)
    logger.info(f"Draw detection : {type(input_image)}")
    img = cv2.resize(
        input_image,
        (img_size_w, img_size_h),
        interpolation=cv2.INTER_LINEAR,
    )

    scores = []
    for pred in filtered_predictions[0]:
        score = pred[0][4]
        scores.append(score)
        pred = xywh2xyxy(pred[:, :4])
        x1 = int(pred[0][0])
        y1 = int(pred[0][1])
        x2 = int(pred[0][2])
        y2 = int(pred[0][3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), thickness=2, color=(0, 255, 0))
        report = {
            "xmin-ymin-w-h": [x1, y1, x2, y2],
            "score": score,
        }
        logger.info(f"{report}")

    if len(filtered_predictions[0]) > 0:

        logger.info(f"{len(filtered_predictions[0])} germes trouvés.")
        logger.info(f"Probabilité minimale : {np.min(scores)*100:.2f}.")
        logger.info(f"Probabilité maximale : {np.max(scores)*100:.2f}.")
        logger.info(f"Probabilité moyenne : {np.mean(scores)*100:.2f}")
    else:
        logger.info("0 germes trouvés")

    return img


def get_detections(session, img_size_w, img_size_h, binary_image):

    input_image = Image.open(io.BytesIO(binary_image))
    logger.info(f"Input : {type(input_image)}")

    img_in = preprocess(input_image, img_size_w, img_size_h)
    logger.info(f"Preprocessing : {img_in.shape}")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})
    logger.info(f"Prediction : {outputs[0].shape}")

    filtered_predictions = postprocess(outputs[0], conf_thres=0.25, iou_thres=0.1)
    logger.info(f"Postprocessing : {len(filtered_predictions[0])}")

    detections = draw_detection(
        input_image, img_size_w, img_size_h, filtered_predictions
    )

    return Image.fromarray(detections).resize((1024, 1024))
