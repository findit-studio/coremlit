"""SCRFD-10GF (``det_10g.onnx``) detection + 5-point landmarks, in numpy and PIL.

**This runs only to CUT FIXTURES.** No gate depends on it, nothing in the crate imports
it, and ``det_10g.onnx`` is never converted or published — it is here because the five
landmarks the ArcFace template is defined against have to come from somewhere, and taking
them from the detector InsightFace itself pairs with ``w600k_r50`` is the one choice that
introduces no new alignment.

Ported from ``python-package/insightface/model_zoo/scrfd.py`` at ``INSIGHTFACE_REV``,
without OpenCV. Two deliberate divergences from that file, both recorded because they move
pixels:

* **resize.** ``SCRFD.detect`` calls ``cv2.resize`` (``INTER_LINEAR``); this uses PIL's
  ``BILINEAR``. The two do not agree byte for byte. It does not matter here: the resize
  feeds the DETECTOR, whose output is five approximate landmarks, and both the ONNX and
  the CoreML arm are then handed the SAME aligned crop. A landmark that moves by a
  fraction of a pixel moves both arms identically and cancels out of every parity and
  known-pairs number this recipe reports.
* **letterbox.** Same top-left paste into a zero canvas as upstream, spelled with numpy.

The alignment that follows detection is NOT ported here — it is ``align_oracle.py``, the
oracle the committed alignment golden is cut with, so the fixtures and the golden are one
specification.
"""
import numpy as np

#: SCRFD's own preprocessing. NOT ArcFace's: the detector divides by 128, the recogniser by
#: 127.5. Two divisors in one pack is exactly the trap issue #115's census names.
INPUT_MEAN = 127.5
INPUT_STD = 128.0
FEAT_STRIDE_FPN = (8, 16, 32)
NUM_ANCHORS = 2
FMC = 3
NMS_THRESH = 0.4
DET_THRESH = 0.5


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def _nms(dets, thresh=NMS_THRESH):
    x1, y1, x2, y2, scores = (dets[:, i] for i in range(5))
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][np.where(ovr <= thresh)[0]]
    return keep


def letterbox(rgb_u8, size):
    """Upstream's ``detect`` preamble: scale the longest side onto ``size`` preserving
    aspect, paste top-left into a zero canvas. Returns (canvas, det_scale)."""
    from PIL import Image

    h, w = rgb_u8.shape[:2]
    im_ratio = h / w
    model_ratio = size[1] / size[0]
    if im_ratio > model_ratio:
        new_h = size[1]
        new_w = int(new_h / im_ratio)
    else:
        new_w = size[0]
        new_h = int(new_w * im_ratio)
    det_scale = new_h / h
    resized = np.asarray(Image.fromarray(rgb_u8).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.zeros((size[1], size[0], 3), np.uint8)
    canvas[:new_h, :new_w, :] = resized
    return canvas, det_scale


class Detector:
    """``det_10g.onnx``, bound to onnxruntime's CPU provider."""

    def __init__(self, path, det_size=(640, 640)):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise SystemExit(f"det_10g.onnx should expose 9 outputs, got "
                             f"{len(self.output_names)}")
        self.det_size = det_size

    def _blob(self, rgb_u8):
        """``cv2.dnn.blobFromImage(img, 1/128, size, (127.5,)*3, swapRB=True)`` on a BGR
        frame — i.e. RGB in, NCHW out, ``(x - 127.5) / 128``."""
        x = (rgb_u8.astype(np.float32) - INPUT_MEAN) / INPUT_STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])

    def detect(self, rgb_u8, threshold=DET_THRESH):
        """Returns (boxes ``[n, 4]``, scores ``[n]``, kps ``[n, 5, 2]``) in ORIGINAL image
        coordinates, ordered by descending score."""
        canvas, det_scale = letterbox(rgb_u8, self.det_size)
        outs = self.session.run(self.output_names, {self.input_name: self._blob(canvas)})
        h_in, w_in = canvas.shape[:2]

        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(FEAT_STRIDE_FPN):
            scores = outs[idx]
            bbox_preds = outs[idx + FMC] * stride
            kps_preds = outs[idx + FMC * 2] * stride
            height, width = h_in // stride, w_in // stride
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape((-1, 2))
            pos = np.where(scores >= threshold)[0]
            scores_l.append(scores[pos])
            boxes_l.append(_distance2bbox(centers, bbox_preds)[pos])
            kps_l.append(_distance2kps(centers, kps_preds).reshape((-1, 5, 2))[pos])

        scores = np.vstack(scores_l).ravel()
        if scores.size == 0:
            return np.zeros((0, 4), np.float32), scores, np.zeros((0, 5, 2), np.float32)
        boxes = np.vstack(boxes_l) / det_scale
        kps = np.vstack(kps_l) / det_scale
        order = scores.argsort()[::-1]
        pre_det = np.hstack((boxes, scores[:, None])).astype(np.float32)[order]
        keep = _nms(pre_det)
        return pre_det[keep, :4], pre_det[keep, 4], kps[order][keep]
