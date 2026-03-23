import numpy as np
import PIL.Image
import PIL.ImageDraw
from ultralytics import YOLO

import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as models
from cvat_sdk.masks import encode_mask


# Load your custom YOLO segmentation model
_model = YOLO("yolo26m-seg.pt")

_PERSON_CLASS_ID = next(id for id, name in _model.names.items() if name == "person")

spec = cvataa.DetectionFunctionSpec(
    labels=[
        cvataa.label_spec("person", _PERSON_CLASS_ID),
    ],
)


def detect(
    context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
) -> list[models.LabeledShapeRequest]:
    conf_threshold = 0.3 if context.conf_threshold is None else context.conf_threshold
    predictions = _model.predict(source=image, conf=conf_threshold, classes=[_PERSON_CLASS_ID])

    results = []
    for result in predictions:
        if result.masks is None or result.masks.xy is None:
            continue

        img_w, img_h = image.size
        for label, mask_tensor in zip(result.boxes.cls, result.masks.data):
            if int(label.item()) != _PERSON_CLASS_ID:
                continue
            _mask_np = mask_tensor.cpu().numpy()
            _mask_resized = np.array(
                PIL.Image.fromarray(_mask_np).resize((img_w, img_h), PIL.Image.NEAREST)
            ).astype(bool)
            if not _mask_resized.any():
                continue
            results.append(
                cvataa.mask(
                    label_id=int(label.item()),
                    points=encode_mask(_mask_resized),
                )
            )

    return results