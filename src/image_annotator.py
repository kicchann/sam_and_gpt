from typing import Any, Dict, List, Optional

import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class ImageAnnotator:
    def __init__(
        self,
        mask_generator: Optional[SamAutomaticMaskGenerator] = None,
    ):
        if mask_generator is not None:
            self._mask_generator: SamAutomaticMaskGenerator = mask_generator
        else:
            sam_checkpoint = r"../weights/sam_vit_l_0b3195.pth"
            model_type = "vit_l"

            device = "cuda"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)

            # https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py
            self._mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                # pred_iou_thresh = 0.95,
                # stability_score_thresh=0.95,
                crop_n_layers=0,
                # crop_nms_thresh= 0.9,
                # crop_overlap_ratio=0.9,
            )

    def annotate(
        self,
        image: np.ndarray,
    ) -> List[Dict[str, Any]]:
        anns = self._mask_generator.generate(image)
        return anns
