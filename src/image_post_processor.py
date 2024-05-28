from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import binary_dilation

from .distance_image_annotator import DistanceImageAnnotator
from .utils import draw_text_with_box


# annotationした画像の後処理
class ImagePostProcessor:
    """ImagePostProcessor
    samでannotationした画像に対して後処理を行うクラス
    """

    def __init__(self, image: np.ndarray, anns: List[Dict[str, Any]]):
        self._image = image
        self._anns = anns
        self._distance_image_annotator = DistanceImageAnnotator(anns)
        # seedを固定
        np.random.seed(0)

    def crop_by_bboxes(self, padding: int = 0) -> np.ndarray:
        # annsで指定されたbbox範囲だけを切り出す
        if len(self._anns) == 0:
            return self._image
        sorted_anns = sorted(self._anns, key=(lambda x: x["area"]), reverse=True)
        new_img = np.zeros(
            (self._image.shape[0], self._image.shape[1], 3), dtype=np.uint8
        )
        # new_imgにannsで指定された部分だけをimgからコピーする
        for ann in sorted_anns:
            bbox = ann["bbox"]
            m = np.zeros((self._image.shape[0], self._image.shape[1]), dtype=bool)
            edge_left = max(0, bbox[0] - padding)
            edge_right = min(self._image.shape[1], bbox[0] + bbox[2] + padding)
            esge_top = max(0, bbox[1] - padding)
            edge_bottom = min(self._image.shape[0], bbox[1] + bbox[3] + padding)
            m[esge_top:edge_bottom, edge_left:edge_right] = True
            new_img[m] = self._image[m]
        return new_img

    def crop_by_segmentations(self, padding: int = 0) -> np.ndarray:
        # annsで指定されたセグメンテーション範囲だけを切り出す
        if len(self._anns) == 0:
            return self._image
        sorted_anns = sorted(self._anns, key=(lambda x: x["area"]), reverse=True)
        new_img = np.zeros(
            (self._image.shape[0], self._image.shape[1], 3), dtype=np.uint8
        )
        # new_imgにannsで指定された部分だけをimgからコピーする
        for ann in sorted_anns:
            m = ann["segmentation"]
            if padding > 0:
                # padding分だけTrueを広げる
                m = binary_dilation(m, iterations=padding)
            new_img[m] = self._image[m]
        return new_img

    def get_anns_img(
        self,
        alpha: float = 0.2,
        color: Optional[tuple] = None,
        add_on_image: bool = True,
        add_boundaries: bool = True,
        only_boundaries: bool = False,
        boundary_thickness: int = 3,
        add_numbers: bool = True,
        color_of_number: tuple = (255, 0, 0),
        background_color_of_number: tuple = (0, 0, 0),
        height_of_number: Union[int, str] = "auto",
    ) -> np.ndarray:
        """get_anns_img

        アノテーションを追加した画像を取得する

        Args:
            alpha (float, optional): annotationの透明度. Defaults to 0.5.
            color (Optional[tuple], optional): annotationの色. Defaults to None. Noneの場合はランダムな色を使用
            add_on_image (bool, optional): annotationを元画像に重ねるかどうか. Defaults to True.
            add_boundaries (bool, optional): 境界線を追加するかどうか. Defaults to True.
            only_boundaries (bool, optional): 塗りつぶしをやめ，境界線のみを追加するかどうか. Defaults to False.
            boundary_thickness (int, optional): 境界線の太さ. Defaults to 3.
            add_numbers (bool, optional): annotationに番号を追加するかどうか. Defaults to True.
            height_of_number (Union[int, str], optional): 番号の高さ. Defaults to "auto".

        Returns:
            np.array: annotationを追加した画像
        """
        # annsで指定されたセグメンテーション範囲を画像として取得
        # seedを固定しているので，同じannsであれば同じ画像が得られる
        mask_image = np.zeros(
            (
                self._image.shape[0],
                self._image.shape[1],
                4,
            ),
            dtype=np.uint8,
        )
        if len(self._anns) == 0:
            return mask_image
        else:
            mask_image[:, :, :3] = 1

        sorted_anns = sorted(self._anns, key=(lambda x: x["area"]), reverse=True)
        for ann in sorted_anns:
            m = ann["segmentation"]
            if color is None:
                color_mask = np.concatenate([np.random.random(3) * 255, [alpha * 255]])
            else:
                color_mask = np.concatenate([color, [alpha * 255]])
            color_mask = color_mask.astype(np.uint8)
            mask_image[m] = color_mask
        if add_boundaries:
            mask_image = self.__add_boundaries(
                mask_image, only_boundaries, boundary_thickness
            )
        if add_on_image:
            mask_image = self.__add_on_img(self._image.copy(), mask_image, alpha=alpha)
        if add_numbers:
            mask_image = self.__add_numbers(
                image=mask_image,
                color=color_of_number,
                background_color=background_color_of_number,
                height=height_of_number,
            )
        return mask_image

    def get_non_anns_img(
        self,
        alpha: float = 0.2,
        color: Optional[tuple] = None,
        add_on_image: bool = False,
    ):
        """get_non_anns_img

        アノテーションされていない部分を色付けした画像を取得する

        Args:
            alpha (float, optional): _description_. Defaults to 0.5.
            color (Optional[tuple], optional): _description_. Defaults to None.
            add_on_image (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        # annsで指定されていない部分を色付けした画像を取得
        mask_image = np.zeros(
            (
                self._image.shape[0],
                self._image.shape[1],
                4,
            ),
            dtype=np.uint8,
        )
        if len(self._anns) == 0:
            return mask_image
        else:
            mask_image[:, :, :3] = 1

        sorted_anns = sorted(self._anns, key=(lambda x: x["area"]), reverse=True)
        for ann in sorted_anns:
            m = ann["segmentation"]
            if color is None:
                color_mask = np.concatenate([np.random.random(3) * 255, [alpha * 255]])
            else:
                color_mask = np.concatenate([color, [alpha * 255]])
            color_mask = color_mask.astype(np.uint8)
            mask_image[m] = color_mask
        # img[:,:,3] がalphaの部分を透明にして，０の部分にcolor_maskを当てる
        mask = mask_image[:, :, 3] == 0
        the_other_mask = mask_image[:, :, 3] != 0
        color_mask = np.concatenate([np.random.random(3) * 255, [alpha * 255]])
        mask_image[mask] = color_mask
        mask_image[the_other_mask] = [0, 0, 0, 0]
        if add_on_image:
            return self.__add_on_img(self._image.copy(), mask_image, alpha=alpha)
        return mask_image

    def __add_on_img(
        self,
        image: np.ndarray,
        mask_image: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        # imgとmask_imgをブレンド
        # img: np.array, (H, W, 3)
        # mask_img: np.array, (H, W, 4)
        # return: np.array, (H, W, 3)
        mask = mask_image[:, :, 3] != 0
        image[mask] = image[mask] * (1 - alpha) + mask_image[mask, :3] * alpha
        return image

    def __add_boundaries(
        self,
        mask_image: np.ndarray,
        only_boundaries: bool = False,
        thickness: int = 3,
    ) -> np.ndarray:
        # mask_imgに境界線を追加する
        # return: np.array, (H, W, 4)
        boundaries = self._distance_image_annotator.get_boundaries(thickness)
        b_img = np.zeros((mask_image.shape[0], mask_image.shape[1], 4), dtype=np.uint8)
        for boundary in boundaries:
            # boundaryでTrueの部分だけmask_imgのalphaを1にする
            b_img[boundary] = [0, 0, 0, 255]
        if only_boundaries:
            mask_image[:, :, 3] = b_img[:, :, 3]
        else:
            mask_image[:, :, 3] = np.maximum(mask_image[:, :, 3], b_img[:, :, 3])
        return mask_image

    def __add_numbers(
        self,
        image: np.ndarray,
        color: tuple = (255, 0, 0),
        background_color: tuple = (0, 0, 0),
        height: Union[int, str] = "auto",
    ) -> np.ndarray:
        coords = self._distance_image_annotator.get_max_distance_coordinates()
        for i, coord in enumerate(coords):
            if type(height) == str and height == "auto":
                image = draw_text_with_box(
                    image,
                    str(i),
                    coord,
                    color=color,
                    background_color=background_color,
                    auto_height=True,
                )
            elif type(height) == int:
                image = draw_text_with_box(
                    image,
                    str(i),
                    coord,
                    color=color,
                    background_color=background_color,
                    height=height,
                )
            else:
                image = draw_text_with_box(
                    image,
                    str(i),
                    coord,
                    color=color,
                    background_color=background_color,
                )
        return image
