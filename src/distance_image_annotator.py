from typing import Any, Dict, List

import cv2
import numpy as np


class DistanceImageAnnotator:
    # 画像の距離画像を使ってannotationする
    def __init__(self, anns: List[Dict[str, Any]]):
        self._anns: List[Dict[str, Any]] = anns

    @staticmethod
    def _distance_transform(mask_array: np.ndarray) -> np.ndarray:
        """_distance_transform

        L1距離を使用して距離画像を取得する．

        Args:
            mask_array (np.array): マスク画像

        Returns:
            np.array: 距離画像
        """
        mask_array = mask_array.astype(np.uint8)
        dist = cv2.distanceTransform(mask_array, cv2.DIST_L1, 5)
        return dist

    def _get_max_distance_coord_from_mask(self, mask_array: np.ndarray) -> tuple:
        """_get_max_distance_coord_from_mask

        距離画像を使って，最大距離の座標を取得する

        Args:
            mask_array (np.array): マスク画像

        Returns:
            Tuple[int, int]: 最大距離の座標 (y, x)
        """
        dist = self._distance_transform(mask_array)
        # 四隅の座標を取得
        # 左上，右上，左下，右下
        corners = [
            (0, 0),
            (0, mask_array.shape[1] - 1),
            (mask_array.shape[0] - 1, 0),
            (mask_array.shape[0] - 1, mask_array.shape[1] - 1),
        ]
        # 最大値を取得
        max_dist = np.max(dist)
        # 最大値の0.9倍以上の座標を取得
        idxs = np.where(dist > max_dist * 0.9)
        # 最大値が複数ある場合は，四隅の座標との距離のばらつきが最も小さい座標を取得
        # 取得できる座標が画像の真ん中に寄る
        min_diff = np.inf
        max_idx: tuple = ()
        for i in range(len(idxs[0])):
            idx = (idxs[0][i], idxs[1][i])
            dists = [
                np.linalg.norm(np.array(idx) - np.array(corner)) for corner in corners
            ]
            diff = np.max(dists) - np.min(dists)
            if diff < min_diff:
                min_diff = diff
                max_idx = idx
        return max_idx

    def _get_boundary_from_mask(self, mask_array: np.ndarray, thickness: int):
        """_get_boundary_from_mask

        距離画像から境界線を取得する

        Args:
            mask_array (np.array): マスク画像
            thickness (int): 境界線の太さ.

        Returns:
            np.array: 境界線のマスク画像
        """
        # マスクの距離変換を行い，境界線を取得する
        # 返り値はsegmentationのようにTrue/Falseの行列で返す
        dist = self._distance_transform(mask_array)
        # 距離が1または2の部分を境界線とする
        for dist_ in range(1, thickness + 1):
            if dist_ == 1:
                boundary = dist == dist_
            else:
                boundary = np.logical_or(boundary, dist == dist_)
        return boundary

    def get_max_distance_coordinates(self) -> List[tuple]:
        """get_max_distance_coordinates

        annotationされたmaskの中で，最大距離の座標を取得する

        Returns:
            List[tuple]: 最大距離の座標のリスト (y, x)
        """
        coords = []
        for ann in self._anns:
            mask_array = ann["segmentation"]
            coord = self._get_max_distance_coord_from_mask(mask_array)
            coords.append(coord)
        return coords

    def get_boundaries(self, thickness: int = 3) -> List[np.array]:
        """get_boundaries

        Args:
            thickness (int): 境界線の太さ. Defaults to 3.

        Returns:
            List[np.array]: 境界線のマスク画像のリスト
        """
        boundaries = []
        for ann in self._anns:
            mask_array = ann["segmentation"]
            boundary = self._get_boundary_from_mask(mask_array, thickness)
            boundaries.append(boundary)
        return boundaries
