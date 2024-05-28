import numpy as np


class AnnotationFilter:
    @staticmethod
    def overlap_ratio(mask1, mask2):
        # 2つのマスクの重なり率を計算する
        # mask1が完全にmask2に含まれる場合，1を返す
        # mask1, mask2: np.array, (H, W)
        # return: float
        return np.sum(mask1 & mask2) / np.sum(mask1)

    def filter_by_overlap_ratio(self, anns, threshold: float = 0.90):
        """filter_by_overlap_ratio

        Args:
            anns (_type_): マスク情報のリスト
            threshold (float): 重なり率の閾値. Defaults to 0.90.

        Returns:
            _type_: 重なり率がthreshold以下のマスク
        """
        new_masks = []
        for i in range(len(anns)):
            mask1 = anns[i]["segmentation"]
            is_overlap = False
            for j in range(len(anns)):
                if i == j:
                    continue
                mask2 = anns[j]["segmentation"]
                if (
                    self.overlap_ratio(mask1, mask2) > threshold
                    and anns[i]["area"] < anns[j]["area"]
                ):
                    is_overlap = True
                    break
            if not is_overlap:
                new_masks.append(anns[i])
        return new_masks

    def filter_by_area_ratio(self, anns, threshold: float = 0.01):
        """filter_by_area_ratio

        Args:
            anns (_type_): マスク情報のリスト
            threshold (float): 面積の割合の閾値. Defaults to 0.01.

        Returns:
            _type_: 面積の割合がthreshold以上のマスク
        """
        new_masks = []
        for mask in anns:
            crop_area = mask["segmentation"].size
            area = mask["area"]
            if area / crop_area > threshold:
                new_masks.append(mask)
        return new_masks
