import numpy as np


class ImageProcessor:
    """Basit görüntü kırpma yardımcı sınıfı."""

    def crop_region(self, image: np.ndarray, bbox: list, padding: int = 0) -> np.ndarray:
        """
        Verilen [x1, y1, x2, y2] koordinatlarına göre görseli kırpar.

        padding: Kutunun etrafına ekstra piksel payı bırakır (pozitif
        değer = daha geniş kırpma). VLM'e bağlamı biraz daha geniş
        vermek istediğimizde kullanışlı; varsayılan 0 = tam kutu.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return image[y1:y2, x1:x2]