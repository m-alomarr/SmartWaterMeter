import cv2
from ultralytics import YOLO


class WaterMeterDetector:
    """
    YOLO tabanlı su sayacı bölge tespit modülü.

    Görevi SADECE konum bulmak: fotoğrafta 'su_sayaci' (tüm sayaç
    gövdesi), 'endeks' (dijital tüketim penceresi) ve 'seri_no' (seri
    numarası bölgesi) nerede, onu bulup kutu (bbox) döndürür.

    Rakamları OKUMA işini bu sınıf yapmaz - o iş OCREngine'e ait.
    """

    def __init__(self, model_path: str = "models/best.pt",
                 conf_threshold: float = 0.25, imgsz: int = 640):
        # imgsz, train.py'deki eğitim çözünürlüğüyle AYNI olmalı.
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def _extract(self, results, offset=(0, 0)) -> dict:
        ox, oy = offset
        found = {"su_sayaci": [], "endeks": [], "seri_no": []}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name not in found:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            found[cls_name].append({
                "bbox": [ox + x1, oy + y1, ox + x2, oy + y2],
                "confidence": conf,
            })

        return found

    def detect(self, image_input) -> dict:
        """
        Dönüş: {"su_sayaci": [...], "endeks": [...], "seri_no": [...]}
        Her eleman: {"bbox": [x1,y1,x2,y2], "confidence": float}
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            if img is None:
                raise FileNotFoundError(f"Görsel okunamadı: {image_input}")
        else:
            img = image_input.copy()

        results = self.model(img, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)[0]
        detections = self._extract(results)

        # Eksik sınıflar için, bulunan sayaç bölgesinde ikinci bir arama yap
        missing = [c for c in ("endeks", "seri_no") if not detections[c]]
        if detections["su_sayaci"] and missing:
            best_meter = max(detections["su_sayaci"], key=lambda x: x["confidence"])
            x1, y1, x2, y2 = best_meter["bbox"]
            cropped = img[y1:y2, x1:x2]

            if cropped.size > 0:
                sub_conf = max(0.10, self.conf_threshold * 0.6)
                sub_results = self.model(cropped, imgsz=self.imgsz, conf=sub_conf, verbose=False)[0]
                sub_detections = self._extract(sub_results, offset=(x1, y1))
                for cls_name in missing:
                    detections[cls_name].extend(sub_detections[cls_name])

        return detections

    def get_best(self, image_input) -> dict:
        """Her sınıf için en yüksek güvenli TEK tespiti döndürür (ya da None)."""
        detections = self.detect(image_input)
        return {
            cls_name: (max(items, key=lambda x: x["confidence"]) if items else None)
            for cls_name, items in detections.items()
        }