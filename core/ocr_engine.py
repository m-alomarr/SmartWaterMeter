#ilk version


import json
import re
import traceback

import cv2
import numpy as np
import ollama


class OCREngine:


    def __init__(self, model_name: str = "qwen2.5vl:7b"):
        self.model_name = model_name

    def read_meter(self, image: np.ndarray) -> dict:

        empty = {"endeks": "", "seri_no": ""}

        if image is None or image.size == 0:
            return empty

        image = self._resize_for_speed(image, max_dimension=1024)

        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            return empty
        image_bytes = encoded.tobytes()

        prompt = self._build_prompt()

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [image_bytes],
                }],
                format="json",  # geçerli JSON dışına çıkamaz - döngü/boş cevap koruması
                keep_alive="30m",
                options={          
                    "num_predict": 150,
                    "temperature": 0.1,
                    "num_ctx": 8192,  # Bağlam penceresini 4096'dan 8192'ye çıkardık!
                },
            )
            raw_text = response["message"]["content"].strip()
            print(f"   [DEBUG] Ham cevap: {raw_text!r}")

            parsed = self._parse_json(raw_text)
            print(f"   [DEBUG] Ayrıştırılmış: {parsed}")
            return parsed

        except Exception:
            print("⚠️ OCR (VLM) Okuma Hatası:")
            print("   Ollama kurulu/çalışıyor mu ve "
                  f"'{self.model_name}' indirilmiş mi kontrol et.")
            traceback.print_exc()
            return empty

    @staticmethod
    def _build_prompt() -> str:
        return (
            "Bu bir su sayaci fotografidir. Iki bilgiyi bulup okumalisin.\n\n"

            "1) ENDEKS: Sayacin cam kapagi altindaki, ortadaki kucuk "
            "dikdortgen pencerede gorunen, yan yana dizili SIYAH mekanik "
            "rakamlardir (kilometre sayaci gibi). Genelde 5 hanelidir ve "
            "cogu zaman ilk haneleri sifirdir (ornek: 00142). Pencerenin "
            "yanindaki KIRMIZI kucuk rakamlari veya ibreleri dahil etme.\n\n"

            "2) SERI_NO: Sayacin dis metal (bakir/sari renkli) halkasi "
            "uzerine kabartma olarak islenmis, kavisli dizilmis rakam "
            "dizisidir. Genelde marka adinin (orn. KASKI, BAYLAN) hemen "
            "yaninda/devaminda bulunur. Genelde 7-8 hanelidir, endeksten "
            "daha uzundur.\n\n"

            "Fotograf yan veya ters donmus olabilir, zihninde duz "
            "cevirerek analiz et. Emin olamasan bile en olasi tahminini "
            "ver; sadece o bolge fotografta hic yoksa bos birak.\n\n"

            "Cevabini SADECE su JSON formatinda ver, baska hicbir metin "
            "ekleme:\n"
            '{"endeks": "rakamlar", "seri_no": "rakamlar"}'
        )

    @staticmethod
    def _resize_for_speed(image: np.ndarray, max_dimension: int = 800) -> np.ndarray:
        h, w = image.shape[:2]
        longest_side = max(h, w)
        if longest_side <= max_dimension:
            return image
        scale = max_dimension / longest_side
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _parse_json(raw_text: str) -> dict:
        matches = re.findall(r"\{[^{}]*\}", raw_text)
        for candidate in reversed(matches):
            try:
                data = json.loads(candidate)
                if "endeks" in data or "seri_no" in data:
                    return {
                        "endeks": "".join(ch for ch in str(data.get("endeks", "")) if ch.isdigit()),
                        "seri_no": "".join(ch for ch in str(data.get("seri_no", "")) if ch.isdigit()),
                    }
            except Exception:
                continue

        endeks_match = re.search(r'"endeks"\s*:\s*"(\d*)"', raw_text)
        seri_match = re.search(r'"seri_no"\s*:\s*"(\d*)"', raw_text)
        return {
            "endeks": endeks_match.group(1) if endeks_match else "",
            "seri_no": seri_match.group(1) if seri_match else "",
        }