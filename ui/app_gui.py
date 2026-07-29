#version 1

import traceback

import cv2
import customtkinter as ctk
from PIL import Image

from core.detector import WaterMeterDetector
from core.processor import ImageProcessor
from core.ocr_engine import OCREngine


class AppGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("KASKİ - Akıllı Su Sayacı Okuma Sistemi")
        self.geometry("1100x650")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.detector = WaterMeterDetector(model_path="models/best.pt", conf_threshold=0.25)
        self.processor = ImageProcessor()
        self.ocr = OCREngine(model_name="qwen2.5vl:7b")

        self.setup_ui()

    # ------------------------------------------------------------------
    def setup_ui(self):
        self.left_frame = ctk.CTkFrame(self, width=550)
        self.left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.btn_select = ctk.CTkButton(
            self.left_frame,
            text="📁 Fotoğraf Seç ve Analiz Et",
            command=self.select_and_process,
            font=("Arial", 14, "bold")
        )
        self.btn_select.pack(pady=15)

        self.lbl_image = ctk.CTkLabel(self.left_frame, text="Lütfen bir su sayacı fotoğrafı seçin", font=("Arial", 12))
        self.lbl_image.pack(fill="both", expand=True, padx=10, pady=10)

        self.right_frame = ctk.CTkFrame(self, width=450)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.lbl_title = ctk.CTkLabel(self.right_frame, text="📊 Analiz Sonuçları", font=("Arial", 18, "bold"))
        self.lbl_title.pack(pady=20)

        self.lbl_endeks_title = ctk.CTkLabel(self.right_frame, text="Endeks (Rakamlar):", font=("Arial", 13, "bold"))
        self.lbl_endeks_title.pack(anchor="w", padx=20, pady=(10, 2))
        self.txt_endeks = ctk.CTkEntry(self.right_frame, font=("Arial", 16, "bold"), width=300, justify="center")
        self.txt_endeks.pack(padx=20, pady=(0, 15))

        self.lbl_seri_title = ctk.CTkLabel(self.right_frame, text="Seri Numarası:", font=("Arial", 13, "bold"))
        self.lbl_seri_title.pack(anchor="w", padx=20, pady=(10, 2))
        self.txt_seri = ctk.CTkEntry(self.right_frame, font=("Arial", 16, "bold"), width=300, justify="center")
        self.txt_seri.pack(padx=20, pady=(0, 15))

        self.lbl_status = ctk.CTkLabel(self.right_frame, text="Durum: Hazır", text_color="gray", font=("Arial", 12))
        self.lbl_status.pack(pady=20)

    # ------------------------------------------------------------------
    def select_and_process(self):
        file_path = ctk.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return

        self.txt_endeks.delete(0, "end")
        self.txt_seri.delete(0, "end")
        self.lbl_status.configure(text="Durum: Analiz ediliyor... (biraz sürebilir)", text_color="yellow")
        self.btn_select.configure(state="disabled")
        self.update()

        try:
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                self.lbl_status.configure(text="Durum: Görsel okunamadı!", text_color="red")
                return

            detections = self.detector.detect(file_path)
            best_su_sayaci = (max(detections["su_sayaci"], key=lambda x: x["confidence"])
                               if detections["su_sayaci"] else None)
            best_endeks = (max(detections["endeks"], key=lambda x: x["confidence"])
                           if detections["endeks"] else None)
            best_seri = (max(detections["seri_no"], key=lambda x: x["confidence"])
                         if detections["seri_no"] else None)

            img_draw = img_cv.copy()
            for item, name in [(best_su_sayaci, "su_sayaci"), (best_endeks, "endeks"), (best_seri, "seri_no")]:
                if item:
                    x1, y1, x2, y2 = item["bbox"]
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_draw, f"{name} {item['confidence']:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(500, 450))
            self.lbl_image.configure(image=ctk_img, text="")

            # --- OKUMA: su_sayaci bölgesini (bağlamıyla birlikte) TEK seferde VLM'e gönder ---
            if best_su_sayaci:
                cropped_meter = self.processor.crop_region(img_cv, best_su_sayaci["bbox"], padding=15)
            else:
                cropped_meter = img_cv

            result = self.ocr.read_meter(cropped_meter)
            endeks_text = result["endeks"] if result["endeks"] else "Okunamadı"
            seri_text = result["seri_no"] if result["seri_no"] else "Okunamadı"

            self.txt_endeks.insert(0, endeks_text)
            self.txt_seri.insert(0, seri_text)
            self.lbl_status.configure(text="Durum: Tamamlandı!", text_color="green")

        except Exception as e:
            print("⚠️ Analiz sırasında hata oluştu:")
            traceback.print_exc()
            self.lbl_status.configure(text=f"Durum: Hata oluştu ({type(e).__name__})", text_color="red")

        finally:
            self.btn_select.configure(state="normal")