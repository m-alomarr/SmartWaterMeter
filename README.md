# SmartWaterMeter — Akıllı Su Sayacı Okuma Sistemi

Su sayacı fotoğraflarından **endeks (tüketim) değerini** ve **seri numarasını** otomatik olarak tespit edip okuyan masaüstü uygulaması. Sistem, nesne tespiti ve görsel dil modelini (Vision-Language Model) birlikte kullanarak, elle veri girişi gerektirmeden sayaç bilgilerini dijitalleştirmeyi amaçlar.

## İçindekiler

- [Genel Bakış](#genel-bakış)
- [Nasıl Çalışır](#nasıl-çalışır)
- [Klasör Yapısı](#klasör-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Veri Seti](#veri-seti)
- [Bilinen Sınırlamalar](#bilinen-sınırlamalar)

## Genel Bakış

Su sayacı okuma işlemi, sahada çalışan personel için zaman alıcı ve hataya açık bir süreçtir. Bu proje, bir su sayacı fotoğrafı yüklendiğinde:

1. Fotoğraftaki sayaç gövdesini, endeks penceresini ve seri numarası bölgesini otomatik olarak tespit eder,
2. Bu bölgeleri ayrı ayrı kırpıp bir görsel dil modeline göndererek üzerlerindeki rakamları okur,
3. Sonuçları ekranda kullanıcıya sunar.

Uygulama tamamen yerel (local) çalışır; görsellerin işlenmesi için herhangi bir bulut servisine veya internete bağımlılık yoktur.

## Nasıl Çalışır

Sistem iki aşamalı bir mimari üzerine kurulmuştur:

**1. Aşama — Bölge Tespiti (YOLO)**
Özel olarak etiketlenmiş bir veri seti üzerinde eğitilmiş bir YOLO modeli (`models/best.pt`), fotoğraf üzerinde üç sınıfı tespit eder: `su_sayaci` (sayacın tamamı), `endeks` (mekanik tüketim penceresi) ve `seri_no` (metal halka üzerindeki kabartma seri numarası). Ana fotoğrafta bir bölge bulunamazsa, sistem önce sayaç gövdesini kırpıp o bölgede ikinci bir tarama yaparak tespit şansını artırır.

**2. Aşama — Okuma (Görsel Dil Modeli)**
Tespit edilen her bölge ayrı ayrı kırpılır ve yerel olarak çalışan bir görsel dil modeline (Ollama üzerinden `qwen2.5vl`) gönderilir. Model, her bölge için özel olarak yazılmış talimatlarla yönlendirilir ve sonucu yapılandırılmış JSON formatında döndürür. Gelen cevap, sayısal olmayan karakterlerden ayıklanarak arayüzde gösterilir.

Bu iki aşamalı yaklaşımın temel avantajı, YOLO'nun hızlı ve güvenilir konum bulma yeteneği ile görsel dil modelinin esnek okuma yeteneğinin bir arada kullanılmasıdır — sadece OCR'a güvenmek yerine, modele "nereye bakması gerektiği" önceden söylenir.

## Klasör Yapısı

```
SmartWaterMeter/
├── core/
│   ├── detector.py      # YOLO tabanlı bölge tespiti
│   ├── ocr_engine.py    # Ollama/VLM ile rakam okuma
│   └── processor.py     # Görüntü kırpma yardımcı fonksiyonları
├── models/
│   └── best.pt          # Eğitilmiş YOLO modeli
├── ui/
│   └── app_gui.py       # customtkinter tabanlı arayüz
├── main.py               # Uygulama giriş noktası
└── requirements.txt      # Python bağımlılıkları
```

## Kurulum

### Gereksinimler

- Python 3.10 veya üzeri
- [Ollama](https://ollama.com) (yerel dil/görsel modellerini çalıştırmak için)
- Daha iyi performans için ekran kartı (GPU) kullanılması önerilir.

### Adımlar

1. Depoyu klonlayın ve proje klasörüne geçin:
   ```bash
   git clone <repo-url>
   cd SmartWaterMeter
   ```

2. Sanal ortam oluşturup etkinleştirin:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS / Linux
   ```

3. Bağımlılıkları yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

4. Ollama'yı kurun ve kullanılacak görsel dil modelini indirin:
   ```bash
   ollama pull qwen2.5vl:7b
   ```

## Kullanım

Uygulamayı başlatmak için:

```bash
python main.py
```

Açılan pencerede **"Fotoğraf Seç ve Analiz Et"** butonuna tıklayarak bir su sayacı fotoğrafı seçin. Sistem otomatik olarak:

- Fotoğraf üzerinde tespit edilen bölgeleri yeşil kutularla işaretler,
- Endeks ve seri numarası değerlerini sağ paneldeki ilgili alanlara yazdırır.

## Veri Seti

Projede kullanılan YOLO modeli, farklı su sayacı fotoğraflarından oluşan özel bir veri seti üzerinde eğitilmiştir. Veri seti manuel olarak toplanmış ve `su_sayaci`, `endeks`, `seri_no` sınıfları için etiketlenmiştir. Veri seti boyutu sınırlı olduğundan, modelin genelleme (generalization) performansı farklı sayaç marka/modellerinde değişkenlik gösterebilir; veri setinin genişletilmesi doğruluğu artıracak en doğrudan yoldur.

## Bilinen Sınırlamalar

- **Donanım:** Analiz süresi kullanılan ekran kartına göre değişir; daha güçlü bir GPU ile işlem belirgin şekilde hızlanır.
- **Küçük veri seti:** Model, sınırlı sayıda örnekle eğitildiği için farklı ışık koşulları, açılar veya sayaç modellerinde tespit doğruluğu düşebilir.
- **Senkron işlem akışı:** Uygulama şu an her fotoğrafı tek tek ve arayüz iş parçacığı (thread) üzerinde işlemektedir; analiz süresince arayüz geçici olarak yanıt vermeyebilir.