# customer-segmentation-api
API sederhana dan siap digunakan untuk segmentasi pelanggan menggunakan algoritma K-Means. Dibangun dengan FastAPI dan scikit-learn, layanan ini menganalisis perilaku pelanggan berdasarkan pengeluaran tahunan dan frekuensi belanja.

# 📦 K-Means Customer Segmentation API

Proyek ini merupakan implementasi sederhana dari **K-Means Clustering** menggunakan `scikit-learn`, dan disajikan sebagai **REST API dengan FastAPI**.

## 🚀 Fitur

- [✓] Dataset sintetik menggunakan `make_blobs`
- [✓] Training model K-Means
- [✓] Simpan model dan scaler ke file `.pkl`
- [✓] Endpoint API untuk prediksi cluster pelanggan

## 📁 Struktur Proyek

```
.
├── app.py              # FastAPI server (endpoint /predict)
├── main.py             # Training model dan save ke pickle
├── requirements.txt    # Daftar dependency Python
├── .gitignore          # File/folder yang diabaikan Git (optional)
└── models/
    ├── kmeans_model.pkl       # File model hasil training
    └── kmeans_scaler.pkl      # Scaler untuk normalisasi
```

## ⚙️ Instalasi

```bash
# 1. Clone repository ini
git clone https://github.com/username/kmeans-customer-segmentation-api.git
cd kmeans-customer-segmentation-api

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan training untuk membuat model
python main.py

# 4. Jalankan server API
uvicorn app:app --reload
```

## 📬 API Endpoint

### POST `/predict`

#### Request Body
```json
{
  "annual_spending": 5000,
  "purchase_frequency": 4
}
```

#### Response
```json
{
  "cluster": 1
}
```

## 📊 Visualisasi (Opsional)

`main.py` menghasilkan visualisasi Metode Elbow dan hasil clustering pelanggan:

- Metode Elbow untuk menentukan jumlah cluster optimal
- Plot cluster pelanggan dengan titik pusat (centroid)

## 🧠 Teknologi yang Digunakan

- Python 3.x
- scikit-learn
- FastAPI
- Pickle
- Matplotlib
- NumPy & Pandas

## 📌 Lisensi

Proyek ini dilisensikan di bawah lisensi MIT — bebas digunakan, dimodifikasi, dan dibagikan.

---

Created with ❤️ by arielshakaramiro
