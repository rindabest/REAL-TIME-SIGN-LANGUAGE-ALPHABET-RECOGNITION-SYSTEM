# REAL-TIME SIGN LANGUAGE ALPHABET RECOGNITION SYSTEM

## Mo ta
He thong nhan dang bang chu cai ngon ngu ky hieu (ASL) theo thoi gian thuc su dung webcam.

## Cong nghe su dung
- **MediaPipe Hands**: Phat hien va trich xuat toa do ban tay (21 landmarks)
- **Random Forest Classifier**: Mo hinh phan loai voi 300 cay quyet dinh
- **OpenCV**: Xu ly anh va hien thi webcam

## Do chinh xac
- **99.39%** tren tap test (11,839 mau)
- Nhan dang **29 lop**: A-Z, del, nothing, space

## Cau truc thu muc
```
SAL2/
├── Model_training.ipynb   # Notebook huan luyen model (chay tren Google Colab)
├── test_cam.py            # Script demo webcam (chay tren may local)
├── RD.pkl                 # File model da huan luyen (~400MB, khong co tren GitHub)
├── .gitignore
└── README.md
```

## Huong dan su dung

### 1. Huan luyen model
- Mo file `Model_training.ipynb` tren Google Colab
- Chay tat ca cac cell de:
  + Trich xuat toa do tu dataset ASL
  + Huan luyen mo hinh Random Forest
  + Xuat file `RD.pkl`
- Tai file `RD.pkl` ve may tinh

### 2. Cai dat thu vien
```bash
pip install opencv-python mediapipe==0.10.14 joblib numpy scikit-learn
```

### 3. Demo webcam
```bash
python test_cam.py
```
- Dua tay vao camera de nhan dang chu cai
- Nhan phim `q` de thoat

## Luu y
- File `RD.pkl` (~400MB) khong duoc push len GitHub do gioi han kich thuoc.
  Ban can tu huan luyen model bang notebook hoac tai tu nguon khac.
- Can su dung **mediapipe==0.10.14** (phien ban moi hon da thay doi API).
- Model duoc huan luyen voi **scikit-learn 1.6.1** tren Colab.
