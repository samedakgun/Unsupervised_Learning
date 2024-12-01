import os
import numpy as np
import cv2
import joblib

# Veri yollarını ayarla
MODEL_DIR = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/models"
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

def extract_color_histogram(image_path, bins=16):
    """
    Tek bir görüntüden renk histogramı özelliklerini çıkarır.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Görüntü yüklenemedi. Dosya yolunu kontrol edin.")
        return None

    # Renk histogramı çıkar
    hist_r = np.histogram(image[:, :, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(image[:, :, 1], bins=bins, range=(0, 256))[0]
    hist_b = np.histogram(image[:, :, 2], bins=bins, range=(0, 256))[0]
    histogram_features = np.hstack((hist_r, hist_g, hist_b))
    return histogram_features.reshape(1, -1)  # Modelin tahmin edebilmesi için 2D dizi döndür

def predict_image_cluster(model, image_path):
    """
    Bir görüntüyü verilen modele göre bir kümeye atar.
    """
    features = extract_color_histogram(image_path)
    if features is not None:
        cluster_label = model.predict(features)[0]
        print("0: NO TUMOR")
        print("1: TUMOR")
        print(f"[INFO] Görüntü, küme {cluster_label} olarak sınıflandırıldı.")
        return cluster_label
    else:
        print("[ERROR] Özellik çıkarılamadı.")
        return None

if __name__ == "__main__":
    # Modeli yükle
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Eğitilen model bulunamadı.")
        exit()

    kmeans = joblib.load(MODEL_PATH)
    print("[INFO] Model başarıyla yüklendi.")

    # Görüntü yolunu belirtin
    image_path = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/data/test/27_jpg.rf.b2a2b9811786cc32a23c46c560f04d07.jpg"  # Burada tahmin edilecek görüntünün yolunu belirtin
    predict_image_cluster(kmeans, image_path)
