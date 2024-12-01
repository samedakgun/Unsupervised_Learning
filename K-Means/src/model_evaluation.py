import os
import numpy as np
import cv2
import joblib
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Veri yollarını ayarla
DATA_DIR = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/data"
MODEL_DIR = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/models"
MODEL_PATH = os.path.join(MODEL_DIR, "kmeans_model.pkl")

def extract_features_with_histograms(image_dir, bins=16):
    """
    Görüntülerden renk histogramları çıkarır.
    """
    features = []
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Renk histogramını çıkar
            hist_r = np.histogram(image[:, :, 0], bins=bins, range=(0, 256))[0]
            hist_g = np.histogram(image[:, :, 1], bins=bins, range=(0, 256))[0]
            hist_b = np.histogram(image[:, :, 2], bins=bins, range=(0, 256))[0]
            hist = np.hstack((hist_r, hist_g, hist_b))  # RGB histogramları birleştir
            features.append(hist)
    return np.array(features)

def evaluate_model_on_dataset(model, features, dataset_name):
    """
    Test veya valid kümesinde modeli değerlendirir.
    """
    print(f"[INFO] {dataset_name} kümesi üzerinde model değerlendirmesi başlıyor...")
    labels = model.predict(features)
    silhouette_avg = silhouette_score(features, labels)
    print(f"[INFO] {dataset_name} Silhouette Skoru: {silhouette_avg:.4f}")
    return labels

def visualize_clusters_2d(features, labels, dataset_name):
    """
    Test veya valid verisi için kümeleri 2D olarak görselleştirir.
    """
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = features[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Küme {label}")
    plt.title(f"{dataset_name} K-Means Kümeleri")
    plt.xlabel("Histogram Kırmızı Bileşen")
    plt.ylabel("Histogram Yeşil Bileşen")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Modeli yükle
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Eğitilen model bulunamadı.")
        exit()

    kmeans = joblib.load(MODEL_PATH)
    print("[INFO] Model başarıyla yüklendi.")

    # Test ve Valid veri kümelerini değerlendirme
    for subset in ["test", "valid"]:
        subset_dir = os.path.join(DATA_DIR, subset)
        features = extract_features_with_histograms(subset_dir)
        labels = evaluate_model_on_dataset(kmeans, features, subset)
        visualize_clusters_2d(features, labels, subset)
