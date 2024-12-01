import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import cv2
import joblib  # Model kaydetme ve yükleme için
from mpl_toolkits.mplot3d import Axes3D  # 3D görselleştirme için

# Veri yollarını ayarla
DATA_DIR = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/data"
MODEL_DIR = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/models"
SUBSETS = ["train", "test", "valid"]

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

def find_optimal_clusters(features, max_clusters=10):
    """
    En iyi küme sayısını bulmak için Elbow Method ve Silhouette Analysis uygular.
    """
    inertias = []
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(features, labels)
        silhouette_scores.append(silhouette_avg)
        print(f"[INFO] Küme Sayısı: {n_clusters}, Silhouette Skoru: {silhouette_avg:.4f}")

    # Elbow grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), inertias, marker='o', label="Inertia")
    plt.xlabel("Küme Sayısı")
    plt.ylabel("Inertia (Toplam Hata)")
    plt.title("Elbow Method")
    plt.legend()
    plt.show()

    # Silhouette grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', label="Silhouette Skoru")
    plt.xlabel("Küme Sayısı")
    plt.ylabel("Silhouette Skoru")
    plt.title("Silhouette Analysis")
    plt.legend()
    plt.show()

def train_kmeans(features, n_clusters):
    """
    K-Means modelini eğitir ve sonuçları döner.
    """
    print("[INFO] K-Means modeli eğitiliyor...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    silhouette_avg = silhouette_score(features, labels)
    print(f"[INFO] Ortalama Silhouette Skoru: {silhouette_avg:.4f}")
    return kmeans, labels

def save_model(model, model_path):
    """
    Eğitilen modeli belirtilen yola kaydeder.
    """
    print(f"[INFO] Model {model_path} konumuna kaydediliyor...")
    joblib.dump(model, model_path)
    print(f"[INFO] Model başarıyla kaydedildi.")

def visualize_clusters_3d(features, labels, n_clusters):
    """
    Kümeleri 3D olarak görselleştirir.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster in range(n_clusters):
        cluster_points = features[labels == cluster]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Küme {cluster}")

    ax.set_title("3D K-Means Kümeleme")
    ax.set_xlabel("Renk Histogramı - Kırmızı")
    ax.set_ylabel("Renk Histogramı - Yeşil")
    ax.set_zlabel("Renk Histogramı - Mavi")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Eğitim verilerinden özellikleri çıkar
    train_dir = os.path.join(DATA_DIR, "train")
    features = extract_features_with_histograms(train_dir)

    # En iyi küme sayısını bulun
    find_optimal_clusters(features, max_clusters=10)

    # K-Means modeli eğit
    n_clusters = 2  # Küme sayısını en uygun değere göre güncelleyin
    kmeans, labels = train_kmeans(features, n_clusters=n_clusters)

    # Modeli kaydet
    os.makedirs(MODEL_DIR, exist_ok=True)  # Klasör yoksa oluştur
    model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    save_model(kmeans, model_path)

    # Kümeleri 3D olarak görselleştir
    visualize_clusters_3d(features, labels, n_clusters)
