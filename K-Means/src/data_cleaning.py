import os
import json

# Veri yollarını ayarla
DATA_DIR = "/Users/samedakgun/PycharmProjects/Unsupervised_Learning/K-Means/data"
SUBSETS = ["train", "test", "valid"]


def validate_data(data_dir, subsets):
    """
    Verileri doğrular ve eksik ya da uyumsuz dosyaları raporlar.
    """
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        json_path = os.path.join(subset_dir, "_annotations.coco.json")

        # JSON dosyasını kontrol et
        if not os.path.exists(json_path):
            print(f"[ERROR] {subset} için JSON dosyası bulunamadı: {json_path}")
            continue

        # JSON dosyasını yükle
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        image_files_in_json = {img['file_name'] for img in annotations.get('images', [])}
        image_files_in_folder = {file for file in os.listdir(subset_dir) if file.endswith(('.jpg', '.png'))}

        # Eksik dosyaları kontrol et
        missing_images = image_files_in_json - image_files_in_folder
        if missing_images:
            print(f"[WARNING] {subset} için eksik görüntü dosyaları: {missing_images}")

        # Fazla dosyaları kontrol et
        extra_files = image_files_in_folder - image_files_in_json
        if extra_files:
            print(f"[INFO] {subset} için JSON'da tanımlanmayan dosyalar: {extra_files}")

        # Raporla
        print(
            f"[INFO] {subset} doğrulama tamamlandı: {len(image_files_in_json)} JSON kaydı, {len(image_files_in_folder)} görüntü dosyası bulundu.")


if __name__ == "__main__":
    validate_data(DATA_DIR, SUBSETS)
