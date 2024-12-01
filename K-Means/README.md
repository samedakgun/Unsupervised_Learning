# Brain Tumor Segmentation Project

## Proje Amacı
Bu proje, K-Means kümeleme algoritmasını kullanarak beyin tümörü segmentasyonu yapmak için geliştirilmiştir. Amacımız, görüntülerin farklı segmentlerini analiz etmek ve tümör bölgelerini tespit etmektir.

## Veri Seti Hakkında
Bu veri seti, [Roboflow](https://roboflow.com) tarafından sağlanmıştır ve 2146 adet görüntü içerir. 
- **Format**: COCO Segmentation formatında etiketlenmiştir.
- **Ön İşleme**: 
  - Görseller otomatik yönlendirme işlemlerine tabi tutuldu.
  - Boyutlar: 640x640 piksele yeniden boyutlandırıldı.
  - Görsellerde artırma teknikleri uygulanmadı.

### Veri Yapısı
- **train/**: Eğitim için kullanılan görüntüler.
- **test/**: Test aşamasında kullanılan görüntüler.
- **valid/**: Doğrulama için kullanılan görüntüler.

## Kullanılan Teknolojiler
- Python
- NumPy, Pandas
- Scikit-learn
- OpenCV
- Matplotlib, Seaborn

## Kurulum
1. Gerekli paketleri yükleyin:
   ```bash
   pip install -r requirements.txt
