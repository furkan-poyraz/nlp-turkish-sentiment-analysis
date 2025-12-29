# Türkçe Duygu Analizi Projesi

Bu proje, Türkçe metinler üzerinde duygu analizi (sentiment analysis) yapmak için geliştirilmiş kapsamlı bir NLP (Doğal Dil İşleme) çalışmasıdır. Proje, farklı makine öğrenmesi ve derin öğrenme modellerini kullanarak Türkçe metinlerdeki duyguları (Pozitif, Negatif, Nötr) sınıflandırmaktadır.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Kullanılan Teknolojiler](#kullanılan-teknolojiler)
- [Veri Seti](#veri-seti)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)

## Proje Hakkında

Bu proje, Türkçe dilinde yazılmış metinlerin duygusal tonunu analiz etmeyi amaçlamaktadır. Sosyal medya yorumları, ürün değerlendirmeleri ve çeşitli metin kaynaklarından toplanan veriler kullanılarak, metinlerin pozitif, negatif veya nötr olarak sınıflandırılması sağlanmaktadır.

### Proje Amaçları

- Türkçe dil yapısına özgü veri ön işleme teknikleri uygulamak
- Farklı model yaklaşımlarını karşılaştırmak (Geleneksel ML, CNN, Transformer modelleri)
- Türkçe duygu analizi için en iyi performansı gösteren modeli belirlemek
- Gerçek dünya uygulamaları için kullanılabilir bir sistem geliştirmek

## Özellikler

- **Kapsamlı Veri Ön İşleme**: Türkçe karakterler, stopwords, stemming işlemleri
- **Çoklu Model Desteği**: TF-IDF, CNN, BERTurk, XLM-RoBERTa modellerinin karşılaştırması
- **Görselleştirme**: Veri dağılımı, model performans metrikleri ve karşılaştırmalı grafikler
- **GPU Desteği**: CUDA destekli hızlandırılmış model eğitimi
- **Modüler Yapı**: Kolayca genişletilebilir ve özelleştirilebilir kod yapısı

## Kullanılan Teknolojiler

### Kütüphaneler ve Framework'ler

- **Transformers**: Hugging Face transformers kütüphanesi (BERT, XLM-RoBERTa)
- **PyTorch**: Derin öğrenme framework'ü
- **Scikit-learn**: Makine öğrenmesi algoritmaları ve değerlendirme metrikleri
- **NLTK**: Doğal dil işleme araçları
- **TurkishStemmer**: Türkçe kelimelerin kök bulma işlemleri
- **Datasets**: Hugging Face datasets kütüphanesi
- **Matplotlib & Seaborn**: Veri görselleştirme
- **Pandas & NumPy**: Veri manipülasyonu ve sayısal işlemler

### Modeller

1. **TF-IDF + Logistic Regression**: Geleneksel makine öğrenmesi yaklaşımı
2. **CNN (Convolutional Neural Network)**: Derin öğrenme tabanlı metin sınıflandırma
3. **BERTurk**: Türkçe için özelleştirilmiş BERT modeli
4. **XLM-RoBERTa**: Çok dilli transformer modeli

## Veri Seti

Proje, `winvoker/turkish-sentiment-analysis-dataset` veri setini kullanmaktadır. Bu veri seti şunları içerir:

- **Kaynak**: Ürün yorumları, sosyal medya metinleri, wiki metinleri
- **Örnek Sayısı**: 5,000 adet (örnekleme yapılmış)
- **Sınıflar**: 
  - Positive (Pozitif)
  - Negative (Negatif)
  - Notr (Nötr)
- **Özellikler**: text (metin), label (etiket), dataset (kaynak)

## Kurulum

### Gereksinimler

Python 3.8 veya üzeri bir sürüm gereklidir.

### Adımlar

1. Repoyu klonlayın:
```bash
git clone <repo-url>
cd nlp-project
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install transformers datasets torch scikit-learn nltk matplotlib seaborn TurkishStemmer sentencepiece
```

3. NLTK stopwords'ü indirin:
```python
import nltk
nltk.download('stopwords')
```

4. Google Colab için GPU ayarlarını kontrol edin (Opsiyonel):
- Runtime > Change runtime type > Hardware accelerator: GPU (T4)

## 💻 Kullanım

### Notebook'u Çalıştırma

1. Jupyter Notebook veya Google Colab'da `NLP_Project.ipynb` dosyasını açın
2. Hücreleri sırasıyla çalıştırın:
   - Kütüphane kurulumları
   - Veri yükleme ve ön işleme
   - Model eğitimi
   - Sonuç değerlendirmesi

### Veri Ön İşleme

Proje şu veri temizleme adımlarını içerir:

```python
def veri_temizleme(metin):
    # 1. Küçük harfe çevirme
    # 2. URL ve mention temizleme
    # 3. Noktalama ve sayı temizleme
    # 4. Stopwords kaldırma
    # 5. Stemming (kök bulma)
    return temiz_metin
```

### Model Eğitimi

Her model için eğitim parametreleri:

- **Epoch Sayısı**: 3
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Loss Function**: Cross Entropy

## 🏗 Model Mimarisi

### 1. TF-IDF + Logistic Regression
- TF-IDF vektörizasyonu (max_features=5000)
- Logistic Regression sınıflandırıcı
- Hızlı eğitim, baseline model

### 2. CNN Modeli
- Embedding Layer (100 boyutlu)
- 1D Convolutional Layers
- MaxPooling ve Dropout
- Dense layers ile sınıflandırma

### 3. BERTurk
- Model: `dbmdz/bert-base-turkish-cased`
- Türkçe için özel eğitilmiş BERT
- Fine-tuning ile eğitim

### 4. XLM-RoBERTa
- Model: `xlm-roberta-base`
- Çok dilli destek
- Transformer tabanlı mimari

## 📈 Sonuçlar

### Model Performans Karşılaştırması

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| TF-IDF + LR | ~0.75 | ~0.74 | ~0.75 | ~0.75 |
| CNN | ~0.80 | ~0.79 | ~0.80 | ~0.80 |
| BERTurk | ~0.86 | ~0.85 | ~0.86 | ~0.86 |
| XLM-RoBERTa | ~0.86 | ~0.85 | ~0.85 | ~0.86 |

### Önemli Bulgular

- Transformer tabanlı modeller (BERTurk, XLM-RoBERTa) geleneksel yöntemlere göre daha iyi performans göstermiştir
- BERTurk ve XLM-RoBERTa benzer performans sergilemiştir
- CNN modeli, TF-IDF'e göre belirgin bir iyileşme sağlamıştır
- Veri temizleme ve ön işleme adımları tüm modellerin performansını olumlu etkilemiştir

### Görselleştirmeler

Proje şu görselleştirmeleri içerir:
- Veri seti sınıf dağılımı
- Eğitim kayıpları (Training Loss)
- Model performans karşılaştırmaları
- Confusion Matrix

## Proje Yapısı

```
NLP_Project/
│
├── NLP_Project.ipynb        # Ana notebook dosyası
├── README.md                # Bu dosya
│
└── Bölümler:
    ├── 1. Veri Seti Çekme ve Kütüphane Kurulumu
    ├── 2. Veri Ön İşleme ve Temizleme
    ├── 3. Veri Yükleme ve İnceleme
    ├── 4. Model Eğitimi (TF-IDF)
    ├── 5. CNN Model Eğitimi
    ├── 6. BERTurk Model Eğitimi
    ├── 7. XLM-RoBERTa Model Eğitimi
    └── 8. Model Karşılaştırması ve Sonuçlar
```

## Gelecek Geliştirmeler

- [ ] Daha fazla model eklenmesi (GPT-based, T5)
- [ ] Hyperparameter optimization
- [ ] Cross-validation implementasyonu
- [ ] Web API geliştirmesi (Flask/FastAPI)
- [ ] Daha büyük veri seti ile eğitim
- [ ] Fine-grained sentiment analysis (5 sınıf)
- [ ] Aspect-based sentiment analysis

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

##  Notlar

- Model eğitimi için GPU kullanımı önerilir (özellikle transformer modelleri için)
- Veri seti büyüklüğü ihtiyaca göre ayarlanabilir
- Hyperparameter'lar deneysel olarak optimize edilebilir
- Türkçe karakterlere özel dikkat gösterilmelidir

## İletişim

Sorularınız veya önerileriniz için lütfen iletişime geçin.

## Teşekkürler

- Hugging Face ekibine transformers kütüphanesi için
- Türkçe NLP topluluğuna katkıları için
- Veri seti sağlayıcılarına

---

**Not**: Bu proje eğitim ve araştırma amaçlıdır. Ticari kullanım için gerekli izinler alınmalıdır.
