# Yüz Kimliği Pro Katmanı

Yüz sistemi Pattern/FAISS indeksinden ayrı bir SQLite veritabanında çalışır.

## Mimari
- InsightFace/ArcFace: yüz tespiti + embedding + cinsiyet sinyali
- `face_index.db`: dosya, yüz, kişi ve kalıcı prototipler
- `person_0001` gibi anonim kişi kimlikleri
- kişi adı değiştirilebilir
- aynı kişi farklı fotoğraflarda kalıcı kümelenir
- bilinen kadın/erkek çelişkisi kişi birleştirmesini engeller
- 5 adede kadar gerçek yüz prototipi centroid yanında tutulur
- görsel sorguda yüz eşleşmesi pattern prefilter tarafından elenmez
- yüz eşleşmesi mevcut pattern sıralamasına ek kanıt olarak girer
- çok yüzlü sorguda her yüz ayrı değerlendirilir

## Arka plan taraması
Uygulama açıldıktan sonra yüz indeksi ayrı bir arka plan iş parçacığında güncellenir.
Yeni/değişen dosyalar sonraki tarama döngüsünde yakalanır. Pattern DB ve FAISS değiştirilmez.

## Kurulum
`requirements_face.txt` kurulmalıdır. İlk InsightFace çalıştırmasında `buffalo_l` modeli
ortam tarafından indirilebilir; internet/model önbelleği yoksa yüz özelliği devre dışı kalır,
desen araması normal şekilde devam eder.

## Arama
- `kadın`
- `erkek`
- `bilinmiyor`
- `Kişi 001`
- `Kişi 001` veya kaydedilmiş kişi adı
- görsel sorgu: seçilen fotoğraftaki yüzlerin arşivdeki eşleşmeleri

Yüz eşleşmesi için tek bir centroid'e güvenilmez; gerçek yüz prototipleri de doğrulanır.
