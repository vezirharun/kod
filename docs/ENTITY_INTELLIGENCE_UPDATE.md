# V12.1 — Varlık Intelligence güncellemesi

Bu sürüm, marka aramasındaki kanıt tabanlı mimariyi görsel varlık aramasına taşır.

## Değişen ana parçalar

- `core/entity_aliases.py`: Varlık alias/ontoloji çözümleme.
- `core/entity_evidence.py`: indekslenmiş varlık kanıtlarını tek kanonik kümede toplar.
- `core/index_enrichments.py`: mevcut metadata'dan `entity_evidence` üretir.
- `core/text_index.py`: varlık kanıtını metin arama skoruna dahil eder ve indeks blob'una ekler.
- `core/universal_visual_intel.py`: nesne kabul tabanı `%26` yerine `%18`; hard-object sonuçları artık ayrı varlık kabul yolu kullanır.
- `core/search_engine.py`: insan/kedi/kuş/balık/köpek gibi temel varlıklar için daha güçlü CLIP prompt topluluğu.
- `core/search_acceptance.py`: varlık sonuçları genel `%60` tekstil eşiğine tekrar takılmaz.
- `core/category_predictions.py`: indekslenmiş varlık kanıtı sonuç detayına aktarılır.
- `scripts/backfill_entity_evidence.py`: mevcut indeks için model çalıştırmadan kanıt backfill işlemi.

## Temel davranış

Önceki akış:

`CLIP → genel skor → %60 genel eşik`

Yeni akış:

`Entity sorgusu → Entity CLIP kanıtı → varlık kabul eşiği (%18) → kanonik kabul → UI`

Böylece `kedi`, `kuş`, `insan` gibi sorguların `0.20–0.30` aralığındaki gerçek görsel kanıtı, tekstil aramasının genel `%60` eşiğinde kaybolmaz.

## Güvenlik / doğruluk

- Leopar, çiçek, zebra ve marka yolları yeniden yazılmadı.
- Genel desen motoru ve FAISS/DINO/CLIP altyapısı korunur.
- Varlık eşiğinin düşürülmesi tek başına her sonucu kabul etmez; UVI hard-object gate, rakip/distractor kanıtı ve kanonik son kabul birlikte çalışır.
- İnsan/kimlik/yüz tarafında biyometrik kimlik iddiası eklenmemiştir.

## Mevcut indeks

Yeni `entity_evidence` alanı mevcut kayıtlar için `scripts/backfill_entity_evidence.py` ile model çalıştırmadan doldurulabilir. Bu işlem yeni görsel nesne tespiti üretmez; mevcut kategori/semantic/object metadata'sını kanonikleştirir. Yeni dosyalar normal indeks akışında alanı üretir.
