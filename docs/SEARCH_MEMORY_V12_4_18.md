# Arama Hafızası v12.4.18

## Sorun
Tekrarlanan metin sorguları yalnızca RAM'deki 12 girişlik arama önbelleğine bağlıydı. Farklı sorgular bu önbelleği doldurunca daha önce çalıştırılmış bir sorgu yeniden tam arama bekliyordu. Ayrıca mevcut LearningMemory yalnızca geri bildirim/öğretim kayıtlarını tutuyor; arama sonuç listesini tutmuyordu.

## Çözüm
- RAM arama önbelleği 128 sorguya çıkarıldı.
- Metin sorguları için kalıcı `search_memory.db` eklendi.
- Önceden çalıştırılmış metin sorgusu uygulama yeniden açılsa bile ilk sonuç listesini anında gösterebilir.
- Eski sonuç gösterilirken canlı arama arka planda devam eder (stale-while-revalidate).
- Canlı arama tamamlandığında kalıcı hafıza güncellenir.
- Semantik arama ayarları cache anahtarına dahil edildi.
- Kullanıcı geri bildirimi/etiket/kategori değişikliklerinde RAM + kalıcı arama hafızası birlikte temizlenir.
- Bu mekanizma leopara özel değildir; tüm metin sorguları için geneldir.
