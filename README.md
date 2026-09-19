# CategoryMatch Hatası Düzeltmesi

Ekrandaki hata:
`'CategoryMatch' object has no attribute 'parent'`

Nedeni: CategoryMatch alanları `parent/child` değil,
`primary_family/primary_subtype`.

Kullanım:
1. `vezir_pattern_search7.rar` dosyasını çıkarın.
2. Bu ZIP içindeki üç dosyayı çıkan ana klasöre kopyalayın.
3. `KategoriMatch_Hatasi_Duzelt.bat` çalıştırın.
4. Programı yeniden açın.

Script önce `core/search_engine.py` dosyasının yedeğini oluşturur.

Beklenen:
- marka -> tüm markalar
- amiri -> yalnız Amiri
- dior -> yalnız Dior
- dolce gabbana -> yalnız Dolce Gabbana
