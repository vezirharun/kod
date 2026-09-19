# V3 Index Pipeline — Preview First Contract

## Hızlı İndeks

1. Kaynak dosya keşfedilir.
2. Feature Preview oluşturulur veya mevcut fiziksel Preview doğrulanır.
3. Preview havuzu hazır olduktan sonra Thumbnail **yalnızca Preview'dan** oluşturulur.
4. Thumbnail + Preview hazır olduğunda Hızlı İndeks tamamlanır.
5. Hash ve Metadata Hızlı İndeks tamamlanma kriteri değildir.

## Genel AI

1. Genel AI kaynak dosyayı görsel analiz için doğrudan açmaz.
2. Fiziksel Preview hazır değilse AI işi bekler.
3. Preview hazırsa Preview havuzundan okur.
4. Hash ve Metadata Genel AI lane'inde gösterilir ve tamamlanır.
5. DINO, OpenCLIP, Texture, Semantic, Pattern DNA, Patch ve isteğe bağlı OCR aynı Genel AI hattında ilerler.
6. Genel AI Final sayacı Hash + Metadata + zorunlu ağır AI çıktılarının tamamını temsil eder; OCR ayrı gösterilir ve opsiyoneldir.

## Sayaç anlamları

- **Hızlı İndeks:** Thumbnail + Preview tamamlanan dosyalar.
- **Genel AI:** Hash + Metadata + ağır AI analizleri tamamlanan dosyalar.
- **Preview:** AI'nin tek görsel giriş kapısı.
- **Hash / Metadata:** Genel AI altında ayrı artifact havuzları.

## Kaynak dosyaya dönüş kuralı

- Thumbnail işi Preview hazır değilse kaynak dosyaya dönmez; Preview'yi bekler.
- Hash Preview hazır değilse kaynak dosyaya dönmez.
- Metadata Preview hazır değilse kaynak dosyaya dönmez.
- DINO/CLIP/Texture/Patch/OCR vb. Preview hazır değilse kaynak dosyaya dönmez.
