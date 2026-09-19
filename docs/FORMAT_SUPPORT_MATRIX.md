# Vezir Pattern Search — Evrensel Format Desteği Araştırması

Tarih: 2026-07-02  
Durum: Mimari araştırma ve uygulama planı; bu belge çalışan sisteme format desteği eklendiği anlamına gelmez.

## 1. Yönetici özeti

Vezir'in genişlemesi bir `SUPPORTED_EXTENSIONS` listesi büyütülerek yapılmamalıdır. Doğru sınır, dosyayı tanıyan, güvenli biçimde metadata/önizleme çıkaran ve sonucu ortak bir varlık sözleşmesine dönüştüren plugin katmanıdır.

Araştırma sonucu:

- Bu belge 150 benzersiz uzantıyı raster, tasarım/vektör, nakış, triko, dokuma, CAD ve Office olarak sınıflandırır.
- Açık veya belgelenmiş değişim formatlarında (`TIFF`, `SVG`, `PDF`, `DST`, `PES`, `WIF`, `DXF`, OOXML/ODF gibi) uygulama içi parser ya da olgun bir kütüphane gerçekçidir.
- Wilcom `EMB`, Wings `NGS`, bazı `ART` sürümleri, Shima/Stoll native projeleri ve moda CAD üretici dosyaları için uzantıyı tanımak tam parse desteği değildir. Bu dosyalar vendor yazılımı/API'si veya doğrulanmış bir export üzerinden işlenmelidir.
- Binary hash tüm dosyalarda framework tarafından üretilebilir. Görsel hash, stitch imzası ve vektör geometri hash'i yalnız ilgili plugin'in normalize edilmiş içeriğinden üretilmelidir.
- Mevcut `pattern_groups` yapısı başlangıç noktasıdır; farklı uzantıları aynı desen ailesinde birleştirmek için dosya kaydı ile mantıksal tasarım varlığı ayrılmalıdır.
- Kod değişikliğine geçmeden önce gerçek müşteri arşivinden, her öncelikli format/sürüm için anonimleştirilmiş bir “golden corpus” gereklidir. Proprietary formatlarda örnek dosya olmadan verilen destek sözü güvenilir olmaz.

## 2. Kanıt ve destek seviyeleri

| Kod | Anlamı |
|---|---|
| L0 | Sadece katalog: uzantı tanınır, içerik parse edilmez |
| L1 | Güvenli temel metadata ve binary hash |
| L2 | Thumbnail/preview üretilebilir |
| L3 | Alan bilgisi çıkarılır: katman, vektör, stitch, örgü veya kalıp verisi |
| L4 | Normalize içerikten AI etiketi/embedding ve aile eşleştirme |
| L5 | Native nesne semantiği yüksek doğrulukla korunur; round-trip hedeflenmez |

Matriste `D` doğrudan/kütüphane ile, `X` harici dönüştürücü ile, `E` üretici export'u ile, `—` yok, `?` ise örnek dosya ve sürüm doğrulaması gerekli demektir. “AI” özgün dosyanın doğrudan modele verilmesi değil, güvenli biçimde üretilmiş raster preview veya normalize semantik veriden embedding üretilmesidir.

## 3. Mevcut sistem denetimi

Bugünkü kodda:

- `core/settings.py` yalnız 11 uzantıyı tarıyor: `tif, tiff, jpg, jpeg, png, bmp, psd, ai, eps, pdf, cdr`.
- `core/scanner.py` dosyayı içerik imzasıyla değil uzantıyla kabul ediyor.
- `core/thumbnailer.py` TIFF için özel yol, diğer rasterlar için pyvips/Pillow yolu kullanıyor.
- `core/preview_renderer.py` PDF için PyMuPDF/Ghostscript, AI/EPS için Ghostscript, PSD için `psd-tools`/Pillow kullanıyor; CDR'ı bilinçli olarak desteklenmiyor işaretliyor.
- `core/indexer.py` thumbnail → preview → hash/texture → OCR/AI → metin indeksi akışını yürütüyor. Ağır format erteleme ve resumable kuyruk zaten var.
- `ui/worker_threads.py` indeks ve aramayı `QThread` üzerinde çalıştırıyor. Yeni parserlar bu sınırı delmemeli.
- `core/db.py` dosya, feature, kuyruk ve `pattern_groups` tablolarına sahip; fakat plugin kimliği/sürümü, normalize metadata, artifact provenance ve logical asset kimliği için şema yok.

Sonuç: mevcut worker, kuyruk, preview cache, FAISS ve arama katmanları korunabilir. Format-spesifik kararlar `settings/scanner/preview_renderer/indexer` içinden çıkarılıp registry üzerinden çağrılmalıdır.

## 4. 150 uzantılık destek matrisi

### 4.1 Raster — 25 uzantı

| Uzantı(lar) | Bugün | Hedef | Thumb | Metadata | AI | Fallback / önerilen araç |
|---|---:|---:|---:|---:|---:|---|
| tif, tiff | D | L4 | D | D | D | pyvips → Pillow/libtiff; çok sayfa ve büyük dosya limitleri |
| jpg, jpeg | D | L4 | D | D | D | Pillow/libjpeg-turbo |
| png | D | L4 | D | D | D | Pillow/libpng |
| bmp | D | L4 | D | D | D | Pillow |
| webp | — | L4 | D | D | D | Pillow/libwebp |
| gif | — | L4 | D | D | D | Pillow; ilk kare + animasyon metadata |
| avif | — | L4 | D | D | D | Pillow/libavif veya pyvips |
| heic, heif | — | L4 | D | D | D | pillow-heif/libheif; lisans ve codec capability check |
| jp2, j2k, jpc | — | L4 | D | D | D | Pillow/OpenJPEG |
| pcx | — | L4 | D | D | D | Pillow |
| tga | — | L4 | D | D | D | Pillow |
| qoi | — | L4 | D | D | D | Pillow veya qoi |
| ppm, pgm, pbm, pnm, pfm | — | L4 | D | D | D | Pillow; PFM renk varyantı capability check |
| sgi | — | L4 | D | D | D | Pillow |
| xpm | — | L4 | D | D | D | Pillow |
| exr | — | L4 | D | D | D | OpenImageIO/OpenEXR; ton eşleme ile preview |

Not: Pillow formatı dosya içeriğinden tanıyabilir; registry uzantıyı yalnız aday seçimi için kullanmalı, magic/probe sonucu yetkili olmalıdır. Pillow'ın güncel format tablosu TIFF, JPEG, PNG, WebP, GIF, AVIF, JPEG 2000, PCX, PFM, SGI ve diğerlerinin gerçek okuma sınırlarını ayrı ayrı belgeliyor.

### 4.2 Adobe, Corel ve vektör/tasarım — 22 uzantı

| Uzantı(lar) | Tür | Bugün | Hedef | Thumb | Metadata/vektör | AI | Fallback / önerilen araç |
|---|---|---:|---:|---:|---:|---:|---|
| psd | Adobe native | D/X | L4 | D | D | D | psd-tools → Pillow; katman/ağaç metadata |
| psb | Adobe native büyük belge | — | L3 | D/? | D/? | D | psd-tools sürüm/corpus doğrulaması; Adobe export |
| pdd | Photoshop varyantı | — | L2 | X | ? | D | Adobe/Imagemagick export |
| ai | Illustrator native | X | L3 | X | X/? | D | PDF-compatible AI: PyMuPDF; aksi halde Ghostscript/Illustrator export |
| ait | Illustrator template | — | L2 | X | ? | D | Illustrator export |
| eps, epsf, ps | PostScript/export | X | L3 | X | D/X | D | Ghostscript `-dSAFER`; embedded preview varsa önce onu kullan |
| pdf | değişim/belge | D/X | L4 | D | D | D | PyMuPDF; sayfa sayısı ve render limitleri |
| svg | açık vektör | — | L4 | D | D | D | defusedxml + resvg/CairoSVG; script ve dış URL kapalı |
| svgz | sıkıştırılmış SVG | — | L4 | D | D | D | boyut limitli gzip + SVG parser |
| cdr | Corel native | L0 | L2 | X/? | X/? | D | LibreOffice/Inkscape/Corel CLI; sürüm bazlı corpus |
| cmx | Corel exchange | — | L2 | X/? | X/? | D | LibreOffice/Corel export |
| cpt | Corel PHOTO-PAINT | — | L2 | X/? | ? | D | Corel/Imagemagick capability check |
| pat | desen/preset; üreticiye göre değişir | — | L1 | ? | ? | ? | magic + producer ayrımı; tek parser varsayma |
| aco, ase | Adobe renk paleti | — | L3 | D* | D | D* | swatch parser; renk kartı preview üret |
| emf, wmf | Windows metafile | — | L3 | D/X | D/X | D | Windows GDI veya LibreOffice/Inkscape sandbox |
| cgm | grafik metafile | — | L2 | X | X/? | D | LibreOffice/Imagemagick delegate |
| odg | OpenDocument drawing | — | L3 | X | D | D | ZIP/XML metadata + LibreOffice headless preview |
| afdesign | Affinity Designer native | — | L0 | E | E | E | Affinity PDF/SVG export; native parser sözü verme |

Adobe ve Corel'in kendi destek tabloları AI/CDR/CMX/PDF/SVG/EPS ile çok sayıda değişim formatını doğruluyor; fakat bir masaüstü uygulamasının açabilmesi, sunucu tarafında açık parser bulunduğu anlamına gelmez.

### 4.3 Nakış — 43 uzantı

`Native/export` sütununda “stitch” makineye giden koordinat/komut dosyasını, “source” düzenlenebilir üretici projesini ifade eder.

| Uzantı(lar) | Aile | Native/export | Parser | Thumb | Stitch | Metadata | AI preview | Fallback |
|---|---|---|---|---:|---:|---:|---:|---|
| dst | Tajima | stitch/export | pyembroidery D | D | D | kısmi | D | gerekmez; renkler çoğu dosyada dışarıdan |
| pes, pec | Brother/Babylock | stitch/export | pyembroidery D | D | D | D | D | PEC/PES karşılıklı; DST son çare |
| exp | Melco/Bernina | stitch/export | pyembroidery D | D | D | kısmi | D | DST; renk paleti ayrı olabilir |
| jef | Janome | stitch/export | pyembroidery D | D | D | D | D | DST/PES |
| vp3 | Husqvarna/Pfaff | stitch/export | pyembroidery D | D | D | D | D | DST/PES |
| hus | Husqvarna legacy | stitch/export | pyembroidery D | D | D | kısmi | D | VP3/DST |
| xxx | Singer/Compucon | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| pcs | Pfaff | stitch/export | pyembroidery D | D | D | kısmi | D | DST/VP3 |
| sew | Janome/Elna legacy | stitch/export | pyembroidery D | D | D | kısmi | D | JEF/DST |
| tap | Happy | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| dsb | Barudan | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| dsz | ZSK | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| ksm | Pfaff/industrial legacy | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| 100, 10o, bro, dat | legacy/makine | stitch/export | pyembroidery D | D | D | kısmi | D | DST; `.dat` magic çakışmasına dikkat |
| emd, exy, fxy, gt, inb | legacy/makine | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| gcode | açık makine komutu | stitch/export | pyembroidery D | D | D | kısmi | D | normalize stitch stream |
| jpx, max, mit, new | legacy/makine | stitch/export | pyembroidery D | D | D | kısmi | D | DST; `.jpx` JPEG2000 ile magic çakışır |
| pcd, pcm, pcq | Pfaff/legacy | stitch/export | pyembroidery D | D | D | kısmi | D | DST; `.pcd` PhotoCD ile çakışır |
| phb, phc, shv, spx | Brother/Husqvarna/legacy | stitch/export | pyembroidery D | D | D | kısmi | D | PES/VP3/DST |
| stc, stx, tbf, u01 | industrial/legacy | stitch/export | pyembroidery D | D | D | kısmi | D | DST |
| ofm | Melco DesignShop | source/native | — | E/X | E/X | E/X | E | DesignShop ile EXP/DST export; vector/project ayarları native kalır |
| emb | Wilcom | source/native | — | E/X | E/X | E/X | E | Wilcom ile DST/PES/EXP export; native nesne iddiası yok |
| art | Bernina/Wilcom sürümleri | source/native | — | E/X | E/X | E/X | E | sürüm tespiti + vendor export |
| ngs | Wings Systems/NGS/XP/Modular | source/native | — | E | E | E | E | Wings üzerinden DST/PES/EXP export; örnek corpus şart |

pyembroidery 40'tan fazla nakış formatını okuyabildiğini ve stitch/jump/trim/stop/color-change gibi komutları normalize ettiğini belgeliyor. Bu nedenle ilk nakış plugin'i için en düşük riskli temel odur. `EMB/ART/NGS` ise aynı sınıfa sokulmamalıdır.

Stitch okunuyorsa plugin şu alanları üretir: `stitch_count`, `color_count`, `jump_count`, `trim_count`, `bounds`, `thread_palette`, `command_histogram`. `satin`, `tatami` ve `applique`, basit stitch dosyalarında çoğunlukla açık nesne türü değildir; segmentasyon/heuristic sonucu oldukları için `derived=true` ve güven skoru ile saklanmalıdır. Native source dosyası okunamıyorsa vendor tarafından üretilmiş DST/PES/EXP kardeş dosya aranır; bulunamazsa yalnız katalog + binary hash yapılır.

### 4.4 Triko — 10 uzantı

| Uzantı | Sistem | Tür | Hedef | Preview/metadata | Öneri/fallback |
|---|---|---|---:|---|---|
| k | Knitout | açık makine komutu | L4 | D/D | resmi Knitout grameri; komut histogramı ve grid preview |
| mdv | Stoll M1 Plus | native proje | L1 | E/E | Stoll M1plus; Sintral/Jacquard/Setup export |
| sin | Stoll Sintral | makine programı | L3 | D/X | kontrollü metin parser + Stoll doğrulaması |
| shp | Stoll shape | native/yardımcı | L1 | E/E | Stoll export; CAD `.shp` ile magic çakışır |
| spf | Stoll pattern | native | L1 | E/E | Stoll desteği ve örnek corpus |
| 000, 999 | Shima SDS/KnitPaint | makine/native veri | L1 | ?/? | Shima vendor doğrulaması; yalnız uzantıyla parse etme |
| pak, kcc, kcl | Shima SDS/KnitPaint | paket/kontrol | L1 | ?/? | Shima export veya lisanslı otomasyon; corpus şart |

Shima SDS-ONE/APEX ve Stoll M1plus için üretici kaynakları tasarım/programlama kabiliyetini doğruluyor; açık native format spesifikasyonu sunmuyor. Stoll kaynağı `.mdv` pattern ve `.sin` Sintral akışını doğruluyor. Açık, makineden bağımsız yol için Knitout `.k` en uygun plugin başlangıcıdır.

### 4.5 Dokuma — 10 uzantı

| Uzantı | Sistem | Tür | Hedef | Thumb | Metadata | Fallback / öneri |
|---|---|---|---:|---:|---:|---|
| wif | Weaving Information File | açık ASCII değişim | L4 | D | D | özel güvenli INI-benzeri parser; warp/weft draft render |
| jc3, jc4, jc5, zc5 | Stäubli/ArahWeave CAM | loom/jacquard | L2 | D/X | X/? | ArahWeave export; sürüm/corpus doğrulaması |
| dis, arm | JD&N/Arah import | dokuma/jacquard | L1 | X/? | X/? | ArahWeave üzerinden WIF/TIFF/PNG |
| qte, qtz | textile CAD import | proprietary | L1 | X/? | X/? | ArahWeave; QTZ sıkıştırma limiti |
| sf2 | Image Tech dobby | loom/dobby | L1 | X/? | X/? | ArahWeave → WIF/TIFF/PNG |

WIF; çözgü/atkı dizisi, renkler ve yoğunluk gibi alanları taşır, ancak repeat, denting, regulator ve bazı iplik özelliklerini taşımaz. Bu kayıp açıkça provenance içinde kaydedilmelidir. ArahWeave'in resmi format tablosu WIF ve JC/DIS/ARM/QTE/QTZ/SF2 ailesinin okuma/yazma sınırlarını listeler.

NedGraphics, Pointcarre, ScotWeave ve EAT için tek bir güvenilir, açık native parser tespit edilmedi. İlk sürümde ürün adı “plugin” değil vendor adapter profili olmalı; WIF/TIFF/PNG veya üretici destekli loom export'u alınmalıdır.

### 4.6 Moda/CAD — 15 uzantı

| Uzantı | Tür/sistem | Hedef | Thumb | Metadata/geometri | Fallback / öneri |
|---|---|---:|---:|---:|---|
| dxf | AutoCAD; AAMA/ASTM dialectleri | L4 | D | D | ezdxf; AAMA/ASTM header/katman profili ayrıca doğrula |
| dwg | AutoCAD binary | L3 | X | X | ODA/LibreDWG lisans incelemesi → DXF |
| igs, iges | IGES değişim | L3 | D/X | D/X | OpenCascade/assimp; 2D kalıp profili |
| plt, hpgl, hgl | plotter değişim | L3 | D | D | güvenli HP-GL parser; komut/koordinat limitleri |
| mdl | Lectra Modaris | native | L1 | E | E | Lectra Pattern Converter/vendor API; DXF AAMA/ASTM export |
| cei | Gerber CutWorks | native/değişim | L1 | E/X | E/X | Gerber/Lectra CutWorks export |
| ntv | Autometrix | native | L1 | E/X | E/X | vendor export → DXF/HPGL |
| cmd, nc | cutter/machine command | üretim | L2 | D* | D | salt-okunur komut parser; asla çalıştırma |
| gem | Gemini CAD | native | L1 | E | E | Gemini DXF AAMA/ASTM export |
| mtm | moda CAD/marker; üreticiye bağlı | L0 | ? | ? | magic + producer tespiti; vendor export |
| pds | pattern-design; üreticiye bağlı | L0 | ? | ? | magic + producer tespiti; uzantı tek başına yeterli değil |

Lectra/Gerber belgeleri AccuMark, Modaris MDL V8, DXF AAMA/ASTM, standart DXF ve IGS arasında üretici dönüştürme akışlarını doğruluyor. “Native parser” yerine desteklenen dönüştürücü kullanmak veri kaybını ve tersine mühendislik riskini azaltır. Optitex, Gemini ve Tukatech için de ilk sözleşme DXF AAMA/ASTM + plotter export olmalıdır.

### 4.7 Office/katalog — 25 uzantı

| Uzantı(lar) | Aile | Hedef | Thumb | Metadata/metin | AI | Fallback / öneri |
|---|---|---:|---:|---:|---:|---|
| doc | Word binary | L3 | X | D/X | D | LibreOffice headless; antiword yalnız metin fallback |
| docx, docm, dotx | Word OOXML | L4 | X | D | D | ZIP/XML + python-docx; makro çalıştırma yok |
| xls | Excel binary | L3 | X | D/X | D | LibreOffice; xlrd sürüm/corpus doğrulaması |
| xlsx, xlsm, xlsb, xltx | Excel modern | L4/L3 | X | D | D | openpyxl; xlsb için pyxlsb; formül çalıştırma yok |
| ppt | PowerPoint binary | L3 | X | D/X | D | LibreOffice headless |
| pptx, pptm, potx, ppsx | PowerPoint OOXML | L4 | X | D | D | python-pptx + LibreOffice preview; makro yok |
| odt, ott, odm | ODF text/template/master | L4 | X | D | D | ZIP/XML + odfpy/LibreOffice |
| ods | ODF spreadsheet | L4 | X | D | D | ZIP/XML + odfpy/LibreOffice |
| odp | ODF presentation | L4 | X | D | D | ZIP/XML + odfpy/LibreOffice |
| rtf | rich text | L3 | X | D | D | LibreOffice/pandoc sandbox |
| csv, tsv | tablo/metin | L3 | D* | D | D | charset/dialect sniff; ilk N satırdan tablo preview |
| pub | Microsoft Publisher | L2 | X | X/? | D | LibreOffice/Publisher PDF export |
| vsd, vsdx | Visio | L3 | X | D/X | D | LibreOffice; VSDX ZIP/XML parser |

Office dosyaları desen/katalog aramasında sayfa/slide render, OCR/metin, ürün kodu, müşteri, renk adı ve gömülü görseller sağlar. Makrolar, linkler, OLE nesneleri ve formüller hiçbir zaman çalıştırılmamalıdır. Microsoft ve LibreOffice format referansları bu uzantı ailelerini doğrular.

### 4.8 Sayım

| Grup | Benzersiz uzantı |
|---|---:|
| Raster | 25 |
| Adobe/Corel/vektör | 22 |
| Nakış | 43 |
| Triko | 10 |
| Dokuma | 10 |
| CAD | 15 |
| Office | 25 |
| **Toplam** | **150** |

Uzantı çakışmaları (`dat`, `jpx`, `pcd`, `shp`, `cmd`, `pds` gibi) registry'nin yalnız suffix ile seçim yapamayacağını kanıtlar. `probe(header, path, mime)` bir güven skoru döndürmeli ve belirsizlikte dosya L0/L1 olarak kalmalıdır.

## 5. Önerilen plugin sözleşmesi

İstenen klasörler korunabilir:

```text
plugins/
  raster/
  adobe/
  corel/
  embroidery/
  knitting/
  weaving/
  cad/
  office/
  pdf/
  vector/
```

Ortak sözleşme `core/formats/` altında olmalı; `plugins/` yalnız uygulamalara ait olmalıdır. Boolean metotlar tek başına yetersizdir. Her işlem destek/kalite/maliyet/bağımlılık bilgisi vermelidir:

```python
class FormatPlugin(Protocol):
    plugin_id: str
    plugin_version: str

    def supported_extensions(self) -> frozenset[str]: ...
    def probe(self, request: ProbeRequest) -> ProbeResult: ...
    def capabilities(self, request: AssetRequest) -> CapabilitySet: ...
    def extract_metadata(self, request: AssetRequest) -> AssetMetadata: ...
    def export_preview(self, request: AssetRequest, target: PreviewSpec) -> Artifact: ...
    def extract_domain_data(self, request: AssetRequest) -> DomainData | None: ...
```

`CapabilitySet`, kullanıcının istediği `can_open`, `can_preview`, `can_thumbnail`, `can_metadata`, `can_hash`, `can_ai_tags`, `can_stitch`, `can_vector` yeteneklerini kapsar; ayrıca `native/library/external/vendor_export`, tahmini maliyet ve eksik bağımlılık nedenini taşır. `can_hash` ikiye ayrılmalıdır:

- `binary_hash`: framework özelliği; okunabilen her dosyada vardır.
- `canonical_hash`: plugin özelliği; normalize piksel, stitch stream veya vektör geometri üzerinden hesaplanır.

`AssetMetadata` alanları:

```text
thumbnail, preview, width, height, colors, file_type, metadata,
semantic_tags, hashes{binary, perceptual, canonical}, texture,
category, family, subfamily, brand_style, designer, customer,
repeat, dpi, created, modified, provenance, warnings
```

Alanların yanında `source`, `confidence`, `derived`, `parser_id`, `parser_version` bulunmalıdır. Dosya sistemi `created/modified` zamanı ile dosya içi üretim zamanları ayrı saklanmalıdır.

## 6. Registry ve güvenli yürütme

Akış:

1. Scanner yalnız yol/stat toplar; uzantı allowlist'i registry'nin birleşik aday listesine dönüşür.
2. Registry ilk 64–256 KiB header ve suffix ile aday pluginleri sıralar.
3. En yüksek güvenli `probe` sonucu seçilir; çakışma kayda geçirilir.
4. Framework binary hash ve temel filesystem metadata üretir.
5. Plugin metadata/domain data çıkarır ve preview artifact üretir.
6. Mevcut feature extractor yalnız normalize preview üzerinde çalışır.
7. Sonuçlar plugin sürümü ve provenance ile atomik yazılır.

Kurallar:

- Parser/thumbnail/metadata/hash/AI işleri UI thread'de çalışmaz. Mevcut `IndexWorker` ve kuyruk korunur; CPU-bound veya güvensiz harici araçlar ayrı process'te çalıştırılır.
- Her görevde timeout, maksimum çıktı boyutu, maksimum sayfa/kare, maksimum piksel, maksimum stitch/vektör komutu ve iptal kontrolü bulunur.
- Ghostscript/LibreOffice/Inkscape/vendor CLI shell string ile değil argüman listesiyle, geçici izole dizinde çalışır.
- SVG dış URL/script, Office makro/OLE/formül, PDF attachment/JavaScript ve machine command hiçbir zaman çalıştırılmaz.
- Cache anahtarı `binary_hash + plugin_id + plugin_version + operation + settings_version` olur. Başarısızlıklar da kısa süreli negatif cache'e girer.
- Bir plugin'in çökmesi dosyayı ve diğer pluginleri etkilemez; sonuç `unsupported`, `dependency_missing`, `corrupt`, `timeout`, `resource_limit` olarak ayrılır.

## 7. Semantic search

Dosya adı sinyallerden yalnız biridir. Arama belgesi şu kaynaklardan oluşur:

```text
filename/path + OCR + parser metadata + domain metrics + visual caption/tags
+ taxonomy aliases + approved brand/style vocabulary + manual labels
```

Örnek sorgu genişletmeleri kontrollü sözlük + embedding birlikte uygulanır:

- `yazılı` → `text, typography, lettering, logo, monogram`
- `kamuflaj` → `camouflage, camo, military, forest, woodland, digital`
- `gucci` → yalnız yetkili marka sözlüğünde `GG, luxury, interlocking monogram`; sonuç “Gucci ürünü” diye doğrulanmaz, `Gucci-benzeri/brand-style signal` olarak işaretlenir.

Marka taklidi ve telif açısından `brand_style` çıkarımı bir kimlik doğrulaması değildir. Kaynak/güven skoru gösterilmeli ve kullanıcı düzeltmesi manual label guard ile korunmalıdır.

## 8. Formatlar arası desen ailesi kümelendirme

Yeni mantıksal model:

```text
design_asset (aynı yaratıcı tasarım)
  └─ asset_variant (renk/repeat/revizyon)
      └─ file_representation (PSD, AI, TIFF, DST, PES, CDR...)
```

Aynı aile kararı uzantıdan bağımsız kanıtlarla verilir:

- raster perceptual/patch embedding benzerliği,
- vektör raster preview + normalize geometri imzası,
- nakış stitch-path raster + stitch canonical hash,
- OCR/semantic tag ve kategori yakınlığı,
- müşteri/klasör/revizyon bilgisi,
- kullanıcı onayı veya reddi.

`Animal → Leopard → {DST, PSD, AI, TIFF, PES, CDR}` örneğinde dosyalar aynı `family/subfamily` altında listelenebilir; ancak otomatik olarak aynı `design_asset` yapılması daha yüksek eşik veya kullanıcı onayı gerektirir. Böylece benzer iki leopar desen yanlışlıkla tek tasarım diye birleşmez.

## 9. Uygulama sırası

### Faz 0 — corpus ve sözleşme

- Her öncelikli format/sürüm için iyi, bozuk, büyük ve parola/edge-case örnekleri topla.
- Plugin Protocol, capability/provenance modeli ve contract testlerini ekle.
- Var olan 11 uzantı için davranış snapshot testleri oluştur; bu testler geçmeden registry'ye geçme.

### Faz 1 — mevcut davranışı plugin arkasına taşı

- `raster`, `pdf`, `adobe`, `corel` adapterları.
- Scanner'da registry discovery; mevcut cache ve DB alanları geriye uyumlu kalır.
- İlk migration yalnız eklemeli olur: `plugin_id`, `plugin_version`, `format_id`, `metadata_json`, `provenance_json`, `canonical_hash`.

### Faz 2 — açık/değişim formatları

- SVG/SVGZ, WIF, DXF/HPGL, OOXML/ODF.
- `pyembroidery` ile önce DST/PES/PEC/EXP/JEF/VP3, sonra doğrulanmış diğer stitch formatları.
- Stitch ve weaving teknik preview rendererları.

### Faz 3 — proprietary adapterlar

- Wilcom/Wings/Shima/Stoll/Lectra/Gerber/Corel/Affinity için lisanslı vendor export adapterları.
- Her vendor/sürüm ayrı capability manifesti; “kurulu değil” durumu normal bir sonuçtur.

### Faz 4 — cross-format family graph

- `design_asset/asset_variant/file_representation` tabloları.
- Conservative auto-link, review queue, split/merge ve manual override.
- Önce offline backfill; mevcut arama sonuç sırasını bir anda değiştirme.

## 10. Kabul kriterleri

- Plugin kaldırıldığında uygulama açılır ve diğer formatlar çalışır.
- Aynı uzantılı ama farklı magic'e sahip testler doğru plugin'e gider veya güvenli biçimde belirsiz kalır.
- Parser kodu ve external converter UI thread'de çalışmaz; thread testi ve event-loop heartbeat testi geçer.
- Bozuk/çok büyük dosya timeout veya resource-limit ile sonlanır; worker havuzu kilitlenmez.
- Mevcut 11 formatın golden sonuçları gerilemez.
- DST/PES/EXP/JEF/VP3 örneklerinde stitch/color/jump sayıları referans araçla karşılaştırılır.
- WIF ve DXF örneklerinde ölçü/renk/repeat kaybı provenance içinde görünür.
- Office makroları, SVG scriptleri, PDF JavaScript ve NC/CMD içeriği çalıştırılmaz.
- Farklı uzantılı aynı tasarım aile önerisi üretir; benzer ama farklı tasarımlar otomatik birleşmez.

## 11. Kaynaklar

- [Pillow — image file formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)
- [ImageMagick — supported formats and delegates](https://imagemagick.org/formats/)
- [Adobe Illustrator — supported file formats](https://helpx.adobe.com/uk/illustrator/kb/supported-file-formats-illustrator.html)
- [CorelDRAW — supported file formats](https://product.corel.com/help/CorelDRAW/540227992/Main/EN/Documentation/CorelDRAW-Supported-file-formats.html)
- [pyembroidery — supported embroidery formats and commands](https://github.com/EmbroidePy/pyembroidery)
- [Ink/Stitch — embroidery read/write format matrix](https://inkstitch.org/docs/file-formats/)
- [Melco DesignShop — OFM/project ve export formatları](https://www.melco-service.com/docs/DS_V9_090309/Working_with_Files_in_DesignShop.htm)
- [Knitout `.k` specification](https://github.com/textiles-lab/knitout)
- [STOLL M1plus — MDV version selection](https://software.stoll.com/m1plus/m1plusknowledgebase/en-US/8250300171.html)
- [STOLL M1plus philosophy / Sintral](https://software.stoll.com/m1plus/m1plusknowledgebase/en-US/8252433035.html)
- [SHIMA SEIKI SDS-ONE APEX system](https://www.shimaseiki.co.jp/product/design/system/)
- [ArahWeave — supported image, WIF and loom formats](https://www.arahne.si/products/arahweave/)
- [ArahWeave manual — WIF capabilities and losses](https://www.arahne.si/wp-content/uploads/manuals/pdf/aweave-EN.pdf)
- [Lectra Modaris — AccuMark/DXF conversion](https://www.lectra.com/en/fashion/products/modaris)
- [Gerber AccuMark — export formats](https://gerber-help.lectra.com/AccuMark/PDS/File_Export_Tab_File.htm)
- [Gerber CutWorks — recognized CAD/cutter formats](https://gerber-help.lectra.com/Cutworks/Using_CutWorks.htm)
- [Microsoft — Office file format reference](https://learn.microsoft.com/en-us/office/compatibility/office-file-format-reference)
- [LibreOffice — OpenDocument XML formats](https://help.libreoffice.org/latest/en-GB/text/shared/00/00000021.html)

## 12. Karar

Bir sonraki adım doğrudan 150 uzantıyı `SUPPORTED_EXTENSIONS` içine eklemek değildir. Faz 0 ve Faz 1 birlikte uygulanmalı; ilk teslimat mevcut 11 formatı davranış değişmeden registry/plugin arkasına taşımalı ve contract testleriyle kilitlemelidir. İlk yeni gerçek domain desteği, açık format ve olgun parser avantajı nedeniyle nakışta DST/PES/PEC/EXP/JEF/VP3; dokumada WIF; CAD'de DXF; Office'te OOXML/ODF olmalıdır.
