"""Hierarchical textile category tree and alias resolution."""

from __future__ import annotations

from dataclasses import dataclass
import re

from core.textile_terms import normalize_turkish

CATEGORY_TREE: dict[str, dict[str, tuple[str, ...]]] = {
    "Animal": {
        "Rabbit": ("rabbit", "tavsan", "tavşan", "bunny"),
        "Cat": ("cat", "kedi", "black cat"),
        "Dog": ("dog", "kopek", "köpek"),
        "Octopus": ("octopus", "ahtapot"),
        "Snake": ("snake animal", "yilan hayvan", "yılan hayvan"),
        "Butterfly": ("butterfly", "kelebek"),
        "Bird": ("bird", "kus", "kuş"),
        "Fish": ("fish", "balik", "balık"),
        "Other Animal": ("other animal", "mixed animal", "hayvan deseni"),
    },
    "Animal Print": {
        "Leopard": ("leopard", "leopar", "jaguar print", "cheetah print"),
        "Zebra": ("zebra", "zebra desen"),
        "Snake Skin": ("snake skin", "yilan derisi", "yılan derisi"),
        "Tiger": ("tiger", "kaplan"),
        "Crocodile": ("crocodile", "timsah"),
        "Cow": ("cow print", "inek deseni"),
        "Giraffe": ("giraffe", "zurafa", "zürafa"),
        "Mixed Animal Print": ("animal print", "hayvan derisi"),
    },
    "Floral": {
        "Rose": ("rose", "gul", "gül"),
        "Daisy": ("daisy", "papatya"),
        "Leaf": ("leaf", "yaprak"),
        "Small Floral": ("small floral", "kucuk cicek", "küçük çiçek"),
        "Ditsy Floral": ("ditsy floral", "citir cicek", "çıtır çiçek"),
        "Big Flower": ("big flower", "buyuk cicek", "büyük çiçek"),
        "Allover Floral": ("allover floral", "metraj cicek", "metraj çiçek"),
        "Seamless Floral": ("seamless floral", "kesintisiz cicek", "kesintisiz çiçek"),
        "Mixed Floral": ("floral", "flower", "cicek", "çiçek"),
    },
    "Camouflage": {
        "Forest Camo": ("forest camo", "orman kamuflaj"),
        "Military Camo": ("military camo", "askeri kamuflaj"),
        "Digital Camo": ("digital camo",),
        "Urban Camo": ("urban camo",),
        "Abstract Camo": ("abstract camo",),
    },
    "Geometric": {
        "Spiral": ("spiral", "swirl", "sarmal"),
        "Optical": ("optical", "op art"),
        "Circle": ("circle", "daire"),
        "Square": ("square", "kare"),
        "Mosaic": ("mosaic", "mozaik"),
        "Line Art": ("line art", "cizgi sanat", "çizgi sanat"),
        "Repeat Geometric": ("geometric", "geometrik", "repeat geometric"),
    },
    "Plaid Check": {
        "Ekose": ("ekose", "tartan"),
        "Check": ("check", "checked"),
        "Plaid": ("plaid",),
        "Houndstooth": ("houndstooth", "kazayagi", "kazayağı"),
        "Gingham": ("gingham",),
    },
    "Stripe": {
        "Vertical Stripe": ("vertical stripe", "dikey cizgi", "dikey çizgi"),
        "Horizontal Stripe": ("horizontal stripe", "yatay cizgi", "yatay çizgi"),
        "Diagonal Stripe": ("diagonal stripe", "capraz cizgi", "çapraz çizgi"),
        "Wavy Stripe": ("wavy stripe", "dalgali cizgi", "dalgalı çizgi"),
    },
    "Lace": {
        "Lace Pattern": ("lace", "dantel"),
        "Embroidery Lace": ("embroidery lace", "nakis dantel", "nakış dantel"),
        "Transparent Lace": ("transparent lace", "seffaf dantel", "şeffaf dantel"),
    },
    "Drawing Illustration": {
        "Cartoon": ("cartoon", "karikatur", "karikatür"),
        "Hand Drawing": ("hand drawing", "el cizimi", "el çizimi"),
        "Animal Drawing": ("animal drawing", "hayvan cizimi", "hayvan çizimi"),
        "Line Illustration": ("line illustration",),
        "Character Drawing": ("character", "karakter"),
    },
    "Typography Text": {
        "Text Pattern": ("text pattern", "yazi deseni", "yazı deseni"),
        "Repeated Text": ("repeated text", "tekrar eden yazi", "tekrar eden yazı"),
        "Newspaper Letter": ("newspaper", "letter", "gazete", "harf deseni"),
        "Brand Text Style": ("brand text", "marka yazisi", "marka yazısı"),
        "Logo Text": ("logo text", "logo yazisi", "logo yazısı"),
    },
    "Monogram Logo": {
        "Repeated Logo": ("repeated logo", "logo desen"),
        "Monogram": ("monogram",),
        "Luxury Style": ("luxury style", "luks desen", "lüks desen"),
        "Emblem": ("emblem", "arma"),
        "Symbol Pattern": ("symbol pattern", "sembol desen"),
    },
    "Paisley Scarf": {
        "Paisley": ("paisley", "sal desen", "şal desen"),
        "Border": ("border", "bordur", "bordür"),
        "Medallion": ("medallion", "madalyon"),
        "Oriental Ethnic": ("oriental", "ethnic", "etnik"),
    },
    "Baroque Ornament": {
        "Baroque": ("baroque", "barok"),
        "Ornament": ("ornament", "susleme", "süsleme"),
        "Chain": ("chain", "zincir"),
        "Gold Ornament": ("gold ornament", "altin susleme", "altın süsleme"),
        "Luxury Ornament": ("luxury ornament",),
    },
    "Marble Abstract": {
        "Marble": ("marble", "mermer"),
        "Fluid Art": ("fluid art", "akiskan boya", "akışkan boya"),
        "Watercolor": ("watercolor", "suluboya"),
        "Color Swirl": ("color swirl", "renk akisi", "renk akışı"),
        "Smoke Ink": ("smoke", "ink", "duman", "murekkep", "mürekkep"),
        "Abstract Paint": ("abstract", "abstract paint"),
    },
    "Soyut": {},
    "Nitelik": {},
    "Texture Ground": {
        "Plain Texture": ("plain texture", "duz doku", "düz doku"),
        "Fabric Texture": ("fabric texture", "kumas doku", "kumaş doku"),
        "Knit Texture": ("knit", "orgu", "örgü"),
        "Denim Texture": ("denim", "kot"),
        "Paper Grain": ("paper grain", "kagit doku", "kağıt doku"),
    },
    "Textile": {
        "Plaid": ("plaid", "ekose", "tartan"),
        "Stripe": ("stripe", "cizgi", "çizgi", "striped"),
        "Floral": ("floral", "flower", "cicek", "çiçek"),
        "Leopard": ("leopard print", "leopar desen", "leopard textile"),
    },
    "Object": {
        "Lantern": ("lantern", "fener"),
        "Crown": ("crown", "tas", "taç", "kral tacı"),
        "Heart": ("heart", "kalp"),
    },
    # Global Object categories: these are separate from textile pattern categories.
    # They are used by the visual-object layer and surfaced in category predictions.
    "Global Object": {
        "Person": ("person", "human", "insan", "kisi", "kişi"),
        "Vehicle": ("vehicle", "arac", "araç", "tasit", "taşıt"),
        "Car": ("car", "araba", "otomobil"),
        "Bicycle": ("bicycle", "bisiklet"),
        "Motorcycle": ("motorcycle", "motosiklet"),
        "Bird": ("bird", "kuş", "kus"),
        "Cat": ("cat", "kedi"),
        "Dog": ("dog", "köpek", "kopek"),
        "Animal": ("animal", "hayvan"),
        "Table": ("table", "masa"),
        "Chair": ("chair", "sandalye"),
        "Phone": ("phone", "telefon"),
        "Laptop": ("laptop", "dizüstü", "dizustu"),
    },
    # Marka ve logo düzeltmeleri ayrı bir sınıf olarak tutulur.
    # Bunlar Pattern DNA ailesi değildir; kullanıcı düzeltme ekranında
    # doğrudan öğretilebilir ve arama metin indeksine yazılır.
    "Kişi": {"Ünlü": ("ünlü", "unlu", "celebrity")},
    "Marka": {
        "Dolce Gabbana": ("dolce gabbana", "dolce", "gabbana", "dg"),
        "Gucci": ("gucci", "gg", "gg supreme", "double g", "interlocking g"),
        "Louis Vuitton": ("louis vuitton", "louis", "vuitton", "lv", "lv monogram"),
        "Yves Saint Laurent": ("yves saint laurent", "saint laurent", "ysl", "saint", "laurent"),
        "Christian Dior": ("christian dior", "dior", "cd", "dior oblique"),
        "Fendi": ("fendi", "ff", "fendi ff"),
        "Burberry": ("burberry", "tb", "nova check"),
        "Michael Kors": ("michael kors", "mk", "michael"),
        "Celine": ("celine", "céline"),
        "Chanel": ("chanel",),
        "Hermes": ("hermes", "hermès"),
        "Prada": ("prada",),
        "Versace": ("versace",),
        "Valentino": ("valentino",),
        "Balenciaga": ("balenciaga",),
        "Givenchy": ("givenchy",),
        "Armani": ("armani", "giorgio armani", "emporio armani"),
        "Moncler": ("moncler",),
        "Moschino": ("moschino",),
        "Ferragamo": ("ferragamo", "salvatore ferragamo"),
        "Balmain": ("balmain",),
        "Alexander McQueen": ("alexander mcqueen", "mcqueen"),
        "Ralph Lauren": ("ralph lauren",),
        "Tommy Hilfiger": ("tommy hilfiger",),
        "Calvin Klein": ("calvin klein", "ck"),
        "Amiri": ("amiri", "amiri los angeles"),
        "Nike": ("nike",),
        "Adidas": ("adidas",),
    },
    "Logo": {
        "Marka Logosu": ("brand logo", "marka logosu", "marka logo", "logo"),
        "Monogram": ("monogram", "monogram logo"),
        "Yazı Logo": ("wordmark", "yazı logo", "yazi logo", "logo text"),
        "Amblem": ("emblem", "amblem", "arma"),
        "Sembol": ("symbol", "sembol"),
        "Tekrarlı Logo": ("repeated logo", "tekrarlı logo", "tekrar eden logo"),
        "Lüks Logo": ("luxury logo", "lüks logo", "luks logo"),
    },
    "Araç Detayı": {
        "Araba": ("araba", "otomobil", "car"), "SUV": ("suv", "arazi aracı", "arazi araci"),
        "Pickup": ("pickup", "pick-up"), "Spor Araba": ("spor araba", "spor otomobil", "sports car"),
        "Sedan": ("sedan",), "Hatchback": ("hatchback",), "Coupe": ("coupe", "coupe car"),
        "Cabrio": ("cabrio", "convertible", "üstü açılır"), "Minivan": ("minivan",),
        "Polis Arabası": ("polis arabası", "polis arabasi", "police car"),
        "Bisiklet": ("bisiklet", "bicycle"), "Yol Bisikleti": ("yol bisikleti", "road bike"),
        "Dağ Bisikleti": ("dağ bisikleti", "dag bisikleti", "mountain bike", "mtb"),
        "Şehir Bisikleti": ("şehir bisikleti", "sehir bisikleti", "city bike"),
        "Elektrikli Bisiklet": ("elektrikli bisiklet", "electric bike", "e-bike"),
        "BMX": ("bmx",), "Scooter": ("scooter", "skuter"), "Kaykay": ("kaykay", "skateboard"),
        "Motosiklet": ("motosiklet", "motorcycle"), "Kamyon": ("kamyon", "truck"),
        "Otobüs": ("otobüs", "otobus", "bus"), "Tren": ("tren", "train"),
        "Tramvay": ("tramvay", "tram"), "Metro": ("metro", "subway"), "Uçak": ("uçak", "ucak", "airplane"),
        "Helikopter": ("helikopter",), "Tekne": ("tekne", "boat"), "Gemi": ("gemi", "ship"),
    },
    "Mobilya Detayı": {
        "Masa": ("masa", "table"), "Yemek Masası": ("yemek masası", "yemek masasi", "dining table"),
        "Sehpa": ("sehpa", "coffee table"), "Çalışma Masası": ("çalışma masası", "calisma masasi", "desk"),
        "Konsol Masa": ("konsol", "console table"), "Yan Sehpa": ("yan sehpa", "side table"),
        "Sandalye": ("sandalye", "chair"), "Ofis Sandalyesi": ("ofis sandalyesi", "office chair"),
        "Berjer": ("berjer", "armchair"), "Kanepe": ("kanepe", "sofa", "couch"),
        "Yatak": ("yatak", "bed"), "Gardırop": ("gardırop", "gardrop", "wardrobe"),
        "Dolap": ("dolap", "cabinet"), "Kitaplık": ("kitaplık", "kitaplik", "bookshelf"),
        "Şifonyer": ("şifonyer", "sifonyer", "dresser"), "Komodin": ("komodin", "nightstand"),
        "Bank": ("bank", "bench"), "Tabure": ("tabure", "stool"), "Raf": ("raf", "shelf"),
    },
    "Araba Markası": {
        "Togg": ("togg",), "BMW": ("bmw",), "Mercedes": ("mercedes", "mercedes-benz"),
        "Audi": ("audi",), "Toyota": ("toyota",), "Ford": ("ford",),
        "Volkswagen": ("volkswagen", "vw"), "Volvo": ("volvo",), "Honda": ("honda",),
        "Hyundai": ("hyundai",), "Kia": ("kia",), "Tesla": ("tesla",),
        "Renault": ("renault",), "Peugeot": ("peugeot",), "Citroen": ("citroen", "citroën"),
        "Fiat": ("fiat",), "Opel": ("opel",), "Porsche": ("porsche",),
        "Lexus": ("lexus",), "Jaguar": ("jaguar",), "Land Rover": ("land rover",),
        "Skoda": ("skoda", "škoda"), "Seat": ("seat",), "Mazda": ("mazda",),
        "Nissan": ("nissan",), "Subaru": ("subaru",), "Suzuki": ("suzuki",),
        "Chevrolet": ("chevrolet", "chevy"), "Jeep": ("jeep",), "Dodge": ("dodge",),
        "Chrysler": ("chrysler",), "Genesis": ("genesis",), "Infiniti": ("infiniti",),
        "Acura": ("acura",),
    },
    "Hayvan Detayı": {
        "Kedi": ("kedi", "cat"), "Yavru Kedi": ("yavru kedi", "kitten", "baby cat"),
        "Persian Cat": ("iran kedisi", "persian cat"), "Siamese Cat": ("siamese cat", "siamese"),
        "British Shorthair": ("british shorthair",), "Maine Coon": ("maine coon",),
        "Scottish Fold": ("scottish fold",), "Köpek": ("köpek", "kopek", "dog"),
        "Yavru Köpek": ("yavru köpek", "yavru kopek", "puppy"), "Labrador": ("labrador",),
        "German Shepherd": ("german shepherd", "alman kurdu"), "Golden Retriever": ("golden retriever",),
        "Bulldog": ("bulldog",), "Poodle": ("poodle", "kaniş", "kanis"),
        "Kuş": ("kuş", "kus", "bird"), "Karga": ("karga", "crow"), "Kartal": ("kartal", "eagle"),
        "Serçe": ("serçe", "serce", "sparrow"), "Güvercin": ("güvercin", "guvercin", "pigeon"),
        "Papağan": ("papağan", "papagan", "parrot"), "Baykuş": ("baykuş", "baykus", "owl"),
        "Flamingo": ("flamingo",), "Tavus Kuşu": ("tavus kuşu", "tavus", "peacock"),
        "Penguen": ("penguen", "penguin"), "Martı": ("martı", "marti", "seagull"),
        "Kelebek": ("kelebek", "butterfly"), "Arı": ("arı", "ari", "bee"),
        "Ahtapot": ("ahtapot", "octopus"), "Timsah": ("timsah", "crocodile"),
        "Zürafa": ("zürafa", "zurafa", "giraffe"), "Fil": ("fil", "elephant"),
        "At": ("at", "horse"), "İnek": ("inek", "cow"), "Koyun": ("koyun", "sheep"),
    },
}

PARENT_ALIASES: dict[str, tuple[str, ...]] = {
    "Animal": ("animal", "hayvan"),
    "Animal Print": ("animal print", "hayvan derisi"),
    "Floral": ("floral", "flower", "cicek", "çiçek"),
    "Camouflage": ("camouflage", "camo", "kamuflaj"),
    "Geometric": ("geometric", "geometrik"),
    "Plaid Check": ("plaid", "check", "ekose"),
    "Stripe": ("stripe", "cizgi", "çizgi"),
    "Lace": ("lace", "dantel"),
    "Drawing Illustration": ("drawing", "illustration", "cizim", "çizim"),
    "Typography Text": ("typography", "text pattern", "yazi deseni", "yazı deseni"),
    "Monogram Logo": ("monogram", "logo", "logo desen"),
    "Paisley Scarf": ("paisley", "sal", "şal"),
    "Baroque Ornament": ("baroque", "barok", "ornament"),
    "Marble Abstract": ("marble", "mermer", "abstract"),
    "Texture Ground": ("texture", "doku"),
    "Textile": ("textile", "tekstil", "kumas desen", "kumaş desen"),
    "Object": ("object", "nesne", "obje"),
    "Global Object": ("global object", "global nesne", "nesne tanıma", "nesne", "obje"),
    "Marka": ("brand", "marka"),
    "Kişi": ("kişi", "kisi", "person"),
    "Logo": ("logo", "logosu", "marka logosu"),
    "Araba Markası": ("car brand", "araba markası", "araba markasi", "otomobil markası", "otomobil markasi"),
    "Hayvan Detayı": ("animal detail", "hayvan türü", "hayvan turu", "kedi türü", "kedi turu", "kuş türü", "kus turu"),
    "Araç Detayı": ("vehicle detail", "araç türü", "arac turu", "bisiklet türü", "bisiklet turu", "araba türü", "araba turu"),
    "Mobilya Detayı": ("furniture detail", "mobilya türü", "mobilya turu", "masa türü", "masa turu"),
}


@dataclass(frozen=True)
class CategoryMatch:
    category_path: str = ""
    primary_family: str = ""
    primary_subtype: str = ""
    aliases: tuple[str, ...] = ()


def category_path(parent: str, child: str = "") -> str:
    return f"{parent}/{child}" if child else parent


def aliases_for_path(path: str) -> tuple[str, ...]:
    parent, _, child = (path or "").partition("/")
    parent = parent.strip()
    child = child.strip()
    aliases = list(PARENT_ALIASES.get(parent, (parent,)))
    if child:
        aliases.extend(CATEGORY_TREE.get(parent, {}).get(child, (child,)))
    return tuple(dict.fromkeys(a for a in aliases if a))


def _category_alias_hit(alias: str, norm: str) -> bool:
    """Kısa alias'larda yanlış alt dize eşleşmesini engelle."""
    token = normalize_turkish(alias or "").strip()
    if not token:
        return False
    clean_norm = re.sub(r"[^a-z0-9]+", " ", normalize_turkish(norm or "")).strip()
    clean_token = re.sub(r"[^a-z0-9]+", " ", token).strip()
    if not clean_norm or not clean_token:
        return False
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(clean_token)}(?![a-z0-9])", clean_norm))


def resolve_category_query(text: str, db_path: str = "") -> CategoryMatch:
    norm = normalize_turkish(text or "")
    if not norm:
        return CategoryMatch()
    child_matches: list[tuple[int, str, str]] = []
    parent_only: list[tuple[int, str]] = []
    for parent, children in CATEGORY_TREE.items():
        for child, aliases in children.items():
            for alias in aliases:
                token = normalize_turkish(alias)
                if token and _category_alias_hit(token, norm):
                    child_matches.append((len(token), parent, child))
        for alias in PARENT_ALIASES.get(parent, ()):
            token = normalize_turkish(alias)
            if token and _category_alias_hit(token, norm):
                parent_only.append((len(token), parent))
    if child_matches:
        # Marka/araç markası/hayvan ayrıntısı gibi özel sınıflar,
        # "DG logo", "BMW araba", "yavru kedi" gibi bileşik sorgularda
        # genel "Logo" veya "Global Object" eşleşmesinin önüne geçsin.
        priority = {"Marka": 8, "Araba Markası": 8, "Hayvan Detayı": 7, "Kişi": 8}
        _, parent, child = max(
            child_matches,
            key=lambda item: (
                item[0] + priority.get(item[1], 0),
                priority.get(item[1], 0),
                item[0],
                item[2],
            ),
        )
    elif parent_only:
        _, parent = max(parent_only, key=lambda item: item[0])
        child = ""
    else:
        # A bare, human-looking proper name (e.g. "Marilyn Monroe" or
        # "Hülya Koçyiğit") belongs to the person-label namespace. This is
        # text/metadata routing only; it does not identify a person from pixels.
        raw = str(text or "").strip()
        tokens = [t for t in re.split(r"\s+", raw) if t]
        if 2 <= len(tokens) <= 5 and all(
            any(ch.isupper() for ch in tok[:1]) for tok in tokens if tok
        ):
            child = re.sub(r"\s+", " ", raw).strip()
            parent = "Kişi"
        else:
            dyn_path = ""
            if db_path:
                try:
                    from core.category_memory import resolve_dynamic_category

                    dyn_path = resolve_dynamic_category(db_path, text) or ""
                except Exception:
                    dyn_path = ""
            if not dyn_path:
                return CategoryMatch()
            parent, _, child = dyn_path.partition("/")
            parent = parent.strip()
            child = child.strip()
    if parent == "Kişi" and child and not any(
        normalize_turkish(child) == normalize_turkish(c)
        for c in CATEGORY_TREE.get("Kişi", {})
    ):
        # Keep the legacy/UI spelling used by the person category tests.
        path = f"Kişi / Ünlü/{child}"
    else:
        path = category_path(parent, child)
    return CategoryMatch(
        path,
        normalize_turkish(parent),
        normalize_turkish(child),
        aliases_for_path(path),
    )


def is_category_descendant(selected_path: str, candidate_path: str) -> bool:
    selected = (selected_path or "").strip("/")
    candidate = (candidate_path or "").strip("/")
    return bool(
        selected and (candidate == selected or candidate.startswith(selected + "/"))
    )


def metadata_for_path(path: str, *, manual: bool = False) -> dict[str, object]:
    parent, _, child = (path or "").partition("/")
    return {
        "category_path": path,
        "primary_family": normalize_turkish(parent),
        "primary_subtype": normalize_turkish(child),
        "manual_category_path": path if manual else "",
        "category_confidence": 1.0 if manual else 0.0,
        "category_source": "manual_user" if manual else "auto",
        "category_aliases": list(aliases_for_path(path)),
    }


LEGACY_CATEGORY_MAP: dict[str, str] = {
    "animal_print": "Animal Print",
    "floral": "Floral",
    "marble_abstract": "Marble Abstract",
    "geometric": "Geometric",
    "plaid_check": "Plaid Check",
    "stripe": "Stripe",
    "lace": "Lace",
    "monogram_logo": "Monogram Logo",
    "paisley": "Paisley Scarf",
    "scarf_border": "Paisley Scarf/Border",
    "baroque": "Baroque Ornament/Baroque",
    "chain": "Baroque Ornament/Chain",
    "texture_ground": "Texture Ground",
    "animal": "Animal",
    "textile": "Textile",
    "object": "Object",
}


def category_from_legacy(family: str, subtype: str = "", animal_type: str = "") -> str:
    parent = LEGACY_CATEGORY_MAP.get((family or "").strip(), "")
    if not parent:
        return ""
    candidate = animal_type or subtype
    if family == "animal_print" and candidate:
        labels = {
            "leopard": "Leopard",
            "zebra": "Zebra",
            "snake": "Snake Skin",
            "tiger": "Tiger",
            "crocodile": "Crocodile",
            "cow": "Cow",
            "giraffe": "Giraffe",
        }
        child = labels.get(candidate, "Mixed Animal Print")
        if parent == "Textile" and candidate == "leopard":
            return category_path("Textile", "Leopard")
        return category_path(parent, child)
    if family == "animal" and candidate:
        animal_labels = {
            "cat": "Cat",
            "dog": "Dog",
            "rabbit": "Rabbit",
            "octopus": "Octopus",
            "snake": "Snake",
            "snake_animal": "Snake",
            "butterfly": "Butterfly",
            "bird": "Bird",
            "fish": "Fish",
        }
        return category_path(parent, animal_labels.get(candidate, "Other Animal"))
    if family == "textile" and candidate:
        textile_labels = {
            "plaid": "Plaid",
            "stripe": "Stripe",
            "floral": "Floral",
            "leopard": "Leopard",
        }
        return category_path("Textile", textile_labels.get(candidate, "Floral"))
    if family == "object" and candidate:
        object_labels = {
            "lantern": "Lantern",
            "crown": "Crown",
            "heart": "Heart",
        }
        return category_path("Object", object_labels.get(candidate, "Heart"))
    subtype_norm = normalize_turkish(candidate)
    for child, aliases in CATEGORY_TREE.get(parent, {}).items():
        if subtype_norm and any(normalize_turkish(a) == subtype_norm for a in aliases):
            return category_path(parent, child)
    return parent


def pattern_fields_for_path(path: str) -> dict[str, str]:
    """Kategori yolunu legacy pattern_family / subtype alanlarına çevir."""
    parent, _, child = (path or "").partition("/")
    parent = parent.strip()
    child = child.strip()
    if not parent:
        return {}
    out: dict[str, str] = {
        "pattern_family": "",
        "pattern_subtype": "",
        "animal_print_type": "",
        "brand_name": "",
    }

    if parent == "Animal Print":
        out["pattern_family"] = "animal_print"
        animal_map = {
            "Leopard": "leopard",
            "Zebra": "zebra",
            "Snake Skin": "snake",
            "Tiger": "tiger",
            "Crocodile": "crocodile",
            "Cow": "cow",
            "Giraffe": "giraffe",
        }
        out["animal_print_type"] = animal_map.get(child, "mixed_animal")
        return out

    if parent == "Animal":
        out["pattern_family"] = "unknown"
        animal_map = {
            "Cat": "cat",
            "Dog": "dog",
            "Rabbit": "rabbit",
            "Octopus": "octopus",
            "Snake": "snake_animal",
            "Butterfly": "butterfly",
            "Bird": "bird",
            "Fish": "fish",
        }
        out["pattern_subtype"] = animal_map.get(child, "animal")
        return out

    if parent == "Floral":
        out["pattern_family"] = "floral"
        floral_map = {
            "Rose": "rose",
            "Daisy": "daisy",
            "Leaf": "leaf",
            "Small Floral": "small_floral",
            "Ditsy Floral": "ditsy_floral",
            "Big Flower": "big_flower",
            "Allover Floral": "mixed_floral",
            "Seamless Floral": "mixed_floral",
        }
        out["pattern_subtype"] = floral_map.get(child, "mixed_floral")
        return out

    if parent == "Camouflage":
        out["pattern_family"] = "texture_ground"
        camo_map = {
            "Forest Camo": "camouflage",
            "Military Camo": "military_camo",
            "Digital Camo": "digital_camo",
            "Urban Camo": "urban_camo",
            "Abstract Camo": "abstract_camo",
        }
        out["pattern_subtype"] = camo_map.get(child, "camouflage")
        return out

    if parent == "Geometric":
        out["pattern_family"] = "geometric"
        return out
    if parent == "Plaid Check":
        out["pattern_family"] = "plaid_check"
        return out
    if parent == "Stripe":
        out["pattern_family"] = "stripe"
        return out
    if parent == "Lace":
        out["pattern_family"] = "lace"
        return out
    if parent == "Drawing Illustration":
        out["pattern_family"] = "unknown"
        out["pattern_subtype"] = "illustration"
        return out
    if parent == "Typography Text":
        out["pattern_family"] = "typography_text"
        out["pattern_subtype"] = "text_pattern"
        return out
    if parent == "Monogram Logo":
        out["pattern_family"] = "monogram_logo"
        return out
    if parent == "Paisley Scarf":
        out["pattern_family"] = "paisley"
        return out
    if parent == "Baroque Ornament":
        out["pattern_family"] = "baroque"
        return out
    if parent == "Marble Abstract":
        out["pattern_family"] = "marble_abstract"
        return out
    if parent == "Texture Ground":
        out["pattern_family"] = "texture_ground"
        return out
    if parent == "Textile":
        out["pattern_family"] = "textile"
        textile_map = {
            "Plaid": "plaid",
            "Stripe": "stripe",
            "Floral": "floral",
            "Leopard": "leopard",
        }
        out["pattern_subtype"] = textile_map.get(child, "floral")
        return out
    if parent == "Object":
        out["pattern_family"] = "object"
        object_map = {"Lantern": "lantern", "Crown": "crown", "Heart": "heart"}
        out["pattern_subtype"] = object_map.get(child, "heart")
        return out
    if parent == "Kişi":
        # Kişi/ünlü baskısı bir içerik etiketi olarak tutulur; yüz kimliği çıkarımı yapmaz.
        out["pattern_family"] = "person_subject"
        person_label = child.rsplit("/", 1)[-1].strip() if child else "person"
        out["pattern_subtype"] = normalize_turkish(person_label)
        return out

    if parent == "Marka":
        out["pattern_family"] = "monogram_logo"
        out["pattern_subtype"] = "brand"
        out["brand_name"] = child
        return out
    if parent == "Logo":
        out["pattern_family"] = "monogram_logo"
        logo_map = {
            "Marka Logosu": "brand_logo",
            "Monogram": "monogram",
            "Yazı Logo": "wordmark",
            "Amblem": "emblem",
            "Sembol": "symbol",
            "Tekrarlı Logo": "repeated_logo",
            "Lüks Logo": "luxury_logo",
        }
        out["pattern_subtype"] = logo_map.get(child, "logo")
        return out
    if parent in {"Araba Markası", "Araç Detayı", "Mobilya Detayı", "Hayvan Detayı"}:
        out["pattern_subtype"] = normalize_turkish(child)
        return out
    return out


def infer_category_path(
    texture_map: dict | None,
    *,
    filename: str = "",
    path: str = "",
    ocr_text: str = "",
) -> str:
    """Doku sınıflandırması + dosya adından otomatik kategori yolu."""
    from core.texture_profile import TextureProfile

    tm = texture_map or {}
    prof = TextureProfile.from_dict(tm)
    family = (prof.pattern_family or tm.get("pattern_family") or "").strip()
    animal = (prof.animal_print_type or tm.get("animal_print_type") or "").strip()
    subtype = (prof.pattern_subtype or tm.get("pattern_subtype") or "").strip()

    # Güvenilir marka kanıtı varsa Monogram/Logo ailesinin altında kaybolmasın.
    # Yalnızca daha önce indekslenmiş metadata/manuel öğrenme kanıtı kullanılır;
    # dosya adı tek başına marka kanıtı değildir.
    tm_brand = str(tm.get("brand_name") or "").strip()
    sem = tm.get("semantic_tags") or {}
    refs = sem.get("brand_references", []) if isinstance(sem, dict) else []
    if tm_brand:
        from core.brand_aliases import resolve_brand_alias, normalize_brand_key
        canonical = resolve_brand_alias(tm_brand) or tm_brand
        return category_path("Marka", canonical.title() if canonical else tm_brand)
    if refs:
        from core.brand_aliases import normalize_brand_key, resolve_brand_alias
        for ref in refs:
            raw_ref = str(ref or "").strip()
            if not raw_ref:
                continue
            # AI/semantic katmanı açıkça bir marka referansı verdiyse marka
            # sözlüğünde bulunmasa bile adı kaybetme. Böylece "Polo" gibi
            # kaynak listesinde olmayan müşteri markaları da Marka altında
            # tutulabilir.
            canonical = resolve_brand_alias(raw_ref) or normalize_turkish(raw_ref)
            if canonical:
                return category_path("Marka", canonical.title())

    # Legacy indekslerde marka adı yalnızca OCR'da bulunabilir. OCR kanıtı,
    # bilinen marka alias'larından biriyle eşleşiyorsa marka kategorisini
    # Monogram/Logo gibi genel ailelerin önüne al.
    if ocr_text:
        from core.brand_aliases import expand_brand_terms, resolve_brand_alias
        for brand in expand_brand_terms(str(ocr_text)):
            canonical = resolve_brand_alias(str(brand)) or str(brand).strip()
            if canonical:
                return category_path("Marka", canonical.title())

    # Person/celebrity labels from explicit text/semantic metadata take
    # precedence over a generic "Typography Text" family. This is still a
    # content-label route; it never identifies a person from pixels.
    person_candidates = []
    if isinstance(sem, dict):
        person_candidates.extend(sem.get("motifs") or [])
        person_candidates.extend(sem.get("entities") or [])
        person_candidates.extend(sem.get("people") or [])
    person_candidates.append(str(ocr_text or ""))
    person_candidates.extend([str(filename or ""), str(path or "")])
    for candidate in person_candidates:
        raw = str(candidate or "").strip()
        tokens = [x for x in re.split(r"\s+", raw) if x]
        if not (2 <= len(tokens) <= 5):
            continue
        # Reject obvious generic text/design phrases.
        norm_raw = normalize_turkish(raw)
        if any(x in norm_raw.split() for x in (
            "logo", "desen", "pattern", "flower", "floral", "text", "yazi",
            "write", "print", "portrait", "portre"
        )):
            # "portrait Marilyn Monroe" is still a useful person cue; strip
            # the generic leading token below instead of rejecting it.
            if "portrait" in norm_raw.split() or "portre" in norm_raw.split():
                tokens = [x for x in tokens if normalize_turkish(x) not in {"portrait", "portre"}]
            else:
                continue
        if len(tokens) >= 2:
            name = re.sub(r"\s+", " ", " ".join(tokens[:4])).strip()
            if all(sum(ch.isalpha() for ch in tok) >= 2 for tok in tokens[:2]):
                # OCR often arrives uppercase; title-case only the generated label.
                display = " ".join(x[:1].upper() + x[1:].lower() for x in name.split())
                return f"Kişi / Ünlü/{display}"

    legacy = category_from_legacy(family, subtype, animal)
    if legacy:
        return legacy

    for text in (
        filename,
        path,
        " ".join(filter(None, [family, animal, subtype, prof.texture_family])),
    ):
        match = resolve_category_query(str(text))
        if match.category_path:
            return match.category_path
    return ""


def apply_auto_category_to_texture_map(
    texture_map: dict | None,
    *,
    filename: str = "",
    path: str = "",
) -> tuple[dict, str]:
    """Manuel etiket yoksa texture_map'e otomatik category_path yazar."""
    from core.manual_label_guard import is_manual_labeled
    from core.texture_profile import TextureProfile

    tm = dict(texture_map or {})
    if is_manual_labeled(tm):
        existing = str(
            tm.get("manual_category_path") or tm.get("category_path") or ""
        ).strip()
        return tm, existing

    cat_path = infer_category_path(tm, filename=filename, path=path)
    if not cat_path:
        return tm, ""

    prof = TextureProfile.from_dict(tm)
    tm.update(
        {
            "category_path": cat_path,
            "category_source": "auto_index",
            "category_aliases": list(aliases_for_path(cat_path)),
            "category_confidence": float(prof.classification_confidence or 0.45),
        }
    )
    return tm, cat_path


def parent_categories(db_path: str = "") -> list[str]:
    roots, _ = build_category_tree_hierarchy(db_path)
    return roots


def child_categories(parent: str, db_path: str = "") -> list[str]:
    _, children = build_category_tree_hierarchy(db_path)
    if parent in children:
        return list(children[parent])
    # Parent may be dynamic-only with different casing vs map keys.
    from core.category_memory import _key

    pk = _key(parent)
    for name, kids in children.items():
        if _key(name) == pk:
            return list(kids)
    return list(CATEGORY_TREE.get(parent, {}).keys())


def _collapse_marka_labels(kids: list[str], db_path: str) -> list[str]:
    try:
        from core.brand_aliases import canonical_brand_display, normalize_brand_key

        collapsed: list[str] = []
        seen: set[str] = set()
        for name in kids:
            key = normalize_brand_key(name)
            if not key or key in seen:
                continue
            seen.add(key)
            collapsed.append(canonical_brand_display(name, db_path) or name)
        return collapsed
    except Exception:
        return kids


def build_category_tree_hierarchy(
    db_path: str = "",
) -> tuple[list[str], dict[str, list[str]]]:
    """One-pass parent→children map (same merge rules as legacy helpers).

    Loads ``concepts`` and ``category_memory`` at most once per call so the
    UI panel does not pay N+1 SQLite opens across ~50 parents.
    """
    roots = list(CATEGORY_TREE.keys())
    concept_rows: list[dict] = []
    dyn_by_parent: dict[str, list[str]] = {}
    if db_path:
        try:
            from core.category_memory import _key, all_dynamic_children_grouped, dynamic_parents

            seen = {_key(p) for p in roots}
            for p in dynamic_parents(db_path):
                k = _key(p)
                if k and k not in seen:
                    seen.add(k)
                    roots.append(p)
        except Exception:
            pass
        try:
            from core.category_memory import _key
            from core.concept_registry import concepts_readonly

            concept_rows = list(concepts_readonly(db_path))
            seen = {_key(p) for p in roots}
            for row in concept_rows:
                status = str(row.get("status") or "active")
                if status in ("inactive", "retired"):
                    continue
                parent = str(row.get("parent") or "").strip()
                if not parent:
                    continue
                k = _key(parent)
                if k and k not in seen:
                    seen.add(k)
                    roots.append(parent)
        except Exception:
            concept_rows = []
        try:
            from core.category_memory import all_dynamic_children_grouped

            dyn_by_parent = all_dynamic_children_grouped(db_path)
        except Exception:
            dyn_by_parent = {}

    from core.category_memory import _key

    children_map: dict[str, list[str]] = {}
    for parent in roots:
        kids = list(CATEGORY_TREE.get(parent, {}).keys())
        seen = {_key(c) for c in kids}
        for c in dyn_by_parent.get(_key(parent), []):
            k = _key(c)
            if k and k not in seen:
                seen.add(k)
                kids.append(c)
        pk = _key(parent)
        for row in concept_rows:
            status = str(row.get("status") or "active")
            if status in ("inactive", "retired"):
                continue
            if _key(str(row.get("parent") or "")) != pk:
                continue
            can = str(row.get("canonical") or "").strip()
            k = _key(can)
            if can and k and k not in seen:
                seen.add(k)
                kids.append(can)
        if parent.casefold() == "marka":
            kids = _collapse_marka_labels(kids, db_path)
        children_map[parent] = kids
    return roots, children_map
