"""Vezir Textile Knowledge Base — sektör bilgisi (aile, motif, terim, ilişki).

Arama skorlaması bu tabanı kullanır. Mevcut taxonomy id'leri korunur;
alt aileler `parent` ile bağlanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

# Mevcut motor / taxonomy ile uyumlu kök aileler
ROOT_FAMILIES: tuple[str, ...] = (
    "animal_print",
    "floral",
    "marble_abstract",
    "geometric",
    "plaid_check",
    "stripe",
    "paisley",
    "lace",
    "monogram_logo",
    "baroque",
    "chain",
    "scarf_border",
    "texture_ground",
    "typography_text",
    "document",
    "plain",
    "unknown",
)

# --- Seed data (üretici genişletir) -----------------------------------------

_ROOT_TR: dict[str, str] = {
    "animal_print": "Hayvan Deseni",
    "floral": "Çiçek",
    "marble_abstract": "Mermer / Soyut",
    "geometric": "Geometrik",
    "plaid_check": "Ekose / Kare",
    "stripe": "Çizgi",
    "paisley": "Paisley / Şal",
    "lace": "Dantel",
    "monogram_logo": "Monogram / Logo",
    "baroque": "Barok",
    "chain": "Zincir",
    "scarf_border": "Şal Bordür",
    "texture_ground": "Zemin Doku",
    "typography_text": "Yazı Deseni",
    "document": "Doküman / Toile",
    "plain": "Düz",
    "unknown": "Bilinmeyen",
}

_FAMILY_SUBTYPES: dict[str, list[str]] = {
    "floral": [
        "rose", "daisy", "tulip", "peony", "orchid", "lily", "sunflower", "poppy",
        "hibiscus", "chrysanthemum", "lavender", "wildflower", "botanical_leaf",
        "fern", "palm_leaf", "tropical_bloom", "ditsy", "small_floral", "big_floral",
        "watercolor_floral", "line_floral", "vintage_floral", "garden_party",
        "herbarium", "blossom_scatter", "flower_border", "rose_garden",
    ],
    "animal_print": [
        "leopard", "cheetah", "jaguar", "tiger", "zebra", "snake", "python",
        "crocodile", "alligator", "giraffe", "cow", "dalmatian", "pony",
        "ostrich", "fish_scale", "mixed_animal", "faux_fur_print", "safari_mix",
    ],
    "geometric": [
        "triangle", "hexagon", "diamond", "circle", "square", "mosaic", "tessellation",
        "optical", "maze", "lattice", "grid", "star", "chevron_geo", "art_deco_geo",
        "islamic_geo", "tribal_geo", "pixel_geo", "3d_cube",
    ],
    "stripe": [
        "pinstripe", "pencil_stripe", "bold_stripe", "candy_stripe", "awning",
        "horizontal", "vertical", "diagonal", "broken_stripe", "space_stripe",
        "multi_stripe", "railroad", "ombre_stripe", "breton",
    ],
    "plaid_check": [
        "tartan", "gingham", "windowpane", "glen_check", "houndstooth", "shepherd",
        "buffalo", "madras", "nova_check", "mini_check", "overcheck", "blanket_plaid",
    ],
    "paisley": [
        "classic_paisley", "mini_paisley", "persian_paisley", "indian_paisley",
        "scarf_paisley", "teardrop", "boteh", "cascade_paisley",
    ],
    "lace": [
        "guipure", "chantilly", "crochet_lace", "eyelet", "broderie", "venise",
        "scallop_lace", "floral_lace", "geometric_lace",
    ],
    "marble_abstract": [
        "marble", "watercolor", "ink_bleed", "smoke", "fluid_art", "pour_paint",
        "geode", "agate", "terrazzo", "granite", "crackled", "nebula", "oil_slick",
    ],
    "baroque": [
        "ornate_scroll", "acanthus", "rococo", "versace_baroque", "medusa",
        "crest", "cartouche", "gilt_ornament",
    ],
    "monogram_logo": [
        "logo_allover", "logo_scatter", "monogram_grid", "emblem_badge",
        "letter_mark", "brand_stripe", "heritage_crest",
    ],
    "chain": [
        "curb_chain", "cable_chain", "figaro", "rope_chain", "jewelry_mix",
        "gold_chain", "silver_chain",
    ],
    "scarf_border": [
        "square_scarf", "long_scarf", "border_frame", "corner_motif",
        "hem_border", "foulard",
    ],
    "texture_ground": [
        "camouflage", "military_camo", "linen_look", "denim_look", "knit_texture",
        "jacquard_ground", "tweed_look", "felt_look", "nubuck", "suede_look",
        "canvas_look", "mesh", "rib", "cable_knit",
    ],
    "typography_text": [
        "slogan", "newspaper", "script_text", "block_letter", "graffiti",
        "ticket_print", "label_text", "map_text",
    ],
    "document": [
        "toile_de_jouy", "engraving", "map_print", "blueprint", "stamp",
        "postcard", "vintage_document",
    ],
    "plain": ["solid", "heather", "melange", "space_dye_solid"],
}

_SCALE_MODS = ("micro", "mini", "midi", "macro", "oversized")
_LAYOUT_MODS = ("allover", "border", "placement", "engineered", "panel")
_STYLE_MODS = (
    "realistic", "stylized", "watercolor", "linework", "silhouette",
    "vintage", "modern", "ethnic", "digital", "hand_drawn",
)

_MOTIF_SEEDS: dict[str, list[tuple[str, str]]] = {
    # (id, display) — synonyms added in builder
    "floral": [
        ("rose", "Gül"), ("daisy", "Papatya"), ("tulip", "Lale"), ("peony", "Şakayık"),
        ("orchid", "Orkide"), ("lily", "Zambak"), ("sunflower", "Ayçiçeği"),
        ("poppy", "Gelincik"), ("hibiscus", "Hibiskus"), ("chrysanthemum", "Krizantem"),
        ("lavender", "Lavanta"), ("violet", "Menekşe"), ("carnation", "Karanfil"),
        ("magnolia", "Manolya"), ("camellia", "Kamelya"), ("lotus", "Nilüfer"),
        ("cherry_blossom", "Kiraz Çiçeği"), ("jasmine", "Yasemin"), ("iris", "Iris"),
        ("marigold", "Kadife Çiçeği"), ("anemone", "Anemon"), ("hydrangea", "Ortanca"),
        ("wisteria", "Mor Salkım"), ("protea", "Protea"), ("poinsettia", "Atatürk Çiçeği"),
        ("clover", "Yonca"), ("thistle", "Devedikeni"), ("wheat", "Başak"),
        ("olive_branch", "Zeytin Dalı"), ("eucalyptus", "Okaliptüs"), ("ivy", "Sarmaşık"),
        ("maple_leaf", "Akçaağaç"), ("oak_leaf", "Meşe"), ("bamboo", "Bambu"),
        ("cactus", "Kaktüs"), ("succulent", "Sukulent"), ("mushroom", "Mantar"),
        ("berry", "Berry"), ("fruit_mix", "Meyve Karışımı"), ("citrus", "Narenciye"),
        ("pineapple", "Ananas"), ("palm", "Palmiye"), ("banana_leaf", "Muz Yaprağı"),
        ("monstera", "Monstera"), ("fern_frond", "Eğrelti"), ("acorn", "Meşe Palamudu"),
        ("pinecone", "Kozalak"), ("willow", "Söğüt"), ("birch", "Huş"),
        ("dandelion", "Karahindiba"), ("forget_me_not", "Unutma Beni"),
        ("sweet_pea", "Bezelye Çiçeği"), ("freesia", "Freesia"), ("ranunculus", "Düğün Çiçeği"),
        ("calla", "Gala"), ("bird_of_paradise", "Cennet Kuşu"), ("plumeria", "Plumeria"),
        ("bougainvillea", "Begonvil"), ("azalea", "Açelya"), ("gardenia", "Gardenya"),
        ("lilac", "Leylak"), ("hyacinth", "Sümbül"), ("daffodil", "Nergis"),
        ("snowdrop", "Kardelen"), ("bluebell", "Çan Çiçeği"), ("foxglove", "Yüksükotu"),
        ("edelweiss", "Edelweiss"), ("heather", "Fundalık"), ("reed", "Saz"),
        ("coral_floral", "Mercan Çiçek"), ("wild_rose", "Yabani Gül"),
        ("tea_rose", "Çay Gülü"), ("english_rose", "İngiliz Gülü"),
    ],
    "animal": [
        ("leopard_spot", "Leopar Benek"), ("cheetah_spot", "Çita Benek"),
        ("tiger_stripe", "Kaplan Çizgi"), ("zebra_stripe", "Zebra"),
        ("snake_scale", "Yılan Pul"), ("python_skin", "Python"),
        ("crocodile_skin", "Timsah"), ("giraffe_spot", "Zürafa"),
        ("cow_hide", "İnek Deseni"), ("dalmatian_spot", "Dalmaçyalı"),
        ("fish_scale", "Balık Pul"), ("peacock_feather", "Tavus"),
        ("eagle", "Kartal"), ("owl", "Baykuş"), ("flamingo", "Flamingo"),
        ("parrot", "Papağan"), ("hummingbird", "Sinekkapan"), ("swan", "Kuğu"),
        ("horse", "At"), ("deer", "Geyik"), ("fox", "Tilki"), ("wolf", "Kurt"),
        ("bear", "Ayı"), ("lion", "Aslan"), ("elephant", "Fil"), ("giraffe", "Zürafa"),
        ("monkey", "Maymun"), ("panda", "Panda"), ("koala", "Koala"),
        ("butterfly", "Kelebek"), ("dragonfly", "Yusufçuk"), ("bee", "Arı"),
        ("ladybug", "Uğur Böceği"), ("beetle", "Böcek"), ("spider_web", "Örümcek Ağı"),
        ("seahorse", "Denizatı"), ("starfish", "Deniz Yıldızı"), ("shell", "Deniz Kabuğu"),
        ("coral", "Mercan"), ("jellyfish", "Denizanası"), ("whale", "Balina"),
        ("dolphin", "Yunus"), ("shark", "Köpekbalığı"), ("octopus", "Ahtapot"),
        ("crab", "Yengeç"), ("lobster", "Istakoz"), ("turtle", "Kaplumbağa"),
        ("dinosaur", "Dinozor"), ("dragon", "Ejderha"), ("phoenix", "Anka"),
        ("unicorn", "Unicorn"), ("mermaid", "Denizkızı"), ("cat", "Kedi"),
        ("dog", "Köpek"), ("rabbit", "Tavşan"), ("squirrel", "Sincap"),
        ("hedgehog", "Kirpi"), ("frog", "Kurbağa"), ("chameleon", "Bukalemun"),
        ("lizard", "Kertenkele"), ("gecko", "Geko"), ("iguana", "İguana"),
    ],
    "geometric": [
        ("circle", "Daire"), ("dot", "Nokta"), ("polka", "Puantiyeli"),
        ("square", "Kare"), ("diamond", "Baklava"), ("triangle", "Üçgen"),
        ("hexagon", "Altıgen"), ("octagon", "Sekizgen"), ("star", "Yıldız"),
        ("cross", "Haç"), ("plus", "Artı"), ("chevron", "Chevron"),
        ("zigzag", "Zigzag"), ("wave", "Dalga"), ("spiral", "Spiral"),
        ("swirl", "Girdap"), ("arabesque", "Arabesk"), ("trellis", "Kafes"),
        ("lattice", "Latis"), ("honeycomb", "Petek"), ("brick", "Tuğla"),
        ("herringbone", "Balıksırtı"), ("houndstooth", "Kazayağı"),
        ("argyle", "Argyle"), ("ikat_geo", "Ikat Geo"), ("pixel", "Piksel"),
        ("moire", "Muar"), ("optical_square", "Optik Kare"),
        ("nested_square", "İç İçe Kare"), ("radial", "Radyal"),
        ("mandala", "Mandal"), ("kaleidoscope", "Kaleydoskop"),
        ("fractal", "Fraktal"), ("tessera", "Mozaik Kare"),
    ],
    "object": [
        ("crown", "Taç"), ("key", "Anahtar"), ("lock", "Kilit"), ("heart", "Kalp"),
        ("anchor", "Çapa"), ("compass", "Pusula"), ("ship", "Gemi"), ("plane", "Uçak"),
        ("car", "Araba"), ("bicycle", "Bisiklet"), ("camera", "Kamera"),
        ("umbrella", "Şemsiye"), ("hat", "Şapka"), ("shoe", "Ayakkabı"),
        ("handbag", "Çanta"), ("perfume", "Parfüm"), ("lipstick", "Ruj"),
        ("sunglasses", "Gözlük"), ("watch", "Saat"), ("jewelry", "Mücevher"),
        ("ring", "Yüzük"), ("earring", "Küpe"), ("necklace", "Kolye"),
        ("bracelet", "Bilezik"), ("scarf_obj", "Fular"), ("fan", "Yelpaze"),
        ("lantern", "Fener"), ("candle", "Mum"), ("book", "Kitap"),
        ("music_note", "Nota"), ("guitar", "Gitar"), ("piano", "Piyano"),
        ("dice", "Zar"), ("playing_card", "İskambil"), ("chess", "Satranç"),
        ("balloon", "Balon"), ("gift", "Hediye"), ("bow", "Fiyonk"),
        ("ribbon", "Kurdele"), ("star_ornament", "Yıldız Süs"),
        ("snowflake", "Kar Tanesi"), ("firework", "Havai Fişek"),
        ("lightning", "Şimşek"), ("cloud", "Bulut"), ("sun", "Güneş"),
        ("moon", "Ay"), ("planet", "Gezegen"), ("constellation", "Takımyıldız"),
        ("map", "Harita"), ("globe", "Küre"), ("flag", "Bayrak"),
        ("medal", "Madalya"), ("trophy", "Kupa"), ("ball", "Top"),
        ("racket", "Raket"), ("ski", "Kayak"), ("surfboard", "Sörf"),
        ("teapot", "Çaydanlık"), ("cup", "Fincan"), ("bottle", "Şişe"),
        ("wine", "Şarap"), ("cocktail", "Kokteyl"), ("ice_cream", "Dondurma"),
        ("cake", "Pasta"), ("donut", "Donut"), ("cookie", "Kurabiye"),
        ("candy", "Şeker"), ("lollipop", "Lolipop"), ("cherry", "Kiraz"),
        ("strawberry", "Çilek"), ("watermelon", "Karpuz"), ("lemon", "Limon"),
        ("orange", "Portakal"), ("apple", "Elma"), ("pear", "Armut"),
        ("grape", "Üzüm"), ("fig", "İncir"), ("pomegranate", "Nar"),
        ("coffee", "Kahve"), ("tea_leaf", "Çay Yaprağı"),
    ],
    "cultural": [
        ("ottoman", "Osmanlı"), ("kilim", "Kilim"), ("suzani", "Suzani"),
        ("ikat", "Ikat"), ("batik", "Batik"), ("shibori", "Shibori"),
        ("tie_dye", "Batik Boya"), ("african_wax", "Afrika Wax"),
        ("kente", "Kente"), ("paisley_boteh", "Boteh"), ("mandala_cult", "Mandal"),
        ("mehndi", "Kına"), ("celtic", "Kelt"), ("nordic", "Nordik"),
        ("japanese_wave", "Japon Dalga"), ("sakura", "Sakura"), ("kanji", "Kanji"),
        ("chinese_cloud", "Çin Bulutu"), ("dragon_cult", "Ejderha Motifi"),
        ("aztec", "Aztek"), ("mayan", "Maya"), ("inca", "İnka"),
        ("navajo", "Navajo"), ("tribal", "Tribal"), ("folk_flower", "Halk Çiçeği"),
        ("greek_key", "Yunan Anahtarı"), ("egyptian", "Mısır"), ("hieroglyph", "Hiyeroglif"),
        ("moroccan", "Fas"), ("andalusian", "Endülüs"), ("persian", "Fars"),
        ("indian_block", "Hint Blok"), ("russian_folk", "Rus Folk"),
        ("scandinavian", "İskandinav"), ("alpine", "Alp"), ("provence", "Provence"),
        ("toile_scene", "Toile Sahne"), ("chintz", "Chintz"),
    ],
    "brand_motif": [
        ("lv_monogram", "LV Monogram"), ("gg_supreme", "GG Supreme"),
        ("cd_oblique", "CD Oblique"), ("ff_logo", "FF Logo"), ("ysl_logo", "YSL"),
        ("tb_check", "TB Check"), ("medusa_head", "Medusa"), ("dg_crown", "DG Crown"),
        ("chanel_cc", "Chanel CC"), ("hermes_H", "Hermès H"), ("celine_triomphe", "Celine"),
        ("fendi_baguette", "Fendi"), ("prada_triangle", "Prada"), ("gucci_horsebit", "Horsebit"),
        ("burberry_nova", "Nova Check"), ("versace_greca", "Greca"),
        ("balenciaga_logo", "Balenciaga"), ("dior_cannage", "Cannage"),
        ("loewe_anagram", "Loewe"), ("bottega_intrecciato", "Intrecciato"),
    ],
}

_MOTIF_SYNONYMS: dict[str, list[str]] = {
    "rose": ["gül", "gul", "roses", "rosa"],
    "daisy": ["papatya", "marguerite"],
    "leopard_spot": ["leopar", "leopard", "leo", "jaguar print"],
    "tiger_stripe": ["kaplan", "tiger"],
    "zebra_stripe": ["zebra"],
    "snake_scale": ["yılan", "yilan", "snake", "python"],
    "paisley_boteh": ["paisley", "şal", "sal", "boteh"],
    "polka": ["puantiye", "dot", "benek"],
    "houndstooth": ["kazayağı", "kazayagi"],
    "herringbone": ["balıksırtı", "baliksirti"],
    "tartan": ["ekose", "plaid", "check"],
    "camouflage": ["kamuflaj", "camo"],
    "monstera": ["monstera leaf", "swiss cheese plant"],
    "cherry_blossom": ["sakura", "kiraz çiçeği"],
    "lv_monogram": ["lv", "louis vuitton", "vuitton"],
    "gg_supreme": ["gg", "gucci"],
    "burberry_nova": ["burberry", "nova check", "tb"],
}

_BRAND_FAMILIES: dict[str, dict[str, Any]] = {
    "louis_vuitton": {
        "label": "Louis Vuitton",
        "aliases": ["lv", "vuitton", "louis"],
        "pattern_family": "monogram_logo",
        "motifs": ["lv_monogram"],
        "styles": ["luxury", "heritage"],
    },
    "gucci": {
        "label": "Gucci",
        "aliases": ["gg", "gg supreme"],
        "pattern_family": "monogram_logo",
        "motifs": ["gg_supreme", "gucci_horsebit"],
        "styles": ["luxury", "maximalist"],
    },
    "dior": {
        "label": "Dior",
        "aliases": ["cd", "christian dior"],
        "pattern_family": "monogram_logo",
        "motifs": ["cd_oblique", "dior_cannage"],
        "styles": ["luxury", "couture"],
    },
    "fendi": {
        "label": "Fendi",
        "aliases": ["ff"],
        "pattern_family": "monogram_logo",
        "motifs": ["ff_logo", "fendi_baguette"],
        "styles": ["luxury"],
    },
    "ysl": {
        "label": "Saint Laurent",
        "aliases": ["ysl", "yves saint laurent", "saint laurent"],
        "pattern_family": "monogram_logo",
        "motifs": ["ysl_logo"],
        "styles": ["luxury", "minimal_logo"],
    },
    "burberry": {
        "label": "Burberry",
        "aliases": ["tb", "nova check"],
        "pattern_family": "plaid_check",
        "motifs": ["burberry_nova", "tb_check"],
        "styles": ["heritage", "british"],
    },
    "versace": {
        "label": "Versace",
        "aliases": ["medusa", "greca"],
        "pattern_family": "baroque",
        "motifs": ["medusa_head", "versace_greca"],
        "styles": ["baroque", "maximalist"],
    },
    "dolce_gabbana": {
        "label": "Dolce & Gabbana",
        "aliases": ["dg", "dolce", "gabbana"],
        "pattern_family": "baroque",
        "motifs": ["dg_crown"],
        "styles": ["baroque", "sicilian"],
    },
    "amiri": {
        "label": "Amiri",
        "aliases": ["amiri", "amiri los angeles"],
        "pattern_family": "typography_text",
        "motifs": [],
        "styles": ["luxury", "streetwear", "rock"],
    },
    "chanel": {
        "label": "Chanel",
        "aliases": ["cc", "chanel"],
        "pattern_family": "monogram_logo",
        "motifs": ["chanel_cc"],
        "styles": ["luxury", "tweed"],
    },
    "hermes": {
        "label": "Hermès",
        "aliases": ["hermes", "hermès"],
        "pattern_family": "scarf_border",
        "motifs": ["hermes_H"],
        "styles": ["luxury", "equestrian", "scarf"],
    },
    "celine": {
        "label": "Celine",
        "aliases": ["céline", "celine"],
        "pattern_family": "monogram_logo",
        "motifs": ["celine_triomphe"],
        "styles": ["luxury", "minimal"],
    },
    "prada": {
        "label": "Prada",
        "aliases": ["prada"],
        "pattern_family": "monogram_logo",
        "motifs": ["prada_triangle"],
        "styles": ["luxury", "minimal"],
    },
    "balenciaga": {
        "label": "Balenciaga",
        "aliases": ["balenciaga"],
        "pattern_family": "typography_text",
        "motifs": ["balenciaga_logo"],
        "styles": ["street", "logo"],
    },
    "loewe": {
        "label": "Loewe",
        "aliases": ["loewe"],
        "pattern_family": "monogram_logo",
        "motifs": ["loewe_anagram"],
        "styles": ["luxury", "craft"],
    },
    "bottega_veneta": {
        "label": "Bottega Veneta",
        "aliases": ["bottega", "intrecciato"],
        "pattern_family": "texture_ground",
        "motifs": ["bottega_intrecciato"],
        "styles": ["luxury", "woven"],
    },
    "michael_kors": {
        "label": "Michael Kors",
        "aliases": ["mk", "michael kors"],
        "pattern_family": "monogram_logo",
        "motifs": ["letter_mark"],
        "styles": ["logo"],
    },
}

_REPEAT_TYPES: dict[str, dict[str, str]] = {
    "straight": {"tr": "Düz tekrar", "en": "straight repeat"},
    "half_drop": {"tr": "Yarım düşme", "en": "half-drop"},
    "half_brick": {"tr": "Yarım tuğla", "en": "half-brick"},
    "mirror": {"tr": "Ayna", "en": "mirror repeat"},
    "rotational": {"tr": "Dönel", "en": "rotational"},
    "seamless": {"tr": "Dikişsiz", "en": "seamless"},
    "allover": {"tr": "Allover / Metraj", "en": "allover"},
    "border": {"tr": "Bordür", "en": "border"},
    "placement": {"tr": "Yerleşim", "en": "placement"},
    "engineered": {"tr": "Engineered", "en": "engineered"},
    "panel": {"tr": "Panel", "en": "panel"},
    "scarf_square": {"tr": "Şal kare", "en": "scarf square"},
    "foulard": {"tr": "Foulard", "en": "foulard"},
    "one_way": {"tr": "Tek yön", "en": "one-way"},
    "two_way": {"tr": "İki yön", "en": "two-way"},
    "four_way": {"tr": "Dört yön", "en": "four-way"},
    "tossed": {"tr": "Saçılı", "en": "tossed"},
    "nested": {"tr": "İç içe", "en": "nested"},
    "stripe_repeat": {"tr": "Çizgi tekrar", "en": "stripe repeat"},
    "check_repeat": {"tr": "Kare tekrar", "en": "check repeat"},
}

_COLOR_FAMILIES: dict[str, dict[str, Any]] = {
    "black_white": {"tr": "Siyah-Beyaz", "aliases": ["bw", "monochrome", "siyah beyaz"]},
    "grayscale": {"tr": "Gri Ton", "aliases": ["grey", "gri"]},
    "navy": {"tr": "Lacivert", "aliases": ["navy blue", "lacivert"]},
    "indigo": {"tr": "İndigo", "aliases": ["indigo denim"]},
    "royal_blue": {"tr": "Kraliyet Mavisi", "aliases": ["royal"]},
    "sky_blue": {"tr": "Gök Mavisi", "aliases": ["light blue"]},
    "teal": {"tr": "Teal", "aliases": ["petrol", "petrol mavisi"]},
    "turquoise": {"tr": "Turkuaz", "aliases": ["turkuaz"]},
    "mint": {"tr": "Nane", "aliases": ["mint green"]},
    "emerald": {"tr": "Zümrüt", "aliases": ["emerald green"]},
    "olive": {"tr": "Zeytin", "aliases": ["olive green", "asker yeşili"]},
    "khaki": {"tr": "Haki", "aliases": ["khaki"]},
    "forest": {"tr": "Orman Yeşili", "aliases": ["forest green"]},
    "lime": {"tr": "Lime", "aliases": ["lime green"]},
    "yellow": {"tr": "Sarı", "aliases": ["sarı", "mustard", "hardal"]},
    "gold": {"tr": "Altın", "aliases": ["gold", "altın"]},
    "orange": {"tr": "Turuncu", "aliases": ["turuncu", "coral"]},
    "coral": {"tr": "Mercan", "aliases": ["coral pink"]},
    "red": {"tr": "Kırmızı", "aliases": ["kırmızı", "scarlet"]},
    "burgundy": {"tr": "Bordo", "aliases": ["bordo", "wine", "maroon"]},
    "pink": {"tr": "Pembe", "aliases": ["pembe", "blush", "fuchsia"]},
    "magenta": {"tr": "Macenta", "aliases": ["magenta"]},
    "purple": {"tr": "Mor", "aliases": ["mor", "violet", "lilac"]},
    "lavender": {"tr": "Lavanta", "aliases": ["lavender"]},
    "brown": {"tr": "Kahverengi", "aliases": ["kahve", "tan", "camel", "cognac"]},
    "beige": {"tr": "Bej", "aliases": ["bej", "sand", "nude"]},
    "cream": {"tr": "Krem", "aliases": ["krem", "ivory", "off white"]},
    "white": {"tr": "Beyaz", "aliases": ["beyaz", "optic white"]},
    "black": {"tr": "Siyah", "aliases": ["siyah"]},
    "metallic": {"tr": "Metalik", "aliases": ["silver", "gold foil", "foil"]},
    "pastel": {"tr": "Pastel", "aliases": ["pastel mix"]},
    "neon": {"tr": "Neon", "aliases": ["fluorescent", "neon mix"]},
    "earth": {"tr": "Toprak", "aliases": ["earth tone", "terracotta"]},
    "jewel": {"tr": "Mücevher Ton", "aliases": ["jewel tone"]},
    "rainbow": {"tr": "Gökkuşağı", "aliases": ["multicolor", "rainbow"]},
}

_FABRIC_TYPES: dict[str, dict[str, str]] = {
    "cotton": {"tr": "Pamuk", "en": "cotton"},
    "poplin": {"tr": "Poplin", "en": "poplin"},
    "voile": {"tr": "Vual", "en": "voile"},
    "chiffon": {"tr": "Şifon", "en": "chiffon"},
    "organza": {"tr": "Organze", "en": "organza"},
    "satin": {"tr": "Saten", "en": "satin"},
    "silk": {"tr": "İpek", "en": "silk"},
    "viscose": {"tr": "Viskon", "en": "viscose"},
    "rayon": {"tr": "Rayon", "en": "rayon"},
    "polyester": {"tr": "Polyester", "en": "polyester"},
    "nylon": {"tr": "Naylon", "en": "nylon"},
    "linen": {"tr": "Keten", "en": "linen"},
    "wool": {"tr": "Yün", "en": "wool"},
    "cashmere": {"tr": "Kaşmir", "en": "cashmere"},
    "jersey": {"tr": "Jersey", "en": "jersey"},
    "interlock": {"tr": "Interlok", "en": "interlock"},
    "rib": {"tr": "Ribana", "en": "rib"},
    "fleece": {"tr": "Polar", "en": "fleece"},
    "terry": {"tr": "Havlu", "en": "terry"},
    "denim": {"tr": "Denim", "en": "denim"},
    "twill": {"tr": "Dimi", "en": "twill"},
    "canvas": {"tr": "Kanvas", "en": "canvas"},
    "gabardine": {"tr": "Gabardin", "en": "gabardine"},
    "crepe": {"tr": "Krep", "en": "crepe"},
    "georgette": {"tr": "Jorjet", "en": "georgette"},
    "tulle": {"tr": "Tül", "en": "tulle"},
    "mesh": {"tr": "File", "en": "mesh"},
    "lace_fabric": {"tr": "Dantel Kumaş", "en": "lace"},
    "jacquard": {"tr": "Jakar", "en": "jacquard"},
    "brocade": {"tr": "Brokar", "en": "brocade"},
    "damask": {"tr": "Damask", "en": "damask"},
    "velvet": {"tr": "Kadife", "en": "velvet"},
    "corduroy": {"tr": "Kadife Fitilli", "en": "corduroy"},
    "suede": {"tr": "Süet", "en": "suede"},
    "leather": {"tr": "Deri", "en": "leather"},
    "faux_leather": {"tr": "Suni Deri", "en": "faux leather"},
    "tweed": {"tr": "Tüvit", "en": "tweed"},
    "boucle": {"tr": "Bukle", "en": "bouclé"},
    "neoprene": {"tr": "Neopren", "en": "neoprene"},
    "spandex": {"tr": "Likra", "en": "spandex/elastane"},
    "modal": {"tr": "Modal", "en": "modal"},
    "tencel": {"tr": "Tencel", "en": "tencel/lyocell"},
    "bamboo_fabric": {"tr": "Bambu Kumaş", "en": "bamboo"},
    "organza_silk": {"tr": "İpek Organze", "en": "silk organza"},
    "habotai": {"tr": "Habotai", "en": "habotai"},
    "dupioni": {"tr": "Dupioni", "en": "dupioni"},
    "chiffon_silk": {"tr": "İpek Şifon", "en": "silk chiffon"},
    "scuba": {"tr": "Scuba", "en": "scuba"},
    "ponte": {"tr": "Ponte", "en": "ponte"},
    "french_terry": {"tr": "French Terry", "en": "french terry"},
}

_STYLE_CLASSES: dict[str, dict[str, str]] = {
    "romantic": {"tr": "Romantik", "en": "romantic"},
    "classic": {"tr": "Klasik", "en": "classic"},
    "modern": {"tr": "Modern", "en": "modern"},
    "minimal": {"tr": "Minimal", "en": "minimal"},
    "maximalist": {"tr": "Maksimalist", "en": "maximalist"},
    "boho": {"tr": "Bohem", "en": "boho"},
    "ethnic": {"tr": "Etnik", "en": "ethnic"},
    "folk": {"tr": "Folk", "en": "folk"},
    "sporty": {"tr": "Sportif", "en": "sporty"},
    "street": {"tr": "Sokak", "en": "street"},
    "luxury": {"tr": "Lüks", "en": "luxury"},
    "heritage": {"tr": "Mirası", "en": "heritage"},
    "vintage": {"tr": "Vintage", "en": "vintage"},
    "retro": {"tr": "Retro", "en": "retro"},
    "couture": {"tr": "Couture", "en": "couture"},
    "resort": {"tr": "Resort", "en": "resort"},
    "beach": {"tr": "Plaj", "en": "beach"},
    "evening": {"tr": "Gece", "en": "evening"},
    "bridal": {"tr": "Gelinlik", "en": "bridal"},
    "lingerie": {"tr": "İç Giyim", "en": "lingerie"},
    "kids": {"tr": "Çocuk", "en": "kids"},
    "menswear": {"tr": "Erkek", "en": "menswear"},
    "womenswear": {"tr": "Kadın", "en": "womenswear"},
    "unisex": {"tr": "Unisex", "en": "unisex"},
    "workwear": {"tr": "İş Giyim", "en": "workwear"},
    "military": {"tr": "Askeri", "en": "military"},
    "safari": {"tr": "Safari", "en": "safari"},
    "nautical": {"tr": "Denizci", "en": "nautical"},
    "preppy": {"tr": "Preppy", "en": "preppy"},
    "punk": {"tr": "Punk", "en": "punk"},
    "gothic": {"tr": "Gotik", "en": "gothic"},
    "art_deco": {"tr": "Art Deco", "en": "art deco"},
    "art_nouveau": {"tr": "Art Nouveau", "en": "art nouveau"},
    "brutalist": {"tr": "Brutalist", "en": "brutalist"},
    "organic": {"tr": "Organik", "en": "organic"},
    "techno": {"tr": "Tekno", "en": "techno"},
    "digital": {"tr": "Dijital", "en": "digital"},
    "handcraft": {"tr": "El İşi", "en": "handcraft"},
    "sustainable": {"tr": "Sürdürülebilir", "en": "sustainable"},
}

_TERM_CATEGORIES: dict[str, list[tuple[str, str]]] = {
    "print": [
        ("digital_print", "Dijital baskı"), ("screen_print", "Şablon baskı"),
        ("rotary_print", "Rotasyon baskı"), ("block_print", "Blok baskı"),
        ("discharge", "Dekape"), ("reactive_print", "Reaktif baskı"),
        ("pigment_print", "Pigment baskı"), ("sublimation", "Süblimasyon"),
        ("heat_transfer", "Isı transfer"), ("foil_print", "Foil baskı"),
        ("flock", "Flock"), ("puff_print", "Kabartma baskı"),
        ("glitter_print", "Sim baskı"), ("burnout", "Yakma baskı"),
        ("devore", "Devore"), ("placement_print", "Yerleşim baskı"),
        ("engineered_print", "Engineered baskı"), ("allover_print", "Allover baskı"),
        ("border_print", "Bordür baskı"), ("panel_print", "Panel baskı"),
    ],
    "structure": [
        ("warp", "Çözgü"), ("weft", "Atkı"), ("selvedge", "Kenar"),
        ("grain", "İplik yönü"), ("bias", "Biye / çapraz"), ("nap", "Tüy yönü"),
        ("handfeel", "El hissi"), ("drape", "Döküm"), ("opacity", "Opaklık"),
        ("transparency", "Şeffaflık"), ("gsm", "Gramaj"), ("yarn_count", "İplik numarası"),
        ("twist", "Büküm"), ("ply", "Kat"), ("denier", "Denye"),
        ("tex", "Tex"), ("gauge", "Fayn"), ("course", "İlmek sırası"),
        ("wale", "İlmek çubuğu"), ("interlock_knit", "Interlok örme"),
    ],
    "finish": [
        ("soft_finish", "Yumuşak apre"), ("peach_finish", "Şeftali apre"),
        ("brushing", "Fırçalama"), ("sanding", "Zımpara"), ("calendering", "Kalandır"),
        ("mercerizing", "Merserize"), ("sanforize", "Sanfor"), ("enzyme_wash", "Enzim yıkama"),
        ("stone_wash", "Taş yıkama"), ("acid_wash", "Asit yıkama"),
        ("garment_dye", "Konfeksiyon boya"), ("overdye", "Üzerine boya"),
        ("coating", "Kaplama"), ("lamination", "Laminasyon"), ("waterproof", "Su geçirmez"),
        ("water_repellent", "Su itici"), ("anti_pill", "Boncuklanma önleyici"),
        ("wrinkle_free", "Buruşmaz"), ("easy_care", "Kolay bakım"),
        ("flame_retardant", "Alev geciktirici"), ("antibacterial", "Antibakteriyel"),
        ("uv_protect", "UV koruma"), ("moisture_wicking", "Nemi uzaklaştırma"),
    ],
    "design": [
        ("motif", "Motif"), ("rapport", "Raport"), ("repeat_unit", "Tekrar ünitesi"),
        ("colorway", "Renk yolu"), ("recolor", "Yeniden renklendirme"),
        ("strike_off", "Strike-off"), ("lab_dip", "Lab dip"), ("tech_pack", "Teknik paket"),
        ("cad", "CAD"), ("vector_art", "Vektör"), ("raster_art", "Raster"),
        ("seamless_tile", "Dikişsiz karo"), ("dpi", "DPI"), ("ppi", "PPI"),
        ("bleed", "Taşma payı"), ("trim", "Kesim"), ("marker", "Pastal"),
        ("grading", "Kalıp grading"), ("fit", "Kalıp uyumu"), ("silhouette", "Siluet"),
        ("collection", "Koleksiyon"), ("capsule", "Kapsül"), ("lookbook", "Lookbook"),
        ("moodboard", "Moodboard"), ("storyboard", "Hikâye panosu"),
        ("trend", "Trend"), ("season", "Sezon"), ("ss", "İlkbahar-Yaz"),
        ("aw", "Sonbahar-Kış"), ("cruise", "Cruise"), ("pre_fall", "Pre-fall"),
    ],
    "market": [
        ("fast_fashion", "Hızlı moda"), ("premium", "Premium"), ("luxury_market", "Lüks pazar"),
        ("private_label", "Private label"), ("oem", "OEM"), ("odm", "ODM"),
        ("wholesale", "Toptan"), ("retail", "Perakende"), ("ecom", "E-ticaret"),
        ("marketplace", "Pazar yeri"), ("showroom", "Showroom"), ("fair", "Fuar"),
        ("buyer", "Alıcı"), ("merchandiser", "Merchandiser"), ("sourcing", "Tedarik"),
        ("moq", "MOQ"), ("lead_time", "Termin"), ("sampling", "Numune"),
        ("bulk", "Seri"), ("qc", "Kalite kontrol"), ("qa", "Kalite güvence"),
    ],
    "pattern_talk": [
        ("scale", "Ölçek"), ("density", "Yoğunluk"), ("contrast", "Kontrast"),
        ("ground", "Zemin"), ("figure", "Figür"), ("positive_space", "Pozitif alan"),
        ("negative_space", "Negatif alan"), ("balance", "Denge"), ("rhythm", "Ritim"),
        ("hierarchy", "Hiyerarşi"), ("focal_point", "Odak noktası"),
        ("directional", "Yönlü"), ("non_directional", "Yönsüz"),
        ("conversational", "Konuşmalı desen"), ("novelty", "Novelty"),
        ("toile", "Toile"), ("chintz_style", "Chintz stil"), ("foulard_style", "Foulard stil"),
        ("liberty_style", "Liberty stil"), ("ditsy_style", "Ditsy stil"),
        ("jumbo", "Jumbo motif"), ("micro_print", "Mikro baskı"),
        ("tone_on_tone", "Ton sür ton"), ("high_contrast", "Yüksek kontrast"),
        ("low_contrast", "Düşük kontrast"), ("multicolor", "Çok renkli"),
        ("two_tone", "İki ton"), ("three_tone", "Üç ton"),
    ],
}

# İlişki grupları: aynı gruptaki motifler birbirine yakın
_RELATION_GROUPS: list[tuple[str, list[str]]] = [
    ("big_cat_spots", ["leopard_spot", "cheetah_spot", "jaguar", "cow_hide", "dalmatian_spot"]),
    ("big_cat_stripes", ["tiger_stripe", "zebra_stripe"]),
    ("reptile", ["snake_scale", "python_skin", "crocodile_skin", "lizard", "gecko", "iguana"]),
    ("garden_roses", ["rose", "tea_rose", "english_rose", "wild_rose", "peony", "camellia"]),
    ("spring_blooms", ["daisy", "tulip", "daffodil", "hyacinth", "lilac", "violet"]),
    ("tropical_flora", ["hibiscus", "plumeria", "bird_of_paradise", "monstera", "palm", "banana_leaf"]),
    ("botanical_leaf", ["fern_frond", "eucalyptus", "ivy", "olive_branch", "maple_leaf", "oak_leaf"]),
    ("sea_life", ["seahorse", "starfish", "shell", "coral", "jellyfish", "whale", "dolphin", "octopus"]),
    ("birds", ["peacock_feather", "eagle", "owl", "flamingo", "parrot", "hummingbird", "swan"]),
    ("insects", ["butterfly", "dragonfly", "bee", "ladybug", "beetle"]),
    ("geo_classic", ["houndstooth", "herringbone", "argyle", "chevron", "zigzag"]),
    ("geo_islamic", ["arabesque", "mandala", "tessera", "star", "hexagon"]),
    ("check_family", ["tartan", "gingham", "windowpane", "buffalo", "madras"]),
    ("luxury_logo", ["lv_monogram", "gg_supreme", "cd_oblique", "ff_logo", "ysl_logo", "chanel_cc"]),
    ("heritage_check", ["burberry_nova", "tb_check", "tartan"]),
    ("baroque_ornament", ["medusa_head", "versace_greca", "dg_crown", "crown", "crest"]),
    ("nautical_set", ["anchor", "compass", "ship", "shell", "starfish", "rope_chain"]),
    ("jewelry_set", ["ring", "earring", "necklace", "bracelet", "jewelry", "gold_chain"]),
    ("fruit_set", ["strawberry", "cherry", "watermelon", "lemon", "orange", "apple", "grape", "pomegranate"]),
    ("sweet_set", ["candy", "lollipop", "donut", "cookie", "cake", "ice_cream"]),
    ("celestial", ["sun", "moon", "star", "planet", "constellation", "cloud", "lightning"]),
    ("winter", ["snowflake", "pinecone", "poinsettia"]),
    ("cultural_east", ["sakura", "japanese_wave", "kanji", "chinese_cloud", "dragon_cult"]),
    ("cultural_anatolia", ["ottoman", "kilim", "suzani", "mehndi"]),
    ("tie_resist", ["ikat", "batik", "shibori", "tie_dye"]),
]


@dataclass
class FamilyDef:
    id: str
    label: str
    parent: str = ""
    root: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class MotifDef:
    id: str
    label: str
    category: str
    synonyms: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)


@dataclass
class TextileKnowledgeBase:
    families: dict[str, FamilyDef]
    motifs: dict[str, MotifDef]
    motif_names: set[str]
    terms: dict[str, str]
    brand_families: dict[str, dict[str, Any]]
    repeat_types: dict[str, dict[str, str]]
    color_families: dict[str, dict[str, Any]]
    fabric_types: dict[str, dict[str, str]]
    style_classes: dict[str, dict[str, str]]
    motif_relations: dict[str, set[str]]
    term_to_family: dict[str, str]
    term_to_motif: dict[str, str]

    def stats(self) -> dict[str, int]:
        return {
            "pattern_families": len(self.families),
            "motif_defs": len(self.motifs),
            "motif_names": len(self.motif_names),
            "textile_terms": len(self.terms),
            "brand_families": len(self.brand_families),
            "repeat_types": len(self.repeat_types),
            "color_families": len(self.color_families),
            "fabric_types": len(self.fabric_types),
            "style_classes": len(self.style_classes),
            "motif_relation_edges": sum(len(v) for v in self.motif_relations.values()) // 2,
        }


def _slug(*parts: str) -> str:
    return "_".join(p for p in parts if p).lower().replace("-", "_").replace(" ", "_")


def _label_join(*parts: str) -> str:
    return " / ".join(p for p in parts if p)


def _build_families() -> dict[str, FamilyDef]:
    out: dict[str, FamilyDef] = {}
    for root in ROOT_FAMILIES:
        out[root] = FamilyDef(
            id=root,
            label=_ROOT_TR.get(root, root),
            parent="",
            root=root,
            tags=["root"],
        )

    # Alt tipler + tek ölçek varyantı → ~300–500 aile
    for root, subtypes in _FAMILY_SUBTYPES.items():
        for sub in subtypes:
            fid = _slug(root, sub)
            out[fid] = FamilyDef(
                id=fid,
                label=_label_join(_ROOT_TR.get(root, root), sub.replace("_", " ").title()),
                parent=root,
                root=root,
                tags=["subtype", sub],
            )
            sid = _slug(fid, "macro")
            out[sid] = FamilyDef(
                id=sid,
                label=_label_join(out[fid].label, "macro"),
                parent=fid,
                root=root,
                tags=["scale", "macro", sub],
            )

    for root in ("floral", "animal_print", "geometric", "marble_abstract", "baroque"):
        for style in ("watercolor", "vintage", "modern", "ethnic", "linework"):
            sid = _slug(root, style)
            if sid not in out:
                out[sid] = FamilyDef(
                    id=sid,
                    label=_label_join(_ROOT_TR.get(root, root), style),
                    parent=root,
                    root=root,
                    tags=["style", style],
                )

    for brand, meta in _BRAND_FAMILIES.items():
        bid = _slug("brand", brand)
        out[bid] = FamilyDef(
            id=bid,
            label=str(meta["label"]),
            parent=str(meta.get("pattern_family") or "monogram_logo"),
            root=str(meta.get("pattern_family") or "monogram_logo"),
            tags=["brand", brand],
        )
    return out


def _build_motifs() -> tuple[dict[str, MotifDef], set[str], dict[str, str]]:
    motifs: dict[str, MotifDef] = {}
    names: set[str] = set()
    term_to_motif: dict[str, str] = {}

    cat_to_family = {
        "floral": "floral",
        "animal": "animal_print",
        "geometric": "geometric",
        "object": "conversational",
        "cultural": "ethnic",
        "brand_motif": "monogram_logo",
    }

    for cat, items in _MOTIF_SEEDS.items():
        for mid, label in items:
            syns = list(_MOTIF_SYNONYMS.get(mid, []))
            # TR/EN otomatik isimler
            syns.extend(
                [
                    mid.replace("_", " "),
                    label.lower(),
                    label,
                ]
            )
            # Ölçekli motif adları → 1000+ isim
            for scale in ("mini", "midi", "macro", "jumbo"):
                syns.append(f"{scale} {mid.replace('_', ' ')}")
                syns.append(f"{scale} {label.lower()}")
            for layout in ("allover", "border", "scatter"):
                syns.append(f"{layout} {mid.replace('_', ' ')}")

            fam = cat_to_family.get(cat, "unknown")
            motifs[mid] = MotifDef(
                id=mid,
                label=label,
                category=cat,
                synonyms=sorted(set(s for s in syns if s)),
                families=[fam] if fam != "conversational" else ["unknown", "document"],
            )
            for n in [mid, label.lower(), *motifs[mid].synonyms]:
                key = n.strip().lower()
                if key:
                    names.add(key)
                    term_to_motif.setdefault(key, mid)
    return motifs, names, term_to_motif


def _build_terms() -> dict[str, str]:
    terms: dict[str, str] = {}
    for cat, items in _TERM_CATEGORIES.items():
        for tid, label in items:
            terms[tid] = label
            terms[tid.replace("_", " ")] = label
            terms[label.lower()] = label
    for rid, meta in _REPEAT_TYPES.items():
        terms[rid] = meta["tr"]
        terms[meta["en"].lower()] = meta["tr"]
        terms[meta["tr"].lower()] = meta["tr"]
    for cid, meta in _COLOR_FAMILIES.items():
        terms[cid] = meta["tr"]
        for a in meta.get("aliases", []):
            terms[str(a).lower()] = meta["tr"]
        terms[meta["tr"].lower()] = meta["tr"]
    for fid, meta in _FABRIC_TYPES.items():
        terms[fid] = meta["tr"]
        terms[meta["en"].lower()] = meta["tr"]
        terms[meta["tr"].lower()] = meta["tr"]
    for sid, meta in _STYLE_CLASSES.items():
        terms[sid] = meta["tr"]
        terms[meta["en"].lower()] = meta["tr"]
        terms[meta["tr"].lower()] = meta["tr"]
    # Ek tekstil jargonu (500+ hedef)
    extras = [
        ("rapport_size", "Raport ölçüsü"), ("color_separation", "Renk ayrımı"),
        ("screen_count", "Şablon sayısı"), ("registration", "Çakıştırma"),
        ("ghosting", "Ghosting"), ("moiré_defect", "Muar hatası"),
        ("crocking", "Sürtme haslığı"), ("lightfastness", "Işık haslığı"),
        ("washfastness", "Yıkama haslığı"), ("rubfastness", "Sürtme haslığı"),
        ("pilling", "Boncuklanma"), ("snagging", "İplik çekmesi"),
        ("skew", "Çarpıklık"), ("bowing", "Kavis"), ("shrinkage", "Çekme"),
        ("elongation", "Uzama"), ("recovery", "Toparlanma"),
        ("hand_loom", "El tezgâhı"), ("power_loom", "Makine tezgâhı"),
        ("circular_knit", "Yuvarlak örme"), ("flat_knit", "Düz örgü"),
        ("warp_knit", "Çözgülü örme"), ("weft_knit", "Atkılı örme"),
        ("single_jersey", "Süprem"), ("double_knit", "Çift örgü"),
        ("lace_knit", "Ajurlu örme"), ("intarsia", "İntarsia"),
        ("jacquard_knit", "Jakar örme"), ("embroidered", "Nakışlı"),
        ("applique", "Aplikasyon"), ("beading", "Boncuk işi"),
        ("sequin_work", "Pul işi"), ("smocking", "Büzgü"),
        ("pleat", "Pile"), ("ruffle", "Fırfır"), ("fringe", "Saçak"),
        ("tassel", "Püskül"), ("piping", "Boru biye"), ("binding", "Biye"),
        ("facing", "Astar yüz"), ("lining", "Astar"), ("interlining", "Telalı"),
        ("fusible", "Ütüyle yapışan"), ("nonwoven", "Dokusuz yüzey"),
        ("spunbond", "Spunbond"), ("meltblown", "Meltblown"),
        ("microfiber", "Mikrofilament"), ("performance_wear", "Performans giysi"),
        ("athleisure", "Athleisure"), ("loungewear", "Ev giyim"),
        ("sleepwear", "Uyku giysisi"), ("swimwear", "Mayo"),
        ("activewear", "Spor giyim"), ("outerwear_shell", "Dış katman"),
        ("insulation", "İzolasyon"), ("down_fill", "Kaz tüyü"),
        ("membrane", "Membran"), ("softshell", "Softshell"),
        ("hardshell", "Hardshell"), ("rainwear", "Yağmurluk"),
        ("windbreaker", "Rüzgarlık"), ("puffer", "Şişme mont"),
        ("bomber", "Bomber"), ("trench", "Trençkot"),
        ("blazer", "Blazer"), ("tailoring", "Terzi işi"),
        ("bespoke", "Sipariş üzerine"), ("made_to_measure", "Ölçüye göre"),
        ("ready_to_wear", "Hazır giyim"), ("haute_couture", "Haute couture"),
        ("demi_couture", "Demi couture"), ("capsule_wardrobe", "Kapsül gardırop"),
        ("core_range", "Ana koleksiyon"), ("fashion_range", "Moda koleksiyon"),
        ("basics", "Basic"), ("essentials", "Essentials"),
        ("statement_print", "Vurucu baskı"), ("hero_print", "Hero desen"),
        ("carryover", "Carryover"), ("bestseller", "Çok satan"),
        ("deadstock", "Deadstock"), ("overstock", "Overstock"),
        ("preorder", "Ön sipariş"), ("drop", "Drop"),
        ("collab", "Kolaborasyon"), ("license", "Lisans"),
        ("ip_print", "IP baskı"), ("character_print", "Karakter baskı"),
    ]
    for tid, label in extras:
        terms[tid] = label
        terms[tid.replace("_", " ")] = label
        terms[label.lower()] = label
    return terms


def _build_relations(motifs: dict[str, MotifDef]) -> dict[str, set[str]]:
    rel: dict[str, set[str]] = {m: set() for m in motifs}
    for _name, group in _RELATION_GROUPS:
        ids = [m for m in group if m in motifs]
        for a in ids:
            for b in ids:
                if a != b:
                    rel[a].add(b)
    # Aynı kategorideki motifler zayıf bağ
    by_cat: dict[str, list[str]] = {}
    for mid, m in motifs.items():
        by_cat.setdefault(m.category, []).append(mid)
    for _cat, ids in by_cat.items():
        for i, a in enumerate(ids):
            for b in ids[i + 1 : i + 6]:
                rel[a].add(b)
                rel[b].add(a)
    return rel


def _build_term_to_family(families: dict[str, FamilyDef]) -> dict[str, str]:
    out: dict[str, str] = {}
    for fid, fam in families.items():
        out[fid] = fam.root or fid
        out[fam.label.lower()] = fam.root or fid
        for t in fam.tags:
            if t not in ("root", "subtype", "scale", "layout", "style", "brand"):
                out.setdefault(t, fam.root or fid)
    for brand, meta in _BRAND_FAMILIES.items():
        out[brand] = str(meta["pattern_family"])
        out[str(meta["label"]).lower()] = str(meta["pattern_family"])
        for a in meta.get("aliases", []):
            out[str(a).lower()] = str(meta["pattern_family"])
    # Kök TR
    for root, label in _ROOT_TR.items():
        out[label.lower()] = root
        out[root] = root
    return out


@lru_cache(maxsize=1)
def get_knowledge_base() -> TextileKnowledgeBase:
    families = _build_families()
    motifs, motif_names, term_to_motif = _build_motifs()
    terms = _build_terms()
    # Motif isimleri de terim sayılır
    for n in motif_names:
        terms.setdefault(n, n)
    relations = _build_relations(motifs)
    return TextileKnowledgeBase(
        families=families,
        motifs=motifs,
        motif_names=motif_names,
        terms=terms,
        brand_families=dict(_BRAND_FAMILIES),
        repeat_types=dict(_REPEAT_TYPES),
        color_families=dict(_COLOR_FAMILIES),
        fabric_types=dict(_FABRIC_TYPES),
        style_classes=dict(_STYLE_CLASSES),
        motif_relations=relations,
        term_to_family=_build_term_to_family(families),
        term_to_motif=term_to_motif,
    )


def root_family(family_id: str) -> str:
    kb = get_knowledge_base()
    fam = kb.families.get(family_id or "")
    if fam:
        return fam.root or fam.id
    if family_id in ROOT_FAMILIES:
        return family_id
    return "unknown"


def family_affinity(family_a: str, family_b: str) -> float:
    """0–1 aile yakınlığı (skorlama için)."""
    a = (family_a or "").strip()
    b = (family_b or "").strip()
    if not a or not b or a in ("unknown", "") or b in ("unknown", ""):
        return 0.0
    if a == b:
        return 1.0
    ra, rb = root_family(a), root_family(b)
    if ra == rb and ra not in ("unknown", "plain"):
        return 0.82
    kb = get_knowledge_base()
    fa, fb = kb.families.get(a), kb.families.get(b)
    if fa and fb and fa.parent and fa.parent == fb.parent:
        return 0.70
    # Bilinen çatışmalar
    conflicts = {
        "animal_print": {"floral", "plaid_check", "geometric", "plain"},
        "floral": {"animal_print", "camouflage"},
        "plaid_check": {"floral", "animal_print"},
        "marble_abstract": {"animal_print"},
    }
    if rb in conflicts.get(ra, set()) or ra in conflicts.get(rb, set()):
        return 0.05
    if ra != rb:
        return 0.15
    return 0.35


def motif_affinity(motif_a: str, motif_b: str) -> float:
    a = (motif_a or "").strip().lower()
    b = (motif_b or "").strip().lower()
    if not a or not b:
        return 0.0
    kb = get_knowledge_base()
    # id veya isim çözümle
    ma = kb.term_to_motif.get(a, a if a in kb.motifs else "")
    mb = kb.term_to_motif.get(b, b if b in kb.motifs else "")
    if not ma or not mb:
        return 0.0
    if ma == mb:
        return 1.0
    if mb in kb.motif_relations.get(ma, set()):
        return 0.78
    ca = kb.motifs[ma].category if ma in kb.motifs else ""
    cb = kb.motifs[mb].category if mb in kb.motifs else ""
    if ca and ca == cb:
        return 0.45
    return 0.1


def color_affinity(color_a: str, color_b: str) -> float:
    a = (color_a or "").strip().lower()
    b = (color_b or "").strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    kb = get_knowledge_base()
    # alias → id
    def _cid(x: str) -> str:
        if x in kb.color_families:
            return x
        for cid, meta in kb.color_families.items():
            if x == meta["tr"].lower() or x in [str(z).lower() for z in meta.get("aliases", [])]:
                return cid
        return x

    ca, cb = _cid(a), _cid(b)
    if ca == cb:
        return 1.0
    warm = {"red", "orange", "coral", "yellow", "gold", "burgundy", "pink", "magenta"}
    cool = {"navy", "indigo", "royal_blue", "sky_blue", "teal", "turquoise", "mint", "emerald", "purple", "lavender"}
    earth = {"brown", "beige", "cream", "olive", "khaki", "forest", "earth"}
    mono = {"black_white", "grayscale", "black", "white", "cream"}
    for group, score in ((warm, 0.72), (cool, 0.72), (earth, 0.75), (mono, 0.8)):
        if ca in group and cb in group:
            return score
    return 0.18


def knowledge_score_boost(
    *,
    query_family: str = "",
    cand_family: str = "",
    query_motif: str = "",
    cand_motif: str = "",
    query_color: str = "",
    cand_color: str = "",
    query_text: str = "",
    cand_filename: str = "",
) -> dict[str, float]:
    """Skor bileşenleri — search_engine pattern_family_score'a eklenir."""
    fam = family_affinity(query_family, cand_family)
    mot = motif_affinity(query_motif, cand_motif)
    col = color_affinity(query_color, cand_color)

    # Metin/dosya adı motif ipucu
    text_boost = 0.0
    kb = get_knowledge_base()
    q = (query_text or "").strip().lower()
    fn = (cand_filename or "").lower()
    if q:
        mid = kb.term_to_motif.get(q)
        if mid:
            if mid.replace("_", " ") in fn or mid in fn:
                text_boost = 0.55
            else:
                related = kb.motif_relations.get(mid, set())
                if any(r.replace("_", " ") in fn for r in related):
                    text_boost = 0.35
        root = kb.term_to_family.get(q)
        if root and root == root_family(cand_family):
            text_boost = max(text_boost, 0.40)

    combined = min(
        1.0,
        0.45 * fam + 0.30 * mot + 0.15 * col + 0.10 * text_boost,
    )
    return {
        "kb_family": round(fam, 4),
        "kb_motif": round(mot, 4),
        "kb_color": round(col, 4),
        "kb_text": round(text_boost, 4),
        "kb_combined": round(combined, 4),
    }


def expand_knowledge_terms(query: str) -> list[str]:
    """textile_terms.expand_query_terms için ek genişletme."""
    q = (query or "").strip().lower()
    if not q:
        return []
    kb = get_knowledge_base()
    out: set[str] = {q}
    mid = kb.term_to_motif.get(q)
    if mid and mid in kb.motifs:
        out.update(kb.motifs[mid].synonyms)
        out.add(mid)
        for rel in kb.motif_relations.get(mid, set()):
            if rel in kb.motifs:
                out.add(rel)
                out.add(kb.motifs[rel].label.lower())
    fam = kb.term_to_family.get(q)
    if fam:
        out.add(fam)
        out.add(_ROOT_TR.get(fam, fam).lower())
    for brand, meta in kb.brand_families.items():
        aliases = [brand, str(meta["label"]).lower(), *[str(a).lower() for a in meta.get("aliases", [])]]
        if q in aliases or any(a in q for a in aliases if len(a) > 2):
            out.update(aliases)
            out.add(str(meta.get("pattern_family") or ""))
    # Renk / kumaş / stil
    for cid, meta in kb.color_families.items():
        if q == cid or q == meta["tr"].lower() or q in [str(a).lower() for a in meta.get("aliases", [])]:
            out.add(cid)
            out.add(meta["tr"].lower())
            out.update(str(a).lower() for a in meta.get("aliases", []))
    return sorted({t for t in out if t}, key=len, reverse=True)


def knowledge_family_hints(query: str) -> dict[str, str]:
    q = (query or "").strip().lower()
    if not q:
        return {}
    kb = get_knowledge_base()
    hints: dict[str, str] = {}
    mid = kb.term_to_motif.get(q)
    if mid and mid in kb.motifs:
        m = kb.motifs[mid]
        if m.families:
            hints["pattern_family"] = m.families[0]
        hints["pattern_type"] = mid
        if m.category == "animal":
            # animal_print_type kabaca
            for key in ("leopard", "tiger", "zebra", "snake", "crocodile", "giraffe", "cow"):
                if key in mid:
                    hints["animal_print_type"] = key
                    break
    fam = kb.term_to_family.get(q)
    if fam and "pattern_family" not in hints:
        hints["pattern_family"] = fam
    for brand, meta in kb.brand_families.items():
        aliases = [brand, str(meta["label"]).lower(), *[str(a).lower() for a in meta.get("aliases", [])]]
        if q in aliases:
            hints["pattern_family"] = str(meta.get("pattern_family") or "monogram_logo")
            hints["brand"] = brand
            break
    return hints
