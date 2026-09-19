"""Universal Visual Intelligence — schema, hierarchy, fine-grained rank (query-time).

Does not load extra models. Reuses CLIP/DINO already in the search engine.
Does not write production index / Pattern DNA.
Forbidden: race / ethnicity classification.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from core.pattern_intelligence_v2 import (
    COMPOSITE_CLIP_MIN,
    PatternCardV2,
    PatternQueryV2,
    parse_pattern_query_v2,
)
from core.textile_terms import normalize_turkish

TIER_EXACT = "EXACT"
TIER_VERY_SIMILAR = "VERY_SIMILAR"
TIER_SAME_SUBTYPE = "SAME_SUBTYPE"
TIER_SAME_FAMILY = "SAME_FAMILY"
TIER_SEMANTIC = "SEMANTIC_SIMILAR"

# These ontology nodes are ordinary text/category concepts in a textile
# catalogue. They must remain searchable from filename/metadata even when CLIP
# is enabled; requiring an object-CLIP hit here would incorrectly zero valid
# clothing searches such as "kot".
TEXT_METADATA_NODES = frozenset({
    "shirt", "tshirt", "dress", "jacket", "coat", "pants", "jeans",
    "skirt", "shoe", "sneaker", "hat", "bag", "backpack",
})

EVIDENCE_EXACT = "object_exact"
EVIDENCE_SUBTYPE = "object_subtype"
EVIDENCE_FAMILY = "object_family"
EVIDENCE_SEMANTIC = "object_semantic"
EVIDENCE_UNKNOWN = "object_unknown"
EVIDENCE_MISMATCH = "object_mismatch"

# Same bar as Pattern Intelligence v2 illustration objects — not a new model.
OBJECT_OWN_MIN = 0.26
OBJECT_MARGIN = 0.03

# Concrete object queries. Textile motif families (leopard/floral/rose) stay on v2.
HARD_OBJECT_NODES = frozenset(
    {
        "vehicle",
        "car",
        "togg",
        "t10x",
        "bmw",
        "mercedes",
        "audi",
        "truck",
        "bus",
        "motorcycle",
        "bird",
        "crow",
        "stork",
        "eagle",
        "sparrow",
        "pigeon",
        "fish",
        "mouse",
        "insect",
        "ant",
        "bee",
        "butterfly",
        "ladybug",
        "person",
        "child",
        "girl",
        "building",
        # Expanded universal object coverage (zero-shot / experimental unless dedicated model exists).
        "cat", "kitten", "persian_cat", "siamese_cat", "british_shorthair", "maine_coon", "scottish_fold",
        "dog", "puppy", "labrador", "german_shepherd", "golden_retriever", "bulldog", "poodle",
        "horse", "cow", "sheep", "goat", "pig", "deer", "fox", "wolf", "bear", "lion", "elephant",
        "giraffe", "monkey", "gorilla", "panda", "koala", "kangaroo", "rabbit", "hamster",
        "parrot", "canary", "owl", "falcon", "hawk", "swan", "duck", "goose", "flamingo", "peacock",
        "penguin", "seagull", "woodpecker", "hummingbird",
        "crocodile", "turtle", "lizard", "frog",
        "spider", "dragonfly", "beetle", "grasshopper", "fly", "mosquito",
        "shark", "whale", "dolphin", "seal", "octopus", "jellyfish",
        "tree", "leaf", "grass", "forest", "mountain", "beach", "river", "lake", "sky", "cloud", "sun", "moon",
        "bicycle", "road_bike", "mountain_bike", "city_bike", "electric_bike", "bmx", "scooter", "skateboard",
        "van", "suv", "pickup", "sports_car", "sedan", "hatchback", "coupe", "convertible", "minivan",
        "train", "tram", "subway", "airplane", "helicopter", "boat", "ship", "tractor", "ambulance", "fire_truck", "police_car",
        "toyota", "ford", "volkswagen", "volvo", "honda", "hyundai", "kia", "tesla", "renault", "peugeot", "citroen",
        "fiat", "opel", "porsche", "lexus", "jaguar", "land_rover", "skoda", "seat", "mazda", "nissan", "subaru", "suzuki",
        "volkswagen_commercial", "chevrolet", "jeep", "dodge", "chrysler", "genesis", "infiniti", "acura",
        "table", "dining_table", "coffee_table", "desk", "console_table", "side_table", "chair", "office_chair", "armchair",
        "sofa", "bed", "wardrobe", "cabinet", "bookshelf", "dresser", "nightstand", "bench", "stool", "shelf",
        "lamp", "mirror", "rug", "curtain", "door", "window",
        "phone", "smartphone", "tablet", "laptop", "desktop_computer", "monitor", "keyboard", "mouse_device", "headphones",
        "camera", "television", "speaker", "watch", "clock", "printer",
        "shirt", "tshirt", "dress", "jacket", "coat", "pants", "jeans", "skirt", "shoe", "sneaker", "hat",
        "bag", "backpack", "brooch", "necklace",
        "cup", "bottle", "glass", "plate", "bowl", "fork", "knife", "spoon", "pan", "pot",
        "apple", "banana", "orange", "lemon", "strawberry", "watermelon", "tomato", "potato", "bread", "cake", "pizza", "burger",
        "ball", "football", "basketball", "tennis_racket", "golf_club", "skis", "surfboard",
        "book", "newspaper", "pen", "pencil", "scissors", "hammer", "screwdriver", "drill", "toolbox",
    }
)

_DISTRACTOR_FOR: dict[str, tuple[str, ...]] = {
    "vehicle": ("flower", "denim"),
    "car": ("flower", "denim"),
    "togg": ("flower", "denim"),
    "bmw": ("flower", "denim"),
    "mercedes": ("flower", "denim"),
    "audi": ("flower", "denim"),
    "truck": ("flower", "denim"),
    "bus": ("flower", "denim"),
    "motorcycle": ("flower", "denim"),
    "bird": ("flower", "denim"),
    "crow": ("flower", "denim"),
    "stork": ("flower", "denim"),
    "eagle": ("flower", "denim"),
    "sparrow": ("flower", "denim"),
    "pigeon": ("flower", "denim"),
    "fish": ("flower", "denim"),
    "mouse": ("flower", "denim"),
    "insect": ("flower", "denim"),
    "butterfly": ("flower", "denim"),
    "strawberry": ("flower", "denim"),
    "cherry": ("flower", "denim"),
    "person": ("flower", "denim"),
    "building": ("flower", "denim"),
}

_DISTRACTOR_PROMPTS: dict[str, str] = {
    "denim": "blue denim jeans fabric texture, not an animal bird fish or car print",
}

_CONFLICT_PATH_TOKENS: dict[str, frozenset[str]] = {
    "bird": frozenset({"denim", "kot", "jeans"}),
    "crow": frozenset({"denim", "kot", "jeans"}),
    "stork": frozenset({"denim", "kot", "jeans"}),
    "fish": frozenset({"floral", "cicek", "dantel"}),
    "car": frozenset({"dantel", "floral"}),
    "vehicle": frozenset({"dantel", "floral"}),
}

SUPPORTED = "supported"
EXPERIMENTAL = "experimental"
PLANNED = "planned"
UNSUPPORTED = "unsupported"

# Honest capability map — do not claim more than the code does.
CAPABILITIES: dict[str, str] = {
    "open_ontology": SUPPORTED,
    "hierarchical_query_parse": SUPPORTED,
    "fine_grained_clip_rival_gate": SUPPORTED,
    "composite_and_search": SUPPORTED,
    "representation_textile_vs_photo": EXPERIMENTAL,
    "visual_dna_query_card": SUPPORTED,
    "identity_clip_clustering": EXPERIMENTAL,
    "face_detection": SUPPORTED,
    "face_counting": SUPPORTED,
    "gender_classification_from_face": EXPERIMENTAL,
    "face_recognition": EXPERIMENTAL,
    "person_reidentification_gallery": UNSUPPORTED,
    "spatial_object_boxes": UNSUPPORTED,
    "race_ethnicity_classification": UNSUPPORTED,
    "dedicated_togg_classifier": UNSUPPORTED,
    "dedicated_bird_species_classifier": UNSUPPORTED,
    "clip_zero_shot_species": EXPERIMENTAL,
    "clip_zero_shot_car_brand": EXPERIMENTAL,
    "object_aware_ranking": SUPPORTED,
    "negative_object_evidence": SUPPORTED,
    "expanded_object_ontology": SUPPORTED,
    "cat_life_stage_and_breed_zero_shot": EXPERIMENTAL,
    "bird_species_zero_shot": EXPERIMENTAL,
    "bicycle_type_zero_shot": EXPERIMENTAL,
    "furniture_type_zero_shot": EXPERIMENTAL,
    "vehicle_type_zero_shot": EXPERIMENTAL,
    "vehicle_brand_zero_shot": EXPERIMENTAL,
    # Global Object AI uses the optional torchvision COCO detector. It is
    # fail-safe and does not replace the existing pattern-search pipeline.
    "global_common_object_detection": SUPPORTED,
    "multi_object_detection": SUPPORTED,
    "object_counting": SUPPORTED,
    "human_instance_separation": SUPPORTED,
    "object_segmentation": UNSUPPORTED,
    "object_spatial_relations": UNSUPPORTED,
    "vehicle_model_classifier": UNSUPPORTED,
    "animal_breed_classifier": EXPERIMENTAL,
}

FORBIDDEN_LABELS = frozenset(
    {
        "race",
        "ethnicity",
        "ırk",
        "irk",
        "caucasian",
        "negroid",
        "mongoloid",
    }
)

# Open hierarchy: (id, parent, level, aliases, clip_prompt, status)
# level: category|family|subtype|species|brand|model
_NODES: list[tuple[str, str, str, tuple[str, ...], str, str]] = [
    ("entity", "", "category", ("varlik",), "", SUPPORTED),
    ("person", "entity", "family", ("insan", "kisi", "kişi", "person", "human", "face"), "photograph of a person, not a textile print", EXPERIMENTAL),
    # Person attributes are query-time semantic concepts. They are intentionally
    # EXPERIMENTAL and never override an explicit face-index identity match.
    ("female_person", "person", "subtype", ("kadın", "kadin", "woman", "women", "female", "bayan"),
     "a real photograph or illustration of a woman, female human person, woman's face or body, portrait or fashion photo, not a textile fabric pattern", EXPERIMENTAL),
    ("male_person", "person", "subtype", ("erkek", "man", "men", "male"),
     "a real photograph or illustration of a man, male human person, man's face or body, portrait or fashion photo, not a textile fabric pattern", EXPERIMENTAL),
    ("child", "person", "subtype", ("çocuk", "cocuk", "child", "kid", "çocuklar"),
     "a real photograph of a child, young person, not a textile print", EXPERIMENTAL),
    ("girl", "child", "subtype", ("kız", "kiz", "girl", "kız çocuğu"),
     "a real photograph of a girl, female child, not a textile print", EXPERIMENTAL),
    ("animal", "entity", "category", ("hayvan", "animal"), "animal", SUPPORTED),
    ("bird", "animal", "family", ("kus", "kuş", "bird"), "bird", EXPERIMENTAL),
    ("crow", "bird", "species", ("karga", "crow"), "crow bird corvus", EXPERIMENTAL),
    ("stork", "bird", "species", ("leylek", "stork"), "stork bird", EXPERIMENTAL),
    ("eagle", "bird", "species", ("kartal", "eagle"), "eagle bird", EXPERIMENTAL),
    ("sparrow", "bird", "species", ("serce", "sparrow"), "sparrow bird", EXPERIMENTAL),
    ("pigeon", "bird", "species", ("guvercin", "pigeon"), "pigeon dove", EXPERIMENTAL),
    ("mammal", "animal", "family", ("memeli", "mammal"), "mammal", EXPERIMENTAL),
    ("mouse", "mammal", "species", ("fare", "mouse", "mice"), "mouse", EXPERIMENTAL),
    ("leopard", "mammal", "species", ("leopar", "leopard"), "leopard", SUPPORTED),
    ("insect", "animal", "family", ("bocek", "böcek", "insect"), "insect", EXPERIMENTAL),
    ("ant", "insect", "species", ("karinca", "karınca", "ant"), "ant insect", EXPERIMENTAL),
    ("bee", "insect", "species", ("ari", "arı", "bee"), "bee insect", EXPERIMENTAL),
    ("butterfly", "insect", "species", ("kelebek", "butterfly"), "butterfly", EXPERIMENTAL),
    ("ladybug", "insect", "species", ("ugur", "ladybug"), "ladybug", EXPERIMENTAL),
    ("fish", "animal", "family", ("balik", "balık", "fish"), "fish", EXPERIMENTAL),
    ("reptile", "animal", "family", ("surungen", "reptile"), "reptile", EXPERIMENTAL),
    ("snake", "reptile", "species", ("yilan", "snake"), "snake", SUPPORTED),
    ("zebra", "mammal", "species", ("zebra",), "zebra", SUPPORTED),
    ("tiger", "mammal", "species", ("kaplan", "tiger"), "tiger", SUPPORTED),
    # Expanded universal object ontology. These are queryable and rankable with the
    # existing CLIP/DINO stack; entries without a dedicated classifier remain EXPERIMENTAL.
    ("cat", "mammal", "family", ("kedi", "cat"), "cat animal", EXPERIMENTAL),
    ("kitten", "cat", "subtype", ("yavru kedi", "kitten", "baby cat"), "kitten young cat", EXPERIMENTAL),
    ("persian_cat", "cat", "breed", ("iran kedisi", "persian cat"), "Persian cat", EXPERIMENTAL),
    ("siamese_cat", "cat", "breed", ("siamese cat", "siamese"), "Siamese cat", EXPERIMENTAL),
    ("british_shorthair", "cat", "breed", ("british shorthair", "british shorthair cat"), "British Shorthair cat", EXPERIMENTAL),
    ("maine_coon", "cat", "breed", ("maine coon",), "Maine Coon cat", EXPERIMENTAL),
    ("scottish_fold", "cat", "breed", ("scottish fold",), "Scottish Fold cat", EXPERIMENTAL),
    ("dog", "mammal", "family", ("kopek", "köpek", "dog"), "dog animal", EXPERIMENTAL),
    ("puppy", "dog", "subtype", ("yavru kopek", "yavru köpek", "puppy", "young dog"), "puppy young dog", EXPERIMENTAL),
    ("labrador", "dog", "breed", ("labrador", "labrador retriever"), "Labrador retriever dog", EXPERIMENTAL),
    ("german_shepherd", "dog", "breed", ("german shepherd", "alman kurdu"), "German Shepherd dog", EXPERIMENTAL),
    ("golden_retriever", "dog", "breed", ("golden retriever",), "Golden Retriever dog", EXPERIMENTAL),
    ("bulldog", "dog", "breed", ("bulldog",), "bulldog dog", EXPERIMENTAL),
    ("poodle", "dog", "breed", ("poodle", "kaniş", "kanis"), "poodle dog", EXPERIMENTAL),
    ("horse", "mammal", "species", ("at", "horse"), "horse animal", EXPERIMENTAL),
    ("cow", "mammal", "species", ("inek", "cow"), "cow animal", EXPERIMENTAL),
    ("sheep", "mammal", "species", ("koyun", "sheep"), "sheep animal", EXPERIMENTAL),
    ("goat", "mammal", "species", ("keçi", "keci", "goat"), "goat animal", EXPERIMENTAL),
    ("pig", "mammal", "species", ("domuz", "pig"), "pig animal", EXPERIMENTAL),
    ("deer", "mammal", "species", ("geyik", "deer"), "deer animal", EXPERIMENTAL),
    ("fox", "mammal", "species", ("tilki", "fox"), "fox animal", EXPERIMENTAL),
    ("wolf", "mammal", "species", ("kurt", "wolf"), "wolf animal", EXPERIMENTAL),
    ("bear", "mammal", "species", ("ayı", "ayi", "bear"), "bear animal", EXPERIMENTAL),
    ("lion", "mammal", "species", ("aslan", "lion"), "lion animal", EXPERIMENTAL),
    ("elephant", "mammal", "species", ("fil", "elephant"), "elephant animal", EXPERIMENTAL),
    ("giraffe", "mammal", "species", ("zürafa", "zurafa", "giraffe"), "giraffe animal", EXPERIMENTAL),
    ("monkey", "mammal", "species", ("maymun", "monkey"), "monkey animal", EXPERIMENTAL),
    ("gorilla", "mammal", "species", ("goril", "gorilla"), "gorilla animal", EXPERIMENTAL),
    ("panda", "mammal", "species", ("panda",), "panda animal", EXPERIMENTAL),
    ("koala", "mammal", "species", ("koala",), "koala animal", EXPERIMENTAL),
    ("kangaroo", "mammal", "species", ("kanguru", "kangaroo"), "kangaroo animal", EXPERIMENTAL),
    ("rabbit", "mammal", "species", ("tavşan", "tavsan", "rabbit", "bunny"), "rabbit animal", EXPERIMENTAL),
    ("hamster", "mammal", "species", ("hamster",), "hamster animal", EXPERIMENTAL),
    ("parrot", "bird", "species", ("papağan", "papagan", "parrot"), "parrot bird", EXPERIMENTAL),
    ("canary", "bird", "species", ("kanarya", "canary"), "canary bird", EXPERIMENTAL),
    ("owl", "bird", "species", ("baykuş", "baykus", "owl"), "owl bird", EXPERIMENTAL),
    ("falcon", "bird", "species", ("doğan", "dogan", "falcon"), "falcon bird", EXPERIMENTAL),
    ("hawk", "bird", "species", ("şahin", "sahin", "hawk"), "hawk bird", EXPERIMENTAL),
    ("swan", "bird", "species", ("kuğu", "kugu", "swan"), "swan bird", EXPERIMENTAL),
    ("duck", "bird", "species", ("ördek", "ordek", "duck"), "duck bird", EXPERIMENTAL),
    ("goose", "bird", "species", ("kaz", "goose"), "goose bird", EXPERIMENTAL),
    ("flamingo", "bird", "species", ("flamingo",), "flamingo bird", EXPERIMENTAL),
    ("peacock", "bird", "species", ("tavus kuşu", "tavus", "peacock"), "peacock bird", EXPERIMENTAL),
    ("penguin", "bird", "species", ("penguen", "penguin"), "penguin bird", EXPERIMENTAL),
    ("seagull", "bird", "species", ("martı", "marti", "seagull"), "seagull bird", EXPERIMENTAL),
    ("woodpecker", "bird", "species", ("ağaçkakan", "agackakan", "woodpecker"), "woodpecker bird", EXPERIMENTAL),
    ("hummingbird", "bird", "species", ("sinek kuşu", "sinek kusu", "hummingbird"), "hummingbird", EXPERIMENTAL),
    ("crocodile", "reptile", "species", ("timsah", "crocodile"), "crocodile animal", EXPERIMENTAL),
    ("turtle", "reptile", "species", ("kaplumbağa", "kaplumbaga", "turtle"), "turtle animal", EXPERIMENTAL),
    ("lizard", "reptile", "species", ("kertenkele", "lizard"), "lizard animal", EXPERIMENTAL),
    ("frog", "animal", "species", ("kurbağa", "kurbaga", "frog"), "frog animal", EXPERIMENTAL),
    ("spider", "insect", "species", ("örümcek", "orumcek", "spider"), "spider", EXPERIMENTAL),
    ("dragonfly", "insect", "species", ("yusufçuk", "yusufcuk", "dragonfly"), "dragonfly", EXPERIMENTAL),
    ("beetle", "insect", "species", ("böcek", "bocek", "beetle"), "beetle insect", EXPERIMENTAL),
    ("grasshopper", "insect", "species", ("çekirge", "cekirge", "grasshopper"), "grasshopper", EXPERIMENTAL),
    ("fly", "insect", "species", ("sinek", "fly"), "fly insect", EXPERIMENTAL),
    ("mosquito", "insect", "species", ("sivrisinek", "sivrisinek", "mosquito"), "mosquito", EXPERIMENTAL),
    ("shark", "fish", "species", ("köpek balığı", "kopek baligi", "shark"), "shark", EXPERIMENTAL),
    ("whale", "animal", "species", ("balina", "whale"), "whale", EXPERIMENTAL),
    ("dolphin", "animal", "species", ("yunus", "dolphin"), "dolphin", EXPERIMENTAL),
    ("seal", "animal", "species", ("fok", "seal"), "seal animal", EXPERIMENTAL),
    ("octopus", "animal", "species", ("ahtapot", "octopus"), "octopus", EXPERIMENTAL),
    ("jellyfish", "animal", "species", ("denizanası", "denizanasi", "jellyfish"), "jellyfish", EXPERIMENTAL),
    ("tree", "plant", "family", ("ağaç", "agac", "tree"), "tree", EXPERIMENTAL),
    ("leaf", "plant", "family", ("yaprak", "leaf"), "leaf", EXPERIMENTAL),
    ("grass", "plant", "family", ("çim", "cim", "grass"), "grass", EXPERIMENTAL),
    ("forest", "entity", "family", ("orman", "forest"), "forest landscape", EXPERIMENTAL),
    ("mountain", "entity", "family", ("dağ", "dag", "mountain"), "mountain landscape", EXPERIMENTAL),
    ("beach", "entity", "family", ("plaj", "beach"), "beach landscape", EXPERIMENTAL),
    ("river", "entity", "family", ("nehir", "river"), "river landscape", EXPERIMENTAL),
    ("lake", "entity", "family", ("göl", "gol", "lake"), "lake landscape", EXPERIMENTAL),
    ("sky", "entity", "family", ("gökyüzü", "gokyuzu", "sky"), "sky", EXPERIMENTAL),
    ("cloud", "entity", "family", ("bulut", "cloud"), "cloud", EXPERIMENTAL),
    ("sun", "entity", "family", ("güneş", "gunes", "sun"), "sun", EXPERIMENTAL),
    ("moon", "entity", "family", ("ay", "moon"), "moon", EXPERIMENTAL),
    ("bicycle", "vehicle", "family", ("bisiklet", "bicycle", "bike"), "bicycle", EXPERIMENTAL),
    ("road_bike", "bicycle", "subtype", ("yol bisikleti", "road bike"), "road bicycle", EXPERIMENTAL),
    ("mountain_bike", "bicycle", "subtype", ("dağ bisikleti", "dag bisikleti", "mountain bike", "mtb"), "mountain bike", EXPERIMENTAL),
    ("city_bike", "bicycle", "subtype", ("şehir bisikleti", "sehir bisikleti", "city bike"), "city bicycle", EXPERIMENTAL),
    ("electric_bike", "bicycle", "subtype", ("elektrikli bisiklet", "electric bike", "e-bike"), "electric bicycle", EXPERIMENTAL),
    ("bmx", "bicycle", "subtype", ("bmx",), "BMX bicycle", EXPERIMENTAL),
    ("scooter", "vehicle", "family", ("scooter", "skuter", "scooter"), "scooter", EXPERIMENTAL),
    ("skateboard", "vehicle", "family", ("kaykay", "skateboard"), "skateboard", EXPERIMENTAL),
    ("van", "vehicle", "family", ("van", "minibüs", "minibus"), "van", EXPERIMENTAL),
    ("suv", "car", "subtype", ("suv", "arazi aracı", "arazi araci"), "SUV car", EXPERIMENTAL),
    ("pickup", "car", "subtype", ("pickup", "pick-up"), "pickup truck", EXPERIMENTAL),
    ("sports_car", "car", "subtype", ("spor araba", "spor otomobil", "sports car"), "sports car", EXPERIMENTAL),
    ("sedan", "car", "subtype", ("sedan",), "sedan car", EXPERIMENTAL),
    ("hatchback", "car", "subtype", ("hatchback",), "hatchback car", EXPERIMENTAL),
    ("coupe", "car", "subtype", ("coupe", "coupe car"), "coupe car", EXPERIMENTAL),
    ("convertible", "car", "subtype", ("cabrio", "convertible", "üstü açılır"), "convertible car", EXPERIMENTAL),
    ("minivan", "car", "subtype", ("minivan",), "minivan", EXPERIMENTAL),
    ("train", "vehicle", "family", ("tren", "train"), "train", EXPERIMENTAL),
    ("tram", "vehicle", "family", ("tramvay", "tram"), "tram", EXPERIMENTAL),
    ("subway", "vehicle", "family", ("metro", "subway"), "subway train", EXPERIMENTAL),
    ("airplane", "vehicle", "family", ("uçak", "ucak", "airplane", "plane"), "airplane", EXPERIMENTAL),
    ("helicopter", "vehicle", "family", ("helikopter", "helicopter"), "helicopter", EXPERIMENTAL),
    ("boat", "vehicle", "family", ("tekne", "boat"), "boat", EXPERIMENTAL),
    ("ship", "vehicle", "family", ("gemi", "ship"), "ship", EXPERIMENTAL),
    ("tractor", "vehicle", "family", ("traktör", "traktor", "tractor"), "tractor", EXPERIMENTAL),
    ("ambulance", "vehicle", "family", ("ambulans", "ambulance"), "ambulance", EXPERIMENTAL),
    ("fire_truck", "vehicle", "family", ("itfaiye aracı", "itfaiye araci", "fire truck"), "fire truck", EXPERIMENTAL),
    ("police_car", "car", "subtype", ("polis arabası", "polis arabasi", "police car"), "police car", EXPERIMENTAL),
    # Car brands: zero-shot brand recognition until dedicated brand/model classifiers are installed.
    ("toyota", "car", "brand", ("toyota",), "Toyota car", EXPERIMENTAL),
    ("ford", "car", "brand", ("ford",), "Ford car", EXPERIMENTAL),
    ("volkswagen", "car", "brand", ("volkswagen", "vw"), "Volkswagen car", EXPERIMENTAL),
    ("volvo", "car", "brand", ("volvo",), "Volvo car", EXPERIMENTAL),
    ("honda", "car", "brand", ("honda",), "Honda car", EXPERIMENTAL),
    ("hyundai", "car", "brand", ("hyundai",), "Hyundai car", EXPERIMENTAL),
    ("kia", "car", "brand", ("kia",), "Kia car", EXPERIMENTAL),
    ("tesla", "car", "brand", ("tesla",), "Tesla car", EXPERIMENTAL),
    ("renault", "car", "brand", ("renault",), "Renault car", EXPERIMENTAL),
    ("peugeot", "car", "brand", ("peugeot",), "Peugeot car", EXPERIMENTAL),
    ("citroen", "car", "brand", ("citroen", "citroën"), "Citroen car", EXPERIMENTAL),
    ("fiat", "car", "brand", ("fiat",), "Fiat car", EXPERIMENTAL),
    ("opel", "car", "brand", ("opel",), "Opel car", EXPERIMENTAL),
    ("porsche", "car", "brand", ("porsche",), "Porsche car", EXPERIMENTAL),
    ("lexus", "car", "brand", ("lexus",), "Lexus car", EXPERIMENTAL),
    ("jaguar", "car", "brand", ("jaguar",), "Jaguar car", EXPERIMENTAL),
    ("land_rover", "car", "brand", ("land rover",), "Land Rover car", EXPERIMENTAL),
    ("skoda", "car", "brand", ("skoda", "škoda"), "Skoda car", EXPERIMENTAL),
    ("seat", "car", "brand", ("seat",), "SEAT car", EXPERIMENTAL),
    ("mazda", "car", "brand", ("mazda",), "Mazda car", EXPERIMENTAL),
    ("nissan", "car", "brand", ("nissan",), "Nissan car", EXPERIMENTAL),
    ("subaru", "car", "brand", ("subaru",), "Subaru car", EXPERIMENTAL),
    ("suzuki", "car", "brand", ("suzuki",), "Suzuki car", EXPERIMENTAL),
    ("chevrolet", "car", "brand", ("chevrolet", "chevy"), "Chevrolet car", EXPERIMENTAL),
    ("jeep", "car", "brand", ("jeep",), "Jeep car", EXPERIMENTAL),
    ("dodge", "car", "brand", ("dodge",), "Dodge car", EXPERIMENTAL),
    ("chrysler", "car", "brand", ("chrysler",), "Chrysler car", EXPERIMENTAL),
    ("genesis", "car", "brand", ("genesis",), "Genesis car", EXPERIMENTAL),
    ("infiniti", "car", "brand", ("infiniti",), "Infiniti car", EXPERIMENTAL),
    ("acura", "car", "brand", ("acura",), "Acura car", EXPERIMENTAL),
    # Furniture / home objects.
    ("table", "entity", "family", ("masa", "table"), "table furniture", EXPERIMENTAL),
    ("dining_table", "table", "subtype", ("yemek masası", "yemek masasi", "dining table"), "dining table", EXPERIMENTAL),
    ("coffee_table", "table", "subtype", ("sehpa", "coffee table"), "coffee table", EXPERIMENTAL),
    ("desk", "table", "subtype", ("çalışma masası", "calisma masasi", "desk"), "desk office table", EXPERIMENTAL),
    ("console_table", "table", "subtype", ("konsol", "console table"), "console table", EXPERIMENTAL),
    ("side_table", "table", "subtype", ("yan sehpa", "side table"), "side table", EXPERIMENTAL),
    ("chair", "entity", "family", ("sandalye", "chair"), "chair furniture", EXPERIMENTAL),
    ("office_chair", "chair", "subtype", ("ofis sandalyesi", "office chair"), "office chair", EXPERIMENTAL),
    ("armchair", "chair", "subtype", ("koltuk", "berjer", "armchair"), "armchair", EXPERIMENTAL),
    ("sofa", "entity", "family", ("kanepe", "koltuk", "sofa", "couch"), "sofa couch", EXPERIMENTAL),
    ("bed", "entity", "family", ("yatak", "bed"), "bed furniture", EXPERIMENTAL),
    ("wardrobe", "entity", "family", ("gardırop", "gardrop", "wardrobe"), "wardrobe", EXPERIMENTAL),
    ("cabinet", "entity", "family", ("dolap", "cabinet"), "cabinet furniture", EXPERIMENTAL),
    ("bookshelf", "entity", "family", ("kitaplık", "kitaplik", "bookshelf"), "bookshelf", EXPERIMENTAL),
    ("dresser", "entity", "family", ("şifonyer", "sifonyer", "dresser"), "dresser", EXPERIMENTAL),
    ("nightstand", "entity", "family", ("komodin", "nightstand"), "nightstand", EXPERIMENTAL),
    ("bench", "entity", "family", ("bank", "bench"), "bench", EXPERIMENTAL),
    ("stool", "entity", "family", ("tabure", "stool"), "stool", EXPERIMENTAL),
    ("shelf", "entity", "family", ("raf", "shelf"), "shelf", EXPERIMENTAL),
    ("lamp", "entity", "family", ("lamba", "abajur", "lamp"), "lamp", EXPERIMENTAL),
    ("mirror", "entity", "family", ("ayna", "mirror"), "mirror", EXPERIMENTAL),
    ("rug", "entity", "family", ("halı", "hali", "rug"), "rug carpet", EXPERIMENTAL),
    ("curtain", "entity", "family", ("perde", "curtain"), "curtain", EXPERIMENTAL),
    ("door", "entity", "family", ("kapı", "kapi", "door"), "door", EXPERIMENTAL),
    ("window", "entity", "family", ("pencere", "window"), "window", EXPERIMENTAL),
    # Electronics / personal items / tools / food / sport.
    ("phone", "entity", "family", ("telefon", "phone"), "phone", EXPERIMENTAL),
    ("smartphone", "phone", "subtype", ("akıllı telefon", "akilli telefon", "smartphone"), "smartphone", EXPERIMENTAL),
    ("tablet", "entity", "family", ("tablet",), "tablet computer", EXPERIMENTAL),
    ("laptop", "entity", "family", ("laptop", "dizüstü", "dizustu"), "laptop computer", EXPERIMENTAL),
    ("desktop_computer", "entity", "family", ("masaüstü bilgisayar", "masaustu bilgisayar", "desktop computer"), "desktop computer", EXPERIMENTAL),
    ("monitor", "entity", "family", ("monitör", "monitor"), "computer monitor", EXPERIMENTAL),
    ("keyboard", "entity", "family", ("klavye", "keyboard"), "computer keyboard", EXPERIMENTAL),
    ("mouse_device", "entity", "family", ("mouse", "bilgisayar faresi", "computer mouse"), "computer mouse device", EXPERIMENTAL),
    ("headphones", "entity", "family", ("kulaklık", "kulaklik", "headphones"), "headphones", EXPERIMENTAL),
    ("camera", "entity", "family", ("kamera", "camera"), "camera", EXPERIMENTAL),
    ("television", "entity", "family", ("televizyon", "tv", "television"), "television", EXPERIMENTAL),
    ("speaker", "entity", "family", ("hoparlör", "hoparlor", "speaker"), "speaker", EXPERIMENTAL),
    ("watch", "entity", "family", ("kol saati", "kol saati", "watch"), "wrist watch", EXPERIMENTAL),
    ("clock", "entity", "family", ("saat", "clock"), "clock", EXPERIMENTAL),
    ("printer", "entity", "family", ("yazıcı", "yazici", "printer"), "printer", EXPERIMENTAL),
    ("shirt", "entity", "family", ("gömlek", "gomlek", "shirt"), "shirt clothing", EXPERIMENTAL),
    ("tshirt", "shirt", "subtype", ("t-shirt", "tişört", "tisort", "tshirt"), "t-shirt", EXPERIMENTAL),
    ("dress", "entity", "family", ("elbise", "dress"), "dress clothing", EXPERIMENTAL),
    ("jacket", "entity", "family", ("ceket", "jacket"), "jacket", EXPERIMENTAL),
    ("coat", "entity", "family", ("palto", "kaban", "coat"), "coat", EXPERIMENTAL),
    ("pants", "entity", "family", ("pantolon", "pants"), "pants clothing", EXPERIMENTAL),
    ("jeans", "pants", "subtype", ("kot", "kot pantolon", "jeans"), "jeans", EXPERIMENTAL),
    ("skirt", "entity", "family", ("etek", "skirt"), "skirt clothing", EXPERIMENTAL),
    ("shoe", "entity", "family", ("ayakkabı", "ayakkabi", "shoe"), "shoe", EXPERIMENTAL),
    ("sneaker", "shoe", "subtype", ("spor ayakkabı", "spor ayakkabi", "sneaker"), "sneaker shoe", EXPERIMENTAL),
    ("hat", "entity", "family", ("şapka", "sapka", "hat"), "hat", EXPERIMENTAL),
    ("bag", "entity", "family", ("çanta", "canta", "bag"), "bag", EXPERIMENTAL),
    ("backpack", "bag", "subtype", ("sırt çantası", "sirt cantasi", "backpack"), "backpack", EXPERIMENTAL),
    ("cup", "entity", "family", ("fincan", "kupa", "cup"), "cup", EXPERIMENTAL),
    ("bottle", "entity", "family", ("şişe", "sise", "bottle"), "bottle", EXPERIMENTAL),
    ("glass", "entity", "family", ("bardak", "glass"), "drinking glass", EXPERIMENTAL),
    ("plate", "entity", "family", ("tabak", "plate"), "plate", EXPERIMENTAL),
    ("bowl", "entity", "family", ("kase", "bowl"), "bowl", EXPERIMENTAL),
    ("fork", "entity", "family", ("çatal", "catal", "fork"), "fork", EXPERIMENTAL),
    ("knife", "entity", "family", ("bıçak", "bicak", "knife"), "knife", EXPERIMENTAL),
    ("spoon", "entity", "family", ("kaşık", "kasik", "spoon"), "spoon", EXPERIMENTAL),
    ("pan", "entity", "family", ("tava", "pan"), "frying pan", EXPERIMENTAL),
    ("pot", "entity", "family", ("tencere", "pot"), "cooking pot", EXPERIMENTAL),
    ("apple", "entity", "family", ("elma", "apple"), "apple fruit", EXPERIMENTAL),
    ("banana", "entity", "family", ("muz", "banana"), "banana fruit", EXPERIMENTAL),
    ("orange", "entity", "family", ("portakal", "orange"), "orange fruit", EXPERIMENTAL),
    ("lemon", "entity", "family", ("limon", "lemon"), "lemon fruit", EXPERIMENTAL),
    ("strawberry", "entity", "family", ("çilek", "cilek", "strawberry"), "strawberry fruit, not a flower", EXPERIMENTAL),
    ("cherry", "entity", "family", ("kiraz", "cherry", "visne", "vişne"), "cherry fruit, not cherry blossom", EXPERIMENTAL),
    ("watermelon", "entity", "family", ("karpuz", "watermelon"), "watermelon", EXPERIMENTAL),
    ("tomato", "entity", "family", ("domates", "tomato"), "tomato", EXPERIMENTAL),
    ("potato", "entity", "family", ("patates", "potato"), "potato", EXPERIMENTAL),
    ("bread", "entity", "family", ("ekmek", "bread"), "bread", EXPERIMENTAL),
    ("cake", "entity", "family", ("pasta", "cake"), "cake", EXPERIMENTAL),
    ("pizza", "entity", "family", ("pizza",), "pizza", EXPERIMENTAL),
    ("burger", "entity", "family", ("hamburger", "burger"), "hamburger", EXPERIMENTAL),
    ("ball", "entity", "family", ("top", "ball"), "ball", EXPERIMENTAL),
    ("football", "ball", "subtype", ("futbol topu", "football"), "football", EXPERIMENTAL),
    ("basketball", "ball", "subtype", ("basketbol topu", "basketball"), "basketball", EXPERIMENTAL),
    ("tennis_racket", "entity", "family", ("tenis raketi", "tenis raketi", "tennis racket"), "tennis racket", EXPERIMENTAL),
    ("golf_club", "entity", "family", ("golf sopası", "golf sopasi", "golf club"), "golf club", EXPERIMENTAL),
    ("skis", "entity", "family", ("kayak", "skis"), "skis", EXPERIMENTAL),
    ("surfboard", "entity", "family", ("sörf tahtası", "sorf tahtasi", "surfboard"), "surfboard", EXPERIMENTAL),
    ("book", "entity", "family", ("kitap", "book"), "book", EXPERIMENTAL),
    ("newspaper", "entity", "family", ("gazete", "newspaper"), "newspaper", EXPERIMENTAL),
    ("pen", "entity", "family", ("kalem", "pen"), "pen", EXPERIMENTAL),
    ("pencil", "entity", "family", ("kurşun kalem", "kursun kalem", "pencil"), "pencil", EXPERIMENTAL),
    ("scissors", "entity", "family", ("makas", "scissors"), "scissors", EXPERIMENTAL),
    ("hammer", "entity", "family", ("çekiç", "cekic", "hammer"), "hammer", EXPERIMENTAL),
    ("screwdriver", "entity", "family", ("tornavida", "screwdriver"), "screwdriver", EXPERIMENTAL),
    ("drill", "entity", "family", ("matkap", "drill"), "power drill", EXPERIMENTAL),
    ("toolbox", "entity", "family", ("alet çantası", "alet cantasi", "toolbox"), "toolbox", EXPERIMENTAL),
    ("plant", "entity", "category", ("bitki", "plant"), "plant", SUPPORTED),
    ("flower", "plant", "family", ("cicek", "çiçek", "flower", "floral"), "flower", SUPPORTED),
    ("rose", "flower", "species", ("gul", "gül", "rose"), "rose flower", SUPPORTED),
    ("daisy", "flower", "species", ("papatya", "daisy"), "daisy flower", SUPPORTED),
    ("tulip", "flower", "species", ("lale", "tulip"), "tulip flower", SUPPORTED),
    ("orchid", "flower", "species", ("orkide", "orchid"), "orchid", EXPERIMENTAL),
    ("peony", "flower", "species", ("sakayik", "peony"), "peony", EXPERIMENTAL),
    ("vehicle", "entity", "category", ("arac", "araç", "vehicle"), "vehicle", EXPERIMENTAL),
    ("car", "vehicle", "family", ("araba", "otomobil", "car"), "car automobile", EXPERIMENTAL),
    ("togg", "car", "brand", ("togg",), "Togg Turkish electric car", EXPERIMENTAL),
    ("t10x", "togg", "model", ("t10x", "togg t10x"), "Togg T10X electric car model", EXPERIMENTAL),
    ("bmw", "car", "brand", ("bmw",), "BMW car", EXPERIMENTAL),
    ("mercedes", "car", "brand", ("mercedes", "mercedes-benz"), "Mercedes-Benz car", EXPERIMENTAL),
    ("audi", "car", "brand", ("audi",), "Audi car", EXPERIMENTAL),
    ("truck", "vehicle", "family", ("kamyon", "truck"), "truck", EXPERIMENTAL),
    ("bus", "vehicle", "family", ("otobus", "bus"), "bus", EXPERIMENTAL),
    ("motorcycle", "vehicle", "family", ("motosiklet", "motorcycle"), "motorcycle", EXPERIMENTAL),
    ("building", "entity", "family", ("bina", "building"), "building architecture", EXPERIMENTAL),
    ("logo", "entity", "family", ("logo", "monogram"), "logo monogram", SUPPORTED),
    ("accessory", "entity", "category", ("aksesuar", "accessory"), "fashion accessory", EXPERIMENTAL),
    ("jewelry", "accessory", "family", ("takı", "taki", "jewelry", "mücevher"),
     "fashion jewelry necklace bracelet earrings, not a textile ornament", EXPERIMENTAL),
    ("brooch", "jewelry", "subtype", ("broş", "bros", "brooch"), "brooch jewelry pin", EXPERIMENTAL),
    ("necklace", "jewelry", "subtype", ("kolye", "necklace", "gerdanlık"), "necklace jewelry", EXPERIMENTAL),
    ("baroque_pattern", "entity", "family", ("barok desen", "barok", "baroque", "baroque pattern"),
     "baroque ornamental textile pattern", EXPERIMENTAL),
    ("plaid", "entity", "family", ("ekose", "kareli", "tartan", "plaid"),
     "tartan plaid check textile ekose", EXPERIMENTAL),
    ("ethnic_print", "entity", "family", ("etnik", "etnik desen", "ethnic print", "tribal print"),
     "ethnic folk textile print, not a racial label", EXPERIMENTAL),
    ("watercolor", "entity", "family", ("suluboya", "watercolor", "watercolor effect"),
     "watercolor wash textile pattern", EXPERIMENTAL),
    ("brushstroke", "entity", "family", ("fırça etkisi", "fırça darbesi", "firca etkisi", "brushstroke", "brush stroke"),
     "visible paint brush stroke texture", EXPERIMENTAL),
    ("paint_splatter", "entity", "family", ("sıçrama", "sicrama", "boya sıçraması", "paint splatter"),
     "paint splatter spray textile texture", EXPERIMENTAL),
    ("rococo", "entity", "family", ("rokoko", "rococo"),
     "rococo ornamental textile pattern", EXPERIMENTAL),
]


def _build_indexes() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    nodes: dict[str, dict[str, Any]] = {}
    alias: dict[str, str] = {}
    for nid, parent, level, aliases, prompt, status in _NODES:
        nodes[nid] = {
            "id": nid,
            "parent": parent,
            "level": level,
            "aliases": aliases,
            "prompt": prompt or nid,
            "status": status,
        }
        alias[nid] = nid
        for a in aliases:
            alias[normalize_turkish(a)] = nid
    return nodes, alias


ONTOLOGY, ALIAS_TO_NODE = _build_indexes()


def node_path(node_id: str) -> list[str]:
    out: list[str] = []
    cur = node_id
    seen: set[str] = set()
    while cur and cur in ONTOLOGY and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = str(ONTOLOGY[cur].get("parent") or "")
    out.reverse()
    return out


def siblings(node_id: str) -> list[str]:
    parent = str((ONTOLOGY.get(node_id) or {}).get("parent") or "")
    if not parent:
        return []
    return [
        nid
        for nid, meta in ONTOLOGY.items()
        if meta.get("parent") == parent and nid != node_id
    ]


def children(node_id: str) -> list[str]:
    return [nid for nid, meta in ONTOLOGY.items() if meta.get("parent") == node_id]


@dataclass
class UniversalQuery:
    raw: str = ""
    node_id: str = ""
    path: list[str] = field(default_factory=list)
    family: str = ""
    leaf: str = ""
    rivals: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    level: str = ""
    pattern_query: dict[str, Any] = field(default_factory=dict)
    status: str = SUPPORTED
    open_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VisualDNA:
    """Query-time card. Not persisted. Compatible with Pattern DNA extras."""

    objects: list[dict[str, Any]] = field(default_factory=list)
    object_hierarchy: list[str] = field(default_factory=list)
    fine_grained: str = "unknown"
    identity_cluster: str = ""
    identity_status: str = EXPERIMENTAL
    motifs: list[dict[str, Any]] = field(default_factory=list)
    pattern_family: str = ""
    brands: list[str] = field(default_factory=list)
    ocr: str = ""
    colors: list[str] = field(default_factory=list)
    texture: str = ""
    geometry: str = ""
    representation: str = "unknown"
    textile: bool | None = None
    composition: str = "unknown"
    base: str = ""
    overlay: list[str] = field(default_factory=list)
    relationships: list[dict[str, str]] = field(default_factory=list)
    confidence: float = 0.0
    search_tier: str = TIER_SEMANTIC
    capability_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_universal_query(text: str) -> UniversalQuery:
    raw = (text or "").strip()
    pq = parse_pattern_query_v2(raw)
    uq = UniversalQuery(raw=raw, pattern_query=pq.to_dict(), required=list(pq.required))
    norm = normalize_turkish(raw)
    hits: list[str] = []
    for alias, nid in sorted(ALIAS_TO_NODE.items(), key=lambda kv: -len(kv[0])):
        if not alias or nid in hits:
            continue
        if _alias_in_text(alias, norm):
            hits.append(nid)
    preferred = ""
    for nid in hits:
        level = ONTOLOGY[nid]["level"]
        if level in ("species", "breed", "brand", "model"):
            preferred = nid
            break
    if not preferred:
        for nid in hits:
            if ONTOLOGY[nid]["level"] in ("family", "subtype"):
                preferred = nid
                break
    uq.node_id = preferred or (hits[0] if hits else "")
    families_hit: list[str] = []
    for nid in hits:
        fam = next(
            (p for p in reversed(node_path(nid)) if ONTOLOGY.get(p, {}).get("level") == "family"),
            nid,
        )
        if fam not in families_hit:
            families_hit.append(fam)
    if uq.node_id:
        uq.path = node_path(uq.node_id)
        uq.level = str(ONTOLOGY[uq.node_id]["level"])
        uq.status = str(ONTOLOGY[uq.node_id]["status"])
        uq.leaf = uq.node_id if uq.level in ("species", "breed", "brand", "model") else ""
        uq.family = next(
            (p for p in reversed(uq.path) if ONTOLOGY.get(p, {}).get("level") in ("family", "category")),
            uq.node_id,
        )
        if uq.leaf:
            uq.rivals = siblings(uq.node_id)
        extra = [n for n in hits if n in ONTOLOGY and n not in uq.required]
        uq.required = list(dict.fromkeys(uq.required + extra))
        # Composite across families: do not treat as a single forced leaf.
        if len(families_hit) >= 2 or str(pq.composition or "") == "composite":
            uq.leaf = ""
    _attach_open_concepts(uq)
    return uq


def _attach_open_concepts(uq: UniversalQuery) -> None:
    """Unknown tokens → open CLIP concepts. Does not replace textile motif queries."""
    if is_textile_discovery_query(uq):
        return
    from core.visual_concept import compile_visual_query

    compiled = compile_visual_query(uq.raw)
    extras: list[str] = []
    for c in compiled.concepts:
        lemma = str(c.concept_id.split(":", 1)[-1] or "").replace("_", " ").strip()
        key = lemma.replace(" ", "_")
        if lemma in ONTOLOGY or key in ONTOLOGY:
            continue
        extras.append(c.concept_id)
    if not extras:
        return
    uq.open_ids = list(dict.fromkeys(list(uq.open_ids) + extras))
    uq.required = list(dict.fromkeys(list(uq.required) + extras))
    if not uq.node_id:
        uq.node_id = extras[0]
        uq.status = EXPERIMENTAL
        uq.family = "entity"
        uq.level = "open"


def _is_gender_only_query(query: UniversalQuery) -> bool:
    from core.textile_terms import normalize_turkish

    return normalize_turkish(getattr(query, "raw", "") or "") in {
        "kadin",
        "kadın",
        "erkek",
        "woman",
        "man",
        "female",
        "male",
        "bayan",
        "women",
        "men",
    }


def _row_has_person_box(dbg: dict[str, Any], row: Any) -> bool:
    wrap = dbg.get("texture_map") if isinstance(dbg.get("texture_map"), dict) else {}
    goi = wrap.get("global_object_intelligence") if isinstance(wrap, dict) else None
    blobs: list[Any] = []
    if isinstance(goi, dict):
        blobs.extend(goi.get("objects") or ())
    dna = wrap.get("visual_concept_dna") if isinstance(wrap, dict) else None
    if isinstance(dna, dict):
        blobs.extend(dna.get("objects") or ())
    for obj in blobs:
        lab = ""
        if isinstance(obj, dict):
            lab = str(obj.get("label") or obj.get("canonical_name") or obj.get("label_tr") or "")
        else:
            lab = str(obj)
        low = lab.lower()
        if any(x in low for x in ("person", "woman", "man", "human", "face", "insan", "kadın", "kadin", "erkek")):
            return True
    if dbg.get("face_gender_match") or dbg.get("face_match"):
        return True
    return False


def is_hard_object_query(query: UniversalQuery) -> bool:
    if str(getattr(query, "node_id", "") or "").startswith("open:"):
        return True
    if any(str(x).startswith("open:") for x in (getattr(query, "open_ids", None) or ())):
        return True
    return bool(query.node_id) and query.node_id in HARD_OBJECT_NODES


def is_textile_discovery_query(query: UniversalQuery) -> bool:
    """Textile motif search stays on Pattern Intel v2; do not treat as a 24-hit object leaf.

    Leopard/zebra/floral/paisley queries share CLIP space with image search.
    Capping them like crow/BMW hides visually matching prints.
    """
    pq = query.pattern_query if isinstance(query.pattern_query, dict) else {}
    family = str(pq.get("family") or "")
    primary = str(pq.get("primary") or "")
    if family in {"illustration_object", "garment_photo"}:
        return False
    if family in {
        "animal_print", "floral", "geometric", "paisley", "plaid_check",
        "stripe", "marble_abstract", "camouflage", "monogram_logo", "baroque",
    }:
        return True
    if primary in {
        "leopard", "zebra", "tiger", "snake", "floral", "rose", "daisy",
        "tulip", "leaf", "paisley", "geometric", "polka_dot", "stripe",
        "baroque", "watercolor", "ethnic_print", "plaid", "brushstroke",
        "paint_splatter", "rococo",
    }:
        return True
    return False


def _path_tokens(*parts: str) -> set[str]:
    blob = normalize_turkish(" ".join(str(p or "") for p in parts))
    return {t for t in re.findall(r"[a-z0-9]+", blob) if len(t) >= 3}


def _object_aliases(query: UniversalQuery) -> set[str]:
    names: set[str] = set()
    nids = [query.node_id, query.leaf, query.family, *query.path, *query.required]
    for nid in nids:
        if not nid or nid not in ONTOLOGY:
            continue
        names.add(normalize_turkish(nid))
        for alias in ONTOLOGY[nid].get("aliases") or ():
            tok = normalize_turkish(alias)
            if tok:
                names.add(tok)
    return {n for n in names if len(n) >= 3}


def _row_path_tokens(row: Any) -> set[str]:
    dbg = getattr(row, "debug", {}) or {}
    return _path_tokens(
        str(getattr(row, "filename", "") or ""),
        str(getattr(row, "path", "") or ""),
        str(dbg.get("filename") or ""),
        str(dbg.get("path") or ""),
    )


def _text_alias_hit(query: UniversalQuery, row: Any) -> bool:
    return bool(_object_aliases(query) & _row_path_tokens(row))


def _path_conflict(query: UniversalQuery, row: Any) -> bool:
    bad = _CONFLICT_PATH_TOKENS.get(query.node_id) or _CONFLICT_PATH_TOKENS.get(query.family) or frozenset()
    if not bad:
        return False
    toks = _row_path_tokens(row)
    if _text_alias_hit(query, row):
        return False
    return bool(toks & bad)


def _sanitize_object_scores(
    query: UniversalQuery,
    scores: dict[str, float],
    dbg: dict[str, Any],
) -> dict[str, float]:
    """Generic query CLIP is semantic similarity, not object evidence."""
    out = {k: float(v or 0) for k, v in scores.items()}
    if not is_hard_object_query(query) or not query.node_id:
        return out
    if dbg.get("clip_object_prompt"):
        return out
    generic = float(dbg.get("clip_score") or 0)
    kids = children(query.node_id)
    child_hi = max((out.get(c, 0.0) for c in kids), default=0.0)
    node_sc = out.get(query.node_id, 0.0)
    if generic and abs(node_sc - generic) < 1e-4 and child_hi < OBJECT_OWN_MIN:
        out[query.node_id] = 0.0
    return out


def _object_own(query: UniversalQuery, scores: dict[str, float]) -> float:
    if not query.node_id:
        return 0.0
    own = float(scores.get(query.node_id, 0.0) or 0.0)
    child_hi = max((float(scores.get(c, 0.0) or 0.0) for c in children(query.node_id)), default=0.0)
    return max(own, child_hi)


def _distractor_hit(query: UniversalQuery, scores: dict[str, float]) -> tuple[float, str]:
    names = _DISTRACTOR_FOR.get(query.node_id) or _DISTRACTOR_FOR.get(query.family) or ()
    best, who = 0.0, ""
    for name in names:
        sc = float(scores.get(name, 0.0) or 0.0)
        if name == "flower":
            sc = max(sc, float(scores.get("floral", 0.0) or 0.0))
        if sc > best:
            best, who = sc, name
    return best, who


def classify_object_evidence(
    query: UniversalQuery,
    scores: dict[str, float],
    *,
    tier: str,
    reason: str,
    row: Any | None = None,
    card: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Return (evidence_level, extra_reason). Missing evidence is not a match."""
    if not query.node_id:
        return EVIDENCE_UNKNOWN, reason
    own = _object_own(query, scores)
    dist, dist_name = _distractor_hit(query, scores)
    req = {str(x) for x in (query.required or [])}
    if dist_name and (
        dist_name in req
        or (dist_name in ("flower", "floral") and req & {"floral", "flower", "rose", "daisy", "tulip", "orchid", "peony"})
    ):
        dist, dist_name = 0.0, ""
    card = card or {}
    card_ids = {
        str(h.get("id") or "")
        for h in list(card.get("objects") or []) + list(card.get("motifs") or [])
        if float(h.get("score") or 0) >= OBJECT_OWN_MIN
    }
    card_hit = bool(
        card_ids
        & {
            query.node_id,
            query.family,
            query.leaf,
            *children(query.node_id),
        }
    )
    conflict = bool(row is not None and _path_conflict(query, row))
    hard = is_hard_object_query(query)
    dbg = getattr(row, "debug", {}) or {} if row is not None else {}
    if dbg.get("object_index_hit"):
        card_hit = True

    if hard and (conflict or (dist >= OBJECT_OWN_MIN and dist > own + OBJECT_MARGIN and own < OBJECT_OWN_MIN)):
        if not dbg.get("object_index_hit"):
            return EVIDENCE_MISMATCH, f"object_mismatch:{dist_name or 'path'}"
    if hard and own < OBJECT_OWN_MIN and not card_hit:
        if tier in (TIER_EXACT, TIER_VERY_SIMILAR, TIER_SAME_SUBTYPE, TIER_SAME_FAMILY):
            return EVIDENCE_UNKNOWN, "no_object_clip"
        return EVIDENCE_UNKNOWN, reason or "unknown"
    if tier == TIER_EXACT:
        return EVIDENCE_EXACT, reason
    if tier == TIER_SAME_SUBTYPE:
        return EVIDENCE_SUBTYPE, reason
    if tier in (TIER_SAME_FAMILY, TIER_VERY_SIMILAR):
        return EVIDENCE_FAMILY, reason
    if own >= OBJECT_OWN_MIN or card_hit:
        return EVIDENCE_SEMANTIC, reason
    return EVIDENCE_UNKNOWN, reason


def _uvi_explain(
    query: UniversalQuery,
    *,
    evidence: str,
    tier: str,
    scores: dict[str, float],
    dbg: dict[str, Any],
    penalty: str = "",
) -> dict[str, Any]:
    own = _object_own(query, scores)
    card = dbg.get("pattern_intel_v2") or {}
    clip = float(dbg.get("clip_score") or scores.get(query.node_id, 0.0) or 0.0)
    if evidence == EVIDENCE_EXACT:
        object_match = query.leaf or query.node_id or "None"
    elif evidence in (EVIDENCE_SUBTYPE, EVIDENCE_FAMILY):
        object_match = query.family or query.node_id or "None"
    else:
        object_match = "None"
    return {
        "object_match": object_match,
        "object_evidence": evidence,
        "pattern_match": round(float(card.get("confidence") or 0.0), 4),
        "clip": round(clip, 4),
        "uvi": round(own, 4),
        "semantic": round(float(dbg.get("hybrid") or dbg.get("semantic_score") or 0.0), 4),
        "tier": tier if evidence != EVIDENCE_MISMATCH else TIER_SEMANTIC,
        "penalty": penalty,
        "representation": str(card.get("representation") or "unknown"),
    }


def fine_grained_tier(
    query: UniversalQuery,
    clip_scores: dict[str, float],
    *,
    v11_visual_win: bool = False,
) -> tuple[str, str]:
    """Return (search_tier, reason). Never force a specific label."""
    scores = {k: float(v or 0) for k, v in (clip_scores or {}).items()}
    if not query.node_id:
        return TIER_SEMANTIC, "no_node"
    own = scores.get(query.node_id, 0.0)
    # parent query (kuş, araba): family match is success
    if query.level in ("family", "category") and not query.leaf:
        child_hi = max((scores.get(c, 0.0) for c in children(query.node_id)), default=0.0)
        fam = max(own, child_hi)
        need = OBJECT_OWN_MIN if is_hard_object_query(query) else COMPOSITE_CLIP_MIN
        # Hard object: generic visual_win is not object proof.
        if fam >= need or (v11_visual_win and not is_hard_object_query(query)):
            return TIER_SAME_FAMILY, "family_query"
        return TIER_SEMANTIC, "weak_family"

    rival_hi = 0.0
    rival_name = ""
    rivals_present = False
    for r in query.rivals:
        if r not in scores:
            continue
        rivals_present = True
        sc = float(scores.get(r, 0.0) or 0.0)
        if sc > rival_hi:
            rival_hi, rival_name = sc, r
    hard = is_hard_object_query(query)
    need = OBJECT_OWN_MIN if hard else COMPOSITE_CLIP_MIN
    if own < need and not (v11_visual_win and not hard):
        if rivals_present and rival_hi >= need:
            return TIER_SAME_FAMILY if rival_name else TIER_SEMANTIC, f"unspecific:{rival_name or 'weak'}"
        return TIER_SEMANTIC, "unknown_subtype"
    # Missing rival CLIP hits must not be treated as a species win.
    if query.rivals and not rivals_present:
        return TIER_VERY_SIMILAR, "no_rival_scores"
    if rival_hi and own <= rival_hi + 0.02:
        return TIER_SAME_FAMILY, f"rival_{rival_name}"
    if rivals_present and own >= need + 0.04 and own > rival_hi + 0.03:
        return TIER_EXACT, "leaf_win"
    if (v11_visual_win and not hard) or own >= need:
        return TIER_VERY_SIMILAR, "leaf_ok"
    return TIER_SEMANTIC, "low_conf"


def build_visual_dna(
    query: UniversalQuery,
    *,
    card: dict[str, Any] | None = None,
    clip_scores: dict[str, float] | None = None,
    ocr: str = "",
    search_tier: str = TIER_SEMANTIC,
    identity_cluster: str = "",
) -> VisualDNA:
    card = card or {}
    scores = clip_scores or {}
    if search_tier == TIER_EXACT:
        fg = query.leaf or query.node_id or "unknown"
    elif search_tier in (TIER_SAME_FAMILY, TIER_SAME_SUBTYPE, TIER_VERY_SIMILAR):
        fg = query.family or query.node_id or "unknown"
    else:
        fg = "unknown"
    dna = VisualDNA(
        objects=list(card.get("objects") or []),
        object_hierarchy=list(query.path),
        fine_grained=fg,
        identity_cluster=identity_cluster,
        identity_status=EXPERIMENTAL if identity_cluster else UNSUPPORTED,
        motifs=list(card.get("motifs") or []),
        pattern_family=str(card.get("family") or query.family or ""),
        brands=[],
        ocr=(ocr or "")[:200],
        colors=[c for c in (query.pattern_query.get("color"), query.pattern_query.get("ground")) if c],
        representation=str(card.get("representation") or "unknown"),
        textile=card.get("textile_pattern"),
        composition=str(card.get("composition") or "unknown"),
        base=str(card.get("base_motif") or ""),
        overlay=list(card.get("overlay_motifs") or []),
        confidence=float(card.get("confidence") or 0.0),
        search_tier=search_tier,
    )
    if query.pattern_query.get("relationship") in ("overlay", "mixed"):
        a = query.pattern_query.get("base_hint") or dna.base
        b = (query.pattern_query.get("overlay_hint") or (dna.overlay[0] if dna.overlay else ""))
        if a and b:
            dna.relationships.append({"from": str(a), "rel": query.pattern_query.get("relationship"), "to": str(b)})
        else:
            dna.relationships.append({"from": "", "rel": "unknown", "to": ""})
    else:
        dna.relationships.append({"from": "", "rel": "unknown", "to": ""})
    notes = []
    if query.status != SUPPORTED:
        notes.append(f"node_status={query.status}")
    if dna.identity_cluster:
        notes.append("identity=experimental_clip_cluster")
    dna.capability_notes = notes
    # never persist forbidden labels
    blob = normalize_turkish(" ".join(str(x) for x in (dna.fine_grained, dna.pattern_family, ocr)))
    if any(x in blob for x in FORBIDDEN_LABELS):
        dna.fine_grained = "unknown"
        dna.capability_notes.append("stripped_forbidden_label")
    _ = scores
    return dna


def cluster_clip_identities(
    vectors: dict[int, np.ndarray],
    *,
    threshold: float = 0.88,
) -> dict[int, str]:
    """Experimental instance grouping from existing CLIP vectors. Not face-ID."""
    ids = [i for i, v in vectors.items() if v is not None and getattr(v, "size", 0)]
    labels: dict[int, str] = {}
    next_i = 1

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        na = float(np.linalg.norm(aa))
        nb = float(np.linalg.norm(bb))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(aa, bb) / (na * nb))

    centroids: list[tuple[str, np.ndarray]] = []
    for fid in ids:
        v = np.asarray(vectors[fid], dtype=np.float32).reshape(-1)
        assigned = ""
        best = -1.0
        for lab, c in centroids:
            s = _cos(v, c)
            if s > best:
                best, assigned = s, lab
        if assigned and best >= threshold:
            labels[fid] = assigned
            # running mean
            for i, (lab, c) in enumerate(centroids):
                if lab == assigned:
                    centroids[i] = (lab, (c + v) / 2.0)
                    break
        else:
            lab = f"instance_{next_i:03d}"
            next_i += 1
            labels[fid] = lab
            centroids.append((lab, v.copy()))
    return labels


def clip_nodes_for_query(query: UniversalQuery, *, max_nodes: int = 10) -> list[str]:
    names: list[str] = []
    if query.node_id:
        names.append(query.node_id)
    names.extend(list(query.rivals)[:5])
    if not query.leaf and query.node_id:
        names.extend(children(query.node_id)[:4])
    if query.family:
        names.append(query.family)
    for c in query.required:
        if c in ONTOLOGY or c in _DISTRACTOR_PROMPTS or str(c).startswith("open:"):
            names.append(c)
    if is_hard_object_query(query):
        names.extend(_DISTRACTOR_FOR.get(query.node_id) or ())
    # Gender/person concepts need their own semantic evidence.  Do not let the
    # rival gender crowd the query out: the person parent is useful as a
    # secondary human-presence signal, while the requested gender remains first.
    if query.node_id in {"female_person", "male_person"}:
        names = [query.node_id, "person", *[n for n in names if n not in {query.node_id, "person"}]]
    names.extend(list(getattr(query, "open_ids", None) or []))
    return list(dict.fromkeys(n for n in names if n))[:max_nodes]


def _alias_in_text(alias: str, norm: str) -> bool:
    a = normalize_turkish(alias)
    if not a:
        return False
    if len(a) <= 2:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", norm))
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(a)}", norm))


def node_clip_prompt(node_id: str) -> str:
    if str(node_id or "").startswith("open:"):
        from core.visual_concept import prompt_for_open_id

        return prompt_for_open_id(node_id)
    if node_id in _DISTRACTOR_PROMPTS:
        return _DISTRACTOR_PROMPTS[node_id]
    meta = ONTOLOGY.get(node_id) or {}
    from core.visual_concept import expand_clip_prompt

    return expand_clip_prompt(str(meta.get("prompt") or node_id))


def apply_universal_ranking(
    query: UniversalQuery,
    rows: list[Any],
    *,
    clip_by_id: dict[int, dict[str, float]] | None = None,
    clip_active: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    clip_by_id = clip_by_id or {}
    hard = is_hard_object_query(query)
    # Gender is a visual attribute query, not a textile-object hard gate.
    # Keep a dedicated acceptance path so CLIP can retrieve women/men figures
    # and illustrations even when the face index is incomplete.
    gender_query = query.node_id in {"female_person", "male_person"}
    gender_cap = None
    stats = {
        "universal_visual_intel": True,
        "node": query.node_id,
        "path": query.path,
        "status": query.status,
        "hard_object": hard,
        "capabilities": {k: v for k, v in CAPABILITIES.items() if v != PLANNED},
        "tiers": {},
        "evidence": {},
        "dropped_unspecific": 0,
        "dropped_mismatch": 0,
    }
    _TIER_RANK = {
        TIER_EXACT: 0,
        TIER_VERY_SIMILAR: 1,
        TIER_SAME_SUBTYPE: 2,
        TIER_SAME_FAMILY: 3,
        TIER_SEMANTIC: 4,
        EVIDENCE_MISMATCH: 6,
    }
    kept: list[Any] = []
    for row in rows:
        fid = int(getattr(row, "file_id", 0) or 0)
        dbg = getattr(row, "debug", {}) or {}
        scores = dict(clip_by_id.get(fid) or {})
        for k, v in (dbg.get("v2_clip") or {}).items():
            scores[k] = max(float(scores.get(k, 0) or 0), float(v or 0))
        if query.node_id and dbg.get("clip_score") and (
            not hard or dbg.get("clip_object_prompt")
        ):
            # Bare kadın/erkek: generic CLIP is not female_person/male_person evidence.
            if not (gender_query and _is_gender_only_query(query)):
                scores[query.node_id] = max(
                    float(scores.get(query.node_id, 0) or 0), float(dbg.get("clip_score") or 0)
                )
        scores = _sanitize_object_scores(query, scores, dbg)
        qev_grade = str((dbg.get("query_evidence_report") or {}).get("visual_grade") or dbg.get("visual_grade") or "")
        visual_ok = qev_grade in ("visual_exact", "visual_strong")
        win = bool(dbg.get("visual_win") or visual_ok) and not hard
        own = _object_own(query, scores)
        if hard and (own >= OBJECT_OWN_MIN or visual_ok):
            win = bool(dbg.get("visual_win") or visual_ok)
        if gender_query:
            face_match = bool(dbg.get("face_gender_match"))
            person_box = _row_has_person_box(dbg, row)
            if _is_gender_only_query(query):
                if face_match or person_box:
                    win = True
            elif face_match or own >= 0.18:
                win = True
        if dbg.get("object_index_hit"):
            win = True
        tier, reason = fine_grained_tier(query, scores, v11_visual_win=win)
        card = dbg.get("pattern_intel_v2") or {}
        evidence, ev_reason = classify_object_evidence(
            query, scores, tier=tier, reason=reason, row=row, card=card
        )
        penalty = ""
        if evidence == EVIDENCE_MISMATCH:
            penalty = ev_reason
            tier = TIER_SEMANTIC
            reason = ev_reason
        elif evidence == EVIDENCE_UNKNOWN and hard and reason in ("family_query",):
            tier = TIER_SEMANTIC
            reason = ev_reason

        drop_hard = (
            hard
            and evidence == EVIDENCE_MISMATCH
            and not win
            and not _text_alias_hit(query, row)
        )
        # Concrete visual objects keep the UVI evidence gate.  Catalogue
        # clothing nodes are the exception: their filename/metadata evidence
        # is authoritative enough for text search and must not be zeroed merely
        # because no object-CLIP score was persisted.
        metadata_passthrough = query.node_id in TEXT_METADATA_NODES
        drop_pad = (
            (clip_active and query.node_id and not win and not metadata_passthrough)
            or drop_hard
        )
        if gender_query:
            gender_ok = bool(dbg.get("face_gender_match")) or _row_has_person_box(dbg, row)
            if not _is_gender_only_query(query):
                gender_ok = gender_ok or own >= 0.18
            if gender_ok:
                drop_pad = False
        if gender_query and _is_gender_only_query(query) and drop_pad:
            stats["dropped_unspecific"] += 1
            continue
        if dbg.get("object_index_hit"):
            drop_pad = False
            win = True
        if (
            str(dbg.get("evidence_type") or "") == "ZERO_SHOT_VISUAL"
            or dbg.get("zero_shot_visual")
        ) and not gender_query:
            drop_pad = False
        if drop_pad and evidence == EVIDENCE_MISMATCH:
            stats["dropped_mismatch"] += 1
            continue
        if drop_pad and query.leaf and (reason.startswith("unspecific") or tier == TIER_SEMANTIC or evidence == EVIDENCE_UNKNOWN):
            stats["dropped_unspecific"] += 1
            continue
        if drop_pad and not query.leaf and (tier == TIER_SEMANTIC or evidence == EVIDENCE_UNKNOWN):
            stats["dropped_unspecific"] += 1
            continue

        dna = build_visual_dna(query, card=card, clip_scores=scores, search_tier=tier)
        uvi_rank = _TIER_RANK.get(tier, 5)
        if gender_query:
            # For gender queries the raw visual score is the useful ordering
            # signal. Face-index matches stay first; UVI matches follow by
            # descending CLIP evidence.  Write one canonical marker that the
            # SearchEngine's final threshold guard can consume.
            if dbg.get("face_gender_match"):
                uvi_rank = 0
                row.debug = {
                    **row.debug,
                    "human_semantic_mode": True,
                    "human_semantic_only": True,
                    "human_semantic_score": 1.0,
                }
            elif own >= 0.18 and (
                not _is_gender_only_query(query) or _row_has_person_box(dbg, row)
            ):
                uvi_rank = 1
                row.debug = {
                    **row.debug,
                    "human_semantic_mode": True,
                    "human_semantic_only": True,
                    "human_semantic_score": round(float(own), 4),
                    "gender_visual_score": round(float(own), 4),
                }
        if evidence == EVIDENCE_MISMATCH:
            uvi_rank = 6
        elif evidence == EVIDENCE_UNKNOWN and hard:
            uvi_rank = max(uvi_rank, 5)
        explain = _uvi_explain(
            query, evidence=evidence, tier=tier, scores=scores, dbg=dbg, penalty=penalty
        )
        row.debug = {
            **dbg,
            "visual_dna": dna.to_dict(),
            "uvi_tier": tier,
            "uvi_reason": reason,
            "uvi_rank": uvi_rank,
            "object_evidence": evidence,
            "uvi_explain": explain,
        }
        if gender_query and dbg.get("face_gender_match"):
            row.debug["human_semantic_mode"] = True
            row.debug["human_semantic_only"] = True
            row.debug["human_semantic_score"] = 1.0
        elif gender_query and not dbg.get("face_gender_match") and own >= 0.18:
            if not _is_gender_only_query(query) or _row_has_person_box(dbg, row):
                row.debug["human_semantic_mode"] = True
                row.debug["human_semantic_only"] = True
                row.debug["human_semantic_score"] = round(float(own), 4)
                row.debug["gender_visual_score"] = round(float(own), 4)
        stats["tiers"][tier] = stats["tiers"].get(tier, 0) + 1
        stats["evidence"][evidence] = stats["evidence"].get(evidence, 0) + 1
        kept.append(row)
    if gender_query:
        kept.sort(key=lambda r: (
            int((r.debug or {}).get("uvi_rank", 5)),
            -float((r.debug or {}).get("gender_visual_score", 0.0) or 0.0),
            -float(r.score or 0),
        ))
    else:
        kept.sort(key=lambda r: (int((r.debug or {}).get("uvi_rank", 5)), -float(r.score or 0)))
    if query.leaf and not gender_query and not is_textile_discovery_query(query):
        kept = kept[:24]
    elif gender_query:
        # Gender search is a discovery query, not a fine-grained textile leaf.
        # Keep enough candidates for the UI to show useful figures/models.
        kept = kept[: (gender_cap or 120)]
    return kept, stats
