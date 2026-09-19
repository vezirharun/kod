import tempfile, os
from core.category_memory import register_root_category, register_category, resolve_dynamic_category, category_suggestions
from core.concept_registry import learn, find, concepts

def test_dynamic_category_fuzzy_and_roots():
    with tempfile.TemporaryDirectory() as d:
        db=os.path.join(d,"x.db")
        assert register_root_category(db,"Sıçramış Boya",aliases=["boya sıçraması"])
        assert resolve_dynamic_category(db,"sıçramış boya")=="Sıçramış Boya"
        assert category_suggestions(db,"sıçramış boya")[0][0]=="Sıçramış Boya"
        assert register_category(db,"Soyut","Çatlak",aliases=["çatlak yüzey"])
        assert resolve_dynamic_category(db,"çatlak")=="Soyut/Çatlak"

def test_concept_learning_positive_negative():
    with tempfile.TemporaryDirectory() as d:
        db=os.path.join(d,"x.db")
        cid=learn(db,"Sıçramış Boya",file_id=1,file_path="a.png",concept_type="attribute")
        assert cid
        assert any(x["canonical"]=="Sıçramış Boya" for x in concepts(db))
        assert find(db,"sıçramış boya")[0]["canonical"]=="Sıçramış Boya"
        from core.concept_registry import add_example
        add_example(db,cid,file_id=2,file_path="b.png",role="negative")
        assert len(concepts(db))==1
