from pathlib import Path
from PIL import Image
from core.db import Database
from core.index_v3 import IndexEngineV3, Mode
from core.index_ssot import count_lanes

def test_v3_light_completion_updates_legacy_counter(tmp_path: Path):
    db=Database(tmp_path/'db.sqlite')
    root=tmp_path/'src'; root.mkdir()
    with db.connect() as c:
        c.execute("INSERT INTO sources(name, root_path, is_active) VALUES (?,?,1)",('src',str(root)))
    for i in range(4): Image.new('RGB',(32,32)).save(root/f'{i}.jpg')
    eng=IndexEngineV3(db, job_db_path=tmp_path/'jobs.sqlite')
    eng.run(mode=Mode.FAST, sources=[{'id':1,'root_path':str(root)}], walk_disk=True)
    lanes=count_lanes(db,[1])
    assert lanes['total']==4
    assert lanes['light_pending']==0
    assert lanes['light_done']==4
