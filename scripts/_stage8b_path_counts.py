import sqlite3

c = sqlite3.connect("data/patterns.db")
for label, sql in [
    ("pips", "SELECT COUNT(*) FROM files WHERE lower(path) LIKE '%pips%' OR lower(filename) LIKE '%pips%'"),
    ("dudak/lip", "SELECT COUNT(*) FROM files WHERE lower(path) LIKE '%dudak%' OR lower(filename) LIKE '%dudak%' OR lower(path) LIKE '%lip%' OR lower(filename) LIKE '%lip%'"),
    ("puantiye/polka", "SELECT COUNT(*) FROM files WHERE lower(path) LIKE '%puantiye%' OR lower(path) LIKE '%polka%' OR lower(filename) LIKE '%puantiye%' OR lower(filename) LIKE '%polka%'"),
]:
    print(label, c.execute(sql).fetchone()[0])
