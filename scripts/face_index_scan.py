#!/usr/bin/env python3
"""Incremental face index scanner.

Usage examples:
  python scripts/face_index_scan.py --limit 500
  python scripts/face_index_scan.py --after-id 50000 --limit 5000
  python scripts/face_index_scan.py --reconcile
"""
from __future__ import annotations
import argparse
from core.settings import AppSettings
from core.face_scanner import FaceIndexScanner

p=argparse.ArgumentParser()
p.add_argument('--limit',type=int,default=0)
p.add_argument('--after-id',type=int,default=0)
p.add_argument('--reconcile',action='store_true')
args=p.parse_args()
s=AppSettings.load()
scanner=FaceIndexScanner(s.db_path,s.face_db_path,threshold=s.face_identity_threshold,min_margin=s.face_identity_min_margin)
if args.reconcile:
    print({'removed':scanner.reconcile(),'stats':scanner.store.stats()})
else:
    print(scanner.scan(after_id=args.after_id,limit=args.limit))
