from core.production.health_monitor import queues_from_status


def test_index_active_is_claimed_not_ai_remaining():
    q = queues_from_status(
        {
            "queue_pending": 10182,
            "pending": 0,
            "claimed_jobs": 1,
            "processing": 1,
            "total": 11799,
            "pending_light_jobs": 0,
            "pending_heavy_jobs": 78000,
            "light_done": 11779,
            "light_failed": 20,
            "general_remaining": 10182,
            "general_completed": 1598,
            "failed_permanent_files": 20,
        }
    )
    assert q["index_active"] == 1
    assert q["INDEX_ACTIVE"] == 1
    assert q["workers"] == 1
    assert q["SOURCE_TOTAL"] == 11799
    assert q["LIGHT_PENDING"] == 0
    assert q["LIGHT_DONE"] == 11779
    assert q["LIGHT_FAILED"] == 20
    assert q["GENERAL_AI_PENDING"] == 10182
    assert q["GENERAL_AI_COMPLETED"] == 1598
    assert q["GENERAL_AI_PENDING"] != q["index_active"]
