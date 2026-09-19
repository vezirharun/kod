"""V12.4.13 evidence explanation formatter."""
def explain_decision(decision):
    parts=[f"Karar: {decision.label or 'yok'}",
           f"Güven: {decision.confidence:.1%}",
           "Durum: kabul" if decision.accepted else "Durum: kabul edilmedi"]
    if decision.contradictory: parts.append("Çelişen güçlü kanıt var.")
    if decision.missing: parts.append("Eksik: "+", ".join(decision.missing))
    if decision.reason: parts.append(decision.reason)
    return " | ".join(parts)
