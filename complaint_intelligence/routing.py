"""Auditable department routing rules."""

from .constants import DEPARTMENT_MAP


def route_department(topic: str, urgency: str = "medium") -> str:
    department = DEPARTMENT_MAP.get(topic, "Customer Experience")
    return f"{department} — Priority queue" if urgency == "high" else department

