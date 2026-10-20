from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from lxml import etree

from errors import ApiError

_DTD = re.compile(rb"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                          huge_tree=True,
                          remove_blank_text=True)


@lru_cache(maxsize=4)
def schema(path: str) -> etree.XMLSchema:
    return etree.XMLSchema(etree.parse(path))


def parse(body: bytes, root_name: str, schema_path: Path) -> etree._Element:
    if not body or not body.strip():
        raise ApiError("XML_MALFORMED", "Пустое тело запроса")
    if _DTD.search(body):
        raise ApiError("XML_DTD_FORBIDDEN")
    try:
        root = etree.fromstring(body, _PARSER)
    except etree.XMLSyntaxError as e:
        raise ApiError("XML_MALFORMED", details=str(e)) from None
    if root.tag != root_name:
        raise ApiError("XML_INVALID", f"Ожидался корневой элемент <{root_name}>",
                       details=f"получен <{root.tag}>")
    xsd = schema(str(schema_path))
    if not xsd.validate(root):
        errs = "; ".join(f"строка {e.line}: {e.message}" for e in list(xsd.error_log)[:3])
        raise ApiError("XML_INVALID", details=errs)
    return root


def text(root: etree._Element, tag: str, default: str | None = None) -> str | None:
    el = root.find(tag)
    return el.text.strip() if el is not None and el.text is not None else default


def flag(root: etree._Element, tag: str, default: bool = False) -> bool:
    v = text(root, tag)
    return default if v is None else v.lower() in ("true", "1")


def to_bytes(el: etree._Element) -> bytes:
    return etree.tostring(el, xml_declaration=True, encoding="UTF-8", pretty_print=True)


def num(v) -> str:
    v = float(v)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def _sub(parent, tag, value=None, **attrs):
    el = etree.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})
    if value is not None:
        el.text = str(value)
    return el


def _estimate_fields(parent, e: dict, level: float) -> None:
    _sub(parent, "EstimateId", e["id"])
    _sub(parent, "ShipmentId", e["shipment_id"])
    _sub(parent, "LoadPct", num(e["load_pct"]))
    _sub(parent, "Interval", low=num(e["interval_low"]), high=num(e["interval_high"]),
         level=f"{level:g}")
    _sub(parent, "Confidence", e["confidence"])
    if e.get("occupied_ratio") is not None:
        _sub(parent, "OccupiedRatioByMask", num(e["occupied_ratio"]))
    _sub(parent, "Model", version=e["model_version"], mode=e["mode"], models=e["n_models"])
    _sub(parent, "ProcessingMs", int(e["processing_ms"]))
    _sub(parent, "CreatedAt", e["created_at"])


def _summary(parent, tag: str, s: dict):
    attrs = dict(estimates=s["count"], aggregateLoadPct=num(s["aggregate"]), method=s["method"])
    if "photos" in s:
        attrs["photos"] = s["photos"]
    return _sub(parent, tag, **attrs)


def _quality(parent, q: dict):
    el = _sub(parent, "PhotoQuality", status=q["status"], width=q["width"], height=q["height"],
              brightness=f"{q['brightness']:g}", darkShare=f"{q['dark_share']:g}",
              brightShare=f"{q['bright_share']:g}", sharpness=f"{q['sharpness']:g}")
    for code, message in q["issues"]:
        _sub(el, "Issue", message, code=code)
    return el


def estimate_response(e: dict, summary: dict, level: float, mask_b64: str | None = None,
                      duplicate: bool = False, quality: dict | None = None):
    root = etree.Element("EstimateResponse")
    _estimate_fields(root, e, level)
    _summary(root, "Shipment", summary)
    if quality is not None:
        _quality(root, quality)
    if duplicate:
        _sub(root, "Duplicate", "true")
    if mask_b64:
        _sub(root, "Mask", mask_b64, encoding="base64", format="png")
    return root


def shipment_response(ship: dict, level: float):
    root = etree.Element("ShipmentResponse")
    _sub(root, "ShipmentId", ship["shipment_id"])
    _summary(root, "Aggregate", ship["summary"])
    _estimate_fields(etree.SubElement(root, "Latest"), ship["history"][-1], level)
    hist = etree.SubElement(root, "History")
    for e in ship["history"]:
        _estimate_fields(etree.SubElement(hist, "Estimate"), e, level)
    if ship.get("feedback"):
        fb = ship["feedback"]
        _sub(root, "Feedback", confirmedLoadPct=num(fb["confirmed_load_pct"]),
             createdAt=fb["created_at"])
    return root


def feedback_response(fb: dict, est: dict, pending: int):
    root = etree.Element("FeedbackResponse")
    _sub(root, "FeedbackId", fb["id"])
    _sub(root, "ShipmentId", fb["shipment_id"])
    _sub(root, "EstimateId", est["id"])
    _sub(root, "PredictedLoadPct", num(est["load_pct"]))
    _sub(root, "ConfirmedLoadPct", num(fb["confirmed_load_pct"]))
    _sub(root, "PendingForRetrain", pending)
    _sub(root, "CreatedAt", fb["created_at"])
    return root


def job_response(job: dict):
    root = etree.Element("JobResponse")
    _sub(root, "JobId", job["id"])
    _sub(root, "Kind", job["kind"])
    _sub(root, "Status", job["status"])
    _sub(root, "CreatedAt", job["created_at"])
    if job.get("finished_at"):
        _sub(root, "FinishedAt", job["finished_at"])
    if job.get("result"):
        _sub(root, "Result", job["result"])
    if job.get("log"):
        _sub(root, "Log", job["log"][-4000:])
    return root


def models_response(models: list[dict]):
    root = etree.Element("ModelsResponse")
    for m in models:
        attrs = dict(version=m["version"], kind=m["kind"],
                     active="true" if m["is_active"] else "false", createdAt=m["created_at"])
        if m.get("meta"):
            attrs["meta"] = m["meta"]
        _sub(root, "Model", **attrs)
    return root


def _accuracy_attrs(m: dict) -> dict:
    attrs = {"confirmed": m["n"]}
    if m["n"]:
        attrs.update(mae=f"{m['mae']:.2f}", within10pp=f"{m['within10']:.1f}",
                     bias=f"{m['bias']:.2f}")
    return attrs


def metrics_response(acc: dict):
    root = etree.Element("MetricsResponse")
    _sub(root, "Overall", **_accuracy_attrs(acc["overall"]))
    for m in acc["by_model"]:
        _sub(root, "ByModel", version=m["version"], mode=m["mode"], **_accuracy_attrs(m))
    return root


def health_response(info: dict, version: str):
    root = etree.Element("HealthResponse")
    _sub(root, "Status", info["status"])
    _sub(root, "Device", info["device"])
    for name, m in info["modes"].items():
        _sub(root, "Mode", name=name, available="true" if m["available"] else "false",
             models=m["models"])
    _sub(root, "ActiveModel", info["active_model"])
    _sub(root, "Version", version)
    return root


def error_response(code: str, message: str, details: str | None = None):
    root = etree.Element("Error")
    _sub(root, "Code", code)
    _sub(root, "Message", message)
    if details:
        _sub(root, "Details", details)
    return root
