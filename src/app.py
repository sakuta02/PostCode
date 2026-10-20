from __future__ import annotations

import base64
import hmac
import logging
import os
import re
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

import images
import quality
import retrain
import xml_io
from errors import ApiError
from settings import Settings
from storage import Storage

VERSION = "1.0.0"
log = logging.getLogger("loadest")
XML = "application/xml"
SHIPMENT_RE = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


def xml(el, status: int = 200) -> Response:
    return Response(xml_io.to_bytes(el), status_code=status, media_type=XML)


class BodyLimit:
    def __init__(self, app, limit: int, limit_mb: float):
        self.app, self.limit, self.limit_mb = app, limit, limit_mb

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            for k, v in scope.get("headers", ()):
                if k == b"content-length" and v.isdigit() and int(v) > self.limit:
                    resp = xml(xml_io.error_response(
                        "PAYLOAD_TOO_LARGE", "Тело запроса превышает допустимый размер",
                        f"лимит {self.limit_mb:g} МБ"), 413)
                    await resp(scope, receive, send)
                    return
        await self.app(scope, receive, send)


def save_photo(img, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    img.save(tmp, format="JPEG", quality=95)
    os.replace(tmp, path)


def create_app(settings: Settings | None = None) -> FastAPI:
    from inference import TorchPredictor

    s = settings or Settings()
    s.ensure_dirs()
    st = Storage(s.db_path)
    pr = TorchPredictor(s)

    fast = Path(s.fast_path)
    if fast.exists():
        st.register_model(fast.stem, str(fast), "base", {"source": "initial"},
                          activate=st.active_model() is None)
    active = st.active_model()
    if active and Path(active["path"]).exists() and active["version"] != pr.fast_version:
        try:
            pr.set_fast(Path(active["path"]), active["version"])
        except ApiError as e:
            log.warning("активная версия %s не загрузилась (%s) — работает базовая модель",
                        active["version"], e.details)
            if fast.exists():
                st.activate_model(fast.stem)

    app = FastAPI(title="loadest — оценка загрузки кузова", version=VERSION,
                  description="XML API. Схема обмена: GET /api/v1/schema")
    app.state.settings, app.state.storage, app.state.predictor = s, st, pr
    body_limit = int(s.max_body_mb * 1024 * 1024)
    app.add_middleware(BodyLimit, limit=body_limit, limit_mb=s.max_body_mb)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return xml(xml_io.error_response(e.code, e.message, e.details), e.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        details = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        return xml(xml_io.error_response("INVALID_ARGUMENT", "Недопустимые параметры запроса",
                                         details), 400)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, e: Exception):
        log.error("необработанная ошибка: %s\n%s", e, traceback.format_exc())
        return xml(xml_io.error_response("INTERNAL_ERROR", "Внутренняя ошибка сервиса"), 500)

    def require_xml(request: Request) -> None:
        ctype = request.headers.get("content-type", "")
        if "xml" not in ctype.lower():
            raise ApiError("UNSUPPORTED_MEDIA_TYPE", details=f"получено: {ctype or 'нет'}")

    async def read_body(request: Request) -> bytes:
        too_large = ApiError("PAYLOAD_TOO_LARGE", details=f"лимит {s.max_body_mb:g} МБ")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > body_limit:
            raise too_large
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > body_limit:
                raise too_large
            chunks.append(chunk)
        return b"".join(chunks)

    async def read_xml(request: Request, root: str):
        require_xml(request)
        body = await read_body(request)
        return await run_in_threadpool(xml_io.parse, body, root, s.schema_path)

    def require_key(request: Request) -> None:
        if not s.api_key:
            return
        given = request.headers.get("x-api-key", "").encode()
        if not hmac.compare_digest(given, s.api_key.encode()):
            raise ApiError("UNAUTHORIZED")

    def check_shipment(shipment_id: str) -> str:
        if not SHIPMENT_RE.match(shipment_id or ""):
            raise ApiError("INVALID_ARGUMENT", "Недопустимый номер перевозки",
                           details="латиница, цифры, «.», «-», «_»; 1–64 символа")
        return shipment_id

    def check_mode(mode: str) -> str:
        if mode not in ("fast", "accurate"):
            raise ApiError("INVALID_ARGUMENT", f"Неизвестный режим {mode}",
                           details="допустимо: fast, accurate")
        return mode

    def b64(png: bytes | None) -> str | None:
        return base64.b64encode(png).decode() if png else None

    def estimate(shipment_id: str, mode: str, return_mask: bool, data: bytes) -> Response:
        t0 = time.perf_counter()
        img, info = images.load_image(data, s)
        sha = info["sha256"]
        photo_quality = quality.assess(img)

        if s.dedup:
            dup = st.find_duplicate(shipment_id, sha, mode, pr.version_for(mode))
            if dup is not None:
                mask = b64(pr.predict(img, mode, True).mask_png) if return_mask else None
                return xml(xml_io.estimate_response(dup, st.summary(shipment_id),
                                                    s.interval_level, mask, duplicate=True,
                                                    quality=photo_quality), 200)

        pred = pr.predict(img, mode, return_mask)
        ms = int((time.perf_counter() - t0) * 1000)

        path = s.images_dir / sha[:2] / f"{sha}.jpg"
        save_photo(img, path)

        est = st.add_estimate(shipment_id, pred, info, str(path), ms)
        return xml(xml_io.estimate_response(est, st.summary(shipment_id),
                                            s.interval_level, b64(pred.mask_png),
                                            quality=photo_quality), 201)

    @app.post("/api/v1/estimates", summary="Оценка загрузки: фото в base64 внутри XML")
    async def post_estimate(request: Request):
        root = await read_xml(request, "EstimateRequest")

        def work() -> Response:
            shipment = check_shipment(xml_io.text(root, "ShipmentId"))
            mode = check_mode(xml_io.text(root, "Mode") or s.default_mode)
            data = images.decode_base64(xml_io.text(root, "Image"))
            return estimate(shipment, mode, xml_io.flag(root, "ReturnMask"), data)

        return await run_in_threadpool(work)

    @app.post("/api/v1/estimates/upload", summary="Оценка загрузки: фото файлом (multipart)")
    async def post_estimate_upload(shipment_id: str = Form(...), mode: str | None = Form(None),
                                   return_mask: bool = Form(False),
                                   file: UploadFile = File(...)):
        check_shipment(shipment_id)
        mode = check_mode(mode or s.default_mode)
        limit = int(s.max_image_mb * 1024 * 1024)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise ApiError("IMAGE_TOO_LARGE", details=f"лимит {s.max_image_mb:g} МБ")
        return await run_in_threadpool(estimate, shipment_id, mode, return_mask, data)

    @app.get("/api/v1/shipments/{shipment_id}", summary="Сохранённые оценки по номеру перевозки")
    def get_shipment(shipment_id: str):
        ship = st.shipment(check_shipment(shipment_id))
        if ship is None:
            raise ApiError("SHIPMENT_NOT_FOUND", details=shipment_id)
        return xml(xml_io.shipment_response(ship, s.interval_level))

    @app.post("/api/v1/feedback", summary="Подтверждённая загрузка для дообучения")
    async def post_feedback(request: Request):
        root = await read_xml(request, "FeedbackRequest")
        shipment = check_shipment(xml_io.text(root, "ShipmentId"))
        confirmed = float(xml_io.text(root, "ConfirmedLoadPct"))
        est_id = xml_io.text(root, "EstimateId")
        if est_id:
            est = st.get_estimate(int(est_id))
            if est is None:
                raise ApiError("ESTIMATE_NOT_FOUND", details=est_id)
            if est["shipment_id"] != shipment:
                raise ApiError("ESTIMATE_SHIPMENT_MISMATCH",
                               details=f"оценка {est_id} относится к {est['shipment_id']}")
        else:
            hist = st.history(shipment)
            if not hist:
                raise ApiError("SHIPMENT_NOT_FOUND", details=shipment)
            est = hist[-1]
        fb = st.add_feedback(shipment, est, confirmed, xml_io.text(root, "Comment"))
        return xml(xml_io.feedback_response(fb, est, len(st.pending_feedback())), 201)

    @app.get("/api/v1/models", summary="Реестр версий модели быстрого режима")
    def get_models():
        return xml(xml_io.models_response(st.models()))

    @app.post("/api/v1/models/{version}/activate", summary="Переключить быстрый режим на версию")
    def activate_model(version: str, request: Request):
        require_key(request)
        row = st.get_model(version)
        if row is None:
            raise ApiError("MODEL_NOT_FOUND", details=version)
        pr.set_fast(Path(row["path"]), version)
        st.activate_model(version)
        return xml(xml_io.models_response(st.models()))

    @app.post("/api/v1/retrain", status_code=202, summary="Запустить дообучение")
    async def post_retrain(request: Request):
        require_key(request)
        body = await read_body(request)
        opts = {"DryRun": False, "Activate": False, "AllowCpu": False}
        min_fb = 1
        if body.strip():
            require_xml(request)
            root = await run_in_threadpool(xml_io.parse, body, "RetrainRequest", s.schema_path)
            opts = {k: xml_io.flag(root, k) for k in opts}
            min_fb = int(xml_io.text(root, "MinFeedback", "1"))
        if st.running_job("retrain"):
            raise ApiError("RETRAIN_BUSY")
        pending = len(st.pending_feedback())
        if not opts["DryRun"] and pending < min_fb:
            raise ApiError("RETRAIN_NO_FEEDBACK",
                           details=f"подтверждено {pending}, требуется {min_fb}")
        if not opts["DryRun"] and pr.device == "cpu" and not opts["AllowCpu"]:
            raise ApiError("RETRAIN_REQUIRES_GPU")
        job = st.create_job("retrain")
        args = (s, st, job["id"], opts["DryRun"], opts["Activate"], pr)
        if s.retrain_async:
            threading.Thread(target=retrain.run, args=args, daemon=True).start()
        else:
            retrain.run(*args)
        return xml(xml_io.job_response(st.job(job["id"])), 202)

    @app.get("/api/v1/jobs/{job_id}", summary="Статус задачи дообучения")
    def get_job(job_id: int):
        job = st.job(job_id)
        if job is None:
            raise ApiError("JOB_NOT_FOUND", details=str(job_id))
        return xml(xml_io.job_response(job))

    @app.get("/api/v1/metrics", summary="Фактическая точность по подтверждённым оценкам")
    def get_metrics():
        return xml(xml_io.metrics_response(st.accuracy()))

    @app.get("/api/v1/health", summary="Состояние сервиса и доступность режимов")
    def health():
        return xml(xml_io.health_response(pr.info(), VERSION))

    @app.get("/api/v1/schema", summary="XSD-схема обмена")
    def get_schema():
        return FileResponse(s.schema_path, media_type=XML, filename="loadest_api.xsd")

    return app


def _lazy_app():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    application = create_app()
    try:
        application.state.predictor.warmup()
    except Exception as e:
        log.warning("прогрев не удался (сервис работает в режиме degraded): %s", e)
    return application


def __getattr__(name):
    if name == "app":
        globals()["app"] = _lazy_app()
        return globals()["app"]
    raise AttributeError(name)
