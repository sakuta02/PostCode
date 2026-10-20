from __future__ import annotations

CATALOG: dict[str, tuple[int, str]] = {
    "UNSUPPORTED_MEDIA_TYPE": (415, "Ожидается Content-Type: application/xml"),
    "PAYLOAD_TOO_LARGE":      (413, "Тело запроса превышает допустимый размер"),
    "XML_MALFORMED":          (400, "Тело запроса не является корректным XML"),
    "XML_DTD_FORBIDDEN":      (400, "DOCTYPE и объявления сущностей запрещены"),
    "XML_INVALID":            (400, "XML не соответствует схеме обмена"),
    "BASE64_INVALID":         (400, "Изображение не удалось декодировать из base64"),
    "INVALID_ARGUMENT":       (400, "Недопустимое значение параметра"),
    "UNAUTHORIZED":           (401, "Нужен ключ доступа в заголовке X-Api-Key"),
    "IMAGE_EMPTY":            (422, "Передано пустое изображение"),
    "IMAGE_UNSUPPORTED_FORMAT": (415, "Формат изображения не поддерживается"),
    "IMAGE_CORRUPTED":        (422, "Изображение повреждено и не может быть прочитано"),
    "IMAGE_TOO_SMALL":        (422, "Изображение слишком маленькое для оценки"),
    "IMAGE_TOO_LARGE":        (413, "Изображение слишком большое"),
    "SHIPMENT_NOT_FOUND":     (404, "Перевозка не найдена"),
    "ESTIMATE_NOT_FOUND":     (404, "Оценка не найдена"),
    "ESTIMATE_SHIPMENT_MISMATCH": (409, "Оценка относится к другой перевозке"),
    "MODEL_NOT_FOUND":        (404, "Версия модели не найдена"),
    "JOB_NOT_FOUND":          (404, "Задача не найдена"),
    "MODEL_UNAVAILABLE":      (503, "Модель для запрошенного режима не загружена"),
    "RETRAIN_BUSY":           (409, "Дообучение уже выполняется"),
    "RETRAIN_NO_FEEDBACK":    (409, "Недостаточно подтверждённых оценок для дообучения"),
    "RETRAIN_REQUIRES_GPU":   (409, "Дообучение на CPU займёт часы; нужен GPU или AllowCpu=true"),
    "INTERNAL_ERROR":         (500, "Внутренняя ошибка сервиса"),
}


class ApiError(Exception):
    def __init__(self, code: str, message: str | None = None, details: str | None = None):
        if code not in CATALOG:
            code, details = "INTERNAL_ERROR", f"неизвестный код ошибки: {code}"
        self.code = code
        self.status, default = CATALOG[code]
        self.message = message or default
        self.details = details
        super().__init__(f"{code}: {self.message}")
