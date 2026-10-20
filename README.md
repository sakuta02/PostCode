# Оценка загрузки грузового кузова по фото

Хакатон Почтатех. Сервис принимает фото грузового отсека и номер перевозки, возвращает загрузку в процентах через XML API, хранит историю по перевозке и дообучается


## Общее описание

1. **Данные.** 716 обучающих и 307 тестовых фото, метка - загрузка 0-100. Распределение таргета: бимодальное с преимущественным значением, кратным пяти
2. **Разметка.** Полигоны пола, кузова и груза в CVAT, предразметка моделью YOLO11-seg.
3. **Модель.** Backbone DINOv2 ViT-B/14, дообучаются последние 4 из 12 блоков. Фото дополняется до квадрата и ужимается до 392x392
4. **Две головы в мультитаске.** Attention-pooling по патч-токенам даёт распределение по 21 бину (0, 5, ..., 100), вторая голова сегментирует патчи на фон, кузов, пол и груз
5. **Декодирование.** Медиана распределения, затем округление к сетке в 5 п.п.
6. **Ансамбль.** 10 моделей (5 фолдов x 2 обучения) в режиме accurate, одна модель на всех данных в режиме fast
7. **Стек.** torch, DINOv2, FastAPI, lxml, pandas, SQLite, Docker


## Метрики

**Public-лидерборд:** MAE 4.6863, 2-е место.

## XML API

Схема обмена schema/loadest_api.xsd. Запросы проверяются по схеме, ошибки тоже возвращаются в XML

| Метод | Путь | Назначение |
|---|---|---|
| POST | /api/v1/estimates | Оценка, фото в base64 внутри XML |
| POST | /api/v1/estimates/upload | Оценка, фото файлом (multipart) |
| GET | /api/v1/shipments/{id} | Сохранённые оценки по перевозке |
| POST | /api/v1/feedback | Подтверждённая загрузка для дообучения |
| POST | /api/v1/retrain | Запуск дообучения |
| GET | /api/v1/jobs/{id} | Статус дообучения |
| GET | /api/v1/models | Версии модели |
| POST | /api/v1/models/{version}/activate | Переключение версии |
| GET | /api/v1/metrics | Фактическая точность по подтверждённым оценкам |
| GET | /api/v1/health | Состояние сервиса |

Дообучение: retrain обучает новую версию, она сравнивается с текущей на контрольных фото и включается, только если не хуже сверх допуска. Управляющие методы защищены ключом LOADEST_API_KEY в заголовке X-Api-Key


## Запуск

Веса (3.9 ГБ) лежат на [Google Drive](https://drive.google.com/drive/folders/1_oldPZE8pzKlcbABOlf9OrnqGdMCLFNN?usp=sharing), в репозитории описания с контрольными суммами:

```
models/
├── ensemble/   mt_fold0-4.pt, mt200_fold0-4.pt
└── fast/       fast_model.pt
```

```bash
# сервис в Docker
docker compose up --build
curl http://localhost:8000/api/v1/health

# или без Docker
pip install -r requirements.txt
export PYTHONPATH=src LOADEST_MODELS_DIR=models
uvicorn app:app --port 8000

curl -X POST http://localhost:8000/api/v1/estimates/upload \
     -F shipment_id=RP-2026-000123 -F file=@photo.jpg
```

Обучение на всех данных (нужен GPU), данные в data/train/{images/*.jpg, train.csv}. Гиперпараметры в src/config.py, любой переопределяется через --set ключ=значение:

```bash
PYTHONPATH=src python -m train --set train_dir=data/train
```

## Участники команды "Башкортостан":

1. https://github.com/sakuta02
2. ...
3. ...
4. ...