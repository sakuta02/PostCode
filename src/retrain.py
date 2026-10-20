from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import images as imgio
from settings import PROJECT_ROOT, Settings
from storage import Storage

EPOCH = 17


def _link(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy(src, dst)


def train_paths(train_dir: Path) -> tuple[Path, Path]:
    labels, images = Path(train_dir) / "train.csv", Path(train_dir) / "images"
    if not (labels.exists() and images.is_dir()):
        raise FileNotFoundError(f"нет исходных данных в {train_dir}: нужны images/ и train.csv")
    return labels, images


def _read_labels(labels: Path) -> pd.DataFrame:
    df = pd.read_csv(labels)
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.jpg$", "", regex=True)
    return df


def build_dataset(s: Settings, st: Storage, out: Path) -> dict:
    labels, images = train_paths(s.train_dir)

    img_out = out / "images"
    if out.exists():
        shutil.rmtree(out)
    img_out.mkdir(parents=True)

    base = _read_labels(labels)
    kept = []
    for image_id in base.image_id:
        src = images / f"{image_id}.jpg"
        if src.exists():
            _link(src, img_out / src.name)
            kept.append(image_id)
    base = base[base.image_id.isin(kept)]

    fb_rows, fb_ids, new_ids, skipped = [], [], [], 0
    for fb in st.confirmed_feedback():
        src = Path(fb["image_path"] or "")
        if not src.exists():
            skipped += 1
            continue
        image_id = f"fb_{fb['id']:06d}"
        Image.open(src).convert("RGB").save(img_out / f"{image_id}.jpg", quality=95)
        fb_rows.append({"image_id": image_id, "load_pct": fb["confirmed_load_pct"]})
        fb_ids.append(fb["id"])
        if fb["used_in_version"] is None:
            new_ids.append(fb["id"])

    fb_df = pd.DataFrame(fb_rows, columns=["image_id", "load_pct"])
    pd.concat([base[["image_id", "load_pct"]], fb_df]).to_csv(out / "train.csv", index=False)

    return {"base_images": len(base), "feedback_images": len(fb_rows),
            "feedback_new": len(new_ids), "feedback_skipped": skipped,
            "total": len(base) + len(fb_rows),
            "feedback_ids": fb_ids, "new_feedback_ids": new_ids, "dataset": str(out)}


def control_set(s: Settings, n: int) -> list[tuple[Path, float]]:
    labels, images = train_paths(s.train_dir)
    df = _read_labels(labels)
    df = df[[(images / f"{i}.jpg").exists() for i in df.image_id]]
    df = df.sort_values(["load_pct", "image_id"]).reset_index(drop=True)
    if df.empty or n <= 0:
        return []
    idx = np.unique(np.linspace(0, len(df) - 1, num=min(n, len(df))).round().astype(int))
    return [(images / f"{df.image_id[i]}.jpg", float(df.load_pct[i])) for i in idx]


def _mae(predictor, ckpt: Path, items: list[tuple[Path, float]], s: Settings) -> float:
    errs = []
    for path, y in items:
        img, _ = imgio.load_image(path.read_bytes(), s)
        errs.append(abs(predictor.predict_paths(img, [ckpt]) - y))
    return round(float(np.mean(errs)), 2)


def quality_gate(s: Settings, predictor, new_ckpt: Path) -> dict:
    items = control_set(s, s.retrain_control_n)
    if not items:
        return {"gate": "skipped", "reason": "нет контрольных фото"}
    current = Path(predictor.fast_path)
    cur_mae = _mae(predictor, current, items, s)
    new_mae = _mae(predictor, new_ckpt, items, s)
    passed = new_mae <= cur_mae + s.retrain_gate_tol
    return {"gate": "passed" if passed else "failed", "control_images": len(items),
            "control_mae_current": cur_mae, "control_mae_new": new_mae,
            "current_version": predictor.fast_version, "tolerance": s.retrain_gate_tol}


def _train_env() -> dict:
    env = dict(os.environ)
    env.pop("LOADEST_SKIP_PRETRAINED", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(PROJECT_ROOT / "src"), env.get("PYTHONPATH")]))
    return env


def run(s: Settings, st: Storage, job_id: int, dry_run: bool, activate: bool,
        predictor) -> None:
    job_dir = s.jobs_dir / f"retrain_{job_id:04d}"
    log: list[str] = []
    try:
        st.update_job(job_id, "running")
        stats = build_dataset(s, st, job_dir / "data")
        log.append(f"набор: {stats['base_images']} исходных + "
                   f"{stats['feedback_images']} подтверждённых "
                   f"(новых {stats['feedback_new']}) = {stats['total']}")
        public = {k: v for k, v in stats.items() if not k.endswith("_ids")}
        if dry_run:
            st.update_job(job_id, "done", {**public, "dry_run": True}, "\n".join(log))
            return

        tag = f"v{job_id}"
        cmd = [sys.executable, "-m", "train",
               "--set", f"train_dir={job_dir / 'data'}",
               "--set", f"out_dir={job_dir / 'out'}",
               "--set", f"run_tag={tag}",
               "--epoch", str(EPOCH)]
        if s.coco_json and Path(s.coco_json).exists():
            cmd += ["--set", f"coco_json={s.coco_json}"]
        else:
            log.append(f"ВНИМАНИЕ: нет разметки масок ({s.coco_json or 'не задана'}) — "
                       "новая версия не будет строить маску груза")
            cmd += ["--set", "coco_json=", "--set", "lambda_seg=0", "--set", "lambda_seg_final=0"]
        log.append("$ " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
                              env=_train_env())
        log.append(proc.stdout[-3000:])
        if proc.returncode:
            log.append(proc.stderr[-3000:])
            raise RuntimeError(f"обучение завершилось с кодом {proc.returncode}")

        ckpt = job_dir / "out" / "checkpoints" / f"{tag}_full_ep{EPOCH}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"не найден чекпоинт {ckpt}")
        dst = s.trained_models_dir / ckpt.name
        shutil.copy(ckpt, dst)

        gate = quality_gate(s, predictor, dst)
        log.append("контроль: " + json.dumps(gate, ensure_ascii=False))
        activated = activate and gate["gate"] != "failed"
        if activate and not activated:
            log.append("версия не активирована: хуже текущей сверх допуска")

        meta = {k: stats[k] for k in ("base_images", "feedback_images", "feedback_new", "total")}
        meta.update({k: v for k, v in gate.items() if k != "reason"})
        st.register_model(tag, str(dst), "retrained", meta, activate=False)
        st.mark_feedback_used(stats["new_feedback_ids"], tag)
        if activated:
            predictor.set_fast(dst, tag)
            st.activate_model(tag)
        else:
            predictor.evict(dst)
        st.update_job(job_id, "done", {**meta, "version": tag, "activated": activated},
                      "\n".join(log))
    except Exception as e:
        log.append(f"ОШИБКА: {e}")
        st.update_job(job_id, "failed", {"error": str(e)}, "\n".join(log))
