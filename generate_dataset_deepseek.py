#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация датасета для CORA (см. system_one_multimodal_training.ipynb) через DeepSeek API.

Что делает
----------
1. Тянет КАРТИНКИ из открытых датасетов HuggingFace (стримингом, без полной загрузки):
     * computer-use / GUI — скриншоты реальных приложений и сайтов, где важно
       «какую кнопку нажать» (ScreenSpot, Click-100k, RICO-Screen2Words, WebSight, AndroidControl);
     * реальный мир — чеки и документы (SROIE), графики/дашборды (ChartQA), еда (Food-101),
       уличные/бытовые сцены (COCO), медицинский рентген.
2. Для каждой картинки собирает текстовые «улики» из аннотаций датасета
   (инструкция и bbox целевого элемента, подписи экрана, OCR-текст чека и company/date/total,
   вопрос-ответ по графику, класс объекта, список детекций...).
3. DeepSeek (учитель) пишет по этим уликам:
     * фазa A — реалистичный бизнес-контекст `state_text` + варианты ответа для
       «заземлённых» вопросов (например конкретная кнопка/элемент из аннотации + дистракторы)
       + 1-2 дополнительных вопроса, специфичных для картинки;
     * фаза B — self-consistency: K независимых сэмплов с температурой, каждый возвращает
       распределение вероятностей по УЖЕ зафиксированным вариантам (closed-set, как в CORA).
4. Вероятности усредняются по K сэмплам, сглаживаются, для заземлённых вопросов
   подмешивается априор one-hot из аннотации датасета (--gt-mix) — так модель учится
   на реальной разметке, а не только на мнении учителя.
5. Пишет train.jsonl / val.jsonl ровно в схеме ноутбука + meta.jsonl с provenance и
   сырыми голосами учителя (для аудита/отладки).

Формат строки (совпадает с notebook):
    {"id": "...", "state_text": "...", "image_path": "images/screenspot/xxx.jpg",
     "questions": [{"key": "...", "text": "...", "options": [...], "target_probs": [...]}]}

Зависимости
-----------
    pip install -U "datasets>=2.19" pillow requests
    # (для ноутбука-потребителя: transformers>=4.49 и т.д. — см. ячейку установки)

Переменные окружения
--------------------
    DEEPSEEK_API_KEY   — обязательно (если не --dry-run)
    DEEPSEEK_BASE_URL  — по умолчанию https://api.deepseek.com/v1
                         (OpenRouter/прокси: https://openrouter.ai/api/v1 и т.п.)
    HF_TOKEN           — для gated-датасетов (не обязательно)

Примеры запуска
---------------
    # небольшой смешанный датасет: 8 источников x 40 примеров, self-consistency K=5
    python generate_dataset_deepseek.py --per-source 40 --k 5 --workers 4 --out-dir data

    # только computer-use сценарии
    python generate_dataset_deepseek.py --family computer_use --per-source 150 --out-dir data

    # только реальный мир
    python generate_dataset_deepseek.py --family real_world --per-source 120 --out-dir data

    # дешёвая отладка пайплайна без API (state_text из аннотаций, probs равномерные)
    python generate_dataset_deepseek.py --per-source 5 --dry-run

    # учитель реально СМОТРИТ на картинку (нужна VL-модель на том же OpenAI-совместимом API)
    python generate_dataset_deepseek.py --use-vision --vision-model deepseek-vl2-chat --per-source 20

    # конкретные источники
    python generate_dataset_deepseek.py --sources screenspot,sroie,chartqa --per-source 60

Нюансы
------
* DeepSeek chat API текстовый: по умолчанию картинка учителю НЕ отправляется, он работает
  по метаданным датасета (инструкция/bbox/подписи/OCR). Поэтому state_text получается
  согласованным с изображением. Если есть VL-эндпоинт — включай --use-vision, качество заметно выше.
* Все источники обёрнуты в try/except: если датасет недоступен/переименован/с другой схемой,
  скрипт warns и продолжает с остальными. Список и схему колонок смотри в SOURCES ниже.
* Картинки уменьшаются до --max-image-side (по умолчанию 1024) и сохраняются в JPEG:
  датасет на несколько тысяч примеров остаётся в единицах ГБ.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:                      # нужен для сохранения/чтения картинок
    Image = None                         # type: ignore[assignment]

try:
    import requests
except ImportError:                      # нужен только для реальных вызовов API
    requests = None

try:
    from datasets import load_dataset
except Exception:  # datasets не установлен — работать можно только в --dry-run без картинок
    load_dataset = None


# --------------------------------------------------------------------------------------
# 1. Конфигурация по умолчанию
# --------------------------------------------------------------------------------------

DEFAULT_MODEL = "deepseek-v4-flash"            # текстовый учитель (поменяй на свой, если endpoint зовётся иначе)
DEFAULT_VISION_MODEL = ""                       # например "deepseek-vl2-chat" при --use-vision
DEFAULT_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

MAX_OPTIONS = 16                # = CFG["max_options"] в ноутбуке
MAX_STATE_CHARS = 1600          # state_text держим короче CFG["max_state_len"] токенов
DEFAULT_MAX_IMAGE_SIDE = 1024
DEFAULT_K = 5                   # self-consistency: сколько сэмплов учителя на пример
DEFAULT_LABEL_SMOOTHING = 0.02
DEFAULT_GT_MIX = 0.30           # доля априора из аннотации датасета в заземлённых вопросах
DEFAULT_TRAIN_RATIO = 0.85

LANG_NAMES = {"ru": "русском", "en": "английском"}


# --------------------------------------------------------------------------------------
# 2. Схемы вопросов (closed-set). Часть слотов — "заземлённые": их варианты
#    придумывает учитель из аннотации датасета (GT + дистракторы).
# --------------------------------------------------------------------------------------

@dataclass
class Question:
    key: str
    text: str
    options: Optional[List[str]] = None     # None => варианты придумывает учитель (grounded slot)
    gt_hint: str = ""                       # что считать правильным ответом (инструкция для учителя)
    grounded: bool = False                  # подмешивать ли one-hot из аннотации при агрегации
    n_options: int = 5                      # сколько вариантов просить у учителя для grounded-слота
    gt_index: Optional[int] = None          # индекс ground-truth варианта (ставится после фазы A)


def _priority_q() -> Question:
    return Question(
        key="priority",
        text="Какой приоритет должен получить этот кейс?",
        options=["low", "normal", "high", "urgent"],
    )


def _human_confirm_q() -> Question:
    return Question(
        key="needs_human_confirmation",
        text="Нужно ли подтверждение человека перед выполнением действия?",
        options=["нет, можно выполнить автоматически", "да, нужно подтверждение оператора"],
    )


SCHEMAS: Dict[str, List[Question]] = {
    # ---------------- computer use: десктоп/мобайл/веб, "какую кнопку нажать" ----------------
    "computer_use": [
        Question(
            key="next_action",
            text="Какое действие агенту выполнить следующим на этом экране?",
            options=None,
            gt_hint="действие и целевой элемент из поля GT_ACTION (инструкция/bbox датасета)",
            grounded=True,
            n_options=5,
        ),
        Question(
            key="target_element",
            text="На какой элемент интерфейса нужно нажать?",
            options=None,
            gt_hint="элемент, указанный в GT_ACTION/GT_ELEMENT (кнопка, поле, иконка, пункт меню)",
            grounded=True,
            n_options=6,
        ),
        Question(
            key="task_stage",
            text="На какой стадии пользовательской задачи находится этот экран?",
            options=["только начали", "середина сценария", "почти завершено", "задача завершена"],
        ),
        Question(
            key="action_reversibility",
            text="Насколько обратимо предложенное действие?",
            options=["полностью обратимо", "частично обратимо", "необратимо"],
        ),
        Question(
            key="wrong_action_risk",
            text="Какой риск, если агент нажмёт не туда?",
            options=["низкий", "средний", "высокий", "критический"],
        ),
        _human_confirm_q(),
    ],

    # ---------------- синтетические веб-страницы (WebSight) ----------------
    "web_ui": [
        Question(
            key="page_type",
            text="Что это за страница?",
            options=["лендинг продукта", "дашборд/аналитика", "форма или чекаут",
                     "статья/блог", "каталог товаров", "другое"],
        ),
        Question(
            key="primary_cta",
            text="Какое целевое действие пользователь, скорее всего, совершит на этой странице?",
            options=None,
            gt_hint="главный CTA/компонент, следующий из HTML или подписи страницы",
            grounded=True,
            n_options=5,
        ),
        Question(
            key="ux_issue_priority",
            text="Насколько срочно нужно править UX этой страницы?",
            options=["не требует правок", "низкий приоритет", "средний приоритет", "высокий приоритет"],
        ),
        Question(
            key="is_mobile_ready",
            text="Похоже, что вёрстка адаптирована под мобильные?",
            options=["да", "частично", "нет", "не определить по скриншоту"],
        ),
        _priority_q(),
    ],

    # ---------------- документы: чеки, счета, формы ----------------
    "documents": [
        Question(
            key="doc_type",
            text="Что это за документ?",
            options=["кассовый чек", "счёт-фактура", "накладная", "анкета/форма",
                     "договор", "другой документ"],
        ),
        Question(
            key="expense_category",
            text="К какой категории расходов отнести этот документ?",
            options=["продукты", "транспорт", "оборудование и ПО", "маркетинг",
                     "представительские", "коммунальные", "прочее"],
        ),
        Question(
            key="needs_manual_review",
            text="Нужна ли ручная проверка бухгалтером?",
            options=["нет, можно провести автоматически", "да, нужна проверка"],
        ),
        Question(
            key="fraud_risk",
            text="Какой риск аномалии или мошенничества по этому документу?",
            options=["низкий", "средний", "высокий"],
        ),
        _priority_q(),
    ],

    # ---------------- графики и дашборды ----------------
    "charts": [
        Question(
            key="trend",
            text="Какая динамика показана на графике?",
            options=["рост", "падение", "без изменений", "нестабильно/смешанно", "не определить"],
        ),
        Question(
            key="business_action",
            text="Какое действие по этому графику разумнее всего?",
            options=["эскалировать руководителю", "продолжать наблюдение",
                     "запросить дополнительные данные", "закрыть вопрос", "запустить проверку"],
        ),
        Question(
            key="data_confidence",
            text="Насколько можно доверять данным на графике?",
            options=["низкая уверенность", "средняя уверенность", "высокая уверенность"],
        ),
        Question(
            key="answer_grounded",
            text="Какой ответ на вопрос по графику верный?",
            options=None,
            gt_hint="правильный ответ из поля GT_ANSWER (аннотация ChartQA) + правдоподобные неверные ответы",
            grounded=True,
            n_options=4,
        ),
        _priority_q(),
    ],

    # ---------------- еда / HoReCa ----------------
    "food": [
        Question(
            key="dish_type",
            text="Какое блюдо на фото?",
            options=None,
            gt_hint="название блюда из поля GT_LABEL (метка класса датасета)",
            grounded=True,
            n_options=5,
        ),
        Question(
            key="course",
            text="К какому курсу отнести блюдо?",
            options=["завтрак", "основное блюдо", "десерт", "напиток", "закуска", "фастфуд"],
        ),
        Question(
            key="quality_risk",
            text="Есть ли на фото признаки проблемы с качеством подачи?",
            options=["нет проблем", "незначительные замечания", "явная проблема"],
        ),
        Question(
            key="complaint_severity",
            text="Если это фото пришло в жалобе клиента, какова её серьёзность?",
            options=["низкая", "средняя", "высокая", "это не жалоба"],
        ),
        _priority_q(),
    ],

    # ---------------- реальные сцены (COCO и подобные) ----------------
    "scenes": [
        Question(
            key="scene_type",
            text="Что за сцена на фотографии?",
            options=["улица и город", "природа", "помещение", "спорт и активность",
                     "животные", "транспорт", "еда", "другое"],
        ),
        Question(
            key="safety_risk",
            text="Есть ли на снимке признаки риска для людей или имущества?",
            options=["нет риска", "низкий риск", "средний риск", "высокий риск"],
        ),
        Question(
            key="next_step",
            text="Что сделать с этим материалом дальше?",
            options=["зафиксировать без действий", "передать в поддержку",
                     "эскалировать ответственному", "запросить дополнительные фото", "архивировать"],
        ),
        Question(
            key="objects_grounded",
            text="Какие объекты точно видны на снимке?",
            options=None,
            gt_hint="объекты из аннотации детекций/подписи (поле GT_FACTS)",
            grounded=True,
            n_options=4,
        ),
        _priority_q(),
    ],

    # ---------------- медицина ----------------
    "medical": [
        Question(
            key="finding_category",
            text="Что видно на снимке?",
            options=["норма", "подозрение на патологию", "явная патология", "снимок низкого качества"],
        ),
        Question(
            key="urgency",
            text="Насколько срочно нужно описать этот снимок?",
            options=["планово", "в течение суток", "срочно", "экстренно"],
        ),
        Question(
            key="needs_radiologist",
            text="Нужно ли второе чтение врачом-рентгенологом (double read)?",
            options=["нет", "да"],
        ),
        _priority_q(),
    ],
}

FAMILIES_OF_SCHEMA = {
    "computer_use": "computer_use",
    "web_ui": "computer_use",
    "documents": "real_world",
    "charts": "real_world",
    "food": "real_world",
    "scenes": "real_world",
    "medical": "real_world",
}


# --------------------------------------------------------------------------------------
# 3. Источники картинок на HuggingFace
#
#    image_keys     — кандидаты колонок с картинкой (берётся первая найденная)
#    hint_keys      — колонки с текстовыми уликами -> имя поля в промпте учителя
#    label_key      — колонка с меткой класса (int) -> пытаемся раскрыть в название
#    gt_*           — откуда взять ground truth для заземлённых вопросов
#    optional=True  — если датасет недоступен/изменил схему, просто warn и идём дальше
# --------------------------------------------------------------------------------------

@dataclass
class Source:
    name: str
    hf_id: str
    family: str                                   # ключ SCHEMAS
    split: str = "train"
    config: Optional[str] = None
    image_keys: Sequence[str] = ("image",)
    hint_keys: Dict[str, str] = field(default_factory=dict)
    label_key: Optional[str] = None
    gt_action_key: Optional[str] = None           # инструкция/действие (computer-use)
    gt_answer_key: Optional[str] = None           # верный ответ (charts/qa)
    gt_facts_keys: Sequence[str] = ()             # факты для заземления (детекции, OCR, сущности)
    html_key: Optional[str] = None                # колонка с HTML (WebSight)
    note: str = ""
    optional: bool = False
    extra_load_kwargs: Dict[str, Any] = field(default_factory=dict)


SOURCES: List[Source] = [
    # ============================ COMPUTER USE / GUI ============================
    Source(
        name="screenspot",
        hf_id="rootsautomation/ScreenSpot",
        family="computer_use",
        split="test",                              # в этом датасете только test
        image_keys=("image",),
        hint_keys={"instruction": "instruction", "platform_or_app": "data_source",
                   "element_type": "data_type", "target_bbox_xyxy": "bbox",
                   "file_name": "file_name"},
        gt_action_key="instruction",
        gt_facts_keys=("instruction", "data_source", "data_type"),
        note="GUI grounding (SeeClick): ~1.2k скриншотов iOS/Android/macOS/Windows/Web, "
             "инструкция + bbox целевого элемента (text/icon)",
    ),
    Source(
        name="screenspot_pro",
        hf_id="likaixin/ScreenSpot-Pro",
        family="computer_use",
        split="train",
        image_keys=("image", "screenshot"),
        hint_keys={"instruction": "instruction", "app": "app", "element_type": "element_type",
                   "target_bbox": "bbox", "industry": "industry"},
        gt_action_key="instruction",
        gt_facts_keys=("instruction", "app"),
        note="ScreenSpot-Pro: 1.5k high-res скриншотов профессионального софта (CAD/IDE/медицина), "
             "инструкция + bbox",
        optional=True,
    ),
    Source(
        name="click100k",
        hf_id="mlfoundations/Click-100k",
        family="computer_use",
        split="train",
        image_keys=("images", "image", "image_path"),   # images = list[PIL]
        hint_keys={"instruction": "easyr1_prompt", "target_bbox_xyxy": "bbox",
                   "image_width": "image_width"},
        gt_action_key="easyr1_prompt",
        gt_facts_keys=("easyr1_prompt",),
        note="Click-100k: 100k desktop-скриншотов с grounding-инструкциями "
             "(смесь ScreenSpot-Pro / OSWorld-G / JEDI)",
        optional=True,
    ),
    Source(
        name="screen2words",
        hf_id="rootsautomation/RICO-Screen2Words",
        family="computer_use",
        split="train",
        image_keys=("image",),
        hint_keys={"screen_summary": "captions", "android_app": "app_package_name",
                   "screen_id": "screenId"},
        gt_facts_keys=("captions", "app_package_name"),
        note="RICO + Screen2Words: 22k реальных скриншотов Android-приложений "
             "и человеческие саммари экрана (что это за экран и зачем)",
    ),
    Source(
        name="androidcontrol",
        hf_id="HarrytheOrange/parsed_AndroidControl",
        family="computer_use",
        split="train",
        image_keys=("screenshot", "image", "screenshot_image", "obs"),
        hint_keys={"goal": "goal", "step_instruction": "instruction", "action": "action",
                   "app": "app", "step_id": "step_id"},
        gt_action_key="instruction",
        gt_facts_keys=("goal", "instruction", "action"),
        note="Parsed AndroidControl: ~15k демонстраций задач в Android-приложениях "
             "(high/low-level инструкции, скриншоты, accessibility tree)",
        optional=True,
    ),
    Source(
        name="websight",
        hf_id="HuggingFaceM4/WebSight",
        config="v0.2",
        family="web_ui",
        split="train",
        image_keys=("image",),
        hint_keys={},
        html_key="generated_html",
        gt_facts_keys=("generated_html",),
        note="WebSight v0.2: 823k синтетических скриншотов сайтов + HTML/CSS "
             "(текст страницы вытаскиваем из HTML)",
        optional=True,
    ),

    # ============================ РЕАЛЬНЫЙ МИР ============================
    Source(
        name="sroie_receipts",
        hf_id="rth/sroie-2019-v2",
        family="documents",
        split="train",
        image_keys=("image",),
        hint_keys={"ocr_and_entities": "objects"},
        gt_facts_keys=("objects",),
        note="ICDAR-2019 SROIE: 973 скана реальных чеков + OCR-тексты + "
             "извлечённые company/date/address/total",
    ),
    Source(
        name="chartqa",
        hf_id="HuggingFaceM4/ChartQA",
        family="charts",
        split="train",
        image_keys=("image",),
        hint_keys={"chart_question": "query", "chart_answer": "label",
                   "human_or_machine": "human_or_machine"},
        gt_answer_key="label",
        gt_facts_keys=("query", "label"),
        note="ChartQA: 32k реальных и синтетических графиков/дашбордов + вопрос + ответ",
    ),
    Source(
        name="food101",
        hf_id="ethz/food101",
        family="food",
        split="train",
        image_keys=("image",),
        hint_keys={},
        label_key="label",
        gt_facts_keys=("label",),
        note="Food-101: 101k реальных фото еды из Foodspotting, 101 класс блюд",
    ),
    Source(
        name="coco_detections",
        hf_id="detection-datasets/coco",
        family="scenes",
        split="train",
        image_keys=("image",),
        hint_keys={"detected_objects": "objects"},
        gt_facts_keys=("objects",),
        note="COCO (detection-упаковка): 122k реальных фото + объекты с категориями и bbox",
        optional=True,
    ),
    Source(
        name="coco_captions",
        hf_id="cat-state/mscoco-1st-caption",
        family="scenes",
        split="train",
        image_keys=("image", "jpg", "png"),
        hint_keys={"caption": "caption", "original_size": "original_width"},
        gt_facts_keys=("caption",),
        note="MS COCO (webdataset-упаковка): 118k реальных фото + первая подпись COCO",
        optional=True,
    ),
    Source(
        name="chest_xray",
        hf_id="trpakov/chest-xray-classification",
        family="medical",
        split="train",
        image_keys=("image",),
        hint_keys={},
        label_key="label",
        gt_facts_keys=("label",),
        note="Рентген грудной клетки (imagefolder): метка normal/pneumonia — "
             "сценарий триажа снимков",
        optional=True,
    ),
]

SOURCE_BY_NAME = {s.name: s for s in SOURCES}


# --------------------------------------------------------------------------------------
# 4. Мелкие утилиты
# --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] WARNING: {msg}", file=sys.stderr, flush=True)


def stable_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[:n]


def clamp_text(s: Optional[str], limit: int) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:limit]


def strip_html(html: str, limit: int = 900) -> str:
    txt = re.sub(r"<(style|script|head)[^>]*>.*?</\1>", " ", html or "", flags=re.S | re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = re.sub(r"&[a-z#0-9]+;", " ", txt)
    return clamp_text(txt, limit)


def is_image_like(value: Any) -> bool:
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, dict):
        return "bytes" in value or "path" in value
    if isinstance(value, (list, tuple)) and value and is_image_like(value[0]):
        return True
    return False


def to_pil(value: Any) -> Optional[Image.Image]:
    """Достаём PIL.Image из того, что вернул datasets (PIL / {'bytes':..} / list / путь)."""
    try:
        if isinstance(value, Image.Image):
            return value
        if isinstance(value, (list, tuple)):
            for v in value:
                img = to_pil(v)
                if img is not None:
                    return img
            return None
        if isinstance(value, dict):
            raw = value.get("bytes")
            if raw:
                return Image.open(io.BytesIO(raw))
            path = value.get("path")
            if path and os.path.exists(path):
                return Image.open(path)
            return None
        if isinstance(value, (bytes, bytearray)):
            return Image.open(io.BytesIO(value))
        if isinstance(value, str) and os.path.exists(value):
            return Image.open(value)
    except Exception as e:                                   # битая/не поддерживаемая картинка
        warn(f"не удалось прочитать картинку: {type(e).__name__}: {e}")
    return None


def summarize(value: Any, max_len: int = 1400, max_items: int = 30) -> str:
    """Универсальный рендер любого значения из датасета в короткий текст для промпта."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)[:max_len]
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if is_image_like(v):
                continue
            s = summarize(v, max_len=max(60, max_len // 3), max_items=max_items)
            if s:
                parts.append(f"{k}: {s}")
        return "; ".join(parts)[:max_len]
    if isinstance(value, (list, tuple)):
        items = []
        for v in list(value)[:max_items]:
            if is_image_like(v):
                continue
            s = summarize(v, max_len=90, max_items=6)
            if s:
                items.append(s)
        return ", ".join(items)[:max_len]
    if isinstance(value, Image.Image):
        return ""
    return str(value)[:max_len]


def parse_json_object(text: str) -> Dict[str, Any]:
    """Вытаскиваем JSON-объект из ответа модели ( markdown-забор, лишний текст )."""
    if not text:
        raise ValueError("пустой ответ модели")
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, flags=re.S)
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("ответ модели не JSON-объект")
    return obj


def normalize_probs(p: Sequence[float]) -> List[float]:
    vals = [max(0.0, float(x)) for x in p]
    s = sum(vals)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals) if vals else []
    return [v / s for v in vals]


def aggregate_votes(votes: Sequence[Optional[Sequence[float]]], n_options: int,
                    smoothing: float = DEFAULT_LABEL_SMOOTHING,
                    gt_index: Optional[int] = None,
                    gt_mix: float = 0.0) -> Optional[List[float]]:
    """Усредняем K распределений учителя (self-consistency) + сглаживание + априор из датасета."""
    acc = [0.0] * n_options
    used = 0
    for v in votes:
        if v is None:
            continue
        v = list(v)
        if len(v) != n_options:
            continue
        p = normalize_probs(v)
        acc = [a + b for a, b in zip(acc, p)]
        used += 1
    if used == 0:
        return None
    p = [a / used for a in acc]
    if smoothing > 0 and n_options:
        p = [(1.0 - smoothing) * x + smoothing / n_options for x in p]
    if gt_index is not None and gt_mix > 0 and 0 <= gt_index < n_options:
        onehot = [0.0] * n_options
        onehot[gt_index] = 1.0
        p = [(1.0 - gt_mix) * a + gt_mix * b for a, b in zip(p, onehot)]
    return [round(x, 6) for x in normalize_probs(p)]


def clean_options(options: Sequence[Any], max_options: int = MAX_OPTIONS,
                  max_words: int = 14) -> List[str]:
    """Дедуп + обрезка вариантов ответа, чтобы closed-set softmax был честным."""
    out: List[str] = []
    seen = set()
    for raw in options:
        if raw is None:
            continue
        s = clamp_text(str(raw), 160)
        s = re.sub(r"^\s*[-*•\d\).\]]+\s*", "", s).strip()
        if not s:
            continue
        if len(s.split()) > max_words:
            s = " ".join(s.split()[:max_words])
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_options:
            break
    return out


def encode_image_b64(path: Path, max_side: int = 768, quality: int = 80) -> str:
    """Миниатюра в base64 для VL-учителя (экономим токены на больших скриншотах)."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------------------
# 5. Клиент DeepSeek (OpenAI-совместимый /chat/completions) c ретраями и rate-limit
# --------------------------------------------------------------------------------------

class TransientAPIError(Exception):
    pass


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 timeout: int = 120, max_retries: int = 6, backoff: float = 1.5,
                 min_interval: float = 0.0):
        if requests is None:
            raise SystemExit("нужен пакет requests: pip install requests")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.min_interval = min_interval
        self.session = requests.Session()
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7,
             max_tokens: int = 1500, model: Optional[str] = None,
             json_mode: bool = True, top_p: float = 0.95) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                r = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise TransientAPIError(f"HTTP {r.status_code}: {r.text[:200]}")
                if r.status_code >= 400:
                    # 4xx (кроме 429) — не временная ошибка: нет смысла долбить
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                with self._lock:
                    self.calls += 1
                    self.tokens_in += int(usage.get("prompt_tokens") or 0)
                    self.tokens_out += int(usage.get("completion_tokens") or 0)
                return content
            except (requests.RequestException, TransientAPIError, KeyError, ValueError) as e:
                last_err = e
                sleep_s = min(60.0, self.backoff * (2 ** attempt)) + random.uniform(0, 1.0)
                warn(f"API-ошибка ({type(e).__name__}: {str(e)[:160]}), ретрай {attempt + 1}/"
                     f"{self.max_retries} через {sleep_s:.1f}s")
                time.sleep(sleep_s)
        raise RuntimeError(f"DeepSeek API не ответил после {self.max_retries} попыток: {last_err}")


# --------------------------------------------------------------------------------------
# 6. Загрузка картинок и «улик» из HuggingFace
# --------------------------------------------------------------------------------------

def load_stream(src: Source, hf_token: Optional[str], seed: int, shuffle: bool):
    """Стриминг датасета; при ошибке с запрошенным сплитом пробуем типовые alternatives."""
    if load_dataset is None:
        raise RuntimeError("не установлен пакет `datasets`: pip install -U datasets")

    kwargs: Dict[str, Any] = dict(src.extra_load_kwargs)
    if src.config:
        args: Tuple[Any, ...] = (src.hf_id, src.config)
    else:
        args = (src.hf_id,)
    if hf_token:
        kwargs["token"] = hf_token

    candidate_splits = [src.split] + [s for s in ("train", "test", "validation", "val")
                                      if s != src.split]
    last_err: Optional[Exception] = None
    for sp in candidate_splits:
        try:
            ds = load_dataset(*args, split=sp, streaming=True, **kwargs)
            if shuffle:
                try:
                    ds = ds.shuffle(seed=seed, buffer_size=1000)
                except Exception:
                    pass
            # прогрев: тянем первую строку, чтобы сразу узнать реальную схему колонок
            first = next(iter(ds))
            return ds, first
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"не удалось загрузить {src.hf_id} (сплиты {candidate_splits}): {last_err}")


def resolve_image(row: Dict[str, Any], src: Source) -> Optional[Image.Image]:
    for key in src.image_keys:
        if key in row and is_image_like(row[key]):
            img = to_pil(row[key])
            if img is not None:
                return img
    # fallback: любая колонка, похожая на картинку
    for key, value in row.items():
        if is_image_like(value):
            img = to_pil(value)
            if img is not None:
                return img
    return None


def label_names(ds_like: Any, key: str) -> Optional[List[str]]:
    try:
        features = getattr(ds_like, "features", None) or {}
        feat = features.get(key)
        names = getattr(feat, "names", None)
        if names:
            return [str(n) for n in names]
        # ClassLabel может лежать вложенным (imagefolder -> 'label')
        if isinstance(feat, dict) and feat.get("names"):
            return [str(n) for n in feat["names"]]
    except Exception:
        pass
    return None


def build_hints(row: Dict[str, Any], src: Source, ds_like: Any) -> Dict[str, str]:
    """Собираем текстовые улики из строки датасета."""
    hints: Dict[str, str] = {}

    for field_name, col in src.hint_keys.items():
        if col not in row:
            continue
        value = row[col]
        if src.html_key and col == src.html_key:
            hints[field_name] = strip_html(str(value))
            continue
        text = summarize(value)
        if text:
            hints[field_name] = text

    # HTML-колонка, если она не попала в hint_keys
    if src.html_key and src.html_key in row and "page_text" not in hints:
        hints["page_text"] = strip_html(str(row[src.html_key]))

    # метка класса -> название
    if src.label_key and src.label_key in row:
        raw_label = row[src.label_key]
        names = label_names(ds_like, src.label_key)
        if names is not None and isinstance(raw_label, int) and 0 <= raw_label < len(names):
            hints["GT_LABEL"] = names[raw_label]
        elif isinstance(raw_label, str):
            hints["GT_LABEL"] = raw_label
        else:
            hints["GT_LABEL"] = f"label_id={raw_label}"

    # ground truth для заземлённых вопросов
    if src.gt_action_key and src.gt_action_key in row:
        hints["GT_ACTION"] = clamp_text(summarize(row[src.gt_action_key]), 400)
    if src.gt_answer_key and src.gt_answer_key in row:
        hints["GT_ANSWER"] = clamp_text(summarize(row[src.gt_answer_key]), 300)
    if src.gt_facts_keys:
        facts = []
        for col in src.gt_facts_keys:
            if col in row:
                s = summarize(row[col], max_len=600)
                if src.html_key and col == src.html_key:
                    s = strip_html(str(row[col]), 600)
                if s:
                    facts.append(f"{col}: {s}")
        if facts:
            hints["GT_FACTS"] = " | ".join(facts)[:1400]
    return {k: v for k, v in hints.items() if v}


def save_image(img: Image.Image, images_root: Path, source_name: str, stem: str,
               max_side: int, fmt: str = "JPEG", quality: int = 88) -> Tuple[Path, str]:
    """Сохраняем картинку, возвращаем (абсолютный путь, md5 содержимого)."""
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    out_dir = images_root / source_name
    out_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    digest = hashlib.md5(buf.getvalue()).hexdigest()
    ext = "jpg" if fmt.upper() in ("JPEG", "JPG") else fmt.lower()
    path = out_dir / f"{stem}_{digest[:8]}.{ext}"
    path.write_bytes(buf.getvalue())
    return path, digest


# --------------------------------------------------------------------------------------
# 7. Промпты учителя (фаза A — контекст и варианты, фаза B — вероятности)
# --------------------------------------------------------------------------------------

SYSTEM_A = """Ты — разметчик-учитель для калиброванной мультимодальной модели закрытого типа (CORA).
Модель не генерирует текст: на каждый вопрос она выдаёт распределение вероятностей по заранее \
заданному списку вариантов. Твоя разметка позже усредняется по нескольким твоим же ответам, \
поэтому будь последовательным и честным про неопределённость.

Правила:
1. Отвечай СТРОГО одним валидным JSON-объектом. Никакого текста вне JSON, никаких markdown-заборов.
2. "state_text": плотный бизнес-контекст, 3-6 предложений. Опирайся на METADATA (это факты о \
картинке), но добавь правдоподобные рабочие детали: кто участник (оператор/аналитик/врач/агент \
автоматизации), номер тикета или заявки, SLA, срок, что было раньше. НЕ противоречь METADATA и \
не выдумывай факты, которые видно опровергнуть по картинке.
3. Для каждого GROUNDED SLOT придумай ровно столько вариантов, сколько указано в n_options:
   - один вариант ДОЛЖЕН соответствовать ground truth из METADATA (можно переформулировать, \
но смысл и целевой объект сохранить);
   - остальные — правдоподобные дистракторы той же granularity и того же стиля;
   - перемешай порядок и верни ground_truth_index (0-based) для каждого слота;
   - варианты короткие: до 12 слов, без «все вышеперечисленное» и «не знаю».
4. "extra_questions": {n_extra} дополнительных вопроса, специфичных ИМЕННО для этой картинки \
(не дублируй основные). Формат каждого: key (snake_case), text, options (2-6 коротких вариантов), \
ground_truth_index (int или null).
5. Не упоминай в state_text и вопросах, что данные взяты из датасета/бенчмарка, и не ссылайся \
на «инструкцию аннотатора» — пиши так, будто это реальная рабочая ситуация."""

SYSTEM_B = """Ты — калиброванный эксперт-разметчик. Для каждого вопроса выдай распределение \
вероятностей ПО УЖЕ ЗАДАННЫМ вариантам ответа (closed-set).

Правила:
1. Отвечай СТРОГО одним JSON-объектом вида {"probs": {"<question_key>": [p1, p2, ...]}}.
2. Длина массива = числу вариантов этого вопроса; сумма = 1.0; все числа >= 0.
3. Порядок вероятностей = порядок вариантов в списке.
4. Будь калиброванным: если METADATA не даёт однозначного ответа, распредели массу между \
правдоподобными вариантами (например 0.55/0.3/0.15). Не ставь 1.0, если есть хоть малая \
альтернатива. Типичный уверенный ответ — 0.6..0.9 на лучшем варианте.
5. Не добавляй свои варианты и не меняй ключи вопросов."""


def render_metadata(hints: Dict[str, str], limit: int = 2600) -> str:
    """Блок METADATA для промпта: по строке на каждое поле, переносы строк сохраняем."""
    if not hints:
        return "(метаданных нет — опирайся только на тип сценария)"
    lines = [f"- {k}: {re.sub(r'[ ]+', ' ', str(v)).strip()}" for k, v in hints.items()]
    return "\n".join(lines)[:limit]


def build_phase_a_messages(src: Source, hints: Dict[str, str], schema: List[Question],
                           n_extra: int, lang: str, vision_b64: Optional[str] = None,
                           ) -> List[Dict[str, Any]]:
    grounded_slots = [q for q in schema if q.grounded]
    slots_json = [
        {"slot_key": q.key, "question": q.text, "n_options": q.n_options, "ground_truth_hint": q.gt_hint}
        for q in grounded_slots
    ]
    fixed_questions = [
        {"key": q.key, "text": q.text, "options": q.options}
        for q in schema if not q.grounded and q.options
    ]

    user_text = f"""СЦЕНАРИЙ: {src.family} (источник картинок: {src.name} — {src.note})

METADATA (факты из аннотации картинки):
{render_metadata(hints)}

FIXED QUESTIONS (варианты уже заданы, их придумывать НЕ надо):
{json.dumps(fixed_questions, ensure_ascii=False, indent=1)}

GROUNDED SLOTS (придумай варианты ответа по ground truth из METADATA):
{json.dumps(slots_json, ensure_ascii=False, indent=1)}

ЗАДАЧА: верни JSON строго такой формы:
{{
  "state_text": "<3-6 предложений контекста, на {LANG_NAMES.get(lang, lang)} языке>",
  "grounded_slots": {{
    "<slot_key>": {{"options": ["...", "..."], "ground_truth_index": 0}}
  }},
  "extra_questions": [
    {{"key": "<snake_case>", "text": "<вопрос>", "options": ["...", "..."], "ground_truth_index": null}}
  ]
}}

Язык всех текстов (state_text, вопросы, варианты ответа): {LANG_NAMES.get(lang, lang)}.
Число extra_questions: {n_extra}."""

    if vision_b64:
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": "SCREENSHOT/PHOTO (само изображение):\n"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{vision_b64}"}},
            {"type": "text", "text": "\n" + user_text},
        ]
        return [{"role": "system", "content": SYSTEM_A.replace("{n_extra}", str(n_extra))},
                {"role": "user", "content": content}]
    return [{"role": "system", "content": SYSTEM_A.replace("{n_extra}", str(n_extra))},
            {"role": "user", "content": user_text}]


def build_phase_b_messages(state_text: str, questions: List[Question], hints: Dict[str, str],
                           lang: str, vision_b64: Optional[str] = None,
                           ) -> List[Dict[str, Any]]:
    qlist = [
        {"key": q.key, "text": q.text, "options": q.options}
        for q in questions
    ]
    user_text = f"""STATE_TEXT (ситуация):
{clamp_text(state_text, MAX_STATE_CHARS)}

METADATA (факты о картинке):
{render_metadata(hints, limit=1800)}

QUESTIONS:
{json.dumps(qlist, ensure_ascii=False, indent=1)}

ЗАДАЧА: верни JSON {{"probs": {{"<key>": [<вероятности по вариантам в том же порядке>]}}}} \
для ВСЕХ перечисленных ключей. Язык рассуждений не важен — нужен только JSON."""

    if vision_b64:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{vision_b64}"}},
            {"type": "text", "text": user_text},
        ]
        return [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": content}]
    return [{"role": "system", "content": SYSTEM_B}, {"role": "user", "content": user_text}]


# --------------------------------------------------------------------------------------
# 8. Сборка одного примера
# --------------------------------------------------------------------------------------

@dataclass
class GenResult:
    example: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def resolve_questions(schema: List[Question], phase_a: Dict[str, Any]) -> List[Question]:
    """Подставляем варианты, придуманные учителем, в grounded-слоты + extra_questions."""
    slots = phase_a.get("grounded_slots") or {}
    resolved: List[Question] = []

    for q in schema:
        if not q.grounded:
            resolved.append(q)
            continue
        raw = slots.get(q.key) or {}
        options = clean_options(raw.get("options") or [])
        if len(options) < 2:
            continue                                   # слот не заполнен — вопрос пропускаем
        gt = raw.get("ground_truth_index", None)
        try:
            gt_idx = int(gt) if gt is not None else None
        except (TypeError, ValueError):
            gt_idx = None
        if gt_idx is not None and not (0 <= gt_idx < len(options)):
            gt_idx = None
        resolved.append(Question(key=q.key, text=q.text, options=options,
                                 grounded=True, gt_hint=q.gt_hint, gt_index=gt_idx))

    # дополнительные вопросы учителя (опционально)
    for extra in (phase_a.get("extra_questions") or [])[:2]:
        if not isinstance(extra, dict):
            continue
        key = clamp_text(str(extra.get("key") or ""), 40)
        text = clamp_text(str(extra.get("text") or ""), 200)
        options = clean_options(extra.get("options") or [])
        if not re.fullmatch(r"[a-z0-9_]{3,40}", key) or not text or len(options) < 2:
            continue
        if any(k.key == key for k in resolved):
            continue
        gt = extra.get("ground_truth_index", None)
        try:
            gt_idx = int(gt) if gt is not None else None
        except (TypeError, ValueError):
            gt_idx = None
        if gt_idx is not None and not (0 <= gt_idx < len(options)):
            gt_idx = None
        resolved.append(Question(key=key, text=text, options=options,
                                 grounded=gt_idx is not None, gt_index=gt_idx))

    return resolved


def dry_state_text(src: Source, hints: Dict[str, str]) -> str:
    """state_text без API: склеиваем факты из аннотации (для --dry-run и smoke-тестов)."""
    parts = [f"Сценарий: {src.family}. Источник материала: {src.name}."]
    for k, v in hints.items():
        parts.append(f"{k}: {v}")
    return clamp_text(" ".join(parts), MAX_STATE_CHARS)


def generate_one(client: Optional[DeepSeekClient], src: Source, row: Dict[str, Any],
                 ds_like: Any, args: argparse.Namespace, images_root: Path,
                 out_root: Path, seen_hashes: set, seen_lock: threading.Lock,
                 rng: random.Random) -> GenResult:
    img = resolve_image(row, src)
    if img is None:
        return GenResult(error=f"{src.name}: в строке нет картинки")

    hints = build_hints(row, src, ds_like)
    schema = SCHEMAS[src.family]

    stem = f"{src.name}_{stable_hash(json.dumps(hints, ensure_ascii=False, sort_keys=True))}"
    try:
        img_path, digest = save_image(img, images_root, src.name, stem, args.max_image_side)
    except Exception as e:
        return GenResult(error=f"{src.name}: не удалось сохранить картинку: {e}")

    with seen_lock:
        if digest in seen_hashes:
            img_path.unlink(missing_ok=True)
            return GenResult(error=f"{src.name}: дубликат картинки {digest[:8]}")
        seen_hashes.add(digest)

    rel_path = os.path.relpath(img_path, out_root).replace(os.sep, "/")
    ex_id = f"{src.name}_{digest[:12]}"

    # ---- dry-run: без API ----
    if args.dry_run or client is None:
        questions = [q for q in schema if q.options]
        out_questions = [
            {"key": q.key, "text": q.text, "options": list(q.options),
             "target_probs": [round(1.0 / len(q.options), 6)] * len(q.options)}
            for q in questions
        ]
        example = {"id": ex_id, "state_text": dry_state_text(src, hints),
                   "image_path": rel_path, "questions": out_questions}
        meta = {"id": ex_id, "source": src.name, "hf_id": src.hf_id, "family": src.family,
                "license_note": src.note, "image": rel_path, "image_md5": digest,
                "teacher": None, "k": 0, "dry_run": True, "hints": hints}
        return GenResult(example=example, meta=meta)

    vision_b64 = None
    if args.use_vision:
        try:
            vision_b64 = encode_image_b64(img_path, max_side=args.vision_image_side)
        except Exception as e:
            warn(f"{src.name}: не удалось подготовить картинку для VL-модели: {e}")

    # ---- фаза A: контекст + варианты для заземлённых вопросов ----
    msgs_a = build_phase_a_messages(src, hints, schema, args.extra_questions, args.lang, vision_b64)
    try:
        raw_a = client.chat(msgs_a, temperature=args.temperature_a,
                            max_tokens=args.max_tokens_a, model=args.vision_model if args.use_vision else None)
        phase_a = parse_json_object(raw_a)
    except Exception as e:
        return GenResult(error=f"{src.name}/{ex_id}: фаза A: {type(e).__name__}: {e}")

    state_text = clamp_text(str(phase_a.get("state_text") or ""), MAX_STATE_CHARS)
    if len(state_text) < 40:
        state_text = clamp_text(dry_state_text(src, hints) + " " + state_text, MAX_STATE_CHARS)

    questions = resolve_questions(schema, phase_a)
    if not questions:
        return GenResult(error=f"{src.name}/{ex_id}: учитель не вернул ни одного валидного вопроса")

    # ---- фаза B: self-consistency, K сэмплов вероятностей ----
    msgs_b = build_phase_b_messages(state_text, questions, hints, args.lang, vision_b64)
    votes: Dict[str, List[Optional[List[float]]]] = {q.key: [] for q in questions}
    errors = 0
    for _ in range(max(1, args.k)):
        try:
            raw_b = client.chat(msgs_b, temperature=args.temperature_b, max_tokens=args.max_tokens_b,
                                model=args.vision_model if args.use_vision else None)
            probs_obj = (parse_json_object(raw_b) or {}).get("probs") or {}
        except Exception as e:
            errors += 1
            warn(f"{src.name}/{ex_id}: фаза B: {type(e).__name__}: {e}")
            continue
        for q in questions:
            v = probs_obj.get(q.key)
            if isinstance(v, (list, tuple)) and len(v) == len(q.options or []):
                votes[q.key].append([float(x) for x in v])

    # ---- агрегация ----
    out_questions = []
    used_gt_mix = 0.0
    for q in questions:
        n = len(q.options or [])
        gt_idx = q.gt_index
        gt_mix = args.gt_mix if (q.grounded and gt_idx is not None) else 0.0
        used_gt_mix = max(used_gt_mix, gt_mix)
        p = aggregate_votes(votes.get(q.key) or [], n, smoothing=args.label_smoothing,
                            gt_index=gt_idx, gt_mix=gt_mix)
        if p is None:
            if errors >= args.k:               # учитель совсем не отвечал — равномерное распределение
                p = [round(1.0 / n, 6)] * n
            else:
                continue                       # вопрос без единого валидного голоса — выкидываем
        out_questions.append({"key": q.key, "text": q.text, "options": list(q.options),
                              "target_probs": p})

    if not out_questions:
        return GenResult(error=f"{src.name}/{ex_id}: не удалось собрать ни одного вопроса с вероятностями")

    example = {"id": ex_id, "state_text": state_text, "image_path": rel_path,
               "questions": out_questions}
    meta = {
        "id": ex_id, "source": src.name, "hf_id": src.hf_id, "family": src.family,
        "dataset_note": src.note, "image": rel_path, "image_md5": digest,
        "teacher": client.model, "vision_teacher": args.vision_model if args.use_vision else None,
        "k": args.k, "k_failed": errors, "label_smoothing": args.label_smoothing,
        "gt_mix": used_gt_mix,
        "hints": hints, "votes": {k: v for k, v in votes.items()},
    }
    return GenResult(example=example, meta=meta)


# --------------------------------------------------------------------------------------
# 9. Обход источников, сплит и запись
# --------------------------------------------------------------------------------------

def iter_rows(ds_like: Any, first_row: Dict[str, Any], limit: int) -> Iterable[Dict[str, Any]]:
    yield first_row
    count = 1
    for row in ds_like:
        if count >= limit:
            break
        yield row
        count += 1


def load_existing_ids(path: Path) -> set:
    ids = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


def select_sources(args: argparse.Namespace) -> List[Source]:
    if args.sources:
        wanted = [s.strip() for s in args.sources.split(",") if s.strip()]
        unknown = [w for w in wanted if w not in SOURCE_BY_NAME]
        if unknown:
            raise SystemExit(f"неизвестные источники: {unknown}. Доступны: {list(SOURCE_BY_NAME)}")
        return [SOURCE_BY_NAME[w] for w in wanted]
    out = []
    for s in SOURCES:
        fam = FAMILIES_OF_SCHEMA.get(s.family, "real_world")
        if args.family == "all" or args.family == fam:
            out.append(s)
    return out


def run(args: argparse.Namespace) -> int:
    if Image is None:
        raise SystemExit("нужен пакет pillow: pip install pillow")
    out_root = Path(args.out_dir).resolve()
    images_root = Path(args.images_dir) if os.path.isabs(args.images_dir) else out_root / args.images_dir
    images_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.dry_run and not api_key:
        raise SystemExit("нужен DEEPSEEK_API_KEY (или --api-key), либо запусти с --dry-run")
    if args.use_vision and not args.vision_model:
        raise SystemExit("--use-vision требует --vision-model <имя VL-модели на том же API>")

    client: Optional[DeepSeekClient] = None
    if not args.dry_run:
        client = DeepSeekClient(
            api_key=api_key, base_url=args.base_url, model=args.model,
            timeout=args.timeout, max_retries=args.max_retries,
            min_interval=(1.0 / args.rps) if args.rps > 0 else 0.0,
        )
        log(f"учитель: {args.model} @ {args.base_url}"
            + (f" | vision: {args.vision_model}" if args.use_vision else ""))

    sources = select_sources(args)
    log(f"источников: {len(sources)} -> {[s.name for s in sources]}; "
        f"по {args.per_source} картинок на источник")

    rng = random.Random(args.seed)
    seen_hashes: set = set()
    seen_lock = threading.Lock()

    all_examples: List[Dict[str, Any]] = []
    all_meta: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, int]] = {}

    done_ids = set()
    if args.resume:
        for p in (out_root / args.train_file, out_root / args.val_file):
            done_ids |= load_existing_ids(p)
        if done_ids:
            log(f"resume: уже сгенерировано {len(done_ids)} примеров — пропускаю их")

    for src in sources:
        t0 = time.time()
        stats[src.name] = {"ok": 0, "failed": 0}
        try:
            ds_like, first_row = load_stream(src, args.hf_token or os.environ.get("HF_TOKEN"),
                                             args.seed, shuffle=not args.no_shuffle)
        except Exception as e:
            msg = f"{src.name} ({src.hf_id}): {type(e).__name__}: {str(e)[:200]}"
            if src.optional:
                warn("пропускаю источник — " + msg)
                continue
            warn("НЕ удалось загрузить источник — " + msg)
            continue

        # тянем с запасом: часть строк отбракуется (дубликаты, битые картинки, ошибки API)
        fetch_limit = int(args.per_source * args.oversample)
        tasks: List[Tuple[Source, Dict[str, Any]]] = []
        try:
            for row in iter_rows(ds_like, first_row, fetch_limit):
                tasks.append((src, row))
        except Exception as e:
            warn(f"{src.name}: стриминг оборвался на {len(tasks)} строках: {type(e).__name__}: {e}")

        if args.limit:
            tasks = tasks[: args.limit]
        log(f"{src.name}: загружено строк {len(tasks)} (за {time.time() - t0:.1f}s), генерирую...")

        results: List[GenResult] = []
        if args.workers <= 1:
            for s, row in tasks:
                results.append(generate_one(client, s, row, ds_like, args, images_root,
                                            out_root, seen_hashes, seen_lock, rng))
                if len([r for r in results if r.example]) >= args.per_source:
                    break
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(generate_one, client, s, row, ds_like, args, images_root,
                                     out_root, seen_hashes, seen_lock, rng): s
                           for s, row in tasks}
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        results.append(GenResult(error=f"{type(e).__name__}: {e}"))
                    if len([r for r in results if r.example]) >= args.per_source:
                        for f in futures:
                            f.cancel()
                        break

        kept = 0
        for r in results:
            if r.example is None or r.example["id"] in done_ids or kept >= args.per_source:
                stats[src.name]["failed"] += 1
                if r.error and args.verbose:
                    warn(r.error)
                continue
            all_examples.append(r.example)
            all_meta.append(r.meta or {})
            stats[src.name]["ok"] += 1
            kept += 1

        log(f"{src.name}: готово {stats[src.name]['ok']}, отброшено {stats[src.name]['failed']} "
            f"({time.time() - t0:.1f}s)")

    if not all_examples:
        warn("не сгенерировано ни одного примера — проверь доступ к HF и ключ API")
        return 1

    # ---- сплит train/val, стратифицированный по источнику ----
    rng.shuffle(all_examples)
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for ex, meta in zip(all_examples, all_meta):
        by_source.setdefault(meta.get("source", "unknown"), []).append(ex)
    train, val = [], []
    for name, items in by_source.items():
        rng.shuffle(items)
        n_train = max(1, int(round(len(items) * args.train_ratio))) if len(items) > 1 else len(items)
        train.extend(items[:n_train])
        val.extend(items[n_train:])
    rng.shuffle(train)
    rng.shuffle(val)

    def write_jsonl(path: Path, items: Sequence[Dict[str, Any]], append: bool = False) -> None:
        mode = "a" if append else "w"
        with path.open(mode, encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    train_path = out_root / args.train_file
    val_path = out_root / args.val_file
    write_jsonl(train_path, train, append=args.resume and bool(done_ids))
    write_jsonl(val_path, val, append=args.resume and bool(done_ids))

    meta_path = out_root / args.meta_file
    with meta_path.open("a" if (args.resume and done_ids) else "w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teacher_model": args.model if client else None,
        "vision_model": args.vision_model if args.use_vision else None,
        "self_consistency_k": args.k,
        "label_smoothing": args.label_smoothing,
        "gt_mix": args.gt_mix,
        "lang": args.lang,
        "n_train": len(train), "n_val": len(val), "n_total": len(all_examples),
        "per_source": stats,
        "images_dir": os.path.relpath(images_root, out_root),
        "max_image_side": args.max_image_side,
    }
    if client:
        summary["api_calls"] = client.calls
        summary["prompt_tokens"] = client.tokens_in
        summary["completion_tokens"] = client.tokens_out
    (out_root / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"готово: {train_path} ({len(train)}), {val_path} ({len(val)}), {meta_path}")
    if client:
        log(f"API: {client.calls} вызовов, {client.tokens_in} prompt / {client.tokens_out} completion токенов")
    log("Дальше: в ноутбуке укажи CFG['train_path']/CFG['val_path'] на эти файлы "
        "(image_path уже относительный к папке датасета).")
    return 0


# --------------------------------------------------------------------------------------
# 10. CLI
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Генерация мультимодального датасета для CORA через DeepSeek API "
                    "(картинки — из датасетов HuggingFace).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", default="data", help="куда писать train.jsonl/val.jsonl/meta.jsonl")
    p.add_argument("--images-dir", default="images", help="подпапка для картинок (относительно --out-dir)")
    p.add_argument("--train-file", default="train.jsonl")
    p.add_argument("--val-file", default="val.jsonl")
    p.add_argument("--meta-file", default="meta.jsonl")

    p.add_argument("--sources", default="", help="список источников через запятую (см. --list-sources)")
    p.add_argument("--family", default="all", choices=["all", "computer_use", "real_world"],
                   help="какое семейство сценариев генерировать")
    p.add_argument("--list-sources", action="store_true", help="показать доступные источники и выйти")
    p.add_argument("--per-source", type=int, default=40, help="сколько примеров на источник")
    p.add_argument("--oversample", type=float, default=1.6,
                   help="во сколько раз больше строк тянуть из HF (запас на брак/дубли)")
    p.add_argument("--limit", type=int, default=0, help="ограничить число строк на источник (0 = без лимита)")
    p.add_argument("--no-shuffle", action="store_true", help="не перемешивать стрим HF (быстрее старт)")
    p.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE,
                   help="максимальная сторона сохраняемой картинки, px")

    p.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
                   help="модель-учитель (DeepSeek chat API)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-совместимый base URL")
    p.add_argument("--api-key", default="", help="API-ключ (иначе берётся из DEEPSEEK_API_KEY)")
    p.add_argument("--hf-token", default="", help="токен HF (иначе HF_TOKEN) для gated-датасетов")
    p.add_argument("--use-vision", action="store_true",
                   help="отправлять учителю саму картинку (нужна VL-модель)")
    p.add_argument("--vision-model", default=os.environ.get("DEEPSEEK_VISION_MODEL", DEFAULT_VISION_MODEL),
                   help="имя VL-модели для --use-vision")
    p.add_argument("--vision-image-side", type=int, default=768, help="сторона миниатюры для VL-модели")

    p.add_argument("--k", type=int, default=DEFAULT_K, help="число сэмплов self-consistency на пример")
    p.add_argument("--extra-questions", type=int, default=1,
                   help="сколько дополнительных вопросов учитель может придумать на пример (0 — выключить)")
    p.add_argument("--temperature-a", type=float, default=0.8, help="температура фазы A (контекст)")
    p.add_argument("--temperature-b", type=float, default=0.7, help="температура фазы B (вероятности)")
    p.add_argument("--max-tokens-a", type=int, default=1600)
    p.add_argument("--max-tokens-b", type=int, default=900)
    p.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    p.add_argument("--gt-mix", type=float, default=DEFAULT_GT_MIX,
                   help="доля one-hot априора из аннотации датасета в заземлённых вопросах")
    p.add_argument("--lang", default="ru", choices=["ru", "en"], help="язык генерируемых текстов")

    p.add_argument("--workers", type=int, default=4, help="параллельных запросов к API")
    p.add_argument("--rps", type=float, default=2.0, help="глобальный лимит запросов в секунду (0 = без лимита)")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--max-retries", type=int, default=6)

    p.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true", help="дописывать к существующим train/val, пропуская готовые id")
    p.add_argument("--dry-run", action="store_true",
                   help="не дёргать API: state_text из аннотаций, вероятности равномерные")
    p.add_argument("--verbose", action="store_true", help="печатать причины отбраковки примеров")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_sources:
        print(f"{'name':<18} {'family':<14} {'hf_id':<42} note")
        for s in SOURCES:
            fam = FAMILIES_OF_SCHEMA.get(s.family, "real_world")
            flag = " (optional)" if s.optional else ""
            print(f"{s.name:<18} {fam:<14} {s.hf_id:<42} {s.note}{flag}")
        print("\nfamily: computer_use = скриншоты/UI ('какие кнопки жмать'), real_world = фото/документы")
        return 0

    random.seed(args.seed)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
