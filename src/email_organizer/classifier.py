"""Cloudflare Clef wrapper: build records, run the joint schema head, apply the threshold.

torch / transformers are imported lazily so the rest of the CLI (and the tests) work
without a GPU stack.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Category, ModelConfig
from .message import Message, clean_body

log = logging.getLogger(__name__)

QUESTION_ID = "category"


@dataclass
class Decision:
    category: str | None  # None when uncertain or on error
    prob: float  # probability of the top option
    probs: dict[str, float] = field(default_factory=dict)
    top: str | None = None  # top option even when below threshold
    error: str | None = None

    @property
    def uncertain(self) -> bool:
        return self.category is None and self.error is None


class Classifier(Protocol):
    def classify(self, messages: Sequence[Message]) -> list[Decision]: ...


def build_record(
    msg: Message,
    categories: Mapping[str, Category],
    *,
    instructions: str = "Which folder should this email be filed in?",
    body_chars: int | None = None,
) -> dict[str, Any]:
    return {
        "state": {
            "from": msg.sender,
            "to": msg.to,
            "subject": msg.subject,
            "date": msg.date,
            "is_mailing_list": msg.is_mailing_list,
            "body": clean_body(msg.text_body, body_chars),
        },
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": instructions,
                "criteria": {cid: cat.criteria for cid, cat in categories.items()},
            }
        },
    }


def decide(probs: Mapping[str, float], threshold: float) -> Decision:
    top = max(probs, key=probs.__getitem__)
    p = float(probs[top])
    return Decision(category=top if p >= threshold else None, prob=p, probs=dict(probs), top=top)


def split_logits(output: Any, questions_per_record: Sequence[int]) -> list[list[Any]]:
    """Normalise the model output into one list of per-question logits per record.

    The model card only shows ``model(batch)[0]`` for a single record, so accept both
    plausible batched layouts: ``out[i]`` = record i's questions, or ``out[0]`` = all
    questions of the batch flattened in record order.
    """
    total = sum(questions_per_record)
    first = list(output[0])
    if len(first) == total:
        out, i = [], 0
        for n in questions_per_record:
            out.append(first[i : i + n])
            i += n
        return out
    if len(first) == questions_per_record[0] and len(output) >= len(questions_per_record):
        return [list(output[i]) for i in range(len(questions_per_record))]
    raise ValueError(f"unexpected model output layout for {len(questions_per_record)} records")


class ClefClassifier:
    def __init__(self, cfg: ModelConfig, categories: Mapping[str, Category]):
        self.cfg = cfg
        self.categories = dict(categories)
        self.model = None
        self.processor = None
        self._fns: dict[str, Any] = {}

    # loading -----------------------------------------------------------------
    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download

        path = str(self.cfg.path) if self.cfg.path else snapshot_download(self.cfg.repo)
        if path not in sys.path:
            sys.path.insert(0, path)
        from joint_schema_model import collate_records, encode_record, load_release_model

        self._fns = {"encode": encode_record, "collate": collate_records}
        kwargs: dict[str, Any] = {}
        if self.cfg.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=list(self.cfg.skip_quant_modules),
            )
        log.info("loading Clef from %s (%s)", path, self.cfg.quantization)
        self.model, self.processor = load_release_model(path, device="cuda", dtype=torch.bfloat16, **kwargs)
        self.model.eval()

    # inference ---------------------------------------------------------------
    def predict(self, records: Sequence[dict[str, Any]]) -> list[dict[str, dict[str, float]]]:
        """Run one forward pass; return {question_id: {option_id: prob}} per record."""
        import torch

        self.load()
        tok = self.processor.tokenizer
        encoded = [
            self._fns["encode"](
                tok, r, max_length=self.cfg.max_length,
                max_state_tokens=self.cfg.max_state_tokens, processor=self.processor,
            )
            for r in records
        ]
        batch = self._fns["collate"](encoded, tok.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            output = self.model(batch)
        per_record = split_logits(output, [len(e.questions) for e in encoded])
        results = []
        for enc, logits in zip(encoded, per_record):
            res = {}
            for q, q_logits in zip(enc.questions, logits):
                probs = q_logits.float().softmax(-1).tolist()
                res[q.question_id] = dict(zip(q.option_ids, probs))
            results.append(res)
        return results

    def _classify_batch(self, messages: Sequence[Message]) -> list[Decision]:
        records = [
            build_record(m, self.categories, instructions=self.cfg.instructions, body_chars=self.cfg.body_chars)
            for m in messages
        ]
        return [decide(p[QUESTION_ID], self.cfg.threshold) for p in self.predict(records)]

    def classify(self, messages: Sequence[Message]) -> list[Decision]:
        import torch

        self.load()
        decisions: list[Decision] = []
        bs = self.cfg.batch_size
        for i in range(0, len(messages), bs):
            chunk = messages[i : i + bs]
            try:
                decisions.extend(self._classify_batch(chunk))
                continue
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log.warning("CUDA OOM on batch of %d; retrying one at a time", len(chunk))
            for m in chunk:
                try:
                    decisions.extend(self._classify_batch([m]))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    decisions.append(Decision(None, 0.0, error="cuda-oom"))
        return decisions


# --- smoke test ------------------------------------------------------------

MODEL_CARD_EXAMPLES: list[dict[str, Any]] = [
    {
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
        "questions": {
            "status": {
                "type": "choice",
                "instructions": "What is the invoice status?",
                "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
            },
            "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
        },
    },
    {
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        },
    },
]


def dtype_report(model: Any, depth: int = 2) -> dict[str, dict[str, int]]:
    """Parameter count per dtype, grouped by module prefix (first `depth` name parts).

    4-bit weights show up as torch.uint8; anything listed as bfloat16 was kept unquantized.
    """
    report: dict[str, dict[str, int]] = {}
    for name, p in model.named_parameters():
        prefix = ".".join(name.split(".")[:depth])
        bucket = report.setdefault(prefix, {})
        bucket[str(p.dtype)] = bucket.get(str(p.dtype), 0) + p.numel()
    return report
