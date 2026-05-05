from __future__ import annotations

import structlog

from app.ingestion.embedders.protocols import SparseVector
from app.observability.tracing import stage_span

logger = structlog.get_logger("app.ingestion.embedders.sparse")


class SpladeSparseEmbedder:
    def __init__(self, model_name: str, hf_token: str | None = None) -> None:
        self._model_name = model_name
        self._hf_token = hf_token
        self._model = None
        self._tokenizer = None

    def _load(self, hf_token: str | None = None) -> None:
        if self._model is not None:
            return

        # Disable the background safetensors conversion thread — it POSTs to
        # safetensors-convert.hf.space which is often unavailable (503).
        try:
            import transformers.safetensors_conversion as _sc
            _sc.auto_conversion = lambda *args, **kwargs: None
        except Exception:
            pass

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, token=hf_token)
        self._model = AutoModelForMaskedLM.from_pretrained(self._model_name, token=hf_token)
        self._model.eval()

    def _compute_sparse(self, text: str) -> SparseVector:
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            output = self._model(**inputs)
        logits = output.logits
        sparse_vec = torch.relu(logits).max(dim=1).values
        agg = torch.log1p(sparse_vec).max(dim=0).values.squeeze()

        non_zero = agg.nonzero().squeeze(-1)
        indices = non_zero.tolist()
        values = agg[non_zero].tolist()

        return SparseVector(indices=indices, values=values)

    async def embed(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        self._load(self._hf_token)
        with stage_span("embed.sparse", model=self._model_name, n=len(texts)):
            return [self._compute_sparse(text) for text in texts]
