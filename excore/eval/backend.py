"""The llama.cpp hook: a thin adapter that exposes full next-token logits on CPU.

It uses the ``llama-cpp-python`` bindings, so the model runs in llama.cpp's own CPU
kernels (AVX2/NEON). Install with ``pip install excore[llama]``, and build the bindings
against the same llama.cpp commit that ``configs/hpc_cpu.yaml`` pins for
``llama-quantize`` and ``llama-bench``; mixed versions can change numerics.

Nothing else in eXCore imports llama_cpp, so any object with the same three methods
(see ``accuracy.LogitsBackend``) can replace this adapter.
"""

from __future__ import annotations

import numpy as np


class LlamaCppBackend:
    def __init__(self, model_path: str, *, n_ctx: int, n_threads: int, n_batch: int = 512,
                 logits_all: bool = True):
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on the host
            raise RuntimeError("llama-cpp-python is not installed (pip install 'excore[llama]')") from exc
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=n_batch,
            n_gpu_layers=0,
            logits_all=logits_all,   # keep False for long-context runs: logits for every position cost n_ctx x vocab floats
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        self.logits_all = logits_all
        self.vocab_size = int(self.llm.n_vocab())

    def tokenize(self, text: str) -> list[int]:
        return list(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))

    def bos_token(self) -> int:
        return int(self.llm.token_bos())

    def logits(self, window: np.ndarray, start: int) -> np.ndarray:
        if not self.logits_all:
            raise RuntimeError("this backend was created with logits_all=False")
        self.llm.reset()
        self.llm.eval([int(t) for t in window])
        return np.array(self.llm.scores[start : len(window)], dtype=np.float32, copy=True)

    def generate(self, prompt: list[int], max_new: int) -> str:
        """Greedy continuation of ``prompt`` as text (used by the long-context guards)."""
        out: list[int] = []
        eos = self.llm.token_eos()
        for tok in self.llm.generate(prompt, top_k=1, top_p=1.0, temp=0.0, reset=True):
            if tok == eos or len(out) >= max_new:
                break
            out.append(int(tok))
        return self.llm.detokenize(out).decode("utf-8", errors="ignore")

    def close(self) -> None:
        self.llm.close()
