"""A deterministic stand-in for a model: logits are a fixed function of the last two tokens."""

import numpy as np


class SyntheticBackend:
    def __init__(self, vocab=400, d=16, seed=0, noise=0.0, noise_seed=1):
        rng = np.random.default_rng(seed)
        self.E = rng.normal(size=(vocab, d))
        self.W = rng.normal(size=(d, vocab)) * 1.5
        self.vocab_size = vocab
        self.noise, self.noise_seed = noise, noise_seed

    def tokenize(self, text):
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def bos_token(self):
        return 0

    def logits(self, window, start):
        window = np.asarray(window)
        h = self.E[window] + 0.5 * self.E[np.roll(window, 1)]
        z = h @ self.W
        if self.noise:
            rng = np.random.default_rng(int(window.sum()) + 1000 * self.noise_seed)
            z = z + rng.normal(size=z.shape) * self.noise
        return z[start:].astype(np.float32)


def sample_texts(n_chars=6000, seed=7):
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "kernel", "tensor", "layer", "quant", "cache", "token"]
    text = " ".join(rng.choice(words, size=n_chars // 6))
    return [("a.txt", text[: n_chars // 2]), ("b.txt", text[n_chars // 2 :])]


class Leaky(SyntheticBackend):
    """Answers a needle question only if the needle lies inside its last `window` tokens."""

    def __init__(self, window):
        super().__init__(vocab=256)
        self.window = window

    def generate(self, prompt, max_new):
        import re

        text = bytes(prompt[-self.window:]).decode("latin-1")
        asked = re.search(r"What is the secret passphrase for vault (\S+?)\?", text)
        head = text.split("Question:")[0]
        if asked:
            found = re.search(rf"passphrase for vault {re.escape(asked.group(1))} is (\S+?)\.", head)
            if found:
                return " " + found.group(1)
        return " unknown"
