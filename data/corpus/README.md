# Public fidelity corpus

Plain-text files (`*.txt`, UTF-8) used to measure accuracy drift on CPU. The corpus is
**hash-locked**: `corpus.lock.json` lists the SHA-256 of every file, and the evaluator
refuses to run if a file is missing, extra, or changed.

To populate it (operators only):

1. Put openly licensed text files in this folder. Prefer varied, targeted text
   (prose, code, dialogue, structured data) and keep the total near the token budget
   in `configs/hpc_cpu.yaml` (`accuracy.max_tokens`).
2. Run `python -c "from excore.eval.accuracy import lock_corpus; lock_corpus('data/corpus')"`.
3. Commit the text files and the updated `corpus.lock.json`.

Do not add the private holdout text here. It lives outside the repository.
