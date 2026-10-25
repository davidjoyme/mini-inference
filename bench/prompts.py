"""
bench/prompts.py - the prompts benchmarks send, shared by bench/test.py and bench/run_engine.py.

    build_prompt("Tell me a fact", 3)                         short prompt
    build_prompt("Tell me a fact", 3, words=1000)             ~1000 words of shared text first:
                                                              every request starts identically,
                                                              which is what prefix caching reuses
    build_prompt("Tell me a fact", 3, words=1000, unique=True)
                                                              same length, but each request starts
                                                              with its own id, so nothing is shared
"""
FILLER = ("The quick brown fox jumps over the lazy dog while the river flows past old stone "
          "bridges and quiet fields of wheat under a pale morning sky. ").split()


def build_prompt(prompt, request_id, words=0, unique=False):
    question = f"{prompt} (request {request_id})"
    if not words:
        return question
    text = " ".join(FILLER[i % len(FILLER)] for i in range(words))
    if unique:
        return f"Request {request_id}. Read this text:\n{text}\n\n{question}"
    return f"Read this text:\n{text}\n\n{question}"