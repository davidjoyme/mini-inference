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