import torch

def sample(logits, temperatures):
    """logits [batch, vocab]; temperatures: one float per row -> token ids [batch]"""
    temps = torch.tensor(temperatures, device=logits.device, dtype=torch.float32)
    greedy = logits.argmax(dim=-1)
    if not bool((temps > 0).any()):
        return greedy
    probs = torch.softmax(logits / temps.clamp(min=1e-5)[:, None], dim=-1)
    sampled = torch.multinomial(probs, 1)[:, 0]
    return torch.where(temps > 0, sampled, greedy)