"""
Train the tiny char-level GPT on tiny Shakespeare.

Run from the repo root:  uv run python -m transformer.gpt.train
"""
import time
import torch

from transformer.gpt.data import CharCodec, get_batch
from transformer.gpt.gpt import GPT, GPTConfig

# baby config
block_size, batch_size = 64, 32
n_layer, n_head, n_embd, dropout = 4, 4, 128, 0.0
lr, max_steps = 1e-3, 3000
eval_interval, eval_iters = 100, 50
seed = 1337

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(seed)

# 1. data: read data/input.txt, build the codec, encode to a LongTensor, 90/10 split into train and val
with open('transformer/gpt/data/input.txt', encoding='utf-8') as f:
    text = f.read()

codec = CharCodec(text)
data = torch.tensor(codec.encode(text), dtype=torch.long)   # (N,) about 1.1M ids

n = int(0.9 * len(data))      # cut index
train_data = data[:n]         # first 90%
val_data = data[n:]           # last 10%

print(f"vocab {codec.vocab_size}  train {len(train_data):,}  val {len(val_data):,}")

# 2. model: GPTConfig, GPT, .to(device); print the parameter count and the device
cfg = GPTConfig(
    block_size=block_size,
    vocab_size=codec.vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
)
model = GPT(cfg).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"device {device}  params {n_params:,}")

# 3. optimizer: torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# 4. estimate_loss(): average loss over eval_iters batches for each split, without training
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size)
            x, y = x.to(device), y.to(device)      # (B, T) each, on the model's device
            _, loss = model(x, y)
            losses[k] = loss.item()                # a Python float, copied back to the CPU
        out[split] = losses.mean().item()          # average over eval_iters batches
    model.train()
    return out

print(estimate_loss()) 

# 5. loop over max_steps: every step, get a batch, move it to device, forward, zero_grad, backward, step.
#    Every eval_interval steps (before that step's update), print train and val loss (4 decimals) and a 200-character sample.
def sample(n_chars):
    model.eval()
    prompt = torch.tensor([codec.encode("\n")], device=device)   # (1, 1)
    out = model.generate(prompt, max_new_tokens=n_chars)          # (1, 1 + n_chars)
    model.train()
    return codec.decode(out[0].tolist())

print("initial sample:")
print(sample(500))

t0 = time.time()
for step in range(max_steps):
    # evaluate every eval_interval steps (including step 0, before any training)
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - t0:.0f}s")
        print(sample(200))
        print("-" * 60)

    # one training step
    x, y = get_batch(train_data, block_size, batch_size)
    x, y = x.to(device), y.to(device)          # (B, T) each
    _, loss = model(x, y)                      # scalar
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 6. final evaluation and a longer sample
losses = estimate_loss()
print(f"final      | train {losses['train']:.4f} | val {losses['val']:.4f} | {time.time() - t0:.0f}s")

# 7. at the end: generate 500 characters from "\n", decode, print
print(sample(500))
