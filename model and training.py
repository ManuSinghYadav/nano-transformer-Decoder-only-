import torch
import torch.nn as nn
import torch.nn.functional as F

!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(chars)
context_window = 128
batch_size = 64
emb_size = 256
max_iters = 5000
lr = 1e-3
head_dim = 32
n_head = emb_size // head_dim
n_layer = 8

itos = {i:v for i, v in enumerate(chars)}
stoi = {v:i for i, v in enumerate(chars)}

encode = lambda x : [stoi[i] for i in x]
decode = lambda x : [itos[i] for i in x]

enocded_data = torch.tensor(encode(text))

def train_test_split():
  n = int(len(enocded_data) * 0.9)
  train = enocded_data[:n]
  test = enocded_data[n:]
  return train, test

train, test = train_test_split()  # Remove

def get_batch(split):
  data = train if split == 'train' else test
  batch = torch.randint(0, len(data) - context_window, (batch_size,))
  ix = torch.stack([data[i: i+context_window] for i in batch])
  yx = torch.stack([data[i+1: i+context_window+1] for i in batch])
  ix, yx = ix.to(device), yx.to(device)
  return ix, yx

class Head(nn.Module):
  def __init__(self):
    super().__init__()
    self.query = nn.Linear(emb_size, head_dim)
    self.key = nn.Linear(emb_size, head_dim)
    self.value = nn.Linear(emb_size, head_dim)

  def forward(self, x):
    B,T,C = x.shape

    q = self.query(x)
    k = self.key(x)
    v = self.value(x)

    atten = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    tril = torch.tril(torch.ones(context_window, context_window, device=device))
    atten = atten.masked_fill(tril[:T, :T]==0, float('-inf'))
    atten = F.softmax(atten, -1)
    out = atten @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self):
    super().__init__()
    self.heads = nn.ModuleList([Head() for i in range(n_head)])

  def forward(self, x):
    return torch.cat([h(x) for h in self.heads], -1)

class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(emb_size, 4*emb_size),
        nn.ReLU(),
        nn.Linear(4*emb_size, emb_size),
    )
  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self):
    super().__init__()
    self.sa_head = MultiHeadAttention()
    self.lm_head = FeedForward()
    self.ln1 = nn.LayerNorm(emb_size)
    self.ln2 = nn.LayerNorm(emb_size)

  def forward(self, x):
    ln1_out = self.ln1(x)
    sa_out = x + self.sa_head(ln1_out)
    ln2_out = self.ln1(sa_out)
    lm_out = x + self.lm_head(ln2_out)
    return lm_out

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.nn_emb = nn.Embedding(vocab_size, emb_size)
    self.nn_pos = nn.Embedding(context_window, emb_size)
    self.blocks = nn.Sequential(*[Block() for i in range(n_layer)])
    self.nn_linear = nn.Linear(emb_size, vocab_size)

  def forward(self, x, y=None):
    B, T = x.shape
    emb = self.nn_emb(x)
    pos_emb = self.nn_pos(torch.arange(T, device=device))
    pos_emb = pos_emb + emb
    block_out = self.blocks(pos_emb)
    logits = self.nn_linear(block_out)

    if y is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = y.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, ix, max_new_tokens):
    for i in range(max_new_tokens):
      ix_cond = ix[:, -context_window:]
      logits, loss = self.forward(ix_cond)
      logits = logits[:,-1,:]
      probs = F.softmax(logits, -1)
      ixn = torch.multinomial(probs, num_samples=1)
      ix = torch.cat((ix, ixn), dim=1)
    return ix

model = BigramLanguageModel().to(device)
print(f"Total Parameters: {sum(i.numel() for i in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training
for i in range(max_iters):
  ix, yx = get_batch('train')
  logits, loss = model(ix, yx)
  optimizer.zero_grad(set_to_none=True)
  if i % 250 == 0:
    print(f"{i:5d} /  {max_iters} : loss is {loss:.4f}")
  loss.backward()
  optimizer.step()

# Generation
print(''.join(decode(model.generate(torch.tensor([[2]], device=device), 1000)[0].tolist())))
