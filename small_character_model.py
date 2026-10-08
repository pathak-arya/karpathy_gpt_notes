import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embedding = 384
num_heads = 6
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

torch.manual_seed(1337)

with open('input.txt') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

ctoi = {char: i for i, char in enumerate(chars)}
itoc = {i: char for i, char in enumerate(chars)}
encode = lambda s: [ctoi[c] for c in s]
decode = lambda arr: ''.join([itoc[int(i)] for i in arr])

# test/train split (bleurgh): first 90% train, last 10% is test
data = torch.tensor(encode(text), dtype=torch.long)
pivot = int(0.9*len(data))
train = data[:pivot]
val = data[pivot:]

# frequency = sum(d == 0 for d in data)/len(data)     # it's 3.5%, zero doesn't even appear all that much

"""
B, T, C = 4, 8, 32
x = torch.randn((B, T, C))
xbow = torch.zeros((B, T, C))
for batch in range(B):
    for t in range(T):
        xprev = x[batch, :t+1]
        xbow[batch, t] = xprev.mean(axis=0, dtype=torch.float)
print(xbow)
^very inefficient (bag-of-words: at each token, take the mean of all prev tokens including that token)

same operation as above but achieved through matrix multiplication:
1 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 -> /2
1 1 1 0 0 0 0 0 -> /3
1 1 1 1 0 0 0 0 -> /4
1 1 1 1 1 0 0 0 -> /5
1 1 1 1 1 1 0 0 -> /6
1 1 1 1 1 1 1 0 -> /7
1 1 1 1 1 1 1 1 -> /8

w = torch.tril(torch.ones(T, T))
w /= w.sum(axis=1, keepdim=True)

print(k.transpose(1, 2).shape)
print(k.transpose(2, 1).shape)
print(k.transpose(-1, -2).shape)
print(k.transpose(-2, -1).shape)
^all are the same (B, head_size, T)
"""

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.query = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.value = torch.nn.Linear(n_embedding, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)                             # (B, T, head_size)
        q = self.query(x)                           # (B, T, head_size)
        v = self.value(x)                           # (B, T, head_size)

        # 1. write forward pass as function, figure out why logits vary on changing input
        # 2. why does removing masking crash loss, but output only 0? ---> (0+0+0+0+0+0+0+0+2.4)/8=0.3 => per-term loss is zero for exposed tokens
        w = q@k.transpose(-2, -1) * C**(-0.5)       # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        w = F.softmax(w, -1)
        w = self.dropout(w)
        out = w@v                                   # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embedding, n_embedding) # each head can only write to its own lane, projection ensures that each attention head can write to entire embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], -1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embedding, 4*n_embedding),
            nn.ReLU(),
            nn.Linear(4*n_embedding, n_embedding),  # ReLU will only output [0, inf), need a linear layer to combine its outputs so that it can again take values over R
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_heads, n_embedding):
        super().__init__()
        head_size = n_embedding//num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ff = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # pre-norm
        x = x + self.ff(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        # self.sa_heads = MultiHeadAttention(num_heads, n_embedding//num_heads)
        # self.ff = FeedForward(n_embedding)
        self.blocks = nn.Sequential(*[Block(num_heads, n_embedding) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embedding)   # this is pre-norm for cm_head
        self.cm_head = nn.Linear(n_embedding, vocab_size)

    # idx is (B, T)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)                                      # produces (B, T, n_embedding) token_embeddings
        position_embeddings = self.position_embedding_table(torch.arange(T, device=idx.device))                  # produces (T, n_embedding) position_embeddings
        added_embeddings = token_embeddings + position_embeddings                               # broadcasts position_embeddings to (B, T, n_embedding)
        # attended_embeddings = self.sa_heads(added_embeddings)
        # final_embeddings = self.ff(attended_embeddings)
        attended_embeddings = self.blocks(added_embeddings)
        final_embeddings = self.ln_f(attended_embeddings)
        logits = self.cm_head(self.ln_f(final_embeddings))                                      # produces (B, T, C) logits

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # idx is (B, T)
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            B, T = idx.shape
            idx_cropped = idx[:, -block_size:]      # crop idx to only the last block_size tokens
            logits, _ = self(idx_cropped)
            last_tokens = logits[:, -1, :]          # take the last token
            probs = F.softmax(last_tokens, -1)
            # print(probs)
            idx_next = torch.multinomial(probs, num_samples=1)
            # print(idx_next)
            idx = torch.concat((idx, idx_next), dim=1)

        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        # !!!!!!!! NEVER aggregate losses, hides real issues (0, 0, 0, 0, 0, 0, 0, 2.4 reads as 0.3) !!!!!!!!
        out[split] = losses.mean()
    model.train()
    return out

def get_batch(split):
    data = train if split == 'train' else val

    index = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i+block_size] for i in index])        # 4 arrays, each containing i:i+8 sections
    y = torch.stack([data[i+1 : i+block_size+1] for i in index])
    x, y = x.to(device), y.to(device)
    return x, y

def evaluate_batch(split):
    xb, yb = get_batch(split)
    print(xb.shape, yb.shape)
    for i in range(batch_size):
        for j in range(block_size):
            print(xb[i, :j+1], yb[i, j])

def train_bigram():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for step in range(max_iters):
        if step % eval_interval == 0:
            losses = estimate_loss()
            print(f'loss at iteration {step}: {losses['train']:.4f} (train) | {losses['val']:.4f} (val)')

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    losses = estimate_loss()
    print(f'loss at iteration {max_iters-1}: {losses['train']:.4f} (train) | {losses['val']:.4f} (val)')

    print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))

model = BigramLanguageModel()
m = model.to(device)
train_bigram()
