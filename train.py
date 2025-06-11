import torch
import torch.nn as nn
from torch.nn import functional as F
import math

batch_size = 32
block_size = 256
eval_interval = 500
max_iters = 5000
lr_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
dropout = 0.3
n_head = 6
n_layer = 6
#-----
torch.manual_seed(1337)

with open('ALL_eminem.txt','r',encoding = 'utf-8') as f:
  text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)


stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text),dtype = torch.long, device = device)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) #generates 4 random numbers, the numbers range from 0 to length of data minus block size
  x = torch.stack([data[i:i+block_size] for i in ix]) #the first random sequence is on top, the second is below the first, and so on
  y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #same logic as x
  return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():

   out = {}
   model.eval()
   #goes through training and validation sets
   for split in ['train','val']:
       #creates an empty array with the number of evaluation iterations
       losses = torch.zeros(eval_iters, device = device)
       for k in range(eval_iters):
          X,Y = get_batch(split) #gets train or val batch
          _, loss = model(X,Y) #gets loss of the model
          losses[k] = loss.item() #passes the loss to the losses array
       out[split] = losses.mean() #gets mean of all the losses
   model.train()
   return out

def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta= 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0/ (theta ** (theta_numerator / head_dim)).to(device)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x, freqs_complex, device):
    head_dim = n_embd//n_head
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], head_dim // 2, 2))
    freqs_complex.unsqueeze(0)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.to(device)


class Head(nn.Module):

   def __init__(self, head_size):
      super().__init__()
      self.key = nn.Linear(n_embd, head_size, bias = False)
      self.query = nn.Linear(n_embd, head_size, bias = False)
      self.value = nn.Linear(n_embd, head_size, bias = False)
      self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size, device = device)))

   def forward(self,x, freqs_complex):
      # get key and query values
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)

      k = apply_rotary_embeddings(k, freqs_complex, x.device)
      q = apply_rotary_embeddings(q, freqs_complex, x.device)
      #compute attention scores
      wei = q @ k.transpose(-2,-1) * C**-0.5
      wei = wei.masked_fill(self.tril[:T,:T] ==0, float("-inf"))
      wei = F.softmax(wei, dim = -1)
      #get value
      v = self.value(x)
      out = wei @ v
      return out

class FeedForward(nn.Module):
   def __init__(self,n_embd):
      super().__init__()
      self.net = nn.Sequential(
         nn.Linear(n_embd, n_embd*4),
         nn.ReLU(),
         nn.Linear(4*n_embd, n_embd),
         nn.Dropout(dropout),
      )
   def forward(self,x):
      return self.net(x)

class Block(nn.Module):
   def __init__(self,n_embd, n_head):
      super().__init__()
      head_size = n_embd//n_head
      self.sa = MultiHeadAttention(n_head, head_size)
      self.ffwd = FeedForward(n_embd)
      self.ln1 = nn.LayerNorm(n_embd)
      self.ln2 = nn.LayerNorm(n_embd)
   def forward(self,x, freqs_complex):
      x = x + self.sa(self.ln1(x), freqs_complex)
      x = x + self.ffwd(self.ln2(x))
      return x

class MultiHeadAttention(nn.Module):
   def __init__(self, num_heads, head_size):
      super().__init__()
      self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
      self.proj = nn.Linear(num_heads * head_size, n_embd)
      self.dropout = nn.Dropout(dropout)

   def forward(self,x, freqs_complex ):
      out = torch.cat([h(x, freqs_complex) for h in self.heads], dim=-1)
      out = self.dropout(self.proj(out))
      return out



class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    # each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
    self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.freqs_complex = precompute_theta_pos_frequencies(n_embd//n_head, block_size , device = device)

  def forward(self, idx, targets = None):
    # idx and targets are both (B,T) tensor of integers batch, time, channels
    B,T = idx.shape
    token_embd = self.token_embedding_table(idx) # (B,T,C)
    x = token_embd #(B,T,C)
    freqs_complex = self.freqs_complex[:T, :]
    for block in self.blocks:
       x = block(x, freqs_complex)
    logits = self.lm_head(x) # (B,T,vocab_size)


    if targets is None:
        loss = None
    else:
        B,T,C = logits.shape
        logits = logits.view(B*T,C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits,targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
          #crop idx to the last block_size tokens
          idx_cond = idx[:,-block_size:]
            # get the predictions
          logits, loss = self(idx_cond)
            # focus only on the last time step
          logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
          probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
          idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
          idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      return idx

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr = lr_rate)

batch_size = 32

for iter in range(max_iters):
   #every once in a while evaluate the loss on the training set and validation sets
   if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f} val loss train loss {losses['val']:.4f} ")
   xb, yb = get_batch("train")

   logits,loss = model(xb,yb)
   optimizer.zero_grad(set_to_none=True)
   loss.backward()
   optimizer.step()


print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long , device=device), max_new_tokens=10000)[0].tolist()))

