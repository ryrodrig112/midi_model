import torch, torch.nn as nn, torch.nn.functional as F


class AttnHead(nn.Module):
    def __init__(self, head_size, n_embed, block_size, dropout=0.5):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.key = nn.Linear(n_embed, head_size, bias=False, device=self.device)
        self.query = nn.Linear(n_embed, head_size, bias=False, device=self.device)
        self.value = nn.Linear(n_embed, head_size, bias=False, device=self.device)
        self.register_buffer('tril', torch.ones(block_size, block_size, device=self.device))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C) # important, k & q are generated independent of eachother
        q = self.query(x)  # (B, T, C)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (
                    C ** -0.5)  # (B, T, 16) @ (B, 16, T) ---> (B, T, T) [resulting matrix contains kq dot product from pairs of tokens]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.5):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.heads = nn.ModuleList([AttnHead(head_size, n_embd, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd, device=self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.5):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = nn.Sequential(nn.Linear(n_embed, 4 * n_embed, device=self.device),
                                 nn.ReLU(),
                                 nn.Linear(4 * n_embed, n_embed, device=self.device),
                                 nn.Dropout(dropout)
                                 )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, num_heads, block_size):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        head_size = n_embed // num_heads  # size of concatenated attn vectors is same size as embedding size
        self.sa = MultiHeadAttn(num_heads, head_size, n_embed, block_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed, device=self.device)
        self.ln2 = nn.LayerNorm(n_embed, device=self.device)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, config=None,
                 n_embd=None, n_head=None, n_blocks=None, block_size=None, vocab_size=None, dropout=0.5):
        super().__init__()

        if config is not None:
            self.n_embd = config.get('n_embd') if n_embd is None else n_embd
            self.n_head = config.get('n_head') if n_head is None else n_head
            self.n_blocks = config.get('n_blocks') if n_blocks is None else n_blocks
            self.block_size = config.get('block_size') if block_size is None else block_size
            self.vocab_size = config.get('vocab_size') if vocab_size is None else vocab_size
            self.dropout = config.get('dropout')
        else:
            assert n_embd is not None and n_head is not None and n_blocks is not None and block_size is not None and vocab_size is not None, "When not using config, all parameters must be provided."
            self.n_embd = n_embd
            self.n_head = n_head
            self.n_blocks = n_blocks
            self.block_size = block_size
            self.vocab_size = vocab_size
            self.dropout = dropout

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd, device=self.device)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd, device=self.device)
        self.blocks = nn.Sequential(*[Block(self.n_embd, self.n_head, self.block_size) for _ in range(
            self.n_blocks)])  # the * unpacks the contents of the list, as seqential cannot take a list
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss



class LayerNorm:  # coded for instruction but used pytorch implentaiton in code
    def __init__(self, dim, eps=1e-5):
        self.gamma = torch.ones(dim)  # sd param
        self.beta = torch.zeros(dim)  # mean param
        self.eps = eps

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / (torch.sqrt(xvar) + x)  # normalize rows to unit var
        self.out = (self.gamma * xhat) + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

