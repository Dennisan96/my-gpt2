import torch.nn as nn 
import torch
from torch.nn import functional as F
from dataclasses import dataclass

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE = 1.0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x) # (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=-1) # (B, T, 3 * n_embd) -> (B, T, n_embd) x 3
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, T, n_embd // n_head)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, T ,n_embd // n_head)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, n_head, T, n_embd // n_head)
        
        # attn_weights = (q @ k.transpose(-2, -1)) / (self.n_embd // self.n_head) ** 0.5 # (B, n_head, T, T)
        # # this is where we implement the causal mask, which disallows attention to future tokens
        # # it is difference from original paper, which uses allows tokens to attend to future tokens
        # attn_weights = attn_weights.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # (B, n_head, T, T)
        # attn_weights = F.softmax(attn_weights, dim=-1) # (B, n_head, T, T)
        # attn = attn_weights @ v # (B, n_head, T, n_embd // n_head)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE = 1.0
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # self.attn = nn.MultiheadAttention(config.n_embd, config.n_head)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    # n_positions: int = 1024
    block_size: int = 1024 # max sequence length
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    bias: bool = True
    dropout: float = 0.0
    # layer_norm_epsilon: float = 1e-5
    # n_inner: int = 3072
    # activation_function: str = "gelu"
    # resid_pdrop: float = 0.1
    # embd_pdrop: float = 0.1
    # attn_pdrop: float = 0.1
    # initializer_range: float = 0.02

class GPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = GPTConfig()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))

        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
        
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos) # (T, n_embd)
        tok_emb = self.transformer['wte'](idx) # (B, T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        loss = None
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
num_return_sequences = 5
max_length = 30

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
# device = 'cpu'
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval() 
model.to(device)
# model = torch.compile(model)

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"tokens: {len(self.tokens)}")
        print(f"1 epoch: {len(self.tokens) // (B * T)} batches")

        self.current_position = 0 

    def next_batch(self):
        buf = self.tokens[self.current_position:self.current_position+self.B*self.T+1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T
        if self.current_position >= len(self.tokens):
            self.current_position = 0
        return x, y

trainer_loader = DataLoaderLite(4, T=32)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = trainer_loader.next_batch()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x.to(device), y.to(device))    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

quit()

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    x.size() # (B, T)
    logits = model(x) # (B, T, vocab_size)
    logits = logits[:, -1, :] # (B, vocab_size)
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B, 50), (B, 50)
    ix = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
    xcol = topk_indices.gather(dim=1, index=ix) # (B, 1)
    x = torch.cat((x, xcol), dim=1) # (B, T)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"sample {i+1}: {decoded}")