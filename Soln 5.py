import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim % self.num_heads == 0
        
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attention_weights = F.softmax(scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return attention_output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        attention_output = self.self_attention(x)
        x = self.norm1(x + attention_output)
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + feed_forward_output)
        return x

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, num_layers):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
# Define the parameters
embed_dim = 256
num_heads = 8
hidden_dim = 512
num_layers = 4

# Create the Transformer model
transformer = Transformer(embed_dim, num_heads, hidden_dim, num_layers)

# Generate dummy input
batch_size = 10
seq_len = 20
input_data = torch.randn(batch_size, seq_len, embed_dim)

# Forward pass
output = transformer(input_data)

print("Output shape:", output.shape)
