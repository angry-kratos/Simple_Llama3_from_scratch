
# Self-Attention and Multi-Head Self-Attention in Detail

Self-attention and multi-head self-attention are key mechanisms in transformer models, including large language models (LLMs) like Llama 3. These mechanisms allow the model to weigh the importance of different words in a sequence, enabling it to understand context and relationships effectively.

## Self-Attention

Self-attention, also known as scaled dot-product attention, is a mechanism that allows each word in a sequence to focus on other words in the sequence. This helps the model capture dependencies between words, regardless of their distance from each other.

### Key Components: Q, K, and V

- **Query (Q)**: A vector that represents the current word we are focusing on.
- **Key (K)**: A vector that represents each word in the sequence.
- **Value (V)**: A vector that also represents each word in the sequence, used to construct the final output.

### Steps in Self-Attention

1. **Linear Transformations**:
   - For each word in the input sequence, we create Query, Key, and Value vectors by multiplying the word embedding by three learned matrices \( W_Q \), \( W_K \), and \( W_V \).

2. **Calculate Scores**:
   - We calculate a score for each pair of words by taking the dot product of the Query vector of the current word with the Key vectors of all words.
   - This gives us a measure of similarity or relevance between the current word and all other words.

3. **Scale Scores**:
   - The scores are scaled by dividing by the square root of the dimension of the Key vectors (\( \sqrt{d_k} \)). This step stabilizes the gradients during training.

4. **Softmax**:
   - We apply the softmax function to the scaled scores to obtain attention weights. The weights represent the importance of each word in the sequence relative to the current word.

5. **Weighted Sum**:
   - We multiply the Value vectors by the attention weights and sum them up. This gives us a new representation for the current word that incorporates information from all other words.

### Mathematical Formulation

Given:
- Input sequence: \( X \in \mathbb{R}^{n 	imes d} \) (where \( n \) is the sequence length and \( d \) is the embedding dimension)
- Learned weight matrices: \( W_Q, W_K, W_V \in \mathbb{R}^{d 	imes d_k} \)

The steps are:
1. \( Q = XW_Q \)
2. \( K = XW_K \)
3. \( V = XW_V \)
4. \( 	ext{Scores} = rac{QK^T}{\sqrt{d_k}} \)
5. \( 	ext{Attention Weights} = 	ext{softmax}(	ext{Scores}) \)
6. \( 	ext{Output} = 	ext{Attention Weights} \cdot V \)

### Code Example (Simplified)

```python
import torch
import torch.nn.functional as F

def self_attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights

# Example usage
query = torch.randn(10, 5, 64)  # (batch_size, sequence_length, embedding_dim)
key = torch.randn(10, 5, 64)
value = torch.randn(10, 5, 64)

output, attention_weights = self_attention(query, key, value)
```

## Multi-Head Self-Attention

Multi-head self-attention is an extension of self-attention that allows the model to focus on different parts of the sequence simultaneously. It achieves this by using multiple sets of Query, Key, and Value weight matrices, known as heads.

### Benefits of Multi-Head Self-Attention

1. **Captures Different Relationships**: Each head can focus on different aspects of the relationships between words.
2. **Improves Expressiveness**: By combining the outputs from multiple heads, the model can represent more complex patterns and dependencies.

### Steps in Multi-Head Self-Attention

1. **Linear Transformations**:
   - For each head, we create separate Query, Key, and Value vectors by multiplying the word embedding by different learned matrices.
   
2. **Self-Attention Calculation**:
   - We apply the self-attention mechanism independently for each head.
   
3. **Concatenation**:
   - We concatenate the outputs from all heads.
   
4. **Final Linear Transformation**:
   - We apply a final linear transformation to the concatenated outputs to produce the final result.

### Mathematical Formulation

Given:
- Number of heads: \( h \)
- Input sequence: \( X \in \mathbb{R}^{n 	imes d} \)
- Learned weight matrices for each head: \( W_Q^i, W_K^i, W_V^i \in \mathbb{R}^{d 	imes d_k} \), and \( W_O \in \mathbb{R}^{h \cdot d_v 	imes d} \)

The steps are:
1. For each head \( i \):
   - \( Q_i = XW_Q^i \)
   - \( K_i = XW_K^i \)
   - \( V_i = XW_V^i \)
   - \( 	ext{Scores}_i = rac{Q_iK_i^T}{\sqrt{d_k}} \)
   - \( 	ext{Attention Weights}_i = 	ext{softmax}(	ext{Scores}_i) \)
   - \( 	ext{Output}_i = 	ext{Attention Weights}_i \cdot V_i \)
   
2. Concatenate the outputs from all heads: \( 	ext{Concat}(	ext{Output}_1, 	ext{Output}_2, ..., 	ext{Output}_h) \)
3. Apply the final linear transformation: \( 	ext{Output} = 	ext{Concat} \cdot W_O \)

### Code Example (Simplified)

```python
import torch
import torch.nn.functional as F

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)
        self.linear_out = torch.nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.linear_out(output)
        
        return output, attention_weights

# Example usage
x = torch.randn(10, 5, 64)  # (batch_size, sequence_length, embedding_dim)
mhsa = MultiHeadSelfAttention(d_model=64, num_heads=8)
output, attention_weights = mhsa(x)
```

### Explanation

- **Initialization**: The `MultiHeadSelfAttention` class is initialized with the model dimension (`d_model`) and the number of heads (`num_heads`).
- **Linear Transformations**: The input `x` is linearly transformed to create Query, Key, and Value vectors for each head.
- **Reshape for Multi-Head**: The vectors are reshaped and transposed to separate the heads.
- **Self-Attention for Each Head**: Self-attention is applied independently for each head.
- **Concatenate and Final Linear Transformation**: The outputs from all heads are concatenated and passed through a final linear layer to produce the final output.

## Conclusion

Self-attention and multi-head self-attention are powerful mechanisms that enable LLMs to understand and generate text by focusing on different parts of the input sequence. By using Query, Key, and Value vectors, these mechanisms allow the model to capture complex dependencies and relationships between words, making them essential components of transformer models.
