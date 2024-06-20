from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import json
import matplotlib.pyplot as plt

class Tokenizer:
    """
    Tokenizer class to handle tokenization and detokenization of text.
    """

    def __init__(self, tokenizer_path):
        """
        Initialize the Tokenizer with the tokenizer path.
        """
        self.tokenizer_path = tokenizer_path
        self.special_tokens = [
            "<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", 
            "<|reserved_special_token_1|>", "<|reserved_special_token_2|>", 
            "<|reserved_special_token_3|>", "<|start_header_id|>", 
            "<|end_header_id|>", "<|reserved_special_token_4|>", "<|eot_id|>"
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
        self.tokenizer = self._load_tokenizer()

        
    def _load_tokenizer(self):
        """
        Load the tokenizer using the tiktoken library.
        """
        mergeable_ranks = load_tiktoken_bpe(self.tokenizer_path)
        return tiktoken.Encoding(
            name=Path(self.tokenizer_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(self.special_tokens)},
        )

    def encode(self, text):
        """
        Encode text into tokens.
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        """
        Decode tokens back into text.
        """
        try:
            decoded_text = self.tokenizer.decode(tokens)
        except Exception as e:
            print(f"Error decoding tokens: {tokens}")
            raise e
        return decoded_text


class ModelLoader:
    """
    ModelLoader class to load the model and configuration files.
    """

    def __init__(self, model_path, config_path):
        """
        Initialize the ModelLoader with the model and config paths.
        """
        self.model = torch.load(model_path)
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """
        Load the model configuration from a JSON file.
        """
        with open(config_path, "r") as f:
            return json.load(f)

    def get_model(self):
        """
        Get the loaded model.
        """
        return self.model

    def get_config(self):
        """
        Get the loaded configuration.
        """
        return self.config


class Embedding:
    """
    Embedding class to handle token embeddings.
    """

    def __init__(self, model, config):
        """
        Initialize the Embedding with the model and configuration.
        """
        self.model = model
        self.dim = config["dim"]
        self.vocab_size = config["vocab_size"]
        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.dim)
        self.embedding_layer.weight.data.copy_(self.model["tok_embeddings.weight"])

    def get_embeddings(self, tokens):
        """
        Get the embeddings for the given tokens.
        """
        tokens_tensor = torch.tensor(tokens)
        return self.embedding_layer(tokens_tensor).to(torch.bfloat16)


class Encoding(Tokenizer):
    """
    Encoding class to handle prompt encoding, inherits from Tokenizer.
    """

    def __init__(self, tokenizer_path, prompt):
        """
        Initialize the Encoding with the tokenizer path and prompt.
        """
        super().__init__(tokenizer_path)  # Initialize the parent class
        self.prompt = prompt

    def enc(self):
        """
        Encode the prompt into tokens.
        """
        tokens = [12800] + self.encode(self.prompt)
        return torch.tensor(tokens)


class Attention:
    """
    Attention class to handle the self-attention mechanism.
    """

    def __init__(self, model, config, token_embeddings_unnormalized):
        """
        Initialize the Attention with the model, configuration, and unnormalized token embeddings.
        """
        self.model = model
        self.config = config
        self.token_embeddings_unnormalized = token_embeddings_unnormalized
        self.n_heads = config["n_heads"]
        self.dim = config["dim"]
        self.n_kv_heads = config["n_kv_heads"]
        self.freqs_cis = self._get_freqs_cis(len(token_embeddings_unnormalized))
        self.norm_eps = config["norm_eps"]

    def rms_norm(self, tensor, norm_weights):
        """
        Apply RMS normalization to a tensor.
        """
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)) * norm_weights

    def _get_freqs_cis(self, length):
        """
        Compute the rotary positional encodings.
        """
        zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64
        freqs = 1.0 / (self.config["rope_theta"] ** zero_to_one_split_into_64_parts)
        freqs_for_each_token = torch.outer(torch.arange(length), freqs)
        return torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

    def attn(self, layer_embedding_norm, layer):
        """
        Compute the attention mechanism for a specific layer.
        """
        qkv_attention_store = []

        q_layer = self.model[f"layers.{layer}.attention.wq.weight"]
        q_layer = q_layer.view(self.n_heads, q_layer.shape[0] // self.n_heads, self.dim)
        k_layer = self.model[f"layers.{layer}.attention.wk.weight"]
        k_layer = k_layer.view(self.n_kv_heads, k_layer.shape[0] // self.n_kv_heads, self.dim)
        v_layer = self.model[f"layers.{layer}.attention.wv.weight"]
        v_layer = v_layer.view(self.n_kv_heads, v_layer.shape[0] // self.n_kv_heads, self.dim)
        w_layer = self.model[f"layers.{layer}.attention.wo.weight"]

        for head in range(self.n_heads):
            q_layer_head = q_layer[head]
            k_layer_head = k_layer[head // 4]
            v_layer_head = v_layer[head // 4]

            q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
            k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
            v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

            q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
            q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
            q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * self.freqs_cis)
            q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

            k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
            k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
            k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * self.freqs_cis)
            k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

            qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128) ** 0.5
            mask = torch.full((len(self.token_embeddings_unnormalized), len(self.token_embeddings_unnormalized)), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            qk_per_token_after_masking = qk_per_token + mask
            qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
            qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
            qkv_attention_store.append(qkv_attention)

        stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
        embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
        return embedding_delta


class FeedForward:
    """
    FeedForward class to handle the feed-forward network.
    """

    def __init__(self, model, config, token_embeddings_unnormalized):
        """
        Initialize the FeedForward with the model, configuration, and unnormalized token embeddings.
        """
        self.model = model
        self.config = config
        self.token_embeddings_unnormalized = token_embeddings_unnormalized
        self.n_layers = config["n_layers"]
        self.dim = config["dim"]
        self.norm_eps = config["norm_eps"]

    def rms_norm(self, tensor, norm_weights):
        """
        Apply RMS normalization to a tensor.
        """
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + self.norm_eps)) * norm_weights

    def feed_forward(self):
        """
        Apply the feed-forward network to the token embeddings.
        """
        final_embedding = self.token_embeddings_unnormalized
        attention = Attention(self.model, self.config, self.token_embeddings_unnormalized)

        for layer in range(self.n_layers):
            layer_embedding_norm = self.rms_norm(final_embedding, self.model[f"layers.{layer}.attention_norm.weight"])
            embedding_delta = attention.attn(layer_embedding_norm, layer)
            embedding_after_edit = final_embedding + embedding_delta
            embedding_after_edit_normalized = self.rms_norm(embedding_after_edit, self.model[f"layers.{layer}.ffn_norm.weight"])

            w1 = self.model[f"layers.{layer}.feed_forward.w1.weight"]
            w2 = self.model[f"layers.{layer}.feed_forward.w2.weight"]
            w3 = self.model[f"layers.{layer}.feed_forward.w3.weight"]

            output_after_feedforward = torch.matmul(
                torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T),
                w2.T
            )
            final_embedding = embedding_after_edit + output_after_feedforward

        return final_embedding


class Llama3:
    """
    Llama3 class to handle the entire process of tokenization, embedding, attention, and decoding.
    """

    def __init__(self, tokenizer_path, model_path, config_path, prompt):
        """
        Initialize the Llama3 model with the tokenizer path, model path, config path, and prompt.
        """
        self.tokenizer = Tokenizer(tokenizer_path)
        self.model_loader = ModelLoader(model_path, config_path)
        self.model = self.model_loader.get_model()
        self.config = self.model_loader.get_config()
        self.prompt = prompt
        self.encoded_prompt = self._encode_prompt()
        self.embeddings = self._get_embeddings()
        self.final_embedding = self._apply_feed_forward()
        self.next_token = self._get_next_token()

    def _encode_prompt(self):
        """
        Encode the prompt using the Encoding class.
        """
        encoder = Encoding(self.tokenizer.tokenizer_path, self.prompt)
        return encoder.enc()

    def _get_embeddings(self):
        """
        Get the embeddings for the encoded prompt.
        """
        embedding = Embedding(self.model, self.config)
        return embedding.get_embeddings(self.encoded_prompt)

    def _apply_feed_forward(self):
        """
        Apply the feed-forward network to the embeddings.
        """
        feed_forward = FeedForward(self.model, self.config, self.embeddings)
        return self._rms_norm(feed_forward.feed_forward(), self.model["norm.weight"])

    def _rms_norm(self, tensor, norm_weights):
        """
        Apply RMS normalization to a tensor.
        """
        eps = self.config["norm_eps"]
        return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + eps)) * norm_weights

    def _get_next_token(self):
        """
        Get the next token from the final embeddings.
        """
        logits = torch.matmul(self.final_embedding[-1], self.model["output.weight"].T)
        return torch.argmax(logits, dim=-1)

    def decode_next_token(self):
        """
        Decode the next token to text.
        """
        return self.tokenizer.decode([self.next_token.item()])


# Example usage of the Llama3 class
llama3 = Llama3(
    tokenizer_path="Meta-Llama-3-8B/tokenizer.model",
    model_path="Meta-Llama-3-8B/consolidated.00.pth",
    config_path="Meta-Llama-3-8B/params.json",
    prompt="the answer to the ultimate question of life, the universe, and everything is "
)

# Print the decoded next token
print(llama3.decode_next_token())
