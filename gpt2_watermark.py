import numpy as np
import torch
import torch.nn.functional as F
import galois
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --------------------------
# LDPC Code Utilities
# --------------------------
class LDPCCode:
    def __init__(self, n=32, g=16, r=16, t=3, eta=0.05):
        self.n, self.g, self.r, self.t, self.eta = n, g, r, t, eta
        self.P, self.G = self._generate_ldpc()

    def _generate_ldpc(self):
        """Generate sparse parity-check matrix P and random generator matrix G from its null space."""
        # Step 1: Generate sparse P
        P = np.zeros((self.r, self.n), dtype=int)
        for i in range(self.r):
            ones = np.random.choice(self.n, self.t, replace=False)
            P[i][ones] = 1

        # Step 2: Compute null space basis (columns of B span ker(P))
        GF2 = galois.GF(2)
        P_gf = GF2(P)
        B = P_gf.null_space().T  # B has shape (n, k)

        # Step 3: Randomly sample g vectors from ker(P) to form G
        k = B.shape[1]  # Dimension of null space
        if k < self.g:
            raise ValueError(f"Null space dimension {k} < required g={self.g}; increase n or reduce g.")

        # Generate a random binary matrix R ∈ F2^(k × g) for linear combinations
        R = GF2.Random((k, self.g))

        # Compute G = B * R over GF(2): each column of G is a random codeword in ker(P)
        G = B @ R  # Matrix multiplication over GF(2)

        return P, G.view(np.ndarray)  # Convert back to NumPy array

    def encode_seed(self):
        """Generate a watermarked seed vector x = G*s ⊕ e."""
        s = np.random.randint(0, 2, self.g)
        e = np.random.binomial(1, self.eta, self.n)
        x = (self.G @ s) % 2  # Matrix multiplication over F2
        x = (x + e) % 2  # Add noise
        return x

    def decode_watermark(self, x_prime):
        """Detect watermark via parity-check weight."""
        GF2 = galois.GF(2)
        P_gf = GF2(self.P)
        x_prime_gf = GF2(x_prime)
        syndrome = (P_gf @ x_prime_gf).tolist()
        weight = np.sum(syndrome)
        threshold = self.r / 2 - self.r**0.5  # Empirical threshold
        return 1 if weight < threshold else 0


# --------------------------
# Token-to-Bit Mapping
# --------------------------
class BitMask:
    def __init__(self, vocab_size, device="cpu"):
        self.vocab_size = vocab_size
        self.bit_mask = self._create_bit_mask()
        self.bit_mask_tensor = torch.tensor(self.bit_mask, dtype=torch.bool, device=device)

    def _create_bit_mask(self):
        """Split vocabulary into two groups for bit assignment."""
        bit_mask = np.zeros(self.vocab_size, dtype=int)
        for token_idx in range(self.vocab_size):
            bit_mask[token_idx] = token_idx % 2  # Even = 0, Odd = 1
        return bit_mask


# --------------------------
# Watermarked Generation
# --------------------------
class WatermarkedGenerator:
    def __init__(self, model, tokenizer, ldpc, bit_mask):
        self.model = model
        self.tokenizer = tokenizer
        self.ldpc = ldpc
        self.bit_mask = bit_mask  # This must be a BitMask instance

    def _watermarked_sample(self, logits, xi):
        """Sample token with watermark bias based on seed bit xi."""
        probs = F.softmax(logits, dim=-1)

        # Remove batch dimension
        probs_squeezed = probs.squeeze()

        # Use bit_mask_tensor for efficient masking
        p_1 = probs_squeezed[self.bit_mask.bit_mask_tensor].sum().item()

        # Bias token selection based on xi and p_1
        if p_1 <= 0.5:
            b = np.random.binomial(1, 2 * xi * p_1)
        else:
            adjusted_p = 1 - 2 * (1 - xi) * (1 - p_1)
            b = np.random.binomial(1, adjusted_p)

        # Sample token from bit group
        b_mask = self.bit_mask.bit_mask_tensor == b
        probs_masked = probs_squeezed[b_mask]
        probs_masked /= probs_masked.sum()

        # Get token indices that match the bit condition
        token_ids = torch.arange(self.bit_mask.vocab_size, device=logits.device)[b_mask]
        token_id = token_ids[torch.multinomial(probs_masked, num_samples=1)]

        return token_id  # Shape: (1,)

    def generate(self, prompt, max_length=50):
        """Generate watermarked text."""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        generated_text = prompt
        x = self.ldpc.encode_seed()  # Generate seed

        for i in range(min(max_length, len(x))):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[:, -1, :]
                xi = x[i]
                token_id = self._watermarked_sample(logits, xi)

                # Convert token_id from (1,) → (1, 1) for concatenation
                input_ids = torch.cat([input_ids, token_id.unsqueeze(0)], dim=1)
                generated_text += self.tokenizer.decode(token_id)

        return generated_text, x


# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Fix attention mask warning
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    vocab_size = tokenizer.vocab_size
    bit_mask = BitMask(vocab_size, device=device)
    ldpc = LDPCCode(n=32, g=16, r=16, t=3, eta=0.05)  # Lower noise for better detection

    w_generator = WatermarkedGenerator(model, tokenizer, ldpc, bit_mask)

    # Generate watermarked text
    prompt = "Once upon a time"
    print("Generating watermarked text...")
    watermarked_text, seed = w_generator.generate(prompt, max_length=32)
    print("Watermarked Text:", watermarked_text)

    # Detect watermark
    print("Detecting watermark...")
    x_prime_bits = []
    tokens = tokenizer.encode(watermarked_text)
    for token in tokens:
        if token < vocab_size:
            x_prime_bits.append(bit_mask.bit_mask[token])
    x_prime = np.array(x_prime_bits[:len(seed)])
    if len(x_prime) < len(seed):
        x_prime = np.pad(x_prime, (0, len(seed) - len(x_prime)), constant_values=0)
    is_watermarked = ldpc.decode_watermark(x_prime)
    print("Watermarked?", "Yes" if is_watermarked else "No")

    # Generate unwatermarked text (baseline)
    print("\nGenerating unwatermarked text...")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    unwatermarked_ids = model.generate(
        input_ids,
        max_length=32,
        pad_token_id=tokenizer.pad_token_id
    )
    unwatermarked_text = tokenizer.decode(unwatermarked_ids[0])
    print("Unwatermarked Text:", unwatermarked_text)
