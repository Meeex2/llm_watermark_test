import random


def extract_bits(text, bit_mask, max_len):
    tokens = bit_mask.tokenizer.encode(text)
    bits = []
    for token in tokens:
        if token < bit_mask.vocab_size:
            bits.append(bit_mask.bit_mask[token])
        if len(bits) >= max_len:
            break
    # Pad with zeros if too short
    if len(bits) < max_len:
        bits += [0] * (max_len - len(bits))
    return bits[:max_len]  # Truncate if too long

# --------------------------
# Attack Functions
# --------------------------

def delete_tokens(text, num_deletions=5):
    tokens = text.split()
    if len(tokens) <= num_deletions:
        return ""  # Avoid empty text
    indices = sorted(random.sample(range(len(tokens)), num_deletions))
    for i in reversed(indices):
        del tokens[i]
    return " ".join(tokens)

def insert_tokens(text, filler_words=["the", "and", "of", "to", "in"], num_insertions=5):
    tokens = text.split()
    for _ in range(num_insertions):
        idx = random.randint(0, len(tokens))
        word = random.choice(filler_words)
        tokens.insert(idx, word)
    return " ".join(tokens)

# --------------------------
# Test Watermark Robustness
# --------------------------

def test_watermark_robustness(w_generator, ldpc, bit_mask, prompt="Once upon a time"):
    print("\n--- Original Watermarked Text ---")
    watermarked_text, seed = w_generator.generate(prompt, max_length=32)
    print("Original Text:", watermarked_text)
    
    # Detect original
    x_prime = extract_bits(watermarked_text, bit_mask, ldpc.n)
    is_watermarked = ldpc.decode_watermark(np.array(x_prime))
    print("Detected:", "Yes" if is_watermarked else "No")

    print("\n--- Deletion Attack (5 tokens removed) ---")
    deleted_text = delete_tokens(watermarked_text, num_deletions=5)
    print("Deleted Text:", deleted_text)
    x_prime_del = extract_bits(deleted_text, bit_mask, ldpc.n)
    is_watermarked_del = ldpc.decode_watermark(np.array(x_prime_del))
    print("Detected:", "Yes" if is_watermarked_del else "No")

    print("\n--- Insertion Attack (5 filler tokens added) ---")
    inserted_text = insert_tokens(watermarked_text, num_insertions=5)
    print("Inserted Text:", inserted_text)
    x_prime_ins = extract_bits(inserted_text, bit_mask, ldpc.n)
    is_watermarked_ins = ldpc.decode_watermark(np.array(x_prime_ins))
    print("Detected:", "Yes" if is_watermarked_ins else "No")

    print("\n--- Unwatermarked Text (Baseline) ---")
    input_ids = w_generator.tokenizer.encode(prompt, return_tensors="pt").to(w_generator.model.device)
    unwatermarked_ids = w_generator.model.generate(
        input_ids, max_length=32, pad_token_id=w_generator.tokenizer.pad_token_id
    )
    unwatermarked_text = w_generator.tokenizer.decode(unwatermarked_ids[0])
    print("Unwatermarked Text:", unwatermarked_text)
    x_unwatermarked = extract_bits(unwatermarked_text, bit_mask, ldpc.n)
    is_unwatermarked = ldpc.decode_watermark(np.array(x_unwatermarked))
    print("Detected:", "Yes" if is_unwatermarked else "No")

# --------------------------
# Run Test
# --------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.config.pad_token_id = tokenizer.pad_token_id

    vocab_size = tokenizer.vocab_size
    bit_mask = BitMask(vocab_size, device=device)
    bit_mask.tokenizer = tokenizer
    ldpc = LDPCCode(n=32, g=16, r=16, t=3, eta=0.05)
    w_generator = WatermarkedGenerator(model, tokenizer, ldpc, bit_mask)

    # Calibrate threshold
    print("Calibrating threshold...")
    threshold = calibrate_threshold(ldpc, bit_mask, w_generator)
    ldpc.threshold = threshold

    # Run robustness test
    test_watermark_robustness(w_generator, ldpc, bit_mask)
