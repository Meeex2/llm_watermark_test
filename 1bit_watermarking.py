import numpy as np
import galois

# Step 1: LDPC Setup
GF2 = galois.GF(2)
n, g, t, r = 100, 50, 3, 50

# Generate sparse parity-check matrix P
P = GF2.Zeros((r, n))
for row in P:
    indices = np.random.choice(n, size=t, replace=False)
    row[indices] = 1

# Compute generator matrix G (null space of P)
null_space = P.null_space()
G = null_space.T  # Shape (n, g) = (100, 50)

# Step 2: Generate pseudorandom codeword x with Bernoulli noise
s = GF2.Random(g)  # Secret vector
e_np = np.random.binomial(1, p=0.1, size=n)  # Noise vector η=0.1
e = GF2(e_np)
x = G @ s + e

# Step 3/4: Generate watermarked text (fixed)
def generate_watermarked(prompt, x):
    t = []
    for i in range(len(x)):
        p_i = 0.5  # Assume model's probability is 0.5 (maximal bias)
        x_i = int(x[i])  # Convert GF2 element to 0/1 integer
        
        if p_i <= 0.5:
            prob = 2 * x_i * p_i  # e.g., 0 or 1.0 if p_i=0.5
        else:
            prob = 1 - 2 * (1 - x_i) * (1 - p_i)
        
        t_i = np.random.binomial(1, prob)
        t.append(t_i)
    return GF2(t)  # Convert final list to GF2 array

watermarked_text = generate_watermarked("Prompt", x)

# Step 5: Detect watermark
def detect_watermark(t, P, threshold):
    syndrome = P @ t
    weight = np.sum(syndrome)
    return weight < threshold

threshold = (0.5 - r**-0.25) * r
is_watermarked = detect_watermark(watermarked_text, P, threshold)
print("Watermark detected:", is_watermarked)
