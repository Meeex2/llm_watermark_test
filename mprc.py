import numpy as np
import sympy as sp
from sympy import Matrix
import math
from scipy.special import ndtri, ndtr
 
class PRC:
    """
    Pseudo-Random error-correcting Code
    (c) tanguy.urvoy@Orange.com 2025
    """
 
    def __init__(
        self,
        N: int = 64,
        R: int = 1024,
        M: int = 64,
        p: int = 2,
        nu: float = 0.01,
        seed: int = 0xADE1EE3111E,
        eps:np.float64 = 0
    ):
        """Pseudo-Random Error-Correcting Code
 
            Generate two matrices : P and G on finite field Z/pZ
            - P is the private key
            - G is the public key
 
            Then Code or Decode a one-bit Watermak in the pseudo random generator.
 
        c.f.
        https://link.springer.com/chapter/10.1007/978-3-031-68391-6_10
 
        Args:
            R (int): Parity-Check space dimension. Defaults to 32.
            N (int): Binary Code length. Defaults to 64.
            M (int): Input noise dimension. Defaults to 256.
            p (int, optional): Finite field size (prime number). Defaults to 2.
            nu (float, optional): Sparse Bernoulli noise. Defaults to 0.05.
            seed (int, optional): Random generator seed. Defaults to 0xAde1eE3111e.
        """
        self.rng = np.random.default_rng(seed)
        assert N > 1 and R>1 and M>1 and p>=2 and nu>=0
        self.p = p
        self.nu = nu
        self.P, self.G = self.sample_PRC(R, N, M)
        self.eps = np.float64(eps)
 
    def get_params(self):
        return {
            "N": self.G.shape[0],
            "R": self.P.shape[0],
            "M": self.G.shape[1],
            "p": self.p,
            "nu": self.nu,
        }
 
    def get_public_key(self):
        return self.G
 
    def get_private_key(self):
        return self.P
 
    def sample_parity_check_matrix(self, R: int, N: int) -> Matrix:
        """Sparse parity-check matrix
 
        Args:
            R (int): Parity-Check space dimension (i.e. Matrix Rank).
            N (int): Binary Code length.
 
        Returns:
            Matrix: RxN Matrix
        """
        #assert R <= N, f"Rank {R=} must be lower or equal to {N=}"
        p = self.p
        P = np.zeros((R, N), dtype=int)
        offsets = np.arange(N)
        self.rng.shuffle(offsets)
        for i in range(R):
            P[i, offsets[i%N]] = 1 + self.rng.integers(p - 1)
            P[i, (offsets[i%N] + self.rng.integers(N - 1)) % N] = 1 + self.rng.integers(
                p - 1
            )
        return Matrix(P)
 
    def sample_code_matrix(self, P: Matrix, M: int, N: int) -> Matrix:
        """Sample Code matrix in P nullspace
 
        Args:
            P (Matrix): RxN sparse parity-check matrix.
            M (int): Input noise dimension.
            N (int): Binary Code length.
 
        Returns:
            Matrix: Noise encoding matrix
        """
        p = self.p
        KerP = P.nullspace(simplify=True, iszerofunc=(lambda x: x % p == 0))
        dker = len(KerP)
 
        Zmat = sp.zeros(N, dker)
        for i in range(dker):
            Zmat[i] = KerP[i] % p
 
        coefs = self.rng.integers(0, p, dker * M, endpoint=True, dtype=int)
        coefs = Matrix(dker, M, coefs.tolist())
 
        return (Zmat * coefs) % p
 
    def sample_vector(self, N: int) -> Matrix:
        """sample a uniformly random vector in finite field
 
        Args:
            N (int): Dimension of the vector
 
        Returns:
            Matrix: A vector
        """
        coefs = self.rng.integers(0, self.p - 1, N, endpoint=True, dtype=int)
        return Matrix(N, 1, coefs.tolist())
 
    def sample_PRC(self, R: int, N: int, M: int):
        """Sample a couple of matrices : P the backdoor and G the encoding matrix
 
        Args:
            R (int): Parity-Check space dimension.
            N (int): Binary Code length.
            M (int): Input noise dimension.
 
        Returns:
            (Matrix,Matrix): A couple of matrices.
        """
        P = self.sample_parity_check_matrix(R=R, N=N)
        G = self.sample_code_matrix(P=P, M=M, N=N)
        return P, G
 
    def encode(self, bit: bool) -> Matrix:
        """Encode one bit into a pseudo-random binary code
 
        Args:
            bit (bool): A boolean
 
        Returns:
            Matrix: A Watermarked pseudo-random vector (Matrix of shape 1xD)
        """
 
        if bit:
            s = self.sample_vector(self.G.shape[1])
            noise = self.rng.binomial(1, self.nu, self.G.shape[0])
            E = Matrix(self.G.shape[0], 1, noise.tolist())
            return ((self.G * s) + E) % self.p
        else:
            return self.sample_vector(self.G.shape[0])
 
    def sparsity_threshold(self) -> float:
        """return theoretical threshold for sparity check
 
        Returns:
            float: Minimum sparsity for Watermark Detection
        """
        R = self.P.shape[0]
        return 0.5 + R ** (-0.25)
 
    def sparsity(self, x: Matrix) -> float:
        """Retrun a True bit Watermarking probability score
 
        Args:
            x (Matrix): A (vertical) Vector of pseudo-random bits
 
        Returns:
            float: a True bit Watermarking probability score
        """
        D = (self.P * x) % self.p
        wt = np.count_nonzero(D)
        spt = 1.0 - float(wt) / D.shape[0]
        return spt
 
    def decode(self, x: Matrix) -> bool:
        """Retrun a True bit Watermarking probability score
 
        Args:
            x (Matrix): A (vertical) Vector of pseudo-random bits
 
        Returns:
            float: a True bit Watermarking probability score
        """
        spt = self.sparsity(x)
        return spt > self.sparsity_threshold()
 
    def code_to_int(self, code) -> int:
        assert (
            self.G.shape[0] <= 64
            ), f"For integer to code translation code length should be lower than 64 but {self.G.shape[0]} is not"
        c: int = 0
        for i in range(len(code.T)):
            c = c * self.p
            c = c + (code.T[i] % self.p)
 
        return c
   
    def int_to_code(self, c: int) -> Matrix:
        assert (
            self.G.shape[0] <= 64
            ), f"For integer to code translation code length should be lower than 64 but {self.G.shape[0]} is notfor integer to code translation"
        co = list()
        for i in range(self.G.shape[0]):
            co.append(c % self.p)
            c = c // self.p
        co.reverse()
        assert len(co)==self.G.shape[0], f"len(co) should be {self.G.shape[0]} instead of {len(co)}"
        return Matrix(self.G.shape[0],1,co)
   
 
    def code_to_ufloat(self, code) -> np.float64:
        c = self.code_to_int(code)
        #c = np.float64(c)
        return np.float64(c / 2 ** self.G.shape[0])
 
    def ufloat_to_code(self, u: np.float64) -> Matrix:
        c = int(u * (2 ** self.G.shape[0]))
        return self.int_to_code(c)
 
    def encode_int(self, bit: bool) -> int:
        v = self.encode(bit)
        return self.code_to_int(v)
 
    def decode_int(self, i: int) -> bool:
        v = self.int_to_code(i)
        return self.decode(v)
 
    def encode_ufloat(self, bit: bool) -> np.float64:
        v = self.encode(bit)
        return self.code_to_ufloat(v)
 
    def decode_ufloat(self, f: np.float64) -> bool:
        v = self.ufloat_to_code(f)
        return self.decode(v)
   
    def encode_gfloat(self, bit:bool) -> np.float64:
        """encode a bit as a gaussian sample
 
        Args:
            bit (bool): input bit
 
        Returns:
            float: a pseudo-random gaussian sample
        """        
        g = self.encode_ufloat(bit)
        g = g * (1.0 - 2*self.eps)
        g = g + self.eps
        return ndtri(g)
       
    def decode_gfloat(self, f:np.float64) -> bool:
        """decode a pseudo-random gaussian sample
 
        Args:
            f (float): a pseudo-random gaussian sample
 
        Returns:
            bool: True if the sample was watermarked
        """        
        u = ndtr(f)
        u = u - self.eps
        u = u / (1.0 - 2*self.eps)
        return self.decode_ufloat(u)
   
   
 
   
from sympy import Matrix
import math
class MPRC(PRC):
    _ALLOWED_SYMBOLS_ = "abcdefghijklmnopqrstuvwxyz 01234"  # 32 symbols → 5 bits per character
    def encode(self, msg: str) -> Matrix:
        """Encode a message into a watermarked pseudo-random vector."""
        bits = []
        for char in msg:
            assert char in self._ALLOWED_SYMBOLS_, f"Invalid character: {char}"
            idx = self._ALLOWED_SYMBOLS_.index(char)
            bin_str = format(idx, "05b")  # 5-bit binary string
            for bit_str in bin_str:
                bits.append(int(bit_str))
        # Encode each bit using PRC and concatenate
        encoded_parts = []
        for bit in bits:
            vec = super().encode(bit)  # Returns Sympy Matrix (N x 1)
            encoded_parts.append(vec)
        # Flatten into a single column vector
        full_code = Matrix.zeros(len(encoded_parts) * encoded_parts[0].shape[0], 1)
        idx = 0
        for vec in encoded_parts:
            for i in range(vec.shape[0]):
                full_code[idx] = vec[i]
                idx += 1
        return full_code
    def decode(self, code_vector: Matrix) -> str:
        """Decode a watermarked vector back into the original message."""
        N = self.G.shape[0]  # Code length per bit
        total_length = code_vector.shape[0]
        assert total_length % N == 0, "Invalid code vector length."
        # Split into N-length segments and decode each bit
        num_bits = total_length // N
        bits = []
        for i in range(num_bits):
            start = i * N
            end = start + N
            segment = code_vector[start:end]  # This is already a Matrix
            bit = super().decode(segment)     # Pass Matrix to super().decode
            bits.append(1 if bit else 0)
        # Convert bits back to message
        msg = ""
        for i in range(0, len(bits), 5):
            if i + 5 > len(bits):
                break  # Skip incomplete group
            group = bits[i:i+5]
            group_str = "".join(str(b) for b in group)
            char_idx = int(group_str, 2)
            msg += self._ALLOWED_SYMBOLS_[char_idx]
        return msg
        
# Initialize MPRC with default parameters
mprc = MPRC()
# Encode a message
message = "hello world 123"
watermarked_data = mprc.encode(message)
# Decode the watermarked data
decoded_message = mprc.decode(watermarked_data)
print(f"Decoded message: {decoded_message}")
