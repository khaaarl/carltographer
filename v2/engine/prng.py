"""PCG32 pseudorandom number generator.

Implements the PCG-XSH-RR variant (32-bit output, 64-bit state).
Reference: https://www.pcg-random.org/
"""

from __future__ import annotations


class PCG32:
    _MASK32 = 0xFFFFFFFF
    _MASK64 = 0xFFFFFFFFFFFFFFFF
    _MUL = 6364136223846793005

    def __init__(self, seed: int, seq: int = 0) -> None:
        self._state: int = 0
        self._inc: int = ((seq << 1) | 1) & self._MASK64
        self._advance()
        self._state = (self._state + seed) & self._MASK64
        self._advance()

    def _advance(self) -> None:
        self._state = (self._state * self._MUL + self._inc) & self._MASK64

    def next_u32(self) -> int:
        old = self._state
        self._advance()
        xorshifted = (((old >> 18) ^ old) >> 27) & self._MASK32
        rot = (old >> 59) & 31
        return (
            (xorshifted >> rot) | (xorshifted << ((-rot) & 31))
        ) & self._MASK32

    def next_float(self) -> float:
        """Uniform float in [0, 1)."""
        return self.next_u32() / (self._MASK32 + 1)

    def next_int(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi] inclusive."""
        return lo + self.next_u32() % (hi - lo + 1)
