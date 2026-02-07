//! PCG32 pseudorandom number generator (PCG-XSH-RR).
//!
//! Must produce bit-identical output to the Python implementation
//! in engine/prng.py for the same (seed, seq) pair.

const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, seq: u64) -> Self {
        let inc = (seq << 1) | 1;
        let mut rng = Pcg32 { state: 0, inc };
        rng.advance();
        rng.state = rng.state.wrapping_add(seed);
        rng.advance();
        rng
    }

    fn advance(&mut self) {
        self.state = self
            .state
            .wrapping_mul(MULTIPLIER)
            .wrapping_add(self.inc);
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.advance();
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << (rot.wrapping_neg() & 31))
    }

    pub fn next_float(&mut self) -> f64 {
        self.next_u32() as f64 / (u32::MAX as f64 + 1.0)
    }

    pub fn next_int(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.next_u32() % (hi - lo + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_values() {
        let mut rng = Pcg32::new(42, 54);
        let expected: [u32; 5] = [
            0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293,
            0xbfa4784b,
        ];
        for exp in expected {
            assert_eq!(rng.next_u32(), exp);
        }
    }

    #[test]
    fn float_range() {
        let mut rng = Pcg32::new(1, 0);
        for _ in 0..1000 {
            let f = rng.next_float();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn int_range() {
        let mut rng = Pcg32::new(1, 0);
        for _ in 0..1000 {
            let v = rng.next_int(3, 7);
            assert!((3..=7).contains(&v));
        }
    }
}
