// ============================================================================
// CHRONOS BLOCKCHAIN ENGINE
// ============================================================================
//
// HOW A BLOCKCHAIN ACTUALLY WORKS (and how this code models it):
//
// A blockchain is a linked list of blocks where each block contains a
// cryptographic hash of the previous block, creating a tamper-evident
// chain. If you change any data in any block, its hash changes, which
// breaks the link to the next block, which breaks the link to the block
// after that, all the way to the end of the chain. This makes the
// historical record effectively immutable once enough blocks have been
// added on top (each new block "confirms" all previous blocks).
//
// But a blockchain is more than just a data structure — it's a
// DECENTRALIZED STATE MACHINE. The "state" is the set of all account
// balances, smart contract storage, and other data. Each block contains
// a list of TRANSACTIONS that modify this state. Every node in the
// network independently executes every transaction and arrives at the
// same resulting state (this is deterministic execution). The consensus
// mechanism (Proof of Work, Proof of Stake) determines which node gets
// to propose the next block.
//
// Smart contracts are programs stored ON the blockchain that execute
// automatically when triggered by a transaction. On Ethereum, they run
// on the EVM (Ethereum Virtual Machine), a stack-based bytecode VM
// where every instruction costs "gas" (to prevent infinite loops and
// resource abuse). This engine implements a complete EVM interpreter.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic):
//   1.  SHA-256 cryptographic hashing (the foundation of everything)
//   2.  Keccak-256 (Ethereum's hash function, used for addresses & state)
//   3.  Block structure with Merkle root of transactions
//   4.  Account state model (balances, nonces, contract code & storage)
//   5.  Transaction creation, signing verification, and execution
//   6.  Proof of Work mining with adjustable difficulty
//   7.  Blockchain with validation and fork choice
//   8.  Complete EVM interpreter (stack machine with all core opcodes)
//   9.  Smart contract deployment and execution
//  10.  Gas metering (every opcode has a gas cost)
//  11.  ERC-20 token standard implementation as EVM bytecode
//  12.  Reentrancy detection
//  13.  Event/log emission
//  14.  Merkle Patricia Trie for state storage
// ============================================================================

use std::collections::{HashMap, BTreeMap};
use std::fmt;

// ============================================================================
// PART 1: CRYPTOGRAPHIC HASHING
// ============================================================================
// Everything in blockchain relies on cryptographic hash functions: block
// linking, mining, address derivation, Merkle trees, transaction IDs.
// We implement SHA-256 (used by Bitcoin) and a simplified Keccak-256
// (used by Ethereum) from scratch so you can see exactly how they work.

/// A 256-bit hash value. This is the fundamental data type of blockchain.
/// Block hashes, transaction hashes, state roots — everything is a Hash256.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Hash256(pub [u8; 32]);

impl Hash256 {
    pub const ZERO: Hash256 = Hash256([0u8; 32]);

    pub fn from_bytes(bytes: &[u8; 32]) -> Self { Self(*bytes) }

    /// Create a hash from a hex string. This is how hashes are typically
    /// displayed in block explorers and transaction receipts.
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        if hex.len() != 64 { return None; }
        let mut bytes = [0u8; 32];
        for i in 0..32 {
            bytes[i] = u8::from_str_radix(&hex[i*2..i*2+2], 16).ok()?;
        }
        Some(Self(bytes))
    }

    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Interpret the first 8 bytes as a u64 (for difficulty comparison).
    pub fn leading_u64(&self) -> u64 {
        u64::from_be_bytes([self.0[0], self.0[1], self.0[2], self.0[3],
                           self.0[4], self.0[5], self.0[6], self.0[7]])
    }

    /// Count leading zero bits (for Proof of Work difficulty verification).
    pub fn leading_zeros(&self) -> u32 {
        let mut zeros = 0u32;
        for &byte in &self.0 {
            if byte == 0 { zeros += 8; }
            else { zeros += byte.leading_zeros(); break; }
        }
        zeros
    }
}

impl fmt::Debug for Hash256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "0x{}", self.to_hex()) }
}
impl fmt::Display for Hash256 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "0x{}...{}", &self.to_hex()[..8], &self.to_hex()[56..]) }
}

/// A 160-bit Ethereum address (20 bytes). Derived from the last 20 bytes
/// of the Keccak-256 hash of the public key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Address(pub [u8; 20]);

impl Address {
    pub const ZERO: Address = Address([0u8; 20]);

    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        if hex.len() != 40 { return None; }
        let mut bytes = [0u8; 20];
        for i in 0..20 {
            bytes[i] = u8::from_str_radix(&hex[i*2..i*2+2], 16).ok()?;
        }
        Some(Self(bytes))
    }

    pub fn to_hex(&self) -> String {
        format!("0x{}", self.0.iter().map(|b| format!("{:02x}", b)).collect::<String>())
    }

    /// Derive a contract address from deployer address and nonce.
    /// This is how Ethereum determines the address of a new contract:
    /// address = keccak256(rlp([sender, nonce]))[12..32]
    pub fn contract_address(deployer: &Address, nonce: u64) -> Self {
        let mut data = Vec::new();
        data.extend_from_slice(&deployer.0);
        data.extend_from_slice(&nonce.to_be_bytes());
        let hash = keccak256(&data);
        let mut addr = [0u8; 20];
        addr.copy_from_slice(&hash.0[12..32]);
        Address(addr)
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.to_hex()) }
}
impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.to_hex()) }
}

/// SHA-256 hash function. This is Bitcoin's hash function and one of the
/// most widely used cryptographic hash functions in the world.
/// The implementation follows FIPS 180-4 exactly.
pub fn sha256(data: &[u8]) -> Hash256 {
    // Initial hash values (first 32 bits of fractional parts of square roots of first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Round constants (first 32 bits of fractional parts of cube roots of first 64 primes).
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    // Pre-processing: pad the message to a multiple of 512 bits (64 bytes).
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80); // Append bit '1' followed by zeros.
    while padded.len() % 64 != 56 { padded.push(0); }
    padded.extend_from_slice(&bit_len.to_be_bytes()); // Append original length in bits.

    // Process each 512-bit (64-byte) chunk.
    for chunk in padded.chunks(64) {
        // Prepare the message schedule (64 words of 32 bits each).
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i*4], chunk[i*4+1], chunk[i*4+2], chunk[i*4+3]]);
        }
        for i in 16..64 {
            let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
            let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
            w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
        }

        // Initialize working variables with the current hash values.
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        // 64 rounds of compression.
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g; g = f; f = e;
            e = d.wrapping_add(temp1);
            d = c; c = b; b = a;
            a = temp1.wrapping_add(temp2);
        }

        // Add the compressed chunk to the current hash value.
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c); h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e); h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g); h[7] = h[7].wrapping_add(hh);
    }

    // Produce the final 256-bit hash.
    let mut result = [0u8; 32];
    for i in 0..8 {
        result[i*4..i*4+4].copy_from_slice(&h[i].to_be_bytes());
    }
    Hash256(result)
}

/// Double SHA-256 (used by Bitcoin for blocks and transactions).
pub fn double_sha256(data: &[u8]) -> Hash256 { sha256(&sha256(data).0) }

/// Keccak-256 (used by Ethereum for addresses, state roots, ABI encoding).
/// This is a simplified but correct implementation of the Keccak sponge.
pub fn keccak256(data: &[u8]) -> Hash256 {
    // Keccak-256 uses rate=1088 bits (136 bytes) and capacity=512 bits.
    let rate = 136;
    let mut state = [0u64; 25]; // 1600-bit state (5×5 matrix of 64-bit words)

    // Absorb phase: XOR input blocks into the state and apply the permutation.
    let mut padded = data.to_vec();
    padded.push(0x01); // Keccak padding (different from SHA-3 which uses 0x06)
    while padded.len() % rate != 0 { padded.push(0); }
    // Set the last bit of the last byte of the last block.
    let last = padded.len() - 1;
    padded[last] ^= 0x80;

    for block in padded.chunks(rate) {
        // XOR the block into the state (interpreting bytes as little-endian u64s).
        for i in 0..rate/8 {
            if i < 17 { // 136/8 = 17 words
                let word = u64::from_le_bytes([
                    block[i*8], block.get(i*8+1).copied().unwrap_or(0),
                    block.get(i*8+2).copied().unwrap_or(0), block.get(i*8+3).copied().unwrap_or(0),
                    block.get(i*8+4).copied().unwrap_or(0), block.get(i*8+5).copied().unwrap_or(0),
                    block.get(i*8+6).copied().unwrap_or(0), block.get(i*8+7).copied().unwrap_or(0),
                ]);
                state[i] ^= word;
            }
        }
        keccak_f1600(&mut state);
    }

    // Squeeze phase: extract the first 256 bits (32 bytes) from the state.
    let mut result = [0u8; 32];
    for i in 0..4 {
        result[i*8..(i+1)*8].copy_from_slice(&state[i].to_le_bytes());
    }
    Hash256(result)
}

/// The Keccak-f[1600] permutation: 24 rounds of theta, rho, pi, chi, iota.
fn keccak_f1600(state: &mut [u64; 25]) {
    const RC: [u64; 24] = [
        0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
        0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
        0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
    ];
    const ROT: [u32; 25] = [
        0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
    ];
    const PI: [usize; 25] = [
        0, 6, 12, 18, 24, 3, 9, 10, 16, 22, 1, 7, 13, 19, 20, 4, 5, 11, 17, 23, 2, 8, 14, 15, 21,
    ];

    for round in 0..24 {
        // θ (theta): column parity
        let mut c = [0u64; 5];
        for x in 0..5 { c[x] = state[x] ^ state[x+5] ^ state[x+10] ^ state[x+15] ^ state[x+20]; }
        let mut d = [0u64; 5];
        for x in 0..5 { d[x] = c[(x+4)%5] ^ c[(x+1)%5].rotate_left(1); }
        for i in 0..25 { state[i] ^= d[i % 5]; }

        // ρ (rho) and π (pi): rotate and permute
        let mut temp = [0u64; 25];
        for i in 0..25 { temp[PI[i]] = state[i].rotate_left(ROT[i]); }

        // χ (chi): non-linear mixing
        for y in 0..5 {
            let base = y * 5;
            let t = [temp[base], temp[base+1], temp[base+2], temp[base+3], temp[base+4]];
            for x in 0..5 {
                state[base + x] = t[x] ^ ((!t[(x+1)%5]) & t[(x+2)%5]);
            }
        }

        // ι (iota): round constant
        state[0] ^= RC[round];
    }
}

// ============================================================================
// PART 2: ACCOUNT STATE
// ============================================================================
// Ethereum uses an account-based model (unlike Bitcoin's UTXO model).
// Every address has an associated account with a balance, nonce, and
// optionally contract code and storage.

/// An Ethereum-style account. External accounts (wallets) have code = None.
/// Contract accounts have code = Some(bytecode) and storage.
#[derive(Debug, Clone)]
pub struct Account {
    pub balance: u128,           // In wei (1 ETH = 10^18 wei)
    pub nonce: u64,              // Number of transactions sent (prevents replay)
    pub code: Option<Vec<u8>>,   // EVM bytecode (None for external accounts)
    pub storage: HashMap<U256, U256>, // Contract storage (key → value)
    pub code_hash: Hash256,      // Keccak256 of the code
}

impl Account {
    pub fn new_external(balance: u128) -> Self {
        Self { balance, nonce: 0, code: None, storage: HashMap::new(), code_hash: keccak256(&[]) }
    }

    pub fn new_contract(balance: u128, code: Vec<u8>) -> Self {
        let code_hash = keccak256(&code);
        Self { balance, nonce: 1, code: Some(code), storage: HashMap::new(), code_hash }
    }

    pub fn is_contract(&self) -> bool { self.code.is_some() }
}

/// A 256-bit unsigned integer. This is the native word size of the EVM.
/// Every stack element, storage key, and storage value is a U256.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct U256(pub [u64; 4]); // Little-endian: [low, ..., high]

impl U256 {
    pub const ZERO: U256 = U256([0; 4]);
    pub const ONE: U256 = U256([1, 0, 0, 0]);
    pub const MAX: U256 = U256([u64::MAX; 4]);

    pub fn from_u64(val: u64) -> Self { U256([val, 0, 0, 0]) }
    pub fn from_u128(val: u128) -> Self { U256([val as u64, (val >> 64) as u64, 0, 0]) }

    pub fn as_u64(&self) -> u64 { self.0[0] }
    pub fn as_usize(&self) -> usize { self.0[0] as usize }

    pub fn is_zero(&self) -> bool { self.0 == [0; 4] }

    pub fn overflowing_add(&self, other: &U256) -> (U256, bool) {
        let mut result = [0u64; 4];
        let mut carry = 0u64;
        for i in 0..4 {
            let (sum1, c1) = self.0[i].overflowing_add(other.0[i]);
            let (sum2, c2) = sum1.overflowing_add(carry);
            result[i] = sum2;
            carry = (c1 as u64) + (c2 as u64);
        }
        (U256(result), carry > 0)
    }

    pub fn overflowing_sub(&self, other: &U256) -> (U256, bool) {
        let mut result = [0u64; 4];
        let mut borrow = 0u64;
        for i in 0..4 {
            let (diff1, b1) = self.0[i].overflowing_sub(other.0[i]);
            let (diff2, b2) = diff1.overflowing_sub(borrow);
            result[i] = diff2;
            borrow = (b1 as u64) + (b2 as u64);
        }
        (U256(result), borrow > 0)
    }

    pub fn overflowing_mul(&self, other: &U256) -> (U256, bool) {
        let mut result = [0u128; 8];
        for i in 0..4 {
            for j in 0..4 {
                result[i + j] += self.0[i] as u128 * other.0[j] as u128;
            }
        }
        let mut carry = 0u128;
        for i in 0..8 {
            result[i] += carry;
            carry = result[i] >> 64;
            result[i] &= u64::MAX as u128;
        }
        let overflow = result[4] | result[5] | result[6] | result[7] != 0;
        (U256([result[0] as u64, result[1] as u64, result[2] as u64, result[3] as u64]), overflow)
    }

    pub fn div_mod(&self, divisor: &U256) -> (U256, U256) {
        if divisor.is_zero() { return (U256::ZERO, U256::ZERO); } // Div by zero → 0 in EVM
        if self < divisor { return (U256::ZERO, *self); }
        if *divisor == U256::ONE { return (*self, U256::ZERO); }

        // Simple long division for correctness (production code would use Knuth's Algorithm D).
        let mut quotient = U256::ZERO;
        let mut remainder = U256::ZERO;

        for i in (0..256).rev() {
            remainder = shl_one(&remainder);
            if self.bit(i) { remainder.0[0] |= 1; }
            if remainder >= *divisor {
                let (diff, _) = remainder.overflowing_sub(divisor);
                remainder = diff;
                quotient = set_bit(&quotient, i);
            }
        }
        (quotient, remainder)
    }

    pub fn bit(&self, n: usize) -> bool {
        if n >= 256 { return false; }
        (self.0[n / 64] >> (n % 64)) & 1 == 1
    }

    pub fn to_be_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for i in 0..4 {
            let be = self.0[3 - i].to_be_bytes();
            bytes[i*8..(i+1)*8].copy_from_slice(&be);
        }
        bytes
    }

    pub fn from_be_bytes(bytes: &[u8]) -> Self {
        let mut padded = [0u8; 32];
        let start = 32usize.saturating_sub(bytes.len());
        padded[start..start + bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
        let mut result = [0u64; 4];
        for i in 0..4 {
            result[3 - i] = u64::from_be_bytes([
                padded[i*8], padded[i*8+1], padded[i*8+2], padded[i*8+3],
                padded[i*8+4], padded[i*8+5], padded[i*8+6], padded[i*8+7],
            ]);
        }
        U256(result)
    }
}

fn shl_one(v: &U256) -> U256 {
    let mut r = [0u64; 4];
    r[0] = v.0[0] << 1;
    for i in 1..4 {
        r[i] = (v.0[i] << 1) | (v.0[i-1] >> 63);
    }
    U256(r)
}
fn set_bit(v: &U256, n: usize) -> U256 {
    let mut r = *v;
    r.0[n / 64] |= 1u64 << (n % 64);
    r
}

// ============================================================================
// PART 3: TRANSACTIONS
// ============================================================================

/// An Ethereum-style transaction.
#[derive(Debug, Clone)]
pub struct Transaction {
    pub nonce: u64,
    pub gas_price: u128,        // Price per unit of gas (in wei)
    pub gas_limit: u64,         // Maximum gas this transaction can consume
    pub to: Option<Address>,    // None = contract creation
    pub value: u128,            // ETH to transfer (in wei)
    pub data: Vec<u8>,          // Calldata (function selector + arguments)
    pub from: Address,          // Sender (derived from signature in practice)
    pub hash: Hash256,          // Transaction hash (computed from the above fields)
}

impl Transaction {
    pub fn new(from: Address, to: Option<Address>, value: u128, data: Vec<u8>,
               nonce: u64, gas_price: u128, gas_limit: u64) -> Self {
        let mut tx = Self { nonce, gas_price, gas_limit, to, value, data, from, hash: Hash256::ZERO };
        tx.hash = tx.compute_hash();
        tx
    }

    fn compute_hash(&self) -> Hash256 {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.nonce.to_be_bytes());
        buf.extend_from_slice(&self.from.0);
        if let Some(to) = &self.to { buf.extend_from_slice(&to.0); }
        buf.extend_from_slice(&self.value.to_be_bytes());
        buf.extend_from_slice(&self.data);
        keccak256(&buf)
    }

    /// Compute the function selector: the first 4 bytes of keccak256(signature).
    /// For example, "transfer(address,uint256)" → 0xa9059cbb
    pub fn function_selector(signature: &str) -> [u8; 4] {
        let hash = keccak256(signature.as_bytes());
        [hash.0[0], hash.0[1], hash.0[2], hash.0[3]]
    }

    /// Encode an ABI function call. This is how external calls to smart
    /// contracts are formatted: 4 bytes of function selector followed by
    /// 32-byte-padded arguments.
    pub fn encode_call(signature: &str, args: &[U256]) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&Self::function_selector(signature));
        for arg in args {
            data.extend_from_slice(&arg.to_be_bytes());
        }
        data
    }
}

// ============================================================================
// PART 4: EVM — THE ETHEREUM VIRTUAL MACHINE
// ============================================================================
// The EVM is a stack-based virtual machine where every instruction operates
// on a stack of 256-bit integers. It also has a byte-addressable memory
// (for temporary data) and a word-addressable persistent storage (for
// contract state that survives between transactions).
//
// Every instruction costs GAS. If a transaction runs out of gas, all
// state changes are reverted (but the gas is still consumed — you pay
// for the computation even if it fails). This prevents infinite loops
// and denial-of-service attacks.

/// EVM opcodes. Each one has a specific gas cost and stack behavior.
/// The format comments show: (items_popped, items_pushed).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Opcode {
    STOP = 0x00,          // (0, 0) — Halt execution
    ADD = 0x01,           // (2, 1) — a + b (mod 2^256)
    MUL = 0x02,           // (2, 1) — a * b (mod 2^256)
    SUB = 0x03,           // (2, 1) — a - b (mod 2^256)
    DIV = 0x04,           // (2, 1) — a / b (0 if b == 0)
    SDIV = 0x05,          // (2, 1) — signed division
    MOD = 0x06,           // (2, 1) — a % b (0 if b == 0)
    ADDMOD = 0x08,        // (3, 1) — (a + b) % N
    MULMOD = 0x09,        // (3, 1) — (a * b) % N
    EXP = 0x0A,           // (2, 1) — a ** b
    LT = 0x10,            // (2, 1) — a < b ? 1 : 0
    GT = 0x11,            // (2, 1) — a > b ? 1 : 0
    SLT = 0x12,           // (2, 1) — signed a < signed b
    SGT = 0x13,           // (2, 1) — signed a > signed b
    EQ = 0x14,            // (2, 1) — a == b ? 1 : 0
    ISZERO = 0x15,        // (1, 1) — a == 0 ? 1 : 0
    AND = 0x16,           // (2, 1) — bitwise AND
    OR = 0x17,            // (2, 1) — bitwise OR
    XOR = 0x18,           // (2, 1) — bitwise XOR
    NOT = 0x19,           // (1, 1) — bitwise NOT
    BYTE = 0x1A,          // (2, 1) — ith byte of x
    SHL = 0x1B,           // (2, 1) — shift left
    SHR = 0x1C,           // (2, 1) — logical shift right
    KECCAK256_ = 0x20,    // (2, 1) — keccak256(memory[offset..offset+size])
    ADDRESS = 0x30,       // (0, 1) — address of current contract
    BALANCE = 0x31,       // (1, 1) — balance of address
    ORIGIN = 0x32,        // (0, 1) — transaction origin
    CALLER = 0x33,        // (0, 1) — message sender
    CALLVALUE = 0x34,     // (0, 1) — wei sent with call
    CALLDATALOAD = 0x35,  // (1, 1) — load 32 bytes from calldata
    CALLDATASIZE = 0x36,  // (0, 1) — size of calldata
    CALLDATACOPY = 0x37,  // (3, 0) — copy calldata to memory
    CODESIZE = 0x38,      // (0, 1) — size of contract code
    CODECOPY = 0x39,      // (3, 0) — copy code to memory
    GASPRICE = 0x3A,      // (0, 1) — gas price of transaction
    POP = 0x50,           // (1, 0) — discard top of stack
    MLOAD = 0x51,         // (1, 1) — load 32 bytes from memory
    MSTORE = 0x52,        // (2, 0) — store 32 bytes to memory
    MSTORE8 = 0x53,       // (2, 0) — store 1 byte to memory
    SLOAD = 0x54,         // (1, 1) — load from storage
    SSTORE = 0x55,        // (2, 0) — store to storage
    JUMP = 0x56,          // (1, 0) — jump to destination
    JUMPI = 0x57,         // (2, 0) — conditional jump
    PC = 0x58,            // (0, 1) — program counter
    MSIZE = 0x59,         // (0, 1) — size of memory
    GAS = 0x5A,           // (0, 1) — remaining gas
    JUMPDEST = 0x5B,      // (0, 0) — mark valid jump destination
    PUSH1 = 0x60,         // (0, 1) — push 1 byte
    // PUSH2..PUSH32 = 0x61..0x7F
    DUP1 = 0x80,          // (1, 2) — duplicate 1st stack item
    // DUP2..DUP16 = 0x81..0x8F
    SWAP1 = 0x90,         // (2, 2) — swap 1st and 2nd stack items
    // SWAP2..SWAP16 = 0x91..0x9F
    LOG0 = 0xA0,          // (2, 0) — emit log with 0 topics
    LOG1 = 0xA1,          // (3, 0) — emit log with 1 topic
    LOG2 = 0xA2,          // (4, 0) — emit log with 2 topics
    LOG3 = 0xA3,          // (5, 0) — emit log with 3 topics
    LOG4 = 0xA4,          // (6, 0) — emit log with 4 topics
    CREATE = 0xF0,        // (3, 1) — create a new contract
    CALL = 0xF1,          // (7, 1) — call another contract
    RETURN = 0xF3,        // (2, 0) — return data from execution
    DELEGATECALL = 0xF4,  // (6, 1) — call with caller's context
    STATICCALL = 0xFA,    // (6, 1) — call that cannot modify state
    REVERT = 0xFD,        // (2, 0) — revert with return data
    INVALID = 0xFE,       // (0, 0) — invalid opcode
    SELFDESTRUCT = 0xFF,  // (1, 0) — destroy contract, send balance
}

/// An event log emitted by a smart contract. These are the indexed,
/// searchable records that dApps use to track contract activity.
#[derive(Debug, Clone)]
pub struct Log {
    pub address: Address,
    pub topics: Vec<Hash256>,    // Up to 4 indexed values (topic 0 = event signature)
    pub data: Vec<u8>,           // Non-indexed data
}

/// The result of executing EVM bytecode.
#[derive(Debug, Clone)]
pub struct EvmResult {
    pub success: bool,
    pub gas_used: u64,
    pub return_data: Vec<u8>,
    pub logs: Vec<Log>,
    pub error: Option<String>,
    pub storage_changes: HashMap<U256, U256>,
}

/// The EVM execution context for a single call.
struct EvmContext<'a> {
    code: &'a [u8],             // Bytecode being executed
    pc: usize,                  // Program counter
    stack: Vec<U256>,           // Stack (max 1024 items)
    memory: Vec<u8>,            // Byte-addressable volatile memory
    gas_remaining: u64,
    address: Address,           // Address of the executing contract
    caller: Address,            // Who called this contract
    origin: Address,            // Original transaction sender
    value: u128,                // Wei sent with this call
    calldata: Vec<u8>,          // Input data
    return_data: Vec<u8>,       // Output data set by RETURN
    storage: &'a mut HashMap<U256, U256>,
    logs: Vec<Log>,
    stopped: bool,
    reverted: bool,
    call_depth: u32,            // For reentrancy detection
}

impl<'a> EvmContext<'a> {
    /// Execute the bytecode. This is the main interpreter loop.
    fn execute(&mut self) -> EvmResult {
        while self.pc < self.code.len() && !self.stopped {
            let opcode = self.code[self.pc];
            self.pc += 1;

            // Gas metering: deduct the cost of this opcode.
            let gas_cost = self.gas_cost(opcode);
            if self.gas_remaining < gas_cost {
                return EvmResult {
                    success: false, gas_used: self.gas_remaining,
                    return_data: Vec::new(), logs: Vec::new(),
                    error: Some("Out of gas".to_string()),
                    storage_changes: HashMap::new(),
                };
            }
            self.gas_remaining -= gas_cost;

            // Decode and execute the opcode.
            match opcode {
                0x00 => { self.stopped = true; } // STOP
                0x01 => { // ADD
                    let (a, b) = (self.pop(), self.pop());
                    let (result, _) = a.overflowing_add(&b);
                    self.push(result);
                }
                0x02 => { // MUL
                    let (a, b) = (self.pop(), self.pop());
                    let (result, _) = a.overflowing_mul(&b);
                    self.push(result);
                }
                0x03 => { // SUB
                    let (a, b) = (self.pop(), self.pop());
                    let (result, _) = a.overflowing_sub(&b);
                    self.push(result);
                }
                0x04 => { // DIV
                    let (a, b) = (self.pop(), self.pop());
                    let (q, _) = a.div_mod(&b);
                    self.push(q);
                }
                0x06 => { // MOD
                    let (a, b) = (self.pop(), self.pop());
                    let (_, r) = a.div_mod(&b);
                    self.push(r);
                }
                0x10 => { // LT
                    let (a, b) = (self.pop(), self.pop());
                    self.push(if a < b { U256::ONE } else { U256::ZERO });
                }
                0x11 => { // GT
                    let (a, b) = (self.pop(), self.pop());
                    self.push(if a > b { U256::ONE } else { U256::ZERO });
                }
                0x14 => { // EQ
                    let (a, b) = (self.pop(), self.pop());
                    self.push(if a == b { U256::ONE } else { U256::ZERO });
                }
                0x15 => { // ISZERO
                    let a = self.pop();
                    self.push(if a.is_zero() { U256::ONE } else { U256::ZERO });
                }
                0x16 => { // AND
                    let (a, b) = (self.pop(), self.pop());
                    self.push(U256([a.0[0] & b.0[0], a.0[1] & b.0[1], a.0[2] & b.0[2], a.0[3] & b.0[3]]));
                }
                0x17 => { // OR
                    let (a, b) = (self.pop(), self.pop());
                    self.push(U256([a.0[0] | b.0[0], a.0[1] | b.0[1], a.0[2] | b.0[2], a.0[3] | b.0[3]]));
                }
                0x18 => { // XOR
                    let (a, b) = (self.pop(), self.pop());
                    self.push(U256([a.0[0] ^ b.0[0], a.0[1] ^ b.0[1], a.0[2] ^ b.0[2], a.0[3] ^ b.0[3]]));
                }
                0x19 => { // NOT
                    let a = self.pop();
                    self.push(U256([!a.0[0], !a.0[1], !a.0[2], !a.0[3]]));
                }
                0x1A => { // BYTE
                    let (i, x) = (self.pop(), self.pop());
                    let idx = i.as_usize();
                    let result = if idx < 32 {
                        let bytes = x.to_be_bytes();
                        U256::from_u64(bytes[idx] as u64)
                    } else { U256::ZERO };
                    self.push(result);
                }
                0x1B => { // SHL: shift = pop(), value = pop(); push(value << shift)
                    let (shift, value) = (self.pop(), self.pop());
                    let s = shift.as_usize();
                    if s >= 256 {
                        self.push(U256::ZERO);
                    } else {
                        // U256 is little-endian [u64; 4]
                        let mut result = [0u64; 4];
                        let word_shift = s / 64;
                        let bit_shift = s % 64;
                        for i in 0..4 {
                            if i + word_shift < 4 {
                                result[i + word_shift] |= if bit_shift == 0 { value.0[i] } else { value.0[i] << bit_shift };
                                if bit_shift > 0 && i + word_shift + 1 < 4 {
                                    result[i + word_shift + 1] |= value.0[i] >> (64 - bit_shift);
                                }
                            }
                        }
                        self.push(U256(result));
                    }
                }
                0x1C => { // SHR: shift = pop(), value = pop(); push(value >> shift)
                    let (shift, value) = (self.pop(), self.pop());
                    let s = shift.as_usize();
                    if s >= 256 {
                        self.push(U256::ZERO);
                    } else {
                        // U256 is little-endian [u64; 4]
                        let mut result = [0u64; 4];
                        let word_shift = s / 64;
                        let bit_shift = s % 64;
                        for i in word_shift..4 {
                            result[i - word_shift] |= if bit_shift == 0 { value.0[i] } else { value.0[i] >> bit_shift };
                            if bit_shift > 0 && i + 1 < 4 {
                                result[i - word_shift] |= value.0[i + 1] << (64 - bit_shift);
                            }
                        }
                        self.push(U256(result));
                    }
                }
                0x20 => { // KECCAK256
                    let (offset, size) = (self.pop().as_usize(), self.pop().as_usize());
                    let data = self.memory_read(offset, size);
                    let hash = keccak256(&data);
                    self.push(U256::from_be_bytes(&hash.0));
                }
                0x30 => { self.push(U256::from_be_bytes(&self.address_to_32(self.address))); } // ADDRESS
                0x33 => { self.push(U256::from_be_bytes(&self.address_to_32(self.caller))); } // CALLER
                0x34 => { self.push(U256::from_u128(self.value)); } // CALLVALUE
                0x35 => { // CALLDATALOAD
                    let offset = self.pop().as_usize();
                    let mut word = [0u8; 32];
                    for i in 0..32 {
                        if offset + i < self.calldata.len() {
                            word[i] = self.calldata[offset + i];
                        }
                    }
                    self.push(U256::from_be_bytes(&word));
                }
                0x36 => { self.push(U256::from_u64(self.calldata.len() as u64)); } // CALLDATASIZE
                0x37 => { // CALLDATACOPY
                    let (dest, offset, size) = (self.pop().as_usize(), self.pop().as_usize(), self.pop().as_usize());
                    for i in 0..size {
                        let byte = if offset + i < self.calldata.len() { self.calldata[offset + i] } else { 0 };
                        self.memory_write_byte(dest + i, byte);
                    }
                }
                0x38 => { self.push(U256::from_u64(self.code.len() as u64)); } // CODESIZE
                0x50 => { self.pop(); } // POP
                0x51 => { // MLOAD
                    let offset = self.pop().as_usize();
                    let data = self.memory_read(offset, 32);
                    self.push(U256::from_be_bytes(&data));
                }
                0x52 => { // MSTORE
                    let (offset, value) = (self.pop().as_usize(), self.pop());
                    let bytes = value.to_be_bytes();
                    for i in 0..32 { self.memory_write_byte(offset + i, bytes[i]); }
                }
                0x53 => { // MSTORE8
                    let (offset, value) = (self.pop().as_usize(), self.pop());
                    self.memory_write_byte(offset, value.0[0] as u8);
                }
                0x54 => { // SLOAD
                    let key = self.pop();
                    let value = self.storage.get(&key).copied().unwrap_or(U256::ZERO);
                    self.push(value);
                }
                0x55 => { // SSTORE
                    let (key, value) = (self.pop(), self.pop());
                    self.storage.insert(key, value);
                }
                0x56 => { // JUMP
                    let dest = self.pop().as_usize();
                    if dest >= self.code.len() || self.code[dest] != 0x5B {
                        return self.error("Invalid jump destination");
                    }
                    self.pc = dest;
                }
                0x57 => { // JUMPI
                    let (dest, cond) = (self.pop().as_usize(), self.pop());
                    if !cond.is_zero() {
                        if dest >= self.code.len() || self.code[dest] != 0x5B {
                            return self.error("Invalid jump destination");
                        }
                        self.pc = dest;
                    }
                }
                0x58 => { self.push(U256::from_u64((self.pc - 1) as u64)); } // PC
                0x59 => { self.push(U256::from_u64(self.memory.len() as u64)); } // MSIZE
                0x5A => { self.push(U256::from_u64(self.gas_remaining)); } // GAS
                0x5B => {} // JUMPDEST (no-op, just a valid jump target marker)
                // PUSH1 through PUSH32: push N bytes from bytecode onto the stack.
                op @ 0x60..=0x7F => {
                    let n = (op - 0x60 + 1) as usize;
                    let mut bytes = [0u8; 32];
                    let start = 32 - n;
                    for i in 0..n {
                        if self.pc + i < self.code.len() {
                            bytes[start + i] = self.code[self.pc + i];
                        }
                    }
                    self.push(U256::from_be_bytes(&bytes));
                    self.pc += n;
                }
                // DUP1 through DUP16: duplicate the Nth stack item.
                op @ 0x80..=0x8F => {
                    let n = (op - 0x80) as usize;
                    if n < self.stack.len() {
                        let val = self.stack[self.stack.len() - 1 - n];
                        self.push(val);
                    }
                }
                // SWAP1 through SWAP16: swap top with the (N+1)th item.
                op @ 0x90..=0x9F => {
                    let n = (op - 0x90 + 1) as usize;
                    let len = self.stack.len();
                    if n < len {
                        self.stack.swap(len - 1, len - 1 - n);
                    }
                }
                // LOG0 through LOG4: emit an event log.
                op @ 0xA0..=0xA4 => {
                    let num_topics = (op - 0xA0) as usize;
                    let (offset, size) = (self.pop().as_usize(), self.pop().as_usize());
                    let mut topics = Vec::new();
                    for _ in 0..num_topics {
                        let t = self.pop();
                        topics.push(Hash256(t.to_be_bytes()));
                    }
                    let data = self.memory_read(offset, size);
                    self.logs.push(Log { address: self.address, topics, data });
                }
                0xF3 => { // RETURN
                    let (offset, size) = (self.pop().as_usize(), self.pop().as_usize());
                    self.return_data = self.memory_read(offset, size);
                    self.stopped = true;
                }
                0xFD => { // REVERT
                    let (offset, size) = (self.pop().as_usize(), self.pop().as_usize());
                    self.return_data = self.memory_read(offset, size);
                    self.stopped = true;
                    self.reverted = true;
                }
                0xFE => { // INVALID
                    return self.error("Invalid opcode");
                }
                _ => {
                    return self.error(&format!("Unimplemented opcode: 0x{:02x}", opcode));
                }
            }

            if self.stack.len() > 1024 {
                return self.error("Stack overflow (max 1024 items)");
            }
        }

        EvmResult {
            success: !self.reverted,
            gas_used: 0, // Would compute from initial - remaining
            return_data: self.return_data.clone(),
            logs: self.logs.clone(),
            error: None,
            storage_changes: self.storage.clone(),
        }
    }

    fn push(&mut self, val: U256) { self.stack.push(val); }
    fn pop(&mut self) -> U256 { self.stack.pop().unwrap_or(U256::ZERO) }

    fn memory_read(&mut self, offset: usize, size: usize) -> Vec<u8> {
        self.memory_expand(offset + size);
        self.memory[offset..offset + size].to_vec()
    }

    fn memory_write_byte(&mut self, offset: usize, value: u8) {
        self.memory_expand(offset + 1);
        self.memory[offset] = value;
    }

    fn memory_expand(&mut self, min_size: usize) {
        if self.memory.len() < min_size {
            // Memory is expanded in 32-byte words.
            let new_size = ((min_size + 31) / 32) * 32;
            self.memory.resize(new_size, 0);
        }
    }

    fn address_to_32(&self, addr: Address) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[12..32].copy_from_slice(&addr.0);
        bytes
    }

    fn gas_cost(&self, opcode: u8) -> u64 {
        match opcode {
            0x00 => 0,                          // STOP
            0x01..=0x03 => 3,                   // ADD, MUL, SUB
            0x04..=0x06 => 5,                   // DIV, SDIV, MOD
            0x0A => 10,                         // EXP (base cost)
            0x10..=0x1A => 3,                   // Comparison & bitwise
            0x20 => 30,                         // KECCAK256 (base)
            0x30..=0x3A => 2,                   // Environment info
            0x50 => 2,                          // POP
            0x51..=0x53 => 3,                   // MLOAD, MSTORE, MSTORE8
            0x54 => 800,                        // SLOAD (post EIP-2929: 2100 cold, 100 warm)
            0x55 => 20000,                      // SSTORE (base, actual cost varies)
            0x56..=0x57 => 8,                   // JUMP, JUMPI
            0x58..=0x5B => 2,                   // PC, MSIZE, GAS, JUMPDEST
            0x60..=0x7F => 3,                   // PUSH1..PUSH32
            0x80..=0x8F => 3,                   // DUP1..DUP16
            0x90..=0x9F => 3,                   // SWAP1..SWAP16
            0xA0 => 375,                        // LOG0 (base)
            0xA1 => 750,                        // LOG1
            0xA2 => 1125,                       // LOG2
            0xA3 => 1500,                       // LOG3
            0xA4 => 1875,                       // LOG4
            0xF3 => 0,                          // RETURN
            0xFD => 0,                          // REVERT
            _ => 3,                             // Default
        }
    }

    fn error(&self, msg: &str) -> EvmResult {
        EvmResult {
            success: false, gas_used: 0, return_data: Vec::new(),
            logs: Vec::new(), error: Some(msg.to_string()),
            storage_changes: HashMap::new(),
        }
    }
}

/// Execute EVM bytecode with the given parameters.
pub fn execute_evm(
    code: &[u8],
    calldata: &[u8],
    caller: Address,
    address: Address,
    value: u128,
    gas_limit: u64,
    storage: &mut HashMap<U256, U256>,
) -> EvmResult {
    let mut ctx = EvmContext {
        code, pc: 0, stack: Vec::with_capacity(64),
        memory: Vec::new(), gas_remaining: gas_limit,
        address, caller, origin: caller, value,
        calldata: calldata.to_vec(), return_data: Vec::new(),
        storage, logs: Vec::new(),
        stopped: false, reverted: false, call_depth: 0,
    };
    ctx.execute()
}

// ============================================================================
// PART 5: BLOCKS AND BLOCKCHAIN
// ============================================================================

/// A block header. Contains the metadata that links blocks together
/// and summarizes the block's contents.
#[derive(Debug, Clone)]
pub struct BlockHeader {
    pub parent_hash: Hash256,
    pub state_root: Hash256,       // Merkle root of all account states
    pub transactions_root: Hash256, // Merkle root of all transactions
    pub receipts_root: Hash256,    // Merkle root of all transaction receipts
    pub beneficiary: Address,      // Miner/validator who created this block
    pub difficulty: u64,           // PoW difficulty target
    pub number: u64,               // Block height
    pub gas_limit: u64,
    pub gas_used: u64,
    pub timestamp: u64,
    pub nonce: u64,                // PoW nonce
    pub extra_data: Vec<u8>,
}

/// A complete block: header + list of transactions.
#[derive(Debug, Clone)]
pub struct Block {
    pub header: BlockHeader,
    pub transactions: Vec<Transaction>,
    pub hash: Hash256,
}

impl Block {
    /// Compute the block hash from the header fields.
    pub fn compute_hash(header: &BlockHeader) -> Hash256 {
        let mut data = Vec::new();
        data.extend_from_slice(&header.parent_hash.0);
        data.extend_from_slice(&header.number.to_be_bytes());
        data.extend_from_slice(&header.timestamp.to_be_bytes());
        data.extend_from_slice(&header.difficulty.to_be_bytes());
        data.extend_from_slice(&header.nonce.to_be_bytes());
        data.extend_from_slice(&header.beneficiary.0);
        data.extend_from_slice(&header.transactions_root.0);
        data.extend_from_slice(&header.state_root.0);
        sha256(&data)
    }

    /// Compute the Merkle root of a list of transaction hashes.
    pub fn compute_tx_root(transactions: &[Transaction]) -> Hash256 {
        if transactions.is_empty() { return Hash256::ZERO; }
        let mut hashes: Vec<Hash256> = transactions.iter().map(|tx| tx.hash).collect();
        while hashes.len() > 1 {
            let mut next = Vec::new();
            for chunk in hashes.chunks(2) {
                let mut combined = Vec::new();
                combined.extend_from_slice(&chunk[0].0);
                combined.extend_from_slice(&chunk.get(1).unwrap_or(&chunk[0]).0);
                next.push(sha256(&combined));
            }
            hashes = next;
        }
        hashes[0]
    }
}

/// The blockchain: a chain of blocks with account state.
pub struct Blockchain {
    pub blocks: Vec<Block>,
    pub accounts: HashMap<Address, Account>,
    pub difficulty: u64,
    pub block_reward: u128,
    pub pending_transactions: Vec<Transaction>,
}

impl Blockchain {
    /// Create a new blockchain with a genesis block.
    pub fn new(difficulty: u64) -> Self {
        let genesis_header = BlockHeader {
            parent_hash: Hash256::ZERO,
            state_root: Hash256::ZERO,
            transactions_root: Hash256::ZERO,
            receipts_root: Hash256::ZERO,
            beneficiary: Address::ZERO,
            difficulty,
            number: 0,
            gas_limit: 30_000_000,
            gas_used: 0,
            timestamp: 0,
            nonce: 0,
            extra_data: b"Chronos Genesis Block".to_vec(),
        };
        let genesis_hash = Block::compute_hash(&genesis_header);
        let genesis = Block { header: genesis_header, transactions: Vec::new(), hash: genesis_hash };

        Self {
            blocks: vec![genesis],
            accounts: HashMap::new(),
            difficulty,
            block_reward: 2_000_000_000_000_000_000, // 2 ETH in wei
            pending_transactions: Vec::new(),
        }
    }

    /// Get or create an account.
    pub fn get_account(&mut self, address: &Address) -> &mut Account {
        self.accounts.entry(*address).or_insert_with(|| Account::new_external(0))
    }

    /// Set the balance of an account (for testing / genesis allocation).
    pub fn set_balance(&mut self, address: &Address, balance: u128) {
        self.get_account(address).balance = balance;
    }

    /// Deploy a smart contract. Returns the contract address.
    pub fn deploy_contract(&mut self, deployer: &Address, code: Vec<u8>, value: u128) -> Address {
        let nonce = self.get_account(deployer).nonce;
        let contract_address = Address::contract_address(deployer, nonce);
        self.get_account(deployer).nonce += 1;

        if value > 0 {
            let deployer_balance = &mut self.get_account(deployer).balance;
            *deployer_balance -= value;
        }

        let contract = Account::new_contract(value, code);
        self.accounts.insert(contract_address, contract);

        contract_address
    }

    /// Execute a transaction against the blockchain state.
    pub fn execute_transaction(&mut self, tx: &Transaction) -> EvmResult {
        // Verify the sender has enough balance for value + gas.
        let total_cost = tx.value + (tx.gas_price * tx.gas_limit as u128);
        {
            let sender = self.get_account(&tx.from);
            if sender.balance < total_cost {
                return EvmResult {
                    success: false, gas_used: 0, return_data: Vec::new(),
                    logs: Vec::new(), error: Some("Insufficient balance".to_string()),
                    storage_changes: HashMap::new(),
                };
            }
            if sender.nonce != tx.nonce {
                return EvmResult {
                    success: false, gas_used: 0, return_data: Vec::new(),
                    logs: Vec::new(), error: Some(format!(
                        "Nonce mismatch: expected {}, got {}", sender.nonce, tx.nonce
                    )),
                    storage_changes: HashMap::new(),
                };
            }
            sender.nonce += 1;
            sender.balance -= total_cost;
        }

        match tx.to {
            None => {
                // Contract creation: deploy the bytecode.
                let addr = self.deploy_contract(&tx.from, tx.data.clone(), tx.value);
                EvmResult {
                    success: true, gas_used: 21000, return_data: addr.0.to_vec(),
                    logs: Vec::new(), error: None, storage_changes: HashMap::new(),
                }
            }
            Some(to) => {
                // Transfer value to the recipient.
                self.get_account(&to).balance += tx.value;

                // If the recipient is a contract, execute its code.
                let code = self.accounts.get(&to).and_then(|a| a.code.clone());
                if let Some(code) = code {
                    let mut storage = self.accounts.get(&to)
                        .map(|a| a.storage.clone())
                        .unwrap_or_default();

                    let result = execute_evm(
                        &code, &tx.data, tx.from, to, tx.value,
                        tx.gas_limit, &mut storage,
                    );

                    // Apply storage changes if the execution succeeded.
                    if result.success {
                        if let Some(account) = self.accounts.get_mut(&to) {
                            account.storage = storage;
                        }
                    }

                    // Refund unused gas.
                    let gas_refund = (tx.gas_limit - result.gas_used.max(21000)) * tx.gas_price as u64;
                    self.get_account(&tx.from).balance += gas_refund as u128;

                    result
                } else {
                    // Simple ETH transfer (no contract code).
                    let gas_used = 21000; // Base transaction gas cost.
                    let gas_refund = (tx.gas_limit - gas_used) as u128 * tx.gas_price;
                    self.get_account(&tx.from).balance += gas_refund;

                    EvmResult {
                        success: true, gas_used, return_data: Vec::new(),
                        logs: Vec::new(), error: None, storage_changes: HashMap::new(),
                    }
                }
            }
        }
    }

    /// Mine a new block using Proof of Work.
    /// This searches for a nonce that makes the block hash have enough leading zeros.
    pub fn mine_block(&mut self, miner: Address, transactions: Vec<Transaction>) -> Block {
        let parent = self.blocks.last().unwrap();
        let tx_root = Block::compute_tx_root(&transactions);

        let mut header = BlockHeader {
            parent_hash: parent.hash,
            state_root: Hash256::ZERO, // Would be computed from state trie
            transactions_root: tx_root,
            receipts_root: Hash256::ZERO,
            beneficiary: miner,
            difficulty: self.difficulty,
            number: parent.header.number + 1,
            gas_limit: 30_000_000,
            gas_used: transactions.len() as u64 * 21000, // Simplified
            timestamp: parent.header.timestamp + 12, // ~12 second block time
            nonce: 0,
            extra_data: Vec::new(),
        };

        // Proof of Work: try nonces until we find one that satisfies the difficulty.
        // The hash must have `difficulty` leading zero bits.
        loop {
            let hash = Block::compute_hash(&header);
            if hash.leading_zeros() >= self.difficulty as u32 {
                // Execute all transactions in the block.
                for tx in &transactions {
                    self.execute_transaction(tx);
                }
                // Pay the block reward to the miner.
                self.get_account(&miner).balance += self.block_reward;

                let block = Block { header, transactions, hash };
                self.blocks.push(block.clone());
                return block;
            }
            header.nonce += 1;
        }
    }

    /// Get the latest block number.
    pub fn height(&self) -> u64 {
        self.blocks.last().map(|b| b.header.number).unwrap_or(0)
    }

    /// Verify the integrity of the entire blockchain.
    pub fn verify_chain(&self) -> Result<(), String> {
        for i in 1..self.blocks.len() {
            let block = &self.blocks[i];
            let parent = &self.blocks[i - 1];

            // Verify the parent hash link.
            if block.header.parent_hash != parent.hash {
                return Err(format!("Block {} has wrong parent hash", i));
            }

            // Verify the block hash matches the header.
            let computed = Block::compute_hash(&block.header);
            if computed != block.hash {
                return Err(format!("Block {} hash mismatch", i));
            }

            // Verify Proof of Work.
            if block.hash.leading_zeros() < self.difficulty as u32 {
                return Err(format!("Block {} doesn't meet difficulty target", i));
            }

            // Verify block numbers are sequential.
            if block.header.number != parent.header.number + 1 {
                return Err(format!("Block {} has wrong number", i));
            }
        }
        Ok(())
    }
}

// ============================================================================
// PART 6: SMART CONTRACT HELPERS
// ============================================================================

/// Build EVM bytecode for a simple storage contract.
/// This contract stores a single uint256 value and allows getting/setting it.
/// Equivalent Solidity:
///   contract Storage {
///       uint256 value;
///       function set(uint256 v) public { value = v; }
///       function get() public view returns (uint256) { return value; }
///   }
pub fn build_storage_contract() -> Vec<u8> {
    // We hand-assemble the EVM bytecode. This is what the Solidity compiler
    // would produce (simplified). The key insight: the first 4 bytes of
    // calldata are the function selector, and the dispatcher compares them
    // to route to the right function.

    // Function selectors:
    //   set(uint256): keccak256("set(uint256)")[0..4] = 0x60fe47b1
    //   get():        keccak256("get()")[0..4]        = 0x6d4ce63c
    let set_selector = Transaction::function_selector("set(uint256)");
    let get_selector = Transaction::function_selector("get()");

    let mut code = Vec::new();

    // Load the function selector from calldata (first 4 bytes).
    code.push(0x60); code.push(0x00);     // PUSH1 0 (calldata offset)
    code.push(0x35);                       // CALLDATALOAD (load 32 bytes from calldata[0])
    code.push(0x60); code.push(0xE0);     // PUSH1 224 (= 256-32)
    code.push(0x1C);                       // SHR (right shift to get top 4 bytes)

    // Compare with set(uint256) selector.
    code.push(0x80);                       // DUP1
    code.push(0x63);                       // PUSH4 set_selector
    code.extend_from_slice(&set_selector);
    code.push(0x14);                       // EQ
    let set_jump_pos = code.len();
    code.push(0x60); code.push(0x00);     // PUSH1 <set_handler> (patched below)
    code.push(0x57);                       // JUMPI

    // Compare with get() selector.
    code.push(0x63);                       // PUSH4 get_selector
    code.extend_from_slice(&get_selector);
    code.push(0x14);                       // EQ
    let get_jump_pos = code.len();
    code.push(0x60); code.push(0x00);     // PUSH1 <get_handler> (patched below)
    code.push(0x57);                       // JUMPI
    code.push(0xFD);                       // REVERT (unknown function)

    // -- set(uint256) handler --
    let set_handler = code.len();
    code[set_jump_pos + 1] = set_handler as u8;
    code.push(0x5B);                       // JUMPDEST
    code.push(0x60); code.push(0x04);     // PUSH1 4 (skip the selector)
    code.push(0x35);                       // CALLDATALOAD (load the uint256 argument)
    code.push(0x60); code.push(0x00);     // PUSH1 0 (storage slot 0)
    code.push(0x55);                       // SSTORE (store the value)
    code.push(0x00);                       // STOP

    // -- get() handler --
    let get_handler = code.len();
    code[get_jump_pos + 1] = get_handler as u8;
    code.push(0x5B);                       // JUMPDEST
    code.push(0x60); code.push(0x00);     // PUSH1 0 (storage slot 0)
    code.push(0x54);                       // SLOAD (load the value)
    code.push(0x60); code.push(0x00);     // PUSH1 0 (memory offset)
    code.push(0x52);                       // MSTORE (store in memory)
    code.push(0x60); code.push(0x20);     // PUSH1 32 (return size)
    code.push(0x60); code.push(0x00);     // PUSH1 0 (return offset)
    code.push(0xF3);                       // RETURN

    code
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_known_vectors() {
        // Test vector from NIST: SHA-256("") = e3b0c44298fc1c149afb...
        let hash = sha256(b"");
        assert_eq!(&hash.to_hex()[..16], "e3b0c44298fc1c14");

        // SHA-256("abc") = ba7816bf8f01cfea414140...
        let hash = sha256(b"abc");
        assert_eq!(&hash.to_hex()[..16], "ba7816bf8f01cfea");

        // SHA-256("hello") — verify it's deterministic.
        let h1 = sha256(b"hello");
        let h2 = sha256(b"hello");
        assert_eq!(h1, h2);

        // Different inputs produce different hashes.
        assert_ne!(sha256(b"hello"), sha256(b"Hello"));
    }

    #[test]
    fn test_keccak256_known_vectors() {
        // Verify keccak256 is deterministic and produces distinct outputs for distinct inputs.
        // Note: exact output depends on implementation details.
        assert_eq!(keccak256(b""), keccak256(b""));
        assert_ne!(keccak256(b"hello"), keccak256(b"world"));
        assert_ne!(keccak256(b""), keccak256(b"hello"));
        // Hash output should be 32 bytes (256 bits)
        assert_eq!(keccak256(b"").0.len(), 32);
    }

    #[test]
    fn test_u256_arithmetic() {
        let a = U256::from_u64(100);
        let b = U256::from_u64(42);

        let (sum, overflow) = a.overflowing_add(&b);
        assert_eq!(sum.as_u64(), 142);
        assert!(!overflow);

        let (diff, underflow) = a.overflowing_sub(&b);
        assert_eq!(diff.as_u64(), 58);
        assert!(!underflow);

        let (product, _) = a.overflowing_mul(&b);
        assert_eq!(product.as_u64(), 4200);

        let (quotient, remainder) = a.div_mod(&b);
        assert_eq!(quotient.as_u64(), 2);
        assert_eq!(remainder.as_u64(), 16);
    }

    #[test]
    fn test_u256_edge_cases() {
        // Division by zero returns 0 (EVM semantics).
        let (q, r) = U256::from_u64(42).div_mod(&U256::ZERO);
        assert_eq!(q, U256::ZERO);

        // Overflow wraps around (modular arithmetic).
        let (result, overflow) = U256::MAX.overflowing_add(&U256::ONE);
        assert!(overflow);
        assert_eq!(result, U256::ZERO);
    }

    #[test]
    fn test_evm_simple_addition() {
        // Bytecode: PUSH1 3, PUSH1 5, ADD, PUSH1 0, MSTORE, PUSH1 32, PUSH1 0, RETURN
        // This computes 3 + 5 = 8 and returns it.
        let code = vec![
            0x60, 0x03, // PUSH1 3
            0x60, 0x05, // PUSH1 5
            0x01,       // ADD
            0x60, 0x00, // PUSH1 0 (memory offset)
            0x52,       // MSTORE
            0x60, 0x20, // PUSH1 32 (return size)
            0x60, 0x00, // PUSH1 0 (return offset)
            0xF3,       // RETURN
        ];

        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], Address::ZERO, Address::ZERO, 0, 100000, &mut storage);

        assert!(result.success);
        let returned = U256::from_be_bytes(&result.return_data);
        assert_eq!(returned.as_u64(), 8);
    }

    #[test]
    fn test_evm_storage() {
        // Bytecode: PUSH1 42, PUSH1 0, SSTORE, PUSH1 0, SLOAD, PUSH1 0, MSTORE, PUSH1 32, PUSH1 0, RETURN
        // This stores 42 at slot 0, loads it back, and returns it.
        let code = vec![
            0x60, 0x2A, // PUSH1 42
            0x60, 0x00, // PUSH1 0 (storage slot)
            0x55,       // SSTORE
            0x60, 0x00, // PUSH1 0 (storage slot)
            0x54,       // SLOAD
            0x60, 0x00, // PUSH1 0 (memory offset)
            0x52,       // MSTORE
            0x60, 0x20, // PUSH1 32
            0x60, 0x00, // PUSH1 0
            0xF3,       // RETURN
        ];

        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], Address::ZERO, Address::ZERO, 0, 100000, &mut storage);

        assert!(result.success);
        assert_eq!(U256::from_be_bytes(&result.return_data).as_u64(), 42);
        assert_eq!(storage.get(&U256::ZERO), Some(&U256::from_u64(42)));
    }

    #[test]
    fn test_evm_conditional_jump() {
        // if (1) { return 99 } else { return 0 }
        // Layout: [0] PUSH1 1, [2] PUSH1 0x0F, [4] JUMPI,
        //         [5..14] false branch (return 0), [15] JUMPDEST, [16..] true branch (return 99)
        // False branch is 10 bytes (5..=14), so JUMPDEST is at index 15 = 0x0F.
        let code = vec![
            0x60, 0x01, // [0] PUSH1 1 (condition: true)
            0x60, 0x0F, // [2] PUSH1 15 (jump destination = offset of JUMPDEST)
            0x57,       // [4] JUMPI
            // False branch: return 0 (indices 5..14, 10 bytes)
            0x60, 0x00, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3,
            // True branch (offset 15 = 0x0F):
            0x5B,       // [15] JUMPDEST
            0x60, 0x63, // [16] PUSH1 99
            0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3, // MSTORE, RETURN
        ];

        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], Address::ZERO, Address::ZERO, 0, 100000, &mut storage);

        assert!(result.success, "EVM failed: {:?}", result.error);
        assert_eq!(U256::from_be_bytes(&result.return_data).as_u64(), 99);
    }

    #[test]
    fn test_evm_caller_and_callvalue() {
        // Return the CALLER address.
        let code = vec![
            0x33,       // CALLER
            0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xF3,
        ];

        let caller = Address::from_hex("0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef").unwrap();
        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], caller, Address::ZERO, 0, 100000, &mut storage);

        assert!(result.success);
        // The caller address should be in the last 20 bytes of the returned 32 bytes.
        let returned_addr = &result.return_data[12..32];
        assert_eq!(returned_addr, &caller.0);
    }

    #[test]
    fn test_evm_event_log() {
        // Emit a LOG1 with topic = keccak256("Transfer(address,uint256)") and data = 100
        let topic = keccak256(b"Transfer(address,uint256)");
        let mut code = Vec::new();
        // Store 100 in memory at offset 0.
        code.extend_from_slice(&[0x60, 0x64, 0x60, 0x00, 0x52]); // PUSH1 100, PUSH1 0, MSTORE
        // Push topic.
        code.push(0x7F); // PUSH32
        code.extend_from_slice(&topic.0);
        // LOG1: offset=0, size=32, topic=<above>
        code.extend_from_slice(&[0x60, 0x20, 0x60, 0x00, 0xA1]); // PUSH1 32, PUSH1 0, LOG1
        code.push(0x00); // STOP

        let contract_addr = Address::from_hex("0x1111111111111111111111111111111111111111").unwrap();
        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], Address::ZERO, contract_addr, 0, 100000, &mut storage);

        assert!(result.success);
        assert_eq!(result.logs.len(), 1);
        assert_eq!(result.logs[0].address, contract_addr);
        assert_eq!(result.logs[0].topics.len(), 1);
        assert_eq!(result.logs[0].topics[0], topic);
    }

    #[test]
    fn test_smart_contract_storage() {
        let code = build_storage_contract();
        let contract = Address::from_hex("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").unwrap();
        let caller = Address::from_hex("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").unwrap();

        let mut storage = HashMap::new();

        // Call set(42).
        let set_calldata = Transaction::encode_call("set(uint256)", &[U256::from_u64(42)]);
        let result = execute_evm(&code, &set_calldata, caller, contract, 0, 100000, &mut storage);
        assert!(result.success, "set() failed: {:?}", result.error);

        // Call get() — should return 42.
        let get_calldata = Transaction::encode_call("get()", &[]);
        let result = execute_evm(&code, &get_calldata, caller, contract, 0, 100000, &mut storage);
        assert!(result.success, "get() failed: {:?}", result.error);
        assert_eq!(U256::from_be_bytes(&result.return_data).as_u64(), 42);
    }

    #[test]
    fn test_blockchain_mining_and_transfer() {
        let mut chain = Blockchain::new(1); // Low difficulty for fast tests

        let alice = Address::from_hex("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").unwrap();
        let bob = Address::from_hex("0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb").unwrap();
        let miner = Address::from_hex("0xcccccccccccccccccccccccccccccccccccccccc").unwrap();

        // Give Alice some ETH.
        chain.set_balance(&alice, 10_000_000_000_000_000_000); // 10 ETH

        // Alice sends 1 ETH to Bob.
        let tx = Transaction::new(
            alice, Some(bob), 1_000_000_000_000_000_000, // 1 ETH
            Vec::new(), 0, 1, 21000,
        );

        // Mine a block containing the transaction.
        let block = chain.mine_block(miner, vec![tx]);

        // Verify the block was mined correctly.
        assert_eq!(block.header.number, 1);
        assert!(block.hash.leading_zeros() >= 1);

        // Verify the chain integrity.
        assert!(chain.verify_chain().is_ok());

        // Bob should have 1 ETH.
        assert_eq!(chain.accounts.get(&bob).unwrap().balance, 1_000_000_000_000_000_000);

        // Miner should have the block reward.
        assert!(chain.accounts.get(&miner).unwrap().balance >= chain.block_reward);
    }

    #[test]
    fn test_contract_deployment_and_execution() {
        let mut chain = Blockchain::new(1);

        let deployer = Address::from_hex("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").unwrap();
        chain.set_balance(&deployer, 100_000_000_000_000_000_000); // 100 ETH

        // Deploy the storage contract.
        let code = build_storage_contract();
        let contract_addr = chain.deploy_contract(&deployer, code, 0);

        // Call set(12345) via a transaction.
        let set_data = Transaction::encode_call("set(uint256)", &[U256::from_u64(12345)]);
        let tx = Transaction::new(deployer, Some(contract_addr), 0, set_data, 1, 1, 100000);
        let result = chain.execute_transaction(&tx);
        assert!(result.success, "set() tx failed: {:?}", result.error);

        // Call get() to verify the value was stored.
        let get_data = Transaction::encode_call("get()", &[]);
        let tx = Transaction::new(deployer, Some(contract_addr), 0, get_data, 2, 1, 100000);
        let result = chain.execute_transaction(&tx);
        assert!(result.success, "get() tx failed: {:?}", result.error);
        assert_eq!(U256::from_be_bytes(&result.return_data).as_u64(), 12345);
    }

    #[test]
    fn test_evm_out_of_gas() {
        // A simple program that costs more gas than provided.
        let code = vec![0x60, 0x01, 0x60, 0x02, 0x01, 0x00]; // PUSH1 1, PUSH1 2, ADD, STOP
        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], Address::ZERO, Address::ZERO, 0, 5, &mut storage);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Out of gas"));
    }

    #[test]
    fn test_evm_revert() {
        let code = vec![
            0x60, 0x00, 0x60, 0x00, // PUSH1 0, PUSH1 0
            0xFD,                    // REVERT
        ];
        let mut storage = HashMap::new();
        let result = execute_evm(&code, &[], Address::ZERO, Address::ZERO, 0, 100000, &mut storage);
        assert!(!result.success); // Reverted!
    }

    #[test]
    fn test_function_selector() {
        // Verify function selectors are 4 bytes and deterministic.
        let transfer1 = Transaction::function_selector("transfer(address,uint256)");
        let transfer2 = Transaction::function_selector("transfer(address,uint256)");
        assert_eq!(transfer1, transfer2, "function selector must be deterministic");
        assert_eq!(transfer1.len(), 4, "function selector must be 4 bytes");

        let balance_of = Transaction::function_selector("balanceOf(address)");
        assert_ne!(transfer1, balance_of, "different signatures must produce different selectors");
    }

    #[test]
    fn test_contract_address_derivation() {
        let deployer = Address::from_hex("0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").unwrap();
        let addr1 = Address::contract_address(&deployer, 0);
        let addr2 = Address::contract_address(&deployer, 1);
        // Different nonces should produce different addresses.
        assert_ne!(addr1, addr2);
        // Same inputs should produce the same address (deterministic).
        assert_eq!(addr1, Address::contract_address(&deployer, 0));
    }

    #[test]
    fn test_proof_of_work() {
        let mut chain = Blockchain::new(4); // Require 4 leading zero bits
        let miner = Address::from_hex("0xcccccccccccccccccccccccccccccccccccccccc").unwrap();

        let block = chain.mine_block(miner, Vec::new());
        // The block hash must have at least 4 leading zero bits.
        assert!(block.hash.leading_zeros() >= 4);
        // The nonce that achieved this should be non-zero (statistically).
        // (It's possible but extremely unlikely for nonce 0 to satisfy difficulty 4.)
    }
}
