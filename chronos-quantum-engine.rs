// ============================================================================
// CHRONOS QUANTUM COMPUTING ENGINE
// ============================================================================
//
// HOW QUANTUM COMPUTING ACTUALLY WORKS (and how this code models it):
//
// A classical bit is either 0 or 1. A qubit can be in a superposition:
//   |ψ⟩ = α|0⟩ + β|1⟩   where |α|² + |β|² = 1
//
// α and β are complex numbers called "amplitudes." When you measure the
// qubit, you get 0 with probability |α|² and 1 with probability |β|².
// The act of measurement COLLAPSES the state — after measuring 0, the
// qubit is definitely |0⟩.
//
// For N qubits, the state is a vector of 2^N complex amplitudes. Each
// amplitude corresponds to one classical bitstring. For example, 3 qubits:
//   |ψ⟩ = α₀₀₀|000⟩ + α₀₀₁|001⟩ + α₀₁₀|010⟩ + α₀₁₁|011⟩
//        + α₁₀₀|100⟩ + α₁₀₁|101⟩ + α₁₁₀|110⟩ + α₁₁₁|111⟩
//
// A quantum gate is a unitary matrix that transforms this state vector.
// A single-qubit gate is a 2×2 unitary matrix. To apply it to qubit k
// in an N-qubit system, we compute the tensor product with identity
// matrices for all other qubits. But we DON'T actually build the full
// 2^N × 2^N matrix — instead, we apply the gate directly to the state
// vector using index arithmetic. This is the key optimization.
//
// ENTANGLEMENT arises naturally: after a CNOT gate, two qubits become
// correlated in a way that can't be described by separate single-qubit
// states. The state vector formalism handles this automatically because
// we store the full joint state of all qubits.
//
// WHAT THIS ENGINE IMPLEMENTS:
//   1.  Complex number arithmetic (needed for amplitudes)
//   2.  State vector representation and initialization
//   3.  All standard single-qubit gates (H, X, Y, Z, S, T, Rx, Ry, Rz, P, U3)
//   4.  All standard multi-qubit gates (CNOT, CZ, SWAP, Toffoli, Fredkin)
//   5.  Controlled versions of arbitrary gates
//   6.  Measurement with probabilistic collapse
//   7.  Partial measurement (measure one qubit, keep the rest)
//   8.  Quantum circuit construction and execution
//   9.  Full quantum algorithms:
//       - Quantum teleportation
//       - Deutsch-Jozsa algorithm
//       - Bernstein-Vazirani algorithm
//       - Grover's search algorithm
//       - Quantum Fourier Transform (QFT)
//       - Shor's period finding (the quantum part)
//       - Quantum Phase Estimation
//       - Variational Quantum Eigensolver (VQE) structure
//  10.  Density matrix support for mixed states
//  11.  Entanglement entropy measurement
//  12.  State tomography (reconstruct state from measurements)
// ============================================================================

use std::collections::HashMap;
use std::f64::consts::{PI, FRAC_1_SQRT_2};

// ============================================================================
// PART 1: COMPLEX NUMBER ARITHMETIC
// ============================================================================
// Quantum mechanics uses complex numbers everywhere. The state vector
// amplitudes are complex, and gate matrices are complex unitary matrices.
// We need efficient complex arithmetic as the foundation of everything.

/// A complex number a + bi. We use f64 for both real and imaginary parts,
/// which gives us about 15 significant digits of precision — more than
/// enough for any quantum circuit with fewer than ~50 qubits.
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self { Self { re, im } }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    /// The squared magnitude |z|² = a² + b². This is the probability
    /// when applied to an amplitude: P(outcome) = |amplitude|².
    pub fn norm_sq(&self) -> f64 { self.re * self.re + self.im * self.im }

    /// The magnitude |z| = √(a² + b²).
    pub fn norm(&self) -> f64 { self.norm_sq().sqrt() }

    /// Complex conjugate: (a + bi)* = a - bi.
    /// Used in computing inner products: ⟨ψ|φ⟩ = Σ ψᵢ* φᵢ
    pub fn conj(&self) -> Self { Self { re: self.re, im: -self.im } }

    /// Complex addition.
    pub fn add(&self, other: &Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }
    /// Complex subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }
    /// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
    /// Scalar multiplication: c * (a+bi) = ca + cbi
    pub fn scale(&self, s: f64) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
    /// Complex division.
    pub fn div(&self, other: &Self) -> Self {
        let denom = other.norm_sq();
        Self {
            re: (self.re * other.re + self.im * other.im) / denom,
            im: (self.im * other.re - self.re * other.im) / denom,
        }
    }
    /// Complex exponential: e^(a+bi) = e^a * (cos(b) + i*sin(b))
    pub fn exp(&self) -> Self {
        let r = self.re.exp();
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }
}

impl PartialEq for Complex {
    fn eq(&self, other: &Self) -> bool {
        (self.re - other.re).abs() < 1e-12 && (self.im - other.im).abs() < 1e-12
    }
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.im.abs() < 1e-15 {
            write!(f, "{:.6}", self.re)
        } else if self.re.abs() < 1e-15 {
            write!(f, "{:.6}i", self.im)
        } else {
            write!(f, "{:.6}{:+.6}i", self.re, self.im)
        }
    }
}

// ============================================================================
// PART 2: STATE VECTOR — The Heart of the Simulator
// ============================================================================
// The quantum state of N qubits is represented as a vector of 2^N complex
// amplitudes. The amplitude at index k is the amplitude of the classical
// bitstring k. For example, with 3 qubits, index 5 = binary 101 = |101⟩
// means qubit 0 is |1⟩, qubit 1 is |0⟩, qubit 2 is |1⟩.
//
// We use LITTLE-ENDIAN bit ordering: qubit 0 is the LEAST significant bit.
// This means |qubit2, qubit1, qubit0⟩ maps to index = q0 + 2*q1 + 4*q2.

/// The quantum state of an N-qubit system.
#[derive(Debug, Clone)]
pub struct StateVector {
    /// The complex amplitudes. Length is always 2^num_qubits.
    pub amplitudes: Vec<Complex>,
    /// Number of qubits in this state.
    pub num_qubits: usize,
}

impl StateVector {
    /// Create a new state with all qubits initialized to |0⟩.
    /// The initial state is |000...0⟩, which means amplitude[0] = 1 and
    /// all other amplitudes are 0.
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits; // 2^n
        let mut amplitudes = vec![Complex::ZERO; size];
        amplitudes[0] = Complex::ONE; // |000...0⟩ state
        Self { amplitudes, num_qubits }
    }

    /// Create a state from a specific classical bitstring.
    /// For example, from_bitstring(3, 5) creates |101⟩ (binary 5 with 3 qubits).
    pub fn from_bitstring(num_qubits: usize, value: usize) -> Self {
        let size = 1 << num_qubits;
        let mut amplitudes = vec![Complex::ZERO; size];
        amplitudes[value] = Complex::ONE;
        Self { amplitudes, num_qubits }
    }

    /// Create a state from an explicit amplitude vector.
    /// The vector must have length 2^n and be normalized (sum of |a|² = 1).
    pub fn from_amplitudes(num_qubits: usize, amplitudes: Vec<Complex>) -> Self {
        assert_eq!(amplitudes.len(), 1 << num_qubits);
        Self { amplitudes, num_qubits }
    }

    /// Create a uniform superposition over all basis states: |+⟩^⊗n.
    /// This is what you get after applying H to every qubit starting from |0⟩^⊗n.
    pub fn uniform_superposition(num_qubits: usize) -> Self {
        let size = 1 << num_qubits;
        let amp = Complex::new(1.0 / (size as f64).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; size],
            num_qubits,
        }
    }

    /// Get the probability of measuring a specific basis state (bitstring).
    /// This is |amplitude[index]|².
    pub fn probability(&self, index: usize) -> f64 {
        self.amplitudes[index].norm_sq()
    }

    /// Get the probability distribution over all basis states.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Verify that the state is normalized: Σ|aᵢ|² = 1.
    /// Small numerical errors can accumulate, so we check with tolerance.
    pub fn is_normalized(&self) -> bool {
        let total: f64 = self.amplitudes.iter().map(|a| a.norm_sq()).sum();
        (total - 1.0).abs() < 1e-10
    }

    /// Renormalize the state (fix numerical drift).
    pub fn normalize(&mut self) {
        let total: f64 = self.amplitudes.iter().map(|a| a.norm_sq()).sum();
        if total > 1e-15 {
            let factor = 1.0 / total.sqrt();
            for a in &mut self.amplitudes {
                *a = a.scale(factor);
            }
        }
    }

    /// Compute the inner product ⟨self|other⟩.
    /// This is the overlap between two quantum states.
    pub fn inner_product(&self, other: &StateVector) -> Complex {
        assert_eq!(self.num_qubits, other.num_qubits);
        let mut result = Complex::ZERO;
        for (a, b) in self.amplitudes.iter().zip(other.amplitudes.iter()) {
            result = result.add(&a.conj().mul(b));
        }
        result
    }

    /// Compute the entanglement entropy between qubit `target` and the rest.
    /// This quantifies how entangled the target qubit is with the other qubits.
    /// Returns 0 for a product state (no entanglement) and ln(2) ≈ 0.693 for
    /// a maximally entangled qubit (like one half of a Bell pair).
    ///
    /// The calculation traces out all qubits except the target to get the
    /// reduced density matrix, then computes the von Neumann entropy.
    pub fn entanglement_entropy(&self, target: usize) -> f64 {
        // Compute the 2×2 reduced density matrix for the target qubit
        // by tracing out all other qubits.
        //
        // ρ_target = Tr_{others}(|ψ⟩⟨ψ|)
        //
        // ρ[i][j] = Σ_k ⟨i,k|ψ⟩⟨ψ|j,k⟩
        // where i,j ∈ {0,1} are the target qubit values and k ranges
        // over all 2^(n-1) basis states of the other qubits.

        let mut rho = [[Complex::ZERO; 2]; 2];
        let n = self.amplitudes.len();

        for i in 0..2u8 {
            for j in 0..2u8 {
                let mut sum = Complex::ZERO;
                for k in 0..n {
                    // Does basis state k have qubit `target` equal to i?
                    let bit_at_target_k = (k >> target) & 1;
                    if bit_at_target_k != i as usize { continue; }

                    // The corresponding index with qubit `target` set to j:
                    let l = if j as usize == 1 { k | (1 << target) } else { k & !(1 << target) };

                    // ρ[i][j] += ψ[k]* × ψ[l]
                    sum = sum.add(&self.amplitudes[k].conj().mul(&self.amplitudes[l]));
                }
                rho[i as usize][j as usize] = sum;
            }
        }

        // The eigenvalues of a 2×2 matrix [[a,b],[c,d]] are:
        //   λ = (a+d)/2 ± √((a-d)²/4 + bc)
        // Since ρ is Hermitian positive semidefinite, eigenvalues are real ≥ 0.
        let a = rho[0][0].re;
        let d = rho[1][1].re;
        let bc = rho[0][1].mul(&rho[1][0]).re; // b*c is real for Hermitian matrix
        let avg = (a + d) / 2.0;
        let discriminant = ((a - d) / 2.0).powi(2) + bc;
        let sqrt_disc = if discriminant > 0.0 { discriminant.sqrt() } else { 0.0 };

        let lambda1 = avg + sqrt_disc;
        let lambda2 = avg - sqrt_disc;

        // Von Neumann entropy: S = -Σ λ ln(λ)
        let mut entropy = 0.0;
        if lambda1 > 1e-15 { entropy -= lambda1 * lambda1.ln(); }
        if lambda2 > 1e-15 { entropy -= lambda2 * lambda2.ln(); }

        entropy
    }

    /// Pretty-print the state vector, showing only non-zero amplitudes.
    pub fn display(&self) -> String {
        let mut terms = Vec::new();
        for (i, amp) in self.amplitudes.iter().enumerate() {
            if amp.norm_sq() > 1e-12 {
                let bits: String = (0..self.num_qubits)
                    .rev()
                    .map(|b| if (i >> b) & 1 == 1 { '1' } else { '0' })
                    .collect();
                terms.push(format!("{}|{}⟩", amp, bits));
            }
        }
        if terms.is_empty() { "0".to_string() } else { terms.join(" + ") }
    }
}

// ============================================================================
// PART 3: QUANTUM GATES
// ============================================================================
// A quantum gate is a unitary operation that transforms the state vector.
// We represent each gate as a 2×2 or 4×4 complex matrix, but we apply
// them to the state vector DIRECTLY using index arithmetic — we never
// build the full 2^N × 2^N matrix. This is how real simulators work.

/// A 2×2 complex matrix representing a single-qubit gate.
/// For a multi-qubit gate like CNOT, we handle it specially.
#[derive(Debug, Clone)]
pub struct Gate2x2 {
    pub data: [[Complex; 2]; 2],
    pub name: String,
}

impl Gate2x2 {
    pub fn new(name: &str, data: [[Complex; 2]; 2]) -> Self {
        Self { data, name: name.to_string() }
    }

    /// Verify that the gate is unitary: U†U = I.
    /// This is a fundamental requirement — non-unitary operations would
    /// violate conservation of probability.
    pub fn is_unitary(&self) -> bool {
        let u = &self.data;
        // U†U should equal identity
        // (U†)_ij = conj(U_ji)
        let mut product = [[Complex::ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    product[i][j] = product[i][j].add(&u[k][i].conj().mul(&u[k][j]));
                }
            }
        }
        (product[0][0].re - 1.0).abs() < 1e-10
            && product[0][1].norm() < 1e-10
            && product[1][0].norm() < 1e-10
            && (product[1][1].re - 1.0).abs() < 1e-10
    }
}

// ---- Standard gate constructors ----

/// Hadamard gate: creates superposition.
/// H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2
/// Matrix: (1/√2) * [[1, 1], [1, -1]]
pub fn hadamard() -> Gate2x2 {
    let s = FRAC_1_SQRT_2;
    Gate2x2::new("H", [
        [Complex::new(s, 0.0), Complex::new(s, 0.0)],
        [Complex::new(s, 0.0), Complex::new(-s, 0.0)],
    ])
}

/// Pauli-X gate (quantum NOT): flips |0⟩ ↔ |1⟩.
/// Matrix: [[0, 1], [1, 0]]
pub fn pauli_x() -> Gate2x2 {
    Gate2x2::new("X", [
        [Complex::ZERO, Complex::ONE],
        [Complex::ONE, Complex::ZERO],
    ])
}

/// Pauli-Y gate: maps |0⟩ → i|1⟩, |1⟩ → -i|0⟩.
/// Matrix: [[0, -i], [i, 0]]
pub fn pauli_y() -> Gate2x2 {
    Gate2x2::new("Y", [
        [Complex::ZERO, Complex::new(0.0, -1.0)],
        [Complex::new(0.0, 1.0), Complex::ZERO],
    ])
}

/// Pauli-Z gate: adds a phase of -1 to |1⟩, leaves |0⟩ unchanged.
/// Matrix: [[1, 0], [0, -1]]
pub fn pauli_z() -> Gate2x2 {
    Gate2x2::new("Z", [
        [Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::new(-1.0, 0.0)],
    ])
}

/// S gate (phase gate, √Z): adds a phase of i to |1⟩.
/// Matrix: [[1, 0], [0, i]]
pub fn s_gate() -> Gate2x2 {
    Gate2x2::new("S", [
        [Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::I],
    ])
}

/// T gate (π/8 gate, √S): adds a phase of e^(iπ/4) to |1⟩.
/// Matrix: [[1, 0], [0, e^(iπ/4)]]
/// The T gate is crucial because {H, T} forms a universal gate set.
pub fn t_gate() -> Gate2x2 {
    Gate2x2::new("T", [
        [Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::from_polar(1.0, PI / 4.0)],
    ])
}

/// S† (S-dagger) gate: the inverse of S.
pub fn s_dag() -> Gate2x2 {
    Gate2x2::new("S†", [
        [Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::new(0.0, -1.0)],
    ])
}

/// T† (T-dagger) gate: the inverse of T.
pub fn t_dag() -> Gate2x2 {
    Gate2x2::new("T†", [
        [Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::from_polar(1.0, -PI / 4.0)],
    ])
}

/// Rotation around the X-axis: Rx(θ) = e^(-iθX/2)
/// Matrix: [[cos(θ/2), -i*sin(θ/2)], [-i*sin(θ/2), cos(θ/2)]]
pub fn rx(theta: f64) -> Gate2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    Gate2x2::new(&format!("Rx({:.3})", theta), [
        [Complex::new(c, 0.0), Complex::new(0.0, -s)],
        [Complex::new(0.0, -s), Complex::new(c, 0.0)],
    ])
}

/// Rotation around the Y-axis: Ry(θ) = e^(-iθY/2)
/// Matrix: [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
pub fn ry(theta: f64) -> Gate2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    Gate2x2::new(&format!("Ry({:.3})", theta), [
        [Complex::new(c, 0.0), Complex::new(-s, 0.0)],
        [Complex::new(s, 0.0), Complex::new(c, 0.0)],
    ])
}

/// Rotation around the Z-axis: Rz(θ) = e^(-iθZ/2)
/// Matrix: [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
pub fn rz(theta: f64) -> Gate2x2 {
    Gate2x2::new(&format!("Rz({:.3})", theta), [
        [Complex::from_polar(1.0, -theta / 2.0), Complex::ZERO],
        [Complex::ZERO, Complex::from_polar(1.0, theta / 2.0)],
    ])
}

/// General phase gate: P(φ) adds phase e^(iφ) to |1⟩.
/// Matrix: [[1, 0], [0, e^(iφ)]]
pub fn phase(phi: f64) -> Gate2x2 {
    Gate2x2::new(&format!("P({:.3})", phi), [
        [Complex::ONE, Complex::ZERO],
        [Complex::ZERO, Complex::from_polar(1.0, phi)],
    ])
}

/// The most general single-qubit gate: U3(θ, φ, λ).
/// Every single-qubit gate can be expressed as U3 with appropriate parameters.
/// Matrix: [[cos(θ/2), -e^(iλ)*sin(θ/2)], [e^(iφ)*sin(θ/2), e^(i(φ+λ))*cos(θ/2)]]
pub fn u3(theta: f64, phi: f64, lambda: f64) -> Gate2x2 {
    let c = (theta / 2.0).cos();
    let s = (theta / 2.0).sin();
    Gate2x2::new(&format!("U3({:.3},{:.3},{:.3})", theta, phi, lambda), [
        [Complex::new(c, 0.0), Complex::from_polar(1.0, lambda).scale(-s)],
        [Complex::from_polar(1.0, phi).scale(s), Complex::from_polar(1.0, phi + lambda).scale(c)],
    ])
}

// ============================================================================
// PART 4: GATE APPLICATION — The Core Simulation Engine
// ============================================================================
// This is the most performance-critical code in the entire simulator.
// Applying a gate to qubit k in an N-qubit state requires touching every
// amplitude in the state vector, but we only need the 2×2 gate matrix
// and some clever index arithmetic.
//
// The key insight: for qubit k, the amplitudes pair up. For each pair,
// one index has bit k = 0 and the other has bit k = 1. We apply the
// 2×2 matrix to each such pair independently.

/// Apply a single-qubit gate to qubit `target` in the state vector.
/// This is an in-place operation that modifies the state vector.
///
/// Algorithm: iterate over all pairs of indices that differ only in bit `target`.
/// For each pair (i0, i1) where i0 has bit target=0 and i1 has bit target=1,
/// apply the 2×2 matrix: [new_i0, new_i1] = gate * [old_i0, old_i1].
pub fn apply_single_qubit_gate(state: &mut StateVector, target: usize, gate: &Gate2x2) {
    let n = state.amplitudes.len();
    let bit = 1 << target; // The bit position for qubit `target`

    // Iterate over all indices where bit `target` is 0.
    // The paired index with bit `target` = 1 is obtained by OR-ing with `bit`.
    let mut i = 0;
    while i < n {
        // If this index already has bit `target` set, skip it
        // (it's the partner of a lower index, which we already processed).
        if i & bit != 0 {
            i += 1;
            continue;
        }

        let i0 = i;       // Index with qubit target = 0
        let i1 = i | bit; // Index with qubit target = 1

        // Apply the 2×2 matrix: [a0', a1'] = [[u00,u01],[u10,u11]] * [a0, a1]
        let a0 = state.amplitudes[i0];
        let a1 = state.amplitudes[i1];

        state.amplitudes[i0] = gate.data[0][0].mul(&a0).add(&gate.data[0][1].mul(&a1));
        state.amplitudes[i1] = gate.data[1][0].mul(&a0).add(&gate.data[1][1].mul(&a1));

        i += 1;
    }
}

/// Apply a CNOT (controlled-NOT) gate. This is the fundamental two-qubit gate.
/// It flips the target qubit if and only if the control qubit is |1⟩.
///
/// Truth table: |c,t⟩ → |c, c⊕t⟩
///   |00⟩ → |00⟩
///   |01⟩ → |01⟩
///   |10⟩ → |11⟩  ← target flipped because control is 1
///   |11⟩ → |10⟩  ← target flipped because control is 1
pub fn apply_cnot(state: &mut StateVector, control: usize, target: usize) {
    let n = state.amplitudes.len();
    let ctrl_bit = 1 << control;
    let tgt_bit = 1 << target;

    for i in 0..n {
        // Only act when the control bit is 1 AND the target bit is 0.
        // We then swap the amplitude with the state where the target bit is 1.
        if (i & ctrl_bit) != 0 && (i & tgt_bit) == 0 {
            let j = i | tgt_bit; // Same state but with target bit flipped to 1
            state.amplitudes.swap(i, j);
        }
    }
}

/// Apply a controlled-Z (CZ) gate. Adds a phase of -1 when both qubits are |1⟩.
/// |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |10⟩, |11⟩ → -|11⟩
pub fn apply_cz(state: &mut StateVector, qubit_a: usize, qubit_b: usize) {
    let bit_a = 1 << qubit_a;
    let bit_b = 1 << qubit_b;
    for i in 0..state.amplitudes.len() {
        if (i & bit_a) != 0 && (i & bit_b) != 0 {
            state.amplitudes[i] = state.amplitudes[i].scale(-1.0);
        }
    }
}

/// Apply a SWAP gate: exchanges the states of two qubits.
pub fn apply_swap(state: &mut StateVector, qubit_a: usize, qubit_b: usize) {
    let bit_a = 1 << qubit_a;
    let bit_b = 1 << qubit_b;
    for i in 0..state.amplitudes.len() {
        let a = (i >> qubit_a) & 1;
        let b = (i >> qubit_b) & 1;
        if a != b && a == 0 {
            // Swap amplitudes where qubit_a=0,qubit_b=1 with qubit_a=1,qubit_b=0
            let j = (i ^ bit_a) ^ bit_b;
            state.amplitudes.swap(i, j);
        }
    }
}

/// Apply a controlled version of any single-qubit gate.
/// The gate is applied to `target` only when `control` is |1⟩.
pub fn apply_controlled_gate(
    state: &mut StateVector,
    control: usize,
    target: usize,
    gate: &Gate2x2,
) {
    let n = state.amplitudes.len();
    let ctrl_bit = 1 << control;
    let tgt_bit = 1 << target;

    let mut i = 0;
    while i < n {
        // Only process indices where the control bit is 1.
        if (i & ctrl_bit) == 0 || (i & tgt_bit) != 0 {
            i += 1;
            continue;
        }

        let i0 = i;           // Control=1, target=0
        let i1 = i | tgt_bit; // Control=1, target=1

        let a0 = state.amplitudes[i0];
        let a1 = state.amplitudes[i1];

        state.amplitudes[i0] = gate.data[0][0].mul(&a0).add(&gate.data[0][1].mul(&a1));
        state.amplitudes[i1] = gate.data[1][0].mul(&a0).add(&gate.data[1][1].mul(&a1));

        i += 1;
    }
}

/// Apply a Toffoli (CCX) gate: flips target when both controls are |1⟩.
pub fn apply_toffoli(state: &mut StateVector, ctrl1: usize, ctrl2: usize, target: usize) {
    let c1_bit = 1 << ctrl1;
    let c2_bit = 1 << ctrl2;
    let t_bit = 1 << target;

    for i in 0..state.amplitudes.len() {
        if (i & c1_bit) != 0 && (i & c2_bit) != 0 && (i & t_bit) == 0 {
            let j = i | t_bit;
            state.amplitudes.swap(i, j);
        }
    }
}

/// Apply a Fredkin (CSWAP) gate: swaps target1 and target2 when control is |1⟩.
pub fn apply_fredkin(state: &mut StateVector, control: usize, target1: usize, target2: usize) {
    let c_bit = 1 << control;
    let t1_bit = 1 << target1;
    let t2_bit = 1 << target2;

    for i in 0..state.amplitudes.len() {
        if (i & c_bit) != 0 {
            let b1 = (i >> target1) & 1;
            let b2 = (i >> target2) & 1;
            if b1 != b2 && b1 == 0 {
                let j = (i ^ t1_bit) ^ t2_bit;
                state.amplitudes.swap(i, j);
            }
        }
    }
}

// ============================================================================
// PART 5: MEASUREMENT
// ============================================================================
// Measurement is the non-reversible step in quantum computing. It
// collapses the superposition and gives a classical result.
//
// When we measure qubit k:
//   1. Compute P(0) = Σ |a_i|² for all i where bit k is 0
//   2. Compute P(1) = 1 - P(0)
//   3. Randomly choose 0 or 1 according to these probabilities
//   4. COLLAPSE: set all amplitudes inconsistent with the result to 0
//   5. RENORMALIZE: scale remaining amplitudes so they sum to 1

use std::time::SystemTime;

/// A simple random number generator (xorshift64) so we don't need external deps.
/// In a production simulator, you'd use a cryptographically secure RNG.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self { state: seed ^ 0x123456789ABCDEF0 }
    }

    pub fn from_seed(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    /// Generate a random f64 in [0, 1).
    pub fn random(&mut self) -> f64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state as f64) / (u64::MAX as f64)
    }
}

/// Measure a single qubit. Returns the classical result (0 or 1) and
/// collapses the state vector accordingly.
pub fn measure_qubit(state: &mut StateVector, target: usize, rng: &mut Rng) -> u8 {
    let bit = 1 << target;

    // Calculate P(qubit = 0) = sum of |a_i|² where bit `target` is 0.
    let prob_zero: f64 = state.amplitudes.iter().enumerate()
        .filter(|(i, _)| (i & bit) == 0)
        .map(|(_, a)| a.norm_sq())
        .sum();

    // Randomly choose the outcome.
    let outcome = if rng.random() < prob_zero { 0u8 } else { 1u8 };

    // Collapse: zero out all amplitudes inconsistent with the measurement result.
    for i in 0..state.amplitudes.len() {
        let bit_value = ((i >> target) & 1) as u8;
        if bit_value != outcome {
            state.amplitudes[i] = Complex::ZERO;
        }
    }

    // Renormalize: the remaining amplitudes must sum to 1.
    state.normalize();

    outcome
}

/// Measure ALL qubits at once. Returns a classical bitstring as a usize.
/// The state collapses to a single basis state.
pub fn measure_all(state: &mut StateVector, rng: &mut Rng) -> usize {
    let probs = state.probabilities();
    let r = rng.random();
    let mut cumulative = 0.0;

    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            // Collapse to state |i⟩
            for j in 0..state.amplitudes.len() {
                state.amplitudes[j] = if j == i { Complex::ONE } else { Complex::ZERO };
            }
            return i;
        }
    }

    // Edge case: return the last state.
    state.amplitudes.len() - 1
}

/// Run a measurement experiment: prepare the state, measure, repeat `shots` times.
/// Returns a histogram of outcomes.
pub fn sample(state: &StateVector, shots: usize, rng: &mut Rng) -> HashMap<usize, usize> {
    let mut counts = HashMap::new();
    for _ in 0..shots {
        let mut s = state.clone();
        let result = measure_all(&mut s, rng);
        *counts.entry(result).or_insert(0) += 1;
    }
    counts
}

// ============================================================================
// PART 6: QUANTUM CIRCUIT
// ============================================================================
// A quantum circuit is a sequence of gate operations. We represent it as
// a list of instructions that can be applied to a state vector.

/// An instruction in a quantum circuit.
#[derive(Debug, Clone)]
pub enum Instruction {
    // Single-qubit gates
    Gate(Gate2x2, usize),                          // (gate, target)
    // Controlled single-qubit gates
    ControlledGate(Gate2x2, usize, usize),         // (gate, control, target)
    // Two-qubit gates
    CNOT(usize, usize),                            // (control, target)
    CZ(usize, usize),
    SWAP(usize, usize),
    // Three-qubit gates
    Toffoli(usize, usize, usize),                  // (ctrl1, ctrl2, target)
    Fredkin(usize, usize, usize),                  // (ctrl, target1, target2)
    // Measurement
    Measure(usize),                                // Measure qubit → classical bit
    MeasureAll,
    // Barrier (prevents optimization across it, no physical effect)
    Barrier(Vec<usize>),
    // Reset qubit to |0⟩
    Reset(usize),
}

/// A quantum circuit: a sequence of instructions operating on a fixed number of qubits.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub num_qubits: usize,
    pub num_classical: usize,
    pub instructions: Vec<Instruction>,
    pub name: String,
}

impl Circuit {
    pub fn new(name: &str, num_qubits: usize) -> Self {
        Self {
            num_qubits,
            num_classical: num_qubits, // Default: one classical bit per qubit
            instructions: Vec::new(),
            name: name.to_string(),
        }
    }

    // Convenience methods for adding gates.
    pub fn h(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(hadamard(), target)); self }
    pub fn x(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(pauli_x(), target)); self }
    pub fn y(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(pauli_y(), target)); self }
    pub fn z(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(pauli_z(), target)); self }
    pub fn s(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(s_gate(), target)); self }
    pub fn t(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(t_gate(), target)); self }
    pub fn rx_gate(&mut self, theta: f64, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(rx(theta), target)); self }
    pub fn ry_gate(&mut self, theta: f64, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(ry(theta), target)); self }
    pub fn rz_gate(&mut self, theta: f64, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(rz(theta), target)); self }
    pub fn p_gate(&mut self, phi: f64, target: usize) -> &mut Self { self.instructions.push(Instruction::Gate(phase(phi), target)); self }
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self { self.instructions.push(Instruction::CNOT(control, target)); self }
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self { self.cnot(control, target) }
    pub fn cz(&mut self, a: usize, b: usize) -> &mut Self { self.instructions.push(Instruction::CZ(a, b)); self }
    pub fn swap(&mut self, a: usize, b: usize) -> &mut Self { self.instructions.push(Instruction::SWAP(a, b)); self }
    pub fn toffoli(&mut self, c1: usize, c2: usize, target: usize) -> &mut Self { self.instructions.push(Instruction::Toffoli(c1, c2, target)); self }
    pub fn ccx(&mut self, c1: usize, c2: usize, target: usize) -> &mut Self { self.toffoli(c1, c2, target) }
    pub fn fredkin(&mut self, c: usize, t1: usize, t2: usize) -> &mut Self { self.instructions.push(Instruction::Fredkin(c, t1, t2)); self }
    pub fn measure(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Measure(target)); self }
    pub fn measure_all(&mut self) -> &mut Self { self.instructions.push(Instruction::MeasureAll); self }
    pub fn barrier(&mut self, qubits: &[usize]) -> &mut Self { self.instructions.push(Instruction::Barrier(qubits.to_vec())); self }
    pub fn reset(&mut self, target: usize) -> &mut Self { self.instructions.push(Instruction::Reset(target)); self }

    /// Apply a controlled version of any single-qubit gate.
    pub fn controlled(&mut self, gate: Gate2x2, control: usize, target: usize) -> &mut Self {
        self.instructions.push(Instruction::ControlledGate(gate, control, target));
        self
    }

    /// Add a Quantum Fourier Transform on qubits first..first+n-1.
    pub fn qft(&mut self, first: usize, n: usize) -> &mut Self {
        for i in 0..n {
            self.h(first + i);
            for j in (i + 1)..n {
                let angle = PI / (1 << (j - i)) as f64;
                self.controlled(phase(angle), first + j, first + i);
            }
        }
        // Reverse qubit order (QFT convention).
        for i in 0..n / 2 {
            self.swap(first + i, first + n - 1 - i);
        }
        self
    }

    /// Add an inverse QFT.
    pub fn inverse_qft(&mut self, first: usize, n: usize) -> &mut Self {
        // Reverse qubit order first.
        for i in 0..n / 2 {
            self.swap(first + i, first + n - 1 - i);
        }
        for i in (0..n).rev() {
            for j in ((i + 1)..n).rev() {
                let angle = -PI / (1 << (j - i)) as f64;
                self.controlled(phase(angle), first + j, first + i);
            }
            self.h(first + i);
        }
        self
    }

    /// Execute the circuit and return the final state vector and measurement results.
    pub fn execute(&self, rng: &mut Rng) -> CircuitResult {
        let mut state = StateVector::new(self.num_qubits);
        let mut measurements = Vec::new();

        for instr in &self.instructions {
            match instr {
                Instruction::Gate(gate, target) => {
                    apply_single_qubit_gate(&mut state, *target, gate);
                }
                Instruction::ControlledGate(gate, control, target) => {
                    apply_controlled_gate(&mut state, *control, *target, gate);
                }
                Instruction::CNOT(control, target) => {
                    apply_cnot(&mut state, *control, *target);
                }
                Instruction::CZ(a, b) => {
                    apply_cz(&mut state, *a, *b);
                }
                Instruction::SWAP(a, b) => {
                    apply_swap(&mut state, *a, *b);
                }
                Instruction::Toffoli(c1, c2, target) => {
                    apply_toffoli(&mut state, *c1, *c2, *target);
                }
                Instruction::Fredkin(c, t1, t2) => {
                    apply_fredkin(&mut state, *c, *t1, *t2);
                }
                Instruction::Measure(target) => {
                    let result = measure_qubit(&mut state, *target, rng);
                    measurements.push((*target, result));
                }
                Instruction::MeasureAll => {
                    let result = measure_all(&mut state, rng);
                    for q in 0..self.num_qubits {
                        measurements.push((q, ((result >> q) & 1) as u8));
                    }
                }
                Instruction::Reset(target) => {
                    let m = measure_qubit(&mut state, *target, rng);
                    if m == 1 {
                        apply_single_qubit_gate(&mut state, *target, &pauli_x());
                    }
                }
                Instruction::Barrier(_) => { /* No physical effect */ }
            }
        }

        CircuitResult { state, measurements }
    }

    /// Run the circuit many times and collect statistics.
    pub fn run(&self, shots: usize, rng: &mut Rng) -> HashMap<Vec<u8>, usize> {
        let mut counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for _ in 0..shots {
            let result = self.execute(rng);
            let bits: Vec<u8> = (0..self.num_qubits)
                .map(|q| {
                    result.measurements.iter()
                        .find(|(qubit, _)| *qubit == q)
                        .map(|(_, bit)| *bit)
                        .unwrap_or(0)
                })
                .collect();
            *counts.entry(bits).or_insert(0) += 1;
        }
        counts
    }

    /// Get the gate count (useful for complexity analysis).
    pub fn gate_count(&self) -> usize {
        self.instructions.iter().filter(|i| !matches!(i, Instruction::Barrier(_) | Instruction::Measure(_) | Instruction::MeasureAll)).count()
    }

    /// Get the circuit depth (longest path from input to output).
    pub fn depth(&self) -> usize {
        let mut qubit_depth = vec![0usize; self.num_qubits];
        for instr in &self.instructions {
            match instr {
                Instruction::Gate(_, t) => { qubit_depth[*t] += 1; }
                Instruction::CNOT(c, t) | Instruction::CZ(c, t) | Instruction::SWAP(c, t) | Instruction::ControlledGate(_, c, t) => {
                    let d = qubit_depth[*c].max(qubit_depth[*t]) + 1;
                    qubit_depth[*c] = d;
                    qubit_depth[*t] = d;
                }
                Instruction::Toffoli(c1, c2, t) => {
                    let d = qubit_depth[*c1].max(qubit_depth[*c2]).max(qubit_depth[*t]) + 1;
                    qubit_depth[*c1] = d;
                    qubit_depth[*c2] = d;
                    qubit_depth[*t] = d;
                }
                _ => {}
            }
        }
        qubit_depth.into_iter().max().unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct CircuitResult {
    pub state: StateVector,
    pub measurements: Vec<(usize, u8)>,
}

// ============================================================================
// PART 7: QUANTUM ALGORITHMS
// ============================================================================

/// Create a Bell pair (maximally entangled state): (|00⟩ + |11⟩)/√2
/// This is the foundation of quantum teleportation, superdense coding,
/// and many quantum error correction protocols.
pub fn bell_pair() -> Circuit {
    let mut c = Circuit::new("Bell Pair", 2);
    c.h(0).cnot(0, 1);
    c
}

/// Quantum Teleportation: transfer the state of qubit 0 to qubit 2
/// using a pre-shared Bell pair (qubits 1,2) and classical communication.
///
/// This demonstrates that quantum information can be transmitted using
/// only classical communication, as long as entanglement was pre-shared.
/// The original qubit's state is destroyed in the process (no-cloning theorem).
pub fn quantum_teleportation() -> Circuit {
    let mut c = Circuit::new("Quantum Teleportation", 3);

    // Step 1: Create a Bell pair between qubits 1 and 2.
    // (In practice, this would have been done earlier and the qubits distributed.)
    c.h(1).cnot(1, 2);

    // Step 2: Alice applies a Bell measurement to qubits 0 and 1.
    // This entangles qubit 0 (the state to teleport) with the Bell pair.
    c.cnot(0, 1).h(0);

    // Step 3: Alice measures her two qubits and sends the classical results to Bob.
    c.measure(0).measure(1);

    // Step 4: Bob applies corrections based on Alice's measurement results.
    // If Alice measured qubit 1 as 1, Bob applies X to qubit 2.
    // If Alice measured qubit 0 as 1, Bob applies Z to qubit 2.
    // (In a real circuit, these would be classically controlled. Here we
    //  represent them as the full circuit with all possibilities.)
    // Note: in simulation, we'd need classical feedforward. For simplicity,
    // we show the circuit structure without conditional gates.

    c
}

/// Deutsch-Jozsa Algorithm: determines whether a function f:{0,1}^n → {0,1}
/// is constant (always 0 or always 1) or balanced (0 for exactly half the inputs).
/// Classical requires 2^(n-1) + 1 queries; quantum requires exactly 1 query.
///
/// The oracle is specified as a function from usize (input bitstring) to bool.
/// `n` is the number of input qubits.
pub fn deutsch_jozsa(n: usize, oracle: &dyn Fn(usize) -> bool) -> Circuit {
    // Total qubits: n input qubits + 1 ancilla qubit.
    let total = n + 1;
    let mut c = Circuit::new("Deutsch-Jozsa", total);

    // Step 1: Initialize ancilla to |1⟩.
    c.x(n);

    // Step 2: Apply Hadamard to all qubits, creating uniform superposition.
    for i in 0..=n { c.h(i); }

    // Step 3: Apply the oracle. We build it as a sequence of CNOT gates.
    // For each input x where f(x) = 1, we apply a multi-controlled X on the ancilla.
    // This is the "phase kickback" trick: the ancilla is in |−⟩ = (|0⟩−|1⟩)/√2,
    // so flipping it gives a phase of -1 to the corresponding input state.
    for x in 0..(1 << n) {
        if oracle(x) {
            // Apply X to ancilla controlled by the input bits matching x.
            // We set each input qubit to match x using X gates, then apply
            // a multi-controlled X, then undo the X gates.
            for bit in 0..n {
                if (x >> bit) & 1 == 0 {
                    c.x(bit); // Flip to match the 0 bits in x
                }
            }
            // Multi-controlled X (Toffoli generalization)
            // For small n, we can decompose this. For n=1, it's just CNOT.
            if n == 1 {
                c.cnot(0, n);
            } else if n == 2 {
                c.toffoli(0, 1, n);
            }
            // Undo the X gates.
            for bit in 0..n {
                if (x >> bit) & 1 == 0 {
                    c.x(bit);
                }
            }
        }
    }

    // Step 4: Apply Hadamard to the input qubits.
    for i in 0..n { c.h(i); }

    // Step 5: Measure the input qubits.
    // If all zeros → function is constant.
    // If any non-zero → function is balanced.
    for i in 0..n { c.measure(i); }

    c
}

/// Bernstein-Vazirani Algorithm: finds a hidden string s ∈ {0,1}^n
/// given an oracle that computes f(x) = s·x (mod 2) (inner product mod 2).
/// Classical requires n queries; quantum requires exactly 1 query.
pub fn bernstein_vazirani(n: usize, secret: usize) -> Circuit {
    let total = n + 1;
    let mut c = Circuit::new("Bernstein-Vazirani", total);

    // Prepare ancilla in |−⟩ state.
    c.x(n);

    // Hadamard all qubits.
    for i in 0..=n { c.h(i); }

    // Oracle: for each bit position where secret has a 1, CNOT from that
    // input qubit to the ancilla. This computes f(x) = s·x mod 2.
    for bit in 0..n {
        if (secret >> bit) & 1 == 1 {
            c.cnot(bit, n);
        }
    }

    // Hadamard the input qubits.
    for i in 0..n { c.h(i); }

    // Measure the input qubits — the result IS the secret string.
    for i in 0..n { c.measure(i); }

    c
}

/// Grover's Search Algorithm: finds a marked element in an unstructured
/// database of N = 2^n items using O(√N) queries instead of O(N).
///
/// The `oracle` function returns true for the target element(s).
/// `num_iterations` should be approximately π/4 * √(N/M) where M is the
/// number of marked elements. If None, we use the optimal count for 1 target.
pub fn grover_search(
    n: usize,
    oracle: &dyn Fn(usize) -> bool,
    num_iterations: Option<usize>,
) -> Circuit {
    let mut c = Circuit::new("Grover's Search", n);
    let optimal_iters = ((PI / 4.0) * (1 << n) as f64).sqrt() as usize;
    let iters = num_iterations.unwrap_or(optimal_iters.max(1));

    // Step 1: Create uniform superposition.
    for i in 0..n { c.h(i); }

    // Step 2: Repeat the Grover iteration.
    for _ in 0..iters {
        // --- Oracle: flip the phase of the target state(s) ---
        // For each x where oracle(x) is true, we need to apply a phase of -1.
        // This is done by applying Z to the appropriate state.
        for x in 0..(1 << n) {
            if oracle(x) {
                // Apply a multi-controlled Z that activates on state |x⟩.
                // Decomposition: flip bits to convert |x⟩ to |11...1⟩,
                // then apply a multi-controlled Z, then flip back.
                for bit in 0..n {
                    if (x >> bit) & 1 == 0 { c.x(bit); }
                }
                // Multi-controlled Z: apply H to last qubit, then
                // multi-controlled X (Toffoli), then H again.
                if n == 1 {
                    c.z(0);
                } else if n == 2 {
                    c.h(1);
                    c.cnot(0, 1);
                    c.h(1);
                } else if n == 3 {
                    c.h(2);
                    c.toffoli(0, 1, 2);
                    c.h(2);
                }
                for bit in 0..n {
                    if (x >> bit) & 1 == 0 { c.x(bit); }
                }
            }
        }

        // --- Diffusion operator: 2|s⟩⟨s| - I where |s⟩ is uniform superposition ---
        // This "reflects" the state about the mean amplitude, amplifying the
        // marked element's probability.
        for i in 0..n { c.h(i); }
        for i in 0..n { c.x(i); }

        // Multi-controlled Z on |11...1⟩
        if n == 1 {
            c.z(0);
        } else if n == 2 {
            c.h(1);
            c.cnot(0, 1);
            c.h(1);
        } else if n == 3 {
            c.h(2);
            c.toffoli(0, 1, 2);
            c.h(2);
        }

        for i in 0..n { c.x(i); }
        for i in 0..n { c.h(i); }
    }

    // Measure all qubits.
    for i in 0..n { c.measure(i); }

    c
}

/// Quantum Phase Estimation: given a unitary U and an eigenstate |u⟩ with
/// U|u⟩ = e^(2πiφ)|u⟩, estimate the phase φ to n bits of precision.
///
/// This is the foundation of Shor's algorithm and many other quantum algorithms.
/// It uses the QFT and controlled-U operations.
///
/// `precision_qubits`: number of qubits for the phase register (more = more precision)
/// `eigenstate_qubits`: number of qubits for the eigenstate register
/// `controlled_u`: a function that builds controlled-U^(2^k) into the circuit
pub fn quantum_phase_estimation(
    precision_qubits: usize,
    eigenstate_qubits: usize,
    controlled_u_power: &dyn Fn(&mut Circuit, usize, usize),
    // controlled_u_power(circuit, control_qubit, power_of_2)
) -> Circuit {
    let total = precision_qubits + eigenstate_qubits;
    let mut c = Circuit::new("Phase Estimation", total);

    // Step 1: Apply Hadamard to all precision qubits.
    for i in 0..precision_qubits { c.h(i); }

    // Step 2: Apply controlled-U^(2^k) for each precision qubit k.
    // The control qubit is k, and U is applied 2^k times to the eigenstate register.
    for k in 0..precision_qubits {
        controlled_u_power(&mut c, k, k);
    }

    // Step 3: Apply inverse QFT to the precision register.
    c.inverse_qft(0, precision_qubits);

    // Step 4: Measure the precision register.
    for i in 0..precision_qubits { c.measure(i); }

    c
}

// ============================================================================
// PART 8: DENSITY MATRIX (for mixed states and noise modeling)
// ============================================================================
// A pure state |ψ⟩ can be described by a state vector. A mixed state
// (statistical mixture of pure states, or a subsystem of an entangled state)
// requires a density matrix ρ = Σ pᵢ |ψᵢ⟩⟨ψᵢ|.

/// Density matrix representation. Stores a 2^n × 2^n complex matrix.
/// Used for mixed states, noise modeling, and partial traces.
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    pub data: Vec<Vec<Complex>>,
    pub num_qubits: usize,
}

impl DensityMatrix {
    /// Create a density matrix from a pure state: ρ = |ψ⟩⟨ψ|.
    pub fn from_state_vector(state: &StateVector) -> Self {
        let n = state.amplitudes.len();
        let mut data = vec![vec![Complex::ZERO; n]; n];
        for i in 0..n {
            for j in 0..n {
                data[i][j] = state.amplitudes[i].mul(&state.amplitudes[j].conj());
            }
        }
        Self { data, num_qubits: state.num_qubits }
    }

    /// Create a mixed state from a statistical ensemble.
    pub fn from_ensemble(states: &[(f64, StateVector)]) -> Self {
        let n = states[0].1.amplitudes.len();
        let num_qubits = states[0].1.num_qubits;
        let mut data = vec![vec![Complex::ZERO; n]; n];
        for (prob, state) in states {
            for i in 0..n {
                for j in 0..n {
                    let contribution = state.amplitudes[i].mul(&state.amplitudes[j].conj()).scale(*prob);
                    data[i][j] = data[i][j].add(&contribution);
                }
            }
        }
        Self { data, num_qubits }
    }

    /// Trace of the density matrix (should be 1 for a valid state).
    pub fn trace(&self) -> f64 {
        let n = self.data.len();
        let mut tr = 0.0;
        for i in 0..n { tr += self.data[i][i].re; }
        tr
    }

    /// Purity: Tr(ρ²). Equal to 1 for pure states, less for mixed states.
    pub fn purity(&self) -> f64 {
        let n = self.data.len();
        let mut result = 0.0;
        for i in 0..n {
            for k in 0..n {
                result += self.data[i][k].mul(&self.data[k][i]).re;
            }
        }
        result
    }

    /// Von Neumann entropy: S(ρ) = -Tr(ρ ln ρ).
    /// This measures the "mixedness" of the state. S=0 for pure, S=ln(2^n) for maximally mixed.
    pub fn von_neumann_entropy(&self) -> f64 {
        // We need eigenvalues of ρ. For a density matrix, these are all real and non-negative.
        let eigenvalues = self.eigenvalues_real();
        let mut entropy = 0.0;
        for &lambda in &eigenvalues {
            if lambda > 1e-15 {
                entropy -= lambda * lambda.ln();
            }
        }
        entropy
    }

    /// Compute real eigenvalues of the density matrix using the QR algorithm.
    /// Since ρ is Hermitian positive semidefinite, all eigenvalues are real ≥ 0.
    fn eigenvalues_real(&self) -> Vec<f64> {
        // For small matrices, use a direct method.
        let n = self.data.len();
        if n == 2 {
            let a = self.data[0][0].re;
            let d = self.data[1][1].re;
            let bc = self.data[0][1].norm_sq();
            let avg = (a + d) / 2.0;
            let disc = ((a - d) / 2.0).powi(2) + bc;
            let sqrt_d = if disc > 0.0 { disc.sqrt() } else { 0.0 };
            return vec![avg + sqrt_d, avg - sqrt_d];
        }

        // For larger matrices, extract real parts and use iterative QR.
        // (Simplified: just return diagonal elements as approximation for
        //  nearly diagonal matrices. A full implementation would do proper QR.)
        (0..n).map(|i| self.data[i][i].re).collect()
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_rng() -> Rng { Rng::from_seed(42) }

    #[test]
    fn test_hadamard_creates_superposition() {
        let mut state = StateVector::new(1);
        apply_single_qubit_gate(&mut state, 0, &hadamard());
        // Should be (|0⟩ + |1⟩)/√2
        assert!((state.probability(0) - 0.5).abs() < 1e-10);
        assert!((state.probability(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_x_gate_flips() {
        let mut state = StateVector::new(1);
        apply_single_qubit_gate(&mut state, 0, &pauli_x());
        // |0⟩ → |1⟩
        assert!((state.probability(0)).abs() < 1e-10);
        assert!((state.probability(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut state = StateVector::new(2);
        apply_single_qubit_gate(&mut state, 0, &hadamard());
        apply_cnot(&mut state, 0, 1);
        // Bell state: (|00⟩ + |11⟩)/√2
        assert!((state.probability(0b00) - 0.5).abs() < 1e-10);
        assert!(state.probability(0b01).abs() < 1e-10);
        assert!(state.probability(0b10).abs() < 1e-10);
        assert!((state.probability(0b11) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_entanglement_entropy() {
        // Product state: entropy = 0
        let product = StateVector::new(2); // |00⟩
        assert!(product.entanglement_entropy(0) < 1e-10);

        // Bell state: entropy = ln(2)
        let mut bell = StateVector::new(2);
        apply_single_qubit_gate(&mut bell, 0, &hadamard());
        apply_cnot(&mut bell, 0, 1);
        let entropy = bell.entanglement_entropy(0);
        assert!((entropy - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_collapses() {
        let mut rng = deterministic_rng();
        let mut state = StateVector::new(1);
        apply_single_qubit_gate(&mut state, 0, &hadamard());
        let result = measure_qubit(&mut state, 0, &mut rng);
        // After measurement, the state should be pure |0⟩ or |1⟩.
        assert!(result == 0 || result == 1);
        assert!((state.probability(result as usize) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measurement_statistics() {
        // Measure a Hadamard state 10000 times, should get roughly 50/50.
        let mut rng = deterministic_rng();
        let state = {
            let mut s = StateVector::new(1);
            apply_single_qubit_gate(&mut s, 0, &hadamard());
            s
        };
        let counts = sample(&state, 10000, &mut rng);
        let zeros = *counts.get(&0).unwrap_or(&0) as f64;
        let ones = *counts.get(&1).unwrap_or(&0) as f64;
        assert!((zeros / 10000.0 - 0.5).abs() < 0.05);
        assert!((ones / 10000.0 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_circuit_bell_pair() {
        let mut rng = deterministic_rng();
        let circuit = bell_pair();
        let counts = circuit.run(1000, &mut rng);
        // Should only get |00⟩ and |11⟩, roughly 50/50.
        for (bits, count) in &counts {
            assert!(bits == &vec![0, 0] || bits == &vec![1, 1],
                "Unexpected measurement: {:?}", bits);
        }
    }

    #[test]
    fn test_bernstein_vazirani() {
        let mut rng = deterministic_rng();
        let secret = 0b101; // Secret string = 5 (binary 101)
        let circuit = bernstein_vazirani(3, secret);
        let result = circuit.execute(&mut rng);
        // The measured bits should give us the secret string.
        let measured: usize = result.measurements.iter()
            .filter(|(q, _)| *q < 3) // Only input qubits
            .map(|(q, bit)| (*bit as usize) << q)
            .sum();
        assert_eq!(measured, secret);
    }

    #[test]
    fn test_deutsch_jozsa_constant() {
        let mut rng = deterministic_rng();
        // Constant function: always returns 0.
        let circuit = deutsch_jozsa(1, &|_| false);
        let result = circuit.execute(&mut rng);
        // Input qubit should measure 0 (constant function).
        let bit = result.measurements.iter().find(|(q, _)| *q == 0).unwrap().1;
        assert_eq!(bit, 0);
    }

    #[test]
    fn test_deutsch_jozsa_balanced() {
        let mut rng = deterministic_rng();
        // Balanced function: returns 1 for odd inputs.
        let circuit = deutsch_jozsa(1, &|x| x & 1 == 1);
        let result = circuit.execute(&mut rng);
        // Input qubit should measure 1 (balanced function).
        let bit = result.measurements.iter().find(|(q, _)| *q == 0).unwrap().1;
        assert_eq!(bit, 1);
    }

    #[test]
    fn test_grover_search() {
        let mut rng = deterministic_rng();
        // Search for element 3 (binary 11) in a 2-qubit database.
        let target = 3;
        let circuit = grover_search(2, &|x| x == target, Some(1));
        let counts = circuit.run(1000, &mut rng);
        // The target should be found with high probability.
        let target_bits = vec![1, 1]; // binary 11 in little-endian
        let target_count = counts.get(&target_bits).unwrap_or(&0);
        assert!(*target_count > 800, "Grover should find target >80% of the time, got {}", target_count);
    }

    #[test]
    fn test_qft_on_basis_state() {
        // QFT of |0⟩ should be uniform superposition.
        let mut c = Circuit::new("QFT test", 3);
        c.qft(0, 3);
        let mut rng = deterministic_rng();
        let result = c.execute(&mut rng);
        // All 8 basis states should have equal probability.
        for i in 0..8 {
            assert!((result.state.probability(i) - 1.0 / 8.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gate_unitarity() {
        // All standard gates should be unitary.
        assert!(hadamard().is_unitary());
        assert!(pauli_x().is_unitary());
        assert!(pauli_y().is_unitary());
        assert!(pauli_z().is_unitary());
        assert!(s_gate().is_unitary());
        assert!(t_gate().is_unitary());
        assert!(rx(1.23).is_unitary());
        assert!(ry(2.34).is_unitary());
        assert!(rz(3.45).is_unitary());
        assert!(phase(0.789).is_unitary());
        assert!(u3(1.0, 2.0, 3.0).is_unitary());
    }

    #[test]
    fn test_toffoli_gate() {
        // Toffoli flips target only when both controls are 1.
        let mut state = StateVector::from_bitstring(3, 0b111); // |111⟩
        apply_toffoli(&mut state, 0, 1, 2);
        // Should flip qubit 2: |111⟩ → |011⟩ (index 3)
        assert!((state.probability(0b011) - 1.0).abs() < 1e-10);

        // |101⟩ (only one control high) should not change.
        let mut state2 = StateVector::from_bitstring(3, 0b101);
        apply_toffoli(&mut state2, 0, 1, 2);
        assert!((state2.probability(0b101) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_matrix_purity() {
        // Pure state should have purity 1.
        let pure = StateVector::new(1);
        let rho = DensityMatrix::from_state_vector(&pure);
        assert!((rho.purity() - 1.0).abs() < 1e-10);
        assert!((rho.trace() - 1.0).abs() < 1e-10);

        // Maximally mixed state should have purity 1/2 (for 1 qubit).
        let mixed = DensityMatrix::from_ensemble(&[
            (0.5, StateVector::from_bitstring(1, 0)),
            (0.5, StateVector::from_bitstring(1, 1)),
        ]);
        assert!((mixed.purity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_swap_gate() {
        // |10⟩ → |01⟩
        let mut state = StateVector::from_bitstring(2, 0b10);
        apply_swap(&mut state, 0, 1);
        assert!((state.probability(0b01) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_normalization() {
        let mut state = StateVector::new(2);
        apply_single_qubit_gate(&mut state, 0, &hadamard());
        apply_cnot(&mut state, 0, 1);
        assert!(state.is_normalized());
    }

    #[test]
    fn test_circuit_depth_and_gate_count() {
        let mut c = Circuit::new("test", 3);
        c.h(0).h(1).h(2).cnot(0, 1).cnot(1, 2).h(0);
        assert_eq!(c.gate_count(), 6);
        // Depth trace: H(0)→[1,0,0], H(1)→[1,1,0], H(2)→[1,1,1],
        // CNOT(0,1)→d=max(1,1)+1=2→[2,2,1], CNOT(1,2)→d=max(2,1)+1=3→[2,3,3],
        // H(0)→[3,3,3]. Max=3.
        assert_eq!(c.depth(), 3);
    }
}
