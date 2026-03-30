// ============================================================================
// CHRONOS TYPE INFERENCE v2 + IR v2 + CODEGEN v2
// ============================================================================
// This file contains the modifications needed for all three downstream
// compiler phases to handle the 25 new domains. The changes cascade:
//   Lexer (new tokens) → Parser (new AST nodes) → Type Inference (new rules)
//     → IR (new opcodes) → Code Generation (new emission)
//
// PART A: Type Inference Extensions
// PART B: IR Opcode Extensions  
// PART C: AST→IR Lowering Extensions
// PART D: Backend Emission Extensions
// ============================================================================

// use std::collections::HashMap; // provided by parent scope in compiler-core

// ============================================================================
// PART A: TYPE INFERENCE EXTENSIONS
// ============================================================================
// The type inference engine must understand the semantics of each domain.
// This means: new type constructors, new constraint kinds, new checking
// passes, and new inference rules.

// --- NEW TYPE CONSTRUCTORS ---
// Add these variants to the InferType enum:

/// Additional InferType variants for domain-specific type inference.
/// These would be merged into the existing InferType enum.
#[derive(Debug, Clone, PartialEq)]
pub enum DomainInferType {
    /// Quantum type: tracks qubit state (pure, entangled, measured).
    /// The type system ensures you don't measure a qubit twice or
    /// use an entangled qubit without considering its partner.
    Qubit(QubitState),
    
    /// Protocol type: carries the current state of a protocol state machine.
    /// The compiler proves you never send a message in an invalid state.
    ProtocolSession { protocol: String, current_state: String },
    
    /// Smart contract storage reference: tracks whether storage was read,
    /// written, or needs commit. Prevents the "check-then-act" bug.
    ContractStorage { slot: String, dirty: bool },
    
    /// Proof obligation: a type that can only be constructed by providing
    /// a proof. This is the Curry-Howard bridge.
    ProofObligation { proposition: String, discharged: bool },
    
    /// Unit-tagged numeric type for dimensional analysis.
    /// The compiler verifies that meters + kilograms is a type error.
    UnitTagged { base: Box<DomainInferType>, unit: PhysicalUnitSignature },
    
    /// CRDT type: tracks the merge semantics (join-semilattice property).
    CrdtValue { crdt_type: String, element: Box<DomainInferType> },
    
    /// Real-time typed value: carries WCET metadata.
    RealtimeBounded { inner: Box<DomainInferType>, wcet_ns: u64 },
    
    /// Tainted value: tracks data provenance for security analysis.
    Tainted { inner: Box<DomainInferType>, source: String, sanitized: bool },
    
    /// Reactive signal type: auto-tracks dependencies.
    Signal { inner: Box<DomainInferType> },
    
    /// Geospatial type with CRS (coordinate reference system) tag.
    /// Prevents mixing EPSG:4326 (lat/lon) with EPSG:3857 (web mercator).
    Geospatial { geometry: String, crs: u32 },
}

/// Physical unit signature for compile-time dimensional analysis.
/// Each dimension is an integer exponent: meter^1 * second^(-2) = acceleration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalUnitSignature {
    pub m: i8,   // meter
    pub kg: i8,  // kilogram
    pub s: i8,   // second
    pub a: i8,   // ampere
    pub k: i8,   // kelvin
    pub mol: i8, // mole
    pub cd: i8,  // candela
}

impl PhysicalUnitSignature {
    pub fn dimensionless() -> Self {
        Self { m: 0, kg: 0, s: 0, a: 0, k: 0, mol: 0, cd: 0 }
    }
    
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            m: self.m + other.m, kg: self.kg + other.kg,
            s: self.s + other.s, a: self.a + other.a,
            k: self.k + other.k, mol: self.mol + other.mol,
            cd: self.cd + other.cd,
        }
    }
    
    pub fn div(&self, other: &Self) -> Self {
        Self {
            m: self.m - other.m, kg: self.kg - other.kg,
            s: self.s - other.s, a: self.a - other.a,
            k: self.k - other.k, mol: self.mol - other.mol,
            cd: self.cd - other.cd,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QubitState {
    Uninitialized,
    Pure,             // Known basis state (|0⟩ or |1⟩)
    Superposition,    // In superposition (after H gate, etc.)
    Entangled(String),// Entangled with another qubit (name)
    Measured,         // Already measured — cannot use as quantum anymore
}

// --- NEW CONSTRAINT KINDS ---
// Add these to the Constraint enum:

#[derive(Debug, Clone)]
pub enum DomainConstraint {
    /// Dimensional analysis: unit signatures must be compatible.
    /// Adding meter + kilogram is a compile error.
    UnitCompatible {
        left_unit: PhysicalUnitSignature,
        right_unit: PhysicalUnitSignature,
        operation: String,
        origin: ConstraintOrigin,
    },
    
    /// Protocol session type: the current state must allow this message.
    ProtocolStateValid {
        protocol: String,
        current_state: String,
        message: String,
        origin: ConstraintOrigin,
    },
    
    /// Quantum linearity: a qubit in state X can only undergo operations
    /// valid for that state. Measured qubits cannot be used as quantum.
    QubitStateValid {
        qubit_name: String,
        required_state: Vec<QubitState>, // Must be one of these
        origin: ConstraintOrigin,
    },
    
    /// Smart contract: reentrancy check. External calls must happen AFTER
    /// all state changes (checks-effects-interactions pattern).
    NoReentrancy {
        state_writes: Vec<String>,
        external_calls: Vec<String>,
        origin: ConstraintOrigin,
    },
    
    /// Proof obligation: this proposition must be proven.
    ProofRequired {
        proposition: String,
        origin: ConstraintOrigin,
    },
    
    /// Security taint: tainted data must be sanitized before reaching a sink.
    TaintSanitized {
        variable: String,
        required_sanitizer: String,
        origin: ConstraintOrigin,
    },
    
    /// WCET: function must complete within the declared deadline.
    WCETBound {
        function: String,
        max_ns: u64,
        origin: ConstraintOrigin,
    },
    
    /// CRS compatibility: spatial operations must use the same CRS.
    CRSCompatible {
        left_crs: u32,
        right_crs: u32,
        origin: ConstraintOrigin,
    },
    
    /// CRDT monotonicity: operations on CRDTs must be monotonic
    /// (the type system verifies the join-semilattice property).
    CrdtMonotonic {
        crdt_type: String,
        operation: String,
        origin: ConstraintOrigin,
    },
}

// ConstraintOrigin is defined in chronos-type-inference.rs (included earlier)

// --- NEW INFERENCE RULES ---
// These methods would be added to TypeInferenceEngine.

/// Domain-specific type inference extensions.
/// Each method handles inference for one domain's AST nodes.
pub trait DomainTypeInference {
    // ==========================================
    // DIMENSIONAL ANALYSIS (unit-of-measure)
    // ==========================================
    // When the user writes:
    //   let distance: f64<meter> = 5.0;
    //   let time: f64<second> = 2.0;
    //   let speed = distance / time;  // inferred: f64<meter * second^-1>
    //   let force = mass * speed / time; // inferred: f64<kg * m * s^-2> = newton
    //
    // The type inference engine tracks unit signatures through arithmetic:
    //   - Addition/subtraction: units must match exactly
    //   - Multiplication: unit exponents add
    //   - Division: unit exponents subtract
    //   - Sqrt: all exponents halve (must be even)
    //   - Assignment: units must match

    fn infer_unit_arithmetic(
        &mut self,
        left_unit: &PhysicalUnitSignature,
        right_unit: &PhysicalUnitSignature,
        operation: &str,
    ) -> Result<PhysicalUnitSignature, String> {
        match operation {
            "+" | "-" => {
                if left_unit == right_unit {
                    Ok(left_unit.clone())
                } else {
                    Err(format!(
                        "Cannot {} quantities with different units: {:?} vs {:?}",
                        if operation == "+" { "add" } else { "subtract" },
                        left_unit, right_unit
                    ))
                }
            }
            "*" => Ok(left_unit.mul(right_unit)),
            "/" => Ok(left_unit.div(right_unit)),
            _ => Ok(PhysicalUnitSignature::dimensionless()),
        }
    }

    // ==========================================
    // QUANTUM TYPE CHECKING
    // ==========================================
    // The type system tracks qubit state to prevent:
    //   - Measuring a qubit twice
    //   - Using a measured qubit in a gate
    //   - Ignoring entanglement constraints
    //
    // This is essentially a linear type system specialized for quantum:
    //   - H gate: Pure → Superposition
    //   - CNOT: (control: any, target: any) → (Entangled, Entangled)
    //   - Measure: any → Measured (consumes the qubit as quantum resource)

    fn check_quantum_gate(
        &mut self,
        gate: &str,
        qubit_states: &[(&str, QubitState)],
    ) -> Result<Vec<(String, QubitState)>, String> {
        match gate {
            "h" | "x" | "y" | "z" | "s" | "t" | "rx" | "ry" | "rz" => {
                // Single-qubit gates: any non-measured state → Superposition
                if qubit_states.len() != 1 {
                    return Err(format!("Gate '{}' requires exactly 1 qubit", gate));
                }
                let (name, state) = &qubit_states[0];
                if *state == QubitState::Measured {
                    return Err(format!(
                        "Qubit '{}' has already been measured and cannot be used in gate '{}'",
                        name, gate
                    ));
                }
                Ok(vec![(name.to_string(), QubitState::Superposition)])
            }
            "cx" | "cnot" | "cz" | "swap" => {
                // Two-qubit gates: both become entangled
                if qubit_states.len() != 2 {
                    return Err(format!("Gate '{}' requires exactly 2 qubits", gate));
                }
                for (name, state) in qubit_states {
                    if *state == QubitState::Measured {
                        return Err(format!("Qubit '{}' already measured", name));
                    }
                }
                let (n0, _) = &qubit_states[0];
                let (n1, _) = &qubit_states[1];
                Ok(vec![
                    (n0.to_string(), QubitState::Entangled(n1.to_string())),
                    (n1.to_string(), QubitState::Entangled(n0.to_string())),
                ])
            }
            "measure" => {
                if qubit_states.len() != 1 {
                    return Err("Measure operates on exactly 1 qubit".to_string());
                }
                let (name, state) = &qubit_states[0];
                if *state == QubitState::Measured {
                    return Err(format!("Qubit '{}' already measured — double measurement", name));
                }
                Ok(vec![(name.to_string(), QubitState::Measured)])
            }
            "toffoli" | "ccx" | "fredkin" | "cswap" => {
                if qubit_states.len() != 3 {
                    return Err(format!("Gate '{}' requires exactly 3 qubits", gate));
                }
                for (name, state) in qubit_states {
                    if *state == QubitState::Measured {
                        return Err(format!("Qubit '{}' already measured", name));
                    }
                }
                Ok(qubit_states.iter()
                    .map(|(n, _)| (n.to_string(), QubitState::Entangled("multi".to_string())))
                    .collect())
            }
            _ => Err(format!("Unknown quantum gate: '{}'", gate)),
        }
    }

    // ==========================================
    // SMART CONTRACT VERIFICATION
    // ==========================================
    // The type checker enforces the checks-effects-interactions pattern:
    //   1. All require() checks must come first.
    //   2. All state modifications (storage writes) come second.
    //   3. External calls come last.
    // Violating this order is flagged as a potential reentrancy vulnerability.

    fn check_contract_reentrancy(
        &mut self,
        statements: &[ContractStatement],
    ) -> Vec<String> {
        let mut errors = Vec::new();
        let mut phase = ContractPhase::Checks;
        
        for (i, stmt) in statements.iter().enumerate() {
            match stmt {
                ContractStatement::Require(_) => {
                    if phase != ContractPhase::Checks {
                        errors.push(format!(
                            "Line {}: require() after state modification or external call. \
                             Move all checks before state changes (CEI pattern).", i
                        ));
                    }
                }
                ContractStatement::StorageWrite(_) => {
                    if phase == ContractPhase::Interactions {
                        errors.push(format!(
                            "Line {}: Storage write after external call — reentrancy risk! \
                             Move all state changes before external calls.", i
                        ));
                    }
                    phase = ContractPhase::Effects;
                }
                ContractStatement::ExternalCall(_) => {
                    phase = ContractPhase::Interactions;
                }
                _ => {}
            }
        }
        errors
    }

    // ==========================================
    // PROOF CHECKING
    // ==========================================
    // The proof checker verifies that each theorem/lemma is correctly proven.
    // This is a simplified version of a tactic-based proof assistant.

    fn check_proof(
        &mut self,
        name: &str,
        statement: &ProofExprAST,
        proof: &ProofBlockAST,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        let mut goals: Vec<ProofExprAST> = vec![statement.clone()];
        
        for tactic in &proof.tactics {
            match tactic {
                TacticAST::Sorry => {
                    errors.push(format!(
                        "WARNING: Theorem '{}' uses 'sorry' — proof obligation not discharged. \
                         This is accepted for development but must be resolved before release.", name
                    ));
                    goals.clear(); // Accept the sorry
                }
                TacticAST::Trivial => {
                    // Check if the current goal is trivially true
                    if goals.is_empty() {
                        errors.push(format!("No remaining goals to prove in '{}'", name));
                    } else {
                        goals.pop(); // Simplified: accept trivial for demo
                    }
                }
                TacticAST::Intro(vars) => {
                    // Introduce universally quantified variables into the context
                    if let Some(goal) = goals.last_mut() {
                        // Would actually strip forall binders from the goal
                    }
                }
                TacticAST::SMT => {
                    // Delegate to an SMT solver (Z3/CVC5)
                    // In a real implementation, this would:
                    // 1. Translate the goal to SMTLIB2
                    // 2. Call z3 as a subprocess
                    // 3. Check the result
                    goals.clear(); // Simplified: trust SMT
                }
                _ => {
                    // Other tactics would be handled similarly
                }
            }
        }
        
        if !goals.is_empty() {
            errors.push(format!(
                "Proof of '{}' is incomplete: {} goal(s) remaining", name, goals.len()
            ));
        }
        
        errors
    }

    // ==========================================
    // PROTOCOL SESSION TYPE CHECKING
    // ==========================================
    // Verifies that protocol operations happen in the correct order
    // according to the declared state machine.

    fn check_protocol_transition(
        &self,
        protocol: &str,
        current_state: &str,
        message: &str,
        state_machine: &StateMachineAST,
    ) -> Result<String, String> {
        for transition in &state_machine.transitions {
            if transition.from == current_state && transition.trigger == message {
                return Ok(transition.to.clone());
            }
        }
        Err(format!(
            "Invalid protocol transition: cannot send '{}' in state '{}' of protocol '{}'",
            message, current_state, protocol
        ))
    }

    // ==========================================
    // SECURITY TAINT TRACKING
    // ==========================================
    // Propagates taint through the data flow graph. Any value derived
    // from a tainted source is itself tainted unless explicitly sanitized.

    fn propagate_taint(
        &mut self,
        var: &str,
        source_vars: &[&str],
        taint_state: &HashMap<String, InferTaintInfo>,
    ) -> Option<InferTaintInfo> {
        for source in source_vars {
            if let Some(info) = taint_state.get(*source) {
                if !info.sanitized {
                    return Some(InferTaintInfo {
                        source: info.source.clone(),
                        path: {
                            let mut p = info.path.clone();
                            p.push(var.to_string());
                            p
                        },
                        sanitized: false,
                    });
                }
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContractPhase { Checks, Effects, Interactions }

#[derive(Debug, Clone)]
pub enum ContractStatement {
    Require(String),
    StorageWrite(String),
    ExternalCall(String),
    Other,
}

#[derive(Debug, Clone)]
pub struct InferTaintInfo {
    pub source: String,
    pub path: Vec<String>,
    pub sanitized: bool,
}

// Placeholder types from parser v2 — all now defined in chronos-parser-v2.rs
// which is included earlier in chronos-compiler-core.

// ============================================================================
// PART B: IR OPCODE EXTENSIONS
// ============================================================================
// New IR opcodes for operations that don't map to existing instructions.
// These are added to the IROpcode enum.

#[derive(Debug, Clone)]
pub enum DomainIROpcode {
    // === QUANTUM OPERATIONS ===
    // These compile to calls to a quantum simulator or hardware driver.
    QuantumAlloc(u32),                  // Allocate N qubits, returns qubit register handle
    QuantumFree(IRValue),               // Deallocate qubit register
    QuantumGate {                       // Apply a gate
        gate: String,                   // "h", "cx", "rz", etc.
        params: Vec<f64>,              // Rotation angles
        qubits: Vec<(IRValue, u32)>,   // (register, index) pairs
    },
    QuantumMeasure {                    // Measure qubit → classical bit
        qubit: (IRValue, u32),
        classical: (IRValue, u32),
    },
    QuantumBarrier(Vec<IRValue>),       // Prevent optimization across this point
    QuantumReset(IRValue, u32),         // Reset qubit to |0⟩
    
    // === PROTOCOL OPERATIONS ===
    ProtocolSend {
        session: IRValue,
        message_type: String,
        payload: IRValue,
    },
    ProtocolReceive {
        session: IRValue,
        expected_type: String,
    },
    ProtocolStateTransition {
        session: IRValue,
        from_state: String,
        to_state: String,
    },
    
    // === BLOCKCHAIN / CONTRACT OPERATIONS ===
    StorageLoad { slot: IRValue },
    StorageStore { slot: IRValue, value: IRValue },
    ContractCall {
        address: IRValue,
        function: String,
        args: Vec<IRValue>,
        value: Option<IRValue>,        // ETH value to send
        gas: Option<IRValue>,
    },
    EmitEvent {
        name: String,
        indexed: Vec<IRValue>,
        data: Vec<IRValue>,
    },
    SelfDestruct(IRValue),
    GetBalance(IRValue),
    GetCaller,
    GetBlockNumber,
    GetTimestamp,
    Revert(IRValue),                    // Revert with error message
    Keccak256(IRValue),                 // Hash function (EVM-specific)
    
    // === DISTRIBUTED OPERATIONS ===
    CrdtMerge { crdt: IRValue, other: IRValue },
    CrdtUpdate { crdt: IRValue, operation: String, args: Vec<IRValue> },
    ConsensusPropose { value: IRValue },
    ConsensusAwait,                     // Wait for consensus
    SagaBegin(String),
    SagaCommit,
    SagaCompensate(String),             // Execute compensation
    
    // === CHANNEL / CONCURRENCY OPERATIONS ===
    ChannelCreate { capacity: u32 },
    ChannelSend { channel: IRValue, value: IRValue },
    ChannelReceive { channel: IRValue },
    ChannelSelect { channels: Vec<IRValue> },  // Go-style select
    SpawnTask { function: String, args: Vec<IRValue> },
    JoinTask(IRValue),
    
    // === GEOSPATIAL OPERATIONS ===
    SpatialDistance(IRValue, IRValue),
    SpatialIntersects(IRValue, IRValue),
    SpatialBuffer(IRValue, f64),
    SpatialTransform(IRValue, u32, u32), // geometry, from_crs, to_crs
    SpatialIndex { operation: String, geometry: IRValue },
    
    // === MULTIMEDIA OPERATIONS ===
    AudioProcess { operation: String, input: IRValue, params: Vec<f64> },
    VideoProcess { operation: String, frame: IRValue, params: Vec<f64> },
    ShaderCompile { stage: String, code: String },
    RenderDraw { pipeline: IRValue, mesh: IRValue, material: IRValue },
    
    // === EMBEDDED DSL EXECUTION ===
    EmbedSQL { query: String, params: Vec<IRValue> },
    EmbedRegex { pattern: String, input: IRValue },
    EmbedGraphQL { query: String, variables: Vec<(String, IRValue)> },
    
    // === SENSOR / ROBOTICS ===
    SensorRead { device: String, channel: String },
    ActuatorWrite { device: String, value: IRValue },
    PIDCompute { controller: IRValue, setpoint: IRValue, measurement: IRValue },
    
    // === OBSERVABILITY ===
    TraceSpanBegin { name: String, attributes: Vec<(String, IRValue)> },
    TraceSpanEnd,
    MetricIncrement { name: String, value: IRValue, labels: Vec<(String, String)> },
    LogEmit { level: String, message: IRValue, fields: Vec<(String, IRValue)> },
    
    // === PROOF / VERIFICATION MARKERS ===
    // These generate no runtime code but are checked at compile time.
    // In debug mode, they become runtime assertions.
    AssertInvariant { expression: IRValue, message: String },
    AssumeProperty { expression: IRValue },
    VerifyProperty { expression: IRValue },
}

// IRValue is defined in chronos-ir-codegen.rs (included later in compiler-core)


// ============================================================================
// PART C: AST → IR LOWERING EXTENSIONS
// ============================================================================
// New lowering logic for domain-specific AST nodes → IR opcodes.
// These methods would be added to the ASTLowering impl.

/// Extension methods for ASTLowering to handle domain items.
pub trait DomainLowering {
    /// Main dispatch for lowering domain items to IR.
    fn lower_domain_item(&mut self, item: &DomainItem) {
        match item {
            DomainItem::ProtocolDecl(p) => self.lower_protocol(p),
            DomainItem::ContractDecl(c) => self.lower_contract(c),
            DomainItem::CircuitDecl(q) => self.lower_quantum_circuit(q),
            DomainItem::TheoremDecl(t) | DomainItem::LemmaDecl(t) => {
                self.lower_theorem(t);
            }
            DomainItem::TestDecl(t) => self.lower_test(t),
            DomainItem::SimulationDecl(s) => self.lower_simulation(s),
            DomainItem::EntityDecl(e) => self.lower_ecs_entity(e),
            DomainItem::SystemDecl(s) => self.lower_ecs_system(s),
            _ => {
                // Generic domain items lower to function calls into
                // the domain-specific runtime library.
            }
        }
    }

    /// Lower a protocol declaration to IR.
    /// Generates: parser function, serializer function, state machine validator.
    fn lower_protocol(&mut self, _protocol: &ProtocolDeclAST) {
        // 1. Generate a struct type for the protocol frame
        // 2. Generate a parse() function that reads fields from a byte buffer
        // 3. Generate a serialize() function that writes fields to a buffer
        // 4. Generate a state machine checker function
        // 5. If the protocol has checksums, generate verify/compute functions
    }

    /// Lower a smart contract to EVM bytecode or Solana BPF.
    fn lower_contract(&mut self, _contract: &ContractDeclAST) {
        // 1. Generate storage layout (slot assignments for EVM)
        // 2. Generate function selector dispatch (EVM: first 4 bytes of keccak)
        // 3. Generate each function with storage load/store ops
        // 4. Generate event emission code
        // 5. Generate modifier wrapper functions
        // 6. Insert reentrancy guards where needed
    }

    /// Lower a quantum circuit to simulator calls or hardware instructions.
    fn lower_quantum_circuit(&mut self, _circuit: &CircuitDeclAST) {
        // 1. Allocate qubit registers
        // 2. Emit gate operations in sequence
        // 3. Emit measurement operations
        // 4. Free qubit registers
        // 5. Optimize: merge adjacent single-qubit gates, cancel inverses
    }

    /// Lower a theorem: in release mode this is a no-op (proofs are erased).
    /// In debug mode, preconditions/postconditions become runtime assertions.
    fn lower_theorem(&mut self, _theorem: &TheoremDeclAST) {
        // Proofs are checked at compile time and erased at runtime.
        // The lowering step only emits code if:
        // - Debug mode: emit runtime assertions for requires/ensures
        // - Contract mode: emit runtime checks for invariants
    }

    fn lower_test(&mut self, _test: &TestDeclAST) {
        // Generate a test runner entry point that:
        // 1. Calls setup if present
        // 2. Runs the test body
        // 3. Checks expectations
        // 4. Reports results
        // 5. Calls teardown if present
    }

    fn lower_simulation(&mut self, _sim: &SimulationDeclAST) {
        // 1. Generate mesh initialization code
        // 2. Generate physics solver setup (call into simulation runtime)
        // 3. Generate time-stepping loop
        // 4. Generate output/checkpoint code
        // 5. Generate parallelization setup (MPI/OpenMP/GPU)
    }

    fn lower_ecs_entity(&mut self, _entity: &EntityDeclAST) {
        // Generate a function that creates an entity with the specified
        // components in the ECS world.
    }

    fn lower_ecs_system(&mut self, _system: &SystemDeclAST) {
        // Generate a function that queries the ECS world for entities
        // matching the system's query, then runs the system body on each.
    }
}

// Placeholder types from parser v2 — all defined in chronos-parser-v2.rs
// which is included earlier in chronos-compiler-core.


// ============================================================================
// PART D: BACKEND EMISSION EXTENSIONS
// ============================================================================
// Each backend (LLVM, CUDA, XLA) needs to know how to emit the new opcodes.

/// Extensions to the LLVM backend for domain-specific opcodes.
pub trait LLVMDomainEmission {
    /// Emit LLVM IR for domain-specific operations.
    fn emit_domain_op(&mut self, opcode: &DomainIROpcode) -> String {
        match opcode {
            // Quantum ops → calls to quantum simulator library
            DomainIROpcode::QuantumAlloc(n) => {
                format!("call i8* @chronos_quantum_alloc(i32 {})", n)
            }
            DomainIROpcode::QuantumGate { gate, params, qubits } => {
                let param_str = params.iter()
                    .map(|p| format!("double {:e}", p))
                    .collect::<Vec<_>>().join(", ");
                format!("call void @chronos_quantum_gate_{0}({1})", gate, param_str)
            }
            DomainIROpcode::QuantumMeasure { qubit, classical } => {
                format!("call i1 @chronos_quantum_measure(i8* %qreg, i32 %qidx)")
            }
            
            // Contract ops → EVM bytecode generation or runtime calls
            DomainIROpcode::StorageLoad { slot } => {
                format!("call i256 @chronos_sload(i256 {})", slot.name)
            }
            DomainIROpcode::StorageStore { slot, value } => {
                format!("call void @chronos_sstore(i256 {}, i256 {})", slot.name, value.name)
            }
            DomainIROpcode::EmitEvent { name, indexed, data } => {
                format!("; emit event {}", name)
            }
            DomainIROpcode::Keccak256(input) => {
                format!("call i256 @chronos_keccak256(i8* {})", input.name)
            }
            
            // Channel ops → runtime calls
            DomainIROpcode::ChannelCreate { capacity } => {
                format!("call i8* @chronos_channel_create(i32 {})", capacity)
            }
            DomainIROpcode::ChannelSend { channel, value } => {
                format!("call void @chronos_channel_send(i8* {}, i8* {})",
                    channel.name, value.name)
            }
            DomainIROpcode::ChannelReceive { channel } => {
                format!("call i8* @chronos_channel_receive(i8* {})", channel.name)
            }
            
            // Geospatial ops → runtime library calls
            DomainIROpcode::SpatialDistance(a, b) => {
                format!("call double @chronos_spatial_distance(i8* {}, i8* {})",
                    a.name, b.name)
            }
            DomainIROpcode::SpatialTransform(geom, from, to) => {
                format!("call i8* @chronos_spatial_transform(i8* {}, i32 {}, i32 {})",
                    geom.name, from, to)
            }
            
            // Observability ops → runtime calls
            DomainIROpcode::TraceSpanBegin { name, .. } => {
                format!("call i8* @chronos_trace_begin(i8* getelementptr ([{} x i8], \
                    [{} x i8]* @.str.{}, i64 0, i64 0))",
                    name.len() + 1, name.len() + 1, name)
            }
            DomainIROpcode::TraceSpanEnd => {
                "call void @chronos_trace_end()".to_string()
            }
            DomainIROpcode::MetricIncrement { name, value, .. } => {
                format!("call void @chronos_metric_inc(i8* @.metric.{}, double {})",
                    name, value.name)
            }
            DomainIROpcode::LogEmit { level, message, .. } => {
                format!("call void @chronos_log_{}(i8* {})", level, message.name)
            }
            
            // Embedded DSL ops → compile-time or runtime processing
            DomainIROpcode::EmbedSQL { query, params } => {
                let param_strs: Vec<String> = params.iter()
                    .map(|p| format!("i8* {}", p.name))
                    .collect();
                format!("call i8* @chronos_sql_exec(i8* @.sql.{}, i32 {}{})",
                    query.len(), params.len(),
                    if params.is_empty() { String::new() }
                    else { format!(", {}", param_strs.join(", ")) })
            }
            DomainIROpcode::EmbedRegex { pattern, input } => {
                format!("call i1 @chronos_regex_match(i8* @.regex.compiled, i8* {})",
                    input.name)
            }
            
            // Proof markers → no-ops in release, assertions in debug
            DomainIROpcode::AssertInvariant { expression, message } => {
                format!("; assert invariant: {}\n  \
                    call void @chronos_assert(i1 {}, i8* @.msg.{})",
                    message, expression.name, message.len())
            }
            DomainIROpcode::AssumeProperty { .. } | DomainIROpcode::VerifyProperty { .. } => {
                "; verified at compile time — no runtime code".to_string()
            }
            
            _ => format!("; unhandled domain opcode: {:?}", opcode),
        }
    }

    /// Generate LLVM declarations for all domain runtime functions.
    fn emit_domain_runtime_declarations(&self) -> String {
        let mut decls = String::new();
        
        decls.push_str("\n; === Domain Runtime Library Declarations ===\n\n");
        
        // Quantum runtime
        decls.push_str("; Quantum computing runtime\n");
        decls.push_str("declare i8* @chronos_quantum_alloc(i32)\n");
        decls.push_str("declare void @chronos_quantum_free(i8*)\n");
        decls.push_str("declare void @chronos_quantum_gate_h(i8*, i32)\n");
        decls.push_str("declare void @chronos_quantum_gate_x(i8*, i32)\n");
        decls.push_str("declare void @chronos_quantum_gate_cx(i8*, i32, i8*, i32)\n");
        decls.push_str("declare void @chronos_quantum_gate_rz(i8*, i32, double)\n");
        decls.push_str("declare i1 @chronos_quantum_measure(i8*, i32)\n");
        decls.push_str("declare void @chronos_quantum_reset(i8*, i32)\n\n");
        
        // Blockchain runtime
        decls.push_str("; Blockchain / smart contract runtime\n");
        decls.push_str("declare i256 @chronos_sload(i256)\n");
        decls.push_str("declare void @chronos_sstore(i256, i256)\n");
        decls.push_str("declare i256 @chronos_keccak256(i8*)\n");
        decls.push_str("declare i256 @chronos_balance(i256)\n");
        decls.push_str("declare i256 @chronos_caller()\n");
        decls.push_str("declare void @chronos_revert(i8*)\n\n");
        
        // Channel / concurrency runtime
        decls.push_str("; Concurrency runtime\n");
        decls.push_str("declare i8* @chronos_channel_create(i32)\n");
        decls.push_str("declare void @chronos_channel_send(i8*, i8*)\n");
        decls.push_str("declare i8* @chronos_channel_receive(i8*)\n");
        decls.push_str("declare i8* @chronos_spawn(i8*, i8*)\n");
        decls.push_str("declare void @chronos_join(i8*)\n\n");
        
        // Geospatial runtime
        decls.push_str("; Geospatial runtime\n");
        decls.push_str("declare double @chronos_spatial_distance(i8*, i8*)\n");
        decls.push_str("declare i1 @chronos_spatial_intersects(i8*, i8*)\n");
        decls.push_str("declare i8* @chronos_spatial_buffer(i8*, double)\n");
        decls.push_str("declare i8* @chronos_spatial_transform(i8*, i32, i32)\n\n");
        
        // Observability runtime
        decls.push_str("; Observability runtime\n");
        decls.push_str("declare i8* @chronos_trace_begin(i8*)\n");
        decls.push_str("declare void @chronos_trace_end()\n");
        decls.push_str("declare void @chronos_metric_inc(i8*, double)\n");
        decls.push_str("declare void @chronos_log_info(i8*)\n");
        decls.push_str("declare void @chronos_log_warn(i8*)\n");
        decls.push_str("declare void @chronos_log_error(i8*)\n\n");
        
        // Embedded DSL runtime
        decls.push_str("; Embedded DSL runtime\n");
        decls.push_str("declare i8* @chronos_sql_exec(i8*, i32, ...)\n");
        decls.push_str("declare i1 @chronos_regex_match(i8*, i8*)\n");
        decls.push_str("declare i8* @chronos_graphql_exec(i8*, i8*)\n\n");
        
        // Multimedia runtime
        decls.push_str("; Multimedia runtime\n");
        decls.push_str("declare i8* @chronos_audio_process(i8*, i8*, i8*)\n");
        decls.push_str("declare i8* @chronos_video_process(i8*, i8*, i8*)\n");
        decls.push_str("declare i8* @chronos_shader_compile(i8*, i8*)\n\n");
        
        // Robotics / sensor runtime
        decls.push_str("; Robotics / sensor runtime\n");
        decls.push_str("declare double @chronos_sensor_read(i8*, i8*)\n");
        decls.push_str("declare void @chronos_actuator_write(i8*, double)\n");
        decls.push_str("declare double @chronos_pid_compute(i8*, double, double)\n\n");
        
        // Assertions
        decls.push_str("; Verification runtime (debug mode only)\n");
        decls.push_str("declare void @chronos_assert(i1, i8*)\n");
        
        decls
    }
}

/// Extensions to the CUDA backend for domain operations that benefit from GPU.
pub trait CUDADomainEmission {
    /// Some domain operations can be GPU-accelerated.
    /// These get custom CUDA kernels instead of CPU runtime calls.
    fn emit_cuda_domain_op(&mut self, opcode: &DomainIROpcode) -> Option<String> {
        match opcode {
            // Quantum simulation benefits enormously from GPU
            DomainIROpcode::QuantumGate { gate, params, qubits } => {
                Some(format!(
                    "// GPU-accelerated quantum gate: {}\n\
                     // State vector updated in parallel across {} amplitudes\n\
                     call void @chronos_quantum_gpu_gate_{}(...);",
                    gate, 1u64 << qubits.len(), gate
                ))
            }
            // Geospatial computations on large datasets
            DomainIROpcode::SpatialDistance(_, _) => {
                Some("// GPU-accelerated batch spatial distance computation\n\
                      call void @chronos_spatial_gpu_batch_distance(...);".to_string())
            }
            // Audio FFT is a classic GPU workload
            DomainIROpcode::AudioProcess { operation, .. } if operation == "fft" => {
                Some("// GPU-accelerated FFT via cuFFT\n\
                      call void @chronos_audio_gpu_fft(...);".to_string())
            }
            _ => None, // Not GPU-accelerable; fall back to CPU
        }
    }
}


// ============================================================================
// PART E: EVM BYTECODE BACKEND (new backend for smart contracts)
// ============================================================================
// Smart contracts need their own backend that generates EVM bytecode
// instead of LLVM IR. This is a new CodeGenBackend implementation.

/// The EVM backend generates Ethereum Virtual Machine bytecode.
pub struct EVMBackend {
    bytecode: Vec<u8>,
    stack_depth: usize,
}

impl EVMBackend {
    pub fn new() -> Self {
        Self { bytecode: Vec::new(), stack_depth: 0 }
    }
    
    fn emit_opcode(&mut self, op: u8) {
        self.bytecode.push(op);
    }
    
    fn emit_push(&mut self, value: &[u8]) {
        let n = value.len();
        assert!(n >= 1 && n <= 32);
        self.emit_opcode(0x5F + n as u8); // PUSH1..PUSH32
        self.bytecode.extend_from_slice(value);
        self.stack_depth += 1;
    }

    /// Generate EVM bytecode for a contract.
    pub fn generate_contract(&mut self, contract: &DomainItem) -> Vec<u8> {
        // 1. Generate constructor bytecode
        // 2. Generate runtime bytecode with function selector dispatch
        // 3. Append contract metadata
        
        // Function selector dispatch:
        // The first 4 bytes of calldata are keccak256(signature)[:4]
        // We compare and jump to the right function.
        
        // Simplified example:
        self.emit_opcode(0x60); self.bytecode.push(0x00);  // PUSH1 0x00
        self.emit_opcode(0x35);                              // CALLDATALOAD
        self.emit_opcode(0x60); self.bytecode.push(0xE0);   // PUSH1 0xE0
        self.emit_opcode(0x1C);                              // SHR (get first 4 bytes)
        // DUP1, PUSH4 <selector>, EQ, PUSH2 <offset>, JUMPI
        // ... for each function
        
        self.bytecode.clone()
    }

    pub fn generate(&mut self, _ir: &IRModule) -> Vec<u8> {
        Vec::new()
    }
}

// EVM opcodes for reference:
// 0x00 STOP, 0x01 ADD, 0x02 MUL, 0x03 SUB, 0x04 DIV,
// 0x10 LT, 0x11 GT, 0x14 EQ, 0x15 ISZERO,
// 0x20 KECCAK256,
// 0x30 ADDRESS, 0x31 BALANCE, 0x32 ORIGIN, 0x33 CALLER,
// 0x34 CALLVALUE, 0x35 CALLDATALOAD,
// 0x54 SLOAD, 0x55 SSTORE,
// 0x56 JUMP, 0x57 JUMPI, 0x5B JUMPDEST,
// 0x60-0x7F PUSH1-PUSH32,
// 0x80-0x8F DUP1-DUP16,
// 0x90-0x9F SWAP1-SWAP16,
// 0xA0-0xA4 LOG0-LOG4,
// 0xF1 CALL, 0xF3 RETURN, 0xFD REVERT, 0xFF SELFDESTRUCT
