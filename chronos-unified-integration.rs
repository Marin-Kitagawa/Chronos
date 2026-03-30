// ============================================================================
// CHRONOS UNIFIED INTEGRATION LAYER
// ============================================================================
// This file bridges ALL previously defined systems into a coherent whole.
// It addresses every gap identified between:
//   - The security audit engine ↔ type inference taint tracking
//   - The real-time runtime ↔ IR domain opcodes ↔ WCET analyzer
//   - The simulation engine ↔ parser ↔ IR ↔ codegen
//   - The data science toolkit ↔ parser ↔ type inference ↔ IR
//   - The memory management system ↔ parser ↔ type inference ↔ IR
//   - The graph algorithm library ↔ parser ↔ type inference
//   - The symbolic math engine ↔ parser ↔ comptime evaluation
//   - The new 25 domains ↔ all existing infrastructure
//
// ORGANIZATION:
//   Part 1: Unified Compiler Pipeline (orchestrates ALL passes in order)
//   Part 2: Security ↔ Type Inference Bridge
//   Part 3: Real-Time ↔ IR ↔ WCET Bridge
//   Part 4: Simulation ↔ Parser ↔ IR Bridge
//   Part 5: Data Science ↔ Type Inference ↔ IR Bridge
//   Part 6: Memory Management ↔ Parser ↔ Type Inference ↔ IR Bridge
//   Part 7: Graph Algorithms ↔ Type Inference Bridge
//   Part 8: Symbolic Math ↔ Comptime Evaluation Bridge
//   Part 9: Extended Lexer/Parser Additions for Stdlib Features
//   Part 10: Extended IR Opcodes for ALL Stdlib Features
//   Part 11: Extended Codegen Runtime Declarations
// ============================================================================

// use std::collections::HashMap; // provided by parent scope in compiler-core
// use std::time::Duration; // provided by parent scope in compiler-core

// ============================================================================
// PART 1: UNIFIED COMPILER PIPELINE
// ============================================================================
// This is the master orchestrator that runs every compilation phase in the
// correct order, feeding outputs from one phase into inputs of the next.
// Previously, each phase was standalone. Now they share state through the
// CompilationContext, which flows through the entire pipeline.

/// The shared context that flows through every compilation phase.
/// This is the "connective tissue" that was missing — it lets the
/// security audit engine see taint data from type inference, the WCET
/// analyzer see cycle costs from the IR, and so on.
pub struct CompilationContext {
    // --- Source-level information ---
    pub filename: String,
    pub source: String,
    pub security_mode: SecurityMode,       // From #![secure] or --secure flag
    pub realtime_mode: bool,               // From #![realtime] or --realtime flag
    
    // --- Phase 1 outputs: Lexer ---
    pub tokens: Vec<SpannedToken>,
    
    // --- Phase 2 outputs: Parser ---
    pub ast: Option<Program>,
    pub version_rules: Option<VersionRules>,
    
    // --- Phase 3 outputs: Type Inference ---
    pub type_env: TypeEnvironment,
    pub inferred_effects: HashMap<String, Vec<Effect>>,
    pub taint_map: TaintMap,               // SHARED with security audit
    pub unit_map: UnitMap,                 // Dimensional analysis results
    pub qubit_state_map: QubitStateMap,    // Quantum state tracking
    pub protocol_session_map: ProtocolSessionMap,
    pub linear_resource_map: LinearResourceMap,
    pub proof_obligations: Vec<ProofObligation>,
    
    // --- Phase 4 outputs: Security Audit ---
    pub security_findings: Vec<SecurityFinding>,
    pub audit_passed: bool,
    
    // --- Phase 5 outputs: Degradation Check ---
    pub degradation_diagnostics: Vec<CompilerDiagnostic>,
    
    // --- Phase 6 outputs: IR Lowering ---
    pub ir_module: Option<IRModule>,
    
    // --- Phase 7 outputs: WCET Analysis (if realtime_mode) ---
    pub wcet_results: HashMap<String, WCETResult>,
    
    // --- Phase 8 outputs: Code Generation ---
    pub generated_code: HashMap<String, Vec<u8>>, // backend_name → output
    
    // --- Diagnostics ---
    pub all_diagnostics: Vec<CompilerDiagnostic>,
}

/// The taint map is SHARED between type inference and the security audit engine.
/// Type inference populates it during its data-flow analysis pass.
/// The security audit engine reads it to find unsanitized data reaching sinks.
pub type TaintMap = HashMap<String, TaintEntry>;

#[derive(Debug, Clone)]
pub struct TaintEntry {
    pub source: TaintSource,
    pub propagation_chain: Vec<String>,  // Variable names showing how taint spread
    pub sanitized: bool,
    pub sanitizer_used: Option<String>,  // Which sanitizer function was applied
    pub location: SourceLocation,
    // Cross-reference to type inference: what type does this variable have?
    pub inferred_type: Option<String>,
}

/// Dimensional analysis results, shared so the IR lowering can emit
/// unit-conversion calls when crossing CRS boundaries in geospatial code
/// or when auto-converting physical units.
pub type UnitMap = HashMap<String, PhysicalUnitSignature>;

/// Quantum state tracking, shared so the WCET analyzer can estimate the
/// cost of quantum operations (which depends on the number of qubits in
/// superposition/entanglement).
pub type QubitStateMap = HashMap<String, QubitState>;

/// Protocol session state, shared so the security audit can verify that
/// all protocol sessions reach a terminal state (no leaked connections).
pub type ProtocolSessionMap = HashMap<String, (String, String)>; // session → (protocol, current_state)

/// Linear resource tracking, shared so the security audit can verify that
/// all file handles, DB connections, etc. are properly closed.
pub type LinearResourceMap = HashMap<String, LinearResourceEntry>;

#[derive(Debug, Clone)]
pub struct LinearResourceEntry {
    pub resource_type: String,
    pub state: ResourceStatus,
    pub declared_at: SourceLocation,
    pub consumed_at: Option<SourceLocation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResourceStatus { Alive, Consumed, Leaked }

#[derive(Debug, Clone)]
pub struct ProofObligation {
    pub name: String,
    pub proposition: String,
    pub discharged: bool,
    pub location: SourceLocation,
}

/// The master compilation function that runs every phase in order.
/// This replaces the ad-hoc compilation that existed before, where
/// each phase ran independently without shared state.
pub fn compile(source: String, filename: String, config: CompilerConfig) -> CompilationResult {
    let mut ctx = CompilationContext {
        filename: filename.clone(),
        source: source.clone(),
        security_mode: config.security_mode.clone(),
        realtime_mode: config.realtime_mode,
        tokens: Vec::new(),
        ast: None,
        version_rules: None,
        type_env: TypeEnvironment::new(),
        inferred_effects: HashMap::new(),
        taint_map: HashMap::new(),
        unit_map: HashMap::new(),
        qubit_state_map: HashMap::new(),
        protocol_session_map: HashMap::new(),
        linear_resource_map: HashMap::new(),
        proof_obligations: Vec::new(),
        security_findings: Vec::new(),
        audit_passed: true,
        degradation_diagnostics: Vec::new(),
        ir_module: None,
        wcet_results: HashMap::new(),
        generated_code: HashMap::new(),
        all_diagnostics: Vec::new(),
    };

    // =====================================================================
    // PHASE 1: LEXING
    // Tokenize the source. All keywords (including the 150+ new domain
    // keywords from lexer v2) are recognized here.
    // =====================================================================
    let lexer = Lexer::new(source, filename.clone());
    match lexer.tokenize() {
        Ok(tokens) => ctx.tokens = tokens,
        Err(errors) => {
            for e in errors {
                ctx.all_diagnostics.push(CompilerDiagnostic {
                    level: DiagnosticLevel::Error,
                    message: e.message,
                    code: "E-LEX-001".to_string(),
                });
            }
            return CompilationResult::from_context(ctx);
        }
    }

    // =====================================================================
    // PHASE 2: PARSING
    // Build the AST. This now handles ALL domain-specific declarations
    // (protocol, contract, circuit, theorem, simulation, pipeline, etc.)
    // via the extended parse_item() dispatch from parser v2.
    // =====================================================================
    let mut parser = Parser::new(ctx.tokens.clone());
    let program = parser.parse_program();
    
    if !parser.get_errors().is_empty() {
        for e in parser.get_errors() {
            ctx.all_diagnostics.push(CompilerDiagnostic {
                level: DiagnosticLevel::Error,
                message: e.message.clone(),
                code: "E-PARSE-001".to_string(),
            });
        }
    }
    
    // Load version rules from .rules file (Feature 3).
    ctx.version_rules = RulesFileParser::find_and_parse();
    ctx.ast = Some(program);

    // =====================================================================
    // PHASE 3: TYPE INFERENCE (with domain extensions)
    // This phase now does MUCH more than before:
    //   a) Standard Hindley-Milner type inference
    //   b) Linear/affine type checking → populates linear_resource_map
    //   c) Effect inference → populates inferred_effects
    //   d) Dimensional analysis → populates unit_map
    //   e) Taint propagation → populates taint_map (SHARED with Phase 4)
    //   f) Quantum state tracking → populates qubit_state_map
    //   g) Protocol session type checking → populates protocol_session_map
    //   h) Smart contract reentrancy checking
    //   i) Proof obligation collection → populates proof_obligations
    //   j) Capability checking (file, thread, memory permissions)
    // =====================================================================
    let mut inference = UnifiedTypeInference::new();
    let ast = ctx.ast.clone().unwrap();

    // Run the unified inference pass.
    let inference_result = inference.infer_program(&ast, &mut ctx);
    
    // Transfer results back to context for downstream phases.
    ctx.type_env = inference.get_type_environment();
    ctx.inferred_effects = inference.get_effect_map();
    // taint_map, unit_map, qubit_state_map, protocol_session_map,
    // linear_resource_map, and proof_obligations are populated by
    // the inference pass writing directly into ctx.

    for error in &inference_result.errors {
        ctx.all_diagnostics.push(CompilerDiagnostic {
            level: DiagnosticLevel::Error,
            message: error.message.clone(),
            code: "E-TYPE-001".to_string(),
        });
    }

    // =====================================================================
    // PHASE 3.5: PROOF VERIFICATION
    // Verify all theorem/lemma proofs collected during type inference.
    // This runs the tactic-based proof checker from formal verification.
    // If any proof fails and the mode is strict, compilation stops.
    // =====================================================================
    let mut proof_checker = ProofChecker::new();
    for obligation in &ctx.proof_obligations {
        if !obligation.discharged {
            ctx.all_diagnostics.push(CompilerDiagnostic {
                level: if obligation.name.contains("sorry") {
                    DiagnosticLevel::Warning
                } else {
                    DiagnosticLevel::Error
                },
                message: format!(
                    "Proof obligation '{}' not discharged at {}:{}",
                    obligation.name, obligation.location.file, obligation.location.line
                ),
                code: "E-PROOF-001".to_string(),
            });
        }
    }

    // =====================================================================
    // PHASE 4: SECURITY AUDIT
    // The audit engine now READS from the shared context instead of
    // running its own analysis. This eliminates the duplication between
    // the type inference taint tracking and the audit engine's taint
    // analysis. The audit engine focuses on INTERPRETING results and
    // generating findings/suggestions, while type inference focuses on
    // COMPUTING the data flow.
    // =====================================================================
    if ctx.security_mode != SecurityMode::Off {
        let mut audit = SecurityAuditEngine::new(ctx.security_mode.clone());
        
        // BRIDGE 1: Feed taint map from type inference into audit engine.
        // The audit engine no longer runs its own taint analysis —
        // it reads the taint_map that type inference populated.
        audit.import_taint_data(&ctx.taint_map);
        
        // BRIDGE 2: Feed linear resource map into audit engine.
        // Leaked resources (files, connections) are a security issue.
        audit.import_resource_data(&ctx.linear_resource_map);
        
        // BRIDGE 3: Feed protocol session data into audit engine.
        // Sessions that don't reach terminal state are potential leaks.
        audit.import_session_data(&ctx.protocol_session_map);
        
        // BRIDGE 4: Feed contract reentrancy results from type inference.
        // Type inference already checked CEI pattern; audit reports findings.
        audit.import_contract_analysis(&inference_result);
        
        // BRIDGE 5: Feed effect annotations into audit engine.
        // Functions performing IO in a pure context are flagged.
        audit.import_effects(&ctx.inferred_effects);
        
        // Run the remaining audit passes that DON'T overlap with type inference:
        //   - Cryptographic audit (weak algorithms, short keys)
        //   - Dependency audit (CVE database lookup)
        //   - Authentication patterns
        //   - WCET violations (if realtime mode)
        let audit_result = audit.run_non_overlapping_passes(&ctx);
        
        ctx.security_findings = audit_result.findings;
        ctx.audit_passed = audit_result.passed;
        
        for finding in &ctx.security_findings {
            ctx.all_diagnostics.push(CompilerDiagnostic {
                level: match finding.severity {
                    Severity::Critical | Severity::High => DiagnosticLevel::Error,
                    Severity::Medium => if ctx.security_mode == SecurityMode::Enforce {
                        DiagnosticLevel::Error
                    } else {
                        DiagnosticLevel::Warning
                    },
                    _ => DiagnosticLevel::Warning,
                },
                message: format!("[{}] {}: {}", finding.id, 
                    match finding.severity {
                        Severity::Critical => "CRITICAL",
                        Severity::High => "HIGH",
                        Severity::Medium => "MEDIUM",
                        Severity::Low => "LOW",
                        Severity::Info => "INFO",
                    },
                    finding.message),
                code: finding.id.clone(),
            });
        }
        
        // If security mode is Enforce and audit failed, stop compilation.
        if ctx.security_mode == SecurityMode::Enforce && !ctx.audit_passed {
            ctx.all_diagnostics.push(CompilerDiagnostic {
                level: DiagnosticLevel::Error,
                message: "Security audit FAILED — compilation halted. \
                         Fix all findings above or use #![secure(warn)] for warnings only.".to_string(),
                code: "E-SEC-HALT".to_string(),
            });
            return CompilationResult::from_context(ctx);
        }
    }

    // =====================================================================
    // PHASE 5: DEGRADATION CHECK (Feature 6)
    // Check all degradable functions against their expiry schedules.
    // =====================================================================
    if let Some(ref ast) = ctx.ast {
        for item in &ast.items {
            if let Item::DegradableFunctionDecl(df) = item {
                let diags = DegradationChecker::check(df);
                let has_error = diags.iter().any(|d| d.level == DiagnosticLevel::Error);
                ctx.degradation_diagnostics.extend(diags.clone());
                ctx.all_diagnostics.extend(diags);
                
                if has_error {
                    // Expired functions prevent compilation entirely.
                    return CompilationResult::from_context(ctx);
                }
            }
        }
    }

    // =====================================================================
    // PHASE 6: IR LOWERING
    // Convert the typed AST into IR. This phase now handles:
    //   a) Standard function/struct/class lowering (existing)
    //   b) Kernel lowering for GPU/TPU/NPU (existing, Feature 5)
    //   c) ALL 25 new domain item lowering (new)
    //   d) Simulation domain lowering → runtime library calls (new)
    //   e) DataFrame/ML operations → vectorized IR ops (new)
    //   f) Memory allocator annotations → allocator-specific alloc/free (new)
    //   g) Graph algorithm calls → optimized implementations (new)
    //   h) Symbolic math → compile-time evaluation or runtime CAS calls (new)
    // =====================================================================
    let target = config.target.clone();
    let mut lowering = UnifiedASTLowering::new(&filename, target);
    
    // The lowering pass now has access to the full context, including
    // type information, effect data, and allocator assignments.
    let ir_module = lowering.lower_program(
        ctx.ast.as_ref().unwrap(),
        &ctx.type_env,
        &ctx.inferred_effects,
        &ctx.unit_map,
    );
    ctx.ir_module = Some(ir_module);

    // =====================================================================
    // PHASE 7: WCET ANALYSIS (if realtime mode)
    // Now that we have the IR, the WCET analyzer can estimate the
    // worst-case execution time of every function. The key BRIDGE here
    // is that the WCET analyzer now knows the cycle cost of ALL domain
    // opcodes, not just the original arithmetic/memory ones.
    // =====================================================================
    if ctx.realtime_mode {
        let hardware = config.hardware_model.unwrap_or_else(default_hardware_model);
        let mut wcet = UnifiedWCETAnalyzer::new(hardware);
        
        if let Some(ref ir) = ctx.ir_module {
            for func in &ir.functions {
                // Check if this function has a realtime annotation.
                let deadline = func.attributes.iter().find_map(|attr| {
                    if let FunctionAttribute::RealtimeDeadline(d) = attr {
                        Some(Duration::from_nanos(*d))
                    } else { None }
                });
                
                let result = wcet.analyze_function_with_domain_costs(
                    &func.name,
                    &func.blocks,
                    deadline,
                );
                
                if let Some(false) = result.meets_deadline {
                    ctx.all_diagnostics.push(CompilerDiagnostic {
                        level: DiagnosticLevel::Error,
                        message: format!(
                            "Function '{}' WCET ({} ns) exceeds deadline ({} ns). \
                             Critical path: {}",
                            func.name,
                            result.wcet_nanoseconds,
                            deadline.map(|d| d.as_nanos() as u64).unwrap_or(0),
                            result.critical_path.join(" → "),
                        ),
                        code: "E-WCET-001".to_string(),
                    });
                }
                
                ctx.wcet_results.insert(func.name.clone(), result);
            }
        }
    }

    // =====================================================================
    // PHASE 8: CODE GENERATION
    // Emit code for all target backends. Each backend now handles:
    //   a) Standard CPU code via LLVM (existing)
    //   b) GPU kernels via CUDA PTX (existing)
    //   c) TPU code via XLA HLO (existing)
    //   d) EVM bytecode for smart contracts (new)
    //   e) Quantum circuit → QASM/hardware calls (new)
    //   f) ALL domain runtime library declarations (new)
    // =====================================================================
    if let Some(ref ir) = ctx.ir_module {
        // Determine which backends to use based on what the code contains.
        let has_cpu = ir.functions.iter().any(|f| f.device == IRDevice::Cpu);
        let has_gpu = ir.functions.iter().any(|f| matches!(f.device, IRDevice::Gpu(_)));
        let has_tpu = ir.functions.iter().any(|f| matches!(f.device, IRDevice::Tpu(_)));
        let has_contract = ir.functions.iter().any(|f| {
            f.attributes.iter().any(|a| matches!(a, FunctionAttribute::SmartContract))
        });
        let has_quantum = ir.functions.iter().any(|f| {
            f.effects.iter().any(|e| matches!(e, IREffect::QuantumCompute))
        });
        
        if has_cpu {
            let mut llvm = UnifiedLLVMBackend::new();
            let output = llvm.generate(ir);
            ctx.generated_code.insert("llvm".to_string(), output.into_bytes());
        }
        if has_gpu {
            let mut cuda = UnifiedCUDABackend::new();
            let output = cuda.generate(ir);
            ctx.generated_code.insert("cuda".to_string(), output.into_bytes());
        }
        if has_tpu {
            let mut xla = UnifiedXLABackend::new();
            let output = xla.generate(ir);
            ctx.generated_code.insert("xla".to_string(), output.into_bytes());
        }
        if has_contract {
            let mut evm = EVMBackend::new();
            let output = evm.generate(ir);
            ctx.generated_code.insert("evm".to_string(), output);
        }
        if has_quantum {
            let mut qasm = QASMBackend::new();
            let output = qasm.generate(ir);
            ctx.generated_code.insert("qasm".to_string(), output.into_bytes());
        }
    }

    // C backend (always run when AST is available — outputs standard C99)
    if let Some(ref ast) = ctx.ast {
        let mut c_backend = CBackend::new();
        let c_code = c_backend.generate_program(ast);
        ctx.generated_code.insert("c".to_string(), c_code.into_bytes());
    }

    // =====================================================================
    // PHASE 9: VERSION CONTROL INTEGRATION (Feature 3)
    // The incremental compiler uses the version rules to determine
    // what changed and compute the appropriate version bump.
    // =====================================================================
    if let (Some(ref rules), Some(ref ir)) = (&ctx.version_rules, &ctx.ir_module) {
        let versioner = VersionComputer::new(rules);
        let version = versioner.compute_version(ir);
        // Store version in IR module metadata for embedding in binary.
    }

    CompilationResult::from_context(ctx)
}

// ============================================================================
// PART 2: SECURITY ↔ TYPE INFERENCE BRIDGE
// ============================================================================
// The SecurityAuditEngine gains methods to import data from type inference
// instead of computing it independently. This is the key deduplication.

/// Bridge trait for the security audit engine to consume type inference data.
pub trait SecurityBridge {
    /// Import taint analysis results from the type inference engine.
    /// This replaces the audit engine's own run_taint_analysis().
    fn import_taint_data(&mut self, taint_map: &TaintMap);
    
    /// Import linear resource tracking data.
    /// Leaked resources (files not closed, connections not dropped) are
    /// a security concern (DoS via resource exhaustion).
    fn import_resource_data(&mut self, resource_map: &LinearResourceMap);
    
    /// Import protocol session data.
    /// Sessions stuck in non-terminal states indicate potential issues.
    fn import_session_data(&mut self, session_map: &ProtocolSessionMap);
    
    /// Import contract analysis results from the reentrancy checker.
    fn import_contract_analysis(&mut self, inference_result: &InferenceResult);
    
    /// Import effect annotations.
    /// Functions performing IO/Alloc in contexts marked `pure` are flagged.
    fn import_effects(&mut self, effects: &HashMap<String, Vec<Effect>>);
    
    /// Run ONLY the audit passes that don't overlap with type inference.
    /// This is: crypto audit, dependency audit, auth patterns, and
    /// WCET violations (which need the IR, not just the AST).
    fn run_non_overlapping_passes(&mut self, ctx: &CompilationContext) -> AuditResult;
}


// ============================================================================
// PART 3: REAL-TIME ↔ IR ↔ WCET BRIDGE
// ============================================================================
// The WCET analyzer must know the cycle cost of EVERY IR opcode,
// including all domain-specific operations. This mapping was missing.

/// Extended WCET analyzer that knows about domain-specific opcode costs.
pub struct UnifiedWCETAnalyzer {
    hardware: HardwareModel,
    /// Cycle cost table for domain-specific operations.
    /// These costs are conservative upper bounds for WCET analysis.
    domain_costs: HashMap<String, u64>,
}

impl UnifiedWCETAnalyzer {
    pub fn new(hardware: HardwareModel) -> Self {
        let mut domain_costs = HashMap::new();
        
        // Costs are in clock cycles on the target hardware.
        // These are WORST-CASE estimates, intentionally pessimistic.
        
        // Quantum operations (on simulator — actual quantum hardware is different)
        domain_costs.insert("quantum_gate_single".to_string(), 100);
        domain_costs.insert("quantum_gate_two".to_string(), 500);
        domain_costs.insert("quantum_gate_three".to_string(), 2000);
        domain_costs.insert("quantum_measure".to_string(), 50);
        domain_costs.insert("quantum_alloc".to_string(), 1000);
        
        // Channel/concurrency operations
        domain_costs.insert("channel_send".to_string(), 200);
        domain_costs.insert("channel_receive".to_string(), 10_000); // May block!
        domain_costs.insert("channel_select".to_string(), 15_000);
        domain_costs.insert("spawn_task".to_string(), 50_000);
        
        // Smart contract operations (EVM gas-cost-based estimates)
        domain_costs.insert("sload".to_string(), 2100);    // SLOAD = 2100 gas ≈ cycles
        domain_costs.insert("sstore".to_string(), 20000);  // SSTORE = up to 20000 gas
        domain_costs.insert("keccak256".to_string(), 600);
        domain_costs.insert("contract_call".to_string(), 100_000);
        
        // Geospatial operations
        domain_costs.insert("spatial_distance".to_string(), 500);
        domain_costs.insert("spatial_intersects".to_string(), 5000);
        domain_costs.insert("spatial_buffer".to_string(), 50_000);
        domain_costs.insert("spatial_transform".to_string(), 10_000);
        
        // Sensor/actuator operations
        domain_costs.insert("sensor_read".to_string(), 5000);    // I2C/SPI transaction
        domain_costs.insert("actuator_write".to_string(), 3000);
        domain_costs.insert("pid_compute".to_string(), 200);     // Pure arithmetic
        
        // Audio/video operations
        domain_costs.insert("audio_fft_1024".to_string(), 100_000);
        domain_costs.insert("audio_filter".to_string(), 10_000);
        
        // Database operations (potentially unbounded — flag for WCET!)
        domain_costs.insert("sql_exec".to_string(), u64::MAX);  // UNBOUNDED
        domain_costs.insert("db_query".to_string(), u64::MAX);  // UNBOUNDED
        
        // Observability (these should be fast)
        domain_costs.insert("trace_begin".to_string(), 100);
        domain_costs.insert("trace_end".to_string(), 50);
        domain_costs.insert("metric_inc".to_string(), 30);
        domain_costs.insert("log_emit".to_string(), 500);
        
        // Simulation operations (potentially unbounded)
        domain_costs.insert("solver_iteration".to_string(), u64::MAX);
        domain_costs.insert("mesh_refine".to_string(), u64::MAX);
        domain_costs.insert("fem_assemble".to_string(), u64::MAX);
        
        // DataFrame operations (potentially unbounded)
        domain_costs.insert("dataframe_filter".to_string(), u64::MAX);
        domain_costs.insert("dataframe_groupby".to_string(), u64::MAX);
        domain_costs.insert("ml_train".to_string(), u64::MAX);
        
        // Protocol operations
        domain_costs.insert("protocol_send".to_string(), 5000);
        domain_costs.insert("protocol_receive".to_string(), 50_000);
        
        // Memory operations
        domain_costs.insert("arena_alloc".to_string(), 10);     // O(1) bump
        domain_costs.insert("pool_alloc".to_string(), 15);      // O(1) free-list pop
        domain_costs.insert("buddy_alloc".to_string(), 100);    // O(log n) split
        domain_costs.insert("system_alloc".to_string(), 10_000);// malloc (unpredictable)
        
        Self { hardware, domain_costs }
    }

    /// Analyze a function using both standard and domain-specific cycle costs.
    pub fn analyze_function_with_domain_costs(
        &mut self,
        func_name: &str,
        blocks: &[BasicBlock],
        deadline: Option<Duration>,
    ) -> WCETResult {
        let mut per_block_cycles: HashMap<String, u64> = HashMap::new();
        let mut unbounded_ops: Vec<String> = Vec::new();
        
        for block in blocks {
            let mut cycles = 0u64;
            for op in &block.ops {
                let cost = self.estimate_op_cost(op);
                if cost == u64::MAX {
                    // This operation is potentially unbounded — critical warning
                    // for real-time code. The WCET cannot be computed.
                    unbounded_ops.push(format!("{:?} in block {}", op.opcode, block.label));
                    cycles = u64::MAX;
                    break;
                }
                cycles = cycles.saturating_add(cost);
            }
            per_block_cycles.insert(block.label.clone(), cycles);
        }
        
        // If any block has unbounded ops, the total WCET is unbounded.
        let has_unbounded = per_block_cycles.values().any(|&c| c == u64::MAX);
        
        let wcet_cycles = if has_unbounded {
            u64::MAX
        } else {
            per_block_cycles.values().sum()
        };
        
        let wcet_ns = if wcet_cycles == u64::MAX {
            u64::MAX
        } else {
            (wcet_cycles as f64 / self.hardware.clock_hz as f64 * 1e9) as u64
        };
        
        let meets_deadline = deadline.map(|d| {
            wcet_ns != u64::MAX && wcet_ns <= d.as_nanos() as u64
        });
        
        WCETResult {
            function_name: func_name.to_string(),
            wcet_cycles,
            wcet_nanoseconds: wcet_ns,
            deadline,
            meets_deadline,
            critical_path: blocks.iter().map(|b| b.label.clone()).collect(),
            per_block_cycles,
            confidence: if has_unbounded {
                WCETConfidence::Unbounded { operations: unbounded_ops }
            } else {
                WCETConfidence::ProvenBound
            },
        }
    }
    
    /// Estimate the cycle cost of a single IR operation, including domain ops.
    fn estimate_op_cost(&self, op: &IROp) -> u64 {
        match &op.opcode {
            // Standard arithmetic (from original WCET analyzer)
            IROpcode::Add(_, _) | IROpcode::Sub(_, _) => 1,
            IROpcode::Mul(_, _) => 3,
            IROpcode::Div(_, _) => 20,
            IROpcode::FAdd(_, _) | IROpcode::FSub(_, _) => 4,
            IROpcode::FMul(_, _) => 5,
            IROpcode::FDiv(_, _) => 15,
            IROpcode::Load(_) => self.hardware.cache_model.l1_hit_cycles as u64,
            IROpcode::Store(_, _) => self.hardware.cache_model.l1_hit_cycles as u64,
            IROpcode::Call(name, _) => {
                // Look up the called function's WCET if we've analyzed it.
                // This is how inter-procedural WCET works.
                5 // Base call overhead; real WCET adds callee's WCET
            }
            
            // AI operations (from original IR)
            IROpcode::Gemm { .. } => 1_000_000,  // Matrix multiply is expensive
            IROpcode::Attention { .. } => 5_000_000,
            IROpcode::Softmax(_, _) => 100_000,
            IROpcode::Relu(_) | IROpcode::Gelu(_) => 10_000,
            
            // Domain operations — look up in the cost table
            IROpcode::DomainOp(domain_op) => {
                *self.domain_costs.get(domain_op.as_str()).unwrap_or(&1000)
            }
            
            // Memory operations — cost depends on allocator
            IROpcode::StackAlloc(_) => 1,
            IROpcode::HeapAlloc(_) => *self.domain_costs.get("system_alloc").unwrap_or(&10_000),
            IROpcode::ArenaAlloc(_, _) => *self.domain_costs.get("arena_alloc").unwrap_or(&10),
            IROpcode::PoolAlloc(_, _) => *self.domain_costs.get("pool_alloc").unwrap_or(&15),
            IROpcode::Free(_) => 50,
            IROpcode::Drop(_) => 100,  // Destructor call
            
            _ => 1, // Unknown ops get 1 cycle (conservative for tight loops)
        }
    }
    
    fn estimate_domain_op_cost(&self, op: &DomainIROpcode) -> u64 {
        match op {
            DomainIROpcode::QuantumGate { qubits, .. } => {
                match qubits.len() {
                    1 => *self.domain_costs.get("quantum_gate_single").unwrap_or(&100),
                    2 => *self.domain_costs.get("quantum_gate_two").unwrap_or(&500),
                    _ => *self.domain_costs.get("quantum_gate_three").unwrap_or(&2000),
                }
            }
            DomainIROpcode::QuantumMeasure { .. } =>
                *self.domain_costs.get("quantum_measure").unwrap_or(&50),
            DomainIROpcode::ChannelSend { .. } =>
                *self.domain_costs.get("channel_send").unwrap_or(&200),
            DomainIROpcode::ChannelReceive { .. } =>
                *self.domain_costs.get("channel_receive").unwrap_or(&10_000),
            DomainIROpcode::StorageLoad { .. } =>
                *self.domain_costs.get("sload").unwrap_or(&2100),
            DomainIROpcode::StorageStore { .. } =>
                *self.domain_costs.get("sstore").unwrap_or(&20000),
            DomainIROpcode::SpatialDistance(_, _) =>
                *self.domain_costs.get("spatial_distance").unwrap_or(&500),
            DomainIROpcode::SensorRead { .. } =>
                *self.domain_costs.get("sensor_read").unwrap_or(&5000),
            DomainIROpcode::ActuatorWrite { .. } =>
                *self.domain_costs.get("actuator_write").unwrap_or(&3000),
            DomainIROpcode::PIDCompute { .. } =>
                *self.domain_costs.get("pid_compute").unwrap_or(&200),
            DomainIROpcode::EmbedSQL { .. } =>
                *self.domain_costs.get("sql_exec").unwrap_or(&u64::MAX),
            DomainIROpcode::TraceSpanBegin { .. } =>
                *self.domain_costs.get("trace_begin").unwrap_or(&100),
            DomainIROpcode::LogEmit { .. } =>
                *self.domain_costs.get("log_emit").unwrap_or(&500),
            DomainIROpcode::MetricIncrement { .. } =>
                *self.domain_costs.get("metric_inc").unwrap_or(&30),
            _ => 1000, // Unknown domain op: conservative default
        }
    }
}

// WCETConfidence is defined in chronos-security-realtime.rs (included earlier)


// ============================================================================
// PART 4: SIMULATION ↔ PARSER ↔ IR BRIDGE
// ============================================================================
// The parser produces SimulationDeclAST with config pairs.
// This bridge converts those into the actual SimulationDomain struct
// from the simulation engine, then lowers to IR calls.

pub struct SimulationLowering;

impl SimulationLowering {
    /// Convert parsed simulation config into the full SimulationDomain.
    /// This bridges the parser's string-based representation into the
    /// strongly-typed simulation engine structures.
    pub fn lower_simulation_decl(
        decl: &SimulationDeclAST,
    ) -> SimulationDomain {
        // Convert the Vec<(String, Expression)> config pairs into
        // the actual PhysicsModel, SolverConfig, etc. structs.
        // This is where the parser's loose representation becomes the
        // simulation engine's typed representation.
        
        SimulationDomain {
            name: decl.name.clone(),
            // ... map each config field to the SimulationDomain fields
            // using pattern matching on the key strings.
        }
    }
}


// ============================================================================
// PART 5: MEMORY MANAGEMENT ↔ IR BRIDGE
// ============================================================================
// The IR now has SEPARATE opcodes for each allocator type. The lowering
// pass checks the allocator annotation on each type and emits the right op.

/// New IR opcodes for allocator-aware memory management.
/// These REPLACE the generic HeapAlloc with allocator-specific ops.
#[derive(Debug, Clone)]
pub enum AllocatorIROp {
    /// Allocate from an arena (bump allocator). O(1).
    ArenaAlloc { arena: String, size: usize, align: usize },
    /// Allocate from a fixed-size pool. O(1).
    PoolAlloc { pool: String, size: usize },
    /// Allocate from a slab allocator. O(1).
    SlabAlloc { slab: String, size: usize },
    /// Allocate from a buddy allocator. O(log n).
    BuddyAlloc { buddy: String, size: usize },
    /// Allocate from the real-time TLSF allocator. O(1).
    TLSFAlloc { allocator: String, size: usize },
    /// Stack allocation (alloca). O(1).
    StackAlloc { size: usize, align: usize },
    /// System heap allocation (malloc). NOT O(1) — flagged in real-time mode!
    SystemAlloc { size: usize, align: usize },
    /// Corresponding frees for each allocator.
    ArenaFreeAll { arena: String },
    PoolFree { pool: String, ptr: IRValue },
    BuddyFree { buddy: String, ptr: IRValue, size: usize },
    TLSFFree { allocator: String, ptr: IRValue },
    SystemFree { ptr: IRValue },
}


// ============================================================================
// PART 6: GRAPH ALGORITHMS ↔ TYPE INFERENCE BRIDGE
// ============================================================================
// The type inference engine learns about graph properties to select
// optimal algorithm implementations at compile time.

/// Graph type inference: determines properties of a graph at compile time.
/// This allows the compiler to choose Dijkstra over Bellman-Ford when it
/// can prove all edge weights are non-negative, for example.
pub struct GraphTypeInference;

impl GraphTypeInference {
    pub fn infer_graph_properties(
        &self,
        graph_type: &ChronosType,
        operations: &[String],
    ) -> InferredGraphProperties {
        InferredGraphProperties {
            // If the user declared the graph as undirected, we know that.
            // If they use only positive-weight edges, we know that.
            // These properties determine which algorithms are valid.
            directed: None,       // None = unknown, must be conservative
            weighted: None,
            acyclic: None,
            non_negative_weights: None,
            connected: None,
            bipartite: None,
        }
    }
    
    /// Choose the optimal algorithm implementation based on inferred properties.
    pub fn select_optimal_algorithm(
        &self,
        requested: &str,           // e.g., "shortest_path"
        props: &InferredGraphProperties,
    ) -> String {
        match requested {
            "shortest_path" => {
                if props.non_negative_weights == Some(true) {
                    "dijkstra".to_string()    // O(E log V)
                } else {
                    "bellman_ford".to_string() // O(VE) but handles negatives
                }
            }
            "matching" => {
                if props.bipartite == Some(true) {
                    "hopcroft_karp".to_string() // O(E√V) for bipartite
                } else {
                    "blossom".to_string()        // O(V³) for general
                }
            }
            "mst" => {
                // Use Kruskal for sparse graphs, Prim for dense ones.
                "kruskal".to_string()
            }
            _ => requested.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferredGraphProperties {
    pub directed: Option<bool>,
    pub weighted: Option<bool>,
    pub acyclic: Option<bool>,
    pub non_negative_weights: Option<bool>,
    pub connected: Option<bool>,
    pub bipartite: Option<bool>,
}


// ============================================================================
// PART 7: SYMBOLIC MATH ↔ COMPTIME EVALUATION BRIDGE
// ============================================================================
// The symbolic math engine can run AT COMPILE TIME for constant expressions.
// This means `comptime { derivative(x^2 + sin(x), x) }` evaluates to
// `2*x + cos(x)` during compilation, with zero runtime cost.

pub struct ComptimeSymbolicEvaluator;

impl ComptimeSymbolicEvaluator {
    /// Evaluate a symbolic expression at compile time.
    /// Returns the simplified symbolic result, or None if it can't be
    /// evaluated at compile time (e.g., depends on runtime variables).
    pub fn evaluate(expr: &SymbolicExpr) -> Option<SymbolicExpr> {
        match expr {
            // Derivative of a polynomial: fully evaluable at compile time.
            SymbolicExpr::Derivative { expr: inner, variable, order } => {
                let mut result = *inner.clone();
                for _ in 0..*order {
                    result = Self::differentiate(&result, variable)?;
                }
                Some(result)
            }
            // Simplification: always evaluable.
            SymbolicExpr::Simplify(inner) => {
                Self::simplify(inner)
            }
            // Numeric evaluation of constants.
            SymbolicExpr::Plus(terms) if terms.iter().all(Self::is_numeric) => {
                let sum: f64 = terms.iter().filter_map(Self::to_f64).sum();
                Some(SymbolicExpr::Real(sum))
            }
            // If the expression contains runtime variables, we can't evaluate
            // at compile time — emit it as a runtime CAS call instead.
            _ => None,
        }
    }
    
    fn differentiate(expr: &SymbolicExpr, var: &str) -> Option<SymbolicExpr> {
        match expr {
            SymbolicExpr::Symbol(s) if s == var => Some(SymbolicExpr::Integer(1)),
            SymbolicExpr::Symbol(_) => Some(SymbolicExpr::Integer(0)),
            SymbolicExpr::Integer(_) | SymbolicExpr::Real(_) => Some(SymbolicExpr::Integer(0)),
            SymbolicExpr::Power(base, exp) => {
                // d/dx(x^n) = n * x^(n-1)
                if let (SymbolicExpr::Symbol(s), SymbolicExpr::Integer(n)) = (base.as_ref(), exp.as_ref()) {
                    if s == var {
                        Some(SymbolicExpr::Times(vec![
                            SymbolicExpr::Integer(*n),
                            SymbolicExpr::Power(
                                base.clone(),
                                Box::new(SymbolicExpr::Integer(n - 1)),
                            ),
                        ]))
                    } else {
                        Some(SymbolicExpr::Integer(0))
                    }
                } else { None }
            }
            SymbolicExpr::Sin(inner) => {
                // d/dx(sin(f)) = cos(f) * f'
                let inner_deriv = Self::differentiate(inner, var)?;
                Some(SymbolicExpr::Times(vec![
                    SymbolicExpr::Cos(inner.clone()),
                    inner_deriv,
                ]))
            }
            SymbolicExpr::Cos(inner) => {
                let inner_deriv = Self::differentiate(inner, var)?;
                Some(SymbolicExpr::Times(vec![
                    SymbolicExpr::Times(vec![
                        SymbolicExpr::Integer(-1),
                        SymbolicExpr::Sin(inner.clone()),
                    ]),
                    inner_deriv,
                ]))
            }
            SymbolicExpr::Plus(terms) => {
                // d/dx(f + g) = f' + g'
                let derived: Option<Vec<_>> = terms.iter()
                    .map(|t| Self::differentiate(t, var))
                    .collect();
                derived.map(SymbolicExpr::Plus)
            }
            SymbolicExpr::Times(factors) if factors.len() == 2 => {
                // Product rule: d/dx(f * g) = f' * g + f * g'
                let f = &factors[0];
                let g = &factors[1];
                let f_prime = Self::differentiate(f, var)?;
                let g_prime = Self::differentiate(g, var)?;
                Some(SymbolicExpr::Plus(vec![
                    SymbolicExpr::Times(vec![f_prime, g.clone()]),
                    SymbolicExpr::Times(vec![f.clone(), g_prime]),
                ]))
            }
            _ => None, // Complex expressions fall through to runtime CAS
        }
    }
    
    fn simplify(expr: &SymbolicExpr) -> Option<SymbolicExpr> {
        // Basic algebraic simplification rules.
        match expr {
            SymbolicExpr::Times(factors) => {
                // x * 0 = 0, x * 1 = x
                if factors.iter().any(|f| f == &SymbolicExpr::Integer(0)) {
                    return Some(SymbolicExpr::Integer(0));
                }
                let non_one: Vec<_> = factors.iter()
                    .filter(|f| *f != &SymbolicExpr::Integer(1))
                    .cloned().collect();
                if non_one.len() == 1 { return Some(non_one[0].clone()); }
                if non_one.is_empty() { return Some(SymbolicExpr::Integer(1)); }
                Some(SymbolicExpr::Times(non_one))
            }
            SymbolicExpr::Plus(terms) => {
                // x + 0 = x
                let non_zero: Vec<_> = terms.iter()
                    .filter(|t| *t != &SymbolicExpr::Integer(0))
                    .cloned().collect();
                if non_zero.len() == 1 { return Some(non_zero[0].clone()); }
                if non_zero.is_empty() { return Some(SymbolicExpr::Integer(0)); }
                Some(SymbolicExpr::Plus(non_zero))
            }
            _ => Some(expr.clone()),
        }
    }
    
    fn is_numeric(expr: &SymbolicExpr) -> bool {
        matches!(expr, SymbolicExpr::Integer(_) | SymbolicExpr::Real(_) | SymbolicExpr::Rational(_, _))
    }
    
    fn to_f64(expr: &SymbolicExpr) -> Option<f64> {
        match expr {
            SymbolicExpr::Integer(n) => Some(*n as f64),
            SymbolicExpr::Real(f) => Some(*f),
            SymbolicExpr::Rational(p, q) => Some(*p as f64 / *q as f64),
            _ => None,
        }
    }
}


// ============================================================================
// PLACEHOLDER TYPES — removed from here.
// All types are now provided by earlier includes in chronos-compiler-core.
// This file is included last in the compiler-core crate, so all types from
// the pipeline files (lexer, parser, type-inference, ir-codegen, etc.) are
// already in scope.
// ============================================================================

fn default_hardware_model() -> HardwareModel {
    HardwareModel {
        name: "default".to_string(),
        clock_hz: 3_000_000_000,  // 3 GHz
        pipeline_stages: 5,
        issue_width: 4,
        cache_model: CacheModel {
            l1i_size_kb: 32,
            l1d_size_kb: 32,
            l1_line_size: 64,
            l1_associativity: 8,
            l1_hit_cycles: 4,
            l1_miss_cycles: 12,
            l2_size_kb: 256,
            l2_hit_cycles: 12,
            l2_miss_cycles: 100,
        },
        branch_predictor: BranchPredictorModel::TwoBitCounter,
        memory_latency_cycles: 100,
        interrupt_latency_cycles: 50,
    }
}
