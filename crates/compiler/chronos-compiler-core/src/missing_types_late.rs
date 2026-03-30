// ============================================================================
// LATE-BINDING ORCHESTRATOR IMPLEMENTATIONS
// ============================================================================
// These wrap the real pipeline implementations with the "Unified" API that
// chronos-unified-integration.rs expects.
// ============================================================================

pub struct RulesFileParser;
impl RulesFileParser {
    pub fn find_and_parse() -> Option<VersionRules> { None }
}

pub struct DegradationChecker;
impl DegradationChecker {
    pub fn check(_d: &DegradableFunctionDecl) -> Vec<CompilerDiagnostic> { Vec::new() }
}

// Wraps the real TypeInferenceEngine.
pub struct UnifiedTypeInference(TypeInferenceEngine);
impl UnifiedTypeInference {
    pub fn new() -> Self { Self(TypeInferenceEngine::new()) }
    pub fn infer_program(&mut self, ast: &Program, _ctx: &mut CompilationContext) -> InferenceResult {
        // Delegate to the real Hindley-Milner engine.
        self.0.infer_program(ast)
    }
    pub fn get_type_environment(&self) -> TypeEnvironment { TypeEnvironment }
    pub fn get_effect_map(&self) -> HashMap<String, Vec<Effect>> { HashMap::new() }
}

// Wraps the real ASTLowering.
pub struct UnifiedASTLowering(ASTLowering);
impl UnifiedASTLowering {
    pub fn new(f: &str, t: CompilationTarget) -> Self {
        Self(ASTLowering::new(f, t))
    }
    pub fn lower_program(
        &mut self,
        a: &Program,
        _t: &TypeEnvironment,
        _e: &HashMap<String, Vec<Effect>>,
        _u: &UnitMap,
    ) -> IRModule {
        // lower_program returns &IRModule; clone to get an owned copy.
        self.0.lower_program(a).clone()
    }
}

// Wraps the real LLVMBackend.
pub struct UnifiedLLVMBackend(LLVMBackend);
impl UnifiedLLVMBackend {
    pub fn new() -> Self { Self(LLVMBackend::new()) }
    pub fn generate(&mut self, m: &IRModule) -> String {
        // LLVMBackend implements CodeGenBackend<Output=String>.
        self.0.generate(m)
    }
}

// Wraps the real CUDABackend.
pub struct UnifiedCUDABackend(CUDABackend);
impl UnifiedCUDABackend {
    pub fn new() -> Self { Self(CUDABackend::new()) }
    pub fn generate(&mut self, m: &IRModule) -> String {
        self.0.generate(m)
    }
}

pub struct UnifiedXLABackend;
impl UnifiedXLABackend {
    pub fn new() -> Self { Self }
    pub fn generate(&mut self, _m: &IRModule) -> String { String::new() }
}

pub struct QASMBackend;
impl QASMBackend {
    pub fn new() -> Self { Self }
    pub fn generate(&mut self, _m: &IRModule) -> String { String::new() }
}

pub struct VersionComputer;
impl VersionComputer {
    pub fn new(_r: &VersionRules) -> Self { Self }
    pub fn compute_version(&self, _m: &IRModule) -> String { "0.1.0".to_string() }
}

pub struct ProofChecker;
impl ProofChecker {
    pub fn new() -> Self { Self }
}
