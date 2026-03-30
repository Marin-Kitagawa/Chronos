// ============================================================================
// MISSING AST TYPES — defined here, not in any pipeline .rs file
// ============================================================================
// These types are referenced throughout the Chronos compiler pipeline but
// were never defined in any individual pipeline source file. They represent
// the "AST layer" that was expected to exist as a separate shared module.
//
// Types defined in pipeline files (and therefore NOT here):
//   - Token, SpannedToken, Lexer, LexerError  → chronos-lexer.rs
//   - UniversalType                            → chronos-stdlib-types.rs
//   - SecurityMode, Severity, WCETResult, etc. → chronos-security-realtime.rs
//   - IRModule, IRFunction, IRDevice, etc.     → chronos-ir-codegen.rs
//   - InferenceResult, TypeError               → chronos-type-inference.rs
//   - DomainIROpcode, PhysicalUnitSignature    → chronos-inference-ir-v2.rs
// ============================================================================

// Note: HashMap and other std imports are already in scope from
// chronos-stdlib-types.rs which is included before this file in lib.rs.

// ============================================================================
// CORE TYPE: ChronosType
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum ChronosType {
    Void, Never, Bool, Char, Str,
    Int8, Int16, Int32, Int64, Int128, IntArbitrary,
    UInt8, UInt16, UInt32, UInt64, UInt128, UIntArbitrary,
    Float16, Float32, Float64, Float128, BFloat16,
    Owned(Box<ChronosType>),
    Linear(Box<ChronosType>),
    Affine(Box<ChronosType>),
    Borrowed { inner: Box<ChronosType>, mutable: bool, lifetime: Lifetime },
    Tuple(Vec<ChronosType>),
    Array { element: Box<ChronosType>, size: Option<usize> },
    Slice { element: Box<ChronosType> },
    Optional(Box<ChronosType>),
    Function { params: Vec<ChronosType>, return_type: Box<ChronosType>, effects: Vec<Effect> },
    Tensor { element: Box<ChronosType>, shape: TensorShape, device: DeviceTarget },
    Named { name: String, module_path: Vec<String> },
    Unknown,
}

// ============================================================================
// TENSOR SHAPE
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum TensorShape {
    Static(Vec<usize>),
    Symbolic(Vec<String>),
    Dynamic,
}

// ============================================================================
// DEVICE TARGET
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceTarget {
    Cpu,
    Gpu { index: u32 },
    Tpu { index: u32 },
    Npu { index: u32 },
    Auto,
}

// ============================================================================
// EFFECTS
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    IO, Alloc, Async, Gpu, Tpu, Network, Throw(String), Diverge, Quantum, Custom(String),
    GpuKernel, NpuKernel, TpuKernel, Pure,
}

// ============================================================================
// LIFETIMES, VARIANCE, TYPE PARAMS, TYPE BOUNDS
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum Lifetime { Named(String), Static, Inferred }

#[derive(Debug, Clone, PartialEq)]
pub enum Variance { Covariant, Contravariant, Invariant }

#[derive(Debug, Clone, PartialEq)]
pub struct TypeParam {
    pub name: String,
    pub variance: Variance,
    pub bounds: Vec<TypeBound>,
    pub default: Option<ChronosType>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeBound { Implements(String), Lifetime(Lifetime) }

// ============================================================================
// VISIBILITY
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum Visibility { Public, Private, Protected, Internal, Crate }

// ============================================================================
// ANNOTATIONS
// ============================================================================
#[derive(Debug, Clone)]
pub struct Annotation {
    pub name: String,
    pub args: Vec<Expression>,
}

// ============================================================================
// EXPRESSIONS
// ============================================================================
#[derive(Debug, Clone)]
pub enum Expression {
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    Identifier(String),
    BinaryOp { left: Box<Expression>, op: BinOp, right: Box<Expression> },
    UnaryOp { op: UnaryOp, expr: Box<Expression> },
    Call { function: Box<Expression>, args: Vec<Expression> },
    MethodCall { object: Box<Expression>, method: String, args: Vec<Expression> },
    FieldAccess { object: Box<Expression>, field: String },
    If { condition: Box<Expression>, then_branch: Box<Expression>, else_branch: Option<Box<Expression>> },
    Match { scrutinee: Box<Expression>, arms: Vec<MatchArm> },
    Lambda { params: Vec<Parameter>, body: Box<Expression>, return_type: Option<ChronosType> },
    Block(Vec<Statement>),
    AiInvoke { skill_name: String, inputs: HashMap<String, Expression> },
    Await(Box<Expression>),
    Return(Box<Expression>),
    TypeAscription { expr: Box<Expression>, ty: ChronosType },
    TensorLiteral { elements: Vec<Expression>, shape: Vec<usize>, device: DeviceTarget },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    And, Or,
    Eq, Neq, Lt, Gt, Lte, Gte,
    BitAnd, BitOr, Xor, Shl, Shr,
    MatMul, Assign,
    AddAssign, SubAssign, MulAssign, DivAssign, ModAssign,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp { Neg, Not, BitNot, Ref, Deref, MutRef }

#[derive(Debug, Clone)]
pub struct MatchArm { pub pattern: Pattern, pub guard: Option<Expression>, pub body: Expression }

#[derive(Debug, Clone)]
pub enum Pattern {
    Wildcard,
    Binding(String),
    Literal(Expression),
    Constructor { name: String, fields: Vec<Pattern> },
    Tuple(Vec<Pattern>),
    Or(Vec<Pattern>),
}

// ============================================================================
// STATEMENTS
// ============================================================================
#[derive(Debug, Clone)]
pub enum Statement {
    Let { name: String, ty: Option<ChronosType>, value: Expression, mutable: bool },
    Assignment { target: Expression, value: Expression },
    Return(Option<Expression>),
    Break,
    Continue,
    While { condition: Expression, body: Vec<Statement> },
    For { binding: String, iterator: Expression, body: Vec<Statement> },
    Drop(String),
    DeviceScope { target: DeviceTarget, body: Vec<Statement> },
    ExprStatement(Expression),
    Require { condition: Expression, message: Option<Expression> },
}

// ============================================================================
// PARAMETER
// ============================================================================
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub ty: ChronosType,
    pub default: Option<Expression>,
    pub is_variadic: bool,
}

// ============================================================================
// FIELD DECLARATION
// ============================================================================
#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: String,
    pub ty: ChronosType,
    pub visibility: Visibility,
    pub mutable: bool,
    pub default: Option<Expression>,
}

// ============================================================================
// FUNCTION SIGNATURE & DECLARATION
// ============================================================================
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub visibility: Visibility,
    pub type_params: Vec<TypeParam>,
    pub params: Vec<Parameter>,
    pub return_type: ChronosType,
    pub effects: Vec<Effect>,
    pub is_async: bool,
    pub is_const: bool,
    pub is_inline: bool,
    pub lifetime_params: Vec<Lifetime>,
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub signature: FunctionSignature,
    pub body: Vec<Statement>,
    pub annotations: Vec<Annotation>,
}

// ============================================================================
// DEGRADABLE FUNCTION
// ============================================================================
#[derive(Debug, Clone)]
pub struct DegradableFunctionDecl {
    pub function: FunctionDecl,
    pub degradation: DegradationSchedule,
}

#[derive(Debug, Clone)]
pub struct DegradationSchedule {
    pub warn_after: ChronosTimestamp,
    pub error_after: Option<ChronosTimestamp>,
    pub expire_after: ChronosTimestamp,
    pub replacement: Option<String>,
    pub reason: Option<String>,
    pub phases: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ChronosTimestamp {
    pub year: u16, pub month: u8, pub day: u8, pub hour: u8, pub minute: u8,
}
impl ChronosTimestamp {
    pub fn new(year: u16, month: u8, day: u8) -> Self {
        Self { year, month, day, hour: 0, minute: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct VersionAnnotation { pub version: String, pub target: String }

#[derive(Debug, Clone)]
pub struct VersionRules;

// ============================================================================
// CLASS DECLARATIONS
// ============================================================================
#[derive(Debug, Clone)]
pub struct ClassDecl {
    pub name: String,
    pub visibility: Visibility,
    pub type_params: Vec<TypeParam>,
    pub superclass: Option<ChronosType>,
    pub interfaces: Vec<ChronosType>,
    pub fields: Vec<FieldDecl>,
    pub methods: Vec<FunctionDecl>,
    pub constructors: Vec<ConstructorDecl>,
    pub destructor: Option<DestructorDecl>,
    pub is_abstract: bool,
    pub is_final: bool,
    pub companion: Option<Box<ClassDecl>>,
}

#[derive(Debug, Clone)]
pub struct ConstructorDecl {
    pub visibility: Visibility, pub params: Vec<Parameter>,
    pub body: Vec<Statement>, pub is_primary: bool,
}

#[derive(Debug, Clone)]
pub struct DestructorDecl { pub body: Vec<Statement> }

// ============================================================================
// DATA CLASS
// ============================================================================
#[derive(Debug, Clone)]
pub struct DataClassDecl {
    pub name: String, pub visibility: Visibility,
    pub type_params: Vec<TypeParam>, pub fields: Vec<FieldDecl>,
    pub auto_derive: Vec<AutoDerive>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AutoDerive { Equals, Hash, ToString, Copy, Destructure, Serialize, Order, Debug }

// ============================================================================
// STRUCT
// ============================================================================
#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String, pub visibility: Visibility,
    pub type_params: Vec<TypeParam>, pub fields: Vec<FieldDecl>,
    pub is_packed: bool, pub is_repr_c: bool,
    pub copy_semantics: CopySemantics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CopySemantics { Copy, Move, Clone }

// ============================================================================
// ENUM
// ============================================================================
#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String, pub type_params: Vec<TypeParam>, pub variants: Vec<EnumVariant>,
}

#[derive(Debug, Clone)]
pub struct EnumVariant { pub name: String, pub fields: VariantFields }

#[derive(Debug, Clone)]
pub enum VariantFields {
    Unit, Tuple(Vec<ChronosType>), Struct(Vec<FieldDecl>),
}

// ============================================================================
// SEALED HIERARCHY
// ============================================================================
#[derive(Debug, Clone)]
pub struct SealedDecl {
    pub name: String, pub type_params: Vec<TypeParam>, pub variants: Vec<SealedVariant>,
}

#[derive(Debug, Clone)]
pub enum SealedVariant {
    DataClass(DataClassDecl), SubClass(ClassDecl), Singleton(String),
}

// ============================================================================
// TRAIT
// ============================================================================
#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String, pub type_params: Vec<TypeParam>,
    pub super_traits: Vec<ChronosType>,
    pub methods: Vec<TraitMethod>, pub associated_types: Vec<AssociatedType>,
}

#[derive(Debug, Clone)]
pub struct TraitMethod {
    pub signature: FunctionSignature, pub default_impl: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
pub struct AssociatedType {
    pub name: String, pub bounds: Vec<TypeBound>, pub default: Option<ChronosType>,
}

// ============================================================================
// IMPL BLOCK
// ============================================================================
#[derive(Debug, Clone)]
pub struct ImplBlock {
    pub type_params: Vec<TypeParam>, pub trait_name: Option<String>,
    pub target_type: ChronosType, pub methods: Vec<FunctionDecl>,
    pub associated_types: Vec<(String, ChronosType)>,
}

// ============================================================================
// TEMPLATE
// ============================================================================
#[derive(Debug, Clone)]
pub struct TemplateDecl {
    pub name: String, pub params: Vec<TemplateParam>,
    pub body: TemplateBody, pub specializations: Vec<TemplateSpecialization>,
}

#[derive(Debug, Clone)]
pub enum TemplateParam {
    Type { name: String, bounds: Vec<TypeBound> },
    Value { name: String, ty: ChronosType },
}

#[derive(Debug, Clone)]
pub enum TemplateBody {
    Class(ClassDecl), Struct(StructDecl), Function(FunctionDecl),
}

#[derive(Debug, Clone)]
pub struct TemplateSpecialization {
    pub args: Vec<ChronosType>, pub body: TemplateBody,
}

// ============================================================================
// TYPE ALIAS, MODULE, IMPORT
// ============================================================================
#[derive(Debug, Clone)]
pub struct TypeAliasDecl {
    pub name: String, pub type_params: Vec<TypeParam>, pub aliased: ChronosType,
}

#[derive(Debug, Clone)]
pub struct ModuleDecl { pub name: String, pub items: Vec<Item> }

#[derive(Debug, Clone)]
pub struct ImportDecl {
    pub path: Vec<String>, pub alias: Option<String>, pub selective: Vec<String>,
}

// ============================================================================
// KERNEL DECLARATION
// ============================================================================
#[derive(Debug, Clone)]
pub struct KernelDecl {
    pub name: String, pub target: DeviceTarget,
    pub params: Vec<KernelParam>, pub return_type: ChronosType,
    pub body: Vec<Statement>, pub launch_config: Option<KernelLaunchConfig>,
    pub memory_annotations: Vec<MemoryAnnotation>,
}

#[derive(Debug, Clone)]
pub struct KernelParam { pub name: String, pub ty: ChronosType, pub memory_space: MemorySpace }

#[derive(Debug, Clone, PartialEq)]
pub enum MemorySpace { Global, Shared, Local, Constant, Registers, Unified, Hbm }

#[derive(Debug, Clone)]
pub struct KernelLaunchConfig {
    pub grid: Expression, pub block: Expression, pub shared_mem_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MemoryAnnotation { pub name: String, pub space: MemorySpace }

// ============================================================================
// AI-NATIVE DECLARATIONS
// ============================================================================
#[derive(Debug, Clone)]
pub struct AiSkillDecl {
    pub name: String, pub description: String,
    pub input_schema: Vec<AiParam>, pub output_schema: AiOutputSchema,
    pub instructions: Vec<AiInstruction>, pub constraints: Vec<AiConstraint>,
    pub examples: Vec<AiExample>, pub model_requirements: Option<AiModelRequirements>,
}

#[derive(Debug, Clone)]
pub struct AiToolDecl {
    pub name: String, pub description: String,
    pub parameters: Vec<AiParam>, pub return_type: AiOutputSchema,
    pub implementation: FunctionDecl,
    pub retry_policy: Option<AiRetryPolicy>, pub rate_limit: Option<AiRateLimit>,
}

#[derive(Debug, Clone)]
pub struct AiPipelineDecl {
    pub name: String, pub stages: Vec<AiPipelineStage>, pub error_handling: AiErrorStrategy,
}

#[derive(Debug, Clone)]
pub struct AiParam {
    pub name: String, pub description: String, pub ty: AiType,
    pub required: bool, pub default: Option<Expression>, pub validation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AiOutputSchema { pub ty: AiType, pub description: String, pub format: Option<String> }

#[derive(Debug, Clone)]
pub enum AiType { Text, Number, Boolean, List(Box<AiType>), Object, ChronosType(ChronosType) }

#[derive(Debug, Clone)]
pub struct AiInstruction { pub kind: AiInstructionKind, pub content: String, pub priority: u8 }

#[derive(Debug, Clone)]
pub enum AiInstructionKind { System, User, Assistant }

#[derive(Debug, Clone)]
pub struct AiConstraint { pub description: String, pub enforcement: ConstraintEnforcement }

#[derive(Debug, Clone)]
pub enum ConstraintEnforcement { CompileTime, Runtime, PostCondition }

#[derive(Debug, Clone)]
pub struct AiExample { pub input: String, pub output: String, pub explanation: Option<String> }

#[derive(Debug, Clone)]
pub struct AiModelRequirements {
    pub min_context_length: Option<usize>, pub required_capabilities: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AiPipelineStage {
    pub name: String, pub skill_or_tool: String,
    pub input_mapping: HashMap<String, String>, pub condition: Option<Expression>,
}

#[derive(Debug, Clone)]
pub enum AiErrorStrategy { Fail, Skip, Retry(u32), Fallback(String) }

#[derive(Debug, Clone)]
pub struct AiRetryPolicy { pub max_retries: u32, pub delay_ms: u64 }

#[derive(Debug, Clone)]
pub struct AiRateLimit { pub requests_per_minute: u32 }

// ============================================================================
// TOP-LEVEL ITEM AND PROGRAM
// ============================================================================
#[derive(Debug, Clone)]
pub enum Item {
    FunctionDecl(FunctionDecl),
    DegradableFunctionDecl(DegradableFunctionDecl),
    ClassDecl(ClassDecl),
    DataClassDecl(DataClassDecl),
    StructDecl(StructDecl),
    EnumDecl(EnumDecl),
    SealedDecl(SealedDecl),
    TraitDecl(TraitDecl),
    ImplBlock(ImplBlock),
    TemplateDecl(TemplateDecl),
    TypeAlias(TypeAliasDecl),
    ModuleDecl(ModuleDecl),
    ImportDecl(ImportDecl),
    KernelDecl(KernelDecl),
    AiSkillDecl(AiSkillDecl),
    AiToolDecl(AiToolDecl),
    AiPipelineDecl(AiPipelineDecl),
}

#[derive(Debug, Clone)]
pub struct Program { pub items: Vec<Item>, pub version_rules: Option<VersionRules> }

// ============================================================================
// DIAGNOSTICS
// ============================================================================
#[derive(Debug, Clone, PartialEq)]
pub enum DiagnosticLevel { Error, Warning, Info, Hint }

#[derive(Debug, Clone)]
pub struct CompilerDiagnostic { pub level: DiagnosticLevel, pub message: String, pub code: String }

// ============================================================================
// COMPILATION RESULT
// ============================================================================
// CompilationTarget is defined in chronos-ir-codegen.rs with full fields.
// CompilationResult is defined here since it's not in any pipeline file.
// It previously referenced CompilationContext (from unified-integration.rs),
// but since all includes land in the same scope, forward refs work.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    pub success: bool,
    pub diagnostics: Vec<CompilerDiagnostic>,
    pub llvm_ir: Option<String>,
    pub c_code: Option<String>,
    pub output_files: HashMap<String, Vec<u8>>,
}
impl CompilationResult {
    pub fn from_context(ctx: CompilationContext) -> Self {
        let llvm_ir = ctx.generated_code.get("llvm")
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let c_code = ctx.generated_code.get("c")
            .map(|b| String::from_utf8_lossy(b).into_owned());
        let success = ctx.audit_passed &&
            !ctx.all_diagnostics.iter().any(|d| d.level == DiagnosticLevel::Error);
        Self {
            success,
            diagnostics: ctx.all_diagnostics,
            llvm_ir,
            c_code,
            output_files: ctx.generated_code,
        }
    }
}

// ============================================================================
// SIMULATION DOMAIN
// ============================================================================
#[derive(Debug, Clone)]
pub struct SimulationDomain { pub name: String }

// ============================================================================
// TYPE ENVIRONMENT
// ============================================================================
#[derive(Debug, Clone)]
pub struct TypeEnvironment;
impl TypeEnvironment { pub fn new() -> Self { Self } }

// ============================================================================
// COMPILER CONFIG
// ============================================================================
// References SecurityMode (from security-realtime.rs) and HardwareModel
// (from security-realtime.rs). This works because all include!() content
// lands in the same Rust module, where names resolve regardless of order.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub security_mode: SecurityMode,
    pub realtime_mode: bool,
    pub target: CompilationTarget,
    pub hardware_model: Option<HardwareModel>,
}

// UnitMap is defined in chronos-unified-integration.rs as:
//   pub type UnitMap = HashMap<String, PhysicalUnitSignature>;
// We don't define it here to avoid the duplicate.
