// ============================================================================
// CHRONOS LEXER — Tokenization using the `logos` crate
// ============================================================================
// The lexer is the first phase of compilation. It takes raw source text and
// produces a stream of tokens. Every keyword, operator, literal, and
// punctuation mark in the Chronos language is defined here.
//
// To use this, add to Cargo.toml:
//   logos = "0.14"
// ============================================================================

use logos::Logos;
use std::fmt; // provided by parent scope in compiler-core
use std::ops::Range;

/// Every possible token in the Chronos language.
/// The `logos` derive macro generates an optimized DFA-based lexer from the
/// regex patterns and string literals we attach to each variant.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r]+")]  // Skip horizontal whitespace (but NOT newlines — they matter for some rules)
pub enum Token {
    // =====================================================================
    // KEYWORDS — Organized by feature area
    // =====================================================================

    // --- Type declarations (Feature 2: classes, structs, templates, etc.) ---
    #[token("class")]       KwClass,
    #[token("struct")]      KwStruct,
    #[token("enum")]        KwEnum,
    #[token("trait")]       KwTrait,
    #[token("impl")]        KwImpl,
    #[token("data")]        KwData,          // Kotlin-style data class: `data class Point(...)`
    #[token("sealed")]      KwSealed,        // Scala/Kotlin sealed hierarchy
    #[token("template")]    KwTemplate,      // C++-style templates
    #[token("type")]        KwType,          // Type alias
    #[token("interface")]   KwInterface,     // Java-style interface (sugar for trait)
    #[token("object")]      KwObject,        // Scala companion object / Kotlin object
    #[token("companion")]   KwCompanion,     // Companion object marker
    #[token("extension")]   KwExtension,     // Kotlin extension functions
    #[token("abstract")]    KwAbstract,
    #[token("final")]       KwFinal,
    #[token("open")]        KwOpen,          // Kotlin: class is inheritable
    #[token("override")]    KwOverride,
    #[token("virtual")]     KwVirtual,       // C++-style virtual dispatch
    #[token("operator")]    KwOperator,      // Operator overloading
    
    // --- Visibility modifiers ---
    #[token("pub")]         KwPub,
    #[token("private")]     KwPrivate,
    #[token("protected")]   KwProtected,
    #[token("internal")]    KwInternal,      // Kotlin module visibility
    #[token("crate")]       KwCrate,         // Rust crate visibility

    // --- Variable & function declarations ---
    #[token("let")]         KwLet,
    #[token("var")]         KwVar,           // Mutable binding (Kotlin/Swift style)
    #[token("const")]       KwConst,         // Compile-time constant
    #[token("static")]      KwStatic,
    #[token("fn")]          KwFn,
    #[token("fun")]         KwFun,           // Alternative function keyword (Kotlin style)
    #[token("async")]       KwAsync,
    #[token("await")]       KwAwait,
    #[token("inline")]      KwInline,
    #[token("constexpr")]   KwConstexpr,     // C++-style compile-time evaluation
    #[token("suspend")]     KwSuspend,       // Kotlin coroutine marker

    // --- Control flow ---
    #[token("if")]          KwIf,
    #[token("else")]        KwElse,
    #[token("match")]       KwMatch,
    #[token("when")]        KwWhen,          // Kotlin-style when expression
    #[token("for")]         KwFor,
    #[token("while")]       KwWhile,
    #[token("loop")]        KwLoop,          // Rust infinite loop
    #[token("do")]          KwDo,
    #[token("break")]       KwBreak,
    #[token("continue")]    KwContinue,
    #[token("return")]      KwReturn,
    #[token("yield")]       KwYield,         // Generator / coroutine yield
    #[token("throw")]       KwThrow,
    #[token("try")]         KwTry,
    #[token("catch")]       KwCatch,
    #[token("finally")]     KwFinally,

    // --- Ownership & memory (Feature 7 & 8: no GC, linear types) ---
    #[token("move")]        KwMove,          // Explicit ownership transfer
    #[token("ref")]         KwRef,           // Borrow reference
    #[token("mut")]         KwMut,           // Mutable qualifier
    #[token("own")]         KwOwn,           // Explicit ownership annotation
    #[token("linear")]      KwLinear,        // Linear type: must be used exactly once
    #[token("affine")]      KwAffine,        // Affine type: used at most once
    #[token("drop")]        KwDrop,          // Explicit resource destruction
    #[token("unsafe")]      KwUnsafe,
    #[token("pin")]         KwPin,           // Pin in memory (for self-referential types)
    
    // --- Module system ---
    #[token("mod")]         KwMod,
    #[token("module")]      KwModule,
    #[token("import")]      KwImport,
    #[token("use")]         KwUse,
    #[token("from")]        KwFrom,
    #[token("as")]          KwAs,
    #[token("package")]     KwPackage,       // Kotlin/Java package declaration
    #[token("export")]      KwExport,

    // --- Type system qualifiers (Feature 1: universal type system) ---
    #[token("where")]       KwWhere,         // Generic constraints
    #[token("is")]          KwIs,            // Type check operator
    #[token("in")]          KwIn,            // Container membership / variance
    #[token("out")]         KwOut,           // Covariance marker (Kotlin)
    #[token("reified")]     KwReified,       // Kotlin reified generics
    #[token("crossinline")] KwCrossinline,   // Kotlin inline function constraint
    #[token("noinline")]    KwNoinline,      // Kotlin inline function constraint
    #[token("typeof")]      KwTypeof,
    #[token("sizeof")]      KwSizeof,        // C/C++ size query
    #[token("alignof")]     KwAlignof,

    // --- Effect system (Feature 8: tracked effects) ---
    #[token("pure")]        KwPure,          // No side effects
    #[token("effect")]      KwEffect,        // Declare an effect
    #[token("handle")]      KwHandle,        // Effect handler
    #[token("perform")]     KwPerform,       // Trigger an effect
    
    // --- AI-native keywords (Feature 4: AI syntax) ---
    #[token("ai")]          KwAi,
    #[token("skill")]       KwSkill,         // AI skill declaration
    #[token("tool")]        KwTool,          // AI tool declaration
    #[token("pipeline")]    KwPipeline,      // AI pipeline chaining
    #[token("instruction")] KwInstruction,   // LLM instruction
    #[token("constraint")]  KwConstraint,    // AI constraint
    #[token("example")]     KwExample,       // Few-shot example
    #[token("persona")]     KwPersona,       // AI persona definition
    #[token("schema")]      KwSchema,        // Input/output schema
    #[token("invoke")]      KwInvoke,        // Call an AI skill

    // --- Hardware / Device keywords (Feature 5: Mojo-inspired) ---
    #[token("kernel")]      KwKernel,        // GPU/TPU/NPU kernel
    #[token("device")]      KwDevice,        // Device scope
    #[token("gpu")]         KwGpu,
    #[token("cpu")]         KwCpu,
    #[token("tpu")]         KwTpu,
    #[token("npu")]         KwNpu,
    #[token("tensor")]      KwTensor,        // First-class tensor type
    #[token("shape")]       KwShape,
    #[token("tile")]        KwTile,          // Memory tiling annotation
    #[token("vectorize")]   KwVectorize,     // SIMD vectorization
    #[token("unroll")]      KwUnroll,        // Loop unrolling
    #[token("parallel")]    KwParallel,      // Parallelization hint
    #[token("shared")]      KwShared,        // Shared memory (GPU)
    #[token("global")]      KwGlobal,        // Global memory (GPU)
    #[token("simd")]        KwSimd,          // SIMD operation marker
    #[token("distributed")] KwDistributed,   // Multi-device distribution

    // --- Degradable functions (Feature 6) ---
    #[token("degradable")]  KwDegradable,    // Mark function as degradable
    #[token("expires")]     KwExpires,       // Expiry date
    #[token("warns")]       KwWarns,         // Warning date
    #[token("replaces")]    KwReplaces,      // Replacement function
    #[token("deprecated")]  KwDeprecated,    // Standard deprecation (non-degradable)

    // --- Version control annotations (Feature 3) ---
    #[token("version")]     KwVersion,
    #[token("track")]       KwTrack,
    #[token("risk")]        KwRisk,
    #[token("branch")]      KwBranch,
    #[token("geo")]         KwGeo,

    // --- Boolean & null literals ---
    #[token("true")]        KwTrue,
    #[token("false")]       KwFalse,
    #[token("null")]        KwNull,          // Explicit null (behind Optional type)
    #[token("nil")]         KwNil,           // Alternative null keyword
    #[token("none")]        KwNone,          // Option::None
    #[token("some")]        KwSome,          // Option::Some
    #[token("self")]        KwSelf_,
    #[token("Self")]        KwSelfType,      // The implementing type in trait impls
    #[token("super")]       KwSuper,
    #[token("this")]        KwThis,          // Alternative self (Java/Kotlin style)
    
    // --- Miscellaneous ---
    #[token("new")]         KwNew,           // Constructor invocation
    #[token("delete")]      KwDelete,        // Explicit deallocation (unsafe context)
    #[token("with")]        KwWith,          // Context manager / resource scope
    #[token("defer")]       KwDefer,         // Go-style deferred execution
    #[token("lazy")]        KwLazy,          // Lazy evaluation
    #[token("val")]         KwVal,           // Immutable binding (Scala/Kotlin)
    #[token("case")]        KwCase,          // Pattern case / case class
    #[token("derive")]      KwDerive,        // Auto-derive trait implementations

    // =====================================================================
    // PRIMITIVE TYPE KEYWORDS (Feature 1)
    // =====================================================================
    #[token("void")]        TyVoid,
    #[token("bool")]        TyBool,
    #[token("i8")]          TyI8,
    #[token("i16")]         TyI16,
    #[token("i32")]         TyI32,
    #[token("i64")]         TyI64,
    #[token("i128")]        TyI128,
    #[token("u8")]          TyU8,
    #[token("u16")]         TyU16,
    #[token("u32")]         TyU32,
    #[token("u64")]         TyU64,
    #[token("u128")]        TyU128,
    #[token("f16")]         TyF16,
    #[token("f32")]         TyF32,
    #[token("f64")]         TyF64,
    #[token("f128")]        TyF128,
    #[token("bf16")]        TyBf16,          // Brain float for AI workloads
    #[token("char")]        TyChar,
    #[token("str")]         TyStr,
    #[token("string")]      TyString,        // Owned heap string
    #[token("int")]         TyInt,           // Arbitrary precision integer
    #[token("uint")]        TyUInt,          // Arbitrary precision unsigned integer
    #[token("usize")]       TyUsize,
    #[token("isize")]       TyIsize,
    #[token("never")]       TyNever,         // Bottom type (function never returns)

    // =====================================================================
    // LITERALS
    // =====================================================================
    
    // Integer literals: decimal, hex, octal, binary, with optional type suffix
    // Examples: 42, 0xFF, 0o77, 0b1010, 42_i64, 1_000_000
    #[regex(r"0[xX][0-9a-fA-F][0-9a-fA-F_]*", |lex| parse_int_hex(lex.slice()))]
    #[regex(r"0[oO][0-7][0-7_]*", |lex| parse_int_oct(lex.slice()))]
    #[regex(r"0[bB][01][01_]*", |lex| parse_int_bin(lex.slice()))]
    #[regex(r"[0-9][0-9_]*", |lex| parse_int_dec(lex.slice()), priority = 2)]
    IntLiteral(i128),

    // Float literals: 3.14, 1.0e10, .5, 1e-3
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9_]+)?", |lex| lex.slice().replace('_', "").parse::<f64>().ok())]
    #[regex(r"[0-9][0-9_]*[eE][+-]?[0-9_]+", |lex| lex.slice().replace('_', "").parse::<f64>().ok())]
    FloatLiteral(f64),

    // String literals: "hello world", with escape sequences
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(s[1..s.len()-1].to_string())
    })]
    StringLiteral(String),

    // Raw string literals: r"no \escapes here", r#"can contain "quotes""#
    #[regex(r#"r"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        Some(s[2..s.len()-1].to_string())
    })]
    RawStringLiteral(String),

    // Multi-line string literals (triple-quoted, like Kotlin/Python)
    #[regex(r#""""[^"]*""""#, |lex| {
        let s = lex.slice();
        Some(s[3..s.len()-3].to_string())
    })]
    MultiLineStringLiteral(String),

    // Character literal: 'a', '\n', '\x41'
    #[regex(r"'([^'\\]|\\.)'", |lex| {
        let s = lex.slice();
        s[1..s.len()-1].chars().next()
    })]
    CharLiteral(char),

    // =====================================================================
    // IDENTIFIERS
    // =====================================================================
    
    // Standard identifier: starts with letter or underscore, then alphanumeric
    // Checked AFTER all keywords (logos handles priority automatically)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string(), priority = 1)]
    Identifier(String),

    // Lifetime identifier: 'a, 'static (Rust-style)
    #[regex(r"'[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice()[1..].to_string())]
    Lifetime(String),
    
    // Annotation: @deprecated, @version, @device(gpu)
    #[regex(r"@[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice()[1..].to_string())]
    Annotation(String),

    // =====================================================================
    // OPERATORS
    // =====================================================================
    
    // Arithmetic
    #[token("+")]   Plus,
    #[token("-")]   Minus,
    #[token("*")]   Star,
    #[token("/")]   Slash,
    #[token("%")]   Percent,
    #[token("**")]  DoubleStar,      // Exponentiation (Python-style)
    
    // Comparison
    #[token("==")]  EqEq,
    #[token("!=")]  NotEq,
    #[token("<")]   Lt,
    #[token(">")]   Gt,
    #[token("<=")]  LtEq,
    #[token(">=")]  GtEq,
    #[token("<=>")] Spaceship,       // Three-way comparison (C++20)
    
    // Logical
    #[token("&&")]  AndAnd,
    #[token("||")]  OrOr,
    #[token("!")]   Bang,
    
    // Bitwise
    #[token("&")]   Ampersand,
    #[token("|")]   Pipe,
    #[token("^")]   Caret,
    #[token("~")]   Tilde,
    #[token("<<")]  Shl,
    #[token(">>")]  Shr,
    
    // Assignment
    #[token("=")]   Eq,
    #[token("+=")]  PlusEq,
    #[token("-=")]  MinusEq,
    #[token("*=")]  StarEq,
    #[token("/=")]  SlashEq,
    #[token("%=")]  PercentEq,
    #[token("&=")]  AmpEq,
    #[token("|=")]  PipeEq,
    #[token("^=")]  CaretEq,
    #[token("<<=")]  ShlEq,
    #[token(">>=")]  ShrEq,
    
    // Special operators
    #[token("->")]  Arrow,           // Return type annotation
    #[token("=>")]  FatArrow,        // Lambda / match arm
    #[token("::")]  PathSep,         // Path separator (Rust/C++)
    #[token("..")]  DotDot,          // Range operator
    #[token("..=")] DotDotEq,        // Inclusive range
    #[token("...")]  Ellipsis,       // Variadic / spread
    #[token("?")]   Question,        // Optional chaining / error propagation
    #[token("?.")]  SafeDot,         // Kotlin safe call operator
    #[token("?:")]  Elvis,           // Kotlin elvis operator
    #[token("!!")]  DoubleBang,      // Kotlin non-null assertion
    #[token("@@")]  AtAt,            // AI invocation operator
    #[token("|>")]  PipeForward,     // Pipe / compose forward (F#/Elixir)
    #[token("<|")]  PipeBackward,    // Pipe backward
    #[token("@")]   At,              // Matrix multiply (Python-style) / decoration
    
    // =====================================================================
    // DELIMITERS & PUNCTUATION
    // =====================================================================
    #[token("(")]   LParen,
    #[token(")")]   RParen,
    #[token("[")]   LBracket,
    #[token("]")]   RBracket,
    #[token("{")]   LBrace,
    #[token("}")]   RBrace,
    // LAngle and RAngle are aliases for Lt/Gt — do not re-define the token pattern.
    // They are kept as variants for parser compatibility but without #[token] attributes
    // to avoid logos conflicts. Use Lt/Gt for the actual < and > tokens.
    LAngle,   // alias for Lt (<)
    RAngle,   // alias for Gt (>)
    #[token(",")]   Comma,
    #[token(";")]   Semicolon,
    #[token(":")]   Colon,
    #[token(".")]   Dot,
    #[token("#")]   Hash,            // Attribute / preprocessor / version annotation

    // =====================================================================
    // COMMENTS (captured as tokens for version control annotations)
    // =====================================================================
    
    // Line comment: // this is a comment
    #[regex(r"//[^\n]*", |lex| lex.slice().to_string())]
    LineComment(String),

    // Doc comment: /// documentation
    #[regex(r"///[^\n]*", |lex| lex.slice().to_string())]
    DocComment(String),

    // Version annotation comment: //! @version(track="stable")
    // These are special comments that the version control system parses.
    #[regex(r"//![^\n]*", |lex| lex.slice().to_string())]
    VersionComment(String),

    // Block comment: /* ... */  (including nested)
    // Note: logos doesn't handle nested block comments well, so we'll handle
    // those in a post-processing step. This catches simple ones.
    #[regex(r"/\*([^*]|\*[^/])*\*/", |lex| lex.slice().to_string())]
    BlockComment(String),

    // =====================================================================
    // NEWLINES (significant for some parsing rules)
    // =====================================================================
    #[token("\n")]  Newline,

    // =====================================================================
    // ERROR RECOVERY
    // =====================================================================
    // Any character that doesn't match the above rules produces an Error.
    // The parser can use this for error recovery and better diagnostics.

    // =====================================================================
    // EXTENDED TOKENS (from lexer-v2 / parser-v2)
    // These keywords are recognized by the parser but added here so that
    // all code compiled in chronos-compiler-core has a single Token enum.
    // They do NOT have #[token] attributes since they should be merged
    // into the logos-derived lexer's keyword patterns (see integration plan).
    // =====================================================================

    // Networking / Protocol
    #[token("protocol")]    KwProtocol,
    #[token("field")]       KwField,
    #[token("state")]       KwState,
    #[token("on")]          KwOn,
    #[token("endpoint")]    KwEndpoint,
    #[token("route")]       KwRoute,
    #[token("transition")]  KwTransition,

    // OS / Systems
    #[token("driver")]      KwDriver,
    #[token("interrupt")]   KwInterrupt,
    #[token("isr")]         KwIsr,
    #[token("syscall")]     KwSyscall,
    #[token("critical")]    KwCritical,
    #[token("register")]    KwRegister,
    #[token("spawn")]       KwSpawn,

    // Distributed
    #[token("consensus")]   KwConsensus,
    #[token("crdt")]        KwCrdt,
    #[token("saga")]        KwSaga,

    // Blockchain / Smart Contracts
    #[token("contract")]    KwContract,
    #[token("storage")]     KwStorage,
    #[token("event")]       KwEvent,
    #[token("modifier")]    KwModifier,
    #[token("payable")]     KwPayable,
    #[token("view")]        KwView,
    #[token("address")]     KwAddress,
    #[token("emit")]        KwEmit,
    #[token("transaction")]  KwTransaction,

    // Proof / Formal Verification
    #[token("theorem")]     KwTheorem,
    #[token("lemma")]       KwLemma,
    #[token("axiom")]       KwAxiom,
    #[token("proof")]       KwProof,
    #[token("forall")]      KwForall,
    #[token("exists")]      KwExists,
    #[token("sorry")]       KwSorry,
    #[token("by")]          KwBy,
    #[token("induction")]   KwInduction,
    #[token("property")]    KwProperty,

    // Database
    #[token("table")]       KwTable,
    #[token("index")]       KwIndex,
    #[token("query")]       KwQuery,

    // Macros / Comptime
    #[token("macro")]       KwMacro,
    #[token("comptime")]    KwComptime,
    #[token("embed")]       KwEmbed,

    // Multimedia
    #[token("scene")]       KwScene,

    // Robotics
    #[token("robot")]       KwRobot,

    // Quantum Computing
    #[token("qubit")]       KwQubit,
    #[token("qreg")]        KwQreg,
    #[token("creg")]        KwCreg,
    #[token("gate")]        KwGate,
    #[token("measure")]     KwMeasure,
    #[token("circuit")]     KwCircuit,

    // GUI / Widgets
    #[token("widget")]      KwWidget,
    #[token("component")]   KwComponent,

    // ECS (Entity-Component-System)
    #[token("entity")]      KwEntity,
    #[token("system")]      KwSystem,
    #[token("world")]       KwWorld,

    // Bioinformatics
    #[token("genome")]      KwGenome,

    // Finance
    #[token("portfolio")]   KwPortfolio,
    #[token("backtest")]    KwBacktest,

    // Geospatial
    #[token("spatial")]     KwSpatial,

    // Testing / Observability
    #[token("test")]        KwTest,
    #[token("bench")]       KwBench,
    #[token("fuzz")]        KwFuzz,
    #[token("trace")]       KwTrace,
    #[token("log")]         KwLog,
    #[token("audit")]       KwAudit,

    // FFI
    #[token("extern")]      KwExtern,
    #[token("foreign")]     KwForeign,

    // Simulation / FEM
    #[token("simulation")]  KwSimulation,
    #[token("fem")]         KwFem,
    #[token("mesh")]        KwMesh,
    #[token("boundary")]    KwBoundary,
    #[token("solve")]       KwSolve,
    #[token("domain")]      KwDomain,
    #[token("physics")]     KwPhysics,

    // Shader
    KwShaderKw,  // internal use, no token literal

    // Pipeline (avoid conflict with existing KwPipeline)
    KwPipelineKw,

    // Misc
    #[token("assert")]      KwAssert,
    #[token("require")]     KwRequire,
    #[token("verify")]      KwVerify,
    #[token("invariant")]   KwInvariant,
    #[token("requires")]    KwRequires,
    #[token("ensures")]     KwEnsures,
    #[token("oracle")]      KwOracle,
    #[token("entangle")]    KwEntangle,
    #[token("superposition")] KwSuperposition,
    #[token("audio")]       KwAudio,
    #[token("video")]       KwVideo,
    #[token("controller")]  KwController,
    #[token("macro_rules")] KwMacroRules,

    // Extra operators
    #[token("<-")]   LeftArrow,       // Channel receive
    #[token("==>")]  LongFatArrow,    // Proof implication

    // Additional type keywords
    TyBigInt,
    TyBigFloat,
    TyComplex,
    TyDecimal,
    TyQuaternion,
    TyRatio,
    TySymbol,
}

// =====================================================================
// Helper functions for parsing integer literals
// =====================================================================

fn parse_int_dec(s: &str) -> Option<i128> {
    s.replace('_', "").parse::<i128>().ok()
}

fn parse_int_hex(s: &str) -> Option<i128> {
    // Strip the 0x/0X prefix and underscores
    let clean = s[2..].replace('_', "");
    i128::from_str_radix(&clean, 16).ok()
}

fn parse_int_oct(s: &str) -> Option<i128> {
    let clean = s[2..].replace('_', "");
    i128::from_str_radix(&clean, 8).ok()
}

fn parse_int_bin(s: &str) -> Option<i128> {
    let clean = s[2..].replace('_', "");
    i128::from_str_radix(&clean, 2).ok()
}

// =====================================================================
// SPANNED TOKEN — Associates each token with its source location
// =====================================================================
// The parser needs to know WHERE each token came from for error messages,
// incremental compilation (Feature 3), and version annotations.

#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Range<usize>,    // Byte offset range in source
    pub line: usize,           // 1-indexed line number
    pub column: usize,         // 1-indexed column number
}

impl fmt::Display for SpannedToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{} {:?}", self.line, self.column, self.token)
    }
}

// =====================================================================
// LEXER DRIVER — Converts source text into a Vec<SpannedToken>
// =====================================================================
// This wraps the logos-generated lexer with line/column tracking and
// filters out noise tokens (plain comments, whitespace) while preserving
// meaningful ones (doc comments, version annotations).

pub struct Lexer {
    /// The original source text (kept for error reporting and span slicing).
    pub source: String,
    /// The file name (for diagnostics).
    pub filename: String,
}

impl Lexer {
    pub fn new(source: String, filename: String) -> Self {
        Self { source, filename }
    }

    /// Tokenize the entire source, producing a vector of spanned tokens.
    /// Plain comments are discarded; doc comments and version annotations
    /// are preserved because the parser and version control system need them.
    pub fn tokenize(&self) -> Result<Vec<SpannedToken>, Vec<LexerError>> {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();
        
        // We track line and column manually because logos only gives byte offsets.
        let mut line = 1usize;
        let mut line_start_offset = 0usize;

        // Create the logos lexer.
        let mut lex = Token::lexer(&self.source);

        while let Some(result) = lex.next() {
            let span = lex.span();
            
            // Calculate line and column from byte offset.
            // We count newlines in the text between the last processed position
            // and the current span start.
            let preceding = &self.source[line_start_offset..span.start];
            for ch in preceding.chars() {
                if ch == '\n' {
                    line += 1;
                    line_start_offset = span.start;  // Approximate
                }
            }
            let column = span.start - line_start_offset + 1;

            match result {
                Ok(token) => {
                    // Filter decisions: what to keep, what to discard.
                    match &token {
                        // Discard plain line comments (but NOT doc or version comments).
                        Token::LineComment(_) => continue,
                        // Discard block comments.
                        Token::BlockComment(_) => continue,
                        // Count newlines but don't emit them as tokens
                        // (the parser uses semicolons, not newlines, for statement
                        // termination — but we could change this for a Python-like mode).
                        Token::Newline => {
                            line += 1;
                            line_start_offset = span.end;
                            continue;
                        }
                        // Everything else is meaningful.
                        _ => {}
                    }

                    tokens.push(SpannedToken {
                        token,
                        span,
                        line,
                        column,
                    });
                }
                Err(_) => {
                    // Collect the error and continue — we want to report ALL
                    // lexer errors, not just the first one.
                    errors.push(LexerError {
                        message: format!(
                            "Unexpected character: '{}'",
                            &self.source[span.start..span.end]
                        ),
                        line,
                        column,
                        span,
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(tokens)
        } else {
            Err(errors)
        }
    }
}

#[derive(Debug, Clone)]
pub struct LexerError {
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub span: Range<usize>,
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[LEX ERROR] {}:{}: {}", self.line, self.column, self.message)
    }
}

// =====================================================================
// CONVENIENCE: Token classification methods
// =====================================================================
// These help the parser ask questions like "is this token an operator?"
// without writing giant match arms everywhere.

impl Token {
    /// Returns true if this token can start a type expression.
    pub fn is_type_start(&self) -> bool {
        matches!(self,
            Token::TyVoid | Token::TyBool |
            Token::TyI8 | Token::TyI16 | Token::TyI32 | Token::TyI64 | Token::TyI128 |
            Token::TyU8 | Token::TyU16 | Token::TyU32 | Token::TyU64 | Token::TyU128 |
            Token::TyF16 | Token::TyF32 | Token::TyF64 | Token::TyF128 | Token::TyBf16 |
            Token::TyChar | Token::TyStr | Token::TyString |
            Token::TyInt | Token::TyUInt | Token::TyUsize | Token::TyIsize |
            Token::TyNever |
            Token::KwTensor |
            Token::KwLinear | Token::KwAffine |
            Token::LParen |      // Tuple type: (i32, str)
            Token::LBracket |    // Array type: [i32; 10]
            Token::Ampersand |   // Reference type: &T, &mut T
            Token::KwFn |        // Function type: fn(i32) -> bool
            Token::Identifier(_) // Named type: MyStruct, Vec<T>
        )
    }

    /// Returns true if this token can start a statement.
    pub fn is_statement_start(&self) -> bool {
        matches!(self,
            Token::KwLet | Token::KwVar | Token::KwVal | Token::KwConst |
            Token::KwReturn | Token::KwBreak | Token::KwContinue |
            Token::KwIf | Token::KwMatch | Token::KwWhen |
            Token::KwFor | Token::KwWhile | Token::KwLoop | Token::KwDo |
            Token::KwDrop | Token::KwDefer |
            Token::KwUnsafe | Token::KwDevice |
            Token::Identifier(_) | Token::KwSelf_ | Token::KwThis
        )
    }

    /// Returns true if this token can start an item (top-level declaration).
    pub fn is_item_start(&self) -> bool {
        matches!(self,
            Token::KwPub | Token::KwPrivate | Token::KwProtected | Token::KwInternal |
            Token::KwClass | Token::KwStruct | Token::KwEnum | Token::KwTrait |
            Token::KwImpl | Token::KwData | Token::KwSealed | Token::KwTemplate |
            Token::KwType | Token::KwInterface | Token::KwObject |
            Token::KwFn | Token::KwFun | Token::KwAsync |
            Token::KwAbstract | Token::KwFinal | Token::KwOpen |
            Token::KwDegradable |
            Token::KwAi | Token::KwKernel |
            Token::KwMod | Token::KwModule | Token::KwImport | Token::KwUse |
            Token::KwConst | Token::KwStatic |
            Token::KwDerive |
            Token::Hash |         // #[attribute]
            Token::Annotation(_)  // @annotation
        )
    }

    /// Returns the binding power (precedence) for binary operators.
    /// Higher number = tighter binding. Returns None for non-operators.
    /// We use Pratt parsing, so we need left and right binding powers.
    pub fn prefix_binding_power(&self) -> Option<u8> {
        match self {
            Token::Minus | Token::Bang | Token::Tilde => Some(25),
            Token::Ampersand => Some(25),  // &expr (reference)
            Token::Star => Some(25),       // *expr (dereference)
            _ => None,
        }
    }

    pub fn infix_binding_power(&self) -> Option<(u8, u8)> {
        // Returns (left_bp, right_bp). Left-associative: left < right.
        // Right-associative: left > right.
        match self {
            // Assignment (right-associative)
            Token::Eq | Token::PlusEq | Token::MinusEq |
            Token::StarEq | Token::SlashEq | Token::PercentEq => Some((2, 1)),
            
            // Pipe forward (left-associative)
            Token::PipeForward => Some((3, 4)),
            
            // Logical OR
            Token::OrOr => Some((5, 6)),
            
            // Logical AND
            Token::AndAnd => Some((7, 8)),
            
            // Comparison (non-associative, but we treat as left for simplicity)
            Token::EqEq | Token::NotEq |
            Token::Lt | Token::Gt | Token::LtEq | Token::GtEq |
            Token::Spaceship => Some((9, 10)),
            
            // Bitwise OR
            Token::Pipe => Some((11, 12)),
            
            // Bitwise XOR
            Token::Caret => Some((13, 14)),
            
            // Bitwise AND
            Token::Ampersand => Some((15, 16)),
            
            // Shift
            Token::Shl | Token::Shr => Some((17, 18)),
            
            // Range
            Token::DotDot | Token::DotDotEq => Some((19, 20)),
            
            // Addition / Subtraction
            Token::Plus | Token::Minus => Some((21, 22)),
            
            // Multiplication / Division / Modulo / Matrix multiply
            Token::Star | Token::Slash | Token::Percent | Token::At => Some((23, 24)),
            
            // Exponentiation (right-associative)
            Token::DoubleStar => Some((28, 27)),
            
            // Member access, method call
            Token::Dot | Token::SafeDot | Token::PathSep => Some((29, 30)),
            
            _ => None,
        }
    }

    pub fn postfix_binding_power(&self) -> Option<u8> {
        match self {
            Token::Question => Some(27),      // ? error propagation
            Token::DoubleBang => Some(27),    // !! non-null assertion
            Token::LParen => Some(27),        // function call
            Token::LBracket => Some(27),      // indexing
            _ => None,
        }
    }
}

// =====================================================================
// TESTS
// =====================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokens() {
        let source = r#"
            fn main() -> i32 {
                let x: i32 = 42;
                let y = 3.14;
                return x;
            }
        "#;
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        // Check that we get the expected token types.
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwFn)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::Identifier(ref s) if s == "main")));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::IntLiteral(42))));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::FloatLiteral(f) if (f - 3.14).abs() < 1e-10)));
    }

    #[test]
    fn test_ai_native_tokens() {
        let source = r#"
            ai skill summarize {
                instruction "Summarize the input document."
                schema { input: str, output: str }
            }
        "#;
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwAi)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwSkill)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwInstruction)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwSchema)));
    }

    #[test]
    fn test_degradable_tokens() {
        let source = r#"
            degradable fn old_handler(req: Request) -> Response
                expires 2027-01-01
                warns 2026-06-01
                replaces new_handler
            {
                // legacy code
            }
        "#;
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwDegradable)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwExpires)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwWarns)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwReplaces)));
    }

    #[test]
    fn test_kernel_tokens() {
        let source = r#"
            kernel gpu matmul<T: Numeric>(
                a: tensor<T, [M, K]>,
                b: tensor<T, [K, N]>
            ) -> tensor<T, [M, N]> {
                tile [16, 16] {
                    parallel { vectorize 8 { /* compute */ } }
                }
            }
        "#;
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwKernel)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwGpu)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwTile)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwParallel)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwVectorize)));
    }

    #[test]
    fn test_linear_type_tokens() {
        let source = r#"
            let file: linear File = File::open("data.txt");
            let conn: affine DbConnection = db.connect();
        "#;
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwLinear)));
        assert!(tokens.iter().any(|t| matches!(t.token, Token::KwAffine)));
    }

    #[test]
    fn test_version_comment() {
        let source = r#"
            //! @version(track="stable", geo="EU")
            fn gdpr_compliant_handler() {}
        "#;
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        // Version comments should be preserved in the token stream.
        assert!(tokens.iter().any(|t| matches!(t.token, Token::VersionComment(_))));
    }

    #[test]
    fn test_operator_precedence() {
        // Verify that our binding powers are correct.
        assert!(Token::Star.infix_binding_power().unwrap().0 > Token::Plus.infix_binding_power().unwrap().0);
        assert!(Token::AndAnd.infix_binding_power().unwrap().0 > Token::OrOr.infix_binding_power().unwrap().0);
        assert!(Token::DoubleStar.infix_binding_power().unwrap().0 > Token::Star.infix_binding_power().unwrap().0);
    }

    #[test]
    fn test_hex_and_binary_literals() {
        let source = "0xFF 0b1010 0o77 1_000_000";
        let lexer = Lexer::new(source.to_string(), "test.chr".to_string());
        let tokens = lexer.tokenize().expect("Lexer should succeed");
        
        let ints: Vec<i128> = tokens.iter()
            .filter_map(|t| if let Token::IntLiteral(n) = t.token { Some(n) } else { None })
            .collect();
        assert_eq!(ints, vec![255, 10, 63, 1_000_000]);
    }
}
