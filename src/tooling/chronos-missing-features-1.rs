// ============================================================================
// CHRONOS EXTENDED FEATURES — PART 1
// ============================================================================
// Missing domains identified after a comprehensive audit:
//
//   1.  Networking & Protocol Engineering (TCP/UDP/QUIC/custom protocols)
//   2.  Operating System Primitives (syscalls, drivers, kernel modules)
//   3.  Distributed Systems (consensus, CRDTs, distributed transactions)
//   4.  Blockchain / Web3 / Smart Contracts
//   5.  Formal Verification & Proof Assistants (Coq/Lean/Agda-level)
//   6.  Database Engine Internals (query planning, storage engines)
//   7.  Compiler & Language Tooling (macros, proc macros, DSL embedding)
//   8.  Audio / Video / Multimedia Processing
//   9.  Robotics & Control Systems
//  10.  Natural Language Processing (built-in tokenizers, parsers, etc.)
//  11.  Quantum Computing Primitives
//  12.  UI / GUI Framework
//  13.  Game Engine Primitives (ECS, physics, rendering pipeline)
//  14.  Bioinformatics (DNA/RNA/protein sequence analysis)
//  15.  Financial Engineering (pricing models, risk analysis)
//  16.  Geospatial / GIS
//  17.  Signal Processing & DSP
//  18.  Build System / Package Manager (self-hosting)
//  19.  Testing Framework (property-based, fuzzing, mutation)
//  20.  Documentation System (literate programming)
//  21.  Accessibility 
//  22.  Internationalization / Localization
//  23.  Interop / FFI (C, C++, Python, JVM, .NET, WASM)
//  24.  Hot code reloading / Live patching
//  25.  Observability (tracing, metrics, structured logging)
// ============================================================================

use std::collections::HashMap;
use std::time::Duration;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

// ============================================================================
// 1. NETWORKING & PROTOCOL ENGINEERING
// ============================================================================
// Chronos treats network protocols as first-class type-safe constructs.
// You don't write raw byte parsing — you declare a protocol's wire format
// and the compiler generates the parser, serializer, and state machine.

/// A protocol declaration. The compiler generates an optimized, zero-copy
/// parser and serializer from this specification.
///
/// Chronos syntax:
/// ```chronos
/// protocol HTTP2Frame {
///     field length: u24;
///     field frame_type: u8;
///     field flags: u8;
///     field reserved: bit;
///     field stream_id: u31;
///     field payload: [u8; length];
///
///     state_machine {
///         Idle -> Open on HEADERS;
///         Open -> HalfClosed on END_STREAM;
///         HalfClosed -> Closed on END_STREAM;
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ProtocolDecl {
    pub name: String,
    pub version: String,
    pub transport: TransportLayer,
    pub fields: Vec<ProtocolField>,
    pub state_machine: Option<ProtocolStateMachine>,
    pub constraints: Vec<ProtocolConstraint>,
    pub endianness: Endianness,
    pub security: Option<ProtocolSecurity>,
}

#[derive(Debug, Clone)]
pub enum TransportLayer {
    TCP, UDP, QUIC, SCTP, Unix, Raw,
    TLS { min_version: String },
    DTLS,
    WebSocket,
    WebTransport,
    Custom { name: String },
}

#[derive(Debug, Clone)]
pub struct ProtocolField {
    pub name: String,
    pub wire_type: WireType,
    pub endianness: Option<Endianness>,
    pub validation: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum WireType {
    /// Fixed-width integer (any bit width, not just power-of-2).
    UInt(u8),            // u1 through u64
    Int(u8),
    Float(u8),           // f16, f32, f64
    FixedBytes(usize),
    /// Variable-length with a length prefix.
    VarBytes { length_field: String },
    /// Null-terminated string.
    CString,
    /// Length-prefixed string.
    LPString { length_bits: u8 },
    /// Bit field within a byte.
    Bits(u8),
    /// Padding (ignored bytes).
    Padding(usize),
    /// Nested protocol.
    Nested(String),
    /// Tagged union (like protobuf oneof).
    TaggedUnion { tag_field: String, variants: Vec<(u64, String, WireType)> },
    /// Repeated field (like protobuf repeated).
    Repeated { count_field: String, element: Box<WireType> },
    /// Optional field (present based on a flag).
    Optional { flag_field: String, bit: u8, inner: Box<WireType> },
}

#[derive(Debug, Clone)]
pub enum Endianness { Big, Little, Native, Network } // Network = Big

#[derive(Debug, Clone)]
pub struct ProtocolStateMachine {
    pub states: Vec<String>,
    pub initial_state: String,
    pub transitions: Vec<StateTransition>,
    pub terminal_states: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from: String,
    pub to: String,
    pub trigger: String,
    pub guard: Option<String>,
    pub action: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ProtocolConstraint {
    Checksum { algorithm: String, field: String, covers: Vec<String> },
    MaxSize(usize),
    Alignment(usize),
    Version { field: String, supported: Vec<u32> },
}

#[derive(Debug, Clone)]
pub struct ProtocolSecurity {
    pub encryption: Option<String>,
    pub authentication: Option<String>,
    pub integrity: Option<String>,
    pub replay_protection: bool,
}

/// Built-in networking primitives beyond raw sockets.
#[derive(Debug, Clone)]
pub enum NetworkPrimitive {
    // --- Server patterns ---
    TCPListener { addr: String, backlog: u32 },
    UDPSocket { addr: String },
    QUICEndpoint { addr: String, cert: String, key: String },
    HTTPServer { addr: String, routes: Vec<Route>, middleware: Vec<Middleware> },
    GRPCServer { addr: String, services: Vec<String> },
    WebSocketServer { addr: String },
    
    // --- Service mesh / microservices ---
    ServiceDiscovery { backend: DiscoveryBackend },
    LoadBalancer { strategy: LBStrategy, health_check: HealthCheck },
    CircuitBreaker { threshold: u32, timeout: Duration, half_open_max: u32 },
    RateLimiter { algorithm: RateLimitAlgo },
    Retry { max_attempts: u32, backoff: BackoffStrategy },
    Bulkhead { max_concurrent: u32, queue_size: u32 },
    
    // --- DNS ---
    DNSResolver { servers: Vec<String>, cache_ttl: Duration },
    DNSServer { zone_file: String },
}

#[derive(Debug, Clone)]
pub struct Route { pub method: String, pub path: String, pub handler: String, pub middleware: Vec<String> }
#[derive(Debug, Clone)]
pub enum Middleware { Logging, Auth(String), CORS(CORSConfig), RateLimit(u32), Compression, Timeout(Duration), Tracing, Metrics }
#[derive(Debug, Clone)]
pub struct CORSConfig { pub origins: Vec<String>, pub methods: Vec<String>, pub headers: Vec<String>, pub max_age: u32 }
#[derive(Debug, Clone)]
pub enum DiscoveryBackend { DNS, Consul, Etcd, Kubernetes, Eureka, ZooKeeper }
#[derive(Debug, Clone)]
pub enum LBStrategy { RoundRobin, WeightedRoundRobin(Vec<u32>), LeastConnections, Random, IPHash, Consistent }
#[derive(Debug, Clone)]
pub struct HealthCheck { pub interval: Duration, pub timeout: Duration, pub path: String, pub threshold: u32 }
#[derive(Debug, Clone)]
pub enum RateLimitAlgo { TokenBucket { rate: f64, burst: u32 }, SlidingWindow { window: Duration, max: u32 }, LeakyBucket { rate: f64 }, FixedWindow { window: Duration, max: u32 } }
#[derive(Debug, Clone)]
pub enum BackoffStrategy { Constant(Duration), Linear(Duration), Exponential { initial: Duration, max: Duration, multiplier: f64 }, Jittered(Box<BackoffStrategy>) }


// ============================================================================
// 2. OPERATING SYSTEM PRIMITIVES
// ============================================================================
// Chronos can be used as a systems programming language for writing OS
// kernels, drivers, and embedded firmware. These primitives give direct
// access to hardware and OS facilities.

/// OS-level primitives that Chronos exposes as built-in constructs.
#[derive(Debug, Clone)]
pub enum OSPrimitive {
    // --- Process management ---
    Process { command: String, args: Vec<String>, env: HashMap<String, String> },
    Thread { stack_size: usize, priority: i32, affinity: Option<Vec<u32>> },
    Fiber,                              // Lightweight cooperative thread
    Coroutine,                          // Stackful coroutine
    
    // --- IPC ---
    Pipe, NamedPipe(String), UnixSocket(String),
    SharedMemory { name: String, size: usize },
    MessageQueue { name: String, max_messages: usize },
    Signal(i32),
    
    // --- File system ---
    File { path: String, mode: FileMode, flags: Vec<FileFlag> },
    Directory(String),
    Symlink { target: String, link: String },
    Watch { path: String, events: Vec<FSEvent> },       // inotify/kqueue/FSEvents
    TempFile, TempDir,
    MemoryMappedFile { path: String, offset: usize, size: usize },
    
    // --- Kernel / Driver ---
    Syscall { number: u64, args: Vec<u64> },
    Interrupt { vector: u8, handler: String },
    DMA { source: usize, dest: usize, size: usize },
    MMIO { base: usize, size: usize },
    PCI { vendor: u16, device: u16 },
    USB { class: u8, subclass: u8, protocol: u8 },
    GPIO { pin: u32, mode: GPIOMode },
    I2C { bus: u8, address: u8 },
    SPI { bus: u8, cs: u8, speed_hz: u32 },
    UART { port: String, baud: u32, config: UARTConfig },
    
    // --- Virtualization ---
    Container { image: String, volumes: Vec<(String, String)>, ports: Vec<(u16, u16)> },
    VM { cpus: u32, memory_mb: u64, disk: String },
    Sandbox { capabilities: Vec<String>, seccomp_filter: Option<String> },
    
    // --- Resource limits ---
    CGroup { cpu_quota: Option<f64>, memory_limit: Option<usize>, io_weight: Option<u32> },
    Namespace(Vec<NamespaceKind>),
    ResourceLimit { resource: RLimitKind, soft: u64, hard: u64 },
}

#[derive(Debug, Clone)]
pub enum FileMode { Read, Write, ReadWrite, Append, Create, Truncate, Exclusive }
#[derive(Debug, Clone)]
pub enum FileFlag { NonBlocking, Sync, DSync, Direct, NoFollow, NoAtime }
#[derive(Debug, Clone)]
pub enum FSEvent { Create, Modify, Delete, Rename, Attrib, Close, Open }
#[derive(Debug, Clone)]
pub enum GPIOMode { Input, Output, InputPullUp, InputPullDown, Alternate(u8) }
#[derive(Debug, Clone)]
pub struct UARTConfig { pub data_bits: u8, pub stop_bits: u8, pub parity: Parity, pub flow_control: FlowControl }
#[derive(Debug, Clone)]
pub enum Parity { None, Even, Odd }
#[derive(Debug, Clone)]
pub enum FlowControl { None, Hardware, Software }
#[derive(Debug, Clone)]
pub enum NamespaceKind { PID, Mount, Network, UTS, IPC, User, Cgroup }
#[derive(Debug, Clone)]
pub enum RLimitKind { OpenFiles, StackSize, AddressSpace, CPUTime, FileSize, Processes }


// ============================================================================
// 3. DISTRIBUTED SYSTEMS
// ============================================================================
// First-class support for building distributed systems with correctness
// guarantees. The compiler can verify distributed protocols against their
// specifications using TLA+-style model checking.

/// Distributed system primitives.
#[derive(Debug, Clone)]
pub enum DistributedPrimitive {
    // --- Consensus ---
    Raft { cluster: Vec<String>, election_timeout: Duration },
    Paxos { acceptors: Vec<String>, quorum_size: usize },
    PBFT { replicas: Vec<String>, fault_tolerance: usize },
    ZAB { servers: Vec<String> },       // ZooKeeper Atomic Broadcast
    
    // --- CRDTs (Conflict-Free Replicated Data Types) ---
    GCounter,                           // Grow-only counter
    PNCounter,                          // Positive-negative counter
    GSet(String),                       // Grow-only set
    ORSet(String),                      // Observed-Remove set
    LWWRegister(String),                // Last-Writer-Wins register
    MVRegister(String),                 // Multi-Value register
    LWWMap(String, String),             // Last-Writer-Wins map
    RGA(String),                        // Replicated Growable Array (for text)
    Sequence(String),                   // Ordered sequence CRDT
    
    // --- Distributed transactions ---
    TwoPhaseCommit { coordinator: String, participants: Vec<String> },
    ThreePhaseCommit { coordinator: String, participants: Vec<String> },
    Saga { steps: Vec<SagaStep> },      // Compensating transactions
    
    // --- Distributed data structures ---
    DistributedLock { backend: LockBackend },
    DistributedQueue { backend: String },
    DistributedMap { sharding: ShardingStrategy, replication: u32 },
    EventLog { partitions: u32, replication: u32 },
    
    // --- Clock synchronization ---
    LamportClock,
    VectorClock { nodes: Vec<String> },
    HybridLogicalClock,
    TrueTime,                           // Google Spanner-style bounded clock
    
    // --- Consistency models ---
    StrongConsistency,
    EventualConsistency { convergence_bound: Option<Duration> },
    CausalConsistency,
    ReadYourWrites,
    SessionConsistency,
    LinearizableOps,
    SerializableOps,
    SnapshotIsolation,
    
    // --- Model checking (TLA+-inspired) ---
    ModelCheck {
        spec: String,                   // TLA+ or Chronos model spec
        invariants: Vec<String>,
        liveness: Vec<String>,
        max_states: usize,
    },
}

#[derive(Debug, Clone)]
pub struct SagaStep {
    pub name: String,
    pub action: String,
    pub compensation: String,           // Undo action if later step fails
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub enum LockBackend { Redis, ZooKeeper, Etcd, DynamoDB, PostgresAdvisory }
#[derive(Debug, Clone)]
pub enum ShardingStrategy { Hash, Range, Geographic, Consistent, Directory }


// ============================================================================
// 4. BLOCKCHAIN / SMART CONTRACTS
// ============================================================================
// Chronos can compile to EVM bytecode, Solana BPF, or custom blockchain VMs.
// Smart contracts are type-checked for reentrancy, integer overflow, and
// access control at compile time.

#[derive(Debug, Clone)]
pub struct SmartContract {
    pub name: String,
    pub target: BlockchainTarget,
    pub state: Vec<StorageVar>,
    pub functions: Vec<ContractFunction>,
    pub events: Vec<ContractEvent>,
    pub modifiers: Vec<ContractModifier>,
    pub security_checks: Vec<ContractSecurityCheck>,
}

#[derive(Debug, Clone)]
pub enum BlockchainTarget {
    EVM { solidity_version: String },
    Solana { anchor: bool },
    CosmWasm,
    NEAR,
    Substrate,
    StarkNet { cairo: bool },
    Aptos { move_lang: bool },
    Custom { vm_name: String },
}

#[derive(Debug, Clone)]
pub struct StorageVar {
    pub name: String,
    pub ty: StorageType,
    pub visibility: StorageVisibility,
    pub slot: Option<u64>,              // Explicit storage slot (EVM)
}

#[derive(Debug, Clone)]
pub enum StorageType {
    UInt256, Int256, Address, Bool, Bytes(u8),
    String_, Array(Box<StorageType>),
    Mapping(Box<StorageType>, Box<StorageType>),
    Struct(String, Vec<(String, StorageType)>),
}

#[derive(Debug, Clone)]
pub enum StorageVisibility { Public, Internal, Private }

#[derive(Debug, Clone)]
pub struct ContractFunction {
    pub name: String,
    pub mutability: StateMutability,
    pub visibility: StorageVisibility,
    pub params: Vec<(String, StorageType)>,
    pub returns: Vec<StorageType>,
    pub modifiers: Vec<String>,
    pub gas_limit: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum StateMutability { Pure, View, Nonpayable, Payable }

#[derive(Debug, Clone)]
pub struct ContractEvent {
    pub name: String,
    pub fields: Vec<(String, StorageType, bool)>, // (name, type, indexed)
}

#[derive(Debug, Clone)]
pub struct ContractModifier {
    pub name: String,
    pub params: Vec<(String, StorageType)>,
    pub body: String,
}

/// Security checks the compiler performs on smart contracts.
#[derive(Debug, Clone)]
pub enum ContractSecurityCheck {
    ReentrancyGuard,                    // Check-Effects-Interactions pattern
    IntegerOverflow,                    // All arithmetic is checked
    AccessControl,                      // Role-based access verification
    FrontrunningProtection,
    FlashLoanAttack,
    OracleManipulation,
    StorageCollision,
    DelegatecallSafety,
    SelfdestructProtection,
    GasGriefing,
    TimestampDependence,
}


// ============================================================================
// 5. FORMAL VERIFICATION & PROOF ASSISTANT
// ============================================================================
// Chronos integrates a Lean4/Coq-level proof assistant directly into the
// language. You can write proofs alongside your code, and the compiler
// verifies them. This goes beyond refinement types into full dependent
// type theory and tactics.

#[derive(Debug, Clone)]
pub enum ProofConstruct {
    // --- Propositions ---
    Prop(String),                       // A named proposition
    Forall { var: String, ty: String, body: Box<ProofConstruct> },
    Exists { var: String, ty: String, body: Box<ProofConstruct> },
    Implies(Box<ProofConstruct>, Box<ProofConstruct>),
    And(Box<ProofConstruct>, Box<ProofConstruct>),
    Or(Box<ProofConstruct>, Box<ProofConstruct>),
    Not(Box<ProofConstruct>),
    Equal(Box<ProofConstruct>, Box<ProofConstruct>),
    
    // --- Types as propositions (Curry-Howard) ---
    PiType { var: String, domain: Box<ProofConstruct>, codomain: Box<ProofConstruct> },
    SigmaType { var: String, fst: Box<ProofConstruct>, snd: Box<ProofConstruct> },
    InductiveType { name: String, constructors: Vec<(String, Vec<ProofConstruct>)> },
    
    // --- Proof terms ---
    Axiom(String),
    Theorem { name: String, statement: Box<ProofConstruct>, proof: Box<ProofTactic> },
    Lemma { name: String, statement: Box<ProofConstruct>, proof: Box<ProofTactic> },
    Definition { name: String, ty: Box<ProofConstruct>, body: Box<ProofConstruct> },
    
    // --- Contract specifications on functions ---
    Precondition(String),
    Postcondition(String),
    Invariant(String),
    LoopVariant(String),                // Decreasing measure for termination
    LoopInvariant(String),
    
    // --- Temporal logic (for distributed systems) ---
    Always(Box<ProofConstruct>),        // □P (globally P)
    Eventually(Box<ProofConstruct>),    // ◇P (eventually P)
    Until(Box<ProofConstruct>, Box<ProofConstruct>), // P U Q
    Leads_To(Box<ProofConstruct>, Box<ProofConstruct>), // P ~> Q
    Fairness(FairnessKind, Box<ProofConstruct>),
}

#[derive(Debug, Clone)]
pub enum FairnessKind { Weak, Strong }

/// Proof tactics — the "instructions" for constructing proofs.
/// These mirror Lean4/Coq tactics.
#[derive(Debug, Clone)]
pub enum ProofTactic {
    // --- Basic tactics ---
    Intro(Vec<String>),                 // Introduce hypotheses
    Apply(String),                      // Apply a lemma/theorem
    Exact(String),                      // Provide the exact proof term
    Assumption,                         // Use a hypothesis from context
    Reflexivity,                        // Prove x = x
    Symmetry,                           // Flip an equality
    Transitivity(String),               // Chain equalities
    
    // --- Structural tactics ---
    Induction { var: String, cases: Vec<(String, Box<ProofTactic>)> },
    Cases { expr: String, arms: Vec<(String, Box<ProofTactic>)> },
    Destruct(String),
    Generalize(String),
    Specialize(String, String),
    
    // --- Rewriting ---
    Rewrite(String, RewriteDirection),
    Simp(Vec<String>),                  // Simplification with lemma set
    Ring,                               // Ring theory solver
    Omega,                              // Linear arithmetic solver
    Linarith,                           // Linear arithmetic over ordered fields
    NormNum,                            // Numeric normalization
    DecideEquality,
    
    // --- Automation ---
    Auto(u32),                          // Auto-solve with search depth
    Trivial,
    Tauto,                              // Propositional tautology checker
    Blast,                              // Classical logic automation
    SMT,                                // Delegate to Z3/CVC5
    
    // --- Composition ---
    Sequence(Vec<ProofTactic>),
    Try(Box<ProofTactic>),              // Try tactic, succeed silently if fails
    Repeat(Box<ProofTactic>),           // Repeat until failure
    First(Vec<ProofTactic>),            // Try each, use first success
    All(Vec<ProofTactic>),              // Apply to all goals
    Focus(usize, Box<ProofTactic>),     // Apply to specific goal
    
    // --- Proof state management ---
    Have { name: String, ty: String, proof: Box<ProofTactic> },
    Suffices { goal: String, proof: Box<ProofTactic> },
    Obtain { vars: Vec<String>, from: String },
    Clear(Vec<String>),
    Rename(String, String),
    
    // --- Sorry (admit without proof — flagged as incomplete) ---
    Sorry,                              // Compiler warning: unproven obligation
}

#[derive(Debug, Clone)]
pub enum RewriteDirection { LeftToRight, RightToLeft }


// ============================================================================
// 6. DATABASE ENGINE INTERNALS
// ============================================================================
// Chronos can be used to build database engines. It provides built-in
// constructs for query planning, storage engines, and indexing.

#[derive(Debug, Clone)]
pub enum DatabasePrimitive {
    // --- Query planning ---
    LogicalPlan(LogicalPlanNode),
    PhysicalPlan(PhysicalPlanNode),
    CostModel { stats: TableStats },
    
    // --- Storage ---
    BTreeIndex { key_type: String, order: u32 },
    HashIndex { key_type: String, buckets: u32 },
    LSMTree { memtable_size: usize, levels: u32, compaction: CompactionStrategy },
    WAL { segment_size: usize, sync_mode: WALSync },
    BufferPool { size: usize, eviction: EvictionPolicy },
    
    // --- Transaction management ---
    MVCC { isolation: IsolationLevel },
    LockManager { mode: LockMode, deadlock_detection: DeadlockDetection },
    
    // --- Replication ---
    PrimaryReplica { sync_mode: ReplicationSync },
    LogicalReplication { publication: String },
}

#[derive(Debug, Clone)]
pub enum LogicalPlanNode {
    Scan { table: String, filter: Option<String> },
    Project { columns: Vec<String>, child: Box<LogicalPlanNode> },
    Filter { predicate: String, child: Box<LogicalPlanNode> },
    Join { left: Box<LogicalPlanNode>, right: Box<LogicalPlanNode>, condition: String, kind: String },
    Aggregate { group_by: Vec<String>, aggs: Vec<String>, child: Box<LogicalPlanNode> },
    Sort { keys: Vec<(String, bool)>, child: Box<LogicalPlanNode> },
    Limit { count: usize, child: Box<LogicalPlanNode> },
    Union(Vec<LogicalPlanNode>),
    Subquery { alias: String, child: Box<LogicalPlanNode> },
}

#[derive(Debug, Clone)]
pub enum PhysicalPlanNode {
    SeqScan(String), IndexScan(String, String),
    NestedLoopJoin, HashJoin, MergeJoin, IndexNestedLoop,
    HashAggregate, SortAggregate, StreamAggregate,
    ExternalSort { memory: usize }, TopNSort(usize),
    Materialize, HashBuild, HashProbe,
}

#[derive(Debug, Clone)]
pub struct TableStats { pub row_count: u64, pub avg_row_size: usize, pub distinct_values: HashMap<String, u64>, pub histograms: HashMap<String, Vec<(f64, f64)>>, pub null_fraction: HashMap<String, f64> }
#[derive(Debug, Clone)]
pub enum CompactionStrategy { Leveled, SizeTiered, FIFO, TimeWindow(Duration), Universal }
#[derive(Debug, Clone)]
pub enum WALSync { FSync, FDataSync, Buffered, None_ }
#[derive(Debug, Clone)]
pub enum EvictionPolicy { LRU, Clock, TwoQ, ARC, LFU, FIFO }
#[derive(Debug, Clone)]
pub enum IsolationLevel { ReadUncommitted, ReadCommitted, RepeatableRead, Serializable, SnapshotIsolation }
#[derive(Debug, Clone)]
pub enum LockMode { Shared, Exclusive, IntentShared, IntentExclusive, SIX, Update }
#[derive(Debug, Clone)]
pub enum DeadlockDetection { WaitFor, Timeout(Duration), WoundWait, WaitDie }
#[derive(Debug, Clone)]
pub enum ReplicationSync { Synchronous, Asynchronous, SemiSync(u32) }


// ============================================================================
// 7. MACRO SYSTEM & DSL EMBEDDING
// ============================================================================
// A truly universal language needs a powerful metaprogramming system.
// Chronos supports hygienic macros, procedural macros, and the ability
// to embed domain-specific languages directly into source code.

#[derive(Debug, Clone)]
pub enum MacroKind {
    /// Declarative macros (pattern matching on syntax, like Rust macro_rules!).
    Declarative {
        name: String,
        rules: Vec<MacroRule>,
    },
    /// Procedural macros (arbitrary Chronos code that transforms AST).
    Procedural {
        name: String,
        kind: ProcMacroKind,
        /// The macro implementation is a Chronos function that takes
        /// TokenStream → TokenStream.
        implementation: String,
    },
    /// DSL embedding — define a mini-language inside Chronos.
    /// The compiler calls a user-defined parser for the DSL block.
    DSL {
        name: String,
        /// Entry delimiter (e.g., `sql { ... }`, `regex { ... }`).
        open: String,
        close: String,
        parser: String,         // Chronos function that parses DSL text
        codegen: String,        // Chronos function that generates Chronos AST
    },
    /// Compile-time code execution (like Zig's comptime or Rust's const fn).
    CompileTime {
        body: String,
    },
}

#[derive(Debug, Clone)]
pub struct MacroRule {
    pub pattern: String,        // Syntax pattern with metavariables
    pub template: String,       // Expansion template
}

#[derive(Debug, Clone)]
pub enum ProcMacroKind {
    /// Attribute macro: `@my_macro fn foo() { ... }`
    Attribute,
    /// Derive macro: `derive(MyTrait)`
    Derive,
    /// Function-like macro: `my_macro!(...)`
    FunctionLike,
}

/// Built-in DSL blocks that the compiler understands natively.
#[derive(Debug, Clone)]
pub enum BuiltinDSL {
    /// Embedded SQL with compile-time type checking against a schema.
    SQL { query: String, schema: Option<String> },
    /// Regular expressions with compile-time validation and optimization.
    Regex { pattern: String },
    /// JSON/YAML literals with compile-time schema validation.
    JSON { value: String, schema: Option<String> },
    /// HTML/XML templates with compile-time checking.
    HTML { template: String },
    /// CSS with scoping.
    CSS { stylesheet: String },
    /// GraphQL queries with compile-time validation.
    GraphQL { query: String, schema: String },
    /// Shell commands with type-safe argument interpolation.
    Shell { command: String },
    /// Assembly blocks with register allocation hints.
    Asm { dialect: AsmDialect, code: String, clobbers: Vec<String> },
    /// GLSL/HLSL/WGSL shader embedding.
    Shader { language: ShaderLanguage, code: String },
    /// LaTeX math rendering.
    LaTeX { expression: String },
    /// Musical notation (for audio applications).
    MusicNotation { score: String },
}

#[derive(Debug, Clone)]
pub enum AsmDialect { Intel, ATT, ARM, RISCV, MIPS, WASM }
#[derive(Debug, Clone)]
pub enum ShaderLanguage { GLSL, HLSL, WGSL, MetalSL, SPIRV }


// ============================================================================
// 8. AUDIO / VIDEO / MULTIMEDIA
// ============================================================================

#[derive(Debug, Clone)]
pub enum MultimediaPrimitive {
    // --- Audio ---
    AudioBuffer { channels: u8, sample_rate: u32, format: AudioFormat },
    Oscillator { waveform: Waveform, frequency: f64 },
    Filter { kind: AudioFilter, cutoff: f64, resonance: f64 },
    Envelope { attack: f64, decay: f64, sustain: f64, release: f64 },
    FFT { size: usize, window: FFTWindow },
    Reverb { room_size: f64, damping: f64, wet: f64 },
    Compressor { threshold: f64, ratio: f64, attack: f64, release: f64 },
    Synthesizer { voices: u32, engine: SynthEngine },
    AudioCodec(AudioCodecKind),
    MIDI { channel: u8, note: u8, velocity: u8 },
    AudioStream { input: AudioDevice, output: AudioDevice, latency: Duration },
    SpeechToText { model: String, language: String },
    TextToSpeech { voice: String, language: String },
    
    // --- Video ---
    VideoFrame { width: u32, height: u32, format: PixelFormat },
    VideoCodec(VideoCodecKind),
    VideoFilter(VideoFilterKind),
    Camera { device: u32, resolution: (u32, u32), fps: u32 },
    ScreenCapture { region: Option<(u32, u32, u32, u32)>, fps: u32 },
    
    // --- Image processing ---
    ImageBuffer { width: u32, height: u32, format: PixelFormat },
    ImageFilter(ImageFilterKind),
    ObjectDetection { model: String },
    OCR { language: String },
    FaceDetection,
    ImageSegmentation { model: String },
    StyleTransfer { style_model: String },
    SuperResolution { scale: u32 },
    
    // --- 3D Rendering ---
    Scene3D,
    Mesh3D { vertices: usize, format: VertexFormat },
    Material { shader: String, textures: Vec<String> },
    Light { kind: LightKind },
    Camera3D { fov: f64, near: f64, far: f64 },
    RayTracer { samples: u32, bounces: u32 },
    PathTracer { samples: u32 },
    Rasterizer { msaa: u32 },
}

#[derive(Debug, Clone)]
pub enum AudioFormat { I16, I24, I32, F32, F64 }
#[derive(Debug, Clone)]
pub enum Waveform { Sine, Square, Sawtooth, Triangle, Noise, Custom(String) }
#[derive(Debug, Clone)]
pub enum AudioFilter { LowPass, HighPass, BandPass, Notch, AllPass, Peaking, LowShelf, HighShelf }
#[derive(Debug, Clone)]
pub enum FFTWindow { Hann, Hamming, Blackman, FlatTop, Kaiser(f64), Rectangular }
#[derive(Debug, Clone)]
pub enum SynthEngine { Subtractive, Additive, FM, Wavetable, Granular, Physical }
#[derive(Debug, Clone)]
pub enum AudioCodecKind { PCM, MP3, AAC, FLAC, Opus, Vorbis, WAV, ALAC }
#[derive(Debug, Clone)]
pub struct AudioDevice { pub name: String, pub channels: u8, pub sample_rate: u32 }
#[derive(Debug, Clone)]
pub enum PixelFormat { RGB8, RGBA8, BGR8, BGRA8, YUV420, YUV422, NV12, R16F, RGBA16F, RGBA32F }
#[derive(Debug, Clone)]
pub enum VideoCodecKind { H264, H265, VP9, AV1, ProRes, DNxHR }
#[derive(Debug, Clone)]
pub enum VideoFilterKind { Scale(u32, u32), Crop(u32, u32, u32, u32), Rotate(f64), ColorCorrect, Stabilize, MotionBlur, ChromaKey(String), Deinterlace, Denoise }
#[derive(Debug, Clone)]
pub enum ImageFilterKind { Blur(f64), Sharpen, EdgeDetect, Threshold(f64), Erosion(u32), Dilation(u32), Histogram, ColorSpace(String), Resize(u32, u32, String), Rotate(f64), FlipH, FlipV }
#[derive(Debug, Clone)]
pub enum VertexFormat { Position, PositionNormal, PositionNormalUV, PositionNormalUVTangent, Custom(Vec<(String, String)>) }
#[derive(Debug, Clone)]
pub enum LightKind { Directional, Point { radius: f64 }, Spot { angle: f64 }, Area { width: f64, height: f64 }, Environment(String) }


// ============================================================================
// 9. ROBOTICS & CONTROL SYSTEMS
// ============================================================================

#[derive(Debug, Clone)]
pub enum RoboticsPrimitive {
    // --- Kinematics ---
    ForwardKinematics { dh_params: Vec<DHParam> },
    InverseKinematics { target: [f64; 6], method: IKMethod },
    JacobianMatrix,
    WorkspaceAnalysis,
    
    // --- Control ---
    PIDController { kp: f64, ki: f64, kd: f64, limits: Option<(f64, f64)> },
    StateSpaceController { A: Vec<Vec<f64>>, B: Vec<Vec<f64>>, C: Vec<Vec<f64>>, D: Vec<Vec<f64>> },
    LQR { Q: Vec<Vec<f64>>, R: Vec<Vec<f64>> },
    MPC { horizon: usize, constraints: Vec<String> }, // Model Predictive Control
    AdaptiveControl { model: String, learning_rate: f64 },
    FuzzyController { rules: Vec<FuzzyRule> },
    SlidingModeControl { surface: String, gain: f64 },
    
    // --- Planning ---
    RRT { max_iterations: usize },
    RRTStar { max_iterations: usize },
    PRM { samples: usize },
    AStar3D { resolution: f64 },
    DWA { max_vel: f64, max_omega: f64 }, // Dynamic Window Approach
    TrajectoryOptimization { method: String },
    
    // --- Perception ---
    SLAM { method: SLAMMethod },
    PointCloudProcessing,
    LidarProcessing { format: String },
    VisualOdometry,
    SensorFusion { sensors: Vec<SensorType>, method: FusionMethod },
    KalmanFilter { state_dim: usize, measurement_dim: usize },
    ExtendedKalmanFilter { state_dim: usize },
    ParticleFilter { particles: usize },
    UnscentedKalmanFilter { state_dim: usize },
    
    // --- Communication ---
    ROS2 { node_name: String, topics: Vec<(String, String)>, services: Vec<(String, String)> },
    MAVLINK { system_id: u8, component_id: u8 },
    CAN { bus: String, baud: u32 },
}

#[derive(Debug, Clone)]
pub struct DHParam { pub a: f64, pub alpha: f64, pub d: f64, pub theta: f64 }
#[derive(Debug, Clone)]
pub enum IKMethod { Analytical, Jacobian, CCD, FABRIK, LMA }
#[derive(Debug, Clone)]
pub struct FuzzyRule { pub antecedents: Vec<(String, String)>, pub consequent: (String, String) }
#[derive(Debug, Clone)]
pub enum SLAMMethod { EKF_SLAM, FastSLAM, GraphSLAM, ORB_SLAM, LIO_SAM, RTAB_MAP }
#[derive(Debug, Clone)]
pub enum SensorType { IMU, GPS, Lidar, Camera, Radar, Sonar, Encoder, ForceTorque, Barometer, Magnetometer }
#[derive(Debug, Clone)]
pub enum FusionMethod { KalmanFusion, ComplementaryFilter, MadgwickFilter, MahonyFilter, ParticleFusion }


// ============================================================================
// 10. QUANTUM COMPUTING
// ============================================================================

#[derive(Debug, Clone)]
pub enum QuantumPrimitive {
    // --- Qubit management ---
    Qubit,
    QubitRegister(usize),
    ClassicalRegister(usize),
    
    // --- Gates ---
    Gate(QuantumGate),
    CustomGate { name: String, matrix: Vec<Vec<(f64, f64)>> }, // Complex matrix
    ControlledGate { control: usize, target: usize, gate: Box<QuantumGate> },
    
    // --- Algorithms ---
    QFT(usize),                         // Quantum Fourier Transform
    Grover { oracle: String, iterations: Option<usize> },
    Shor { number: u64 },
    VQE { ansatz: String, optimizer: String },
    QAOA { layers: usize },
    QuantumWalk { graph: String },
    QuantumTeleportation,
    SuperdenseCoding,
    QuantumErrorCorrection { code: QECCode },
    
    // --- Measurement ---
    Measure { qubit: usize, basis: MeasurementBasis },
    MeasureAll,
    
    // --- Simulation ---
    StateVector { qubits: usize },
    DensityMatrix { qubits: usize },
    StabilizerSimulation { qubits: usize },
    TensorNetwork { qubits: usize },
    
    // --- Hardware targets ---
    IBMQuantum { backend: String },
    GoogleSycamore,
    IonQ,
    Rigetti,
    DWave { topology: String },
    Simulator { shots: usize },
}

#[derive(Debug, Clone)]
pub enum QuantumGate {
    // Single-qubit gates
    H, X, Y, Z, S, T, SDag, TDag,
    RX(f64), RY(f64), RZ(f64),
    Phase(f64), U3(f64, f64, f64),
    // Two-qubit gates  
    CNOT, CZ, SWAP, ISWAP, SqrtSWAP,
    // Three-qubit gates
    Toffoli, Fredkin,
}

#[derive(Debug, Clone)]
pub enum QECCode { Steane, Shor9, SurfaceCode, ColorCode, ToricCode, RotatedSurface }
#[derive(Debug, Clone)]
pub enum MeasurementBasis { Computational, Hadamard, Custom(Vec<Vec<(f64, f64)>>) }


// ============================================================================
// 11. GUI / UI FRAMEWORK
// ============================================================================

#[derive(Debug, Clone)]
pub enum PlotKind {
    Line,
    Bar,
    Scatter,
    Histogram,
    Pie,
}

#[derive(Debug, Clone)]
pub enum UIPrimitive {
    // --- Layout ---
    Column(Vec<UIPrimitive>),
    Row(Vec<UIPrimitive>),
    Stack(Vec<UIPrimitive>),
    Grid { rows: u32, cols: u32, children: Vec<(u32, u32, Box<UIPrimitive>)> },
    ScrollView(Box<UIPrimitive>),
    Padding { all: f64, child: Box<UIPrimitive> },
    Center(Box<UIPrimitive>),
    Expanded(Box<UIPrimitive>),
    SizedBox { width: f64, height: f64, child: Option<Box<UIPrimitive>> },
    
    // --- Widgets ---
    Text { content: String, style: TextStyle },
    Button { label: String, on_click: String },
    TextField { placeholder: String, on_change: String },
    Checkbox { checked: bool, on_toggle: String },
    RadioGroup { options: Vec<String>, selected: usize, on_change: String },
    Slider { min: f64, max: f64, value: f64, on_change: String },
    Toggle { on: bool, on_toggle: String },
    Dropdown { options: Vec<String>, selected: usize, on_change: String },
    DatePicker { value: String, on_change: String },
    ColorPicker { value: String, on_change: String },
    Image { source: String, width: f64, height: f64 },
    Icon { name: String, size: f64 },
    ProgressBar { value: f64 },
    Spinner,
    
    // --- Navigation ---
    TabBar { tabs: Vec<(String, Box<UIPrimitive>)>, selected: usize },
    NavigationBar { title: String, leading: Option<Box<UIPrimitive>>, trailing: Option<Box<UIPrimitive>> },
    Drawer { content: Box<UIPrimitive> },
    BottomSheet { content: Box<UIPrimitive> },
    Dialog { title: String, content: Box<UIPrimitive>, actions: Vec<(String, String)> },
    
    // --- Data display ---
    Table { columns: Vec<TableColumn>, rows: Vec<Vec<String>>, sortable: bool, pagination: Option<usize> },
    ListView { items: Vec<Box<UIPrimitive>>, virtual_scroll: bool },
    TreeView { nodes: Vec<TreeNode> },
    Chart(Box<crate::PlotKind>),         // Integrates with plotting system
    Map { center: (f64, f64), zoom: u32 },
    
    // --- Rich content ---
    Markdown(String),
    CodeBlock { language: String, code: String, line_numbers: bool },
    LaTeX(String),
    WebView { url: String },
    
    // --- Composite / state ---
    Stateful { state: String, build: String },
    Animated { child: Box<UIPrimitive>, animation: Animation },
    Conditional { condition: String, then_: Box<UIPrimitive>, else_: Option<Box<UIPrimitive>> },
    ForEach { items: String, builder: String },
    
    // --- Accessibility ---
    Semantics { label: String, role: String, child: Box<UIPrimitive> },
}

#[derive(Debug, Clone)]
pub struct TextStyle { pub font_size: f64, pub font_weight: String, pub color: String, pub font_family: Option<String> }
#[derive(Debug, Clone)]
pub struct TableColumn { pub header: String, pub width: Option<f64>, pub sortable: bool }
#[derive(Debug, Clone)]
pub struct TreeNode { pub label: String, pub children: Vec<TreeNode>, pub expanded: bool }
#[derive(Debug, Clone)]
pub struct Animation { pub duration_ms: u32, pub curve: String, pub property: String, pub from: f64, pub to: f64 }


// ============================================================================
// 12. GAME ENGINE PRIMITIVES
// ============================================================================

#[derive(Debug, Clone)]
pub enum GamePrimitive {
    // --- Entity Component System ---
    Entity(u64),
    Component { name: String, fields: Vec<(String, String)> },
    System { name: String, query: Vec<ComponentQuery>, run: String },
    World { entities: usize, archetypes: Vec<Vec<String>> },
    
    // --- Physics ---
    RigidBody2D { mass: f64, inertia: f64, restitution: f64 },
    RigidBody3D { mass: f64, inertia_tensor: [f64; 9], restitution: f64 },
    Collider2D(Collider2DShape),
    Collider3D(Collider3DShape),
    Joint(JointKind),
    RayCast { origin: [f64; 3], direction: [f64; 3], max_distance: f64 },
    NavMesh { bake_settings: NavMeshSettings },
    
    // --- Rendering ---
    Sprite { texture: String, region: Option<(f64, f64, f64, f64)> },
    TileMap { tile_size: u32, layers: Vec<Vec<Vec<u32>>> },
    ParticleSystem { max_particles: u32, emitter: ParticleEmitter },
    Shader { vertex: String, fragment: String },
    RenderPipeline { passes: Vec<RenderPass> },
    
    // --- Audio ---
    AudioSource { clip: String, spatial: bool, loop_: bool },
    AudioListener,
    
    // --- Input ---
    InputAction { name: String, bindings: Vec<InputBinding> },
    
    // --- Animation ---
    AnimationClip { keyframes: Vec<Keyframe>, loop_mode: LoopMode },
    StateMachine { states: Vec<AnimState>, transitions: Vec<AnimTransition> },
    SkeletalMesh { bones: u32, mesh: String },
    BlendTree(Vec<(String, f64)>),
    IKChain { bones: Vec<String>, target: String },
    
    // --- Networking ---
    NetworkedEntity { sync_rate: f64, interpolation: InterpMethod },
    RPCCall { name: String, reliability: Reliability },
    Lobby { max_players: u32 },
}

#[derive(Debug, Clone)]
pub struct ComponentQuery { pub component: String, pub access: QueryAccess, pub filter: Option<String> }
#[derive(Debug, Clone)]
pub enum QueryAccess { Read, Write, Optional, With, Without }
#[derive(Debug, Clone)]
pub enum Collider2DShape { Circle(f64), Rectangle(f64, f64), Capsule(f64, f64), Polygon(Vec<(f64, f64)>) }
#[derive(Debug, Clone)]
pub enum Collider3DShape { Sphere(f64), Box3(f64, f64, f64), Capsule3(f64, f64), ConvexHull(Vec<[f64; 3]>), TriMesh(String) }
#[derive(Debug, Clone)]
pub enum JointKind { Fixed, Hinge { axis: [f64; 3], limits: Option<(f64, f64)> }, Ball, Slider { axis: [f64; 3] }, Spring { stiffness: f64, damping: f64 } }
#[derive(Debug, Clone)]
pub struct NavMeshSettings { pub agent_radius: f64, pub agent_height: f64, pub max_slope: f64, pub step_height: f64 }
#[derive(Debug, Clone)]
pub struct ParticleEmitter { pub rate: f64, pub lifetime: f64, pub speed: f64, pub shape: String }
#[derive(Debug, Clone)]
pub enum RenderPass { Shadow, GBuffer, Lighting, PostProcess(String), UI }
#[derive(Debug, Clone)]
pub struct InputBinding { pub device: String, pub key: String }
#[derive(Debug, Clone)]
pub struct Keyframe { pub time: f64, pub value: f64, pub interpolation: InterpMethod }
#[derive(Debug, Clone)]
pub enum LoopMode { Once, Loop, PingPong, Clamp }
#[derive(Debug, Clone)]
pub struct AnimState { pub name: String, pub clip: String, pub speed: f64 }
#[derive(Debug, Clone)]
pub struct AnimTransition { pub from: String, pub to: String, pub condition: String, pub duration: f64 }
#[derive(Debug, Clone)]
pub enum InterpMethod { None_, Linear, Hermite, CatmullRom, Bezier }
#[derive(Debug, Clone)]
pub enum Reliability { Unreliable, Reliable, ReliableOrdered }


// ============================================================================
// 13. BIOINFORMATICS
// ============================================================================

#[derive(Debug, Clone)]
pub enum BioinformaticsPrimitive {
    // --- Sequence types ---
    DNASequence(String),
    RNASequence(String),
    ProteinSequence(String),
    GenomicRegion { chromosome: String, start: u64, end: u64, strand: Strand },
    
    // --- Alignment ---
    NeedlemanWunsch { gap_penalty: f64, substitution_matrix: String },
    SmithWaterman { gap_penalty: f64, substitution_matrix: String },
    MultipleAlignment { method: MSAMethod },
    BLAST { database: String, evalue: f64 },
    HMM { model: String },              // Hidden Markov Model for sequence profiles
    
    // --- Phylogenetics ---
    PhylogeneticTree { method: PhyloMethod },
    EvolutionaryDistance { model: EvolutionModel },
    
    // --- Structural ---
    PDBStructure { pdb_id: String },
    ProteinFolding { method: FoldingMethod },
    MolecularDocking { receptor: String, ligand: String },
    
    // --- Genomics ---
    VariantCalling { method: String },
    GeneExpression { method: ExpressionMethod },
    GWAS { phenotype: String, covariates: Vec<String> },
    Methylation { method: String },
    
    // --- File formats ---
    FASTA, FASTQ, SAM, BAM, VCF, GFF, BED, PDB, MMCIF,
}

#[derive(Debug, Clone)]
pub enum Strand { Plus, Minus }
#[derive(Debug, Clone)]
pub enum MSAMethod { ClustalW, MUSCLE, MAFFT, TCoffee, Kalign }
#[derive(Debug, Clone)]
pub enum PhyloMethod { NeighborJoining, UPGMA, MaximumLikelihood, Bayesian, MaxParsimony }
#[derive(Debug, Clone)]
pub enum EvolutionModel { JukesCantor, Kimura2P, HKY, GTR, WAG, LG }
#[derive(Debug, Clone)]
pub enum FoldingMethod { AlphaFold, RoseTTAFold, Rosetta, ESMFold }
#[derive(Debug, Clone)]
pub enum ExpressionMethod { RNASeq, Microarray, ScRNASeq, SpatialTranscriptomics }


// ============================================================================
// 14. FINANCIAL ENGINEERING
// ============================================================================

#[derive(Debug, Clone)]
pub enum FinancialPrimitive {
    // --- Pricing models ---
    BlackScholes { spot: f64, strike: f64, rate: f64, vol: f64, time: f64, option_type: OptionType },
    Binomial { steps: u32, spot: f64, strike: f64, rate: f64, vol: f64, time: f64 },
    MonteCarloPricing { paths: u64, steps: u32, model: StochasticProcess },
    HestonModel { v0: f64, theta: f64, kappa: f64, xi: f64, rho: f64 },
    SABR { alpha: f64, beta: f64, rho: f64, nu: f64 },
    LocalVol { surface: Vec<Vec<f64>> },
    
    // --- Risk ---
    VaR { confidence: f64, horizon: Duration, method: VaRMethod },
    CVaR { confidence: f64 },
    Greeks { delta: bool, gamma: bool, theta: bool, vega: bool, rho: bool },
    StressTest { scenarios: Vec<(String, HashMap<String, f64>)> },
    CreditRisk { model: CreditModel },
    
    // --- Fixed Income ---
    BondPricing { face: f64, coupon: f64, maturity: f64, yield_curve: Vec<(f64, f64)> },
    YieldCurve { method: YieldCurveMethod },
    InterestRateSwap { notional: f64, fixed_rate: f64, float_index: String },
    CDS { spread: f64, recovery: f64, notional: f64 },
    
    // --- Portfolio ---
    PortfolioOptimization { method: PortfolioMethod },
    BackTest { strategy: String, data: String, start: String, end: String },
    RiskParity { assets: Vec<String> },
    FactorModel { factors: Vec<String> },
    
    // --- Order management ---
    OrderBook { levels: u32 },
    LimitOrder { side: Side, price: f64, quantity: f64 },
    MarketOrder { side: Side, quantity: f64 },
    MatchingEngine { algorithm: MatchingAlgo },
    VWAP { target_quantity: f64, time_horizon: Duration },
    TWAP { target_quantity: f64, slices: u32 },
}

#[derive(Debug, Clone)]
pub enum OptionType { Call, Put, BinaryCall, BinaryPut, Asian, Barrier(BarrierType), Lookback, American, Bermudan(Vec<f64>) }
#[derive(Debug, Clone)]
pub enum BarrierType { UpAndIn(f64), UpAndOut(f64), DownAndIn(f64), DownAndOut(f64) }
#[derive(Debug, Clone)]
pub enum StochasticProcess { GBM { mu: f64, sigma: f64 }, Heston, JumpDiffusion { lambda: f64, mu_j: f64, sigma_j: f64 }, OrnsteinUhlenbeck { theta: f64, mu: f64, sigma: f64 } }
#[derive(Debug, Clone)]
pub enum VaRMethod { Historical, Parametric, MonteCarlo(u64) }
#[derive(Debug, Clone)]
pub enum CreditModel { Merton, KMV, CreditMetrics, CreditRiskPlus }
#[derive(Debug, Clone)]
pub enum YieldCurveMethod { Bootstrap, NelsonSiegel, Svensson, CubicSpline }
#[derive(Debug, Clone)]
pub enum PortfolioMethod { MeanVariance, BlackLitterman, MinVariance, MaxSharpe, RiskParity, HRP }
#[derive(Debug, Clone)]
pub enum Side { Buy, Sell }
#[derive(Debug, Clone)]
pub enum MatchingAlgo { PriceTime, ProRata, TimeWeighted }


// ============================================================================
// 15. GEOSPATIAL / GIS
// ============================================================================

#[derive(Debug, Clone)]
pub enum GeospatialPrimitive {
    // --- Geometry types (OGC Simple Features) ---
    Point { lat: f64, lon: f64, alt: Option<f64>, srid: u32 },
    LineString(Vec<(f64, f64)>),
    Polygon { exterior: Vec<(f64, f64)>, holes: Vec<Vec<(f64, f64)>> },
    MultiPoint(Vec<(f64, f64)>),
    MultiLineString(Vec<Vec<(f64, f64)>>),
    MultiPolygon(Vec<Vec<Vec<(f64, f64)>>>),
    GeometryCollection(Vec<GeospatialPrimitive>),
    
    // --- Coordinate Reference Systems ---
    CRS { epsg: u32 },
    Transform { from_epsg: u32, to_epsg: u32 },
    
    // --- Spatial operations ---
    Buffer { geometry: Box<GeospatialPrimitive>, distance: f64 },
    Intersection(Box<GeospatialPrimitive>, Box<GeospatialPrimitive>),
    Union(Vec<GeospatialPrimitive>),
    Difference(Box<GeospatialPrimitive>, Box<GeospatialPrimitive>),
    Centroid(Box<GeospatialPrimitive>),
    ConvexHull(Box<GeospatialPrimitive>),
    Voronoi(Vec<(f64, f64)>),
    Delaunay(Vec<(f64, f64)>),
    
    // --- Spatial indexing ---
    RTreeIndex(Vec<GeospatialPrimitive>),
    QuadTreeIndex(Vec<GeospatialPrimitive>),
    H3Index { resolution: u8 },          // Uber H3 hexagonal grid
    S2Index { level: u8 },               // Google S2 geometry
    GeoHash { precision: u8 },
    
    // --- Raster ---
    Raster { width: u32, height: u32, bands: u8, crs: u32 },
    DEM { resolution: f64 },             // Digital Elevation Model
    Hillshade { azimuth: f64, altitude: f64 },
    SlopeAspect,
    Viewshed { observer: (f64, f64, f64), radius: f64 },
    
    // --- Routing ---
    Routing { algorithm: RoutingAlgo, network: String },
    Isochrone { center: (f64, f64), time_limit: Duration, mode: TravelMode },
    Geocode { address: String },
    ReverseGeocode { lat: f64, lon: f64 },
    
    // --- File formats ---
    Shapefile(String), GeoJSON(String), GeoPackage(String),
    GeoTIFF(String), KML(String), WKT(String), WKB(Vec<u8>),
    GeoParquet(String), FlatGeobuf(String),
    
    // --- Services ---
    WMS { url: String, layers: Vec<String> },
    WFS { url: String, typename: String },
    WMTS { url: String },
    TileServer { url_template: String, zoom_range: (u8, u8) },
}

#[derive(Debug, Clone)]
pub enum RoutingAlgo { Dijkstra, AStar, ContractionHierarchies, TransitNodeRouting, OSRM }
#[derive(Debug, Clone)]
pub enum TravelMode { Driving, Walking, Cycling, Transit }


// ============================================================================
// 16. OBSERVABILITY & TESTING
// ============================================================================

#[derive(Debug, Clone)]
pub enum ObservabilityPrimitive {
    // --- Tracing ---
    Span { name: String, parent: Option<String>, attributes: HashMap<String, String> },
    TraceExporter(TraceExporterKind),
    
    // --- Metrics ---
    Counter { name: String, labels: Vec<String> },
    Gauge { name: String, labels: Vec<String> },
    Histogram { name: String, buckets: Vec<f64>, labels: Vec<String> },
    Summary { name: String, quantiles: Vec<f64> },
    MetricsExporter(MetricsExporterKind),
    
    // --- Logging ---
    StructuredLog { level: LogLevel, fields: Vec<(String, String)> },
    LogExporter(LogExporterKind),
    
    // --- Profiling ---
    CPUProfile { duration: Duration },
    MemoryProfile,
    AllocationProfile,
    BlockProfile,
    MutexProfile,
    FlameGraph,
}

#[derive(Debug, Clone)]
pub enum TraceExporterKind { Jaeger, Zipkin, OTLP, DataDog, NewRelic }
#[derive(Debug, Clone)]
pub enum MetricsExporterKind { Prometheus, StatsD, OTLP, DataDog, CloudWatch, InfluxDB }
#[derive(Debug, Clone)]
pub enum LogExporterKind { Stdout, File(String), Syslog, OTLP, ElasticSearch, Loki, CloudWatch }
#[derive(Debug, Clone)]
pub enum LogLevel { Trace, Debug, Info, Warn, Error, Fatal }

/// Built-in testing framework.
#[derive(Debug, Clone)]
pub enum TestPrimitive {
    UnitTest { name: String, body: String },
    PropertyTest { name: String, generators: Vec<Generator>, property: String, iterations: u64 },
    FuzzTest { name: String, corpus: String, target: String },
    MutationTest { source: String, test_suite: String },
    BenchmarkTest { name: String, body: String, iterations: u64, warmup: u64 },
    IntegrationTest { name: String, setup: Option<String>, body: String, teardown: Option<String> },
    SnapshotTest { name: String, body: String, snapshot_path: String },
    ApprovalTest { name: String, body: String, approved_path: String },
    ContractTest { provider: String, consumer: String, pact: String },
    ChaosTest { target: String, faults: Vec<FaultInjection> },
    PerformanceTest { name: String, slo: SLO },
}

#[derive(Debug, Clone)]
pub enum Generator {
    Bool, Int(i64, i64), Float(f64, f64), String_(usize, usize),
    Vec_(Box<Generator>, usize, usize), OneOf(Vec<String>),
    Regex(String), Custom(String),
}

#[derive(Debug, Clone)]
pub enum FaultInjection { LatencyInject(Duration), ErrorInject(String), PartitionInject(Vec<String>), ResourceExhaustion(String), ClockSkew(Duration), ProcessKill(String) }
#[derive(Debug, Clone)]
pub struct SLO { pub p50: Duration, pub p95: Duration, pub p99: Duration, pub error_rate: f64 }


// ============================================================================
// 17. INTEROPERABILITY / FFI
// ============================================================================

#[derive(Debug, Clone)]
pub enum FFIBridge {
    /// Call C functions directly.
    C { header: String, link: Vec<String> },
    /// Call C++ (with name mangling support).
    CPP { header: String, namespace: Option<String>, link: Vec<String> },
    /// Python interop — call Python or be called from Python.
    Python { module: String, class: Option<String> },
    /// JVM interop — call Java/Kotlin/Scala code.
    JVM { classpath: String, class: String },
    /// .NET interop — call C#/F# code.
    DotNet { assembly: String, namespace: String },
    /// JavaScript/TypeScript interop via WASM.
    JavaScript { module: String },
    /// Swift interop (for Apple platforms).
    Swift { framework: String },
    /// Go interop (cgo-style).
    Go { module: String },
    /// Rust interop (share types directly).
    Rust { crate_: String },
    /// WASM component model.
    WASM { wit: String },
    /// gRPC interop (language-agnostic via protobuf).
    GRPC { proto: String },
}

/// Hot code reloading — replace code in a running system.
/// Critical for game development, live performances, and long-running servers.
#[derive(Debug, Clone)]
pub struct HotReload {
    pub mode: HotReloadMode,
    pub watch_paths: Vec<String>,
    pub on_reload: Option<String>,       // Callback function
    pub preserve_state: bool,
    pub rollback_on_error: bool,
}

#[derive(Debug, Clone)]
pub enum HotReloadMode {
    /// Replace individual functions.
    FunctionLevel,
    /// Replace entire modules.
    ModuleLevel,
    /// Replace the entire program (graceful restart).
    ProcessLevel,
    /// Erlang-style code replacement with state migration.
    ErlangStyle { state_migration: String },
}


// ============================================================================
// 18. INTERNATIONALIZATION & LOCALIZATION
// ============================================================================

#[derive(Debug, Clone)]
pub struct I18nConfig {
    pub default_locale: String,
    pub supported_locales: Vec<String>,
    pub fallback_chain: Vec<String>,
    pub message_format: MessageFormat,
    pub pluralization: PluralizationRules,
    pub number_format: HashMap<String, NumberFormat>,
    pub date_format: HashMap<String, DateFormat>,
    pub currency_format: HashMap<String, CurrencyFormat>,
    pub direction: HashMap<String, TextDirection>,
    pub collation: HashMap<String, CollationRules>,
}

#[derive(Debug, Clone)]
pub enum MessageFormat { ICU, Fluent, Gettext, Custom(String) }
#[derive(Debug, Clone)]
pub enum PluralizationRules { CLDR, Custom(Vec<(String, String)>) }
#[derive(Debug, Clone)]
pub struct NumberFormat { pub decimal_separator: char, pub thousands_separator: char, pub grouping: Vec<u8> }
#[derive(Debug, Clone)]
pub struct DateFormat { pub pattern: String, pub calendar: CalendarSystem }
#[derive(Debug, Clone)]
pub struct CurrencyFormat { pub symbol: String, pub position: SymbolPosition, pub decimal_digits: u8 }
#[derive(Debug, Clone)]
pub enum TextDirection { LTR, RTL, Auto }
#[derive(Debug, Clone)]
pub enum CalendarSystem { Gregorian, Islamic, Hebrew, Chinese, Japanese, Buddhist, Persian }
#[derive(Debug, Clone)]
pub enum SymbolPosition { Before, After }
#[derive(Debug, Clone)]
pub struct CollationRules { pub locale: String, pub sensitivity: String, pub case_first: String }


// ============================================================================
// 19. DOCUMENTATION & LITERATE PROGRAMMING
// ============================================================================

#[derive(Debug, Clone)]
pub struct DocSystem {
    /// Documentation can be written inline and compiled into multiple formats.
    pub format: DocFormat,
    /// Cross-reference resolution (links between documentation and code).
    pub cross_refs: bool,
    /// Automatic API documentation generation.
    pub api_docs: bool,
    /// Runnable examples in documentation (tested during compilation).
    pub doctests: bool,
    /// Generate diagrams from code structure.
    pub auto_diagrams: bool,
    /// Changelog generation from version control annotations (Feature 3).
    pub changelog: bool,
    /// Architecture Decision Records.
    pub adrs: bool,
}

#[derive(Debug, Clone)]
pub enum DocFormat {
    Markdown, AsciiDoc, ReStructuredText, HTML, PDF, EPUB, ManPage,
    /// Literate programming: the documentation IS the source file.
    /// Code blocks within prose are extracted and compiled.
    LiterateProgramming { tangle: bool, weave: bool },
    /// Jupyter-compatible notebook format.
    Notebook,
}
