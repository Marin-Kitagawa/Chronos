// ============================================================================
// CHRONOS PARSER v2 — DOMAIN-SPECIFIC PARSING EXTENSIONS
// ============================================================================
// This file contains the NEW AST nodes and the NEW parsing methods that must
// be added to the parser from chronos-parser. Each domain gets:
//   1. New AST nodes (added to the Item enum)
//   2. A parsing method (added to the Parser impl)
//   3. Integration into the parse_item() dispatch
//
// INTEGRATION INSTRUCTIONS:
// 1. Add each DomainItem variant to the Item enum.
// 2. Add each parse_* method to the Parser impl block.
// 3. Add match arms in parse_item() for the new keyword tokens.
// ============================================================================

// use std::collections::HashMap; // provided by parent scope in compiler-core

// ============================================================================
// NEW AST NODES — These are the domain-specific declarations the parser
// must be able to construct. Each one mirrors the data structures defined
// in the extended features artifacts.
// ============================================================================

/// New Item variants to add to the existing Item enum.
/// These represent top-level declarations for each domain.
#[derive(Debug, Clone)]
pub enum DomainItem {
    // Domain 1: Networking
    ProtocolDecl(ProtocolDeclAST),
    EndpointDecl(EndpointDeclAST),
    
    // Domain 2: OS
    DriverDecl(DriverDeclAST),
    InterruptHandler(InterruptHandlerAST),
    RegisterDecl(RegisterDeclAST),
    
    // Domain 3: Distributed Systems
    ConsensusDecl(ConsensusDeclAST),
    CrdtDecl(CrdtDeclAST),
    SagaDecl(SagaDeclAST),
    
    // Domain 4: Blockchain
    ContractDecl(ContractDeclAST),
    
    // Domain 5: Formal Verification
    TheoremDecl(TheoremDeclAST),
    LemmaDecl(TheoremDeclAST),     // Same structure as theorem
    AxiomDecl(AxiomDeclAST),
    
    // Domain 6: Database
    TableDecl(TableDeclAST),
    IndexDecl(IndexDeclAST),
    
    // Domain 7: Macros
    MacroDecl(MacroDeclAST),
    ComptimeBlock(ComptimeBlockAST),
    EmbedBlock(EmbedBlockAST),
    
    // Domain 8: Multimedia
    AudioDecl(MultimediaDeclAST),
    VideoDecl(MultimediaDeclAST),
    SceneDecl(SceneDeclAST),
    ShaderDecl(ShaderDeclAST),
    
    // Domain 9: Robotics
    RobotDecl(RobotDeclAST),
    ControllerDecl(ControllerDeclAST),
    
    // Domain 10: Quantum
    CircuitDecl(CircuitDeclAST),
    
    // Domain 11: GUI
    WidgetDecl(WidgetDeclAST),
    ComponentDecl(ComponentDeclAST),
    
    // Domain 12: Game
    EntityDecl(EntityDeclAST),
    SystemDecl(SystemDeclAST),
    WorldDecl(WorldDeclAST),
    
    // Domain 13: Bio
    GenomeDecl(GenomeDeclAST),
    
    // Domain 14: Finance
    PortfolioDecl(PortfolioDeclAST),
    BacktestDecl(BacktestDeclAST),
    
    // Domain 15: Geospatial
    SpatialDecl(SpatialDeclAST),
    
    // Domain 16: Testing
    TestDecl(TestDeclAST),
    BenchDecl(BenchDeclAST),
    FuzzDecl(FuzzDeclAST),
    PropertyTestDecl(PropertyTestDeclAST),
    
    // Domain 17: FFI
    ExternDecl(ExternDeclAST),
    ForeignBlock(ForeignBlockAST),
    
    // Domain 18: Simulation
    SimulationDecl(SimulationDeclAST),
    FemDecl(FemDeclAST),
    
    // Domain 19: Pipeline
    PipelineDecl(PipelineDeclAST),
    
    // Domain 20: Security
    AuditBlock(AuditBlockAST),
}

// ============================================================================
// AST NODE DEFINITIONS — One struct per domain declaration
// ============================================================================
// I'm defining each AST node to capture the syntactic structure that the
// parser extracts. The parser does NOT do semantic analysis — it just
// builds the tree. Type checking and validation happen later.

// --- DOMAIN 1: Protocol ---
// Syntax: protocol Name { field x: u16; state_machine { ... } }
#[derive(Debug, Clone)]
pub struct ProtocolDeclAST {
    pub name: String,
    pub transport: Option<String>,       // tcp, udp, quic, etc.
    pub endianness: Option<String>,      // big, little
    pub fields: Vec<ProtocolFieldAST>,
    pub state_machine: Option<StateMachineAST>,
    pub constraints: Vec<(String, Vec<String>)>, // (kind, args)
}

#[derive(Debug, Clone)]
pub struct ProtocolFieldAST {
    pub name: String,
    pub wire_type: String,               // "u16", "u24", "[u8; length]", etc.
    pub validation: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StateMachineAST {
    pub states: Vec<String>,
    pub initial: String,
    pub transitions: Vec<TransitionAST>,
}

#[derive(Debug, Clone)]
pub struct TransitionAST {
    pub from: String,
    pub to: String,
    pub trigger: String,
    pub guard: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct EndpointDeclAST {
    pub path: String,
    pub methods: Vec<(String, FunctionDecl)>,   // (GET/POST/..., handler)
    pub middleware: Vec<String>,
}

// --- DOMAIN 2: OS ---
#[derive(Debug, Clone)]
pub struct DriverDeclAST {
    pub name: String,
    pub bus: String,                     // pci, usb, i2c, spi, etc.
    pub device_id: Vec<(String, String)>,// (vendor, device) pairs
    pub functions: Vec<FunctionDecl>,
    pub registers: Vec<RegisterDeclAST>,
}

#[derive(Debug, Clone)]
pub struct InterruptHandlerAST {
    pub vector: Expression,
    pub body: Vec<Statement>,
    pub priority: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct RegisterDeclAST {
    pub name: String,
    pub address: Expression,
    pub access: String,                  // "rw", "ro", "wo"
    pub fields: Vec<RegisterFieldAST>,
}

#[derive(Debug, Clone)]
pub struct RegisterFieldAST {
    pub name: String,
    pub bits: (u8, u8),                  // (start_bit, end_bit)
    pub access: String,
    pub reset_value: Option<Expression>,
}

// --- DOMAIN 3: Distributed ---
#[derive(Debug, Clone)]
pub struct ConsensusDeclAST {
    pub name: String,
    pub algorithm: String,               // raft, paxos, pbft
    pub config: Vec<(String, Expression)>,
    pub handlers: Vec<FunctionDecl>,
}

#[derive(Debug, Clone)]
pub struct CrdtDeclAST {
    pub name: String,
    pub crdt_type: String,               // gcounter, pncounter, orset, etc.
    pub element_type: Option<ChronosType>,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Debug, Clone)]
pub struct SagaDeclAST {
    pub name: String,
    pub steps: Vec<SagaStepAST>,
    pub on_failure: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
pub struct SagaStepAST {
    pub name: String,
    pub action: Vec<Statement>,
    pub compensation: Vec<Statement>,
    pub timeout: Option<Expression>,
}

// --- DOMAIN 4: Blockchain ---
// Syntax: contract ERC20 { storage balances: mapping(address => u256); ... }
#[derive(Debug, Clone)]
pub struct ContractDeclAST {
    pub name: String,
    pub target: Option<String>,          // evm, solana, etc.
    pub inherits: Vec<String>,
    pub storage: Vec<StorageVarAST>,
    pub events: Vec<EventDeclAST>,
    pub modifiers: Vec<ModifierDeclAST>,
    pub functions: Vec<FunctionDecl>,
    pub constructor: Option<FunctionDecl>,
}

#[derive(Debug, Clone)]
pub struct StorageVarAST {
    pub name: String,
    pub ty: ChronosType,
    pub visibility: Visibility,
    pub initial_value: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct EventDeclAST {
    pub name: String,
    pub fields: Vec<(String, ChronosType, bool)>, // (name, type, indexed)
}

#[derive(Debug, Clone)]
pub struct ModifierDeclAST {
    pub name: String,
    pub params: Vec<Parameter>,
    pub body: Vec<Statement>,
}

// --- DOMAIN 5: Formal Verification ---
// Syntax: theorem add_comm: forall a b: Nat, a + b = b + a { proof by { ... } }
#[derive(Debug, Clone)]
pub struct TheoremDeclAST {
    pub name: String,
    pub statement: ProofExprAST,
    pub proof: Option<ProofBlockAST>,
}

#[derive(Debug, Clone)]
pub struct AxiomDeclAST {
    pub name: String,
    pub statement: ProofExprAST,
    // Axioms have no proof — they are assumed true.
}

/// Proof expressions form their own sub-AST. The parser constructs this
/// tree, and the proof checker verifies it during type checking.
#[derive(Debug, Clone)]
pub enum ProofExprAST {
    Forall { var: String, ty: ChronosType, body: Box<ProofExprAST> },
    Exists { var: String, ty: ChronosType, body: Box<ProofExprAST> },
    Implies(Box<ProofExprAST>, Box<ProofExprAST>),
    And(Box<ProofExprAST>, Box<ProofExprAST>),
    Or(Box<ProofExprAST>, Box<ProofExprAST>),
    Not(Box<ProofExprAST>),
    Equal(Box<Expression>, Box<Expression>),
    Predicate(String, Vec<Expression>),
    // Temporal logic operators
    Always(Box<ProofExprAST>),
    Eventually(Box<ProofExprAST>),
    Until(Box<ProofExprAST>, Box<ProofExprAST>),
    LeadsTo(Box<ProofExprAST>, Box<ProofExprAST>),
}

/// A proof block contains tactics that construct the proof.
#[derive(Debug, Clone)]
pub struct ProofBlockAST {
    pub tactics: Vec<TacticAST>,
}

#[derive(Debug, Clone)]
pub enum TacticAST {
    Intro(Vec<String>),
    Apply(String),
    Exact(Expression),
    Rewrite(String, bool),              // (lemma_name, left_to_right)
    Induction { var: String, cases: Vec<(String, ProofBlockAST)> },
    Cases(String, Vec<(String, ProofBlockAST)>),
    Simp(Vec<String>),
    Ring, Omega, Linarith,
    Auto(Option<u32>),
    Trivial,
    SMT,
    Have { name: String, ty: ProofExprAST, proof: ProofBlockAST },
    Sequence(Vec<TacticAST>),
    Sorry,
}

/// Function contract annotations parsed from `requires` and `ensures`.
/// These are attached to FunctionDecl as new fields.
#[derive(Debug, Clone)]
pub struct FunctionContract {
    pub preconditions: Vec<ProofExprAST>,
    pub postconditions: Vec<ProofExprAST>,
    pub invariants: Vec<ProofExprAST>,
    pub decreases: Option<Expression>,
}

// --- DOMAIN 6: Database ---
#[derive(Debug, Clone)]
pub struct TableDeclAST {
    pub name: String,
    pub columns: Vec<ColumnDeclAST>,
    pub primary_key: Vec<String>,
    pub unique_constraints: Vec<Vec<String>>,
    pub foreign_keys: Vec<ForeignKeyAST>,
    pub indexes: Vec<IndexDeclAST>,
    pub checks: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub struct ColumnDeclAST {
    pub name: String,
    pub ty: ChronosType,
    pub nullable: bool,
    pub default: Option<Expression>,
    pub auto_increment: bool,
}

#[derive(Debug, Clone)]
pub struct ForeignKeyAST {
    pub columns: Vec<String>,
    pub ref_table: String,
    pub ref_columns: Vec<String>,
    pub on_delete: String,
    pub on_update: String,
}

#[derive(Debug, Clone)]
pub struct IndexDeclAST {
    pub name: String,
    pub table: String,
    pub columns: Vec<(String, bool)>,    // (column, ascending)
    pub unique: bool,
    pub method: Option<String>,          // btree, hash, gin, gist
}

// --- DOMAIN 7: Macros ---
#[derive(Debug, Clone)]
pub struct MacroDeclAST {
    pub name: String,
    pub kind: MacroKindAST,
    pub rules: Vec<MacroRuleAST>,
}

#[derive(Debug, Clone)]
pub enum MacroKindAST {
    Declarative,
    ProcAttribute,
    ProcDerive,
    ProcFunctionLike,
}

#[derive(Debug, Clone)]
pub struct MacroRuleAST {
    pub pattern: Vec<MacroPatternToken>,
    pub expansion: Vec<MacroPatternToken>,
}

#[derive(Debug, Clone)]
pub enum MacroPatternToken {
    Literal(String),
    MetaVar(String, String),             // ($name, fragment_kind)
    Repetition(Vec<MacroPatternToken>, String), // (pattern, separator)
}

#[derive(Debug, Clone)]
pub struct ComptimeBlockAST {
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct EmbedBlockAST {
    pub language: String,                // sql, regex, html, graphql, etc.
    pub content: String,                 // Raw content of the embedded DSL
    pub schema: Option<String>,          // Optional schema for validation
}

// --- DOMAINS 8-19: Simplified AST nodes ---
// These follow the same pattern: a name, some configuration fields,
// and a body. I'll define them more compactly.

#[derive(Debug, Clone)]
pub struct MultimediaDeclAST {
    pub name: String,
    pub kind: String,                    // "audio", "video", "image"
    pub config: Vec<(String, Expression)>,
    pub pipeline: Vec<(String, Vec<Expression>)>, // processing steps
}

#[derive(Debug, Clone)]
pub struct SceneDeclAST {
    pub name: String,
    pub objects: Vec<SceneObjectAST>,
    pub lights: Vec<(String, Vec<(String, Expression)>)>,
    pub camera: Vec<(String, Expression)>,
}

#[derive(Debug, Clone)]
pub struct SceneObjectAST {
    pub name: String,
    pub mesh: String,
    pub material: String,
    pub transform: Vec<(String, Expression)>,
}

#[derive(Debug, Clone)]
pub struct ShaderDeclAST {
    pub name: String,
    pub stage: String,                   // vertex, fragment, compute
    pub language: String,                // glsl, hlsl, wgsl
    pub code: String,
    pub inputs: Vec<(String, ChronosType)>,
    pub outputs: Vec<(String, ChronosType)>,
}

#[derive(Debug, Clone)]
pub struct RobotDeclAST {
    pub name: String,
    pub kinematics: Vec<(String, Expression)>,
    pub sensors: Vec<(String, String, Vec<(String, Expression)>)>,
    pub actuators: Vec<(String, String, Vec<(String, Expression)>)>,
    pub controllers: Vec<ControllerDeclAST>,
    pub behaviors: Vec<FunctionDecl>,
}

#[derive(Debug, Clone)]
pub struct ControllerDeclAST {
    pub name: String,
    pub kind: String,                    // pid, lqr, mpc, etc.
    pub params: Vec<(String, Expression)>,
    pub body: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
pub struct CircuitDeclAST {
    pub name: String,
    pub qubits: Vec<(String, Option<usize>)>,   // (name, optional_size)
    pub classical: Vec<(String, Option<usize>)>, // classical registers
    pub gates: Vec<QuantumGateAST>,
    pub measurements: Vec<MeasurementAST>,
}

#[derive(Debug, Clone)]
pub struct QuantumGateAST {
    pub gate: String,                    // h, x, cx, rz, etc.
    pub params: Vec<Expression>,         // rotation angles
    pub targets: Vec<(String, Option<Expression>)>, // (register, index)
}

#[derive(Debug, Clone)]
pub struct MeasurementAST {
    pub qubit: (String, Option<Expression>),
    pub classical: (String, Option<Expression>),
    pub basis: Option<String>,
}

#[derive(Debug, Clone)]
pub struct WidgetDeclAST {
    pub name: String,
    pub props: Vec<(String, ChronosType, Option<Expression>)>,
    pub state: Vec<(String, ChronosType, Expression)>,
    pub build: Vec<UINodeAST>,
    pub methods: Vec<FunctionDecl>,
}

#[derive(Debug, Clone)]
pub struct UINodeAST {
    pub widget_type: String,
    pub props: Vec<(String, Expression)>,
    pub children: Vec<UINodeAST>,
    pub event_handlers: Vec<(String, String)>,
    pub condition: Option<Expression>,
    pub for_each: Option<(String, Expression)>,
}

#[derive(Debug, Clone)]
pub struct ComponentDeclAST {
    pub name: String,
    pub props: Vec<(String, ChronosType)>,
    pub slots: Vec<String>,
    pub body: WidgetDeclAST,
}

#[derive(Debug, Clone)]
pub struct EntityDeclAST {
    pub name: String,
    pub components: Vec<(String, Vec<(String, Expression)>)>,
}

#[derive(Debug, Clone)]
pub struct SystemDeclAST {
    pub name: String,
    pub queries: Vec<SystemQueryAST>,
    pub body: Vec<Statement>,
    pub schedule: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SystemQueryAST {
    pub components: Vec<(String, bool)>, // (component_name, mutable)
    pub filters: Vec<(String, bool)>,    // (component_name, with_or_without)
}

#[derive(Debug, Clone)]
pub struct WorldDeclAST {
    pub name: String,
    pub systems: Vec<String>,
    pub resources: Vec<(String, ChronosType, Option<Expression>)>,
    pub startup: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
pub struct GenomeDeclAST {
    pub name: String,
    pub reference: String,
    pub annotations: Vec<(String, Expression)>,
    pub analyses: Vec<(String, Vec<(String, Expression)>)>,
}

#[derive(Debug, Clone)]
pub struct PortfolioDeclAST {
    pub name: String,
    pub assets: Vec<(String, Expression)>,
    pub constraints: Vec<Expression>,
    pub objective: String,
    pub rebalance: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BacktestDeclAST {
    pub name: String,
    pub strategy: String,
    pub data_source: String,
    pub date_range: (String, String),
    pub config: Vec<(String, Expression)>,
    pub metrics: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SpatialDeclAST {
    pub name: String,
    pub crs: Option<u32>,
    pub geometry_type: String,
    pub index: Option<String>,
    pub operations: Vec<(String, Vec<Expression>)>,
}

#[derive(Debug, Clone)]
pub struct TestDeclAST {
    pub name: String,
    pub body: Vec<Statement>,
    pub expected: Option<Expression>,
    pub should_panic: bool,
    pub timeout: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct BenchDeclAST {
    pub name: String,
    pub body: Vec<Statement>,
    pub iterations: Option<Expression>,
    pub warmup: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct FuzzDeclAST {
    pub name: String,
    pub corpus: Option<String>,
    pub body: Vec<Statement>,
    pub max_runs: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct PropertyTestDeclAST {
    pub name: String,
    pub generators: Vec<(String, ChronosType, Option<Expression>)>,
    pub property_body: Vec<Statement>,
    pub iterations: Option<Expression>,
}

#[derive(Debug, Clone)]
pub struct ExternDeclAST {
    pub abi: String,                     // "C", "C++", "Python", "JVM", etc.
    pub link_name: Option<String>,
    pub declarations: Vec<FunctionSignature>,
    pub link_libs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ForeignBlockAST {
    pub language: String,
    pub module: String,
    pub imports: Vec<(String, Option<String>)>, // (name, alias)
    pub config: Vec<(String, Expression)>,
}

#[derive(Debug, Clone)]
pub struct SimulationDeclAST {
    pub name: String,
    pub domain: Vec<(String, Expression)>,
    pub physics: Vec<(String, Vec<(String, Expression)>)>,
    pub mesh: Vec<(String, Expression)>,
    pub boundary_conditions: Vec<(String, Vec<(String, Expression)>)>,
    pub solver: Vec<(String, Expression)>,
    pub output: Vec<(String, Expression)>,
    pub time: Option<Vec<(String, Expression)>>,
}

#[derive(Debug, Clone)]
pub struct FemDeclAST {
    pub name: String,
    pub analysis_type: String,
    pub elements: Vec<(String, Expression)>,
    pub materials: Vec<(String, Vec<(String, Expression)>)>,
    pub loads: Vec<(String, Vec<(String, Expression)>)>,
    pub constraints: Vec<(String, Vec<(String, Expression)>)>,
    pub solver: Vec<(String, Expression)>,
}

#[derive(Debug, Clone)]
pub struct PipelineDeclAST {
    pub name: String,
    pub sources: Vec<(String, String, Vec<(String, Expression)>)>,
    pub transforms: Vec<(String, Vec<Expression>)>,
    pub sinks: Vec<(String, String, Vec<(String, Expression)>)>,
    pub schedule: Option<String>,
    pub on_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AuditBlockAST {
    pub mode: String,
    pub config: Vec<(String, Expression)>,
    pub body: Vec<Item>,
}


// ============================================================================
// PARSER EXTENSIONS — The parse_item() dispatch must be extended
// ============================================================================
// The critical change is in parse_item(), which is the main dispatch table.
// When the parser sees a keyword token, it needs to route to the appropriate
// parsing method. Here is the extended dispatch logic plus representative
// parsing methods for the most structurally interesting domains.

// ADD THESE MATCH ARMS to parse_item():
/*
fn parse_item(&mut self) -> Option<Item> {
    let visibility = self.parse_visibility();
    let annotations = self.parse_annotations();
    
    match self.peek()? {
        // ... existing arms (fn, class, struct, enum, trait, etc.) ...
        
        // === NEW DOMAIN DISPATCHES ===
        
        // Domain 1: Networking
        Token::KwProtocol => Some(Item::DomainItem(
            DomainItem::ProtocolDecl(self.parse_protocol_decl())
        )),
        Token::KwEndpoint => Some(Item::DomainItem(
            DomainItem::EndpointDecl(self.parse_endpoint_decl())
        )),
        
        // Domain 2: OS
        Token::KwDriver => Some(Item::DomainItem(
            DomainItem::DriverDecl(self.parse_driver_decl())
        )),
        Token::KwInterrupt | Token::KwIsr => Some(Item::DomainItem(
            DomainItem::InterruptHandler(self.parse_interrupt_handler())
        )),
        Token::KwRegister => Some(Item::DomainItem(
            DomainItem::RegisterDecl(self.parse_register_decl())
        )),
        
        // Domain 3: Distributed
        Token::KwConsensus => Some(Item::DomainItem(
            DomainItem::ConsensusDecl(self.parse_consensus_decl())
        )),
        Token::KwCrdt => Some(Item::DomainItem(
            DomainItem::CrdtDecl(self.parse_crdt_decl())
        )),
        Token::KwSaga => Some(Item::DomainItem(
            DomainItem::SagaDecl(self.parse_saga_decl())
        )),
        
        // Domain 4: Blockchain
        Token::KwContract => Some(Item::DomainItem(
            DomainItem::ContractDecl(self.parse_contract_decl())
        )),
        
        // Domain 5: Proof
        Token::KwTheorem => Some(Item::DomainItem(
            DomainItem::TheoremDecl(self.parse_theorem_decl())
        )),
        Token::KwLemma => Some(Item::DomainItem(
            DomainItem::LemmaDecl(self.parse_theorem_decl())
        )),
        Token::KwAxiom => Some(Item::DomainItem(
            DomainItem::AxiomDecl(self.parse_axiom_decl())
        )),
        
        // Domain 6: Database
        Token::KwTable => Some(Item::DomainItem(
            DomainItem::TableDecl(self.parse_table_decl())
        )),
        
        // Domain 7: Macros
        Token::KwMacro => Some(Item::DomainItem(
            DomainItem::MacroDecl(self.parse_macro_decl())
        )),
        Token::KwComptime => Some(Item::DomainItem(
            DomainItem::ComptimeBlock(self.parse_comptime_block())
        )),
        Token::KwEmbed => Some(Item::DomainItem(
            DomainItem::EmbedBlock(self.parse_embed_block())
        )),
        
        // Domain 8: Multimedia
        Token::KwScene => Some(Item::DomainItem(
            DomainItem::SceneDecl(self.parse_scene_decl())
        )),
        Token::KwShaderKw => Some(Item::DomainItem(
            DomainItem::ShaderDecl(self.parse_shader_decl())
        )),
        
        // Domain 9: Robotics
        Token::KwRobot => Some(Item::DomainItem(
            DomainItem::RobotDecl(self.parse_robot_decl())
        )),
        
        // Domain 10: Quantum
        Token::KwCircuit => Some(Item::DomainItem(
            DomainItem::CircuitDecl(self.parse_circuit_decl())
        )),
        
        // Domain 11: GUI
        Token::KwWidget => Some(Item::DomainItem(
            DomainItem::WidgetDecl(self.parse_widget_decl())
        )),
        Token::KwComponent => Some(Item::DomainItem(
            DomainItem::ComponentDecl(self.parse_component_decl())
        )),
        
        // Domain 12: Game
        Token::KwEntity => Some(Item::DomainItem(
            DomainItem::EntityDecl(self.parse_entity_decl())
        )),
        Token::KwSystem => Some(Item::DomainItem(
            DomainItem::SystemDecl(self.parse_system_decl())
        )),
        Token::KwWorld => Some(Item::DomainItem(
            DomainItem::WorldDecl(self.parse_world_decl())
        )),
        
        // Domain 13: Bio
        Token::KwGenome => Some(Item::DomainItem(
            DomainItem::GenomeDecl(self.parse_genome_decl())
        )),
        
        // Domain 14: Finance
        Token::KwPortfolio => Some(Item::DomainItem(
            DomainItem::PortfolioDecl(self.parse_portfolio_decl())
        )),
        Token::KwBacktest => Some(Item::DomainItem(
            DomainItem::BacktestDecl(self.parse_backtest_decl())
        )),
        
        // Domain 16: Testing
        Token::KwTest => Some(Item::DomainItem(
            DomainItem::TestDecl(self.parse_test_decl())
        )),
        Token::KwBench => Some(Item::DomainItem(
            DomainItem::BenchDecl(self.parse_bench_decl())
        )),
        Token::KwFuzz => Some(Item::DomainItem(
            DomainItem::FuzzDecl(self.parse_fuzz_decl())
        )),
        Token::KwProperty => Some(Item::DomainItem(
            DomainItem::PropertyTestDecl(self.parse_property_test())
        )),
        
        // Domain 17: FFI
        Token::KwExtern => Some(Item::DomainItem(
            DomainItem::ExternDecl(self.parse_extern_decl())
        )),
        Token::KwForeign => Some(Item::DomainItem(
            DomainItem::ForeignBlock(self.parse_foreign_block())
        )),
        
        // Domain 18: Simulation
        Token::KwSimulation => Some(Item::DomainItem(
            DomainItem::SimulationDecl(self.parse_simulation_decl())
        )),
        Token::KwFem => Some(Item::DomainItem(
            DomainItem::FemDecl(self.parse_fem_decl())
        )),
        
        // Domain 19: Pipeline
        Token::KwPipelineKw => Some(Item::DomainItem(
            DomainItem::PipelineDecl(self.parse_pipeline_decl())
        )),
        
        // Domain 20: Security
        Token::KwAudit => Some(Item::DomainItem(
            DomainItem::AuditBlock(self.parse_audit_block())
        )),
        
        _ => None,
    }
}
*/

// ============================================================================
// REPRESENTATIVE PARSING METHODS
// ============================================================================
// I'll implement the most structurally interesting parsers in detail.
// The remaining ones follow the same pattern: consume keyword, parse name,
// parse brace-enclosed body with domain-specific fields.

/// These methods would be added to the existing Parser impl block.
/// I'm wrapping them in a trait for compilation clarity.
pub trait DomainParsing {
    // Core helpers assumed available from the base parser:
    fn peek(&self) -> Option<&Token>;
    fn advance(&mut self) -> Option<&SpannedToken>;
    fn expect(&mut self, expected: &Token) -> bool;
    fn expect_identifier(&mut self) -> String;
    fn eat(&mut self, expected: &Token) -> bool;
    fn is_eof(&self) -> bool;
    fn parse_expression(&mut self, min_bp: u8) -> Expression;
    fn parse_block(&mut self) -> Vec<Statement>;
    fn parse_type(&mut self) -> ChronosType;
    fn parse_param_list(&mut self) -> Vec<Parameter>;
    fn parse_function_decl(&mut self, vis: Visibility, ann: Vec<Annotation>) -> FunctionDecl;
    fn parse_visibility(&mut self) -> Visibility;
    fn parse_annotations(&mut self) -> Vec<Annotation>;
    fn error_here(&mut self, msg: &str);

    // === PROTOCOL PARSING ===
    // Syntax: protocol HTTP2Frame on tcp big_endian { ... }
    fn parse_protocol_decl(&mut self) -> ProtocolDeclAST {
        self.advance(); // consume 'protocol'
        let name = self.expect_identifier();
        
        // Optional transport and endianness.
        let mut transport = None;
        let mut endianness = None;
        while let Some(Token::Identifier(ref s)) = self.peek().cloned() {
            match s.as_str() {
                "tcp" | "udp" | "quic" | "sctp" | "tls" | "websocket" => {
                    transport = Some(s.clone()); self.advance();
                }
                "big_endian" | "little_endian" | "network" => {
                    endianness = Some(s.clone()); self.advance();
                }
                _ => break,
            }
        }
        
        self.expect(&Token::LBrace);
        
        let mut fields = Vec::new();
        let mut state_machine = None;
        let mut constraints = Vec::new();
        
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            match self.peek() {
                Some(Token::KwField) => {
                    self.advance();
                    let fname = self.expect_identifier();
                    self.expect(&Token::Colon);
                    // Protocol wire types are parsed as strings because they
                    // include non-standard widths like u24, u31, bit, etc.
                    let wtype = self.expect_identifier();
                    let validation = if self.eat(&Token::KwWhere) {
                        Some(self.expect_identifier())
                    } else { None };
                    self.eat(&Token::Semicolon);
                    fields.push(ProtocolFieldAST { name: fname, wire_type: wtype, validation });
                }
                Some(Token::KwState) => {
                    state_machine = Some(self.parse_state_machine());
                }
                _ => { self.advance(); }
            }
        }
        
        self.expect(&Token::RBrace);
        ProtocolDeclAST { name, transport, endianness, fields, state_machine, constraints }
    }
    
    fn parse_state_machine(&mut self) -> StateMachineAST {
        self.advance(); // consume 'state'
        // Optionally consume '_machine' or 'machine' if present
        if let Some(Token::Identifier(ref s)) = self.peek() {
            if s == "machine" || s == "_machine" { self.advance(); }
        }
        self.expect(&Token::LBrace);
        
        let mut transitions = Vec::new();
        let mut states = Vec::new();
        
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let from = self.expect_identifier();
            if !states.contains(&from) { states.push(from.clone()); }
            self.expect(&Token::Arrow); // ->
            let to = self.expect_identifier();
            if !states.contains(&to) { states.push(to.clone()); }
            let trigger = if self.eat(&Token::KwOn) {
                self.expect_identifier()
            } else { String::new() };
            let guard = None;
            self.eat(&Token::Semicolon);
            transitions.push(TransitionAST { from, to, trigger, guard });
        }
        
        self.expect(&Token::RBrace);
        let initial = states.first().cloned().unwrap_or_default();
        StateMachineAST { states, initial, transitions }
    }

    // === SMART CONTRACT PARSING ===
    // Syntax: contract ERC20 target evm { storage ...; event ...; fn ...; }
    fn parse_contract_decl(&mut self) -> ContractDeclAST {
        self.advance(); // consume 'contract'
        let name = self.expect_identifier();
        
        let target = if let Some(Token::Identifier(ref s)) = self.peek() {
            if matches!(s.as_str(), "evm" | "solana" | "cosmwasm" | "near" | "substrate") {
                let t = s.clone(); self.advance(); Some(t)
            } else { None }
        } else { None };
        
        // Optional inheritance: contract Token : ERC20, Ownable { ... }
        let mut inherits = Vec::new();
        if self.eat(&Token::Colon) {
            loop {
                inherits.push(self.expect_identifier());
                if !self.eat(&Token::Comma) { break; }
            }
        }
        
        self.expect(&Token::LBrace);
        
        let mut storage = Vec::new();
        let mut events = Vec::new();
        let mut modifiers = Vec::new();
        let mut functions = Vec::new();
        let mut constructor = None;
        
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let vis = self.parse_visibility();
            let anns = self.parse_annotations();

            match self.peek().cloned() {
                Some(Token::KwStorage) => {
                    self.advance();
                    let sname = self.expect_identifier();
                    self.expect(&Token::Colon);
                    let ty = self.parse_type();
                    let initial = if self.eat(&Token::Eq) {
                        Some(self.parse_expression(0))
                    } else { None };
                    self.eat(&Token::Semicolon);
                    storage.push(StorageVarAST { name: sname, ty, visibility: vis, initial_value: initial });
                }
                Some(Token::KwEvent) => {
                    self.advance();
                    let ename = self.expect_identifier();
                    self.expect(&Token::LParen);
                    let mut fields = Vec::new();
                    while self.peek() != Some(&Token::RParen) && !self.is_eof() {
                        let indexed = self.eat(&Token::KwIndex);
                        let fname = self.expect_identifier();
                        self.expect(&Token::Colon);
                        let fty = self.parse_type();
                        fields.push((fname, fty, indexed));
                        self.eat(&Token::Comma);
                    }
                    self.expect(&Token::RParen);
                    self.eat(&Token::Semicolon);
                    events.push(EventDeclAST { name: ename, fields });
                }
                Some(Token::KwModifier) => {
                    self.advance();
                    let mname = self.expect_identifier();
                    let params = if self.peek() == Some(&Token::LParen) {
                        self.advance();
                        let p = self.parse_param_list();
                        self.expect(&Token::RParen);
                        p
                    } else { Vec::new() };
                    let body = self.parse_block();
                    modifiers.push(ModifierDeclAST { name: mname, params, body });
                }
                Some(Token::KwNew) | Some(Token::Identifier(_)) => {
                    self.advance();
                    let f = self.parse_function_decl(vis, anns);
                    constructor = Some(f);
                }
                Some(Token::KwFn) | Some(Token::KwFun) |
                Some(Token::KwPayable) | Some(Token::KwView) => {
                    let f = self.parse_function_decl(vis, anns);
                    functions.push(f);
                }
                _ => { self.advance(); }
            }
        }
        
        self.expect(&Token::RBrace);
        ContractDeclAST { name, target, inherits, storage, events, modifiers, functions, constructor }
    }

    // === THEOREM/PROOF PARSING ===
    // Syntax: theorem add_comm: forall a b: Nat, a + b = b + a { proof by { ... } }
    fn parse_theorem_decl(&mut self) -> TheoremDeclAST {
        self.advance(); // consume 'theorem' or 'lemma'
        let name = self.expect_identifier();
        self.expect(&Token::Colon);
        let statement = self.parse_proof_expr();
        let proof = if self.peek() == Some(&Token::LBrace) || self.peek() == Some(&Token::KwProof) {
            if self.eat(&Token::KwProof) { /* optional 'proof' keyword */ }
            Some(self.parse_proof_block())
        } else {
            self.eat(&Token::Semicolon);
            None
        };
        TheoremDeclAST { name, statement, proof }
    }
    
    fn parse_axiom_decl(&mut self) -> AxiomDeclAST {
        self.advance(); // consume 'axiom'
        let name = self.expect_identifier();
        self.expect(&Token::Colon);
        let statement = self.parse_proof_expr();
        self.eat(&Token::Semicolon);
        AxiomDeclAST { name, statement }
    }
    
    fn parse_proof_expr(&mut self) -> ProofExprAST {
        match self.peek() {
            Some(Token::KwForall) => {
                self.advance();
                let var = self.expect_identifier();
                self.expect(&Token::Colon);
                let ty = self.parse_type();
                self.eat(&Token::Comma);
                let body = self.parse_proof_expr();
                ProofExprAST::Forall { var, ty, body: Box::new(body) }
            }
            Some(Token::KwExists) => {
                self.advance();
                let var = self.expect_identifier();
                self.expect(&Token::Colon);
                let ty = self.parse_type();
                self.eat(&Token::Comma);
                let body = self.parse_proof_expr();
                ProofExprAST::Exists { var, ty, body: Box::new(body) }
            }
            _ => {
                // Parse as an expression, then check for = or ==> or <=>
                let lhs = self.parse_expression(0);
                if self.eat(&Token::EqEq) {
                    let rhs = self.parse_expression(0);
                    ProofExprAST::Equal(Box::new(lhs), Box::new(rhs))
                } else if self.eat(&Token::LongFatArrow) {
                    let rhs = self.parse_proof_expr();
                    ProofExprAST::Implies(
                        Box::new(ProofExprAST::Predicate("_".to_string(), vec![lhs])),
                        Box::new(rhs),
                    )
                } else {
                    ProofExprAST::Predicate("_".to_string(), vec![lhs])
                }
            }
        }
    }
    
    fn parse_proof_block(&mut self) -> ProofBlockAST {
        // A proof block can be: { tactic1; tactic2; ... }
        // or: by { tactic1; tactic2; ... }
        self.eat(&Token::KwBy);
        self.expect(&Token::LBrace);
        
        let mut tactics = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            tactics.push(self.parse_tactic());
            self.eat(&Token::Semicolon);
        }
        self.expect(&Token::RBrace);
        ProofBlockAST { tactics }
    }
    
    fn parse_tactic(&mut self) -> TacticAST {
        match self.peek().cloned() {
            Some(Token::Identifier(ref s)) if s == "intro" => {
                self.advance();
                let mut vars = Vec::new();
                while let Some(Token::Identifier(_)) = self.peek() {
                    vars.push(self.expect_identifier());
                }
                TacticAST::Intro(vars)
            }
            Some(Token::Identifier(ref s)) if s == "apply" => {
                self.advance();
                TacticAST::Apply(self.expect_identifier())
            }
            Some(Token::Identifier(ref s)) if s == "rewrite" => {
                self.advance();
                let ltr = !self.eat(&Token::LeftArrow);
                TacticAST::Rewrite(self.expect_identifier(), ltr)
            }
            Some(Token::Identifier(ref s)) if s == "simp" => {
                self.advance();
                let mut lemmas = Vec::new();
                if self.eat(&Token::LBracket) {
                    while self.peek() != Some(&Token::RBracket) {
                        lemmas.push(self.expect_identifier());
                        self.eat(&Token::Comma);
                    }
                    self.expect(&Token::RBracket);
                }
                TacticAST::Simp(lemmas)
            }
            Some(Token::KwInduction) => {
                self.advance();
                let var = self.expect_identifier();
                self.expect(&Token::LBrace);
                let mut cases = Vec::new();
                while self.peek() != Some(&Token::RBrace) && !self.is_eof() {
                    let case_name = self.expect_identifier();
                    self.expect(&Token::FatArrow);
                    let case_proof = self.parse_proof_block();
                    cases.push((case_name, case_proof));
                }
                self.expect(&Token::RBrace);
                TacticAST::Induction { var, cases }
            }
            Some(Token::Identifier(ref s)) if s == "ring" => { self.advance(); TacticAST::Ring }
            Some(Token::Identifier(ref s)) if s == "omega" => { self.advance(); TacticAST::Omega }
            Some(Token::Identifier(ref s)) if s == "linarith" => { self.advance(); TacticAST::Linarith }
            Some(Token::Identifier(ref s)) if s == "auto" => {
                self.advance();
                let depth = if let Some(Token::IntLiteral(n)) = self.peek() {
                    let n = *n as u32; self.advance(); Some(n)
                } else { None };
                TacticAST::Auto(depth)
            }
            Some(Token::Identifier(ref s)) if s == "trivial" => { self.advance(); TacticAST::Trivial }
            Some(Token::Identifier(ref s)) if s == "smt" => { self.advance(); TacticAST::SMT }
            Some(Token::KwSorry) => { self.advance(); TacticAST::Sorry }
            Some(Token::Identifier(ref s)) if s == "exact" => {
                self.advance();
                TacticAST::Exact(self.parse_expression(0))
            }
            Some(Token::Identifier(ref s)) if s == "have" => {
                self.advance();
                let name = self.expect_identifier();
                self.expect(&Token::Colon);
                let ty = self.parse_proof_expr();
                let proof = self.parse_proof_block();
                TacticAST::Have { name, ty, proof }
            }
            _ => {
                self.error_here("Expected proof tactic");
                self.advance();
                TacticAST::Trivial
            }
        }
    }

    // === QUANTUM CIRCUIT PARSING ===
    // Syntax: circuit Teleport { qubit q[3]; creg c[3]; h q[0]; cx q[0], q[1]; ... }
    fn parse_circuit_decl(&mut self) -> CircuitDeclAST {
        self.advance(); // consume 'circuit'
        let name = self.expect_identifier();
        self.expect(&Token::LBrace);
        
        let mut qubits = Vec::new();
        let mut classical = Vec::new();
        let mut gates = Vec::new();
        let mut measurements = Vec::new();
        
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            match self.peek() {
                Some(Token::KwQubit) | Some(Token::KwQreg) => {
                    self.advance();
                    let qname = self.expect_identifier();
                    let size = if self.eat(&Token::LBracket) {
                        let n = if let Some(Token::IntLiteral(v)) = self.peek() {
                            let v = *v as usize; self.advance(); Some(v)
                        } else { None };
                        self.expect(&Token::RBracket);
                        n
                    } else { None };
                    self.eat(&Token::Semicolon);
                    qubits.push((qname, size));
                }
                Some(Token::KwCreg) => {
                    self.advance();
                    let cname = self.expect_identifier();
                    let size = if self.eat(&Token::LBracket) {
                        let n = if let Some(Token::IntLiteral(v)) = self.peek() {
                            let v = *v as usize; self.advance(); Some(v)
                        } else { None };
                        self.expect(&Token::RBracket);
                        n
                    } else { None };
                    self.eat(&Token::Semicolon);
                    classical.push((cname, size));
                }
                Some(Token::KwMeasure) => {
                    self.advance();
                    let qreg = self.expect_identifier();
                    let qidx = if self.eat(&Token::LBracket) {
                        let e = self.parse_expression(0);
                        self.expect(&Token::RBracket);
                        Some(e)
                    } else { None };
                    self.expect(&Token::Arrow);
                    let creg = self.expect_identifier();
                    let cidx = if self.eat(&Token::LBracket) {
                        let e = self.parse_expression(0);
                        self.expect(&Token::RBracket);
                        Some(e)
                    } else { None };
                    self.eat(&Token::Semicolon);
                    measurements.push(MeasurementAST {
                        qubit: (qreg, qidx), classical: (creg, cidx), basis: None
                    });
                }
                Some(Token::KwGate) | Some(Token::Identifier(_)) => {
                    // Parse gate application: h q[0]; cx q[0], q[1]; rz(pi/4) q[2];
                    let gate_name = self.expect_identifier();
                    let params = if self.eat(&Token::LParen) {
                        let mut p = Vec::new();
                        while self.peek() != Some(&Token::RParen) && !self.is_eof() {
                            p.push(self.parse_expression(0));
                            self.eat(&Token::Comma);
                        }
                        self.expect(&Token::RParen);
                        p
                    } else { Vec::new() };
                    let mut targets = Vec::new();
                    loop {
                        let treg = self.expect_identifier();
                        let tidx = if self.eat(&Token::LBracket) {
                            let e = self.parse_expression(0);
                            self.expect(&Token::RBracket);
                            Some(e)
                        } else { None };
                        targets.push((treg, tidx));
                        if !self.eat(&Token::Comma) { break; }
                    }
                    self.eat(&Token::Semicolon);
                    gates.push(QuantumGateAST { gate: gate_name, params, targets });
                }
                _ => { self.advance(); }
            }
        }
        
        self.expect(&Token::RBrace);
        CircuitDeclAST { name, qubits, classical, gates, measurements }
    }

    // === TEST DECLARATION PARSING ===
    // Syntax: test "should add two numbers" { expect(1 + 1 == 2); }
    fn parse_test_decl(&mut self) -> TestDeclAST {
        self.advance(); // consume 'test'
        let name = if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
            self.advance(); s
        } else {
            self.expect_identifier()
        };
        let timeout = if let Some(Token::Identifier(ref s)) = self.peek() {
            if s == "timeout" {
                self.advance();
                Some(self.parse_expression(0))
            } else { None }
        } else { None };
        let body = self.parse_block();
        TestDeclAST { name, body, expected: None, should_panic: false, timeout }
    }

    // === GENERIC DOMAIN BLOCK PARSER ===
    // Many domains follow the pattern: keyword Name { key: value; key: value; }
    // This utility parses that pattern into a Vec<(String, Expression)>.
    fn parse_config_block(&mut self) -> Vec<(String, Expression)> {
        self.expect(&Token::LBrace);
        let mut config = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let key = self.expect_identifier();
            self.expect(&Token::Colon);
            let value = self.parse_expression(0);
            self.eat(&Token::Semicolon);
            self.eat(&Token::Comma);
            config.push((key, value));
        }
        self.expect(&Token::RBrace);
        config
    }
    
    // All remaining domain parsers (simulation, FEM, pipeline, robot, etc.)
    // follow the same structural pattern as parse_protocol_decl but with
    // their own keyword sets. Each one:
    //   1. Consumes the leading keyword
    //   2. Parses the name
    //   3. Parses a brace-enclosed body with domain-specific sub-keywords
    //   4. Returns the appropriate AST node
    //
    // I'll show one more to illustrate the pattern, then the rest would
    // be mechanical variants.

    // === SIMULATION PARSING ===
    // Syntax:
    //   simulation FluidFlow {
    //       domain 3d { width: 10.0; height: 5.0; depth: 5.0 }
    //       physics incompressible_ns { reynolds: 1000.0 }
    //       mesh cartesian { cells: [100, 50, 50] }
    //       boundary wall_left { type: "no_slip" }
    //       time { start: 0.0; end: 10.0; dt: 0.001 }
    //       solve { method: "gmres"; tolerance: 1e-6 }
    //       output { format: "vtk"; every: 100 }
    //   }
    fn parse_simulation_decl(&mut self) -> SimulationDeclAST {
        self.advance(); // consume 'simulation'
        let name = self.expect_identifier();
        self.expect(&Token::LBrace);
        
        let mut domain = Vec::new();
        let mut physics = Vec::new();
        let mut mesh = Vec::new();
        let mut boundary_conditions = Vec::new();
        let mut solver = Vec::new();
        let mut output = Vec::new();
        let mut time = None;
        
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            match self.peek().cloned() {
                Some(Token::KwDomain) => {
                    self.advance();
                    let kind = self.expect_identifier();
                    domain = self.parse_config_block();
                    domain.insert(0, ("kind".to_string(),
                        Expression::StringLiteral(kind)));
                }
                Some(Token::KwPhysics) => {
                    self.advance();
                    let model = self.expect_identifier();
                    let config = self.parse_config_block();
                    physics.push((model, config));
                }
                Some(Token::KwMesh) => {
                    self.advance();
                    let kind = self.expect_identifier();
                    mesh = self.parse_config_block();
                    mesh.insert(0, ("kind".to_string(),
                        Expression::StringLiteral(kind)));
                }
                Some(Token::KwBoundary) => {
                    self.advance();
                    let bname = self.expect_identifier();
                    let config = self.parse_config_block();
                    boundary_conditions.push((bname, config));
                }
                Some(Token::KwSolve) | Some(Token::Identifier(_)) => {
                    self.advance();
                    solver = self.parse_config_block();
                }
                _ => { self.advance(); }
            }
        }
        
        self.expect(&Token::RBrace);
        SimulationDeclAST {
            name, domain, physics, mesh, boundary_conditions,
            solver, output, time,
        }
    }

    // Stub signatures for all remaining domain parsers. In a complete
    // implementation, each follows the same keyword-name-brace pattern.
    fn parse_endpoint_decl(&mut self) -> EndpointDeclAST;
    fn parse_driver_decl(&mut self) -> DriverDeclAST;
    fn parse_interrupt_handler(&mut self) -> InterruptHandlerAST;
    fn parse_register_decl(&mut self) -> RegisterDeclAST;
    fn parse_consensus_decl(&mut self) -> ConsensusDeclAST;
    fn parse_crdt_decl(&mut self) -> CrdtDeclAST;
    fn parse_saga_decl(&mut self) -> SagaDeclAST;
    fn parse_table_decl(&mut self) -> TableDeclAST;
    fn parse_macro_decl(&mut self) -> MacroDeclAST;
    fn parse_comptime_block(&mut self) -> ComptimeBlockAST;
    fn parse_embed_block(&mut self) -> EmbedBlockAST;
    fn parse_scene_decl(&mut self) -> SceneDeclAST;
    fn parse_shader_decl(&mut self) -> ShaderDeclAST;
    fn parse_robot_decl(&mut self) -> RobotDeclAST;
    fn parse_widget_decl(&mut self) -> WidgetDeclAST;
    fn parse_component_decl(&mut self) -> ComponentDeclAST;
    fn parse_entity_decl(&mut self) -> EntityDeclAST;
    fn parse_system_decl(&mut self) -> SystemDeclAST;
    fn parse_world_decl(&mut self) -> WorldDeclAST;
    fn parse_genome_decl(&mut self) -> GenomeDeclAST;
    fn parse_portfolio_decl(&mut self) -> PortfolioDeclAST;
    fn parse_backtest_decl(&mut self) -> BacktestDeclAST;
    fn parse_bench_decl(&mut self) -> BenchDeclAST;
    fn parse_fuzz_decl(&mut self) -> FuzzDeclAST;
    fn parse_property_test(&mut self) -> PropertyTestDeclAST;
    fn parse_extern_decl(&mut self) -> ExternDeclAST;
    fn parse_foreign_block(&mut self) -> ForeignBlockAST;
    fn parse_fem_decl(&mut self) -> FemDeclAST;
    fn parse_pipeline_decl(&mut self) -> PipelineDeclAST;
    fn parse_audit_block(&mut self) -> AuditBlockAST;

    // === TYPE PARSING EXTENSIONS ===
    // The parse_type() method needs new arms for domain-specific types.
    // Add these to the existing parse_type() match:
    /*
    Some(Token::TyU256) => { self.advance(); ChronosType::UInt256 }
    Some(Token::TyI256) => { self.advance(); ChronosType::Int256 }
    Some(Token::TyU512) => { self.advance(); ChronosType::UInt512 }
    Some(Token::TyDecimal) => { self.advance(); ChronosType::Decimal128 }
    Some(Token::TyComplex) => { self.advance(); ChronosType::Complex128 }
    Some(Token::TyQuaternion) => { self.advance(); ChronosType::Quaternion }
    Some(Token::TyRatio) => { self.advance(); ChronosType::Ratio }
    Some(Token::TyBigInt) => { self.advance(); ChronosType::IntArbitrary }
    Some(Token::TyBigFloat) => { self.advance(); ChronosType::FloatArbitrary }
    Some(Token::TySymbol) => { self.advance(); ChronosType::Symbolic }
    Some(Token::KwQubit) => { self.advance(); ChronosType::Qubit }
    Some(Token::KwAddress) => { self.advance(); ChronosType::Address }
    
    // Unit-of-measure types: f64<meter>, f32<kilogram * meter / second^2>
    // After parsing a numeric type, check for <unit> suffix:
    // (this would be added as a postfix check in parse_type)
    */
    
    // === EXPRESSION PARSING EXTENSIONS ===
    // The parse_prefix() method needs new arms for domain expressions.
    // Add these to parse_prefix():
    /*
    // Quantum ket notation: |0⟩, |+⟩, |ψ⟩
    Some(Token::Pipe) if is_quantum_context => {
        self.advance();
        let state = self.expect_identifier();
        self.expect(&Token::Gt); // Using > as ket closer
        Expression::QuantumKet(state)
    }
    
    // Channel operations: <- channel (receive)
    Some(Token::LeftArrow) => {
        self.advance();
        let channel = self.parse_expression(25);
        Expression::ChannelReceive(Box::new(channel))
    }
    
    // Embed expression: sql { SELECT * FROM users }
    // embed regex { ^[a-z]+$ }
    Some(Token::KwEmbed) => {
        self.advance();
        let lang = self.expect_identifier();
        // Read raw content until matching close brace
        // (actual implementation would use a raw block scanner)
        self.expect(&Token::LBrace);
        let content = String::new(); // simplified
        self.expect(&Token::RBrace);
        Expression::EmbeddedDSL { language: lang, content }
    }
    
    // Assert expression: assert(x > 0, "x must be positive")
    Some(Token::KwAssert) => {
        self.advance();
        self.expect(&Token::LParen);
        let condition = self.parse_expression(0);
        let message = if self.eat(&Token::Comma) {
            if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
                self.advance(); Some(s)
            } else { None }
        } else { None };
        self.expect(&Token::RParen);
        Expression::Assert { condition: Box::new(condition), message }
    }
    
    // Emit (blockchain event): emit Transfer(from, to, amount);
    Some(Token::KwEmit) => {
        self.advance();
        let event_name = self.expect_identifier();
        self.expect(&Token::LParen);
        let args = self.parse_expression_list();
        self.expect(&Token::RParen);
        Expression::EmitEvent { name: event_name, args }
    }
    */

    // === STATEMENT PARSING EXTENSIONS ===
    // Add these to parse_statement():
    /*
    // Spawn: spawn fiber { ... }
    Some(Token::KwSpawn) => {
        self.advance();
        let kind = self.expect_identifier(); // thread, fiber, process
        let body = self.parse_block();
        Statement::Spawn { kind, body }
    }
    
    // Critical section: critical { ... } (disables interrupts)
    Some(Token::KwCritical) => {
        self.advance();
        let body = self.parse_block();
        Statement::CriticalSection(body)
    }
    
    // Transaction: transaction { ... commit; }
    Some(Token::KwTransaction) => {
        self.advance();
        let body = self.parse_block();
        Statement::Transaction(body)
    }
    
    // Trace span: trace "operation_name" { ... }
    Some(Token::KwTrace) => {
        self.advance();
        let span_name = if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
            self.advance(); s
        } else { self.expect_identifier() };
        let body = self.parse_block();
        Statement::TraceSpan { name: span_name, body }
    }
    
    // Log: log info "message";
    Some(Token::KwLog) => {
        self.advance();
        let level = self.expect_identifier();
        let message = self.parse_expression(0);
        self.eat(&Token::Semicolon);
        Statement::Log { level, message }
    }
    
    // Require (blockchain): require(condition, "error message");
    Some(Token::KwRequire) => {
        self.advance();
        self.expect(&Token::LParen);
        let condition = self.parse_expression(0);
        let message = if self.eat(&Token::Comma) {
            Some(self.parse_expression(0))
        } else { None };
        self.expect(&Token::RParen);
        self.eat(&Token::Semicolon);
        Statement::Require { condition, message }
    }
    */
}

// NOTE: Placeholder types removed — when this file is compiled as part of
// chronos-compiler-core, the real definitions are provided by earlier includes.
// Individual standalone crates that include only this file should provide
// their own stubs if needed.
