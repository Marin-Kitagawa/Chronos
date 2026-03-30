// ============================================================================
// CHRONOS LEXER v2 — ADDITIONS FOR ALL NEW DOMAINS
// ============================================================================
// This file contains ONLY the new tokens that must be added to the Token enum
// in the original lexer. Each group corresponds to one of the 25 new domains.
// In the real codebase, these would be merged into the single Token enum.
//
// WHY THE LEXER MUST CHANGE:
// The lexer is the compiler's "vocabulary". If Chronos can't recognize the
// word "protocol" or "qubit" as a keyword, the parser will never see them —
// they'll be parsed as ordinary identifiers, causing confusing errors.
// Every domain-specific construct needs its own keyword(s) so the parser
// can unambiguously dispatch to the right parsing routine.
// ============================================================================

use logos::Logos;

/// New tokens to merge into the main Token enum.
/// Each token has a `#[token("...")]` attribute for logos.
///
/// INTEGRATION INSTRUCTIONS:
/// Copy these variants into the Token enum in chronos-lexer.
/// logos automatically assigns correct priority to exact string matches
/// over the Identifier regex, so keyword reservation is handled for free.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r]+")]
pub enum ExtendedTokens {

    // =================================================================
    // DOMAIN 1: NETWORKING & PROTOCOL ENGINEERING
    // =================================================================
    // These keywords enable the `protocol MyProto { ... }` declaration
    // syntax and the built-in server/client patterns.
    
    #[token("protocol")]    KwProtocol,      // protocol HTTP2Frame { ... }
    #[token("field")]       KwField,         // field length: u24;
    #[token("state")]       KwState,         // state_machine { ... }
    #[token("transition")]  KwTransition,    // Idle -> Open on HEADERS;
    #[token("on")]          KwOn,            // trigger for state transitions
    #[token("endpoint")]    KwEndpoint,      // endpoint "/api/users" { ... }
    #[token("route")]       KwRoute,         // route GET "/users" -> handler;
    #[token("middleware")]  KwMiddleware,    // middleware logging { ... }
    #[token("listen")]      KwListen,        // listen "0.0.0.0:8080";
    #[token("connect")]     KwConnect,       // connect "peer:9090";
    #[token("stream")]      KwStream,        // stream processing

    // =================================================================
    // DOMAIN 2: OPERATING SYSTEM PRIMITIVES
    // =================================================================
    // These enable writing kernel code, device drivers, and embedded
    // firmware directly in Chronos.
    
    #[token("syscall")]     KwSyscall,       // syscall(NR_write, fd, buf, len)
    #[token("interrupt")]   KwInterrupt,     // interrupt 0x80 { ... }
    #[token("driver")]      KwDriver,        // driver USBMassStorage { ... }
    #[token("register")]    KwRegister,      // register CONTROL @ 0x4000_0000 { ... }
    #[token("volatile")]    KwVolatile,      // volatile read/write for MMIO
    #[token("dma")]         KwDma,           // dma transfer src -> dst, len;
    #[token("isr")]         KwIsr,           // isr timer_tick() { ... }
    #[token("critical")]    KwCritical,      // critical { ... } (disable interrupts)
    #[token("atomic")]      KwAtomic,        // atomic operations
    #[token("barrier")]     KwBarrier,       // memory barrier / fence
    #[token("process")]     KwProcess,       // process management
    #[token("spawn")]       KwSpawn,         // spawn thread/process/fiber
    #[token("fiber")]       KwFiber,         // lightweight cooperative thread
    #[token("sandbox")]     KwSandbox,       // sandboxed execution context
    #[token("namespace")]   KwNamespace,     // OS namespace (also C++ namespace)

    // =================================================================
    // DOMAIN 3: DISTRIBUTED SYSTEMS
    // =================================================================
    
    #[token("consensus")]   KwConsensus,     // consensus raft { ... }
    #[token("crdt")]        KwCrdt,          // crdt GCounter { ... }
    #[token("replicated")]  KwReplicated,    // replicated var count: i64;
    #[token("saga")]        KwSaga,          // saga OrderFlow { ... }
    #[token("compensate")]  KwCompensate,    // compensate { ... } (saga undo)
    #[token("partition")]   KwPartition,     // partition tolerance config
    #[token("shard")]       KwShard,         // shard key = user_id;
    #[token("quorum")]      KwQuorum,        // quorum(3) { ... }
    #[token("eventually")]  KwEventually,    // eventually consistent
    #[token("linearizable")]KwLinearizable,  // linearizable operations
    #[token("causal")]      KwCausal,        // causal ordering

    // =================================================================
    // DOMAIN 4: BLOCKCHAIN / SMART CONTRACTS
    // =================================================================
    
    #[token("contract")]    KwContract,      // contract ERC20 { ... }
    #[token("storage")]     KwStorage,       // storage mapping balances: ...
    #[token("event")]       KwEvent,         // event Transfer(from, to, amount)
    #[token("modifier")]    KwModifier,      // modifier onlyOwner { ... }
    #[token("payable")]     KwPayable,       // payable fn deposit() { ... }
    #[token("view")]        KwView,          // view fn getBalance() -> u256
    #[token("emit")]        KwEmit,          // emit Transfer(a, b, amount);
    #[token("require")]     KwRequire,       // require(balance >= amount);
    #[token("revert")]      KwRevert,        // revert("Insufficient funds");
    #[token("mapping")]     KwMapping,       // mapping(address => uint256)
    #[token("address")]     KwAddress,       // address type (20-byte)
    #[token("wei")]         KwWei,           // wei literal suffix
    #[token("gwei")]        KwGwei,          // gwei literal suffix
    #[token("ether")]       KwEther,         // ether literal suffix

    // =================================================================
    // DOMAIN 5: FORMAL VERIFICATION / PROOF ASSISTANT
    // =================================================================
    // These turn Chronos into a language where you can write AND prove
    // your code correct. The compiler verifies every proof.
    
    #[token("theorem")]     KwTheorem,       // theorem add_comm: ... { ... }
    #[token("lemma")]       KwLemma,         // lemma helper: ... { ... }
    #[token("proof")]       KwProof,         // proof { intro x; ... }
    #[token("axiom")]       KwAxiom,         // axiom extensionality: ...
    #[token("forall")]      KwForall,        // forall x: Nat, P(x)
    #[token("exists")]      KwExists,        // exists x: Nat, P(x)
    #[token("assume")]      KwAssume,        // assume h: P;
    #[token("show")]        KwShow,          // show Q;
    #[token("qed")]         KwQed,           // end of proof
    #[token("sorry")]       KwSorry,         // unproven (generates warning)
    #[token("by")]          KwBy,            // proof by { tactic; ... }
    #[token("induction")]   KwInduction,     // induction on n { ... }
    #[token("invariant")]   KwInvariant,     // loop invariant x > 0;
    #[token("requires")]    KwRequires,      // fn precondition
    #[token("ensures")]     KwEnsures,       // fn postcondition
    #[token("decreases")]   KwDecreases,     // termination measure
    #[token("assert")]      KwAssert,        // compile-time verified assert
    #[token("verify")]      KwVerify,        // trigger SMT verification

    // =================================================================
    // DOMAIN 6: DATABASE INTERNALS
    // =================================================================
    
    #[token("table")]       KwTable,         // table Users { ... }
    #[token("index")]       KwIndex,         // index on Users(email);
    #[token("query")]       KwQuery,         // query plan compilation
    #[token("transaction")] KwTransaction,   // transaction { ... }
    #[token("commit")]      KwCommit,        // commit;
    #[token("rollback")]    KwRollback,      // rollback;
    #[token("isolation")]   KwIsolation,     // isolation = serializable;

    // =================================================================
    // DOMAIN 7: MACRO SYSTEM & DSL EMBEDDING
    // =================================================================
    
    #[token("macro")]       KwMacro,         // macro my_macro!(...) { ... }
    #[token("quote")]       KwQuote,         // quote { ... } (AST quoting)
    #[token("unquote")]     KwUnquote,       // $expr (splice into quoted AST)
    #[token("comptime")]    KwComptime,      // comptime { ... } (compile-time eval)
    #[token("embed")]       KwEmbed,         // embed sql { SELECT ... }

    // =================================================================
    // DOMAIN 8: MULTIMEDIA
    // =================================================================
    
    #[token("audio")]       KwAudio,         // audio { sample_rate: 44100 }
    #[token("video")]       KwVideo,         // video { codec: h265 }
    #[token("image")]       KwImage,         // image processing blocks
    #[token("render")]      KwRender,        // render { ... } (3D scene)
    #[token("shader")]      KwShaderKw,      // shader vertex { ... }
    #[token("scene")]       KwScene,         // scene { ... } (3D scene graph)
    #[token("material")]    KwMaterial,      // material PBR { ... }

    // =================================================================
    // DOMAIN 9: ROBOTICS & CONTROL SYSTEMS
    // =================================================================
    
    #[token("robot")]       KwRobot,         // robot ArmController { ... }
    #[token("controller")]  KwController,    // controller PID { ... }
    #[token("sensor")]      KwSensor,        // sensor imu: IMU { ... }
    #[token("actuator")]    KwActuator,      // actuator motor: Motor { ... }
    #[token("planner")]     KwPlanner,       // planner RRT { ... }
    #[token("filter")]      KwFilter,        // filter kalman: EKF { ... }
    #[token("pid")]         KwPid,           // pid(kp=1.0, ki=0.1, kd=0.01)

    // =================================================================
    // DOMAIN 10: QUANTUM COMPUTING
    // =================================================================
    
    #[token("qubit")]       KwQubit,         // qubit q;
    #[token("qreg")]        KwQreg,          // qreg q[5];  (quantum register)
    #[token("creg")]        KwCreg,          // creg c[5];  (classical register)
    #[token("gate")]        KwGate,          // gate h q[0];
    #[token("measure")]     KwMeasure,       // measure q[0] -> c[0];
    #[token("superposition")]KwSuperposition,// superposition { ... }
    #[token("entangle")]    KwEntangle,      // entangle q[0], q[1];
    #[token("circuit")]     KwCircuit,       // circuit Teleport { ... }
    #[token("oracle")]      KwOracle,        // oracle f { ... }

    // =================================================================
    // DOMAIN 11: GUI / UI
    // =================================================================
    
    #[token("widget")]      KwWidget,        // widget MyButton { ... }
    #[token("layout")]      KwLayout,        // layout column { ... }
    #[token("style")]       KwStyle,         // style { color: red; }
    #[token("bind")]        KwBind,          // bind text to model.name;
    #[token("animate")]     KwAnimate,       // animate opacity 0->1 in 300ms;
    #[token("component")]   KwComponent,     // component Card { ... }
    #[token("slot")]        KwSlot,          // slot header { ... }
    #[token("reactive")]    KwReactive,      // reactive var count = 0;

    // =================================================================
    // DOMAIN 12: GAME ENGINE
    // =================================================================
    
    #[token("entity")]      KwEntity,        // entity Player { ... }
    #[token("system")]      KwSystem,        // system Movement { ... }
    #[token("world")]       KwWorld,         // world GameWorld { ... }
    #[token("prefab")]      KwPrefab,        // prefab Bullet { ... }
    #[token("collider")]    KwCollider,      // collider sphere(1.0) { ... }
    #[token("rigidbody")]   KwRigidBody,     // rigidbody { mass: 10.0 }
    #[token("navmesh")]     KwNavmesh,       // navmesh { ... }
    #[token("tilemap")]     KwTilemap,       // tilemap { ... }

    // =================================================================
    // DOMAIN 13: BIOINFORMATICS
    // =================================================================
    
    #[token("dna")]         KwDna,           // dna seq = "ATCGATCG";
    #[token("rna")]         KwRna,           // rna seq = "AUCGAUCG";
    #[token("protein")]     KwProtein,       // protein seq = "MVLK...";
    #[token("genome")]      KwGenome,        // genome hg38 { ... }
    #[token("align")]       KwAlign,         // align(seq1, seq2, method=SW);
    #[token("fold")]        KwFold,          // fold protein { method: alphafold }

    // =================================================================
    // DOMAIN 14: FINANCIAL ENGINEERING
    // =================================================================
    
    #[token("option")]      KwOption_,       // option call { strike: 100 }
    #[token("portfolio")]   KwPortfolio,     // portfolio MyPort { ... }
    #[token("backtest")]    KwBacktest,      // backtest strategy { ... }
    #[token("order")]       KwOrder,         // order limit buy 100 @ 50.0;
    #[token("position")]    KwPosition,      // position tracking
    #[token("risk")]        KwRiskKw,        // risk var(0.99) { ... }
    #[token("hedge")]       KwHedge,         // hedge delta { ... }

    // =================================================================
    // DOMAIN 15: GEOSPATIAL / GIS
    // =================================================================
    
    #[token("point")]       KwPoint,         // point(lat, lon)
    #[token("polygon")]     KwPolygon,       // polygon [p1, p2, p3]
    #[token("geometry")]    KwGeometry,      // geometry operations
    #[token("crs")]         KwCrs,           // crs EPSG:4326
    #[token("spatial")]     KwSpatial,       // spatial index / query
    #[token("raster")]      KwRaster,        // raster DEM { ... }

    // =================================================================
    // DOMAIN 16: OBSERVABILITY & TESTING
    // =================================================================
    
    #[token("trace")]       KwTrace,         // trace span "operation" { ... }
    #[token("metric")]      KwMetric,        // metric counter requests_total;
    #[token("log")]         KwLog,           // log info "message";
    #[token("test")]        KwTest,          // test "should add" { ... }
    #[token("bench")]       KwBench,         // bench "sort perf" { ... }
    #[token("fuzz")]        KwFuzz,          // fuzz parse_input { ... }
    #[token("property")]    KwProperty,      // property "commutative" { ... }
    #[token("expect")]      KwExpect,        // expect(result == 42);
    #[token("mock")]        KwMock,          // mock HttpClient { ... }

    // =================================================================
    // DOMAIN 17: FFI & HOT RELOAD
    // =================================================================
    
    #[token("extern")]      KwExtern,        // extern "C" fn malloc(...);
    #[token("foreign")]     KwForeign,       // foreign python { ... }
    #[token("link")]        KwLink,          // link "libfoo.so";
    #[token("hotreload")]   KwHotReload,     // hotreload module game_logic;
    #[token("live")]        KwLive,          // live fn update() { ... }

    // =================================================================
    // DOMAIN 18: I18N & L10N
    // =================================================================
    
    #[token("locale")]      KwLocale,        // locale "en-US" { ... }
    #[token("translate")]   KwTranslate,     // translate "hello" { ... }
    #[token("plural")]      KwPlural,        // plural(count) { ... }
    #[token("format")]      KwFormat,        // format currency(amount);

    // =================================================================
    // DOMAIN 19: DOCUMENTATION
    // =================================================================
    // Note: Doc comments (///) are already in the lexer. These keywords
    // support literate programming and structured documentation.
    
    #[token("doc")]         KwDoc,           // doc section "Architecture" { ... }
    #[token("notebook")]    KwNotebook,      // notebook { ... } (Jupyter-like)

    // =================================================================
    // DOMAIN 20: SIMULATION (extended from existing)
    // =================================================================
    
    #[token("simulation")]  KwSimulation,    // simulation FluidFlow { ... }
    #[token("mesh")]        KwMesh,          // mesh triangular { ... }
    #[token("boundary")]    KwBoundary,      // boundary wall { no_slip: true }
    #[token("solve")]       KwSolve,         // solve navier_stokes { ... }
    #[token("timestep")]    KwTimestep,      // timestep adaptive { ... }
    #[token("domain")]      KwDomain,        // domain 3d { ... }
    #[token("physics")]     KwPhysics,       // physics incompressible_ns { ... }
    #[token("fem")]         KwFem,           // fem linear_static { ... }

    // =================================================================
    // DOMAIN 21: DATA PIPELINE (extended from existing)
    // =================================================================
    
    #[token("pipeline")]    KwPipelineKw,    // pipeline ETL_Daily { ... }
    #[token("source")]      KwSource,        // source csv("data.csv") { ... }
    #[token("sink")]        KwSink,          // sink parquet("output/") { ... }
    #[token("transform")]   KwTransform,     // transform { filter(...) }
    #[token("scrape")]      KwScrape,        // scrape "https://..." { ... }
    #[token("validate")]    KwValidate,      // validate { not_null(col) }
    #[token("schedule")]    KwSchedule,      // schedule cron("0 * * * *");

    // =================================================================
    // DOMAIN 22: SECURITY (extended from existing)
    // =================================================================
    
    #[token("secure")]      KwSecure,        // #![secure] at file level
    #[token("encrypt")]     KwEncrypt,       // encrypt aes256gcm { ... }
    #[token("sign")]        KwSign,          // sign ed25519 message;
    #[token("audit")]       KwAudit,         // audit { ... } block
    #[token("capability")]  KwCapability,    // capability<File, Read>
    #[token("taint")]       KwTaint,         // taint(user_input)
    #[token("sanitize")]    KwSanitize,      // sanitize html(input)

    // =================================================================
    // TYPE KEYWORDS FOR NEW DOMAINS
    // =================================================================
    // These are type-level keywords that the type parser needs to recognize.
    
    #[token("u256")]        TyU256,          // Blockchain 256-bit uint
    #[token("i256")]        TyI256,
    #[token("u512")]        TyU512,
    #[token("decimal")]     TyDecimal,       // Exact decimal arithmetic
    #[token("complex")]     TyComplex,       // Complex number type
    #[token("quaternion")]  TyQuaternion,    // Quaternion type
    #[token("ratio")]       TyRatio,         // Exact rational number
    #[token("fixed")]       TyFixed,         // Fixed-point type
    #[token("symbol")]      TySymbol,        // Symbolic expression type
    #[token("bigint")]      TyBigInt,        // Arbitrary precision integer
    #[token("bigfloat")]    TyBigFloat,      // Arbitrary precision float

    // --- Unit-of-measure suffixes (compile-time dimensional analysis) ---
    // These let you write: let distance: f64<meter> = 5.0;
    //                      let force = mass * acceleration; // auto-deduced: newton
    #[token("meter")]       UnitMeter,
    #[token("kilogram")]    UnitKilogram,
    #[token("second")]      UnitSecond,
    #[token("ampere")]      UnitAmpere,
    #[token("kelvin")]      UnitKelvin,
    #[token("mole")]        UnitMole,
    #[token("candela")]     UnitCandela,
    #[token("radian")]      UnitRadian,

    // --- Special operator tokens for new domains ---
    #[token("|->")]  KetArrow,       // Quantum state notation: |0⟩ |-> |1⟩
    #[token("<-")]   LeftArrow,      // Channel receive: msg <- channel
    #[token("~>")]   TildeArrow,     // Leads-to in temporal logic
    #[token("#>")]   HashArrow,      // Pipeline composition: source #> transform #> sink
    #[token("==>")]  LongFatArrow,   // Proof implication
    #[token("<=>")]  Biconditional,  // Logical equivalence / bidirectional
}

// ============================================================================
// TOKEN CLASSIFICATION EXTENSIONS
// ============================================================================
// These methods must be added to the existing Token impl block to help the
// parser recognize which tokens can start domain-specific declarations.

/// Classification helpers to add to the existing Token impl.
pub trait TokenClassificationExt {
    /// Returns true if this token starts a domain-specific declaration.
    fn is_domain_declaration_start(&self) -> bool;
    
    /// Returns true if this token is a unit-of-measure keyword.
    fn is_unit_keyword(&self) -> bool;
    
    /// Returns true if this token starts a proof/verification construct.
    fn is_proof_start(&self) -> bool;
    
    /// Returns true if this token starts a quantum computing construct.
    fn is_quantum_start(&self) -> bool;
}

// In the real implementation, these would be match arms in the Token enum's
// impl block. Here we show the logic:

/*
impl Token {
    pub fn is_domain_declaration_start(&self) -> bool {
        matches!(self,
            // Networking
            Token::KwProtocol | Token::KwEndpoint | Token::KwRoute |
            // OS
            Token::KwDriver | Token::KwInterrupt | Token::KwSyscall |
            // Distributed
            Token::KwConsensus | Token::KwCrdt | Token::KwSaga |
            // Blockchain
            Token::KwContract |
            // Proof
            Token::KwTheorem | Token::KwLemma | Token::KwAxiom |
            // Database
            Token::KwTable | Token::KwIndex |
            // Macros
            Token::KwMacro | Token::KwComptime | Token::KwEmbed |
            // Multimedia
            Token::KwAudio | Token::KwVideo | Token::KwScene |
            // Robotics
            Token::KwRobot | Token::KwController |
            // Quantum
            Token::KwCircuit | Token::KwQubit |
            // GUI
            Token::KwWidget | Token::KwComponent |
            // Game
            Token::KwEntity | Token::KwSystem | Token::KwWorld |
            // Bio
            Token::KwGenome |
            // Finance
            Token::KwPortfolio | Token::KwBacktest |
            // Geo
            Token::KwSpatial |
            // Observability
            Token::KwTest | Token::KwBench | Token::KwFuzz |
            // FFI
            Token::KwExtern | Token::KwForeign |
            // Simulation
            Token::KwSimulation | Token::KwFem |
            // Pipeline
            Token::KwPipelineKw |
            // Security
            Token::KwAudit
        )
    }
    
    pub fn is_unit_keyword(&self) -> bool {
        matches!(self,
            Token::UnitMeter | Token::UnitKilogram | Token::UnitSecond |
            Token::UnitAmpere | Token::UnitKelvin | Token::UnitMole |
            Token::UnitCandela | Token::UnitRadian
        )
    }
    
    pub fn is_proof_start(&self) -> bool {
        matches!(self,
            Token::KwTheorem | Token::KwLemma | Token::KwAxiom |
            Token::KwProof | Token::KwForall | Token::KwExists |
            Token::KwInvariant | Token::KwRequires | Token::KwEnsures |
            Token::KwVerify | Token::KwAssert
        )
    }
    
    pub fn is_quantum_start(&self) -> bool {
        matches!(self,
            Token::KwQubit | Token::KwQreg | Token::KwCreg |
            Token::KwGate | Token::KwMeasure | Token::KwCircuit |
            Token::KwOracle | Token::KwEntangle | Token::KwSuperposition
        )
    }
}
*/

// ============================================================================
// BINDING POWER EXTENSIONS
// ============================================================================
// New operators need precedence levels for the Pratt parser.

/*
Add to Token::infix_binding_power():

    // Pipeline composition: source #> transform #> sink
    Token::HashArrow => Some((3, 4)),     // Same as PipeForward
    
    // Channel operations: msg <- channel
    Token::LeftArrow => Some((2, 1)),     // Right-associative like assignment
    
    // Proof implication: P ==> Q
    Token::LongFatArrow => Some((3, 4)),
    
    // Logical biconditional: P <=> Q
    Token::Biconditional => Some((3, 4)),
    
    // Quantum state transition: |0⟩ |-> |1⟩
    Token::KetArrow => Some((3, 4)),
    
    // Temporal leads-to: P ~> Q
    Token::TildeArrow => Some((3, 4)),
*/
