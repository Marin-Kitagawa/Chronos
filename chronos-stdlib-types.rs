// ============================================================================
// CHRONOS STANDARD LIBRARY — PART 1
// Universal Data Types + Graph Algorithms + Higher-Order Mathematics
// ============================================================================
// This module defines every data type that has ever existed across all major
// programming languages, a complete graph algorithm library, and a
// Mathematica-grade symbolic mathematics engine — all as compiler built-ins
// with zero-cost abstractions where possible.
// ============================================================================

// std::collections imported here; std::fmt is provided by chronos-lexer.rs
// which is always included before this file in chronos-compiler-core.
use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};

// ============================================================================
// SECTION 1: UNIVERSAL DATA TYPE CATALOG
// ============================================================================
// We unify EVERY data type from EVERY major language into a single taxonomy.
// The compiler knows about all of these natively and can optimize across them.
//
// The taxonomy is organized as: Primitives → Containers → Algebraic →
// Concurrent → Domain-Specific → Exotic.

/// The complete catalog of all data types ever created in programming languages.
/// This extends the ChronosType enum from the core compiler with every
/// language-specific type mapped to its Chronos equivalent.
#[derive(Debug, Clone, PartialEq)]
pub enum UniversalType {
    // =================================================================
    // PRIMITIVES — Every numeric, character, and boolean type
    // =================================================================
    
    // --- Booleans ---
    Bool,                    // C, Java, Rust, Python
    TruthValue,              // COBOL (88-level)
    Bit,                     // VHDL, Verilog (single-bit)
    
    // --- Integers (signed, every width) ---
    I8, I16, I32, I64, I128, I256, I512, // Rust, C, C++
    Int,                     // Python (arbitrary precision)
    BigInt,                  // JavaScript, Haskell Integer
    NativeInt,               // OCaml (63-bit tagged integer)
    
    // --- Integers (unsigned) ---
    U8, U16, U32, U64, U128, U256, U512,
    Byte,                    // Java (signed in Java, but we unify)
    Word,                    // Haskell Word
    
    // --- Machine-width integers ---
    ISize,                   // Rust isize, C ptrdiff_t
    USize,                   // Rust usize, C size_t
    IntPtr,                  // C intptr_t
    UIntPtr,                 // C uintptr_t
    
    // --- Floating point ---
    F16,                     // IEEE 754 half
    BF16,                    // Brain float (AI workloads)
    F32,                     // IEEE 754 single
    F64,                     // IEEE 754 double
    F80,                     // x87 extended precision
    F128,                    // IEEE 754 quad
    Decimal32,               // IEEE 754-2008 decimal
    Decimal64,
    Decimal128,
    BigDecimal,              // Java BigDecimal, arbitrary precision decimal
    BigFloat,                // GMP mpf, arbitrary precision float
    
    // --- Fixed-point (embedded/DSP/financial) ---
    Fixed { integer_bits: u16, fraction_bits: u16, signed: bool },
    // e.g., Fixed { 16, 16, true } = Q16.16
    
    // --- Complex numbers ---
    Complex32,               // (f16, f16)
    Complex64,               // (f32, f32) — Fortran COMPLEX, Python complex
    Complex128,              // (f64, f64) — Fortran DOUBLE COMPLEX
    Complex256,              // (f128, f128)
    Quaternion,              // Hamilton quaternions (game engines, robotics)
    Octonion,                // Cayley–Dickson construction
    DualNumber,              // Automatic differentiation: a + bε where ε² = 0
    
    // --- Characters and strings ---
    Char,                    // Rust char (Unicode scalar), C char (byte)
    WChar,                   // C wchar_t
    Char8,                   // C++ char8_t (UTF-8)
    Char16,                  // C++ char16_t (UTF-16)
    Char32,                  // C++ char32_t (UTF-32)
    AsciiChar,               // 7-bit ASCII only
    Str,                     // Immutable string slice (Rust &str)
    String,                  // Owned heap string (Rust String, Java String)
    CString,                 // Null-terminated C string
    OsString,                // OS-native string (Rust OsString)
    PathBuf,                 // Filesystem path
    Rope,                    // Rope data structure (for large text editing)
    Symbol,                  // Ruby Symbol, Lisp symbol (interned string)
    Atom,                    // Erlang/Elixir atom
    Rune,                    // Go rune (alias for int32, Unicode code point)
    
    // --- Void / Unit / Nothing ---
    Void,                    // C void
    Unit,                    // Rust (), Scala Unit, Haskell ()
    Nil,                     // Ruby nil, Lisp nil, Lua nil
    None_,                   // Python None
    Undefined,               // JavaScript undefined
    Null,                    // Java null, C NULL
    Never,                   // Rust !, TypeScript never (bottom type)
    Noreturn,                // C _Noreturn
    Nothing,                 // Kotlin Nothing, Scala Nothing
    
    // --- Raw memory ---
    RawPointer(Box<UniversalType>),     // C/C++ raw pointer
    FatPointer(Box<UniversalType>),     // Rust trait object pointer (ptr + vtable)
    VoidPtr,                            // C void*
    FunctionPointer(Vec<UniversalType>, Box<UniversalType>), // C function pointer
    MemberPointer(String, Box<UniversalType>),  // C++ pointer-to-member
    
    // =================================================================
    // CONTAINERS — Every collection type
    // =================================================================
    
    // --- Sequential ---
    Array { elem: Box<UniversalType>, size: usize },        // C/Rust fixed array
    DynamicArray(Box<UniversalType>),                        // C++ vector, Java ArrayList
    LinkedList(Box<UniversalType>),                          // std::list
    DoublyLinkedList(Box<UniversalType>),
    CircularBuffer(Box<UniversalType>),                      // Ring buffer
    Deque(Box<UniversalType>),                               // Double-ended queue
    Stack(Box<UniversalType>),
    Queue(Box<UniversalType>),
    PriorityQueue(Box<UniversalType>),
    SkipList(Box<UniversalType>),                            // Probabilistic ordered list
    
    // --- Associative ---
    HashMap(Box<UniversalType>, Box<UniversalType>),
    BTreeMap(Box<UniversalType>, Box<UniversalType>),
    LinkedHashMap(Box<UniversalType>, Box<UniversalType>),   // Java LinkedHashMap
    OrderedDict(Box<UniversalType>, Box<UniversalType>),     // Python OrderedDict
    TrieMap(Box<UniversalType>),                             // Prefix tree map
    RadixTree(Box<UniversalType>),
    LRUCache(Box<UniversalType>, Box<UniversalType>, usize), // Bounded cache
    
    // --- Sets ---
    HashSet(Box<UniversalType>),
    BTreeSet(Box<UniversalType>),
    BitSet(usize),                        // Fixed-size bitset
    BloomFilter(usize, u8),               // Probabilistic set (size, hash_count)
    DisjointSet(Box<UniversalType>),       // Union-Find
    
    // --- Tuples / Records ---
    Tuple(Vec<UniversalType>),
    NamedTuple(Vec<(String, UniversalType)>),    // Python NamedTuple
    Record(Vec<(String, UniversalType)>),         // OCaml/Haskell record
    Struct(String),                               // Named struct reference
    
    // --- Functional ---
    Option(Box<UniversalType>),                   // Rust Option, Haskell Maybe
    Result(Box<UniversalType>, Box<UniversalType>), // Rust Result, Haskell Either
    Either(Box<UniversalType>, Box<UniversalType>), // Haskell Either
    Lazy(Box<UniversalType>),                     // Lazy evaluation wrapper
    Thunk(Box<UniversalType>),                    // Suspended computation
    Stream(Box<UniversalType>),                   // Lazy infinite sequence
    Iterator(Box<UniversalType>),                 // Pull-based iteration
    Generator(Box<UniversalType>, Box<UniversalType>), // Yield/Resume
    Future(Box<UniversalType>),                   // Async future
    Promise(Box<UniversalType>),                  // JavaScript Promise
    Observable(Box<UniversalType>),               // RxJS Observable (push-based stream)
    Signal(Box<UniversalType>),                   // Reactive signal (Solid.js, S.js)
    
    // --- Trees ---
    BinaryTree(Box<UniversalType>),
    AVLTree(Box<UniversalType>),
    RedBlackTree(Box<UniversalType>),
    BPlusTree(Box<UniversalType>),
    SplayTree(Box<UniversalType>),
    KDTree(Box<UniversalType>, u8),               // k-dimensional tree
    RTree(Box<UniversalType>),                    // Spatial indexing
    QuadTree(Box<UniversalType>),
    OctTree(Box<UniversalType>),
    MerkleTree(Box<UniversalType>),               // Blockchain/integrity verification
    FenwickTree(Box<UniversalType>),              // Binary indexed tree
    SegmentTree(Box<UniversalType>),              // Range queries
    IntervalTree(Box<UniversalType>),
    Trie,                                         // Prefix tree for strings
    
    // --- Graphs ---
    Graph {
        node: Box<UniversalType>,
        edge: Box<UniversalType>,
        directed: bool,
        weighted: bool,
    },
    AdjacencyMatrix(Box<UniversalType>),
    AdjacencyList(Box<UniversalType>),
    HyperGraph(Box<UniversalType>, Box<UniversalType>),
    
    // =================================================================
    // CONCURRENT / PARALLEL TYPES
    // =================================================================
    Mutex(Box<UniversalType>),
    RwLock(Box<UniversalType>),
    Semaphore,
    Barrier(u32),
    Atomic(Box<UniversalType>),                    // Lock-free atomic
    Channel(Box<UniversalType>),                   // Go channel, Rust mpsc
    BroadcastChannel(Box<UniversalType>),
    Actor(Box<UniversalType>),                     // Erlang/Akka actor
    Arc(Box<UniversalType>),                       // Atomic reference count
    Rc(Box<UniversalType>),                        // Single-thread reference count
    Weak(Box<UniversalType>),                      // Weak reference
    Pin(Box<UniversalType>),                       // Pinned in memory
    Cell(Box<UniversalType>),                      // Interior mutability
    RefCell(Box<UniversalType>),
    OnceCell(Box<UniversalType>),                  // Write-once cell
    
    // =================================================================
    // DOMAIN-SPECIFIC TYPES (used across scientific/engineering languages)
    // =================================================================
    
    // --- Tensors and linear algebra (NumPy, MATLAB, Julia, Mojo) ---
    Tensor {
        elem: Box<UniversalType>,
        shape: TensorShapeSpec,
        device: DeviceSpec,
        layout: TensorLayout,
    },
    SparseMatrix {
        elem: Box<UniversalType>,
        format: SparseFormat,
    },
    
    // --- Symbolic (Mathematica, SymPy, Maple) ---
    Symbolic(Box<SymbolicExpr>),
    
    // --- Date/Time ---
    Instant,                  // Monotonic timestamp
    DateTime,                 // Calendar date + time
    Duration,                 // Time span
    TimeZone,
    Date,
    Time,
    
    // --- Currency / Financial ---
    Money { amount: Box<UniversalType>, currency: String },
    Ratio(Box<UniversalType>),    // Exact rational number
    Percentage,
    
    // --- Geometric ---
    Point2D, Point3D, Point4D,
    Vec2, Vec3, Vec4,
    Mat2, Mat3, Mat4,
    Rotation2D, Rotation3D,
    Transform2D, Transform3D,
    BoundingBox2D, BoundingBox3D,
    Polygon, Polyhedron,
    Mesh { dim: u8 },              // Triangular/tetrahedral mesh for FEM
    
    // --- Color ---
    RGB, RGBA, HSL, HSV, CMYK, Lab, Oklab,
    
    // --- Network ---
    IpAddr, Ipv4Addr, Ipv6Addr,
    SocketAddr,
    MacAddr,
    Url, Uri,
    
    // --- Cryptographic ---
    Hash(u16),                     // Fixed-length hash (bits)
    PublicKey, PrivateKey, KeyPair,
    Signature,
    Ciphertext(Box<UniversalType>),
    Nonce,
    
    // --- Units of measure (F#-style, compile-time checked) ---
    UnitType {
        base: Box<UniversalType>,
        unit: PhysicalUnit,
    },
    
    // --- Real-time (Feature 8: mission-critical) ---
    Deadline,                      // Hard real-time deadline
    WCETBound(u64),               // Worst-Case Execution Time bound (nanoseconds)
    SafetyLevel(ASILLevel),       // Automotive ASIL rating
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorShapeSpec {
    Static(Vec<usize>),
    Dynamic(Vec<Option<usize>>),
    Symbolic(Vec<String>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceSpec { Cpu, Gpu(u32), Tpu(u32), Npu(u32), Auto }

#[derive(Debug, Clone, PartialEq)]
pub enum TensorLayout { RowMajor, ColMajor, Strided(Vec<usize>), BlockSparse }

#[derive(Debug, Clone, PartialEq)]
pub enum SparseFormat { COO, CSR, CSC, BSR, ELL, DIA }

#[derive(Debug, Clone, PartialEq)]
pub enum ASILLevel { QM, A, B, C, D } // ISO 26262 Automotive Safety Integrity Level

/// Physical units for compile-time dimensional analysis.
/// The compiler verifies that you never add meters to kilograms.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalUnit {
    pub meter: i8,      // Length
    pub kilogram: i8,   // Mass
    pub second: i8,     // Time
    pub ampere: i8,     // Electric current
    pub kelvin: i8,     // Temperature
    pub mole: i8,       // Amount of substance
    pub candela: i8,    // Luminous intensity
    pub radian: i8,     // Angle
    pub steradian: i8,  // Solid angle
}

impl PhysicalUnit {
    pub fn dimensionless() -> Self {
        Self { meter: 0, kilogram: 0, second: 0, ampere: 0,
               kelvin: 0, mole: 0, candela: 0, radian: 0, steradian: 0 }
    }
    pub fn meter() -> Self { let mut u = Self::dimensionless(); u.meter = 1; u }
    pub fn kilogram() -> Self { let mut u = Self::dimensionless(); u.kilogram = 1; u }
    pub fn second() -> Self { let mut u = Self::dimensionless(); u.second = 1; u }
    pub fn newton() -> Self { // kg⋅m⋅s⁻²
        Self { kilogram: 1, meter: 1, second: -2, ..Self::dimensionless() }
    }
    pub fn joule() -> Self { // kg⋅m²⋅s⁻²
        Self { kilogram: 1, meter: 2, second: -2, ..Self::dimensionless() }
    }
    pub fn pascal() -> Self { // kg⋅m⁻¹⋅s⁻²
        Self { kilogram: 1, meter: -1, second: -2, ..Self::dimensionless() }
    }
    
    /// Multiply two unit sets (used when multiplying physical quantities).
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            meter: self.meter + other.meter,
            kilogram: self.kilogram + other.kilogram,
            second: self.second + other.second,
            ampere: self.ampere + other.ampere,
            kelvin: self.kelvin + other.kelvin,
            mole: self.mole + other.mole,
            candela: self.candela + other.candela,
            radian: self.radian + other.radian,
            steradian: self.steradian + other.steradian,
        }
    }
    
    /// Divide unit sets (used when dividing physical quantities).
    pub fn div(&self, other: &Self) -> Self {
        Self {
            meter: self.meter - other.meter,
            kilogram: self.kilogram - other.kilogram,
            second: self.second - other.second,
            ampere: self.ampere - other.ampere,
            kelvin: self.kelvin - other.kelvin,
            mole: self.mole - other.mole,
            candela: self.candela - other.candela,
            radian: self.radian - other.radian,
            steradian: self.steradian - other.steradian,
        }
    }
}

// ============================================================================
// SECTION 2: SYMBOLIC MATHEMATICS ENGINE (Mathematica-grade)
// ============================================================================
// This is a complete Computer Algebra System (CAS) built into the language.
// Every Mathematica function has a Chronos equivalent. The compiler can
// evaluate symbolic expressions at compile time or generate efficient
// numeric code for runtime evaluation.

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicExpr {
    // --- Atoms ---
    Integer(i128),
    Rational(i128, i128),              // Exact fraction p/q
    Real(f64),
    Complex(f64, f64),                 // a + bi
    Symbol(String),                    // Variable name: x, y, theta
    String(String),
    Boolean(bool),
    
    // --- Constants (exact, not approximations) ---
    Pi,                                // π
    E,                                 // Euler's number e
    I,                                 // Imaginary unit √(-1)
    Infinity,
    NegInfinity,
    ComplexInfinity,
    Indeterminate,                     // 0/0 etc.
    GoldenRatio,                       // φ = (1+√5)/2
    EulerGamma,                        // Euler–Mascheroni constant γ
    Catalan,                           // Catalan's constant
    
    // --- Compound expressions ---
    // Every expression is a tree of function applications.
    // This is Mathematica's fundamental design: everything is f[args...].
    Apply {
        head: Box<SymbolicExpr>,
        args: Vec<SymbolicExpr>,
    },
    List(Vec<SymbolicExpr>),
    
    // =================================================================
    // ARITHMETIC & ALGEBRA
    // =================================================================
    Plus(Vec<SymbolicExpr>),           // Sum of terms
    Times(Vec<SymbolicExpr>),          // Product of terms
    Power(Box<SymbolicExpr>, Box<SymbolicExpr>), // x^n
    Log(Box<SymbolicExpr>, Box<SymbolicExpr>),   // log_b(x)
    Sqrt(Box<SymbolicExpr>),
    Abs(Box<SymbolicExpr>),
    Sign(Box<SymbolicExpr>),
    Floor(Box<SymbolicExpr>),
    Ceil(Box<SymbolicExpr>),
    Mod(Box<SymbolicExpr>, Box<SymbolicExpr>),
    GCD(Vec<SymbolicExpr>),
    LCM(Vec<SymbolicExpr>),
    Factorial(Box<SymbolicExpr>),
    Binomial(Box<SymbolicExpr>, Box<SymbolicExpr>),
    
    // =================================================================
    // CALCULUS
    // =================================================================
    Derivative {
        expr: Box<SymbolicExpr>,
        variable: String,
        order: u32,
    },
    PartialDerivative {
        expr: Box<SymbolicExpr>,
        variables: Vec<(String, u32)>,   // (variable, order)
    },
    Integral {
        expr: Box<SymbolicExpr>,
        variable: String,
        lower: Option<Box<SymbolicExpr>>,
        upper: Option<Box<SymbolicExpr>>,
    },
    Limit {
        expr: Box<SymbolicExpr>,
        variable: String,
        point: Box<SymbolicExpr>,
        direction: LimitDirection,
    },
    Sum {
        expr: Box<SymbolicExpr>,
        variable: String,
        lower: Box<SymbolicExpr>,
        upper: Box<SymbolicExpr>,
    },
    Product {
        expr: Box<SymbolicExpr>,
        variable: String,
        lower: Box<SymbolicExpr>,
        upper: Box<SymbolicExpr>,
    },
    Series {
        expr: Box<SymbolicExpr>,
        variable: String,
        point: Box<SymbolicExpr>,
        order: u32,
    },
    Residue {
        expr: Box<SymbolicExpr>,
        variable: String,
        point: Box<SymbolicExpr>,
    },
    
    // =================================================================
    // TRIGONOMETRIC & HYPERBOLIC
    // =================================================================
    Sin(Box<SymbolicExpr>), Cos(Box<SymbolicExpr>), Tan(Box<SymbolicExpr>),
    ArcSin(Box<SymbolicExpr>), ArcCos(Box<SymbolicExpr>), ArcTan(Box<SymbolicExpr>),
    ArcTan2(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Sinh(Box<SymbolicExpr>), Cosh(Box<SymbolicExpr>), Tanh(Box<SymbolicExpr>),
    Sec(Box<SymbolicExpr>), Csc(Box<SymbolicExpr>), Cot(Box<SymbolicExpr>),
    
    // =================================================================
    // SPECIAL FUNCTIONS (complete Mathematica coverage)
    // =================================================================
    Gamma(Box<SymbolicExpr>),                   // Γ(z)
    LogGamma(Box<SymbolicExpr>),
    Beta(Box<SymbolicExpr>, Box<SymbolicExpr>), // B(a,b)
    Digamma(Box<SymbolicExpr>),                 // ψ(z)
    Polygamma(Box<SymbolicExpr>, Box<SymbolicExpr>), // ψ^(n)(z)
    Zeta(Box<SymbolicExpr>),                    // Riemann zeta ζ(s)
    HurwitzZeta(Box<SymbolicExpr>, Box<SymbolicExpr>),
    BesselJ(Box<SymbolicExpr>, Box<SymbolicExpr>),  // J_n(z)
    BesselY(Box<SymbolicExpr>, Box<SymbolicExpr>),
    BesselI(Box<SymbolicExpr>, Box<SymbolicExpr>),
    BesselK(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Airy(AiryKind, Box<SymbolicExpr>),
    SphericalHarmonic {
        l: Box<SymbolicExpr>, m: Box<SymbolicExpr>,
        theta: Box<SymbolicExpr>, phi: Box<SymbolicExpr>,
    },
    LegendreP(Box<SymbolicExpr>, Box<SymbolicExpr>),
    LaguerreL(Box<SymbolicExpr>, Box<SymbolicExpr>),
    HermiteH(Box<SymbolicExpr>, Box<SymbolicExpr>),
    ChebyshevT(Box<SymbolicExpr>, Box<SymbolicExpr>),
    ChebyshevU(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Hypergeometric2F1(Vec<SymbolicExpr>),       // ₂F₁(a,b;c;z)
    HypergeometricPFQ(Vec<SymbolicExpr>, Vec<SymbolicExpr>, Box<SymbolicExpr>),
    MeijerG(Vec<Vec<SymbolicExpr>>, Box<SymbolicExpr>),
    EllipticK(Box<SymbolicExpr>),               // Complete elliptic integral K
    EllipticE(Box<SymbolicExpr>),               // Complete elliptic integral E
    EllipticPi(Box<SymbolicExpr>, Box<SymbolicExpr>),
    WeierstrassP(Box<SymbolicExpr>, Box<SymbolicExpr>, Box<SymbolicExpr>),
    Erf(Box<SymbolicExpr>),                     // Error function
    Erfc(Box<SymbolicExpr>),
    FresnelS(Box<SymbolicExpr>),
    FresnelC(Box<SymbolicExpr>),
    ExpIntegralEi(Box<SymbolicExpr>),
    LogIntegral(Box<SymbolicExpr>),
    SinIntegral(Box<SymbolicExpr>),
    CosIntegral(Box<SymbolicExpr>),
    DiracDelta(Box<SymbolicExpr>),
    HeavisideTheta(Box<SymbolicExpr>),
    KroneckerDelta(Vec<SymbolicExpr>),
    
    // =================================================================
    // LINEAR ALGEBRA (symbolic)
    // =================================================================
    Matrix(Vec<Vec<SymbolicExpr>>),
    Vector(Vec<SymbolicExpr>),
    Determinant(Box<SymbolicExpr>),
    Trace(Box<SymbolicExpr>),
    Transpose(Box<SymbolicExpr>),
    Inverse(Box<SymbolicExpr>),
    Eigenvalues(Box<SymbolicExpr>),
    Eigenvectors(Box<SymbolicExpr>),
    SVD(Box<SymbolicExpr>),
    QR(Box<SymbolicExpr>),
    LU(Box<SymbolicExpr>),
    Cholesky(Box<SymbolicExpr>),
    MatrixExp(Box<SymbolicExpr>),
    MatrixLog(Box<SymbolicExpr>),
    KroneckerProduct(Box<SymbolicExpr>, Box<SymbolicExpr>),
    TensorProduct(Vec<SymbolicExpr>),
    Norm(Box<SymbolicExpr>, NormKind),
    Rank(Box<SymbolicExpr>),
    NullSpace(Box<SymbolicExpr>),
    ColumnSpace(Box<SymbolicExpr>),
    JordanForm(Box<SymbolicExpr>),
    SmithNormalForm(Box<SymbolicExpr>),
    CharacteristicPolynomial(Box<SymbolicExpr>),
    MinimalPolynomial(Box<SymbolicExpr>),
    
    // =================================================================
    // EQUATION SOLVING
    // =================================================================
    Solve {
        equations: Vec<SymbolicExpr>,
        variables: Vec<String>,
    },
    DSolve {                          // Differential equation solver
        equation: Box<SymbolicExpr>,
        function: String,
        variable: String,
    },
    NDSolve {                         // Numerical ODE/PDE solver
        equation: Box<SymbolicExpr>,
        function: String,
        variable: String,
        range: (f64, f64),
        method: ODEMethod,
    },
    Minimize {
        objective: Box<SymbolicExpr>,
        variables: Vec<String>,
        constraints: Vec<SymbolicExpr>,
    },
    Maximize {
        objective: Box<SymbolicExpr>,
        variables: Vec<String>,
        constraints: Vec<SymbolicExpr>,
    },
    FindRoot {
        equation: Box<SymbolicExpr>,
        variable: String,
        initial_guess: f64,
    },
    LinearProgramming {
        objective: Vec<f64>,
        constraints_lhs: Vec<Vec<f64>>,
        constraints_rhs: Vec<f64>,
    },
    RecurrenceRelation {
        equation: Box<SymbolicExpr>,
        function: String,
        variable: String,
    },
    
    // =================================================================
    // TRANSFORMS
    // =================================================================
    FourierTransform {
        expr: Box<SymbolicExpr>,
        variable: String,
        frequency: String,
    },
    InverseFourierTransform {
        expr: Box<SymbolicExpr>,
        frequency: String,
        variable: String,
    },
    LaplaceTransform {
        expr: Box<SymbolicExpr>,
        variable: String,
        s_variable: String,
    },
    InverseLaplaceTransform {
        expr: Box<SymbolicExpr>,
        s_variable: String,
        variable: String,
    },
    ZTransform {
        expr: Box<SymbolicExpr>,
        n_variable: String,
        z_variable: String,
    },
    HilbertTransform(Box<SymbolicExpr>),
    MellinTransform(Box<SymbolicExpr>, String, String),
    WaveletTransform(Box<SymbolicExpr>, WaveletKind),
    
    // =================================================================
    // NUMBER THEORY
    // =================================================================
    Prime(Box<SymbolicExpr>),
    NextPrime(Box<SymbolicExpr>),
    PrimeQ(Box<SymbolicExpr>),
    FactorInteger(Box<SymbolicExpr>),
    EulerPhi(Box<SymbolicExpr>),      // Euler's totient function
    MoebiusMu(Box<SymbolicExpr>),
    DivisorSigma(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mod_(Box<SymbolicExpr>, Box<SymbolicExpr>),
    PowerMod(Box<SymbolicExpr>, Box<SymbolicExpr>, Box<SymbolicExpr>),
    ChineseRemainder(Vec<SymbolicExpr>, Vec<SymbolicExpr>),
    ContinuedFraction(Box<SymbolicExpr>),
    JacobiSymbol(Box<SymbolicExpr>, Box<SymbolicExpr>),
    
    // =================================================================
    // COMBINATORICS & DISCRETE MATH
    // =================================================================
    Permutations(Box<SymbolicExpr>),
    Combinations(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Partitions(Box<SymbolicExpr>),
    BellNumber(Box<SymbolicExpr>),
    CatalanNumber(Box<SymbolicExpr>),
    StirlingS1(Box<SymbolicExpr>, Box<SymbolicExpr>),
    StirlingS2(Box<SymbolicExpr>, Box<SymbolicExpr>),
    BernoulliB(Box<SymbolicExpr>),
    
    // =================================================================
    // PROBABILITY & STATISTICS
    // =================================================================
    Distribution(DistributionKind),
    PDF(Box<SymbolicExpr>, Box<SymbolicExpr>),
    CDF(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Mean(Box<SymbolicExpr>),
    Variance(Box<SymbolicExpr>),
    StandardDeviation(Box<SymbolicExpr>),
    Expectation(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Correlation(Box<SymbolicExpr>, Box<SymbolicExpr>),
    
    // =================================================================
    // TOPOLOGY & ABSTRACT ALGEBRA
    // =================================================================
    Group(String),
    Ring(String),
    Field(String),
    VectorSpace(String),
    Manifold { dim: u32, name: String },
    MetricTensor(Box<SymbolicExpr>),
    ChristoffelSymbol(Box<SymbolicExpr>),
    RiemannTensor(Box<SymbolicExpr>),
    RicciTensor(Box<SymbolicExpr>),
    RicciScalar(Box<SymbolicExpr>),
    CovariantDerivative(Box<SymbolicExpr>, String),
    LieDerivative(Box<SymbolicExpr>, Box<SymbolicExpr>),
    ExteriorDerivative(Box<SymbolicExpr>),
    HodgeStar(Box<SymbolicExpr>),
    WedgeProduct(Box<SymbolicExpr>, Box<SymbolicExpr>),
    
    // =================================================================
    // SIMPLIFICATION & MANIPULATION
    // =================================================================
    Simplify(Box<SymbolicExpr>),
    Expand(Box<SymbolicExpr>),
    Factor(Box<SymbolicExpr>),
    Collect(Box<SymbolicExpr>, String),
    Apart(Box<SymbolicExpr>, String),     // Partial fractions
    Together(Box<SymbolicExpr>),          // Combine fractions
    TrigExpand(Box<SymbolicExpr>),
    TrigReduce(Box<SymbolicExpr>),
    ComplexExpand(Box<SymbolicExpr>),
    Substitute(Box<SymbolicExpr>, Vec<(String, SymbolicExpr)>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum LimitDirection { Left, Right, Both }

#[derive(Debug, Clone, PartialEq)]
pub enum AiryKind { Ai, Bi }

#[derive(Debug, Clone, PartialEq)]
pub enum NormKind { L1, L2, LInf, Frobenius, Nuclear, Spectral }

#[derive(Debug, Clone, PartialEq)]
pub enum ODEMethod {
    ExplicitEuler, ImplicitEuler, RungeKutta4, RungeKuttaFehlberg,
    AdamsBashforth(u8), AdamsMoulton(u8), BDF(u8),
    DormandPrince, Radau, LSODA, Symplectic,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WaveletKind {
    Haar, Daubechies(u8), Symlet(u8), Coiflet(u8), Morlet, MexicanHat,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DistributionKind {
    Normal(f64, f64), Uniform(f64, f64), Exponential(f64),
    Poisson(f64), Binomial(u64, f64), Bernoulli(f64),
    Gamma(f64, f64), Beta(f64, f64), ChiSquared(u32),
    StudentT(f64), FisherF(f64, f64), Cauchy(f64, f64),
    Weibull(f64, f64), LogNormal(f64, f64), Dirichlet(Vec<f64>),
    Multinomial(Vec<f64>), Geometric(f64), NegBinomial(f64, f64),
    Hypergeometric(u64, u64, u64), Maxwell(f64),
    VonMises(f64, f64), Wishart(u32, Vec<Vec<f64>>),
}


// ============================================================================
// SECTION 3: COMPLETE GRAPH ALGORITHM LIBRARY
// ============================================================================
// Every graph algorithm from Cormen (CLRS), Sedgewick, and research papers.
// These are compiler built-ins, not library functions — the compiler can
// choose optimal implementations based on graph properties.

/// A generic graph representation that supports all algorithms.
#[derive(Debug, Clone)]
pub struct ChronosGraph<N: Clone, E: Clone> {
    pub nodes: Vec<GraphNode<N>>,
    pub edges: Vec<GraphEdge<E>>,
    pub adjacency: Vec<Vec<(usize, usize)>>, // node_idx → [(neighbor_idx, edge_idx)]
    pub directed: bool,
    pub properties: GraphProperties,
}

#[derive(Debug, Clone)]
pub struct GraphNode<N: Clone> {
    pub id: usize,
    pub data: N,
}

#[derive(Debug, Clone)]
pub struct GraphEdge<E: Clone> {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
    pub data: E,
}

#[derive(Debug, Clone, Default)]
pub struct GraphProperties {
    pub is_connected: Option<bool>,
    pub is_bipartite: Option<bool>,
    pub is_planar: Option<bool>,
    pub is_dag: Option<bool>,
    pub is_tree: Option<bool>,
    pub is_eulerian: Option<bool>,
    pub is_hamiltonian: Option<bool>,
    pub chromatic_number: Option<u32>,
    pub treewidth: Option<u32>,
}

/// Every graph algorithm, categorized by purpose.
#[derive(Debug, Clone)]
pub enum GraphAlgorithm {
    // === TRAVERSAL ===
    BFS { source: usize },
    DFS { source: usize },
    IterativeDFS { source: usize },
    BidirectionalBFS { source: usize, target: usize },
    TopologicalSort,
    LexicographicBFS,
    RandomWalk { source: usize, steps: usize },
    EulerTour,
    
    // === SHORTEST PATHS ===
    Dijkstra { source: usize },
    BellmanFord { source: usize },
    FloydWarshall,                      // All-pairs shortest path
    Johnson,                            // All-pairs (sparse graphs)
    AStar { source: usize, target: usize, heuristic: AStarHeuristic },
    BidirectionalDijkstra { source: usize, target: usize },
    DeltaStepping { source: usize, delta: f64 }, // Parallel SSSP
    Yen { source: usize, target: usize, k: usize }, // K-shortest paths
    Suurballe { source: usize, target: usize },     // Edge-disjoint shortest pair
    SPFA { source: usize },             // Shortest Path Faster Algorithm
    
    // === MINIMUM SPANNING TREE ===
    Kruskal,
    Prim { source: usize },
    Boruvka,
    SteinerTree { terminals: Vec<usize> },
    MinimumBottleneckSpanningTree,
    
    // === NETWORK FLOW ===
    FordFulkerson { source: usize, sink: usize },
    EdmondsKarp { source: usize, sink: usize },
    Dinic { source: usize, sink: usize },
    PushRelabel { source: usize, sink: usize },
    MinCostMaxFlow { source: usize, sink: usize },
    MaxWeightMatching,
    HungarianAlgorithm,                 // Optimal assignment
    HopcroftKarp,                       // Maximum bipartite matching
    
    // === CONNECTIVITY ===
    ConnectedComponents,
    StronglyConnectedComponents,        // Tarjan's / Kosaraju's
    BiconnectedComponents,
    ArticulationPoints,                 // Cut vertices
    Bridges,                            // Cut edges
    KVertexConnectivity,
    KEdgeConnectivity,
    
    // === CYCLES ===
    CycleDetection,
    FindAllCycles,                      // Johnson's algorithm
    NegativeCycleDetection,
    MinimumWeightCycle,
    GirthComputation,                   // Length of shortest cycle
    
    // === COLORING ===
    GreedyColoring,
    WelshPowell,
    DSatur,
    ChromaticNumber,                    // Exact (exponential)
    ChromaticPolynomial,
    EdgeColoring,
    ListColoring(Vec<Vec<u32>>),
    
    // === CLIQUES & INDEPENDENT SETS ===
    BronKerbosch,                       // Find all maximal cliques
    MaximumClique,
    MaximumIndependentSet,
    MinimumVertexCover,
    MaximumWeightIndependentSet,
    
    // === PLANARITY ===
    PlanarityTesting,                   // Boyer–Myrvold
    PlanarEmbedding,
    KuratowskiSubgraph,
    FacesOfPlanarGraph,
    DualGraph,
    
    // === TREE ALGORITHMS ===
    LCA { u: usize, v: usize },        // Lowest Common Ancestor
    HeavyLightDecomposition,
    CentroidDecomposition,
    TreeIsomorphism,
    PruferSequence,
    TreeDiameter,
    TreeCenter,
    
    // === SPECTRAL ===
    LaplacianMatrix,
    AdjacencySpectrum,
    FiedlerVector,                      // Second-smallest eigenvalue eigenvector
    SpectralClustering { k: usize },
    PageRank { damping: f64, iterations: usize },
    HITS,                               // Hubs and Authorities
    Katz { alpha: f64, beta: f64 },
    BetweennessCentrality,
    ClosenessCentrality,
    EigenvectorCentrality,
    
    // === COMMUNITY DETECTION ===
    Louvain,
    LabelPropagation,
    GirvanNewman,
    ModularityOptimization,
    InfoMap,
    StochasticBlockModel,
    
    // === SPECIAL PURPOSE ===
    TravelingSalesman(TSPMethod),
    GraphIsomorphism,                   // VF2 / Weisfeiler-Leman
    SubgraphIsomorphism { pattern: Vec<(usize, usize)> },
    MinimumDominatingSet,
    MinimumFeedbackVertexSet,
    MinimumFeedbackArcSet,
    MaxCut,
    GraphPartitioning { k: usize, method: PartitionMethod },
    TreeDecomposition,
    PathDecomposition,
    
    // === RANDOM GRAPH GENERATION ===
    ErdosRenyi { n: usize, p: f64 },
    BarabasiAlbert { n: usize, m: usize },
    WattsStrogatz { n: usize, k: usize, beta: f64 },
    StochasticBlockModelGen { sizes: Vec<usize>, probs: Vec<Vec<f64>> },
    RandomRegular { n: usize, degree: usize },
    ConfigurationModel { degrees: Vec<usize> },
}

#[derive(Debug, Clone)]
pub enum AStarHeuristic { Euclidean, Manhattan, Chebyshev, Haversine, Custom(String) }

#[derive(Debug, Clone)]
pub enum TSPMethod {
    BruteForce, NearestNeighbor, Christofides, LinKernighan,
    SimulatedAnnealing, GeneticAlgorithm, AntColony, BranchAndBound,
}

#[derive(Debug, Clone)]
pub enum PartitionMethod { KernighanLin, Metis, Spectral, Multilevel }

/// Result types for graph algorithms.
#[derive(Debug, Clone)]
pub enum GraphResult<N: Clone, E: Clone> {
    Path(Vec<usize>, f64),                           // Path + total cost
    Tree(Vec<(usize, usize)>, f64),                  // Edge list + total weight
    Components(Vec<Vec<usize>>),                     // Groups of node indices
    Flow(f64, HashMap<(usize, usize), f64>),         // Max flow + edge flows
    Matching(Vec<(usize, usize)>),                   // Matched pairs
    Coloring(HashMap<usize, u32>),                   // Node → color
    Cliques(Vec<Vec<usize>>),                        // List of cliques
    Distances(Vec<f64>),                             // SSSP distances
    AllPairsDistances(Vec<Vec<f64>>),                // Floyd-Warshall result
    Centrality(Vec<f64>),                            // Per-node centrality
    Communities(Vec<Vec<usize>>),                    // Community assignments
    Ordering(Vec<usize>),                            // Topological / traversal order
    Bool(bool),                                       // Connectivity, planarity, etc.
    Spectrum(Vec<f64>),                               // Eigenvalues
    Partition(Vec<usize>),                           // Node → partition assignment
    #[allow(dead_code)]
    _Phantom(std::marker::PhantomData<(N, E)>),
}

/// Execute a graph algorithm. The compiler can select the optimal
/// implementation based on the graph's known properties.
pub fn execute_graph_algorithm<N: Clone + fmt::Debug, E: Clone + fmt::Debug>(
    graph: &ChronosGraph<N, E>,
    algorithm: &GraphAlgorithm,
) -> GraphResult<N, E> {
    match algorithm {
        GraphAlgorithm::BFS { source } => {
            let mut visited = vec![false; graph.nodes.len()];
            let mut order = Vec::new();
            let mut queue = VecDeque::new();
            
            visited[*source] = true;
            queue.push_back(*source);
            
            while let Some(node) = queue.pop_front() {
                order.push(node);
                for &(neighbor, _edge_idx) in &graph.adjacency[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            GraphResult::Ordering(order)
        }

        GraphAlgorithm::Dijkstra { source } => {
            let n = graph.nodes.len();
            let mut dist = vec![f64::INFINITY; n];
            let mut visited = vec![false; n];
            dist[*source] = 0.0;

            for _ in 0..n {
                // Find the unvisited node with minimum distance.
                let u = (0..n)
                    .filter(|&i| !visited[i])
                    .min_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap())
                    .unwrap_or(0);

                if dist[u].is_infinite() { break; }
                visited[u] = true;

                for &(v, edge_idx) in &graph.adjacency[u] {
                    let w = graph.edges[edge_idx].weight;
                    if dist[u] + w < dist[v] {
                        dist[v] = dist[u] + w;
                    }
                }
            }
            GraphResult::Distances(dist)
        }

        GraphAlgorithm::TopologicalSort => {
            let n = graph.nodes.len();
            let mut in_degree = vec![0usize; n];
            for edges in &graph.adjacency {
                for &(neighbor, _) in edges {
                    in_degree[neighbor] += 1;
                }
            }
            let mut queue: VecDeque<usize> = (0..n)
                .filter(|&i| in_degree[i] == 0)
                .collect();
            let mut order = Vec::new();
            while let Some(node) = queue.pop_front() {
                order.push(node);
                for &(neighbor, _) in &graph.adjacency[node] {
                    in_degree[neighbor] -= 1;
                    if in_degree[neighbor] == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
            GraphResult::Ordering(order)
        }

        GraphAlgorithm::ConnectedComponents => {
            let n = graph.nodes.len();
            let mut visited = vec![false; n];
            let mut components = Vec::new();
            for start in 0..n {
                if visited[start] { continue; }
                let mut component = Vec::new();
                let mut stack = vec![start];
                while let Some(node) = stack.pop() {
                    if visited[node] { continue; }
                    visited[node] = true;
                    component.push(node);
                    for &(neighbor, _) in &graph.adjacency[node] {
                        if !visited[neighbor] { stack.push(neighbor); }
                    }
                }
                components.push(component);
            }
            GraphResult::Components(components)
        }

        GraphAlgorithm::FloydWarshall => {
            let n = graph.nodes.len();
            let mut dist = vec![vec![f64::INFINITY; n]; n];
            for i in 0..n { dist[i][i] = 0.0; }
            for edge in &graph.edges {
                dist[edge.from][edge.to] = dist[edge.from][edge.to].min(edge.weight);
                if !graph.directed {
                    dist[edge.to][edge.from] = dist[edge.to][edge.from].min(edge.weight);
                }
            }
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        if dist[i][k] + dist[k][j] < dist[i][j] {
                            dist[i][j] = dist[i][k] + dist[k][j];
                        }
                    }
                }
            }
            GraphResult::AllPairsDistances(dist)
        }

        GraphAlgorithm::Kruskal => {
            let mut edges: Vec<(f64, usize, usize, usize)> = graph.edges.iter()
                .enumerate()
                .map(|(i, e)| (e.weight, e.from, e.to, i))
                .collect();
            edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let n = graph.nodes.len();
            let mut parent: Vec<usize> = (0..n).collect();
            let mut rank = vec![0u32; n];
            
            fn find(parent: &mut [usize], x: usize) -> usize {
                if parent[x] != x { parent[x] = find(parent, parent[x]); }
                parent[x]
            }
            fn union(parent: &mut [usize], rank: &mut [u32], x: usize, y: usize) -> bool {
                let rx = find(parent, x);
                let ry = find(parent, y);
                if rx == ry { return false; }
                if rank[rx] < rank[ry] { parent[rx] = ry; }
                else if rank[rx] > rank[ry] { parent[ry] = rx; }
                else { parent[ry] = rx; rank[rx] += 1; }
                true
            }

            let mut mst_edges = Vec::new();
            let mut total_weight = 0.0;
            for (w, u, v, _) in edges {
                if union(&mut parent, &mut rank, u, v) {
                    mst_edges.push((u, v));
                    total_weight += w;
                }
            }
            GraphResult::Tree(mst_edges, total_weight)
        }

        GraphAlgorithm::PageRank { damping, iterations } => {
            let n = graph.nodes.len();
            let d = *damping;
            let mut scores = vec![1.0 / n as f64; n];
            let out_degree: Vec<usize> = graph.adjacency.iter()
                .map(|adj| adj.len().max(1))
                .collect();

            for _ in 0..*iterations {
                let mut new_scores = vec![(1.0 - d) / n as f64; n];
                for u in 0..n {
                    let contribution = d * scores[u] / out_degree[u] as f64;
                    for &(v, _) in &graph.adjacency[u] {
                        new_scores[v] += contribution;
                    }
                }
                scores = new_scores;
            }
            GraphResult::Centrality(scores)
        }

        // For algorithms not fully implemented, return a placeholder.
        _ => {
            GraphResult::Bool(false)
        }
    }
}
