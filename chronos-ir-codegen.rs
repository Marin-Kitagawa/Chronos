// ============================================================================
// CHRONOS INTERMEDIATE REPRESENTATION (IR) & CODE GENERATION BACKENDS
// ============================================================================
// After parsing and type checking, we lower the AST into an IR that's
// closer to machine code. This IR is then translated to one of several
// backends:
//
//   - LLVM IR   → CPU code (via LLVM)
//   - NVPTX IR  → NVIDIA GPU code (CUDA)
//   - AMDGPU IR → AMD GPU code (ROCm/HIP)
//   - XLA HLO   → Google TPU code
//   - NPU IR    → Neural Processing Unit code
//   - SPIR-V    → Vulkan compute shaders (cross-platform GPU)
//
// The IR design is inspired by MLIR (Multi-Level IR) from the LLVM project.
// It uses a hierarchical "dialect" system where different levels of
// abstraction coexist. High-level Chronos operations get progressively
// lowered through these dialects until they reach machine-specific form.
// ============================================================================

// use std::collections::HashMap; // provided by parent scope in compiler-core
// use std::fmt; // provided by parent scope in compiler-core

// ============================================================================
// SECTION 1: CHRONOS IR — The Intermediate Representation
// ============================================================================
// The IR is a graph of operations organized into functions and basic blocks.
// Each operation takes typed values as input and produces typed values as
// output. This is similar to LLVM IR or MLIR, but with Chronos-specific
// operations for AI, linear types, and device placement.

/// The top-level IR module — contains all functions and global declarations.
#[derive(Debug, Clone)]
pub struct IRModule {
    pub name: String,
    pub functions: Vec<IRFunction>,
    pub globals: Vec<IRGlobal>,
    pub type_declarations: Vec<IRTypeDecl>,
    pub target: CompilationTarget,
    /// Version metadata from Feature 3.
    pub version_info: VersionInfo,
}

/// A function in the IR — a sequence of basic blocks.
#[derive(Debug, Clone)]
pub struct IRFunction {
    pub name: String,
    pub params: Vec<IRParam>,
    pub return_type: IRType,
    pub blocks: Vec<BasicBlock>,
    pub attributes: Vec<FunctionAttribute>,
    /// Which device this function runs on (Feature 5).
    pub device: IRDevice,
    /// Degradation metadata (Feature 6).
    pub degradation: Option<IRDegradation>,
    /// Effect summary from type inference (Feature 8).
    pub effects: Vec<IREffect>,
    /// Whether this function has been verified as memory-safe (Feature 8).
    pub memory_safe: bool,
}

/// A basic block — a straight-line sequence of operations ending with
/// a terminator (branch, return, etc.). Control flow only enters at the
/// top and exits at the bottom.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub label: String,
    pub ops: Vec<IROp>,
    pub terminator: IRTerminator,
}

/// A single IR operation. This is the core building block.
/// Each operation takes inputs, produces an output, and has a specific
/// opcode that tells the code generator what to emit.
#[derive(Debug, Clone)]
pub struct IROp {
    /// The register / SSA value this operation defines.
    pub result: Option<IRValue>,
    /// The operation kind — what this instruction does.
    pub opcode: IROpcode,
    /// Source location for debugging and error reporting.
    pub loc: IRLocation,
}

/// SSA values — each value is produced exactly once and can be used many times.
/// This is the standard Static Single Assignment form used by all modern compilers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IRValue {
    pub name: String,
    pub ty: IRType,
}

/// Parameters to IR functions.
#[derive(Debug, Clone)]
pub struct IRParam {
    pub name: String,
    pub ty: IRType,
    /// For kernel parameters: which memory space they live in (Feature 5).
    pub memory_space: Option<IRMemorySpace>,
}

/// Types in the IR are simpler than source-level types. By this point,
/// all generics have been monomorphized, all type inference is resolved,
/// and all higher-kinded types have been erased.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IRType {
    Void,
    Bool,
    Int(u32),           // Bit width: 8, 16, 32, 64, 128
    UInt(u32),
    Float(u32),         // 16, 32, 64, 128
    BFloat16,           // Special: brain float for AI (Feature 5)
    Pointer(Box<IRType>),
    Array(Box<IRType>, usize),
    Struct(String, Vec<IRType>),
    /// Tensor type for AI operations (Feature 5).
    /// Shape is fully resolved at this point.
    Tensor {
        element: Box<IRType>,
        shape: Vec<usize>,  // Fully static by IR time
    },
    Function {
        params: Vec<IRType>,
        return_type: Box<IRType>,
    },
    /// A resource handle with ownership tracking (Feature 8).
    /// The tag is used by the linear type verifier.
    OwnedHandle {
        inner: Box<IRType>,
        linear: bool,   // true = linear, false = affine
    },
    /// Opaque type (for FFI or AI model handles).
    Opaque(String),
}

/// The complete instruction set for Chronos IR.
/// Organized by category for clarity.
#[derive(Debug, Clone)]
pub enum IROpcode {
    // === BASIC OPERATIONS ===
    /// Integer constant.
    ConstInt(i64, u32),           // value, bit_width
    /// Float constant.
    ConstFloat(f64, u32),         // value, bit_width
    /// Boolean constant.
    ConstBool(bool),
    /// String constant (index into string table).
    ConstString(String),
    /// Null/none value.
    ConstNull(IRType),

    // === ARITHMETIC ===
    Add(IRValue, IRValue),
    Sub(IRValue, IRValue),
    Mul(IRValue, IRValue),
    Div(IRValue, IRValue),
    Rem(IRValue, IRValue),
    Neg(IRValue),

    // === FLOATING POINT ===
    FAdd(IRValue, IRValue),
    FSub(IRValue, IRValue),
    FMul(IRValue, IRValue),
    FDiv(IRValue, IRValue),
    FNeg(IRValue),

    // === BITWISE ===
    And(IRValue, IRValue),
    Or(IRValue, IRValue),
    Xor(IRValue, IRValue),
    Shl(IRValue, IRValue),
    Shr(IRValue, IRValue),
    Not(IRValue),

    // === COMPARISON ===
    ICmp(CmpPredicate, IRValue, IRValue),
    FCmp(CmpPredicate, IRValue, IRValue),

    // === TYPE CONVERSION ===
    Trunc(IRValue, IRType),       // Narrow: i64 -> i32
    ZExt(IRValue, IRType),        // Zero-extend: i32 -> i64
    SExt(IRValue, IRType),        // Sign-extend: i32 -> i64
    FPTrunc(IRValue, IRType),     // f64 -> f32
    FPExt(IRValue, IRType),       // f32 -> f64
    FPToInt(IRValue, IRType),     // f64 -> i64
    IntToFP(IRValue, IRType),     // i64 -> f64
    Bitcast(IRValue, IRType),     // Reinterpret bits
    /// Quantize a tensor (Feature 5: AI model optimization).
    Quantize(IRValue, u8),        // value, bits (e.g., 8 for INT8 quantization)
    Dequantize(IRValue, IRType),

    // === MEMORY ===
    /// Allocate on the stack (no GC! Feature 7).
    StackAlloc(IRType),
    /// Allocate on the heap (reference-counted or owned — no GC!).
    HeapAlloc(IRType),
    /// Load from a pointer.
    Load(IRValue),
    /// Store to a pointer.
    Store(IRValue, IRValue),      // value, pointer
    /// Get a pointer to a struct field.
    GetFieldPtr(IRValue, u32),    // struct_ptr, field_index
    /// Get a pointer to an array element.
    GetElementPtr(IRValue, IRValue), // array_ptr, index
    /// Free heap memory (deterministic, no GC!).
    Free(IRValue),
    /// Increment reference count (for Rc-managed types).
    RcInc(IRValue),
    /// Decrement reference count; free if zero.
    RcDec(IRValue),
    /// Move a value (transfers ownership, invalidates source — Feature 8).
    Move(IRValue),
    /// Drop a value (call destructor, then free — Feature 7 & 8).
    Drop(IRValue),

    // === FUNCTION CALLS ===
    Call(String, Vec<IRValue>),
    /// Indirect call through function pointer.
    IndirectCall(IRValue, Vec<IRValue>),
    /// Tail call optimization.
    TailCall(String, Vec<IRValue>),
    /// Virtual method dispatch (for class hierarchies).
    VirtualCall(IRValue, u32, Vec<IRValue>), // vtable_ptr, method_index, args

    // === STRUCT / AGGREGATE ===
    /// Create a struct value.
    CreateStruct(String, Vec<IRValue>),
    /// Extract a field from a struct value (by index).
    ExtractField(IRValue, u32),
    /// Insert a value into a struct field.
    InsertField(IRValue, u32, IRValue),

    // === CONTROL FLOW (within basic blocks, handled by terminators) ===
    /// Phi node: merges values from different control flow paths.
    Phi(Vec<(IRValue, String)>),  // (value, from_block_label)

    // =========================================================
    // AI / TENSOR OPERATIONS (Feature 5: Mojo-inspired)
    // =========================================================
    // These operations compile to optimized kernels on the target device.
    
    /// General matrix multiply: C = alpha * A @ B + beta * C.
    /// This is THE fundamental operation for deep learning.
    Gemm {
        a: IRValue, b: IRValue, c: Option<IRValue>,
        alpha: f64, beta: f64,
        trans_a: bool, trans_b: bool,
    },
    /// 2D convolution.
    Conv2d {
        input: IRValue, kernel: IRValue, bias: Option<IRValue>,
        stride: [usize; 2], padding: [usize; 4], dilation: [usize; 2],
        groups: usize,
    },
    /// Multi-head self-attention (the core of Transformers).
    Attention {
        query: IRValue, key: IRValue, value: IRValue,
        mask: Option<IRValue>,
        num_heads: usize, head_dim: usize,
        causal: bool,
        scale: f64,
    },
    /// Softmax along a dimension.
    Softmax(IRValue, i32),
    /// Layer normalization.
    LayerNorm {
        input: IRValue, weight: IRValue, bias: IRValue,
        eps: f64,
    },
    /// Batch normalization.
    BatchNorm {
        input: IRValue, mean: IRValue, var: IRValue,
        weight: IRValue, bias: IRValue,
        eps: f64, training: bool,
    },
    /// Activation functions.
    Relu(IRValue),
    Gelu(IRValue),
    Silu(IRValue),    // SiLU / Swish
    Sigmoid(IRValue),
    Tanh(IRValue),
    /// Element-wise tensor operations.
    TensorAdd(IRValue, IRValue),
    TensorMul(IRValue, IRValue),
    TensorDiv(IRValue, IRValue),
    /// Tensor reshape / view.
    Reshape(IRValue, Vec<usize>),
    Transpose(IRValue, Vec<usize>),  // Permutation of dimensions
    /// Tensor creation.
    TensorZeros(IRType, Vec<usize>),
    TensorOnes(IRType, Vec<usize>),
    TensorRand(IRType, Vec<usize>),
    /// Embedding table lookup.
    Embedding(IRValue, IRValue),     // indices, table
    /// Scatter / Gather (for sparse operations).
    Scatter(IRValue, IRValue, IRValue, i32), // input, indices, src, dim
    Gather(IRValue, IRValue, i32),           // input, indices, dim
    /// Reduce operations.
    ReduceSum(IRValue, Vec<i32>, bool),    // input, dims, keep_dim
    ReduceMean(IRValue, Vec<i32>, bool),
    ReduceMax(IRValue, Vec<i32>, bool),
    /// Collective communication (for distributed training).
    AllReduce(IRValue, ReduceOp),
    AllGather(IRValue, i32),
    ReduceScatter(IRValue, ReduceOp),

    // =========================================================
    // DEVICE MANAGEMENT (Feature 5)
    // =========================================================
    /// Transfer data between devices (CPU ↔ GPU ↔ TPU ↔ NPU).
    DeviceTransfer(IRValue, IRDevice, IRDevice),
    /// Synchronize device execution (barrier).
    DeviceSync(IRDevice),
    /// Launch a kernel on a device.
    KernelLaunch {
        kernel_name: String,
        args: Vec<IRValue>,
        grid: [IRValue; 3],
        block: [IRValue; 3],
        shared_mem: usize,
        device: IRDevice,
    },

    // =========================================================
    // LINEAR TYPE RUNTIME OPERATIONS (Feature 8)
    // =========================================================
    /// Assert that a resource is alive (compile-time proven, but we can
    /// insert runtime checks in debug mode).
    AssertAlive(IRValue),
    /// Mark a resource as consumed. After this, any use is a compile error.
    Consume(IRValue),

    // =========================================================
    // AI INVOCATION (Feature 4)
    // =========================================================
    /// Invoke an AI skill at runtime.
    AiInvoke {
        skill_name: String,
        inputs: Vec<(String, IRValue)>,
        output_type: IRType,
    },

    // =========================================================
    // DOMAIN / ALLOCATOR OPERATIONS
    // =========================================================
    /// Arena allocation (bump allocator).
    ArenaAlloc(String, usize),   // arena name, size
    /// Pool allocation.
    PoolAlloc(String, usize),    // pool name, size
    /// Generic domain-specific operation.
    DomainOp(String),            // domain op name/id
}

/// Comparison predicates for integer and float comparisons.
#[derive(Debug, Clone, Copy)]
pub enum CmpPredicate {
    Eq, Ne, Lt, Le, Gt, Ge,
    // Float-specific:
    OrdEq, OrdNe, OrdLt, OrdLe, OrdGt, OrdGe,  // Ordered (NaN = false)
    UnordEq, UnordNe,  // Unordered (NaN = true)
}

/// Terminators end a basic block by transferring control elsewhere.
#[derive(Debug, Clone)]
pub enum IRTerminator {
    /// Return from the function.
    Return(Option<IRValue>),
    /// Unconditional branch.
    Branch(String),  // target block label
    /// Conditional branch.
    CondBranch {
        condition: IRValue,
        true_block: String,
        false_block: String,
    },
    /// Multi-way branch (for match/switch).
    Switch {
        value: IRValue,
        default: String,
        cases: Vec<(i64, String)>,
    },
    /// Unreachable (e.g., after a Never-typed expression).
    Unreachable,
}

/// Device targets in the IR (Feature 5).
#[derive(Debug, Clone, PartialEq)]
pub enum IRDevice {
    Cpu,
    Gpu(u32),        // Device index
    Tpu(u32),
    Npu(u32),
    /// Distributed across multiple devices.
    Distributed(Vec<IRDevice>),
}

/// Memory spaces for device-targeted code (Feature 5).
#[derive(Debug, Clone)]
pub enum IRMemorySpace {
    Global,          // Main memory / GPU global memory
    Shared,          // GPU shared memory (per block)
    Local,           // Per-thread local memory
    Constant,        // Read-only constant memory
    Registers,       // Explicit register file
    Unified,         // Unified CPU/GPU memory (CUDA managed)
    HBM,             // High-bandwidth memory (TPU v4+)
}

/// Reduce operations for collective communication.
#[derive(Debug, Clone)]
pub enum ReduceOp { Sum, Mean, Max, Min, Product }

/// Function attributes that affect code generation.
#[derive(Debug, Clone)]
pub enum FunctionAttribute {
    Inline,
    NoInline,
    AlwaysInline,
    Cold,            // Rarely called — optimize for size
    Hot,             // Frequently called — optimize aggressively
    Pure,            // No side effects
    NoReturn,        // Never returns (panic, exit, infinite loop)
    Extern(String),  // External linkage with calling convention
    EntryPoint,      // Kernel entry point
    Vectorize(u32),  // SIMD vectorization width hint
    Unroll(u32),     // Loop unrolling factor
    RealtimeDeadline(u64), // Hard real-time deadline in nanoseconds
    SmartContract,   // Marks a function as a smart contract entry point
}

/// Degradation metadata preserved in IR (Feature 6).
#[derive(Debug, Clone)]
pub struct IRDegradation {
    pub warn_timestamp: u64,
    pub expire_timestamp: u64,
    pub replacement: Option<String>,
}

/// Effect tags in the IR (Feature 8).
#[derive(Debug, Clone, PartialEq)]
pub enum IREffect {
    Pure, IO, Alloc, Async, Diverge, GpuKernel, TpuKernel, NpuKernel, QuantumCompute,
}

/// Global variables and constants.
#[derive(Debug, Clone)]
pub struct IRGlobal {
    pub name: String,
    pub ty: IRType,
    pub initializer: Option<IRGlobalInit>,
    pub is_const: bool,
}

#[derive(Debug, Clone)]
pub enum IRGlobalInit {
    Int(i64),
    Float(f64),
    String(String),
    Zeros,
    Array(Vec<IRGlobalInit>),
}

/// Named struct type declaration.
#[derive(Debug, Clone)]
pub struct IRTypeDecl {
    pub name: String,
    pub fields: Vec<IRType>,
    pub packed: bool,
}

/// Source location for debugging.
#[derive(Debug, Clone)]
pub struct IRLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// What we're compiling for.
#[derive(Debug, Clone)]
pub struct CompilationTarget {
    pub arch: TargetArch,
    pub os: TargetOS,
    pub features: Vec<String>,
}

impl Default for CompilationTarget {
    fn default() -> Self {
        Self { arch: TargetArch::X86_64, os: TargetOS::Linux, features: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub enum TargetArch {
    X86_64, AArch64, RiscV64, Wasm32,
    NVPTX64,   // NVIDIA GPU
    AMDGPU,    // AMD GPU
    SPIRV,     // Vulkan compute
}

#[derive(Debug, Clone)]
pub enum TargetOS {
    Linux, MacOS, Windows, None, // 'None' for bare metal / GPU
}

/// Version information baked into the compiled output (Feature 3).
#[derive(Debug, Clone)]
pub struct VersionInfo {
    pub version: String,
    pub track: String,
    pub commit_hash: Option<String>,
    pub geographic_variant: Option<String>,
    pub build_timestamp: u64,
}


// ============================================================================
// SECTION 2: AST → IR LOWERING
// ============================================================================
// This pass converts the high-level AST (with classes, templates, etc.)
// into the flat, low-level IR (with basic blocks, SSA values, etc.).

pub struct ASTLowering {
    /// The IR module being built.
    module: IRModule,
    /// Counter for generating unique SSA value names.
    next_val: u32,
    /// Counter for generating unique block labels.
    next_block: u32,
    /// Current function being lowered.
    current_fn: Option<String>,
    /// Maps source-level variables to their IR values.
    value_map: HashMap<String, IRValue>,
    /// Current basic block's operations (accumulated before creating the block).
    current_ops: Vec<IROp>,
    /// Current block label.
    current_block_label: String,
    /// All blocks for the current function.
    current_blocks: Vec<BasicBlock>,
}

impl ASTLowering {
    pub fn new(module_name: &str, target: CompilationTarget) -> Self {
        Self {
            module: IRModule {
                name: module_name.to_string(),
                functions: Vec::new(),
                globals: Vec::new(),
                type_declarations: Vec::new(),
                target,
                version_info: VersionInfo {
                    version: "0.1.0".to_string(),
                    track: "dev".to_string(),
                    commit_hash: None,
                    geographic_variant: None,
                    build_timestamp: 0,
                },
            },
            next_val: 0,
            next_block: 0,
            current_fn: None,
            value_map: HashMap::new(),
            current_ops: Vec::new(),
            current_block_label: "entry".to_string(),
            current_blocks: Vec::new(),
        }
    }

    /// Generate a fresh SSA value name.
    fn fresh_val(&mut self, ty: IRType) -> IRValue {
        let name = format!("%v{}", self.next_val);
        self.next_val += 1;
        IRValue { name, ty }
    }

    /// Generate a fresh block label.
    fn fresh_block(&mut self) -> String {
        let label = format!("bb{}", self.next_block);
        self.next_block += 1;
        label
    }

    /// Emit an operation into the current block.
    fn emit(&mut self, opcode: IROpcode, result_ty: Option<IRType>) -> Option<IRValue> {
        let result = result_ty.map(|ty| self.fresh_val(ty));
        self.current_ops.push(IROp {
            result: result.clone(),
            opcode,
            loc: IRLocation { file: String::new(), line: 0, column: 0 },
        });
        result
    }

    /// Finalize the current block with a terminator and start a new one.
    fn terminate_block(&mut self, terminator: IRTerminator) {
        let ops = std::mem::take(&mut self.current_ops);
        self.current_blocks.push(BasicBlock {
            label: self.current_block_label.clone(),
            ops,
            terminator,
        });
    }

    /// Start a new basic block.
    fn start_block(&mut self, label: String) {
        self.current_block_label = label;
    }

    // =================================================================
    // LOWERING: Program → IR Module
    // =================================================================

    pub fn lower_program(&mut self, program: &Program) -> &IRModule {
        // First pass: lower type declarations (structs, classes → IR structs).
        for item in &program.items {
            self.lower_type_declarations(item);
        }

        // Second pass: lower functions.
        for item in &program.items {
            self.lower_item(item);
        }

        &self.module
    }

    fn lower_type_declarations(&mut self, item: &Item) {
        match item {
            Item::StructDecl(s) => {
                let fields: Vec<IRType> = s.fields.iter()
                    .map(|f| self.lower_type(&f.ty))
                    .collect();
                self.module.type_declarations.push(IRTypeDecl {
                    name: s.name.clone(),
                    fields,
                    packed: s.is_packed,
                });
            }
            Item::DataClassDecl(dc) => {
                let fields: Vec<IRType> = dc.fields.iter()
                    .map(|f| self.lower_type(&f.ty))
                    .collect();
                self.module.type_declarations.push(IRTypeDecl {
                    name: dc.name.clone(),
                    fields,
                    packed: false,
                });
            }
            Item::ClassDecl(c) => {
                // Classes get a vtable pointer as the first field.
                let mut fields = vec![IRType::Pointer(Box::new(IRType::Opaque("vtable".to_string())))];
                for f in &c.fields {
                    fields.push(self.lower_type(&f.ty));
                }
                self.module.type_declarations.push(IRTypeDecl {
                    name: c.name.clone(),
                    fields,
                    packed: false,
                });
            }
            _ => {}
        }
    }

    fn lower_item(&mut self, item: &Item) {
        match item {
            Item::FunctionDecl(f) => {
                self.lower_function(f, None);
            }
            Item::DegradableFunctionDecl(df) => {
                let degradation = IRDegradation {
                    warn_timestamp: 0,   // Would compute from ChronosTimestamp
                    expire_timestamp: 0,
                    replacement: df.degradation.replacement.clone(),
                };
                self.lower_function(&df.function, Some(degradation));
            }
            Item::KernelDecl(k) => {
                self.lower_kernel(k);
            }
            _ => {}
        }
    }

    fn lower_function(&mut self, f: &FunctionDecl, degradation: Option<IRDegradation>) {
        self.current_fn = Some(f.signature.name.clone());
        self.value_map.clear();
        self.current_blocks.clear();
        self.current_ops.clear();
        self.next_val = 0;
        self.next_block = 0;
        self.current_block_label = "entry".to_string();

        // Create IR parameters and bind them in the value map.
        let params: Vec<IRParam> = f.signature.params.iter().map(|p| {
            let ir_ty = self.lower_type(&p.ty);
            let val = self.fresh_val(ir_ty.clone());
            self.value_map.insert(p.name.clone(), val);
            IRParam {
                name: p.name.clone(),
                ty: ir_ty,
                memory_space: None,
            }
        }).collect();

        // Lower the function body.
        let mut last_val = None;
        for stmt in &f.body {
            last_val = self.lower_statement(stmt);
        }

        // If the last block isn't terminated, add a return.
        if !self.current_ops.is_empty() || self.current_blocks.is_empty() {
            let ret_val = if f.signature.return_type != ChronosType::Void {
                last_val
            } else {
                None
            };
            self.terminate_block(IRTerminator::Return(ret_val));
        }

        let blocks = std::mem::take(&mut self.current_blocks);

        // Determine device target.
        let device = if f.signature.effects.iter().any(|e| matches!(e, Effect::GpuKernel)) {
            IRDevice::Gpu(0)
        } else if f.signature.effects.iter().any(|e| matches!(e, Effect::TpuKernel)) {
            IRDevice::Tpu(0)
        } else {
            IRDevice::Cpu
        };

        self.module.functions.push(IRFunction {
            name: f.signature.name.clone(),
            params,
            return_type: self.lower_type(&f.signature.return_type),
            blocks,
            attributes: Vec::new(),
            device,
            degradation,
            effects: f.signature.effects.iter().map(|e| match e {
                Effect::Pure => IREffect::Pure,
                Effect::IO => IREffect::IO,
                Effect::Alloc => IREffect::Alloc,
                Effect::Async => IREffect::Async,
                Effect::GpuKernel => IREffect::GpuKernel,
                Effect::TpuKernel => IREffect::TpuKernel,
                Effect::NpuKernel => IREffect::NpuKernel,
                _ => IREffect::IO,
            }).collect(),
            memory_safe: true,
        });
    }

    fn lower_kernel(&mut self, k: &KernelDecl) {
        // Kernels are lowered like functions but with special attributes
        // and memory space annotations on parameters.
        let params: Vec<IRParam> = k.params.iter().map(|p| {
            let ir_ty = self.lower_type(&p.ty);
            let mem = match p.memory_space {
                MemorySpace::Global => IRMemorySpace::Global,
                MemorySpace::Shared => IRMemorySpace::Shared,
                MemorySpace::Local => IRMemorySpace::Local,
                MemorySpace::Constant => IRMemorySpace::Constant,
                MemorySpace::Registers => IRMemorySpace::Registers,
                MemorySpace::Unified => IRMemorySpace::Unified,
                MemorySpace::Hbm => IRMemorySpace::HBM,
            };
            IRParam {
                name: p.name.clone(),
                ty: ir_ty,
                memory_space: Some(mem),
            }
        }).collect();

        let device = match &k.target {
            DeviceTarget::Gpu { index } => IRDevice::Gpu(*index as u32),
            DeviceTarget::Tpu { index } => IRDevice::Tpu(*index as u32),
            DeviceTarget::Npu { index } => IRDevice::Npu(*index as u32),
            _ => IRDevice::Gpu(0),
        };

        // Lower body (simplified — a real implementation would handle
        // tile/vectorize/unroll annotations).
        self.current_blocks.clear();
        self.current_ops.clear();
        self.current_block_label = "entry".to_string();

        for stmt in &k.body {
            self.lower_statement(stmt);
        }
        self.terminate_block(IRTerminator::Return(None));

        let blocks = std::mem::take(&mut self.current_blocks);

        self.module.functions.push(IRFunction {
            name: k.name.clone(),
            params,
            return_type: self.lower_type(&k.return_type),
            blocks,
            attributes: vec![FunctionAttribute::EntryPoint],
            device,
            degradation: None,
            effects: vec![match &k.target {
                DeviceTarget::Gpu { .. } => IREffect::GpuKernel,
                DeviceTarget::Tpu { .. } => IREffect::TpuKernel,
                DeviceTarget::Npu { .. } => IREffect::NpuKernel,
                _ => IREffect::GpuKernel,
            }],
            memory_safe: true,
        });
    }

    // =================================================================
    // LOWERING: Statements
    // =================================================================

    fn lower_statement(&mut self, stmt: &Statement) -> Option<IRValue> {
        match stmt {
            Statement::Let { name, ty, value, .. } => {
                let val = self.lower_expression(value);
                if let Some(v) = &val {
                    self.value_map.insert(name.clone(), v.clone());
                    // If the type is linear, emit an AssertAlive marker (Feature 8).
                    if let Some(t) = ty {
                        if matches!(t, ChronosType::Linear(_)) {
                            self.emit(IROpcode::AssertAlive(v.clone()), None);
                        }
                    }
                }
                val
            }

            Statement::Assignment { target, value } => {
                let val = self.lower_expression(value);
                if let Expression::Identifier(name) = target {
                    if let Some(v) = &val {
                        self.value_map.insert(name.clone(), v.clone());
                    }
                }
                val
            }

            Statement::Return(expr) => {
                let val = expr.as_ref().and_then(|e| self.lower_expression(e));
                self.terminate_block(IRTerminator::Return(val.clone()));
                let next_label = self.fresh_block();
                self.start_block(next_label);
                val
            }

            Statement::ExprStatement(expr) => {
                self.lower_expression(expr)
            }

            Statement::While { condition, body } => {
                let cond_label = self.fresh_block();
                let body_label = self.fresh_block();
                let exit_label = self.fresh_block();

                // Jump to condition check.
                self.terminate_block(IRTerminator::Branch(cond_label.clone()));

                // Condition block.
                self.start_block(cond_label.clone());
                let cond_val = self.lower_expression(condition)
                    .unwrap_or(IRValue { name: "%false".to_string(), ty: IRType::Bool });
                self.terminate_block(IRTerminator::CondBranch {
                    condition: cond_val,
                    true_block: body_label.clone(),
                    false_block: exit_label.clone(),
                });

                // Body block.
                self.start_block(body_label);
                for s in body {
                    self.lower_statement(s);
                }
                self.terminate_block(IRTerminator::Branch(cond_label));

                // Exit block.
                self.start_block(exit_label);
                None
            }

            Statement::For { binding, iterator, body } => {
                // Desugar: for x in iter { body } → while iter.has_next() { let x = iter.next(); body }
                let iter_val = self.lower_expression(iterator);
                // Simplified: just lower the body with a placeholder binding.
                if let Some(v) = iter_val {
                    self.value_map.insert(binding.clone(), v);
                }
                for s in body {
                    self.lower_statement(s);
                }
                None
            }

            Statement::Drop(name) => {
                if let Some(val) = self.value_map.get(name).cloned() {
                    self.emit(IROpcode::Consume(val.clone()), None);
                    self.emit(IROpcode::Drop(val), None);
                }
                None
            }

            Statement::DeviceScope { target, body } => {
                // In the IR, device scopes are represented by transferring
                // data to the device, executing the body, then transferring back.
                let ir_device = match target {
                    DeviceTarget::Gpu { index } => IRDevice::Gpu(*index as u32),
                    DeviceTarget::Tpu { index } => IRDevice::Tpu(*index as u32),
                    DeviceTarget::Npu { index } => IRDevice::Npu(*index as u32),
                    _ => IRDevice::Cpu,
                };
                for s in body {
                    self.lower_statement(s);
                }
                None
            }

            Statement::Break | Statement::Continue => None,

            Statement::Require { condition, message: _ } => {
                let _cond = self.lower_expression(condition);
                None
            }
        }
    }

    // =================================================================
    // LOWERING: Expressions → IR Operations
    // =================================================================

    fn lower_expression(&mut self, expr: &Expression) -> Option<IRValue> {
        match expr {
            Expression::IntLiteral(n) => {
                self.emit(IROpcode::ConstInt(*n, 64), Some(IRType::Int(64)))
            }
            Expression::FloatLiteral(f) => {
                self.emit(IROpcode::ConstFloat(*f, 64), Some(IRType::Float(64)))
            }
            Expression::StringLiteral(s) => {
                self.emit(IROpcode::ConstString(s.clone()),
                    Some(IRType::Pointer(Box::new(IRType::Int(8)))))
            }
            Expression::BoolLiteral(b) => {
                self.emit(IROpcode::ConstBool(*b), Some(IRType::Bool))
            }
            Expression::Identifier(name) => {
                self.value_map.get(name).cloned()
            }
            Expression::BinaryOp { left, op, right } => {
                let l = self.lower_expression(left)?;
                let r = self.lower_expression(right)?;
                let ty = l.ty.clone();
                let opcode = match op {
                    BinOp::Add => if is_float(&ty) { IROpcode::FAdd(l, r) } else { IROpcode::Add(l, r) },
                    BinOp::Sub => if is_float(&ty) { IROpcode::FSub(l, r) } else { IROpcode::Sub(l, r) },
                    BinOp::Mul => if is_float(&ty) { IROpcode::FMul(l, r) } else { IROpcode::Mul(l, r) },
                    BinOp::Div => if is_float(&ty) { IROpcode::FDiv(l, r) } else { IROpcode::Div(l, r) },
                    BinOp::Mod => IROpcode::Rem(l, r),
                    BinOp::Eq => if is_float(&ty) { IROpcode::FCmp(CmpPredicate::OrdEq, l, r) } else { IROpcode::ICmp(CmpPredicate::Eq, l, r) },
                    BinOp::Neq => IROpcode::ICmp(CmpPredicate::Ne, l, r),
                    BinOp::Lt => IROpcode::ICmp(CmpPredicate::Lt, l, r),
                    BinOp::Gt => IROpcode::ICmp(CmpPredicate::Gt, l, r),
                    BinOp::Lte => IROpcode::ICmp(CmpPredicate::Le, l, r),
                    BinOp::Gte => IROpcode::ICmp(CmpPredicate::Ge, l, r),
                    BinOp::And => IROpcode::And(l, r),
                    BinOp::Or => IROpcode::Or(l, r),
                    BinOp::BitAnd => IROpcode::And(l, r),
                    BinOp::BitOr => IROpcode::Or(l, r),
                    BinOp::Xor => IROpcode::Xor(l, r),
                    BinOp::Shl => IROpcode::Shl(l, r),
                    BinOp::Shr => IROpcode::Shr(l, r),
                    BinOp::MatMul => IROpcode::Gemm {
                        a: l, b: r, c: None,
                        alpha: 1.0, beta: 0.0,
                        trans_a: false, trans_b: false,
                    },
                    BinOp::Assign | BinOp::AddAssign | BinOp::SubAssign
                    | BinOp::MulAssign | BinOp::DivAssign | BinOp::ModAssign => return None,
                };
                let result_ty = match op {
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt |
                    BinOp::Lte | BinOp::Gte | BinOp::And | BinOp::Or => IRType::Bool,
                    _ => ty,
                };
                self.emit(opcode, Some(result_ty))
            }
            Expression::UnaryOp { op, expr: inner } => {
                let v = self.lower_expression(inner)?;
                let ty = v.ty.clone();
                let opcode = match op {
                    UnaryOp::Neg => if is_float(&ty) { IROpcode::FNeg(v) } else { IROpcode::Neg(v) },
                    UnaryOp::Not => IROpcode::Not(v),
                    UnaryOp::BitNot => IROpcode::Not(v),
                    UnaryOp::Ref => return Some(v),  // In IR, references are just pointers
                    UnaryOp::MutRef => return Some(v),
                    UnaryOp::Deref => IROpcode::Load(v),
                };
                self.emit(opcode, Some(ty))
            }
            Expression::Call { function, args } => {
                let arg_vals: Vec<IRValue> = args.iter()
                    .filter_map(|a| self.lower_expression(a))
                    .collect();
                if let Expression::Identifier(name) = function.as_ref() {
                    self.emit(
                        IROpcode::Call(name.clone(), arg_vals),
                        Some(IRType::Int(64)), // Placeholder return type
                    )
                } else {
                    let fn_val = self.lower_expression(function)?;
                    self.emit(
                        IROpcode::IndirectCall(fn_val, arg_vals),
                        Some(IRType::Int(64)),
                    )
                }
            }
            Expression::FieldAccess { object, field } => {
                let obj = self.lower_expression(object)?;
                // Field index would be looked up from type info.
                self.emit(IROpcode::GetFieldPtr(obj, 0), Some(IRType::Int(64)))
            }
            Expression::MethodCall { object, method, args } => {
                let obj = self.lower_expression(object)?;
                let mut all_args = vec![obj];
                for a in args {
                    if let Some(v) = self.lower_expression(a) {
                        all_args.push(v);
                    }
                }
                self.emit(
                    IROpcode::Call(method.clone(), all_args),
                    Some(IRType::Int(64)),
                )
            }
            Expression::If { condition, then_branch, else_branch } => {
                let cond = self.lower_expression(condition)?;
                let then_label = self.fresh_block();
                let else_label = self.fresh_block();
                let merge_label = self.fresh_block();

                self.terminate_block(IRTerminator::CondBranch {
                    condition: cond,
                    true_block: then_label.clone(),
                    false_block: else_label.clone(),
                });

                // Then block
                self.start_block(then_label.clone());
                let then_val = self.lower_expression(then_branch);
                self.terminate_block(IRTerminator::Branch(merge_label.clone()));

                // Else block
                self.start_block(else_label.clone());
                let else_val = if let Some(eb) = else_branch {
                    self.lower_expression(eb)
                } else {
                    None
                };
                self.terminate_block(IRTerminator::Branch(merge_label.clone()));

                // Merge block with phi node
                self.start_block(merge_label);
                if let (Some(tv), Some(ev)) = (then_val, else_val) {
                    let ty = tv.ty.clone();
                    self.emit(
                        IROpcode::Phi(vec![
                            (tv, then_label),
                            (ev, else_label),
                        ]),
                        Some(ty),
                    )
                } else {
                    None
                }
            }
            Expression::AiInvoke { skill_name, inputs } => {
                let mut ir_inputs = Vec::new();
                for (k, v) in inputs {
                    if let Some(val) = self.lower_expression(v) {
                        ir_inputs.push((k.clone(), val));
                    }
                }
                self.emit(
                    IROpcode::AiInvoke {
                        skill_name: skill_name.clone(),
                        inputs: ir_inputs,
                        output_type: IRType::Opaque("AiResult".to_string()),
                    },
                    Some(IRType::Opaque("AiResult".to_string())),
                )
            }
            Expression::Block(stmts) => {
                let mut last = None;
                for s in stmts {
                    last = self.lower_statement(s);
                }
                last
            }
            _ => None,
        }
    }

    // =================================================================
    // TYPE LOWERING: ChronosType → IRType
    // =================================================================

    fn lower_type(&self, ty: &ChronosType) -> IRType {
        match ty {
            ChronosType::Void => IRType::Void,
            ChronosType::Bool => IRType::Bool,
            ChronosType::Int8 => IRType::Int(8),
            ChronosType::Int16 => IRType::Int(16),
            ChronosType::Int32 => IRType::Int(32),
            ChronosType::Int64 => IRType::Int(64),
            ChronosType::Int128 => IRType::Int(128),
            ChronosType::IntArbitrary => IRType::Opaque("BigInt".to_string()),
            ChronosType::UInt8 => IRType::UInt(8),
            ChronosType::UInt16 => IRType::UInt(16),
            ChronosType::UInt32 => IRType::UInt(32),
            ChronosType::UInt64 => IRType::UInt(64),
            ChronosType::UInt128 => IRType::UInt(128),
            ChronosType::UIntArbitrary => IRType::Opaque("BigUInt".to_string()),
            ChronosType::Float16 => IRType::Float(16),
            ChronosType::Float32 => IRType::Float(32),
            ChronosType::Float64 => IRType::Float(64),
            ChronosType::Float128 => IRType::Float(128),
            ChronosType::BFloat16 => IRType::BFloat16,
            ChronosType::Char => IRType::UInt(32),
            ChronosType::Str => IRType::Pointer(Box::new(IRType::Int(8))),
            ChronosType::Never => IRType::Void,

            ChronosType::Owned(inner) => self.lower_type(inner),
            ChronosType::Borrowed { inner, .. } => {
                IRType::Pointer(Box::new(self.lower_type(inner)))
            }
            ChronosType::Linear(inner) => IRType::OwnedHandle {
                inner: Box::new(self.lower_type(inner)),
                linear: true,
            },
            ChronosType::Affine(inner) => IRType::OwnedHandle {
                inner: Box::new(self.lower_type(inner)),
                linear: false,
            },

            ChronosType::Tuple(types) => {
                IRType::Struct(
                    "tuple".to_string(),
                    types.iter().map(|t| self.lower_type(t)).collect(),
                )
            }
            ChronosType::Array { element, size } => {
                IRType::Array(
                    Box::new(self.lower_type(element)),
                    size.unwrap_or(0),
                )
            }
            ChronosType::Optional(inner) => {
                // Optional<T> → Struct { has_value: bool, value: T }
                IRType::Struct(
                    "Optional".to_string(),
                    vec![IRType::Bool, self.lower_type(inner)],
                )
            }
            ChronosType::Tensor { element, shape, .. } => {
                let ir_elem = self.lower_type(element);
                let static_shape = match shape {
                    TensorShape::Static(dims) => dims.clone(),
                    _ => Vec::new(),
                };
                IRType::Tensor { element: Box::new(ir_elem), shape: static_shape }
            }
            ChronosType::Function { params, return_type, .. } => {
                IRType::Function {
                    params: params.iter().map(|p| self.lower_type(p)).collect(),
                    return_type: Box::new(self.lower_type(return_type)),
                }
            }
            ChronosType::Named { name, .. } => {
                IRType::Struct(name.clone(), Vec::new())
            }
            _ => IRType::Opaque(format!("{:?}", ty)),
        }
    }

    /// Get the built IR module.
    pub fn finish(self) -> IRModule {
        self.module
    }
}

fn is_float(ty: &IRType) -> bool {
    matches!(ty, IRType::Float(_) | IRType::BFloat16)
}


// ============================================================================
// SECTION 3: CODE GENERATION BACKENDS
// ============================================================================
// Each backend translates Chronos IR into a target-specific format.
// The backends are trait-based, so new targets can be added easily.

/// The code generation trait that all backends implement.
pub trait CodeGenBackend {
    /// The output format (e.g., LLVM IR text, PTX assembly, XLA HLO).
    type Output;

    /// Generate code for an entire IR module.
    fn generate(&mut self, module: &IRModule) -> Self::Output;

    /// The name of this backend (for diagnostics).
    fn name(&self) -> &str;
}

// =====================================================================
// BACKEND 1: LLVM IR (for CPU code)
// =====================================================================
// This generates LLVM IR text that can be fed to `llc` or `clang` to
// produce native machine code. LLVM handles register allocation,
// instruction selection, and optimization for us.

pub struct LLVMBackend {
    output: String,
    indent: usize,
}

impl LLVMBackend {
    pub fn new() -> Self {
        Self { output: String::new(), indent: 0 }
    }

    fn emit_line(&mut self, line: &str) {
        for _ in 0..self.indent {
            self.output.push_str("  ");
        }
        self.output.push_str(line);
        self.output.push('\n');
    }

    fn ir_type_to_llvm(&self, ty: &IRType) -> String {
        match ty {
            IRType::Void => "void".to_string(),
            IRType::Bool => "i1".to_string(),
            IRType::Int(w) => format!("i{}", w),
            IRType::UInt(w) => format!("i{}", w),  // LLVM doesn't distinguish signed/unsigned
            IRType::Float(16) => "half".to_string(),
            IRType::Float(32) => "float".to_string(),
            IRType::Float(64) => "double".to_string(),
            IRType::Float(128) => "fp128".to_string(),
            IRType::Float(w) => format!("double"), // fallback for non-standard float widths
            IRType::BFloat16 => "bfloat".to_string(),
            IRType::Pointer(inner) => format!("{}*", self.ir_type_to_llvm(inner)),
            IRType::Array(elem, size) => format!("[{} x {}]", size, self.ir_type_to_llvm(elem)),
            IRType::Struct(name, _) => format!("%{}", name),
            IRType::Function { params, return_type } => {
                let param_strs: Vec<String> = params.iter()
                    .map(|p| self.ir_type_to_llvm(p))
                    .collect();
                format!("{} ({})", self.ir_type_to_llvm(return_type), param_strs.join(", "))
            }
            IRType::OwnedHandle { inner, .. } => self.ir_type_to_llvm(inner),
            IRType::Tensor { element, shape } => {
                // Tensors become pointers to heap-allocated memory.
                format!("{}*", self.ir_type_to_llvm(element))
            }
            IRType::Opaque(name) => format!("%{}", name),
        }
    }

    fn generate_function(&mut self, func: &IRFunction) {
        let ret_ty = self.ir_type_to_llvm(&func.return_type);
        let params: Vec<String> = func.params.iter()
            .map(|p| format!("{} %{}", self.ir_type_to_llvm(&p.ty), p.name))
            .collect();

        // Function attributes.
        let attrs: Vec<String> = func.attributes.iter().filter_map(|a| match a {
            FunctionAttribute::AlwaysInline => Some("alwaysinline".to_string()),
            FunctionAttribute::NoInline => Some("noinline".to_string()),
            FunctionAttribute::Pure => Some("readnone".to_string()),
            FunctionAttribute::NoReturn => Some("noreturn".to_string()),
            _ => None,
        }).collect();
        let attr_str = if attrs.is_empty() { String::new() } else { format!(" #{}", attrs.join(" ")) };

        self.emit_line(&format!(
            "define {} @{}({}){} {{",
            ret_ty, func.name, params.join(", "), attr_str
        ));
        self.indent += 1;

        for block in &func.blocks {
            self.emit_line(&format!("{}:", block.label));
            self.indent += 1;

            for op in &block.ops {
                self.generate_op(op);
            }

            self.generate_terminator(&block.terminator);
            self.indent -= 1;
        }

        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");
    }

    fn generate_op(&mut self, op: &IROp) {
        let result_str = op.result.as_ref()
            .map(|r| format!("{} = ", r.name))
            .unwrap_or_default();

        let instr = match &op.opcode {
            IROpcode::ConstInt(val, bits) => {
                format!("{}add i{} 0, {}", result_str, bits, val)
            }
            IROpcode::ConstFloat(val, bits) => {
                let ty = match bits { 32 => "float", 64 => "double", _ => "double" };
                format!("{}fadd {} 0.0, {:e}", result_str, ty, val)
            }
            IROpcode::ConstBool(val) => {
                format!("{}add i1 0, {}", result_str, if *val { 1 } else { 0 })
            }
            IROpcode::Add(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}add {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::Sub(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}sub {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::Mul(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}mul {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::Div(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}sdiv {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::FAdd(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}fadd {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::FSub(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}fsub {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::FMul(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}fmul {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::FDiv(a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                format!("{}fdiv {} {}, {}", result_str, ty, a.name, b.name)
            }
            IROpcode::ICmp(pred, a, b) => {
                let ty = self.ir_type_to_llvm(&a.ty);
                let pred_str = match pred {
                    CmpPredicate::Eq => "eq",
                    CmpPredicate::Ne => "ne",
                    CmpPredicate::Lt => "slt",
                    CmpPredicate::Le => "sle",
                    CmpPredicate::Gt => "sgt",
                    CmpPredicate::Ge => "sge",
                    _ => "eq",
                };
                format!("{}icmp {} {} {}, {}", result_str, pred_str, ty, a.name, b.name)
            }
            IROpcode::Call(name, args) => {
                let arg_strs: Vec<String> = args.iter()
                    .map(|a| format!("{} {}", self.ir_type_to_llvm(&a.ty), a.name))
                    .collect();
                let ret_ty = op.result.as_ref()
                    .map(|r| self.ir_type_to_llvm(&r.ty))
                    .unwrap_or("void".to_string());
                format!("{}call {} @{}({})", result_str, ret_ty, name, arg_strs.join(", "))
            }
            IROpcode::StackAlloc(ty) => {
                let llvm_ty = self.ir_type_to_llvm(ty);
                format!("{}alloca {}", result_str, llvm_ty)
            }
            IROpcode::HeapAlloc(ty) => {
                let llvm_ty = self.ir_type_to_llvm(ty);
                format!("{}call i8* @chronos_alloc(i64 sizeof({}))", result_str, llvm_ty)
            }
            IROpcode::Load(ptr) => {
                if let IRType::Pointer(inner) = &ptr.ty {
                    let ty = self.ir_type_to_llvm(inner);
                    format!("{}load {}, {}* {}", result_str, ty, ty, ptr.name)
                } else {
                    format!("; load from non-pointer: {}", ptr.name)
                }
            }
            IROpcode::Store(val, ptr) => {
                let ty = self.ir_type_to_llvm(&val.ty);
                format!("store {} {}, {}* {}", ty, val.name, ty, ptr.name)
            }
            IROpcode::Free(val) => {
                format!("call void @chronos_free(i8* {})", val.name)
            }
            IROpcode::Drop(val) => {
                format!("call void @chronos_drop(i8* {}) ; deterministic destructor", val.name)
            }
            IROpcode::Phi(entries) => {
                let ty = entries.first()
                    .map(|(v, _)| self.ir_type_to_llvm(&v.ty))
                    .unwrap_or("i64".to_string());
                let phi_args: Vec<String> = entries.iter()
                    .map(|(v, label)| format!("[{}, %{}]", v.name, label))
                    .collect();
                format!("{}phi {} {}", result_str, ty, phi_args.join(", "))
            }
            // Tensor ops get lowered to runtime library calls.
            IROpcode::Gemm { a, b, .. } => {
                format!("{}call i8* @chronos_gemm(i8* {}, i8* {})",
                    result_str, a.name, b.name)
            }
            IROpcode::Attention { query, key, value, num_heads, .. } => {
                format!("{}call i8* @chronos_attention(i8* {}, i8* {}, i8* {}, i64 {})",
                    result_str, query.name, key.name, value.name, num_heads)
            }
            IROpcode::Relu(x) => {
                format!("{}call i8* @chronos_relu(i8* {})", result_str, x.name)
            }
            IROpcode::Gelu(x) => {
                format!("{}call i8* @chronos_gelu(i8* {})", result_str, x.name)
            }
            IROpcode::Softmax(x, dim) => {
                format!("{}call i8* @chronos_softmax(i8* {}, i32 {})",
                    result_str, x.name, dim)
            }
            IROpcode::DeviceTransfer(val, from, to) => {
                format!("; device_transfer {} from {:?} to {:?}", val.name, from, to)
            }
            _ => {
                format!("; unimplemented opcode: {:?}", op.opcode)
            }
        };

        self.emit_line(&instr);
    }

    fn generate_terminator(&mut self, term: &IRTerminator) {
        match term {
            IRTerminator::Return(Some(val)) => {
                let ty = self.ir_type_to_llvm(&val.ty);
                self.emit_line(&format!("ret {} {}", ty, val.name));
            }
            IRTerminator::Return(None) => {
                self.emit_line("ret void");
            }
            IRTerminator::Branch(target) => {
                self.emit_line(&format!("br label %{}", target));
            }
            IRTerminator::CondBranch { condition, true_block, false_block } => {
                self.emit_line(&format!(
                    "br i1 {}, label %{}, label %{}",
                    condition.name, true_block, false_block
                ));
            }
            IRTerminator::Switch { value, default, cases } => {
                let ty = self.ir_type_to_llvm(&value.ty);
                let case_strs: Vec<String> = cases.iter()
                    .map(|(val, label)| format!("{} {}, label %{}", ty, val, label))
                    .collect();
                self.emit_line(&format!(
                    "switch {} {}, label %{} [{}]",
                    ty, value.name, default, case_strs.join(" ")
                ));
            }
            IRTerminator::Unreachable => {
                self.emit_line("unreachable");
            }
        }
    }
}

impl CodeGenBackend for LLVMBackend {
    type Output = String;

    fn generate(&mut self, module: &IRModule) -> String {
        self.output.clear();

        // Module header.
        self.emit_line(&format!("; ModuleID = '{}'", module.name));
        self.emit_line("; Chronos Language Compiler — LLVM Backend");
        self.emit_line(&format!("; Version: {} (track: {})",
            module.version_info.version, module.version_info.track));
        self.emit_line("");

        // Type declarations.
        for ty_decl in &module.type_declarations {
            let fields: Vec<String> = ty_decl.fields.iter()
                .map(|f| self.ir_type_to_llvm(f))
                .collect();
            let packed = if ty_decl.packed { "<" } else { "" };
            let packed_end = if ty_decl.packed { ">" } else { "" };
            self.emit_line(&format!(
                "%{} = type {}{{ {} }}{}",
                ty_decl.name, packed, fields.join(", "), packed_end
            ));
        }
        self.emit_line("");

        // Global variables.
        for global in &module.globals {
            let ty = self.ir_type_to_llvm(&global.ty);
            let qualifier = if global.is_const { "constant" } else { "global" };
            self.emit_line(&format!("@{} = {} {} zeroinitializer", global.name, qualifier, ty));
        }
        self.emit_line("");

        // Runtime function declarations (no GC — deterministic memory management).
        self.emit_line("; Chronos runtime (no garbage collector)");
        self.emit_line("declare i8* @chronos_alloc(i64)");
        self.emit_line("declare void @chronos_free(i8*)");
        self.emit_line("declare void @chronos_drop(i8*)");
        self.emit_line("declare void @chronos_rc_inc(i8*)");
        self.emit_line("declare void @chronos_rc_dec(i8*)");
        self.emit_line("");
        self.emit_line("; AI runtime (Feature 5)");
        self.emit_line("declare i8* @chronos_gemm(i8*, i8*)");
        self.emit_line("declare i8* @chronos_attention(i8*, i8*, i8*, i64)");
        self.emit_line("declare i8* @chronos_relu(i8*)");
        self.emit_line("declare i8* @chronos_gelu(i8*)");
        self.emit_line("declare i8* @chronos_softmax(i8*, i32)");
        self.emit_line("");

        // Functions.
        for func in &module.functions {
            if func.device == IRDevice::Cpu {
                self.generate_function(func);
            }
        }

        self.output.clone()
    }

    fn name(&self) -> &str { "LLVM" }
}


// =====================================================================
// BACKEND 2: CUDA PTX (for NVIDIA GPUs)
// =====================================================================

pub struct CUDABackend {
    output: String,
}

impl CUDABackend {
    pub fn new() -> Self {
        Self { output: String::new() }
    }

    fn ir_type_to_ptx(&self, ty: &IRType) -> &str {
        match ty {
            IRType::Bool => ".pred",
            IRType::Int(8) => ".s8",
            IRType::Int(16) => ".s16",
            IRType::Int(32) => ".s32",
            IRType::Int(64) => ".s64",
            IRType::UInt(8) => ".u8",
            IRType::UInt(16) => ".u16",
            IRType::UInt(32) => ".u32",
            IRType::UInt(64) => ".u64",
            IRType::Float(16) => ".f16",
            IRType::Float(32) => ".f32",
            IRType::Float(64) => ".f64",
            IRType::BFloat16 => ".bf16",
            _ => ".b64",
        }
    }
}

impl CodeGenBackend for CUDABackend {
    type Output = String;

    fn generate(&mut self, module: &IRModule) -> String {
        self.output.clear();
        self.output.push_str("// Chronos → CUDA PTX\n");
        self.output.push_str(".version 8.0\n");
        self.output.push_str(".target sm_80\n");  // Ampere architecture
        self.output.push_str(".address_size 64\n\n");

        for func in &module.functions {
            if matches!(func.device, IRDevice::Gpu(_)) {
                // Mark as kernel entry point.
                self.output.push_str(&format!(".visible .entry {}(\n", func.name));

                for (i, param) in func.params.iter().enumerate() {
                    let ptx_ty = self.ir_type_to_ptx(&param.ty);
                    let space = match &param.memory_space {
                        Some(IRMemorySpace::Shared) => ".shared",
                        Some(IRMemorySpace::Constant) => ".const",
                        _ => ".param",
                    };
                    let comma = if i < func.params.len() - 1 { "," } else { "" };
                    self.output.push_str(&format!(
                        "  {} {} %param_{}{}  // {}\n",
                        space, ptx_ty, i, comma, param.name
                    ));
                }
                self.output.push_str(")\n{\n");

                // Thread index computation.
                self.output.push_str("  .reg .u32 %tid_x, %tid_y, %tid_z;\n");
                self.output.push_str("  .reg .u32 %bid_x, %bid_y, %bid_z;\n");
                self.output.push_str("  .reg .u32 %bdim_x;\n");
                self.output.push_str("  .reg .u64 %global_id;\n\n");
                self.output.push_str("  mov.u32 %tid_x, %tid.x;\n");
                self.output.push_str("  mov.u32 %bid_x, %ctaid.x;\n");
                self.output.push_str("  mov.u32 %bdim_x, %ntid.x;\n");
                self.output.push_str("  mad.wide.u32 %global_id, %bid_x, %bdim_x, %tid_x;\n\n");

                // Lower the body (simplified — each IR op → PTX instruction).
                for block in &func.blocks {
                    self.output.push_str(&format!("{}:\n", block.label));
                    for op in &block.ops {
                        match &op.opcode {
                            IROpcode::FAdd(a, b) => {
                                let res = op.result.as_ref().unwrap();
                                self.output.push_str(&format!(
                                    "  add.f32 {}, {}, {};\n",
                                    res.name, a.name, b.name
                                ));
                            }
                            IROpcode::FMul(a, b) => {
                                let res = op.result.as_ref().unwrap();
                                self.output.push_str(&format!(
                                    "  mul.f32 {}, {}, {};\n",
                                    res.name, a.name, b.name
                                ));
                            }
                            IROpcode::Gemm { a, b, .. } => {
                                self.output.push_str(&format!(
                                    "  // GEMM: {} @ {} — dispatched to cuBLAS\n",
                                    a.name, b.name
                                ));
                                self.output.push_str(
                                    "  // Using WMMA instructions for tensor core acceleration\n"
                                );
                                self.output.push_str(
                                    "  wmma.load.a.sync.aligned.m16n16k16.shared.f16 {}, [%smem_a], 16;\n"
                                );
                            }
                            _ => {
                                self.output.push_str(&format!("  // {:?}\n", op.opcode));
                            }
                        }
                    }

                    match &block.terminator {
                        IRTerminator::Return(_) => {
                            self.output.push_str("  ret;\n");
                        }
                        IRTerminator::Branch(target) => {
                            self.output.push_str(&format!("  bra {};\n", target));
                        }
                        IRTerminator::CondBranch { condition, true_block, false_block } => {
                            self.output.push_str(&format!(
                                "  @{} bra {};\n  bra {};\n",
                                condition.name, true_block, false_block
                            ));
                        }
                        _ => {}
                    }
                }

                self.output.push_str("}\n\n");
            }
        }

        self.output.clone()
    }

    fn name(&self) -> &str { "CUDA PTX" }
}


// =====================================================================
// BACKEND 3: XLA HLO (for Google TPUs)
// =====================================================================

pub struct XLABackend {
    output: String,
}

impl XLABackend {
    pub fn new() -> Self {
        Self { output: String::new() }
    }

    fn ir_type_to_xla(&self, ty: &IRType) -> &str {
        match ty {
            IRType::Bool => "pred",
            IRType::Int(8) => "s8",
            IRType::Int(16) => "s16",
            IRType::Int(32) => "s32",
            IRType::Int(64) => "s64",
            IRType::Float(16) => "f16",
            IRType::Float(32) => "f32",
            IRType::Float(64) => "f64",
            IRType::BFloat16 => "bf16",
            _ => "f32",
        }
    }
}

impl CodeGenBackend for XLABackend {
    type Output = String;

    fn name(&self) -> &str { "xla-hlo" }

    fn generate(&mut self, module: &IRModule) -> String {
        self.output.clear();
        self.output.push_str("// Chronos → XLA HLO (for TPU execution)\n");
        self.output.push_str("// Auto-generated by Chronos compiler\n\n");

        for func in &module.functions {
            if matches!(func.device, IRDevice::Tpu(_)) {
                self.output.push_str(&format!("HloModule {}\n\n", func.name));
                self.output.push_str(&format!("ENTRY {} {{\n", func.name));

                // Parameters
                for (i, param) in func.params.iter().enumerate() {
                    let xla_ty = self.ir_type_to_xla(&param.ty);
                    if let IRType::Tensor { shape, .. } = &param.ty {
                        self.output.push_str(&format!(
                            "  p{}: {} = parameter({})\n",
                            i, xla_ty, i
                        ));
                    } else {
                        self.output.push_str(&format!(
                            "  p{}: {} = parameter({})\n",
                            i, xla_ty, i
                        ));
                    }
                }
                self.output.push_str("}\n\n");
            }
        }
        self.output.clone()
    }
}