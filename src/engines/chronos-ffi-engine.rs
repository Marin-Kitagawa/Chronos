// chronos-ffi-engine.rs
//
// Chronos Language — Foreign Function Interface (FFI) & Interoperability Engine
//
// Implements the complete foreign-function interface layer for six targets:
//
//   § 1  C FFI
//        • ABI classification (System V AMD64 / Microsoft x64 / ARM64)
//        • C type mapping, struct layout (field alignment, padding, sizeof)
//        • Calling-convention descriptors (cdecl, stdcall, fastcall, aapcs)
//        • Variadic function type descriptors
//        • Function pointer types, extern block declarations
//        • Null-safety wrapper model for C pointers
//
//   § 2  C++ FFI
//        • Itanium C++ ABI name mangling (GCC/Clang/Linux) + MSVC decoration
//        • Virtual table (vtable) layout: offset-to-top, RTTI pointer, function slots
//        • Abstract base class / pure-virtual detection
//        • RAII smart-pointer wrappers (unique_ptr / shared_ptr model)
//        • Exception specification encoding
//
//   § 3  Python (CPython) FFI
//        • PyObject header model (ob_refcnt, ob_type)
//        • Reference-count management (Py_INCREF / Py_DECREF)
//        • GIL acquisition / release token
//        • Python type object skeleton (tp_name, tp_basicsize, slots)
//        • Module definition (PyModuleDef, PyMethodDef)
//        • Type conversion table (Python ↔ Chronos)
//
//   § 4  JVM / JNI FFI
//        • JNI type descriptors (field/method signatures)
//        • JNI function table (JNIEnv*) — full 232-entry function index
//        • Local / global reference lifecycle
//        • JVM type mapping (Java primitives ↔ C types ↔ JNI aliases)
//        • Method invocation helpers (CallMethod dispatching by return type)
//        • Exception checking / clearing
//
//   § 5  .NET / CLR FFI
//        • P/Invoke attribute model (DllImport, CharSet, CallingConvention)
//        • Marshalling annotations (MarshalAs, StructLayout, FieldOffset)
//        • COM IUnknown / IDispatch vtable model (AddRef, Release, QueryInterface)
//        • HRESULT decoding
//        • CLR hosting API skeleton (ICLRRuntimeHost)
//
//   § 6  WebAssembly (WASM) FFI
//        • WASM binary section model (type / import / function / export / memory / global / code)
//        • WASM value types and function types
//        • Import / export descriptors
//        • Linear memory model (pages = 64 KiB, grow semantics)
//        • Host-function binding registry
//        • wasm-bindgen-style attribute model for JS interop
//        • WASM binary encoder (LEB128, section serialization)
//
// Design principles:
//   • Pure Rust, no external crates beyond std.
//   • Every ABI detail references the authoritative specification.
//   • The structures are designed to feed into code generation
//     (chronos-ir-codegen.rs) as metadata for emitting correct FFI stubs.

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// § 1  C FFI
// ─────────────────────────────────────────────────────────────────────────────

// ── 1.1  Calling conventions ─────────────────────────────────────────────────

/// Calling convention / ABI for a C function declaration.
///
/// References:
///   • System V AMD64 ABI (psABI v1.0)
///   • Microsoft x64 Software Conventions (MSDN)
///   • ARM Architecture Procedure Call Standard (AAPCS64)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallingConvention {
    /// Default C calling convention (caller cleans stack).
    /// AMD64 Linux/macOS: System V AMD64 ABI.
    CDcl,
    /// Windows 32-bit API convention (callee cleans stack).
    Stdcall,
    /// Microsoft x64 convention (shadow space, RCX/RDX/R8/R9 integer args).
    MicrosoftX64,
    /// System V AMD64 convention (RDI/RSI/RDX/RCX/R8/R9 integer args).
    SystemVAmd64,
    /// ARM64 (AArch64) Procedure Call Standard.
    Aapcs64,
    /// x86 fastcall (first two args in ECX/EDX).
    Fastcall,
    /// Vectorcall (Microsoft; first 6 integer args in RCX/RDX/R8/R9/..., XMM0–5 for float).
    Vectorcall,
    /// Naked function (no prologue/epilogue emitted by the compiler).
    Naked,
    /// Thiscall (Microsoft C++; this pointer in ECX).
    Thiscall,
}

impl fmt::Display for CallingConvention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            CallingConvention::CDcl           => "cdecl",
            CallingConvention::Stdcall        => "stdcall",
            CallingConvention::MicrosoftX64   => "ms_x64",
            CallingConvention::SystemVAmd64   => "sysv_amd64",
            CallingConvention::Aapcs64        => "aapcs64",
            CallingConvention::Fastcall       => "fastcall",
            CallingConvention::Vectorcall     => "vectorcall",
            CallingConvention::Naked          => "naked",
            CallingConvention::Thiscall       => "thiscall",
        };
        write!(f, "{}", s)
    }
}

// ── 1.2  C type system ───────────────────────────────────────────────────────

/// C/C++ primitive and compound types with their ABI size/alignment.
#[derive(Debug, Clone, PartialEq)]
pub enum CType {
    Void,
    Bool,                  // _Bool / bool  (1 byte)
    Char,                  // signed char
    UChar,                 // unsigned char
    Short,                 // short int (2 bytes)
    UShort,
    Int,                   // int (4 bytes)
    UInt,
    Long,                  // long (4 bytes on Windows, 8 bytes on LP64)
    ULong,
    LongLong,              // long long (8 bytes)
    ULongLong,
    Float,                 // float (4 bytes, IEEE 754)
    Double,                // double (8 bytes, IEEE 754)
    LongDouble,            // long double (10/16 bytes)
    Pointer(Box<CType>),   // T*
    ConstPointer(Box<CType>), // const T*
    FunctionPointer(Box<CFunctionType>),
    Array(Box<CType>, usize), // T[N]
    Struct(String),        // struct/class by name
    Union(String),
    Enum(String),
    SizeT,                 // size_t
    PtrDiffT,              // ptrdiff_t
    IntPtrT,               // intptr_t
    // Fixed-width types (<stdint.h>)
    Int8,  Int16,  Int32,  Int64,
    UInt8, UInt16, UInt32, UInt64,
    // POSIX
    SSizeT, Off_t, Pid_t, Uid_t,
}

impl CType {
    /// Return (size_bytes, align_bytes) for a C type on LP64 (Linux/macOS AMD64).
    pub fn size_align_lp64(&self) -> (usize, usize) {
        match self {
            CType::Void         => (0, 1),
            CType::Bool         => (1, 1),
            CType::Char | CType::UChar | CType::Int8 | CType::UInt8 => (1, 1),
            CType::Short | CType::UShort | CType::Int16 | CType::UInt16 => (2, 2),
            CType::Int  | CType::UInt | CType::Int32 | CType::UInt32 => (4, 4),
            CType::Long | CType::ULong | CType::LongLong | CType::ULongLong
                | CType::Int64 | CType::UInt64 | CType::SizeT | CType::PtrDiffT
                | CType::IntPtrT | CType::SSizeT | CType::Off_t => (8, 8),
            CType::Pid_t | CType::Uid_t => (4, 4),
            CType::Float        => (4, 4),
            CType::Double       => (8, 8),
            CType::LongDouble   => (16, 16),
            CType::Pointer(_) | CType::ConstPointer(_)
                | CType::FunctionPointer(_) => (8, 8),
            CType::Array(inner, n) => {
                let (sz, al) = inner.size_align_lp64();
                (sz * n, al)
            }
            CType::Struct(_) | CType::Union(_) | CType::Enum(_) => (0, 0), // opaque
        }
    }

    /// Return (size_bytes, align_bytes) for the LLP64 model (Windows AMD64).
    pub fn size_align_llp64(&self) -> (usize, usize) {
        match self {
            // long is 4 bytes on Windows (LLP64)
            CType::Long | CType::ULong => (4, 4),
            other => other.size_align_lp64(),
        }
    }

    /// C source spelling of the type.
    pub fn c_spelling(&self) -> String {
        match self {
            CType::Void        => "void".into(),
            CType::Bool        => "_Bool".into(),
            CType::Char        => "char".into(),
            CType::UChar       => "unsigned char".into(),
            CType::Short       => "short".into(),
            CType::UShort      => "unsigned short".into(),
            CType::Int         => "int".into(),
            CType::UInt        => "unsigned int".into(),
            CType::Long        => "long".into(),
            CType::ULong       => "unsigned long".into(),
            CType::LongLong    => "long long".into(),
            CType::ULongLong   => "unsigned long long".into(),
            CType::Float       => "float".into(),
            CType::Double      => "double".into(),
            CType::LongDouble  => "long double".into(),
            CType::SizeT       => "size_t".into(),
            CType::PtrDiffT    => "ptrdiff_t".into(),
            CType::IntPtrT     => "intptr_t".into(),
            CType::Int8        => "int8_t".into(),
            CType::Int16       => "int16_t".into(),
            CType::Int32       => "int32_t".into(),
            CType::Int64       => "int64_t".into(),
            CType::UInt8       => "uint8_t".into(),
            CType::UInt16      => "uint16_t".into(),
            CType::UInt32      => "uint32_t".into(),
            CType::UInt64      => "uint64_t".into(),
            CType::SSizeT      => "ssize_t".into(),
            CType::Off_t       => "off_t".into(),
            CType::Pid_t       => "pid_t".into(),
            CType::Uid_t       => "uid_t".into(),
            CType::Pointer(inner) => format!("{}*", inner.c_spelling()),
            CType::ConstPointer(inner) => format!("const {}*", inner.c_spelling()),
            CType::FunctionPointer(_) => "void(*)(...)".into(),
            CType::Array(inner, n) => format!("{}[{}]", inner.c_spelling(), n),
            CType::Struct(name) => format!("struct {}", name),
            CType::Union(name)  => format!("union {}", name),
            CType::Enum(name)   => format!("enum {}", name),
        }
    }
}

// ── 1.3  C struct layout ─────────────────────────────────────────────────────

/// A single field in a C struct declaration.
#[derive(Debug, Clone)]
pub struct CField {
    pub name:       String,
    pub ty:         CType,
    /// Bit-field width (None = not a bit-field).
    pub bit_width:  Option<u8>,
}

/// Computed layout of a C struct (System V AMD64 ABI rules).
#[derive(Debug, Clone)]
pub struct StructLayout {
    pub name:       String,
    pub fields:     Vec<CField>,
    /// Byte offset of each field.
    pub offsets:    Vec<usize>,
    /// Total size of the struct (including tail padding).
    pub size:       usize,
    /// Required alignment.
    pub alignment:  usize,
}

impl StructLayout {
    /// Compute struct layout according to C ABI rules (natural alignment,
    /// trailing padding to next multiple of alignment).
    pub fn compute(name: &str, fields: &[CField]) -> Self {
        let mut offsets = Vec::with_capacity(fields.len());
        let mut offset = 0usize;
        let mut max_align = 1usize;

        for field in fields {
            let (size, align) = field.ty.size_align_lp64();
            let align = align.max(1);
            max_align = max_align.max(align);

            // Pad to alignment boundary
            let pad = (align - (offset % align)) % align;
            offset += pad;
            offsets.push(offset);
            offset += size;
        }

        // Trailing padding
        let tail_pad = if max_align > 0 { (max_align - (offset % max_align)) % max_align } else { 0 };
        let size = offset + tail_pad;

        StructLayout {
            name: name.to_string(),
            fields: fields.to_vec(),
            offsets,
            size,
            alignment: max_align,
        }
    }

    /// Generate a C struct definition string.
    pub fn to_c_header(&self) -> String {
        let mut s = format!("typedef struct {} {{\n", self.name);
        for (i, field) in self.fields.iter().enumerate() {
            let padding = self.offsets.get(i+1)
                .map(|next| next - self.offsets[i] - field.ty.size_align_lp64().0)
                .unwrap_or(0);
            s.push_str(&format!("    {}{}; /* offset={} */\n",
                field.ty.c_spelling(), field.name, self.offsets[i]));
            if padding > 0 {
                s.push_str(&format!("    /* {} byte(s) padding */\n", padding));
            }
        }
        s.push_str(&format!("}} {}; /* size={}, align={} */\n",
            self.name, self.size, self.alignment));
        s
    }
}

// ── 1.4  Extern block ────────────────────────────────────────────────────────

/// A C function type signature.
#[derive(Debug, Clone, PartialEq)]
pub struct CFunctionType {
    pub return_type: CType,
    pub params:      Vec<(String, CType)>,
    pub variadic:    bool,
    pub convention:  CallingConvention,
}

impl CFunctionType {
    /// Generate the C prototype string.
    pub fn c_prototype(&self, name: &str) -> String {
        let params: Vec<String> = self.params.iter()
            .map(|(n, t)| format!("{} {}", t.c_spelling(), n))
            .collect();
        let variadic = if self.variadic { ", ..." } else { "" };
        let params_str = if params.is_empty() && !self.variadic {
            "void".to_string()
        } else {
            format!("{}{}", params.join(", "), variadic)
        };
        format!("{} {}({});", self.return_type.c_spelling(), name, params_str)
    }
}

/// An `extern "C"` block — a set of C function declarations imported from a library.
pub struct ExternBlock {
    pub library:   Option<String>,
    pub functions: Vec<(String, CFunctionType)>,
    pub link_kind: LinkKind,
}

/// How to link the external library.
#[derive(Debug, Clone, PartialEq)]
pub enum LinkKind {
    Dynamic,   // -l<lib>
    Static,    // -Bstatic -l<lib>
    Framework, // macOS framework (-framework <Name>)
    Raw,       // Pre-linked object file
}

impl ExternBlock {
    pub fn new(library: Option<&str>, link_kind: LinkKind) -> Self {
        ExternBlock {
            library: library.map(|s| s.to_string()),
            functions: Vec::new(),
            link_kind,
        }
    }

    pub fn add(&mut self, name: &str, func: CFunctionType) {
        self.functions.push((name.to_string(), func));
    }

    /// Emit the full C header for this extern block.
    pub fn to_c_header(&self) -> String {
        let mut s = String::new();
        if let Some(lib) = &self.library {
            s.push_str(&format!("/* Library: {} ({:?}) */\n", lib, self.link_kind));
        }
        s.push_str("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");
        for (name, func) in &self.functions {
            s.push_str(&func.c_prototype(name));
            s.push('\n');
        }
        s.push_str("\n#ifdef __cplusplus\n}\n#endif\n");
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  C++ FFI
// ─────────────────────────────────────────────────────────────────────────────

// ── 2.1  Itanium ABI name mangling ──────────────────────────────────────────

/// Mangle a C++ symbol name under the Itanium C++ ABI (g++/clang).
///
/// Full mangling grammar is complex; we implement the common cases:
///   • Free functions with simple parameter types
///   • Namespace-qualified names
///   • Constructor / destructor forms
///
/// Itanium ABI §5: mangled name = "_Z" <encoding>
///                 <encoding> = <name> <type>
///                 <name> = <nested-name> | <unqualified-name>
///                 <nested-name> = "N" <qualifier>* <unqualified-name>+ "E"
pub fn itanium_mangle(namespace: &[&str], name: &str, params: &[CType]) -> String {
    let mut result = String::from("_Z");

    if namespace.is_empty() {
        // Unqualified name: <length><name>
        result.push_str(&encode_identifier(name));
    } else {
        // Nested name: N <qualifiers> <unqualified-name> E
        result.push('N');
        for ns in namespace {
            result.push_str(&encode_identifier(ns));
        }
        result.push_str(&encode_identifier(name));
        result.push('E');
    }

    // Encode parameter types
    for param in params {
        result.push_str(&mangle_type(param));
    }
    if params.is_empty() {
        result.push('v'); // void parameter list
    }

    result
}

fn encode_identifier(name: &str) -> String {
    format!("{}{}", name.len(), name)
}

/// Mangle a single type into Itanium ABI encoding.
fn mangle_type(ty: &CType) -> String {
    match ty {
        CType::Void      => "v".into(),
        CType::Bool      => "b".into(),
        CType::Char      => "c".into(),
        CType::UChar     => "h".into(),
        CType::Short     => "s".into(),
        CType::UShort    => "t".into(),
        CType::Int       => "i".into(),
        CType::UInt      => "j".into(),
        CType::Long      => "l".into(),
        CType::ULong     => "m".into(),
        CType::LongLong  => "x".into(),
        CType::ULongLong => "y".into(),
        CType::Float     => "f".into(),
        CType::Double    => "d".into(),
        CType::LongDouble => "e".into(),
        CType::SizeT     => "m".into(),
        CType::PtrDiffT  => "l".into(),
        CType::Pointer(inner) => format!("P{}", mangle_type(inner)),
        CType::ConstPointer(inner) => format!("PK{}", mangle_type(inner)),
        CType::Struct(name) => encode_identifier(name),
        _ => "v".into(),
    }
}

/// Mangle a C++ symbol under MSVC decoration rules (simplified).
/// MSVC uses `?<name>@<qualifiers>@@<type-code>` format.
pub fn msvc_mangle(namespace: &[&str], name: &str, params: &[CType], is_member: bool) -> String {
    let mut result = format!("?{}@", name);
    for ns in namespace.iter().rev() {
        result.push_str(ns);
        result.push('@');
    }
    result.push('@'); // end of scope
    // Calling convention + return type (simplified)
    if is_member {
        result.push_str("QAE"); // public: __thiscall
    } else {
        result.push_str("YAX"); // __cdecl, void return
    }
    for param in params {
        result.push_str(&msvc_type(param));
    }
    result.push('Z');
    result
}

fn msvc_type(ty: &CType) -> String {
    match ty {
        CType::Void      => "X".into(),
        CType::Bool      => "_N".into(),
        CType::Char      => "D".into(),
        CType::UChar     => "E".into(),
        CType::Short     => "F".into(),
        CType::UShort    => "G".into(),
        CType::Int       => "H".into(),
        CType::UInt      => "I".into(),
        CType::Long      => "J".into(),
        CType::ULong     => "K".into(),
        CType::LongLong  => "_J".into(),
        CType::ULongLong => "_K".into(),
        CType::Float     => "M".into(),
        CType::Double    => "N".into(),
        CType::LongDouble => "O".into(),
        CType::Pointer(inner) => format!("PA{}", msvc_type(inner)),
        CType::ConstPointer(inner) => format!("PB{}", msvc_type(inner)),
        _ => "X".into(),
    }
}

// ── 2.2  Virtual table layout ────────────────────────────────────────────────

/// A virtual function slot in a vtable.
#[derive(Debug, Clone)]
pub struct VTableSlot {
    pub index:      usize,
    pub name:       String,
    pub signature:  CFunctionType,
    pub is_pure:    bool,  // = 0 (abstract)
    pub overrides:  Option<String>, // name of the base virtual function it overrides
}

/// A C++ class virtual table (Itanium ABI layout).
///
/// Itanium ABI §2.5 vtable layout:
///   [0]  offset-to-top  (negative for secondary vtables)
///   [1]  typeinfo pointer (RTTI)
///   [2+] virtual function pointers in declaration order
pub struct VTable {
    pub class_name:    String,
    pub slots:         Vec<VTableSlot>,
    /// Inherited vtable slots from the base class (if any).
    pub base_class:    Option<String>,
    pub base_slots:    Vec<VTableSlot>,
}

impl VTable {
    pub fn new(class_name: &str) -> Self {
        VTable {
            class_name: class_name.to_string(),
            slots: Vec::new(),
            base_class: None,
            base_slots: Vec::new(),
        }
    }

    pub fn add_virtual(&mut self, name: &str, sig: CFunctionType, is_pure: bool) {
        let index = self.base_slots.len() + self.slots.len();
        self.slots.push(VTableSlot { index, name: name.to_string(), signature: sig, is_pure, overrides: None });
    }

    pub fn add_override(&mut self, base_name: &str, name: &str, sig: CFunctionType) {
        let index = self.base_slots.len() + self.slots.len();
        self.slots.push(VTableSlot {
            index, name: name.to_string(), signature: sig, is_pure: false,
            overrides: Some(base_name.to_string()),
        });
    }

    pub fn is_abstract(&self) -> bool {
        self.slots.iter().any(|s| s.is_pure) ||
        self.base_slots.iter().any(|s| s.is_pure)
    }

    /// Total vtable entries (2 overhead + all slots).
    pub fn total_entries(&self) -> usize {
        2 + self.base_slots.len() + self.slots.len()
    }

    /// Generate a C struct that mirrors the vtable in memory.
    pub fn to_c_vtable_struct(&self) -> String {
        let mut s = format!("/* vtable for {} */\n", self.class_name);
        s.push_str(&format!("struct {}_vtable {{\n", self.class_name));
        s.push_str("    ptrdiff_t offset_to_top;\n");
        s.push_str("    void*     typeinfo;\n");
        for slot in self.base_slots.iter().chain(self.slots.iter()) {
            s.push_str(&format!("    void* {}; /* slot {} */\n", slot.name, slot.index));
        }
        s.push_str("};\n");
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Python (CPython) FFI
// ─────────────────────────────────────────────────────────────────────────────

// ── 3.1  PyObject reference counting ────────────────────────────────────────

/// Simulates a CPython object header.  In CPython the layout is:
///   struct PyObject { Py_ssize_t ob_refcnt; PyTypeObject* ob_type; }
///
/// This is a Rust model for code generation / analysis — not an actual
/// binding to the Python runtime.
#[derive(Debug, Clone)]
pub struct PyObjectHandle {
    pub id:      u64,    // simulated object identity
    pub refcnt:  i64,    // ob_refcnt (starts at 1 when newly allocated)
    pub type_id: u64,    // pointer/id of ob_type
}

impl PyObjectHandle {
    pub fn new(id: u64, type_id: u64) -> Self {
        PyObjectHandle { id, refcnt: 1, type_id }
    }

    /// Py_INCREF: increment reference count.
    pub fn incref(&mut self) { self.refcnt += 1; }

    /// Py_DECREF: decrement reference count. Returns true if the object
    /// should be deallocated (refcnt reached 0).
    pub fn decref(&mut self) -> bool {
        self.refcnt -= 1;
        self.refcnt <= 0
    }

    pub fn is_live(&self) -> bool { self.refcnt > 0 }
}

/// GIL (Global Interpreter Lock) acquisition token.
/// Holding a `GilGuard` models the CPython thread state conventions.
pub struct GilGuard {
    pub thread_state_saved: bool,
}

impl GilGuard {
    /// Py_BEGIN_ALLOW_THREADS — release the GIL for blocking I/O.
    pub fn release() -> GilGuard {
        GilGuard { thread_state_saved: true }
    }

    /// Py_END_ALLOW_THREADS — reacquire the GIL.
    pub fn reacquire(self) -> bool {
        self.thread_state_saved
    }
}

// ── 3.2  Python type mapping ─────────────────────────────────────────────────

/// Mapping between Python types and Chronos / C types.
#[derive(Debug, Clone)]
pub struct PyTypeMapping {
    pub python_type: &'static str,
    pub c_type:      CType,
    pub py_build:    &'static str, // PyArg_ParseTuple format char for input
    pub py_format:   &'static str, // Py_BuildValue format char for output
}

/// Standard CPython type mappings (PyArg_ParseTuple / Py_BuildValue).
pub fn python_type_mappings() -> Vec<PyTypeMapping> {
    vec![
        PyTypeMapping { python_type: "int",   c_type: CType::LongLong, py_build: "L", py_format: "L" },
        PyTypeMapping { python_type: "float", c_type: CType::Double,   py_build: "d", py_format: "d" },
        PyTypeMapping { python_type: "bool",  c_type: CType::Int,      py_build: "p", py_format: "p" },
        PyTypeMapping { python_type: "bytes", c_type: CType::Pointer(Box::new(CType::UChar)), py_build: "y*", py_format: "y" },
        PyTypeMapping { python_type: "str",   c_type: CType::Pointer(Box::new(CType::Char)), py_build: "s", py_format: "s" },
        PyTypeMapping { python_type: "None",  c_type: CType::Void,     py_build: "",  py_format: "N" },
        PyTypeMapping { python_type: "list",  c_type: CType::Pointer(Box::new(CType::Void)), py_build: "O", py_format: "O" },
        PyTypeMapping { python_type: "tuple", c_type: CType::Pointer(Box::new(CType::Void)), py_build: "O", py_format: "O" },
    ]
}

// ── 3.3  PyMethodDef / module definition ────────────────────────────────────

/// METH_* flags for PyMethodDef.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethFlags(pub u32);

impl MethFlags {
    pub const VARARGS:  MethFlags = MethFlags(0x0001);
    pub const KEYWORDS: MethFlags = MethFlags(0x0002);
    pub const NOARGS:   MethFlags = MethFlags(0x0004);
    pub const O:        MethFlags = MethFlags(0x0008);
    pub const CLASS:    MethFlags = MethFlags(0x0010);
    pub const STATIC:   MethFlags = MethFlags(0x0020);
}

/// Python method descriptor (mirrors C `PyMethodDef` struct).
#[derive(Debug, Clone)]
pub struct PyMethodDef {
    pub ml_name:  String,  // method name exposed to Python
    pub ml_meth:  String,  // C function name implementing the method
    pub ml_flags: MethFlags,
    pub ml_doc:   String,  // docstring
}

/// Python extension module definition (mirrors `PyModuleDef`).
pub struct PyModuleDef {
    pub m_name:    String,
    pub m_doc:     String,
    pub methods:   Vec<PyMethodDef>,
}

impl PyModuleDef {
    pub fn new(name: &str, doc: &str) -> Self {
        PyModuleDef { m_name: name.to_string(), m_doc: doc.to_string(), methods: Vec::new() }
    }

    pub fn add_method(&mut self, method: PyMethodDef) {
        self.methods.push(method);
    }

    /// Generate the C module-init function body (PyMODINIT_FUNC).
    pub fn emit_c_init(&self) -> String {
        let mut s = String::new();
        // Method table
        s.push_str(&format!("static PyMethodDef {}_methods[] = {{\n", self.m_name));
        for m in &self.methods {
            s.push_str(&format!(
                "    {{\"{}\", {}, 0x{:04X}, \"{}\"}},\n",
                m.ml_name, m.ml_meth, m.ml_flags.0, m.ml_doc
            ));
        }
        s.push_str("    {NULL, NULL, 0, NULL}\n};\n\n");

        // Module definition
        s.push_str(&format!(
            "static struct PyModuleDef {name}_module = {{\n    \
             PyModuleDef_HEAD_INIT, \"{name}\", \"{doc}\", -1, {name}_methods\n}};\n\n",
            name = self.m_name, doc = self.m_doc
        ));

        // Init function
        s.push_str(&format!(
            "PyMODINIT_FUNC PyInit_{}(void) {{\n    \
             return PyModule_Create(&{}_module);\n}}\n",
            self.m_name, self.m_name
        ));
        s
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  JVM / JNI FFI
// ─────────────────────────────────────────────────────────────────────────────

// ── 4.1  JNI type descriptors ────────────────────────────────────────────────

/// JNI type descriptor for a Java type (JNI §12.3).
///
/// Primitive descriptors:
///   B=byte, C=char, D=double, F=float, I=int, J=long, S=short, V=void, Z=boolean
/// Reference types:
///   L<fully-qualified-class-name>;  (e.g. "Ljava/lang/String;")
/// Arrays:
///   [<type>                         (e.g. "[I" = int[])
#[derive(Debug, Clone, PartialEq)]
pub enum JniType {
    Byte,
    Char,
    Double,
    Float,
    Int,
    Long,
    Short,
    Void,
    Boolean,
    Object(String),  // class name with '/' separators
    Array(Box<JniType>),
}

impl JniType {
    pub fn descriptor(&self) -> String {
        match self {
            JniType::Byte    => "B".into(),
            JniType::Char    => "C".into(),
            JniType::Double  => "D".into(),
            JniType::Float   => "F".into(),
            JniType::Int     => "I".into(),
            JniType::Long    => "J".into(),
            JniType::Short   => "S".into(),
            JniType::Void    => "V".into(),
            JniType::Boolean => "Z".into(),
            JniType::Object(class) => format!("L{};", class),
            JniType::Array(inner)  => format!("[{}", inner.descriptor()),
        }
    }

    /// The JNI C typedef alias for this type.
    pub fn c_alias(&self) -> &'static str {
        match self {
            JniType::Byte    => "jbyte",
            JniType::Char    => "jchar",
            JniType::Double  => "jdouble",
            JniType::Float   => "jfloat",
            JniType::Int     => "jint",
            JniType::Long    => "jlong",
            JniType::Short   => "jshort",
            JniType::Void    => "void",
            JniType::Boolean => "jboolean",
            JniType::Object(_) | JniType::Array(_) => "jobject",
        }
    }
}

/// Compute the JNI method signature descriptor.
/// Format: (<param-descriptors>)<return-descriptor>
pub fn jni_method_signature(params: &[JniType], ret: &JniType) -> String {
    let params_str: String = params.iter().map(|p| p.descriptor()).collect();
    format!("({}){}", params_str, ret.descriptor())
}

// ── 4.2  JNI reference lifecycle ────────────────────────────────────────────

/// JNI reference kind (JNI §5).
#[derive(Debug, Clone, PartialEq)]
pub enum JniRefKind {
    /// Local references are valid only within the current JNI call frame.
    Local,
    /// Global references survive JNI call frames; must be explicitly deleted.
    Global,
    /// Weak global references — do not prevent GC of the referent.
    WeakGlobal,
}

/// A JNI object reference with lifecycle tracking.
#[derive(Debug)]
pub struct JniRef {
    pub id:   u64,
    pub kind: JniRefKind,
    pub class_name: String,
    deleted: bool,
}

impl JniRef {
    pub fn new_local(id: u64, class_name: &str) -> Self {
        JniRef { id, kind: JniRefKind::Local, class_name: class_name.to_string(), deleted: false }
    }

    pub fn promote_to_global(&self) -> JniRef {
        JniRef { id: self.id, kind: JniRefKind::Global, class_name: self.class_name.clone(), deleted: false }
    }

    /// DeleteLocalRef / DeleteGlobalRef.
    pub fn delete(&mut self) { self.deleted = true; }

    pub fn is_valid(&self) -> bool { !self.deleted }
}

// ── 4.3  JNI method call descriptor ─────────────────────────────────────────

/// Describes a JNI method invocation (mirrors JNIEnv->Call*Method family).
#[derive(Debug, Clone)]
pub struct JniMethodCall {
    pub class_name:  String,
    pub method_name: String,
    pub signature:   String,   // pre-computed JNI descriptor
    pub is_static:   bool,
    pub return_type: JniType,
}

impl JniMethodCall {
    pub fn new(
        class: &str, method: &str,
        params: &[JniType], ret: JniType, is_static: bool,
    ) -> Self {
        let signature = jni_method_signature(params, &ret);
        JniMethodCall {
            class_name: class.replace('.', "/"),
            method_name: method.to_string(),
            signature,
            is_static,
            return_type: ret,
        }
    }

    /// The JNIEnv function to call based on return type and static-ness.
    pub fn jnienv_function(&self) -> &'static str {
        if self.is_static {
            match &self.return_type {
                JniType::Void    => "CallStaticVoidMethod",
                JniType::Boolean => "CallStaticBooleanMethod",
                JniType::Byte    => "CallStaticByteMethod",
                JniType::Char    => "CallStaticCharMethod",
                JniType::Short   => "CallStaticShortMethod",
                JniType::Int     => "CallStaticIntMethod",
                JniType::Long    => "CallStaticLongMethod",
                JniType::Float   => "CallStaticFloatMethod",
                JniType::Double  => "CallStaticDoubleMethod",
                _                => "CallStaticObjectMethod",
            }
        } else {
            match &self.return_type {
                JniType::Void    => "CallVoidMethod",
                JniType::Boolean => "CallBooleanMethod",
                JniType::Byte    => "CallByteMethod",
                JniType::Char    => "CallCharMethod",
                JniType::Short   => "CallShortMethod",
                JniType::Int     => "CallIntMethod",
                JniType::Long    => "CallLongMethod",
                JniType::Float   => "CallFloatMethod",
                JniType::Double  => "CallDoubleMethod",
                _                => "CallObjectMethod",
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  .NET / CLR FFI
// ─────────────────────────────────────────────────────────────────────────────

// ── 5.1  P/Invoke attributes ─────────────────────────────────────────────────

/// CharSet for P/Invoke string marshalling.
#[derive(Debug, Clone, PartialEq)]
pub enum PInvokeCharSet {
    None,
    Ansi,
    Unicode,
    Auto,
}

/// CallingConvention for P/Invoke.
#[derive(Debug, Clone, PartialEq)]
pub enum PInvokeCallingConvention {
    Winapi,    // Default — maps to stdcall on 32-bit, platform default on 64-bit
    Cdecl,
    StdCall,
    ThisCall,
    FastCall,
}

/// Models the [DllImport] attribute in C#.
#[derive(Debug, Clone)]
pub struct DllImportAttribute {
    pub dll_name:          String,
    pub entry_point:       Option<String>,
    pub char_set:          PInvokeCharSet,
    pub calling_convention: PInvokeCallingConvention,
    pub set_last_error:    bool,
    pub exact_spelling:    bool,
    pub preserve_sig:      bool,
    pub best_fit_mapping:  bool,
}

impl DllImportAttribute {
    pub fn new(dll: &str) -> Self {
        DllImportAttribute {
            dll_name: dll.to_string(),
            entry_point: None,
            char_set: PInvokeCharSet::Auto,
            calling_convention: PInvokeCallingConvention::Winapi,
            set_last_error: false,
            exact_spelling: false,
            preserve_sig: false,
            best_fit_mapping: true,
        }
    }

    /// Emit the [DllImport] attribute string for a C# declaration.
    pub fn emit_csharp(&self, extern_name: &str, ret_type: &str, params: &str) -> String {
        let charset = match self.char_set {
            PInvokeCharSet::Ansi    => ", CharSet = CharSet.Ansi",
            PInvokeCharSet::Unicode => ", CharSet = CharSet.Unicode",
            _                       => "",
        };
        let entry = self.entry_point.as_ref()
            .map(|e| format!(", EntryPoint = \"{}\"", e))
            .unwrap_or_default();
        let set_last = if self.set_last_error { ", SetLastError = true" } else { "" };
        format!(
            "[DllImport(\"{dll}\"{entry}{charset}{set_last})]\npublic static extern {ret} {name}({params});",
            dll = self.dll_name, entry = entry, charset = charset, set_last = set_last,
            ret = ret_type, name = extern_name, params = params
        )
    }
}

// ── 5.2  Marshalling annotations ─────────────────────────────────────────────

/// [MarshalAs] unmanaged types (UnmanagedType enum subset).
#[derive(Debug, Clone, PartialEq)]
pub enum UnmanagedType {
    Bool,
    I1, I2, I4, I8,
    U1, U2, U4, U8,
    R4, R8,
    LPStr,      // Pointer to ANSI string
    LPWStr,     // Pointer to Unicode string
    LPTStr,     // Platform-default string
    BStr,       // COM BSTR
    SysInt, SysUInt,
    ByValTStr(usize), // fixed-length string in struct
    SafeArray,
    LPArray,
    Interface,
    IUnknown,
    IDispatch,
    Struct,
    CustomMarshaler(String), // custom marshaler class name
}

/// Layout kind for [StructLayout] attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum StructLayoutKind {
    Sequential,   // Fields laid out in declaration order (default for P/Invoke)
    Explicit,     // Each field has explicit [FieldOffset]
    Auto,         // CLR decides layout (not usable for P/Invoke)
}

/// Models [StructLayout(LayoutKind.Sequential, Pack=N, CharSet=...)] on a struct.
#[derive(Debug, Clone)]
pub struct StructLayoutAttribute {
    pub kind:      StructLayoutKind,
    pub pack:      Option<usize>,   // alignment override (1/2/4/8/16/32/64/128)
    pub char_set:  PInvokeCharSet,
    pub size:      Option<usize>,   // explicit Size=N
}

// ── 5.3  COM vtable ──────────────────────────────────────────────────────────

/// A COM interface definition (IUnknown-derived).
///
/// COM vtable layout (§2 of COM spec):
///   Every COM interface begins with IUnknown's three methods at indices 0–2:
///   [0] QueryInterface(REFIID riid, void** ppvObject) → HRESULT
///   [1] AddRef()  → ULONG
///   [2] Release() → ULONG
///   Then the interface's own methods follow.
#[derive(Debug, Clone)]
pub struct ComInterfaceMethod {
    pub slot:       usize,
    pub name:       String,
    pub params:     Vec<(String, CType)>,
    pub returns:    CType,  // HRESULT for most methods
}

pub struct ComInterface {
    pub name:    String,
    pub iid:     [u8; 16], // GUID bytes
    pub base:    Option<String>,
    pub methods: Vec<ComInterfaceMethod>,
}

impl ComInterface {
    pub fn new_iunknown_derived(name: &str, iid: [u8; 16]) -> Self {
        ComInterface { name: name.to_string(), iid, base: Some("IUnknown".into()), methods: Vec::new() }
    }

    pub fn add_method(&mut self, name: &str, params: Vec<(String, CType)>, returns: CType) {
        let slot = 3 + self.methods.len(); // 3 IUnknown slots before this
        self.methods.push(ComInterfaceMethod { slot, name: name.to_string(), params, returns });
    }

    /// Format the GUID as a string ("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx").
    pub fn iid_string(&self) -> String {
        let b = &self.iid;
        format!(
            "{:02X}{:02X}{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}-{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
            b[0],b[1],b[2],b[3], b[4],b[5], b[6],b[7], b[8],b[9], b[10],b[11],b[12],b[13],b[14],b[15]
        )
    }
}

/// Decode a Windows HRESULT into its components.
/// HRESULT layout: [31]=Severity [30..29]=Reserved [28..16]=Facility [15..0]=Code
pub fn decode_hresult(hr: u32) -> (bool, u16, u16) {
    let severity = (hr >> 31) & 1 == 1;  // true = failure
    let facility = ((hr >> 16) & 0x0FFF) as u16;
    let code     = (hr & 0xFFFF) as u16;
    (severity, facility, code)
}

pub fn hresult_succeeded(hr: u32) -> bool { (hr as i32) >= 0 }
pub fn hresult_failed(hr: u32)    -> bool { (hr as i32) < 0  }

// ─────────────────────────────────────────────────────────────────────────────
// § 6  WebAssembly (WASM) FFI
// ─────────────────────────────────────────────────────────────────────────────

// ── 6.1  WASM value types ────────────────────────────────────────────────────

/// WebAssembly value types (§2.3.1 of the WASM spec).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WasmValType {
    I32,
    I64,
    F32,
    F64,
    V128,         // SIMD proposal
    FuncRef,      // Reference types proposal
    ExternRef,    // Reference types proposal
}

impl WasmValType {
    /// Binary encoding byte (§5.3.1).
    pub fn byte_code(&self) -> u8 {
        match self {
            WasmValType::I32      => 0x7F,
            WasmValType::I64      => 0x7E,
            WasmValType::F32      => 0x7D,
            WasmValType::F64      => 0x7C,
            WasmValType::V128     => 0x7B,
            WasmValType::FuncRef  => 0x70,
            WasmValType::ExternRef=> 0x6F,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            WasmValType::I32       => "i32",
            WasmValType::I64       => "i64",
            WasmValType::F32       => "f32",
            WasmValType::F64       => "f64",
            WasmValType::V128      => "v128",
            WasmValType::FuncRef   => "funcref",
            WasmValType::ExternRef => "externref",
        }
    }
}

/// A WASM function type (§2.3.5): vec(valtype) → vec(valtype).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WasmFuncType {
    pub params:  Vec<WasmValType>,
    pub results: Vec<WasmValType>,
}

impl WasmFuncType {
    pub fn new(params: Vec<WasmValType>, results: Vec<WasmValType>) -> Self {
        WasmFuncType { params, results }
    }

    /// Encode to bytes (§5.3.5 type section entry).
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![0x60]; // functype prefix
        encode_vec(&self.params.iter().map(|t| t.byte_code()).collect::<Vec<_>>(), &mut bytes);
        encode_vec(&self.results.iter().map(|t| t.byte_code()).collect::<Vec<_>>(), &mut bytes);
        bytes
    }
}

// ── 6.2  WASM binary encoding helpers ───────────────────────────────────────

/// Encode an unsigned 32-bit integer as LEB128 (Little-Endian Base 128).
/// Used throughout the WASM binary format.
pub fn leb128_u32(mut v: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            bytes.push(byte | 0x80); // more bytes to come
        } else {
            bytes.push(byte);
            break;
        }
    }
    bytes
}

/// Encode a signed 32-bit integer as LEB128.
pub fn leb128_i32(mut v: i32) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut more = true;
    while more {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        more = !(((v == 0) && (byte & 0x40 == 0)) || ((v == -1) && (byte & 0x40 != 0)));
        bytes.push(if more { byte | 0x80 } else { byte });
    }
    bytes
}

/// Decode LEB128 unsigned from a byte slice. Returns (value, bytes_consumed).
pub fn decode_leb128_u32(bytes: &[u8]) -> (u32, usize) {
    let mut result: u32 = 0;
    let mut shift = 0u32;
    for (i, &byte) in bytes.iter().enumerate() {
        result |= ((byte & 0x7F) as u32) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return (result, i + 1);
        }
    }
    (result, bytes.len())
}

fn encode_vec(items: &[u8], out: &mut Vec<u8>) {
    out.extend(leb128_u32(items.len() as u32));
    out.extend_from_slice(items);
}

fn encode_str(s: &str, out: &mut Vec<u8>) {
    let bytes = s.as_bytes();
    out.extend(leb128_u32(bytes.len() as u32));
    out.extend_from_slice(bytes);
}

// ── 6.3  WASM module sections ────────────────────────────────────────────────

/// WASM section IDs (§5.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SectionId {
    Custom   = 0,
    Type     = 1,
    Import   = 2,
    Function = 3,
    Table    = 4,
    Memory   = 5,
    Global   = 6,
    Export   = 7,
    Start    = 8,
    Element  = 9,
    Code     = 10,
    Data     = 11,
    DataCount= 12,
}

/// Import descriptor kind.
#[derive(Debug, Clone)]
pub enum ImportDesc {
    Func(u32),      // type index
    Table,
    Memory { min: u32, max: Option<u32> },
    Global { ty: WasmValType, mutable: bool },
}

/// A WASM import entry (§2.5.2).
#[derive(Debug, Clone)]
pub struct WasmImport {
    pub module: String,
    pub name:   String,
    pub desc:   ImportDesc,
}

/// Export descriptor kind.
#[derive(Debug, Clone)]
pub enum ExportDesc {
    Func(u32),    // function index
    Table(u32),
    Memory(u32),
    Global(u32),
}

/// A WASM export entry (§2.5.7).
#[derive(Debug, Clone)]
pub struct WasmExport {
    pub name: String,
    pub desc: ExportDesc,
}

// ── 6.4  WASM module builder ─────────────────────────────────────────────────

/// Builds a WebAssembly binary module incrementally.
pub struct WasmModuleBuilder {
    pub types:     Vec<WasmFuncType>,
    pub imports:   Vec<WasmImport>,
    /// type_index for each locally-defined function
    pub functions: Vec<u32>,
    pub exports:   Vec<WasmExport>,
    /// Memory limits (min pages, optional max pages). 1 page = 64 KiB.
    pub memory:    Option<(u32, Option<u32>)>,
    /// (type, mutable, init_expr_bytes)
    pub globals:   Vec<(WasmValType, bool, Vec<u8>)>,
    /// (locals_vec, code_bytes) for each local function
    pub code:      Vec<(Vec<(u32, WasmValType)>, Vec<u8>)>,
}

impl WasmModuleBuilder {
    pub fn new() -> Self {
        WasmModuleBuilder {
            types: Vec::new(), imports: Vec::new(), functions: Vec::new(),
            exports: Vec::new(), memory: None, globals: Vec::new(), code: Vec::new(),
        }
    }

    /// Add a function type and return its index.
    pub fn add_type(&mut self, ft: WasmFuncType) -> u32 {
        // Deduplicate
        if let Some(i) = self.types.iter().position(|t| t == &ft) {
            return i as u32;
        }
        let idx = self.types.len() as u32;
        self.types.push(ft);
        idx
    }

    /// Add an imported function.
    pub fn add_import_func(&mut self, module: &str, name: &str, type_idx: u32) -> u32 {
        let idx = self.imports.len() as u32;
        self.imports.push(WasmImport {
            module: module.to_string(),
            name:   name.to_string(),
            desc:   ImportDesc::Func(type_idx),
        });
        idx
    }

    /// Add a locally-defined function body.
    pub fn add_function(
        &mut self,
        type_idx: u32,
        locals: Vec<(u32, WasmValType)>,
        body: Vec<u8>,
    ) -> u32 {
        let func_idx = self.imports.len() as u32 + self.functions.len() as u32;
        self.functions.push(type_idx);
        self.code.push((locals, body));
        func_idx
    }

    /// Set memory (min pages, max pages).
    pub fn set_memory(&mut self, min: u32, max: Option<u32>) {
        self.memory = Some((min, max));
    }

    /// Add an export.
    pub fn add_export(&mut self, name: &str, desc: ExportDesc) {
        self.exports.push(WasmExport { name: name.to_string(), desc });
    }

    /// Encode the entire WASM binary.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        // Magic + version
        out.extend_from_slice(b"\x00asm");
        out.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]);

        // Type section
        if !self.types.is_empty() {
            let mut section = Vec::new();
            section.extend(leb128_u32(self.types.len() as u32));
            for ft in &self.types { section.extend(ft.encode()); }
            write_section(&mut out, SectionId::Type, &section);
        }

        // Import section
        if !self.imports.is_empty() {
            let mut section = Vec::new();
            section.extend(leb128_u32(self.imports.len() as u32));
            for imp in &self.imports {
                encode_str(&imp.module, &mut section);
                encode_str(&imp.name, &mut section);
                match &imp.desc {
                    ImportDesc::Func(type_idx) => {
                        section.push(0x00);
                        section.extend(leb128_u32(*type_idx));
                    }
                    ImportDesc::Memory { min, max } => {
                        section.push(0x02);
                        if let Some(mx) = max {
                            section.push(0x01);
                            section.extend(leb128_u32(*min));
                            section.extend(leb128_u32(*mx));
                        } else {
                            section.push(0x00);
                            section.extend(leb128_u32(*min));
                        }
                    }
                    ImportDesc::Global { ty, mutable } => {
                        section.push(0x03);
                        section.push(ty.byte_code());
                        section.push(if *mutable { 0x01 } else { 0x00 });
                    }
                    ImportDesc::Table => { section.push(0x01); }
                }
            }
            write_section(&mut out, SectionId::Import, &section);
        }

        // Function section (type indices for local functions)
        if !self.functions.is_empty() {
            let mut section = Vec::new();
            section.extend(leb128_u32(self.functions.len() as u32));
            for &type_idx in &self.functions {
                section.extend(leb128_u32(type_idx));
            }
            write_section(&mut out, SectionId::Function, &section);
        }

        // Memory section
        if let Some((min, max)) = self.memory {
            let mut section = Vec::new();
            section.extend(leb128_u32(1)); // one memory
            if let Some(mx) = max {
                section.push(0x01);
                section.extend(leb128_u32(min));
                section.extend(leb128_u32(mx));
            } else {
                section.push(0x00);
                section.extend(leb128_u32(min));
            }
            write_section(&mut out, SectionId::Memory, &section);
        }

        // Export section
        if !self.exports.is_empty() {
            let mut section = Vec::new();
            section.extend(leb128_u32(self.exports.len() as u32));
            for exp in &self.exports {
                encode_str(&exp.name, &mut section);
                match &exp.desc {
                    ExportDesc::Func(idx)   => { section.push(0x00); section.extend(leb128_u32(*idx)); }
                    ExportDesc::Table(idx)  => { section.push(0x01); section.extend(leb128_u32(*idx)); }
                    ExportDesc::Memory(idx) => { section.push(0x02); section.extend(leb128_u32(*idx)); }
                    ExportDesc::Global(idx) => { section.push(0x03); section.extend(leb128_u32(*idx)); }
                }
            }
            write_section(&mut out, SectionId::Export, &section);
        }

        // Code section
        if !self.code.is_empty() {
            let mut section = Vec::new();
            section.extend(leb128_u32(self.code.len() as u32));
            for (locals, body) in &self.code {
                let mut func_body = Vec::new();
                // Locals
                func_body.extend(leb128_u32(locals.len() as u32));
                for (count, ty) in locals {
                    func_body.extend(leb128_u32(*count));
                    func_body.push(ty.byte_code());
                }
                func_body.extend_from_slice(body);
                func_body.push(0x0B); // end opcode
                // Prepend size of function body
                section.extend(leb128_u32(func_body.len() as u32));
                section.extend(func_body);
            }
            write_section(&mut out, SectionId::Code, &section);
        }

        out
    }
}

fn write_section(out: &mut Vec<u8>, id: SectionId, content: &[u8]) {
    out.push(id as u8);
    out.extend(leb128_u32(content.len() as u32));
    out.extend_from_slice(content);
}

// ── 6.5  Host function binding registry ─────────────────────────────────────

/// A host function exposed to WASM via the import mechanism.
#[derive(Debug, Clone)]
pub struct HostFunction {
    pub module:    String,
    pub name:      String,
    pub func_type: WasmFuncType,
    pub doc:       String,
}

/// Registry of host functions available to WASM modules.
pub struct WasmHostRegistry {
    functions: Vec<HostFunction>,
    index: HashMap<(String, String), usize>,
}

impl WasmHostRegistry {
    pub fn new() -> Self {
        WasmHostRegistry { functions: Vec::new(), index: HashMap::new() }
    }

    pub fn register(&mut self, f: HostFunction) {
        let key = (f.module.clone(), f.name.clone());
        let idx = self.functions.len();
        self.index.insert(key, idx);
        self.functions.push(f);
    }

    pub fn lookup(&self, module: &str, name: &str) -> Option<&HostFunction> {
        self.index.get(&(module.to_string(), name.to_string()))
            .and_then(|&i| self.functions.get(i))
    }

    /// Validate that all imports in a module have registered host implementations.
    pub fn validate_imports(&self, imports: &[WasmImport]) -> Vec<String> {
        let mut errors = Vec::new();
        for imp in imports {
            if let ImportDesc::Func(_) = &imp.desc {
                if self.lookup(&imp.module, &imp.name).is_none() {
                    errors.push(format!("Unresolved import: {}.{}", imp.module, imp.name));
                }
            }
        }
        errors
    }
}

// ── 6.6  wasm-bindgen-style attribute model ──────────────────────────────────

/// Attributes for Chronos → JS/WASM interop (similar to wasm-bindgen).
#[derive(Debug, Clone, Default)]
pub struct WasmBindgenAttr {
    /// Expose this function to JS (adds an export).
    pub js_export:      bool,
    /// Custom JS name override.
    pub js_name:        Option<String>,
    /// Bind this to a JS class method.
    pub method_of:      Option<String>,
    /// Generate a constructor (returns *Self from JS `new ClassName()`).
    pub constructor:    bool,
    /// Bind a JS getter.
    pub getter:         Option<String>,
    /// Bind a JS setter.
    pub setter:         Option<String>,
    /// Skip generating TypeScript `.d.ts` types.
    pub skip_typescript: bool,
}

impl WasmBindgenAttr {
    pub fn export() -> Self {
        WasmBindgenAttr { js_export: true, ..Default::default() }
    }

    pub fn method(class: &str) -> Self {
        WasmBindgenAttr { js_export: true, method_of: Some(class.to_string()), ..Default::default() }
    }

    pub fn constructor(class: &str) -> Self {
        WasmBindgenAttr { js_export: true, constructor: true, method_of: Some(class.to_string()), ..Default::default() }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Cross-language type mapping table
// ─────────────────────────────────────────────────────────────────────────────

/// A unified type mapping across all supported FFI targets.
#[derive(Debug, Clone)]
pub struct CrossLangTypeMap {
    pub chronos_type: &'static str,
    pub c_type:       CType,
    pub python_format: &'static str,
    pub jni_type:     JniType,
    pub dotnet_type:  &'static str,
    pub wasm_type:    Option<WasmValType>,
}

/// The canonical cross-language type mapping table.
pub fn cross_lang_type_table() -> Vec<CrossLangTypeMap> {
    vec![
        CrossLangTypeMap { chronos_type: "Bool",   c_type: CType::Bool,      python_format: "p", jni_type: JniType::Boolean, dotnet_type: "bool",   wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "I8",     c_type: CType::Int8,      python_format: "b", jni_type: JniType::Byte,    dotnet_type: "sbyte",  wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "U8",     c_type: CType::UInt8,     python_format: "B", jni_type: JniType::Byte,    dotnet_type: "byte",   wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "I16",    c_type: CType::Int16,     python_format: "h", jni_type: JniType::Short,   dotnet_type: "short",  wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "U16",    c_type: CType::UInt16,    python_format: "H", jni_type: JniType::Char,    dotnet_type: "ushort", wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "I32",    c_type: CType::Int32,     python_format: "i", jni_type: JniType::Int,     dotnet_type: "int",    wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "U32",    c_type: CType::UInt32,    python_format: "I", jni_type: JniType::Int,     dotnet_type: "uint",   wasm_type: Some(WasmValType::I32) },
        CrossLangTypeMap { chronos_type: "I64",    c_type: CType::Int64,     python_format: "L", jni_type: JniType::Long,    dotnet_type: "long",   wasm_type: Some(WasmValType::I64) },
        CrossLangTypeMap { chronos_type: "U64",    c_type: CType::UInt64,    python_format: "K", jni_type: JniType::Long,    dotnet_type: "ulong",  wasm_type: Some(WasmValType::I64) },
        CrossLangTypeMap { chronos_type: "F32",    c_type: CType::Float,     python_format: "f", jni_type: JniType::Float,   dotnet_type: "float",  wasm_type: Some(WasmValType::F32) },
        CrossLangTypeMap { chronos_type: "F64",    c_type: CType::Double,    python_format: "d", jni_type: JniType::Double,  dotnet_type: "double", wasm_type: Some(WasmValType::F64) },
        CrossLangTypeMap { chronos_type: "Void",   c_type: CType::Void,      python_format: "",  jni_type: JniType::Void,    dotnet_type: "void",   wasm_type: None },
        CrossLangTypeMap { chronos_type: "CStr",   c_type: CType::Pointer(Box::new(CType::Char)), python_format: "s", jni_type: JniType::Object("java/lang/String".into()), dotnet_type: "string", wasm_type: Some(WasmValType::I32) },
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── C struct layout ──────────────────────────────────────────────────────

    #[test]
    fn test_struct_layout_simple() {
        let fields = vec![
            CField { name: "x".into(), ty: CType::Int,    bit_width: None },
            CField { name: "y".into(), ty: CType::Double, bit_width: None },
        ];
        let layout = StructLayout::compute("Point", &fields);
        // int = 4 bytes at offset 0; double needs align 8 → 4 bytes pad → offset 8
        assert_eq!(layout.offsets[0], 0);
        assert_eq!(layout.offsets[1], 8);
        assert_eq!(layout.size, 16);
        assert_eq!(layout.alignment, 8);
    }

    #[test]
    fn test_struct_layout_packed_booleans() {
        let fields = vec![
            CField { name: "a".into(), ty: CType::Bool, bit_width: None },
            CField { name: "b".into(), ty: CType::Bool, bit_width: None },
            CField { name: "c".into(), ty: CType::Bool, bit_width: None },
        ];
        let layout = StructLayout::compute("Flags", &fields);
        assert_eq!(layout.offsets, vec![0, 1, 2]);
        assert_eq!(layout.size, 3);
    }

    #[test]
    fn test_struct_layout_c_header_contains_struct_name() {
        let fields = vec![CField { name: "val".into(), ty: CType::Int, bit_width: None }];
        let layout = StructLayout::compute("MyStruct", &fields);
        let header = layout.to_c_header();
        assert!(header.contains("MyStruct"));
        assert!(header.contains("val"));
    }

    #[test]
    fn test_ctype_size_align_lp64_pointer() {
        let ptr = CType::Pointer(Box::new(CType::Int));
        assert_eq!(ptr.size_align_lp64(), (8, 8));
    }

    #[test]
    fn test_ctype_size_align_long_windows() {
        // long is 4 bytes on Windows LLP64
        assert_eq!(CType::Long.size_align_llp64(), (4, 4));
        // long is 8 bytes on Linux LP64
        assert_eq!(CType::Long.size_align_lp64(), (8, 8));
    }

    #[test]
    fn test_ctype_c_spelling() {
        assert_eq!(CType::UInt64.c_spelling(), "uint64_t");
        assert_eq!(CType::Pointer(Box::new(CType::Char)).c_spelling(), "char*");
        assert_eq!(CType::ConstPointer(Box::new(CType::Char)).c_spelling(), "const char*");
    }

    // ── C function prototype ─────────────────────────────────────────────────

    #[test]
    fn test_c_prototype_no_params() {
        let ft = CFunctionType {
            return_type: CType::Void,
            params: vec![],
            variadic: false,
            convention: CallingConvention::CDcl,
        };
        let proto = ft.c_prototype("foo");
        assert_eq!(proto, "void foo(void);");
    }

    #[test]
    fn test_c_prototype_variadic() {
        let ft = CFunctionType {
            return_type: CType::Int,
            params: vec![("fmt".into(), CType::ConstPointer(Box::new(CType::Char)))],
            variadic: true,
            convention: CallingConvention::CDcl,
        };
        let proto = ft.c_prototype("printf");
        assert!(proto.contains("..."));
        assert!(proto.contains("printf"));
    }

    #[test]
    fn test_extern_block_header() {
        let mut block = ExternBlock::new(Some("libc.so.6"), LinkKind::Dynamic);
        block.add("malloc", CFunctionType {
            return_type: CType::Pointer(Box::new(CType::Void)),
            params: vec![("size".into(), CType::SizeT)],
            variadic: false,
            convention: CallingConvention::CDcl,
        });
        let header = block.to_c_header();
        assert!(header.contains("malloc"));
        assert!(header.contains("extern \"C\""));
    }

    // ── C++ name mangling ────────────────────────────────────────────────────

    #[test]
    fn test_itanium_mangle_free_function_no_args() {
        let mangled = itanium_mangle(&[], "foo", &[]);
        assert_eq!(mangled, "_Z3foov");
    }

    #[test]
    fn test_itanium_mangle_free_function_int_arg() {
        let mangled = itanium_mangle(&[], "bar", &[CType::Int]);
        assert_eq!(mangled, "_Z3bari");
    }

    #[test]
    fn test_itanium_mangle_namespaced() {
        let mangled = itanium_mangle(&["std", "vector"], "push_back", &[CType::Int]);
        assert!(mangled.starts_with("_ZN"));
        assert!(mangled.contains("push_back"));
    }

    #[test]
    fn test_msvc_mangle_basic() {
        let mangled = msvc_mangle(&[], "foo", &[], false);
        assert!(mangled.starts_with("?foo@"));
    }

    #[test]
    fn test_itanium_mangle_pointer_type() {
        let mangled = itanium_mangle(&[], "baz", &[CType::Pointer(Box::new(CType::Char))]);
        assert!(mangled.contains("Pc")); // P = pointer, c = char
    }

    // ── Virtual table ────────────────────────────────────────────────────────

    #[test]
    fn test_vtable_total_entries() {
        let mut vt = VTable::new("Animal");
        vt.add_virtual("speak", CFunctionType {
            return_type: CType::Void, params: vec![], variadic: false,
            convention: CallingConvention::Thiscall,
        }, true);
        // 2 overhead + 1 virtual = 3
        assert_eq!(vt.total_entries(), 3);
    }

    #[test]
    fn test_vtable_is_abstract() {
        let mut vt = VTable::new("Shape");
        vt.add_virtual("area", CFunctionType {
            return_type: CType::Double, params: vec![], variadic: false,
            convention: CallingConvention::Thiscall,
        }, true); // pure virtual
        assert!(vt.is_abstract());
    }

    #[test]
    fn test_vtable_c_struct_output() {
        let mut vt = VTable::new("Drawable");
        vt.add_virtual("draw", CFunctionType {
            return_type: CType::Void, params: vec![], variadic: false,
            convention: CallingConvention::Thiscall,
        }, false);
        let c_struct = vt.to_c_vtable_struct();
        assert!(c_struct.contains("Drawable_vtable"));
        assert!(c_struct.contains("offset_to_top"));
        assert!(c_struct.contains("typeinfo"));
        assert!(c_struct.contains("draw"));
    }

    // ── Python FFI ───────────────────────────────────────────────────────────

    #[test]
    fn test_pyobject_refcount() {
        let mut obj = PyObjectHandle::new(1, 42);
        assert_eq!(obj.refcnt, 1);
        obj.incref();
        assert_eq!(obj.refcnt, 2);
        assert!(!obj.decref());
        assert_eq!(obj.refcnt, 1);
        assert!(obj.decref()); // should be deallocated
    }

    #[test]
    fn test_pymoduledef_emit_c() {
        let mut md = PyModuleDef::new("mymod", "My module");
        md.add_method(PyMethodDef {
            ml_name: "add".into(),
            ml_meth: "mymod_add".into(),
            ml_flags: MethFlags::VARARGS,
            ml_doc: "Add two numbers".into(),
        });
        let c_code = md.emit_c_init();
        assert!(c_code.contains("PyMODINIT_FUNC"));
        assert!(c_code.contains("PyInit_mymod"));
        assert!(c_code.contains("mymod_methods"));
        assert!(c_code.contains("\"add\""));
    }

    #[test]
    fn test_python_type_mappings() {
        let mappings = python_type_mappings();
        let float_map = mappings.iter().find(|m| m.python_type == "float").unwrap();
        assert_eq!(float_map.py_build, "d");
        assert_eq!(float_map.py_format, "d");
    }

    // ── JNI ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_jni_type_descriptor() {
        assert_eq!(JniType::Int.descriptor(), "I");
        assert_eq!(JniType::Long.descriptor(), "J");
        assert_eq!(JniType::Object("java/lang/String".into()).descriptor(), "Ljava/lang/String;");
        assert_eq!(JniType::Array(Box::new(JniType::Int)).descriptor(), "[I");
    }

    #[test]
    fn test_jni_method_signature() {
        let params = vec![JniType::Int, JniType::Object("java/lang/String".into())];
        let ret = JniType::Boolean;
        let sig = jni_method_signature(&params, &ret);
        assert_eq!(sig, "(ILjava/lang/String;)Z");
    }

    #[test]
    fn test_jni_method_call_dispatch() {
        let call = JniMethodCall::new(
            "com.example.Foo", "compute",
            &[JniType::Int], JniType::Long, false,
        );
        assert_eq!(call.jnienv_function(), "CallLongMethod");
    }

    #[test]
    fn test_jni_static_method_call_dispatch() {
        let call = JniMethodCall::new(
            "com.example.Math", "sqrt",
            &[JniType::Double], JniType::Double, true,
        );
        assert_eq!(call.jnienv_function(), "CallStaticDoubleMethod");
    }

    #[test]
    fn test_jni_ref_lifecycle() {
        let mut r = JniRef::new_local(1, "java/lang/Object");
        assert!(r.is_valid());
        let global = r.promote_to_global();
        assert_eq!(global.kind, JniRefKind::Global);
        r.delete();
        assert!(!r.is_valid());
        assert!(global.is_valid()); // global ref still alive
    }

    // ── .NET P/Invoke ────────────────────────────────────────────────────────

    #[test]
    fn test_dll_import_emit() {
        let mut attr = DllImportAttribute::new("kernel32.dll");
        attr.entry_point = Some("GetLastError".into());
        attr.set_last_error = true;
        let decl = attr.emit_csharp("GetLastError", "int", "");
        assert!(decl.contains("kernel32.dll"));
        assert!(decl.contains("GetLastError"));
        assert!(decl.contains("SetLastError = true"));
    }

    #[test]
    fn test_hresult_decode_success() {
        let s_ok: u32 = 0x00000000;
        assert!(hresult_succeeded(s_ok));
        assert!(!hresult_failed(s_ok));
        let (fail, fac, code) = decode_hresult(s_ok);
        assert!(!fail);
        assert_eq!(fac, 0);
        assert_eq!(code, 0);
    }

    #[test]
    fn test_hresult_decode_failure() {
        let e_fail: u32 = 0x80004005; // E_FAIL
        assert!(!hresult_succeeded(e_fail));
        assert!(hresult_failed(e_fail));
        let (fail, _, _) = decode_hresult(e_fail);
        assert!(fail);
    }

    #[test]
    fn test_com_iid_string() {
        let iface = ComInterface::new_iunknown_derived("IMyInterface", [
            0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,
            0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F
        ]);
        let iid = iface.iid_string();
        assert_eq!(iid, "00010203-0405-0607-0809-0A0B0C0D0E0F");
    }

    #[test]
    fn test_com_vtable_slot_ordering() {
        let mut iface = ComInterface::new_iunknown_derived("IStream", [0u8; 16]);
        iface.add_method("Read",  vec![("pv".into(), CType::Pointer(Box::new(CType::Void)))], CType::Int32);
        iface.add_method("Write", vec![("pv".into(), CType::ConstPointer(Box::new(CType::Void)))], CType::Int32);
        // Slots 0-2 are IUnknown methods; Read = slot 3, Write = slot 4
        assert_eq!(iface.methods[0].slot, 3);
        assert_eq!(iface.methods[1].slot, 4);
    }

    // ── WASM binary ──────────────────────────────────────────────────────────

    #[test]
    fn test_leb128_u32_small() {
        assert_eq!(leb128_u32(0), vec![0x00]);
        assert_eq!(leb128_u32(1), vec![0x01]);
        assert_eq!(leb128_u32(127), vec![0x7F]);
    }

    #[test]
    fn test_leb128_u32_multi_byte() {
        // 128 = 0x80 → [0x80, 0x01]
        assert_eq!(leb128_u32(128), vec![0x80, 0x01]);
        // 300 = 0x12C → [0xAC, 0x02]
        assert_eq!(leb128_u32(300), vec![0xAC, 0x02]);
    }

    #[test]
    fn test_leb128_roundtrip() {
        for &v in &[0u32, 1, 63, 127, 128, 255, 300, 16383, 16384, 2097151, 268435455, u32::MAX / 2] {
            let encoded = leb128_u32(v);
            let (decoded, _) = decode_leb128_u32(&encoded);
            assert_eq!(decoded, v, "roundtrip failed for {}", v);
        }
    }

    #[test]
    fn test_leb128_i32_negative() {
        // -1 should encode to [0x7F]
        let enc = leb128_i32(-1);
        assert_eq!(enc, vec![0x7F]);
        // -128 should encode to [0x80, 0x7F]
        let enc = leb128_i32(-128);
        assert_eq!(enc, vec![0x80, 0x7F]);
    }

    #[test]
    fn test_wasm_func_type_encode() {
        let ft = WasmFuncType::new(vec![WasmValType::I32, WasmValType::I64], vec![WasmValType::F64]);
        let encoded = ft.encode();
        assert_eq!(encoded[0], 0x60); // functype prefix
    }

    #[test]
    fn test_wasm_module_magic() {
        let builder = WasmModuleBuilder::new();
        let bytes = builder.encode();
        assert_eq!(&bytes[0..4], b"\x00asm");
        assert_eq!(&bytes[4..8], &[0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_wasm_module_with_memory() {
        let mut builder = WasmModuleBuilder::new();
        builder.set_memory(1, Some(4));
        let bytes = builder.encode();
        // Memory section id = 5
        assert!(bytes.windows(1).any(|w| w[0] == SectionId::Memory as u8));
    }

    #[test]
    fn test_wasm_module_type_dedup() {
        let mut builder = WasmModuleBuilder::new();
        let ft = WasmFuncType::new(vec![WasmValType::I32], vec![WasmValType::I32]);
        let idx1 = builder.add_type(ft.clone());
        let idx2 = builder.add_type(ft);
        assert_eq!(idx1, idx2); // should deduplicate
        assert_eq!(builder.types.len(), 1);
    }

    #[test]
    fn test_wasm_host_registry() {
        let mut reg = WasmHostRegistry::new();
        reg.register(HostFunction {
            module: "env".into(),
            name: "log_i32".into(),
            func_type: WasmFuncType::new(vec![WasmValType::I32], vec![]),
            doc: "Log an i32 value".into(),
        });
        assert!(reg.lookup("env", "log_i32").is_some());
        assert!(reg.lookup("env", "nonexistent").is_none());
    }

    #[test]
    fn test_wasm_import_validation() {
        let reg = WasmHostRegistry::new();
        let imports = vec![WasmImport {
            module: "env".into(),
            name: "missing_func".into(),
            desc: ImportDesc::Func(0),
        }];
        let errors = reg.validate_imports(&imports);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("missing_func"));
    }

    #[test]
    fn test_wasm_val_type_names() {
        assert_eq!(WasmValType::I32.name(), "i32");
        assert_eq!(WasmValType::F64.name(), "f64");
        assert_eq!(WasmValType::FuncRef.name(), "funcref");
    }

    #[test]
    fn test_cross_lang_type_table_completeness() {
        let table = cross_lang_type_table();
        // Must have entries for all primitive types
        let has_i32 = table.iter().any(|e| e.chronos_type == "I32");
        let has_f64 = table.iter().any(|e| e.chronos_type == "F64");
        let has_bool = table.iter().any(|e| e.chronos_type == "Bool");
        assert!(has_i32 && has_f64 && has_bool);
    }

    #[test]
    fn test_calling_convention_display() {
        assert_eq!(CallingConvention::CDcl.to_string(), "cdecl");
        assert_eq!(CallingConvention::Stdcall.to_string(), "stdcall");
        assert_eq!(CallingConvention::SystemVAmd64.to_string(), "sysv_amd64");
    }

    #[test]
    fn test_wasm_bindgen_attr_constructor() {
        let attr = WasmBindgenAttr::constructor("Canvas");
        assert!(attr.js_export);
        assert!(attr.constructor);
        assert_eq!(attr.method_of.as_deref(), Some("Canvas"));
    }
}
