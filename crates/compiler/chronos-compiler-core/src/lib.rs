//! Chronos Compiler Core
//!
//! This crate unifies all compiler pipeline files into a single compilation
//! unit, solving cross-file type dependency issues.
//!
//! All pipeline files are included into the same module scope.
//! Each file's duplicate `use` statements are suppressed via the
//! `#[allow(unused_imports)]` attribute or by removing them from the source.

// ============================================================================
// GLOBAL LINT SUPPRESSION
// These allow attributes suppress harmless issues that arise from combining
// multiple independently-written source files into one module.
// ============================================================================
#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    unused_mut,
    non_snake_case,
    unused_assignments,
    unused_macros,
    irrefutable_let_patterns,
    unreachable_patterns,
    non_camel_case_types,
    clippy::all,
)]

use std::time::Duration;

// Step 1: Lexer — defines Token, SpannedToken, Lexer, LexerError
// (also imports std::fmt and std::ops::Range into crate scope)
include!("../../../../src/compiler/chronos-lexer.rs");

// Step 2: Stdlib types — defines SymbolicExpr and other catalog types
// Note: chronos-stdlib-types.rs has its own `use` statements for
// std::collections::* and std::fmt — these will be E0252 conflicts.
// Handled by removing/commenting them in the source.
include!("../../../../src/compiler/chronos-stdlib-types.rs");

// Step 3: Missing AST types (ChronosType, Expression, Program, etc.)
include!("missing_types.rs");

// Step 4: Parser — uses Token, ChronosType, Expression, Statement, etc.
include!("../../../../src/compiler/chronos-parser.rs");

// Step 5: Parser V2 — extends the parser with domain-specific AST nodes
include!("../../../../src/compiler/chronos-parser-v2.rs");

// Step 6: Type inference — uses ChronosType, Effect, Program, etc.
include!("../../../../src/compiler/chronos-type-inference.rs");

// Step 7: Inference IR V2 — extends type inference with domain opcodes
include!("../../../../src/compiler/chronos-inference-ir-v2.rs");

// Step 8: IR codegen — uses IRModule, IRFunction, ChronosType, etc.
include!("../../../../src/compiler/chronos-ir-codegen.rs");

// Step 9: Security + real-time analysis
include!("../../../../src/compiler/chronos-security-realtime.rs");

// Step 10: Unified integration — orchestrates all phases
include!("../../../../src/compiler/chronos-unified-integration.rs");

// Step 11: C code generation backend (AST → C99)
include!("../../../../src/compiler/chronos-c-backend.rs");

// Step 12: Late-binding stubs — defines orchestrator types that depend
// on types from ALL previous includes
include!("missing_types_late.rs");
