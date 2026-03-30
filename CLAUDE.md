# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Chronos is a next-generation systems programming language compiler and runtime implemented in Rust (~30,000 lines across 22 standalone `.rs` files). It unifies features from C++, Scala, Rust, Kotlin, TypeScript, and AI frameworks.

## Build

There is **no Cargo.toml or build system** yet. The source files are standalone Rust modules designed to become crates in a Cargo workspace. The only external dependency is `logos = "0.14"` (used by the lexer).

To compile any individual file (once a Cargo workspace exists):
```
cargo build
cargo test
```

Currently, files reference each other's types by convention but are not wired together with `mod`/`use` across crate boundaries.

## Architecture

### Compiler Pipeline

```
Source → Lexer → Parser → Type Inference → Security Audit → Degradation Check → IR Lowering → WCET Analysis → Code Generation
```

Orchestrated by `CompilationContext` in `chronos-unified-integration.rs`, which threads shared state through all phases.

### Core Compiler (4 files)

| File | Role |
|------|------|
| `chronos-lexer.rs` | DFA tokenizer via `logos` crate. 100+ token types including version annotations and AI syntax |
| `chronos-parser.rs` | Recursive descent for declarations, Pratt parsing for expressions. Produces the AST |
| `chronos-type-inference.rs` | Hindley-Milner with subtyping, linear/affine type checking, effect inference, taint tracking |
| `chronos-ir-codegen.rs` | SSA-based IR definition and backend code generation (LLVM/NVPTX/XLA targets) |

### V2 Extensions (3 files)

`chronos-lexer-v2.rs`, `chronos-parser-v2.rs`, `chronos-inference-ir-v2.rs` — extend the core compiler with tokens, AST nodes, and type rules for 25 domain-specific features. These are additive layers on top of the core.

### Integration Layer (2 files)

| File | Role |
|------|------|
| `chronos-unified-integration.rs` | Master orchestrator. Bridges all phases via `CompilationContext`. Contains all cross-system integration |
| `chronos-stdlib-types.rs` | Universal `ChronosType` enum cataloging types from all supported languages |

### Cross-Cutting Concerns

| File | Role |
|------|------|
| `chronos-security-realtime.rs` | 9-pass security audit (`#![secure]` mode) + WCET analysis (`#![realtime]` mode) |

### Domain Engines (9 files)

Each is a self-contained implementation with its own types, algorithms, and integration points:

- `chronos-graph-engine.rs` — 30+ graph algorithms (BFS, Dijkstra, A*, MST, SCC, max flow, etc.)
- `chronos-cas-engine.rs` — Computer Algebra System (symbolic differentiation, simplification, equation solving)
- `chronos-network-engine.rs` — Full networking stack (IPv4, TCP with state machine, DNS, HTTP, checksums)
- `chronos-os-engine.rs` — OS primitives (page frame allocator, buddy/slab allocators, CFS scheduler, inode filesystem, IPC)
- `chronos-distributed-engine.rs` — Distributed systems (Lamport/vector/hybrid clocks, Raft consensus, CRDTs, consistent hashing, gossip)
- `chronos-blockchain-engine.rs` — SHA-256, Keccak-256, complete EVM interpreter with gas metering, smart contracts
- `chronos-proof-engine.rs` — Formal verification via Curry-Howard, Calculus of Constructions, tactics engine
- `chronos-quantum-engine.rs` — Quantum simulator (2^N amplitudes, gates, Grover/QFT algorithms, entanglement)
- `chronos-simulation-datascience.rs` — Statistics, linear algebra (QR/SVD), ODE solvers, FEM, particle systems

### Key Design Patterns

- **The `CompilationContext` struct** is the central data flow mechanism — every phase reads from and writes to it
- **V2 files extend V1** — the lexer-v2/parser-v2/inference-ir-v2 files add domain tokens and AST nodes without modifying the originals
- **Each engine is standalone** — domain engines define their own types and algorithms; integration happens in `chronos-unified-integration.rs`
- **Security and real-time are compiler modes** — `#![secure]` triggers 9 audit passes that can block compilation; `#![realtime]` enables WCET analysis

### Type System

The `ChronosType` enum in `chronos-stdlib-types.rs` is the universal type catalog. It includes:
- Primitives (I8–I512, F16–F128)
- Memory-managed types (Linear, Affine — enables no-GC guarantees)
- Domain types (Tensor, Graph, Qubit, Quantity with units, Address, U256)
- All standard containers (Vec, HashMap, BTreeMap, etc.)

## Notes

- `chronos-graph-engine(1).rs` appears to be a duplicate of `chronos-graph-engine.rs`
- `chat.md` (97KB) contains design discussion/history
- `chronos-missing-features-1.rs` is documentation of extended domain features, not executable code
