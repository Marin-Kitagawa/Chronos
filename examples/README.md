# Chronos Example Programs

These examples are written to match the token set and grammar rules implemented in
`chronos-lexer.rs` and `chronos-parser.rs`. Every syntax choice here traces back to
a concrete path in those two files — nothing is invented.

---

## How to build and run

Once the Cargo workspace and a `chronos` binary exist:

```
chronos build hello.chr
chronos run   hello.chr
```

During development you can also drive the compiler directly through `cargo`:

```
cargo run -- build examples/hello.chr
cargo run -- run   examples/hello.chr
```

---

## Language syntax reference

The following is derived solely from reading `chronos-lexer.rs` (Token enum) and
`chronos-parser.rs` (all `parse_*` methods).

### Functions

```
fn name(param1: Type1, param2: Type2) -> ReturnType {
    // body
}
```

- Both `fn` and `fun` are accepted (`Token::KwFn` / `Token::KwFun`).
- Parameters always use `name: Type` — the colon is mandatory (`parse_param_list`).
- The return type is introduced by `->` (`Token::Arrow`). Omitting `->` defaults to `void`.
- Prefix `pub`, `private`, `protected`, `internal`, or `crate` sets visibility.
- Prefix `async` makes a function asynchronous.
- An expression-body shorthand is also accepted: `fn square(x: i32) -> i32 = x * x;`

### Variable bindings

```
let name: Type = expr;   // immutable (let / val)
var name: Type = expr;   // mutable
```

- `let` and `val` produce immutable bindings (`KwLet` / `KwVal`).
- `var` produces a mutable binding (`KwVar`).
- The type annotation (`: Type`) is optional; the parser infers it when absent.
- The `= expr` initialiser is **required** by `parse_let_statement`.
- Trailing semicolons are consumed but not required in all positions.

### Assignment

```
target = value;
```

Any expression on the left-hand side followed by `=` is treated as an assignment
statement. Compound assignment operators (`+=`, `-=`, `*=`, …) are also lexed.

### Control flow

```
if condition {
    // then
} else if other_condition {
    // else-if chain
} else {
    // else
}
```

The condition is a plain expression — **no surrounding parentheses are required**
(though they are allowed because `(expr)` is itself a valid expression).

```
while condition {
    // body
}
```

```
for binding in iterator {
    // body
}
```

```
return expr;
return;        // returns unit/void
break;
continue;
```

### Match / when

```
match value {
    pattern       => expr,
    pattern if guard => expr,
    _             => expr,
}
```

`when` is an accepted alias for `match` (`KwWhen`).

Patterns can be:
- A wildcard: `_`
- A binding: `name`
- A constructor: `Some(inner)`
- A literal: `42`, `"hello"`, `true`, `false`
- A tuple: `(a, b)`

### Structs

```
struct Name {
    field1: Type1,
    field2: Type2,
}
```

Fields inside `{}` are parsed by `parse_field_list_in_braces`: each field is
`name: Type` terminated by a comma or semicolon.

### Classes

```
class Name : SuperClass, Interface1 {
    let field: Type;
    var mutable_field: Type;

    fn method(self: Name) -> ReturnType {
        // body
    }
}
```

`abstract class` and `final class` are also accepted.

### Data classes (Kotlin style)

```
data class Point(x: f64, y: f64)
```

The primary-constructor fields become the struct fields and equality / hash /
copy implementations are auto-derived.

### Enums

```
enum Direction {
    North,
    South,
    East,
    West,
}

enum Option<T> {
    Some(T),
    None,
}
```

Variants may be unit, tuple `(Types…)`, or struct `{ fields }`.

### impl blocks

```
impl TypeName {
    fn method(self: TypeName, ...) -> ReturnType {
        // body
    }
}

impl TraitName for TypeName {
    fn required_method(self: TypeName) -> ReturnType {
        // body
    }
}
```

### Traits and interfaces

```
trait Printable : Display + Debug {
    fn print(self: Self);
    fn debug_str(self: Self) -> string {
        // optional default implementation
        return "default";
    }
}
```

`interface` is accepted as a synonym for `trait`.

### Type aliases

```
type Meters = f64;
type Grid<T> = Vec<Vec<T>>;
```

### Generics

```
fn identity<T>(x: T) -> T {
    return x;
}

struct Pair<A, B> {
    first: A,
    second: B,
}
```

Bounds use `T: Trait` inside `<…>` and can be extended with `where`:

```
fn show<T: Display>(value: T) where T: Clone {
    // ...
}
```

### Type system

Primitive numeric types: `i8` `i16` `i32` `i64` `i128` `u8` `u16` `u32` `u64`
`u128` `f16` `f32` `f64` `f128` `bf16` `bool` `char` `str` `string` `int` `uint`
`usize` `isize`.

Compound: `[T]` (slice), `[T; N]` (fixed array), `(A, B)` (tuple),
`fn(A, B) -> R` (function type), `T?` (optional / nullable).

Memory qualifiers: `linear T` (must use exactly once), `affine T` (at most once),
`&T` / `&mut T` (borrowed reference).

### Lambdas / closures

```
let double = |x: i32| x * 2;
let add    = |a: i32, b: i32| -> i32 { return a + b; };
```

### Modules and imports

```
mod geometry {
    pub struct Point { x: f64, y: f64 }
}

import std::collections;
use std::io;
use std::fmt::{ Display, Debug };
import geometry as geo;
```

Both `import` and `use` are accepted. Path separators can be `::` or `.`.

### Visibility modifiers

`pub`, `private`, `protected`, `internal`, `crate`. Default (no keyword) is
`private` as implemented in `parse_visibility`.

### Annotations

```
@deprecated
@device(gpu)
fn heavy_compute(data: tensor<f32, [1024]>) -> tensor<f32, [1024]> { ... }
```

Annotations start with `@` (`Token::Annotation`) and may carry a parenthesised
argument list of expressions.

### Special features

**Degradable functions** (Feature 6 — functions with an expiry date):

```
@expires("2027-01-01")
@warns("2026-06-01")
@replaces("new_compute")
degradable fn old_compute(x: f32) -> f32 { ... }
```

**GPU kernels** (Feature 5 — Mojo-inspired):

```
kernel gpu matmul(a: tensor<f32, [M, K]>, b: tensor<f32, [K, N]>) -> tensor<f32, [M, N]> {
    // ...
}
```

**AI skills / tools / pipelines** (Feature 4):

```
ai skill Summarize {
    description "Summarize input text"
    schema { text: str, max_words: int }
    instruction "Return a concise summary."
    constraint "Never exceed max_words."
}

ai tool WebSearch fn search(query: string) -> string {
    // implementation
}

ai pipeline ResearchBot {
    fetch -> WebSearch,
    summarize -> Summarize,
}
```

**Pipe-forward operator**:

```
let result = data |> transform |> summarize;
// equivalent to: summarize(transform(data))
```

**AI invocation operator** `@@`:

```
let summary = @@Summarize { text: article, max_words: 100 };
```

---

## Files in this directory

| File | Description |
|------|-------------|
| `hello.chr` | Minimal hello world: `fn main`, `let` binding, function call |
| `fibonacci.chr` | Recursive and iterative Fibonacci: `fn`, `if/else`, `while`, `var`, `return` |
| `structs.chr` | Point and Rect: `struct`, `impl`, field access, method calls, `type` alias |
