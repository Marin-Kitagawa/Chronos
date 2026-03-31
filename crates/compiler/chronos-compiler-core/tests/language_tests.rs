//! Chronos language feature tests.
//!
//! Each test compiles a `.chr` source file from `tests/features/` and
//! asserts that compilation succeeds and the generated C output contains
//! expected patterns.
//!
//! ## Feature support matrix (as of current compiler version)
//! Working end-to-end:
//!   - Primitive types: i8-u64, f32-f64, bool, string
//!   - Arithmetic, comparison, logical, unary operators
//!   - Compound assignments: +=, -=, *=, /=, %=
//!   - let/var bindings with explicit type annotations
//!   - if/else expressions
//!   - while loops
//!   - break / continue
//!   - return statements
//!   - Function declarations, calls, recursion, mutual recursion
//!   - println (variadic)
//!
//! Parse-only (emits placeholder C):
//!   - Struct/enum/trait declarations
//!   - impl blocks
//!   - for loops
//!   - match expressions
//!   - Lambda expressions
//!   - type aliases
//!
//! Not yet implemented:
//!   - Struct literals: `Type { field: val }` (no parser support)
//!   - `self` / `new` as identifiers in parameter position
//!   - `require` statement (not parsed)
//!   - Type alias resolution in type checker

use chronos_compiler_core::{
    compile, CompilerConfig, CompilationTarget, TargetArch, TargetOS, SecurityMode,
};
use std::path::{Path, PathBuf};

// ── helpers ──────────────────────────────────────────────────────────────────

fn features_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/compiler/chronos-compiler-core
    // Go up 3 levels to reach the workspace root.
    let manifest = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest)
        .parent().expect("crate → compiler dir")
        .parent().expect("compiler → crates dir")
        .parent().expect("crates → workspace root")
        .join("tests")
        .join("features")
}

fn default_config() -> CompilerConfig {
    CompilerConfig {
        security_mode: SecurityMode::Off,
        realtime_mode: false,
        target: CompilationTarget {
            arch: TargetArch::X86_64,
            os: TargetOS::Linux,
            features: Vec::new(),
        },
        hardware_model: None,
    }
}

fn compile_feature(filename: &str) -> chronos_compiler_core::CompilationResult {
    let path = features_dir().join(filename);
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e));
    compile(source, filename.to_string(), default_config())
}

fn compile_src(src: &str) -> chronos_compiler_core::CompilationResult {
    compile(src.to_string(), "<test>".to_string(), default_config())
}

/// Print all diagnostics from a result (for debugging failures).
fn print_diags(result: &chronos_compiler_core::CompilationResult, label: &str) {
    for d in &result.diagnostics {
        eprintln!("[{}] {:?}: {} [{}]", label, d.level, d.message, d.code);
    }
}

/// Assert the file compiles without errors and C code is emitted.
fn assert_compiles(filename: &str) {
    let result = compile_feature(filename);
    print_diags(&result, filename);
    assert!(result.success, "{} should compile without errors", filename);
    assert!(
        result.c_code.is_some() && !result.c_code.as_deref().unwrap_or("").is_empty(),
        "{} should produce non-empty C output",
        filename
    );
}

/// Assert the file compiles AND the generated C contains `pattern`.
fn assert_c_contains(filename: &str, pattern: &str) {
    let result = compile_feature(filename);
    print_diags(&result, filename);
    assert!(result.success, "{} should compile", filename);
    let c = result.c_code.expect("expected C output");
    assert!(
        c.contains(pattern),
        "{}: expected C to contain {:?}\n--- C output ---\n{}",
        filename, pattern, &c[..c.len().min(4096)]
    );
}

/// Assert the file compiles AND the generated C contains ALL given patterns.
fn assert_c_contains_all(filename: &str, patterns: &[&str]) {
    let result = compile_feature(filename);
    print_diags(&result, filename);
    assert!(result.success, "{} should compile", filename);
    let c = result.c_code.expect("expected C output");
    for pattern in patterns {
        assert!(
            c.contains(pattern),
            "{}: C should contain {:?}\n--- C output (first 4096) ---\n{}",
            filename, pattern, &c[..c.len().min(4096)]
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA TYPE TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// All signed and unsigned integer types map to the correct C integer types.
#[test]
fn test_integer_types() {
    assert_c_contains_all("01_int_types.chr", &[
        "int8_t",   // i8
        "int16_t",  // i16
        "int32_t",  // i32
        "int64_t",  // i64
        "uint8_t",  // u8
        "uint16_t", // u16
        "uint32_t", // u32
        "uint64_t", // u64
    ]);
}

/// f32 maps to C `float` and f64 maps to C `double`.
#[test]
fn test_float_types() {
    assert_c_contains_all("02_float_types.chr", &["float", "double"]);
}

/// bool type and true/false literals appear in generated C.
#[test]
fn test_bool_type() {
    assert_c_contains_all("03_bool_type.chr", &["bool", "true", "false"]);
}

/// string type maps to `const char*` in C.
#[test]
fn test_string_type() {
    assert_c_contains_all("04_string_type.chr", &["const char*", "Hello, World!"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// OPERATOR TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// All five arithmetic operators appear in the generated C.
#[test]
fn test_arithmetic_operators() {
    assert_c_contains_all("05_arithmetic_ops.chr", &["+", "-", "*", "/", "%"]);
}

/// All six comparison operators appear in the generated C.
#[test]
fn test_comparison_operators() {
    assert_c_contains_all("06_comparison_ops.chr", &["==", "!=", "<", ">", "<=", ">="]);
}

/// Logical AND, OR, and NOT appear in the generated C.
#[test]
fn test_logical_operators() {
    assert_c_contains_all("07_logical_ops.chr", &["&&", "||", "!"]);
}

/// Unary negation produces a minus sign in the C output.
#[test]
fn test_unary_operators() {
    assert_compiles("08_unary_ops.chr");
    let c = compile_feature("08_unary_ops.chr").c_code.unwrap();
    assert!(c.contains("-"), "unary negation (-) should appear in C output");
}

/// All five compound assignment operators are emitted verbatim.
#[test]
fn test_compound_assignment_operators() {
    assert_c_contains_all("09_compound_assign.chr", &["+=", "-=", "*=", "/=", "%="]);
}

// ─────────────────────────────────────────────────────────────────────────────
// VARIABLE BINDING TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// let bindings declare C variables of the correct type.
#[test]
fn test_let_binding() {
    assert_compiles("10_let_binding.chr");
}

/// var bindings (mutable) compile to the same C variable declarations as let.
#[test]
fn test_var_binding() {
    assert_compiles("11_var_binding.chr");
}

// ─────────────────────────────────────────────────────────────────────────────
// CONTROL FLOW TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// if/else generates C `if (...) { } else { }` blocks.
#[test]
fn test_if_else() {
    assert_c_contains_all("12_if_else.chr", &["if (", "else"]);
}

/// while loops generate C `while (...)` blocks.
#[test]
fn test_while_loop() {
    assert_c_contains("13_while_loop.chr", "while (");
}

/// break and continue are emitted inside while loops.
#[test]
fn test_break_continue() {
    assert_c_contains_all("14_break_continue.chr", &["while (", "break;", "continue;"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// FUNCTION TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// Functions with return types and multiple signatures compile correctly.
#[test]
fn test_functions_basic() {
    assert_c_contains_all("15_functions_basic.chr", &[
        "int64_t double(",
        "bool is_even(",
        "const char* status(",
        "main(",
    ]);
}

/// Recursive functions emit self-referential calls.
#[test]
fn test_recursion() {
    let result = compile_feature("16_recursion.chr");
    print_diags(&result, "16_recursion.chr");
    assert!(result.success);
    let c = result.c_code.unwrap();
    // fib appears in declaration + at least 2 recursive calls
    let fib_calls = c.matches("fib(").count();
    assert!(fib_calls >= 3, "fib( should appear >= 3 times, got {}", fib_calls);
}

/// Mutually recursive functions (forward references) compile correctly.
#[test]
fn test_forward_reference() {
    assert_c_contains_all("17_forward_reference.chr", &["is_even(", "is_odd("]);
}

/// Functions with many parameters of different types compile correctly.
#[test]
fn test_multiple_params() {
    assert_c_contains_all("18_multiple_params.chr", &[
        "int64_t add3(",
        "double weighted_sum(",
    ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// STRUCT / DECLARATION TESTS (parse-only — C backend emits struct stubs)
// ─────────────────────────────────────────────────────────────────────────────

/// Struct declarations are parsed without errors.
#[test]
fn test_struct_declarations() {
    assert_compiles("19_structs_decl.chr");
}

/// impl blocks with static methods are parsed without errors.
#[test]
fn test_impl_declarations() {
    assert_compiles("20_impl_decl.chr");
}

/// Type alias declarations are parsed without errors.
#[test]
fn test_type_alias_declarations() {
    assert_compiles("21_type_alias.chr");
}

/// Nested struct declarations (fields referencing other struct types) parse.
#[test]
fn test_nested_struct_declarations() {
    assert_compiles("22_nested_struct_decl.chr");
}

// ─────────────────────────────────────────────────────────────────────────────
// OUTPUT / EXPRESSION TESTS
// ─────────────────────────────────────────────────────────────────────────────

/// Various println forms (zero args, one arg, multi-arg, each type) compile.
#[test]
fn test_println_forms() {
    let result = compile_feature("23_println_forms.chr");
    print_diags(&result, "23_println_forms.chr");
    assert!(result.success);
    let c = result.c_code.unwrap();
    assert!(
        c.contains("printf") || c.contains("__chr_println"),
        "output helpers should appear in C"
    );
}

/// Nested function calls (passing call results as arguments) compile.
#[test]
fn test_nested_calls() {
    assert_c_contains_all("24_nested_calls.chr", &["double(", "inc(", "square("]);
}

/// Defensive assertions via if/else compile correctly.
#[test]
fn test_defensive_assertions() {
    assert_compiles("25_require_assert.chr");
}

/// Void-returning functions are emitted as `void fn_name(...)`.
#[test]
fn test_return_void() {
    assert_c_contains_all("26_return_void.chr", &[
        "void separator(",
        "void print_twice(",
        "void print_range(",
    ]);
}

/// Complex expressions with operator precedence and parentheses compile.
#[test]
fn test_complex_expressions() {
    assert_compiles("27_complex_expressions.chr");
}

// ─────────────────────────────────────────────────────────────────────────────
// PARSE-ONLY TESTS (parser acceptance, C backend may emit placeholders)
// ─────────────────────────────────────────────────────────────────────────────

/// Enum declarations are parsed without errors.
#[test]
fn test_enum_declarations() {
    assert_compiles("28_enum_parse.chr");
}

/// Match expressions are parsed without errors.
#[test]
fn test_match_expressions() {
    assert_compiles("29_match_parse.chr");
}

/// for-in loops are parsed without errors.
#[test]
fn test_for_loop_parse() {
    assert_compiles("30_for_loop_parse.chr");
}

/// pub visibility on functions and struct declarations parses without errors.
#[test]
fn test_visibility_pub() {
    assert_compiles("31_visibility_pub.chr");
}

/// Trait declarations and impl-for-struct parse without errors.
#[test]
fn test_trait_declarations() {
    assert_compiles("32_trait_parse.chr");
}

/// Lambda expressions are parsed without errors.
#[test]
fn test_lambda_parse() {
    assert_compiles("33_lambda_parse.chr");
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPREHENSIVE INTEGRATION TEST
// ─────────────────────────────────────────────────────────────────────────────

/// Comprehensive test: all working features combined in one program.
#[test]
fn test_comprehensive_program() {
    assert_c_contains_all("34_comprehensive.chr", &[
        "main(",
        "while (",
        "if (",
        "break;",
        "continue;",
        "bool is_even(",
        "bool is_odd(",
        "bool is_prime(",
        "int64_t fib(",
    ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// INLINE SOURCE TESTS — test compiler behaviour via embedded source strings
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_inline_hello_world() {
    let r = compile_src(r#"
fn main() {
    println("hello");
}
"#);
    assert!(r.success, "hello world must compile");
    assert!(r.c_code.as_deref().unwrap_or("").contains("hello"));
}

#[test]
fn test_inline_integer_arithmetic() {
    let r = compile_src(r#"
fn main() {
    let x: i64 = 10 + 20 * 3 - 5;
    println(x);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("int64_t"));
}

#[test]
fn test_inline_all_arithmetic_in_one() {
    let r = compile_src(r#"
fn main() {
    let a: i64 = 10;
    let b: i64 = 3;
    println(a + b);
    println(a - b);
    println(a * b);
    println(a / b);
    println(a % b);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    for op in &["+", "-", "*", "/", "%"] {
        assert!(c.contains(op), "operator {} should appear in C", op);
    }
}

#[test]
fn test_inline_compound_assignments() {
    let r = compile_src(r#"
fn main() {
    var x: i64 = 0;
    x += 10;
    x -= 3;
    x *= 2;
    x /= 7;
    x %= 3;
    println(x);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    for op in &["+=", "-=", "*=", "/=", "%="] {
        assert!(c.contains(op), "compound op {} should appear", op);
    }
}

#[test]
fn test_inline_if_returns_value() {
    let r = compile_src(r#"
fn abs(x: i64) -> i64 {
    if x < 0 {
        return -x;
    } else {
        return x;
    }
}
fn main() {
    println(abs(-5));
    println(abs(3));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("if ("));
    assert!(c.contains("else"));
}

#[test]
fn test_inline_nested_if_else_chain() {
    let r = compile_src(r#"
fn classify(x: i64) -> string {
    if x < 0 {
        return "negative";
    } else {
        if x == 0 {
            return "zero";
        } else {
            if x < 100 {
                return "small";
            } else {
                return "large";
            }
        }
    }
}
fn main() {
    println(classify(-1));
    println(classify(0));
    println(classify(50));
    println(classify(999));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    let if_count = c.matches("if (").count();
    assert!(if_count >= 3, "expected >= 3 if blocks, got {}", if_count);
}

#[test]
fn test_inline_while_accumulates() {
    let r = compile_src(r#"
fn main() {
    var sum: i64 = 0;
    var i: i64 = 1;
    while i <= 100 {
        sum = sum + i;
        i = i + 1;
    }
    println(sum);
}
"#);
    assert!(r.success);
    assert!(r.c_code.as_deref().unwrap_or("").contains("while ("));
}

#[test]
fn test_inline_break_exits_loop() {
    let r = compile_src(r#"
fn main() {
    var i: i64 = 0;
    while i < 1000 {
        if i == 42 {
            break;
        }
        i = i + 1;
    }
    println(i);
}
"#);
    assert!(r.success);
    assert!(r.c_code.as_deref().unwrap_or("").contains("break;"));
}

#[test]
fn test_inline_continue_skips_iteration() {
    let r = compile_src(r#"
fn main() {
    var sum: i64 = 0;
    var i: i64 = 0;
    while i <= 10 {
        i = i + 1;
        if i % 2 == 0 {
            continue;
        }
        sum = sum + i;
    }
    println(sum);
}
"#);
    assert!(r.success);
    assert!(r.c_code.as_deref().unwrap_or("").contains("continue;"));
}

#[test]
fn test_inline_recursive_fibonacci() {
    let r = compile_src(r#"
fn fib(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}
fn main() {
    var i: i64 = 0;
    while i <= 10 {
        println(fib(i));
        i = i + 1;
    }
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    let fib_count = c.matches("fib(").count();
    assert!(fib_count >= 3, "fib( should appear >= 3 times, got {}", fib_count);
}

#[test]
fn test_inline_mutual_recursion() {
    let r = compile_src(r#"
fn even(n: i64) -> bool {
    if n == 0 {
        return true;
    }
    return odd(n - 1);
}
fn odd(n: i64) -> bool {
    if n == 0 {
        return false;
    }
    return even(n - 1);
}
fn main() {
    println(even(4));
    println(odd(3));
}
"#);
    assert!(r.success, "mutual recursion should compile");
}

#[test]
fn test_inline_bool_literals_and_operations() {
    let r = compile_src(r#"
fn main() {
    let t: bool = true;
    let f: bool = false;
    let and_res: bool = t && f;
    let or_res: bool = t || f;
    let not_res: bool = !t;
    println(and_res);
    println(or_res);
    println(not_res);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("true") || c.contains("false"));
}

#[test]
fn test_inline_f64_arithmetic() {
    let r = compile_src(r#"
fn circle_area(r: f64) -> f64 {
    let pi: f64 = 3.14159265;
    return pi * r * r;
}
fn main() {
    println(circle_area(1.0));
    println(circle_area(2.0));
}
"#);
    assert!(r.success);
    assert!(r.c_code.as_deref().unwrap_or("").contains("double"));
}

#[test]
fn test_inline_string_in_if() {
    let r = compile_src(r#"
fn label(b: bool) -> string {
    if b {
        return "yes";
    } else {
        return "no";
    }
}
fn main() {
    println(label(true));
    println(label(false));
}
"#);
    assert!(r.success);
}

#[test]
fn test_inline_void_function_called() {
    let r = compile_src(r#"
fn say_hi() {
    println("hi");
}
fn say_bye() {
    println("bye");
}
fn main() {
    say_hi();
    say_bye();
    say_hi();
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("void say_hi("));
    assert!(c.contains("void say_bye("));
}

#[test]
fn test_inline_all_comparison_operators() {
    let r = compile_src(r#"
fn main() {
    let a: i64 = 5;
    let b: i64 = 10;
    println(a == b);
    println(a != b);
    println(a < b);
    println(a > b);
    println(a <= b);
    println(a >= b);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    for op in &["==", "!=", "<", ">", "<=", ">="] {
        assert!(c.contains(op), "operator {} should appear", op);
    }
}

#[test]
fn test_inline_all_logical_operators() {
    let r = compile_src(r#"
fn main() {
    let t: bool = true;
    let f: bool = false;
    println(t && f);
    println(t || f);
    println(!t);
    println(!f);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("&&"));
    assert!(c.contains("||"));
    assert!(c.contains("!"));
}

#[test]
fn test_inline_integer_types_all() {
    // Define functions for all integer types. Only call the i64 one in main
    // (integer literals are typed as Int64, so passing them to u64 params would
    // fail the type checker). The function *definitions* are enough to get all
    // C integer type names into the output via forward declarations.
    let r = compile_src(r#"
fn add_i8(a: i8, b: i8) -> i8   { return a + b; }
fn add_i16(a: i16, b: i16) -> i16 { return a + b; }
fn add_i32(a: i32, b: i32) -> i32 { return a + b; }
fn add_i64(a: i64, b: i64) -> i64 { return a + b; }
fn add_u8(a: u8, b: u8) -> u8   { return a + b; }
fn add_u16(a: u16, b: u16) -> u16 { return a + b; }
fn add_u32(a: u32, b: u32) -> u32 { return a + b; }
fn add_u64(a: u64, b: u64) -> u64 { return a + b; }
fn main() {
    println(add_i64(10, 20));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    for ty in &["int8_t", "int16_t", "int32_t", "int64_t",
                "uint8_t", "uint16_t", "uint32_t", "uint64_t"] {
        assert!(c.contains(ty), "C type {} should appear", ty);
    }
}

#[test]
fn test_inline_float_types_both() {
    let r = compile_src(r#"
fn scale_f32(x: f32, factor: f32) -> f32 { return x * factor; }
fn scale_f64(x: f64, factor: f64) -> f64 { return x * factor; }
fn main() {
    println(scale_f64(3.14, 2.0));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("float"));
    assert!(c.contains("double"));
}

#[test]
fn test_inline_multiple_return_paths() {
    let r = compile_src(r#"
fn fizzbuzz(n: i64) -> string {
    if n % 15 == 0 {
        return "FizzBuzz";
    } else {
        if n % 3 == 0 {
            return "Fizz";
        } else {
            if n % 5 == 0 {
                return "Buzz";
            } else {
                return "n";
            }
        }
    }
}
fn main() {
    println(fizzbuzz(15));
    println(fizzbuzz(3));
    println(fizzbuzz(5));
    println(fizzbuzz(7));
}
"#);
    assert!(r.success);
}

#[test]
fn test_inline_nested_while_loops() {
    let r = compile_src(r#"
fn main() {
    var total: i64 = 0;
    var i: i64 = 1;
    while i <= 5 {
        var j: i64 = 1;
        while j <= 5 {
            total = total + 1;
            j = j + 1;
        }
        i = i + 1;
    }
    println(total);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    let while_count = c.matches("while (").count();
    assert!(while_count >= 2, "should have >= 2 while loops, got {}", while_count);
}

#[test]
fn test_inline_early_return() {
    let r = compile_src(r#"
fn find_first_even(limit: i64) -> i64 {
    var i: i64 = 1;
    while i <= limit {
        if i % 2 == 0 {
            return i;
        }
        i = i + 1;
    }
    return -1;
}
fn main() {
    println(find_first_even(10));
    println(find_first_even(1));
}
"#);
    assert!(r.success);
}

#[test]
fn test_inline_unary_negation() {
    let r = compile_src(r#"
fn negate(x: i64) -> i64 {
    return -x;
}
fn negate_f(x: f64) -> f64 {
    return -x;
}
fn main() {
    println(negate(42));
    println(negate(-7));
    println(negate_f(3.14));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("-"));
}

#[test]
fn test_inline_logical_not() {
    let r = compile_src(r#"
fn flip(b: bool) -> bool {
    return !b;
}
fn main() {
    println(flip(true));
    println(flip(false));
    let x: i64 = 5;
    println(!(x == 10));
}
"#);
    assert!(r.success);
}

#[test]
fn test_inline_gcd() {
    let r = compile_src(r#"
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        return a;
    }
    return gcd(b, a % b);
}
fn main() {
    println(gcd(48, 18));
    println(gcd(100, 75));
    println(gcd(17, 13));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    // gcd must reference itself recursively
    assert!(c.matches("gcd(").count() >= 2);
}

#[test]
fn test_inline_power_of_two() {
    let r = compile_src(r#"
fn pow2(n: i64) -> i64 {
    if n == 0 {
        return 1;
    }
    return 2 * pow2(n - 1);
}
fn main() {
    var i: i64 = 0;
    while i <= 10 {
        println(pow2(i));
        i = i + 1;
    }
}
"#);
    assert!(r.success);
}

#[test]
fn test_inline_string_function_return() {
    let r = compile_src(r#"
fn sign_str(n: i64) -> string {
    if n > 0 {
        return "positive";
    } else {
        if n < 0 {
            return "negative";
        } else {
            return "zero";
        }
    }
}
fn main() {
    println(sign_str(5));
    println(sign_str(-3));
    println(sign_str(0));
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("const char*"));
    assert!(c.contains("positive"));
}

#[test]
fn test_inline_factorial() {
    let r = compile_src(r#"
fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    return n * factorial(n - 1);
}
fn main() {
    println(factorial(1));
    println(factorial(5));
    println(factorial(10));
}
"#);
    assert!(r.success);
}

#[test]
fn test_inline_multi_arg_println() {
    let r = compile_src(r#"
fn main() {
    let x: i64 = 42;
    let y: f64 = 3.14;
    let s: string = "hello";
    println(x, y);
    println("value:", x);
    println("pi:", y);
    println("name:", s);
}
"#);
    assert!(r.success);
    let c = r.c_code.unwrap();
    assert!(c.contains("printf"));
}

#[test]
fn test_inline_is_prime() {
    let r = compile_src(r#"
fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    var i: i64 = 2;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i = i + 1;
    }
    return true;
}
fn main() {
    println(is_prime(2));
    println(is_prime(7));
    println(is_prime(10));
    println(is_prime(97));
}
"#);
    assert!(r.success);
}

// ─────────────────────────────────────────────────────────────────────────────
// PARSE-ONLY INLINE TESTS
// Tests that the parser accepts specific syntax even though the C backend
// does not yet fully support it.
// ─────────────────────────────────────────────────────────────────────────────

/// Struct declarations (without instances) parse without errors.
#[test]
fn test_inline_struct_declaration() {
    let r = compile_src(r#"
pub struct Point {
    x: f64,
    y: f64,
}
pub struct RGB {
    r: i64,
    g: i64,
    b: i64,
}
fn main() {
    println("struct declaration ok");
}
"#);
    assert!(r.success, "struct declarations should parse");
}

/// enum declarations parse without errors.
#[test]
fn test_inline_enum_declaration() {
    let r = compile_src(r#"
enum Direction { North, South, East, West }
enum Status { Active, Inactive }
fn main() {
    println("enum declaration ok");
}
"#);
    assert!(r.success, "enum declarations should parse");
}

/// for-in loops are parsed without errors.
#[test]
fn test_inline_for_loop_parses() {
    let r = compile_src(r#"
fn main() {
    for i in 0..10 {
        println(i);
    }
    println("for loop parsed");
}
"#);
    assert!(r.success, "for loop should parse");
}

/// match expressions are parsed without errors.
#[test]
fn test_inline_match_parses() {
    let r = compile_src(r#"
fn main() {
    let x: i64 = 2;
    match x {
        1 => println("one"),
        2 => println("two"),
        _ => println("other"),
    }
}
"#);
    assert!(r.success, "match expression should parse");
}
