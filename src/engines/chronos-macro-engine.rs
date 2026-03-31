// ============================================================================
// CHRONOS MACRO & LANGUAGE TOOLING ENGINE
// ============================================================================
//
// HOW MACROS ACTUALLY WORK (and why they matter for a universal language):
//
// Macros are programs that run at compile time and generate code. They're
// the mechanism by which a language extends itself — without macros, every
// new syntactic construct must be baked into the compiler. With macros,
// users can add domain-specific syntax that looks and feels native.
//
// There are three fundamentally different approaches to macros:
//
// 1. DECLARATIVE MACROS (pattern → template): The user writes patterns
//    that match syntax fragments and templates that produce replacement
//    code. Think of Rust's `macro_rules!` or Scheme's `syntax-rules`.
//    These are safe (they can't produce malformed code if the patterns
//    are correct) but limited in power.
//
// 2. PROCEDURAL MACROS: The user writes arbitrary code that receives a
//    token stream and returns a modified token stream. Think of Rust's
//    proc macros, Scala's macro annotations, or Lisp's defmacro. These
//    are maximally powerful but harder to reason about — a proc macro
//    can do anything, including reading files, calling APIs, or running
//    the compiler recursively.
//
// 3. COMPILE-TIME EXECUTION (comptime): Instead of transforming syntax,
//    execute arbitrary code at compile time and splice the result into
//    the program. Think of Zig's `comptime`, C++'s constexpr/consteval,
//    or D's CTFE. This is the cleanest model — no token manipulation,
//    just running code early.
//
// Chronos supports all three, plus EMBEDDED DSLs — domain-specific
// languages (SQL, regex, JSON, GraphQL, shell, assembly, shaders, LaTeX)
// that are validated at compile time and lowered to efficient code.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  Token stream representation (the currency of macro expansion)
//   2.  Declarative macro definition and pattern matching
//   3.  Declarative macro expansion with hygiene
//   4.  Procedural macro infrastructure (attribute, derive, function-like)
//   5.  Compile-time expression evaluator (comptime)
//   6.  Embedded DSL framework with compile-time validation:
//       - SQL (syntax validation, table/column reference checking)
//       - Regex (compile-time pattern compilation and validation)
//       - JSON (compile-time parsing and type generation)
//       - Shell (command validation and injection detection)
//   7.  Macro hygiene (preventing accidental name capture)
//   8.  Quasi-quotation (constructing token trees programmatically)
//   9.  Derive macro system (auto-generating trait implementations)
//  10.  Attribute macro system (transforming annotated items)
// ============================================================================

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// PART 1: TOKEN STREAM
// ============================================================================
// The token stream is the universal interface for macros. Every macro —
// declarative, procedural, or DSL — operates on token streams. A token
// stream is a flat sequence of tokens that can be parsed, matched against
// patterns, or constructed programmatically.
//
// The key design decision is whether tokens carry span information (their
// position in the source code) and hygiene context (which macro expansion
// they came from). Chronos tokens carry both, because:
//   - Spans enable accurate error messages that point to the macro call
//     site, not the macro definition
//   - Hygiene prevents macros from accidentally capturing variables from
//     the call site or leaking internal variables to the caller

/// A span in the source code: byte offset range + file ID.
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub file_id: u32,
    /// Hygiene context: which macro expansion created this token.
    /// Tokens from different expansions with the same name don't collide.
    pub hygiene_ctx: u32,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Span { start, end, file_id: 0, hygiene_ctx: 0 }
    }

    pub fn synthetic() -> Self {
        Span { start: 0, end: 0, file_id: u32::MAX, hygiene_ctx: 0 }
    }
}

/// A single token in the stream.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    CharLiteral(char),

    // Identifiers and keywords
    Ident(String),
    Keyword(String),
    Lifetime(String),   // 'a

    // Punctuation and operators
    Plus, Minus, Star, Slash, Percent,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Not,
    Assign,         // =
    PlusAssign,     // +=
    MinusAssign,    // -=
    Arrow,          // ->
    FatArrow,       // =>
    Dot, DotDot, DotDotDot,  // . .. ...
    Comma, Semicolon, Colon, ColonColon,
    Ampersand, Pipe, Caret, Tilde,
    Hash, At, Dollar,
    Question,       // ?

    // Delimiters (paired)
    OpenParen, CloseParen,      // ( )
    OpenBracket, CloseBracket,  // [ ]
    OpenBrace, CloseBrace,      // { }

    // Special
    Newline,
    Eof,

    // Macro-specific
    MacroVar(String),   // $name in macro patterns
    MacroRep,           // $(...) repetition marker
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::IntLiteral(i) => write!(f, "{}", i),
            TokenKind::FloatLiteral(v) => write!(f, "{}", v),
            TokenKind::StringLiteral(s) => write!(f, "\"{}\"", s),
            TokenKind::BoolLiteral(b) => write!(f, "{}", b),
            TokenKind::CharLiteral(c) => write!(f, "'{}'", c),
            TokenKind::Ident(s) | TokenKind::Keyword(s) => write!(f, "{}", s),
            TokenKind::Lifetime(s) => write!(f, "'{}", s),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Star => write!(f, "*"),
            TokenKind::Slash => write!(f, "/"),
            TokenKind::Percent => write!(f, "%"),
            TokenKind::Eq => write!(f, "=="),
            TokenKind::Ne => write!(f, "!="),
            TokenKind::Lt => write!(f, "<"),
            TokenKind::Le => write!(f, "<="),
            TokenKind::Gt => write!(f, ">"),
            TokenKind::Ge => write!(f, ">="),
            TokenKind::And => write!(f, "&&"),
            TokenKind::Or => write!(f, "||"),
            TokenKind::Not => write!(f, "!"),
            TokenKind::Assign => write!(f, "="),
            TokenKind::PlusAssign => write!(f, "+="),
            TokenKind::MinusAssign => write!(f, "-="),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::DotDot => write!(f, ".."),
            TokenKind::DotDotDot => write!(f, "..."),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::ColonColon => write!(f, "::"),
            TokenKind::Ampersand => write!(f, "&"),
            TokenKind::Pipe => write!(f, "|"),
            TokenKind::Caret => write!(f, "^"),
            TokenKind::Tilde => write!(f, "~"),
            TokenKind::Hash => write!(f, "#"),
            TokenKind::At => write!(f, "@"),
            TokenKind::Dollar => write!(f, "$"),
            TokenKind::Question => write!(f, "?"),
            TokenKind::OpenParen => write!(f, "("),
            TokenKind::CloseParen => write!(f, ")"),
            TokenKind::OpenBracket => write!(f, "["),
            TokenKind::CloseBracket => write!(f, "]"),
            TokenKind::OpenBrace => write!(f, "{{"),
            TokenKind::CloseBrace => write!(f, "}}"),
            TokenKind::Newline => write!(f, "\\n"),
            TokenKind::Eof => write!(f, "EOF"),
            TokenKind::MacroVar(s) => write!(f, "${}", s),
            TokenKind::MacroRep => write!(f, "$(...)"),
        }
    }
}

/// A token with span information.
#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

impl Token {
    pub fn new(kind: TokenKind) -> Self {
        Token { kind, span: Span::synthetic() }
    }

    pub fn with_span(kind: TokenKind, span: Span) -> Self {
        Token { kind, span }
    }
}

/// A stream of tokens — the universal macro interface.
#[derive(Debug, Clone)]
pub struct TokenStream {
    pub tokens: Vec<Token>,
}

impl TokenStream {
    pub fn new() -> Self {
        TokenStream { tokens: Vec::new() }
    }

    pub fn from_tokens(tokens: Vec<Token>) -> Self {
        TokenStream { tokens }
    }

    pub fn push(&mut self, token: Token) {
        self.tokens.push(token);
    }

    pub fn extend(&mut self, other: &TokenStream) {
        self.tokens.extend(other.tokens.iter().cloned());
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Convert the token stream back to source text.
    pub fn to_string(&self) -> String {
        let mut result = String::new();
        for (i, token) in self.tokens.iter().enumerate() {
            if i > 0 {
                // Add space between tokens except after openers and before closers
                let prev = &self.tokens[i - 1].kind;
                let curr = &token.kind;
                let no_space = matches!(prev,
                    TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace |
                    TokenKind::Dot | TokenKind::ColonColon | TokenKind::Hash | TokenKind::At |
                    TokenKind::Dollar)
                    || matches!(curr,
                    TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace |
                    TokenKind::OpenParen | TokenKind::OpenBracket |
                    TokenKind::Comma | TokenKind::Semicolon | TokenKind::Dot |
                    TokenKind::ColonColon);
                if !no_space {
                    result.push(' ');
                }
            }
            result.push_str(&format!("{}", token.kind));
        }
        result
    }
}

/// A token tree: either a single token or a delimited group.
/// This is the recursive structure that macros actually match against.
#[derive(Debug, Clone)]
pub enum TokenTree {
    Token(Token),
    Group {
        delimiter: Delimiter,
        stream: TokenStream,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Delimiter {
    Paren,    // ( )
    Bracket,  // [ ]
    Brace,    // { }
    None,     // implicit grouping (no delimiters in output)
}

// ============================================================================
// PART 2: DECLARATIVE MACROS
// ============================================================================
// Declarative macros work by pattern matching: you define one or more
// "arms", each with a pattern and a template. When the macro is invoked,
// the engine tries each pattern in order until one matches, then expands
// the corresponding template with the captured bindings.
//
// Pattern fragments capture different syntactic categories:
//   $name:expr   — matches an expression
//   $name:ident  — matches an identifier
//   $name:ty     — matches a type
//   $name:pat    — matches a pattern
//   $name:tt     — matches a single token tree
//   $name:block  — matches a { ... } block
//   $name:literal — matches a literal value
//
// Repetitions use $(...),* or $(...),+ syntax:
//   $($x:expr),*  — matches zero or more comma-separated expressions
//   $($x:expr),+  — matches one or more comma-separated expressions

/// Fragment specifier: what kind of syntax a macro variable can match.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentSpec {
    Expr,      // any expression
    Ident,     // an identifier
    Ty,        // a type
    Pat,       // a pattern
    Tt,        // a single token tree
    Block,     // a { ... } block
    Literal,   // a literal value
    Item,      // a top-level item (fn, struct, etc.)
    Stmt,      // a statement
    Path,      // a path (a::b::c)
}

/// A pattern element in a declarative macro arm.
#[derive(Debug, Clone)]
pub enum MacroPattern {
    /// Match a specific token literally
    Literal(TokenKind),
    /// Capture a fragment: $name:spec
    Fragment { name: String, spec: FragmentSpec },
    /// Repetition: $( pattern ),* or $( pattern ),+
    Repetition {
        patterns: Vec<MacroPattern>,
        separator: Option<TokenKind>,
        kind: RepKind,
    },
    /// A sequence of patterns that must match in order
    Sequence(Vec<MacroPattern>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RepKind {
    ZeroOrMore,  // *
    OneOrMore,   // +
    ZeroOrOne,   // ?
}

/// A template element for macro expansion.
#[derive(Debug, Clone)]
pub enum MacroTemplate {
    /// Emit a literal token
    Literal(TokenKind),
    /// Substitute a captured variable
    Variable(String),
    /// Repeat a template for each captured repetition
    Repetition {
        templates: Vec<MacroTemplate>,
        separator: Option<TokenKind>,
    },
    /// A sequence of template elements
    Sequence(Vec<MacroTemplate>),
}

/// One arm of a declarative macro: pattern → template.
#[derive(Debug, Clone)]
pub struct MacroArm {
    pub pattern: MacroPattern,
    pub template: MacroTemplate,
}

/// A declarative macro definition.
#[derive(Debug, Clone)]
pub struct DeclarativeMacro {
    pub name: String,
    pub arms: Vec<MacroArm>,
    /// Hygiene context for this macro's expansion
    pub hygiene_ctx: u32,
}

/// Bindings captured during pattern matching.
#[derive(Debug, Clone)]
pub struct MacroBindings {
    /// Single-value bindings: $name → token stream
    pub singles: HashMap<String, TokenStream>,
    /// Repetition bindings: $name → vec of token streams (one per iteration)
    pub repetitions: HashMap<String, Vec<TokenStream>>,
}

impl MacroBindings {
    pub fn new() -> Self {
        MacroBindings {
            singles: HashMap::new(),
            repetitions: HashMap::new(),
        }
    }
}

/// The declarative macro engine.
pub struct DeclarativeMacroEngine {
    macros: HashMap<String, DeclarativeMacro>,
    next_hygiene: u32,
}

impl DeclarativeMacroEngine {
    pub fn new() -> Self {
        DeclarativeMacroEngine {
            macros: HashMap::new(),
            next_hygiene: 1,
        }
    }

    /// Register a declarative macro.
    pub fn define(&mut self, mac: DeclarativeMacro) {
        self.macros.insert(mac.name.clone(), mac);
    }

    /// Expand a macro invocation. Returns the expanded token stream or an error.
    pub fn expand(&mut self, name: &str, input: &TokenStream) -> Result<TokenStream, String> {
        let mac = self.macros.get(name)
            .ok_or_else(|| format!("Undefined macro: {}", name))?
            .clone();

        let hygiene = self.next_hygiene;
        self.next_hygiene += 1;

        // Try each arm until one matches
        for arm in &mac.arms {
            if let Some(bindings) = self.match_pattern(&arm.pattern, input, 0).map(|(b, _)| b) {
                return self.expand_template(&arm.template, &bindings, hygiene);
            }
        }

        Err(format!("No matching arm for macro '{}' with input: {}",
                     name, input.to_string()))
    }

    /// Try to match a pattern against a token stream starting at position `pos`.
    /// Returns the bindings and the position after the match, or None.
    fn match_pattern(&self, pattern: &MacroPattern, input: &TokenStream, pos: usize)
        -> Option<(MacroBindings, usize)>
    {
        match pattern {
            MacroPattern::Literal(expected) => {
                if pos < input.len() && input.tokens[pos].kind == *expected {
                    Some((MacroBindings::new(), pos + 1))
                } else {
                    None
                }
            }

            MacroPattern::Fragment { name, spec } => {
                if pos >= input.len() {
                    return None;
                }
                // Try to match a fragment of the specified kind
                let (captured, end) = self.match_fragment(spec, input, pos)?;
                let mut bindings = MacroBindings::new();
                bindings.singles.insert(name.clone(), captured);
                Some((bindings, end))
            }

            MacroPattern::Repetition { patterns, separator, kind } => {
                let mut all_bindings = MacroBindings::new();
                let mut current_pos = pos;
                let mut iteration_count = 0;

                loop {
                    // Try to match the sequence of patterns
                    let mut iter_bindings = MacroBindings::new();
                    let mut iter_pos = current_pos;
                    let mut matched_all = true;

                    for pat in patterns {
                        if let Some((b, new_pos)) = self.match_pattern(pat, input, iter_pos) {
                            Self::merge_bindings(&mut iter_bindings, &b);
                            iter_pos = new_pos;
                        } else {
                            matched_all = false;
                            break;
                        }
                    }

                    if !matched_all {
                        break;
                    }

                    iteration_count += 1;
                    current_pos = iter_pos;

                    // Merge into repetition bindings
                    for (name, stream) in &iter_bindings.singles {
                        all_bindings.repetitions
                            .entry(name.clone())
                            .or_insert_with(Vec::new)
                            .push(stream.clone());
                    }

                    // Try to match separator
                    if let Some(sep) = separator {
                        if current_pos < input.len() && input.tokens[current_pos].kind == *sep {
                            current_pos += 1;
                        } else {
                            break;
                        }
                    }
                }

                match kind {
                    RepKind::OneOrMore if iteration_count == 0 => None,
                    _ => Some((all_bindings, current_pos)),
                }
            }

            MacroPattern::Sequence(patterns) => {
                let mut bindings = MacroBindings::new();
                let mut current_pos = pos;

                for pat in patterns {
                    let (b, new_pos) = self.match_pattern(pat, input, current_pos)?;
                    Self::merge_bindings(&mut bindings, &b);
                    current_pos = new_pos;
                }

                Some((bindings, current_pos))
            }
        }
    }

    /// Match a fragment specifier against the token stream.
    fn match_fragment(&self, spec: &FragmentSpec, input: &TokenStream, pos: usize)
        -> Option<(TokenStream, usize)>
    {
        if pos >= input.len() {
            return None;
        }

        match spec {
            FragmentSpec::Ident => {
                if let TokenKind::Ident(_) = &input.tokens[pos].kind {
                    let stream = TokenStream::from_tokens(vec![input.tokens[pos].clone()]);
                    Some((stream, pos + 1))
                } else {
                    None
                }
            }
            FragmentSpec::Literal => {
                match &input.tokens[pos].kind {
                    TokenKind::IntLiteral(_) | TokenKind::FloatLiteral(_) |
                    TokenKind::StringLiteral(_) | TokenKind::BoolLiteral(_) |
                    TokenKind::CharLiteral(_) => {
                        let stream = TokenStream::from_tokens(vec![input.tokens[pos].clone()]);
                        Some((stream, pos + 1))
                    }
                    _ => None,
                }
            }
            FragmentSpec::Expr => {
                // Simplified: match tokens until we hit a comma, semicolon,
                // closing delimiter, or EOF. Real parsers do recursive descent here.
                let mut end = pos;
                let mut depth = 0;
                while end < input.len() {
                    match &input.tokens[end].kind {
                        TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => {
                            depth += 1;
                        }
                        TokenKind::CloseParen | TokenKind::CloseBracket | TokenKind::CloseBrace => {
                            if depth == 0 { break; }
                            depth -= 1;
                        }
                        TokenKind::Comma | TokenKind::Semicolon if depth == 0 => break,
                        TokenKind::Eof => break,
                        _ => {}
                    }
                    end += 1;
                }
                if end > pos {
                    let stream = TokenStream::from_tokens(input.tokens[pos..end].to_vec());
                    Some((stream, end))
                } else {
                    None
                }
            }
            FragmentSpec::Ty | FragmentSpec::Path => {
                // Simplified: match identifier possibly followed by :: and more identifiers
                let mut end = pos;
                while end < input.len() {
                    match &input.tokens[end].kind {
                        TokenKind::Ident(_) => { end += 1; }
                        TokenKind::ColonColon if end > pos => { end += 1; }
                        TokenKind::Lt => {
                            // Generic parameters: skip until matching >
                            let mut depth = 1;
                            end += 1;
                            while end < input.len() && depth > 0 {
                                match &input.tokens[end].kind {
                                    TokenKind::Lt => depth += 1,
                                    TokenKind::Gt => depth -= 1,
                                    _ => {}
                                }
                                end += 1;
                            }
                        }
                        _ => break,
                    }
                }
                if end > pos {
                    let stream = TokenStream::from_tokens(input.tokens[pos..end].to_vec());
                    Some((stream, end))
                } else {
                    None
                }
            }
            FragmentSpec::Block => {
                if input.tokens[pos].kind == TokenKind::OpenBrace {
                    let mut depth = 1;
                    let mut end = pos + 1;
                    while end < input.len() && depth > 0 {
                        match &input.tokens[end].kind {
                            TokenKind::OpenBrace => depth += 1,
                            TokenKind::CloseBrace => depth -= 1,
                            _ => {}
                        }
                        end += 1;
                    }
                    let stream = TokenStream::from_tokens(input.tokens[pos..end].to_vec());
                    Some((stream, end))
                } else {
                    None
                }
            }
            FragmentSpec::Tt => {
                // A single token tree: one token or a matched delimiter pair
                match &input.tokens[pos].kind {
                    TokenKind::OpenParen | TokenKind::OpenBracket | TokenKind::OpenBrace => {
                        let close = match &input.tokens[pos].kind {
                            TokenKind::OpenParen => TokenKind::CloseParen,
                            TokenKind::OpenBracket => TokenKind::CloseBracket,
                            TokenKind::OpenBrace => TokenKind::CloseBrace,
                            _ => unreachable!(),
                        };
                        let mut depth = 1;
                        let mut end = pos + 1;
                        while end < input.len() && depth > 0 {
                            if input.tokens[end].kind == input.tokens[pos].kind { depth += 1; }
                            if input.tokens[end].kind == close { depth -= 1; }
                            end += 1;
                        }
                        let stream = TokenStream::from_tokens(input.tokens[pos..end].to_vec());
                        Some((stream, end))
                    }
                    _ => {
                        let stream = TokenStream::from_tokens(vec![input.tokens[pos].clone()]);
                        Some((stream, pos + 1))
                    }
                }
            }
            FragmentSpec::Pat | FragmentSpec::Stmt | FragmentSpec::Item => {
                // Simplified: treat like expr
                self.match_fragment(&FragmentSpec::Expr, input, pos)
            }
        }
    }

    fn merge_bindings(target: &mut MacroBindings, source: &MacroBindings) {
        for (k, v) in &source.singles {
            target.singles.insert(k.clone(), v.clone());
        }
        for (k, v) in &source.repetitions {
            target.repetitions.entry(k.clone())
                .or_insert_with(Vec::new)
                .extend(v.iter().cloned());
        }
    }

    /// Expand a template with captured bindings.
    fn expand_template(&self, template: &MacroTemplate, bindings: &MacroBindings,
                       hygiene: u32) -> Result<TokenStream, String>
    {
        match template {
            MacroTemplate::Literal(kind) => {
                let mut token = Token::new(kind.clone());
                // Apply hygiene context to generated identifiers
                if let TokenKind::Ident(name) = &kind {
                    token.span.hygiene_ctx = hygiene;
                }
                Ok(TokenStream::from_tokens(vec![token]))
            }

            MacroTemplate::Variable(name) => {
                bindings.singles.get(name)
                    .cloned()
                    .ok_or_else(|| format!("Unbound macro variable: ${}", name))
            }

            MacroTemplate::Repetition { templates, separator } => {
                // Find a repetition binding to determine iteration count
                let rep_count = bindings.repetitions.values()
                    .next()
                    .map(|v| v.len())
                    .unwrap_or(0);

                let mut result = TokenStream::new();
                for i in 0..rep_count {
                    if i > 0 {
                        if let Some(sep) = separator {
                            result.push(Token::new(sep.clone()));
                        }
                    }

                    // Create bindings for this iteration
                    let mut iter_bindings = MacroBindings::new();
                    for (name, values) in &bindings.repetitions {
                        if i < values.len() {
                            iter_bindings.singles.insert(name.clone(), values[i].clone());
                        }
                    }
                    // Also include non-repetition bindings
                    for (name, value) in &bindings.singles {
                        iter_bindings.singles.insert(name.clone(), value.clone());
                    }

                    for tmpl in templates {
                        let expanded = self.expand_template(tmpl, &iter_bindings, hygiene)?;
                        result.extend(&expanded);
                    }
                }
                Ok(result)
            }

            MacroTemplate::Sequence(templates) => {
                let mut result = TokenStream::new();
                for tmpl in templates {
                    let expanded = self.expand_template(tmpl, bindings, hygiene)?;
                    result.extend(&expanded);
                }
                Ok(result)
            }
        }
    }
}

// ============================================================================
// PART 3: PROCEDURAL MACROS
// ============================================================================
// Procedural macros are functions that take a token stream and return
// a token stream. They come in three flavors:
//
// 1. Function-like: invoked as `my_macro!(...)`, receives the contents
//    of the invocation as a token stream.
//
// 2. Attribute: invoked as `#[my_attr(...)] fn foo() { ... }`, receives
//    both the attribute arguments and the annotated item.
//
// 3. Derive: invoked as `#[derive(MyTrait)]` on a struct/enum, receives
//    the struct/enum definition and produces additional implementations.
//
// In Chronos, proc macros are represented as closures over token streams.
// In a real implementation, they'd be separate compilation units loaded
// as dynamic libraries (like Rust's proc macro crates).

/// The type of a procedural macro.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcMacroKind {
    FunctionLike,
    Attribute,
    Derive,
}

/// A procedural macro handler — takes input token stream(s), returns output.
/// For function-like: (input) → output
/// For attribute: (attr_args, item) → output
/// For derive: (item) → additional_impls
pub type ProcMacroFn = fn(&TokenStream, Option<&TokenStream>) -> Result<TokenStream, String>;

/// A registered procedural macro.
pub struct ProcMacro {
    pub name: String,
    pub kind: ProcMacroKind,
    pub handler: ProcMacroFn,
}

/// The procedural macro registry.
pub struct ProcMacroRegistry {
    macros: HashMap<String, ProcMacro>,
}

impl ProcMacroRegistry {
    pub fn new() -> Self {
        ProcMacroRegistry { macros: HashMap::new() }
    }

    pub fn register(&mut self, mac: ProcMacro) {
        self.macros.insert(mac.name.clone(), mac);
    }

    /// Invoke a function-like proc macro.
    pub fn invoke_function_like(&self, name: &str, input: &TokenStream)
        -> Result<TokenStream, String>
    {
        let mac = self.macros.get(name)
            .ok_or_else(|| format!("Undefined proc macro: {}", name))?;
        if mac.kind != ProcMacroKind::FunctionLike {
            return Err(format!("'{}' is not a function-like macro", name));
        }
        (mac.handler)(input, None)
    }

    /// Invoke an attribute proc macro.
    pub fn invoke_attribute(&self, name: &str, attr_args: &TokenStream,
                            item: &TokenStream) -> Result<TokenStream, String>
    {
        let mac = self.macros.get(name)
            .ok_or_else(|| format!("Undefined attribute macro: {}", name))?;
        if mac.kind != ProcMacroKind::Attribute {
            return Err(format!("'{}' is not an attribute macro", name));
        }
        (mac.handler)(attr_args, Some(item))
    }

    /// Invoke a derive proc macro.
    pub fn invoke_derive(&self, name: &str, item: &TokenStream)
        -> Result<TokenStream, String>
    {
        let mac = self.macros.get(name)
            .ok_or_else(|| format!("Undefined derive macro: {}", name))?;
        if mac.kind != ProcMacroKind::Derive {
            return Err(format!("'{}' is not a derive macro", name));
        }
        (mac.handler)(item, None)
    }
}

// ============================================================================
// PART 4: COMPILE-TIME EXECUTION (comptime)
// ============================================================================
// comptime blocks are evaluated during compilation. Any expression that
// can be fully evaluated at compile time (no I/O, no dynamic dispatch,
// no external state) is eligible. The result is spliced into the program
// as a literal value.
//
// This is similar to Zig's comptime, C++ constexpr, and D's CTFE.
// The evaluator is a tree-walking interpreter for a subset of the language.

/// A compile-time value.
#[derive(Debug, Clone, PartialEq)]
pub enum ComptimeValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Char(char),
    Array(Vec<ComptimeValue>),
    Tuple(Vec<ComptimeValue>),
    Struct { name: String, fields: Vec<(String, ComptimeValue)> },
    Null,
    // A type itself as a value (for type-level computation)
    Type(String),
}

impl fmt::Display for ComptimeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComptimeValue::Int(i) => write!(f, "{}", i),
            ComptimeValue::Float(v) => write!(f, "{}", v),
            ComptimeValue::Bool(b) => write!(f, "{}", b),
            ComptimeValue::String(s) => write!(f, "\"{}\"", s),
            ComptimeValue::Char(c) => write!(f, "'{}'", c),
            ComptimeValue::Array(a) => {
                write!(f, "[")?;
                for (i, v) in a.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            ComptimeValue::Tuple(t) => {
                write!(f, "(")?;
                for (i, v) in t.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            ComptimeValue::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            ComptimeValue::Null => write!(f, "null"),
            ComptimeValue::Type(t) => write!(f, "type({})", t),
        }
    }
}

/// A compile-time expression.
#[derive(Debug, Clone)]
pub enum ComptimeExpr {
    Literal(ComptimeValue),
    Var(String),
    BinOp { left: Box<ComptimeExpr>, op: ComptimeOp, right: Box<ComptimeExpr> },
    UnaryOp { op: ComptimeUnaryOp, expr: Box<ComptimeExpr> },
    If { cond: Box<ComptimeExpr>, then_: Box<ComptimeExpr>, else_: Box<ComptimeExpr> },
    Let { name: String, value: Box<ComptimeExpr>, body: Box<ComptimeExpr> },
    FnCall { name: String, args: Vec<ComptimeExpr> },
    Index { array: Box<ComptimeExpr>, index: Box<ComptimeExpr> },
    Field { object: Box<ComptimeExpr>, field: String },
    ArrayLiteral(Vec<ComptimeExpr>),
    For { var: String, start: Box<ComptimeExpr>, end: Box<ComptimeExpr>, body: Box<ComptimeExpr>, acc: Box<ComptimeExpr> },
    Block(Vec<ComptimeExpr>),
    StringConcat(Vec<ComptimeExpr>),
}

#[derive(Debug, Clone)]
pub enum ComptimeOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
}

#[derive(Debug, Clone)]
pub enum ComptimeUnaryOp {
    Neg, Not, BitNot,
}

/// The compile-time evaluator.
pub struct ComptimeEvaluator {
    /// Compile-time functions
    functions: HashMap<String, ComptimeFn>,
    /// Named constants
    constants: HashMap<String, ComptimeValue>,
}

type ComptimeFn = fn(&[ComptimeValue]) -> Result<ComptimeValue, String>;

impl ComptimeEvaluator {
    pub fn new() -> Self {
        let mut eval = ComptimeEvaluator {
            functions: HashMap::new(),
            constants: HashMap::new(),
        };
        eval.register_builtins();
        eval
    }

    fn register_builtins(&mut self) {
        // Math functions
        self.functions.insert("abs".to_string(), |args| {
            match args.first() {
                Some(ComptimeValue::Int(i)) => Ok(ComptimeValue::Int(i.abs())),
                Some(ComptimeValue::Float(f)) => Ok(ComptimeValue::Float(f.abs())),
                _ => Err("abs requires a numeric argument".to_string()),
            }
        });
        self.functions.insert("min".to_string(), |args| {
            match (args.get(0), args.get(1)) {
                (Some(ComptimeValue::Int(a)), Some(ComptimeValue::Int(b))) =>
                    Ok(ComptimeValue::Int(*a.min(b))),
                _ => Err("min requires two integer arguments".to_string()),
            }
        });
        self.functions.insert("max".to_string(), |args| {
            match (args.get(0), args.get(1)) {
                (Some(ComptimeValue::Int(a)), Some(ComptimeValue::Int(b))) =>
                    Ok(ComptimeValue::Int(*a.max(b))),
                _ => Err("max requires two integer arguments".to_string()),
            }
        });
        self.functions.insert("len".to_string(), |args| {
            match args.first() {
                Some(ComptimeValue::Array(a)) => Ok(ComptimeValue::Int(a.len() as i64)),
                Some(ComptimeValue::String(s)) => Ok(ComptimeValue::Int(s.len() as i64)),
                _ => Err("len requires an array or string argument".to_string()),
            }
        });
        self.functions.insert("type_name".to_string(), |args| {
            match args.first() {
                Some(ComptimeValue::Int(_)) => Ok(ComptimeValue::String("i64".to_string())),
                Some(ComptimeValue::Float(_)) => Ok(ComptimeValue::String("f64".to_string())),
                Some(ComptimeValue::Bool(_)) => Ok(ComptimeValue::String("bool".to_string())),
                Some(ComptimeValue::String(_)) => Ok(ComptimeValue::String("String".to_string())),
                Some(ComptimeValue::Array(_)) => Ok(ComptimeValue::String("Array".to_string())),
                Some(ComptimeValue::Type(t)) => Ok(ComptimeValue::String(t.clone())),
                _ => Ok(ComptimeValue::String("unknown".to_string())),
            }
        });
        self.functions.insert("stringify".to_string(), |args| {
            match args.first() {
                Some(v) => Ok(ComptimeValue::String(format!("{}", v))),
                None => Err("stringify requires an argument".to_string()),
            }
        });
        self.functions.insert("concat".to_string(), |args| {
            let mut result = String::new();
            for arg in args {
                match arg {
                    ComptimeValue::String(s) => result.push_str(s),
                    other => result.push_str(&format!("{}", other)),
                }
            }
            Ok(ComptimeValue::String(result))
        });
        self.functions.insert("pow".to_string(), |args| {
            match (args.get(0), args.get(1)) {
                (Some(ComptimeValue::Int(base)), Some(ComptimeValue::Int(exp))) => {
                    if *exp < 0 {
                        Err("Negative exponent not supported for integers".to_string())
                    } else {
                        Ok(ComptimeValue::Int(base.pow(*exp as u32)))
                    }
                }
                _ => Err("pow requires two integer arguments".to_string()),
            }
        });
    }

    /// Define a compile-time constant.
    pub fn define_constant(&mut self, name: &str, value: ComptimeValue) {
        self.constants.insert(name.to_string(), value);
    }

    /// Evaluate a compile-time expression.
    pub fn eval(&self, expr: &ComptimeExpr) -> Result<ComptimeValue, String> {
        self.eval_with_env(expr, &HashMap::new())
    }

    fn eval_with_env(&self, expr: &ComptimeExpr, env: &HashMap<String, ComptimeValue>)
        -> Result<ComptimeValue, String>
    {
        match expr {
            ComptimeExpr::Literal(val) => Ok(val.clone()),

            ComptimeExpr::Var(name) => {
                env.get(name)
                    .or_else(|| self.constants.get(name))
                    .cloned()
                    .ok_or_else(|| format!("Undefined comptime variable: {}", name))
            }

            ComptimeExpr::BinOp { left, op, right } => {
                let lv = self.eval_with_env(left, env)?;
                let rv = self.eval_with_env(right, env)?;
                self.eval_binop(op, &lv, &rv)
            }

            ComptimeExpr::UnaryOp { op, expr } => {
                let val = self.eval_with_env(expr, env)?;
                match op {
                    ComptimeUnaryOp::Neg => match val {
                        ComptimeValue::Int(i) => Ok(ComptimeValue::Int(-i)),
                        ComptimeValue::Float(f) => Ok(ComptimeValue::Float(-f)),
                        _ => Err("Cannot negate non-numeric value".to_string()),
                    },
                    ComptimeUnaryOp::Not => match val {
                        ComptimeValue::Bool(b) => Ok(ComptimeValue::Bool(!b)),
                        _ => Err("Cannot apply 'not' to non-boolean value".to_string()),
                    },
                    ComptimeUnaryOp::BitNot => match val {
                        ComptimeValue::Int(i) => Ok(ComptimeValue::Int(!i)),
                        _ => Err("Cannot apply bitwise not to non-integer value".to_string()),
                    },
                }
            }

            ComptimeExpr::If { cond, then_, else_ } => {
                let cond_val = self.eval_with_env(cond, env)?;
                match cond_val {
                    ComptimeValue::Bool(true) => self.eval_with_env(then_, env),
                    ComptimeValue::Bool(false) => self.eval_with_env(else_, env),
                    _ => Err("If condition must be boolean".to_string()),
                }
            }

            ComptimeExpr::Let { name, value, body } => {
                let val = self.eval_with_env(value, env)?;
                let mut new_env = env.clone();
                new_env.insert(name.clone(), val);
                self.eval_with_env(body, &new_env)
            }

            ComptimeExpr::FnCall { name, args } => {
                let arg_values: Vec<ComptimeValue> = args.iter()
                    .map(|a| self.eval_with_env(a, env))
                    .collect::<Result<_, _>>()?;

                if let Some(func) = self.functions.get(name) {
                    func(&arg_values)
                } else {
                    Err(format!("Undefined comptime function: {}", name))
                }
            }

            ComptimeExpr::Index { array, index } => {
                let arr = self.eval_with_env(array, env)?;
                let idx = self.eval_with_env(index, env)?;
                match (&arr, &idx) {
                    (ComptimeValue::Array(a), ComptimeValue::Int(i)) => {
                        a.get(*i as usize).cloned()
                            .ok_or_else(|| format!("Index {} out of bounds (len {})", i, a.len()))
                    }
                    (ComptimeValue::String(s), ComptimeValue::Int(i)) => {
                        s.chars().nth(*i as usize)
                            .map(ComptimeValue::Char)
                            .ok_or_else(|| format!("Index {} out of bounds", i))
                    }
                    _ => Err("Cannot index into non-array value".to_string()),
                }
            }

            ComptimeExpr::Field { object, field } => {
                let obj = self.eval_with_env(object, env)?;
                match obj {
                    ComptimeValue::Struct { fields, .. } => {
                        fields.iter()
                            .find(|(name, _)| name == field)
                            .map(|(_, v)| v.clone())
                            .ok_or_else(|| format!("No field '{}' on struct", field))
                    }
                    _ => Err("Cannot access field on non-struct value".to_string()),
                }
            }

            ComptimeExpr::ArrayLiteral(elements) => {
                let vals: Vec<ComptimeValue> = elements.iter()
                    .map(|e| self.eval_with_env(e, env))
                    .collect::<Result<_, _>>()?;
                Ok(ComptimeValue::Array(vals))
            }

            ComptimeExpr::For { var, start, end, body, acc } => {
                let start_val = match self.eval_with_env(start, env)? {
                    ComptimeValue::Int(i) => i,
                    _ => return Err("For loop bounds must be integers".to_string()),
                };
                let end_val = match self.eval_with_env(end, env)? {
                    ComptimeValue::Int(i) => i,
                    _ => return Err("For loop bounds must be integers".to_string()),
                };
                let mut accumulator = self.eval_with_env(acc, env)?;
                let mut new_env = env.clone();

                for i in start_val..end_val {
                    new_env.insert(var.clone(), ComptimeValue::Int(i));
                    new_env.insert("__acc".to_string(), accumulator.clone());
                    accumulator = self.eval_with_env(body, &new_env)?;
                }
                Ok(accumulator)
            }

            ComptimeExpr::Block(exprs) => {
                let mut result = ComptimeValue::Null;
                let mut new_env = env.clone();
                for expr in exprs {
                    result = self.eval_with_env(expr, &new_env)?;
                }
                Ok(result)
            }

            ComptimeExpr::StringConcat(parts) => {
                let mut result = String::new();
                for part in parts {
                    let val = self.eval_with_env(part, env)?;
                    result.push_str(&format!("{}", val));
                }
                Ok(ComptimeValue::String(result))
            }
        }
    }

    fn eval_binop(&self, op: &ComptimeOp, left: &ComptimeValue, right: &ComptimeValue)
        -> Result<ComptimeValue, String>
    {
        match op {
            ComptimeOp::Add => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a + b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Float(a + b)),
                (ComptimeValue::String(a), ComptimeValue::String(b)) =>
                    Ok(ComptimeValue::String(format!("{}{}", a, b))),
                _ => Err("Type mismatch in addition".to_string()),
            },
            ComptimeOp::Sub => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a - b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Float(a - b)),
                _ => Err("Type mismatch in subtraction".to_string()),
            },
            ComptimeOp::Mul => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a * b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Float(a * b)),
                _ => Err("Type mismatch in multiplication".to_string()),
            },
            ComptimeOp::Div => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                    if *b == 0 { Err("Division by zero".to_string()) }
                    else { Ok(ComptimeValue::Int(a / b)) }
                }
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => {
                    if *b == 0.0 { Err("Division by zero".to_string()) }
                    else { Ok(ComptimeValue::Float(a / b)) }
                }
                _ => Err("Type mismatch in division".to_string()),
            },
            ComptimeOp::Mod => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => {
                    if *b == 0 { Err("Modulo by zero".to_string()) }
                    else { Ok(ComptimeValue::Int(a % b)) }
                }
                _ => Err("Type mismatch in modulo".to_string()),
            },
            ComptimeOp::Eq => Ok(ComptimeValue::Bool(left == right)),
            ComptimeOp::Ne => Ok(ComptimeValue::Bool(left != right)),
            ComptimeOp::Lt => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a < b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a < b)),
                (ComptimeValue::String(a), ComptimeValue::String(b)) => Ok(ComptimeValue::Bool(a < b)),
                _ => Err("Type mismatch in comparison".to_string()),
            },
            ComptimeOp::Le => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a <= b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a <= b)),
                _ => Err("Type mismatch in comparison".to_string()),
            },
            ComptimeOp::Gt => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a > b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a > b)),
                _ => Err("Type mismatch in comparison".to_string()),
            },
            ComptimeOp::Ge => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Bool(a >= b)),
                (ComptimeValue::Float(a), ComptimeValue::Float(b)) => Ok(ComptimeValue::Bool(a >= b)),
                _ => Err("Type mismatch in comparison".to_string()),
            },
            ComptimeOp::And => match (left, right) {
                (ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => Ok(ComptimeValue::Bool(*a && *b)),
                _ => Err("Logical AND requires booleans".to_string()),
            },
            ComptimeOp::Or => match (left, right) {
                (ComptimeValue::Bool(a), ComptimeValue::Bool(b)) => Ok(ComptimeValue::Bool(*a || *b)),
                _ => Err("Logical OR requires booleans".to_string()),
            },
            ComptimeOp::BitAnd => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a & b)),
                _ => Err("Bitwise AND requires integers".to_string()),
            },
            ComptimeOp::BitOr => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a | b)),
                _ => Err("Bitwise OR requires integers".to_string()),
            },
            ComptimeOp::BitXor => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a ^ b)),
                _ => Err("Bitwise XOR requires integers".to_string()),
            },
            ComptimeOp::Shl => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a << b)),
                _ => Err("Shift left requires integers".to_string()),
            },
            ComptimeOp::Shr => match (left, right) {
                (ComptimeValue::Int(a), ComptimeValue::Int(b)) => Ok(ComptimeValue::Int(a >> b)),
                _ => Err("Shift right requires integers".to_string()),
            },
        }
    }
}

// ============================================================================
// PART 5: EMBEDDED DSL FRAMEWORK
// ============================================================================
// Embedded DSLs let you write domain-specific code inline (SQL queries,
// regex patterns, JSON literals, shell commands, etc.) with compile-time
// validation. The compiler parses and validates the DSL at compile time,
// catching errors early, and generates efficient code for the DSL construct.
//
// Each DSL has:
//   - A parser that validates the syntax
//   - A type checker that verifies references (e.g., table/column names in SQL)
//   - A code generator that produces Chronos IR
//
// The key insight is that DSLs embedded in strings ("SELECT * FROM users")
// are invisible to the compiler — it can't check them. By making them
// first-class syntax (sql! { SELECT * FROM users }), the compiler can
// validate them at compile time and generate optimized code.

/// A DSL validation result.
#[derive(Debug, Clone)]
pub struct DslValidation {
    pub valid: bool,
    pub errors: Vec<DslError>,
    pub warnings: Vec<DslWarning>,
    pub output_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DslError {
    pub message: String,
    pub position: usize,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DslWarning {
    pub message: String,
    pub position: usize,
}

/// Schema information for SQL validation.
#[derive(Debug, Clone)]
pub struct SqlSchema {
    pub tables: HashMap<String, Vec<SqlColumn>>,
}

#[derive(Debug, Clone)]
pub struct SqlColumn {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

/// The SQL DSL validator.
pub struct SqlDsl {
    schema: Option<SqlSchema>,
}

impl SqlDsl {
    pub fn new() -> Self {
        SqlDsl { schema: None }
    }

    pub fn with_schema(schema: SqlSchema) -> Self {
        SqlDsl { schema: Some(schema) }
    }

    /// Validate a SQL string at compile time.
    pub fn validate(&self, sql: &str) -> DslValidation {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let trimmed = sql.trim();

        if trimmed.is_empty() {
            errors.push(DslError {
                message: "Empty SQL statement".to_string(),
                position: 0,
                suggestion: None,
            });
            return DslValidation { valid: false, errors, warnings, output_type: None };
        }

        // Parse the statement type
        let upper = trimmed.to_uppercase();
        let stmt_type = if upper.starts_with("SELECT") { "SELECT" }
            else if upper.starts_with("INSERT") { "INSERT" }
            else if upper.starts_with("UPDATE") { "UPDATE" }
            else if upper.starts_with("DELETE") { "DELETE" }
            else if upper.starts_with("CREATE") { "CREATE" }
            else if upper.starts_with("DROP") { "DROP" }
            else if upper.starts_with("ALTER") { "ALTER" }
            else {
                errors.push(DslError {
                    message: format!("Unknown SQL statement type: {}", trimmed.split_whitespace().next().unwrap_or("")),
                    position: 0,
                    suggestion: Some("Valid statements: SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, ALTER".to_string()),
                });
                return DslValidation { valid: false, errors, warnings, output_type: None };
            };

        // Check for unterminated strings
        let single_quotes = trimmed.matches('\'').count();
        if single_quotes % 2 != 0 {
            errors.push(DslError {
                message: "Unterminated string literal".to_string(),
                position: trimmed.rfind('\'').unwrap_or(0),
                suggestion: Some("Add a closing single quote".to_string()),
            });
        }

        // Check for balanced parentheses
        let mut paren_depth = 0i32;
        for (i, ch) in trimmed.char_indices() {
            match ch {
                '(' => paren_depth += 1,
                ')' => {
                    paren_depth -= 1;
                    if paren_depth < 0 {
                        errors.push(DslError {
                            message: "Unmatched closing parenthesis".to_string(),
                            position: i,
                            suggestion: None,
                        });
                    }
                }
                _ => {}
            }
        }
        if paren_depth > 0 {
            errors.push(DslError {
                message: format!("{} unclosed parenthesis(es)", paren_depth),
                position: trimmed.len(),
                suggestion: Some("Add closing parentheses".to_string()),
            });
        }

        // Validate SELECT structure
        if stmt_type == "SELECT" {
            if !upper.contains("FROM") && !upper.contains("SELECT 1") && !upper.contains("SELECT COUNT") {
                warnings.push(DslWarning {
                    message: "SELECT without FROM clause".to_string(),
                    position: 0,
                });
            }

            // Check for SELECT *
            if upper.contains("SELECT *") || upper.contains("SELECT  *") {
                warnings.push(DslWarning {
                    message: "SELECT * is discouraged — specify columns explicitly".to_string(),
                    position: upper.find("*").unwrap_or(0),
                });
            }

            // Validate table references against schema
            if let Some(schema) = &self.schema {
                if let Some(from_pos) = upper.find("FROM") {
                    let after_from = &trimmed[from_pos + 4..].trim();
                    let table_name = after_from.split_whitespace().next().unwrap_or("");
                    let table_lower = table_name.to_lowercase();
                    if !schema.tables.contains_key(&table_lower) && !table_name.is_empty() {
                        errors.push(DslError {
                            message: format!("Table '{}' does not exist", table_name),
                            position: from_pos + 5,
                            suggestion: {
                                let tables: Vec<&String> = schema.tables.keys().collect();
                                if tables.is_empty() { None }
                                else { Some(format!("Available tables: {}", tables.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "))) }
                            },
                        });
                    }
                }
            }
        }

        // Validate INSERT structure
        if stmt_type == "INSERT" {
            if !upper.contains("INTO") {
                errors.push(DslError {
                    message: "INSERT missing INTO keyword".to_string(),
                    position: 6,
                    suggestion: Some("Use INSERT INTO table_name ...".to_string()),
                });
            }
            if !upper.contains("VALUES") && !upper.contains("SELECT") {
                errors.push(DslError {
                    message: "INSERT missing VALUES or SELECT clause".to_string(),
                    position: trimmed.len(),
                    suggestion: Some("Add VALUES (...) or a SELECT subquery".to_string()),
                });
            }
        }

        // Check for SQL injection patterns (compile-time security check)
        if trimmed.contains("--") {
            warnings.push(DslWarning {
                message: "SQL comment detected — may indicate injection vulnerability if user input is interpolated".to_string(),
                position: trimmed.find("--").unwrap_or(0),
            });
        }
        if upper.contains("; DROP") || upper.contains(";DROP") {
            errors.push(DslError {
                message: "Potential SQL injection: multiple statements detected".to_string(),
                position: trimmed.find(';').unwrap_or(0),
                suggestion: Some("Use parameterized queries instead of string concatenation".to_string()),
            });
        }

        let output_type = match stmt_type {
            "SELECT" => Some("ResultSet".to_string()),
            "INSERT" | "UPDATE" | "DELETE" => Some("AffectedRows(u64)".to_string()),
            _ => Some("()".to_string()),
        };

        DslValidation {
            valid: errors.is_empty(),
            errors,
            warnings,
            output_type,
        }
    }
}

/// The Regex DSL validator — compiles regex patterns at compile time.
pub struct RegexDsl;

impl RegexDsl {
    /// Validate a regex pattern at compile time.
    pub fn validate(pattern: &str) -> DslValidation {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if pattern.is_empty() {
            errors.push(DslError {
                message: "Empty regex pattern".to_string(),
                position: 0,
                suggestion: Some("Provide a pattern like '\\d+' or '[a-z]+'".to_string()),
            });
            return DslValidation { valid: false, errors, warnings, output_type: None };
        }

        // Check for balanced brackets
        let mut bracket_depth = 0i32;
        let mut in_char_class = false;
        let mut escaped = false;
        let mut paren_depth = 0i32;

        for (i, ch) in pattern.char_indices() {
            if escaped {
                escaped = false;
                // Validate escape sequences
                if !in_char_class && !"dDwWsSntrfvbBaAzZpP0123456789.^$*+?{}[]()|\\"
                    .contains(ch) {
                    warnings.push(DslWarning {
                        message: format!("Unnecessary escape: \\{}", ch),
                        position: i - 1,
                    });
                }
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            match ch {
                '[' if !in_char_class => {
                    in_char_class = true;
                    bracket_depth += 1;
                }
                ']' if in_char_class => {
                    in_char_class = false;
                    bracket_depth -= 1;
                }
                '(' if !in_char_class => paren_depth += 1,
                ')' if !in_char_class => {
                    paren_depth -= 1;
                    if paren_depth < 0 {
                        errors.push(DslError {
                            message: "Unmatched closing parenthesis".to_string(),
                            position: i,
                            suggestion: None,
                        });
                    }
                }
                '*' | '+' | '?' if !in_char_class => {
                    // Check for nothing to quantify
                    if i == 0 {
                        errors.push(DslError {
                            message: format!("Quantifier '{}' has nothing to repeat", ch),
                            position: i,
                            suggestion: Some("A quantifier must follow an expression".to_string()),
                        });
                    }
                }
                '{' if !in_char_class => {
                    // Validate quantifier syntax: {n}, {n,}, {n,m}
                    if let Some(close) = pattern[i..].find('}') {
                        let inner = &pattern[i+1..i+close];
                        let parts: Vec<&str> = inner.split(',').collect();
                        let valid = match parts.len() {
                            1 => parts[0].parse::<u32>().is_ok(),
                            2 => {
                                parts[0].parse::<u32>().is_ok()
                                    && (parts[1].is_empty() || parts[1].parse::<u32>().is_ok())
                            }
                            _ => false,
                        };
                        if !valid {
                            errors.push(DslError {
                                message: format!("Invalid quantifier: {{{}}}", inner),
                                position: i,
                                suggestion: Some("Use {{n}}, {{n,}}, or {{n,m}}".to_string()),
                            });
                        }
                    } else {
                        errors.push(DslError {
                            message: "Unclosed quantifier brace".to_string(),
                            position: i,
                            suggestion: Some("Add a closing '}'".to_string()),
                        });
                    }
                }
                _ => {}
            }
        }

        if escaped {
            errors.push(DslError {
                message: "Trailing backslash".to_string(),
                position: pattern.len() - 1,
                suggestion: Some("Escape it as \\\\".to_string()),
            });
        }
        if bracket_depth > 0 {
            errors.push(DslError {
                message: "Unclosed character class '['".to_string(),
                position: pattern.rfind('[').unwrap_or(0),
                suggestion: Some("Add a closing ']'".to_string()),
            });
        }
        if paren_depth > 0 {
            errors.push(DslError {
                message: format!("{} unclosed group(s)", paren_depth),
                position: pattern.len(),
                suggestion: Some("Add closing parentheses".to_string()),
            });
        }

        // Warn about common mistakes
        if pattern == ".*" {
            warnings.push(DslWarning {
                message: "'.*' matches everything — is this intentional?".to_string(),
                position: 0,
            });
        }

        DslValidation {
            valid: errors.is_empty(),
            errors,
            warnings,
            output_type: Some("CompiledRegex".to_string()),
        }
    }
}

/// The JSON DSL validator — parses JSON at compile time and generates
/// typed accessors.
pub struct JsonDsl;

impl JsonDsl {
    /// Validate a JSON string at compile time.
    pub fn validate(json: &str) -> DslValidation {
        let mut errors = Vec::new();
        let trimmed = json.trim();

        if trimmed.is_empty() {
            errors.push(DslError {
                message: "Empty JSON".to_string(),
                position: 0,
                suggestion: None,
            });
            return DslValidation { valid: false, errors, warnings: vec![], output_type: None };
        }

        // Simple recursive descent JSON validator
        match Self::validate_value(trimmed, 0) {
            Ok((_, end)) => {
                let remaining = trimmed[end..].trim();
                if !remaining.is_empty() {
                    errors.push(DslError {
                        message: format!("Unexpected content after JSON value: '{}'",
                                         &remaining[..remaining.len().min(20)]),
                        position: end,
                        suggestion: None,
                    });
                }
            }
            Err(e) => errors.push(e),
        }

        let output_type = if trimmed.starts_with('{') {
            Some("JsonObject".to_string())
        } else if trimmed.starts_with('[') {
            Some("JsonArray".to_string())
        } else {
            Some("JsonValue".to_string())
        };

        DslValidation {
            valid: errors.is_empty(),
            errors,
            warnings: vec![],
            output_type,
        }
    }

    fn validate_value(s: &str, pos: usize) -> Result<((), usize), DslError> {
        let trimmed = s[pos..].trim_start();
        let offset = s.len() - s[pos..].len() + (s[pos..].len() - trimmed.len());

        if trimmed.is_empty() {
            return Err(DslError {
                message: "Unexpected end of JSON".to_string(),
                position: pos,
                suggestion: None,
            });
        }

        let first = trimmed.chars().next().unwrap();
        match first {
            '"' => Self::validate_string(s, offset),
            '{' => Self::validate_object(s, offset),
            '[' => Self::validate_array(s, offset),
            't' | 'f' | 'n' => Self::validate_keyword(s, offset),
            '-' | '0'..='9' => Self::validate_number(s, offset),
            _ => Err(DslError {
                message: format!("Unexpected character: '{}'", first),
                position: offset,
                suggestion: None,
            }),
        }
    }

    fn validate_string(s: &str, pos: usize) -> Result<((), usize), DslError> {
        if !s[pos..].starts_with('"') {
            return Err(DslError {
                message: "Expected string".to_string(), position: pos, suggestion: None,
            });
        }
        let mut i = pos + 1;
        let mut escaped = false;
        while i < s.len() {
            let ch = s.as_bytes()[i] as char;
            if escaped {
                escaped = false;
                i += 1;
                continue;
            }
            if ch == '\\' { escaped = true; i += 1; continue; }
            if ch == '"' { return Ok(((), i + 1)); }
            i += 1;
        }
        Err(DslError {
            message: "Unterminated string".to_string(), position: pos, suggestion: None,
        })
    }

    fn validate_object(s: &str, pos: usize) -> Result<((), usize), DslError> {
        let mut i = pos + 1; // skip '{'
        i += s[i..].len() - s[i..].trim_start().len(); // skip whitespace
        if i < s.len() && s.as_bytes()[i] as char == '}' {
            return Ok(((), i + 1));
        }
        loop {
            i += s[i..].len() - s[i..].trim_start().len();
            let (_, end) = Self::validate_string(s, i)?;
            i = end;
            i += s[i..].len() - s[i..].trim_start().len();
            if i >= s.len() || s.as_bytes()[i] as char != ':' {
                return Err(DslError {
                    message: "Expected ':' after object key".to_string(),
                    position: i, suggestion: None,
                });
            }
            i += 1;
            let (_, end) = Self::validate_value(s, i)?;
            i = end;
            i += s[i..].len() - s[i..].trim_start().len();
            if i >= s.len() {
                return Err(DslError {
                    message: "Unterminated object".to_string(), position: pos, suggestion: None,
                });
            }
            match s.as_bytes()[i] as char {
                '}' => return Ok(((), i + 1)),
                ',' => { i += 1; }
                _ => return Err(DslError {
                    message: "Expected ',' or '}' in object".to_string(),
                    position: i, suggestion: None,
                }),
            }
        }
    }

    fn validate_array(s: &str, pos: usize) -> Result<((), usize), DslError> {
        let mut i = pos + 1;
        i += s[i..].len() - s[i..].trim_start().len();
        if i < s.len() && s.as_bytes()[i] as char == ']' {
            return Ok(((), i + 1));
        }
        loop {
            let (_, end) = Self::validate_value(s, i)?;
            i = end;
            i += s[i..].len() - s[i..].trim_start().len();
            if i >= s.len() {
                return Err(DslError {
                    message: "Unterminated array".to_string(), position: pos, suggestion: None,
                });
            }
            match s.as_bytes()[i] as char {
                ']' => return Ok(((), i + 1)),
                ',' => { i += 1; }
                _ => return Err(DslError {
                    message: "Expected ',' or ']' in array".to_string(),
                    position: i, suggestion: None,
                }),
            }
        }
    }

    fn validate_keyword(s: &str, pos: usize) -> Result<((), usize), DslError> {
        for kw in &["true", "false", "null"] {
            if s[pos..].starts_with(kw) {
                return Ok(((), pos + kw.len()));
            }
        }
        Err(DslError {
            message: format!("Invalid keyword starting at position {}", pos),
            position: pos,
            suggestion: Some("Valid keywords: true, false, null".to_string()),
        })
    }

    fn validate_number(s: &str, pos: usize) -> Result<((), usize), DslError> {
        let mut i = pos;
        if i < s.len() && s.as_bytes()[i] == b'-' { i += 1; }
        if i >= s.len() || !s.as_bytes()[i].is_ascii_digit() {
            return Err(DslError {
                message: "Invalid number".to_string(), position: pos, suggestion: None,
            });
        }
        while i < s.len() && s.as_bytes()[i].is_ascii_digit() { i += 1; }
        if i < s.len() && s.as_bytes()[i] == b'.' {
            i += 1;
            while i < s.len() && s.as_bytes()[i].is_ascii_digit() { i += 1; }
        }
        if i < s.len() && (s.as_bytes()[i] == b'e' || s.as_bytes()[i] == b'E') {
            i += 1;
            if i < s.len() && (s.as_bytes()[i] == b'+' || s.as_bytes()[i] == b'-') { i += 1; }
            while i < s.len() && s.as_bytes()[i].is_ascii_digit() { i += 1; }
        }
        Ok(((), i))
    }
}

/// The Shell DSL validator — validates shell commands at compile time
/// and detects potential injection vulnerabilities.
pub struct ShellDsl;

impl ShellDsl {
    pub fn validate(command: &str) -> DslValidation {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        if command.trim().is_empty() {
            errors.push(DslError {
                message: "Empty shell command".to_string(),
                position: 0,
                suggestion: None,
            });
            return DslValidation { valid: false, errors, warnings, output_type: None };
        }

        // Check for dangerous commands
        let dangerous = ["rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
                         ":(){ :|:& };:", "chmod -R 777 /"];
        for pattern in &dangerous {
            if command.contains(pattern) {
                errors.push(DslError {
                    message: format!("Dangerous command detected: '{}'", pattern),
                    position: command.find(pattern).unwrap_or(0),
                    suggestion: Some("This command could cause catastrophic data loss".to_string()),
                });
            }
        }

        // Check for unescaped variable interpolation (injection risk)
        let mut in_single_quote = false;
        let mut in_double_quote = false;
        for (i, ch) in command.char_indices() {
            match ch {
                '\'' if !in_double_quote => in_single_quote = !in_single_quote,
                '"' if !in_single_quote => in_double_quote = !in_double_quote,
                '$' if !in_single_quote => {
                    // Check if this is an unquoted variable
                    if !in_double_quote {
                        warnings.push(DslWarning {
                            message: "Unquoted variable expansion — potential injection risk".to_string(),
                            position: i,
                        });
                    }
                }
                '`' => {
                    warnings.push(DslWarning {
                        message: "Backtick command substitution — use $() instead".to_string(),
                        position: i,
                    });
                }
                _ => {}
            }
        }

        // Check for unterminated quotes
        if in_single_quote {
            errors.push(DslError {
                message: "Unterminated single quote".to_string(),
                position: command.rfind('\'').unwrap_or(0),
                suggestion: None,
            });
        }
        if in_double_quote {
            errors.push(DslError {
                message: "Unterminated double quote".to_string(),
                position: command.rfind('"').unwrap_or(0),
                suggestion: None,
            });
        }

        // Check for pipe chains without error handling
        if command.contains('|') && !command.contains("set -o pipefail") {
            warnings.push(DslWarning {
                message: "Pipe chain without 'set -o pipefail' — errors in earlier commands may be silently ignored".to_string(),
                position: command.find('|').unwrap_or(0),
            });
        }

        DslValidation {
            valid: errors.is_empty(),
            errors,
            warnings,
            output_type: Some("CommandResult".to_string()),
        }
    }
}

// ============================================================================
// PART 6: DERIVE MACRO IMPLEMENTATIONS
// ============================================================================
// Derive macros auto-generate trait implementations for structs and enums.
// They inspect the structure definition (fields, types, attributes) and
// produce the boilerplate code that would otherwise be written by hand.

/// Information about a struct/enum extracted from tokens for derive macros.
#[derive(Debug, Clone)]
pub struct DeriveInput {
    pub name: String,
    pub kind: DeriveItemKind,
    pub fields: Vec<DeriveField>,
    pub variants: Vec<DeriveVariant>,  // for enums
    pub attributes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeriveItemKind {
    Struct,
    Enum,
    Union,
}

#[derive(Debug, Clone)]
pub struct DeriveField {
    pub name: String,
    pub ty: String,
    pub attributes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DeriveVariant {
    pub name: String,
    pub fields: Vec<DeriveField>,
}

/// Parse a token stream into a DeriveInput (simplified).
pub fn parse_derive_input(tokens: &TokenStream) -> Result<DeriveInput, String> {
    let mut name = String::new();
    let mut kind = DeriveItemKind::Struct;
    let mut fields = Vec::new();
    let mut variants = Vec::new();
    let mut attributes = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        match &tokens.tokens[i].kind {
            TokenKind::Hash => {
                // Attribute: # [ ... ]
                if i + 1 < tokens.len() && tokens.tokens[i + 1].kind == TokenKind::OpenBracket {
                    let mut depth = 1;
                    let mut j = i + 2;
                    let mut attr = String::new();
                    while j < tokens.len() && depth > 0 {
                        match &tokens.tokens[j].kind {
                            TokenKind::OpenBracket => depth += 1,
                            TokenKind::CloseBracket => {
                                depth -= 1;
                                if depth == 0 { break; }
                            }
                            _ => {}
                        }
                        attr.push_str(&format!("{}", tokens.tokens[j].kind));
                        j += 1;
                    }
                    attributes.push(attr);
                    i = j + 1;
                    continue;
                }
            }
            TokenKind::Keyword(kw) if kw == "struct" => {
                kind = DeriveItemKind::Struct;
                i += 1;
                if i < tokens.len() {
                    if let TokenKind::Ident(n) = &tokens.tokens[i].kind {
                        name = n.clone();
                    }
                }
            }
            TokenKind::Keyword(kw) if kw == "enum" => {
                kind = DeriveItemKind::Enum;
                i += 1;
                if i < tokens.len() {
                    if let TokenKind::Ident(n) = &tokens.tokens[i].kind {
                        name = n.clone();
                    }
                }
            }
            TokenKind::Ident(id) if name.is_empty() && (id == "struct" || id == "enum") => {
                kind = if id == "struct" { DeriveItemKind::Struct } else { DeriveItemKind::Enum };
                i += 1;
                if i < tokens.len() {
                    if let TokenKind::Ident(n) = &tokens.tokens[i].kind {
                        name = n.clone();
                    }
                }
            }
            TokenKind::OpenBrace => {
                // Parse fields or variants
                i += 1;
                while i < tokens.len() && tokens.tokens[i].kind != TokenKind::CloseBrace {
                    if let TokenKind::Ident(field_name) = &tokens.tokens[i].kind {
                        let fname = field_name.clone();
                        i += 1;
                        // Look for : type
                        if i < tokens.len() && tokens.tokens[i].kind == TokenKind::Colon {
                            i += 1;
                            if i < tokens.len() {
                                let ty = format!("{}", tokens.tokens[i].kind);
                                if kind == DeriveItemKind::Struct {
                                    fields.push(DeriveField {
                                        name: fname,
                                        ty,
                                        attributes: vec![],
                                    });
                                }
                                i += 1;
                            }
                        } else if kind == DeriveItemKind::Enum {
                            variants.push(DeriveVariant {
                                name: fname,
                                fields: vec![],
                            });
                        }
                    }
                    // Skip commas
                    if i < tokens.len() && tokens.tokens[i].kind == TokenKind::Comma {
                        i += 1;
                    } else {
                        i += 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    Ok(DeriveInput { name, kind, fields, variants, attributes })
}

/// Generate a Debug implementation for a struct.
pub fn derive_debug(input: &DeriveInput) -> TokenStream {
    let mut tokens = Vec::new();

    // impl Debug for Name { fn fmt(...) { ... } }
    tokens.push(Token::new(TokenKind::Ident("impl".into())));
    tokens.push(Token::new(TokenKind::Ident("Debug".into())));
    tokens.push(Token::new(TokenKind::Ident("for".into())));
    tokens.push(Token::new(TokenKind::Ident(input.name.clone())));
    tokens.push(Token::new(TokenKind::OpenBrace));
    tokens.push(Token::new(TokenKind::Ident("fn".into())));
    tokens.push(Token::new(TokenKind::Ident("fmt".into())));
    tokens.push(Token::new(TokenKind::OpenParen));
    tokens.push(Token::new(TokenKind::Ampersand));
    tokens.push(Token::new(TokenKind::Ident("self".into())));
    tokens.push(Token::new(TokenKind::Comma));
    tokens.push(Token::new(TokenKind::Ident("f".into())));
    tokens.push(Token::new(TokenKind::Colon));
    tokens.push(Token::new(TokenKind::Ampersand));
    tokens.push(Token::new(TokenKind::Ident("mut".into())));
    tokens.push(Token::new(TokenKind::Ident("Formatter".into())));
    tokens.push(Token::new(TokenKind::CloseParen));
    tokens.push(Token::new(TokenKind::Arrow));
    tokens.push(Token::new(TokenKind::Ident("Result".into())));
    tokens.push(Token::new(TokenKind::OpenBrace));

    // f.debug_struct("Name").field("x", &self.x).field("y", &self.y).finish()
    tokens.push(Token::new(TokenKind::Ident("f".into())));
    tokens.push(Token::new(TokenKind::Dot));
    tokens.push(Token::new(TokenKind::Ident("debug_struct".into())));
    tokens.push(Token::new(TokenKind::OpenParen));
    tokens.push(Token::new(TokenKind::StringLiteral(input.name.clone())));
    tokens.push(Token::new(TokenKind::CloseParen));

    for field in &input.fields {
        tokens.push(Token::new(TokenKind::Dot));
        tokens.push(Token::new(TokenKind::Ident("field".into())));
        tokens.push(Token::new(TokenKind::OpenParen));
        tokens.push(Token::new(TokenKind::StringLiteral(field.name.clone())));
        tokens.push(Token::new(TokenKind::Comma));
        tokens.push(Token::new(TokenKind::Ampersand));
        tokens.push(Token::new(TokenKind::Ident("self".into())));
        tokens.push(Token::new(TokenKind::Dot));
        tokens.push(Token::new(TokenKind::Ident(field.name.clone())));
        tokens.push(Token::new(TokenKind::CloseParen));
    }

    tokens.push(Token::new(TokenKind::Dot));
    tokens.push(Token::new(TokenKind::Ident("finish".into())));
    tokens.push(Token::new(TokenKind::OpenParen));
    tokens.push(Token::new(TokenKind::CloseParen));

    tokens.push(Token::new(TokenKind::CloseBrace));
    tokens.push(Token::new(TokenKind::CloseBrace));

    TokenStream::from_tokens(tokens)
}

/// Generate an Eq/PartialEq implementation for a struct.
pub fn derive_eq(input: &DeriveInput) -> TokenStream {
    let mut tokens = Vec::new();

    tokens.push(Token::new(TokenKind::Ident("impl".into())));
    tokens.push(Token::new(TokenKind::Ident("PartialEq".into())));
    tokens.push(Token::new(TokenKind::Ident("for".into())));
    tokens.push(Token::new(TokenKind::Ident(input.name.clone())));
    tokens.push(Token::new(TokenKind::OpenBrace));
    tokens.push(Token::new(TokenKind::Ident("fn".into())));
    tokens.push(Token::new(TokenKind::Ident("eq".into())));
    tokens.push(Token::new(TokenKind::OpenParen));
    tokens.push(Token::new(TokenKind::Ampersand));
    tokens.push(Token::new(TokenKind::Ident("self".into())));
    tokens.push(Token::new(TokenKind::Comma));
    tokens.push(Token::new(TokenKind::Ident("other".into())));
    tokens.push(Token::new(TokenKind::Colon));
    tokens.push(Token::new(TokenKind::Ampersand));
    tokens.push(Token::new(TokenKind::Ident(input.name.clone())));
    tokens.push(Token::new(TokenKind::CloseParen));
    tokens.push(Token::new(TokenKind::Arrow));
    tokens.push(Token::new(TokenKind::Ident("bool".into())));
    tokens.push(Token::new(TokenKind::OpenBrace));

    // self.field1 == other.field1 && self.field2 == other.field2 && ...
    for (i, field) in input.fields.iter().enumerate() {
        if i > 0 {
            tokens.push(Token::new(TokenKind::And));
        }
        tokens.push(Token::new(TokenKind::Ident("self".into())));
        tokens.push(Token::new(TokenKind::Dot));
        tokens.push(Token::new(TokenKind::Ident(field.name.clone())));
        tokens.push(Token::new(TokenKind::Eq));
        tokens.push(Token::new(TokenKind::Ident("other".into())));
        tokens.push(Token::new(TokenKind::Dot));
        tokens.push(Token::new(TokenKind::Ident(field.name.clone())));
    }

    tokens.push(Token::new(TokenKind::CloseBrace));
    tokens.push(Token::new(TokenKind::CloseBrace));

    TokenStream::from_tokens(tokens)
}

/// Generate a Clone implementation for a struct.
pub fn derive_clone(input: &DeriveInput) -> TokenStream {
    let mut tokens = Vec::new();

    tokens.push(Token::new(TokenKind::Ident("impl".into())));
    tokens.push(Token::new(TokenKind::Ident("Clone".into())));
    tokens.push(Token::new(TokenKind::Ident("for".into())));
    tokens.push(Token::new(TokenKind::Ident(input.name.clone())));
    tokens.push(Token::new(TokenKind::OpenBrace));
    tokens.push(Token::new(TokenKind::Ident("fn".into())));
    tokens.push(Token::new(TokenKind::Ident("clone".into())));
    tokens.push(Token::new(TokenKind::OpenParen));
    tokens.push(Token::new(TokenKind::Ampersand));
    tokens.push(Token::new(TokenKind::Ident("self".into())));
    tokens.push(Token::new(TokenKind::CloseParen));
    tokens.push(Token::new(TokenKind::Arrow));
    tokens.push(Token::new(TokenKind::Ident(input.name.clone())));
    tokens.push(Token::new(TokenKind::OpenBrace));

    // Name { field1: self.field1.clone(), field2: self.field2.clone() }
    tokens.push(Token::new(TokenKind::Ident(input.name.clone())));
    tokens.push(Token::new(TokenKind::OpenBrace));
    for (i, field) in input.fields.iter().enumerate() {
        if i > 0 { tokens.push(Token::new(TokenKind::Comma)); }
        tokens.push(Token::new(TokenKind::Ident(field.name.clone())));
        tokens.push(Token::new(TokenKind::Colon));
        tokens.push(Token::new(TokenKind::Ident("self".into())));
        tokens.push(Token::new(TokenKind::Dot));
        tokens.push(Token::new(TokenKind::Ident(field.name.clone())));
        tokens.push(Token::new(TokenKind::Dot));
        tokens.push(Token::new(TokenKind::Ident("clone".into())));
        tokens.push(Token::new(TokenKind::OpenParen));
        tokens.push(Token::new(TokenKind::CloseParen));
    }
    tokens.push(Token::new(TokenKind::CloseBrace));

    tokens.push(Token::new(TokenKind::CloseBrace));
    tokens.push(Token::new(TokenKind::CloseBrace));

    TokenStream::from_tokens(tokens)
}

// ============================================================================
// PART 7: QUASI-QUOTATION
// ============================================================================
// Quasi-quotation provides a convenient way to construct token streams
// programmatically. Instead of building tokens one by one, you write
// template code with "holes" that are filled in with computed values.
//
// quote! { fn #name() -> #return_type { #body } }
//
// The # markers indicate interpolation points where runtime values are
// spliced into the token stream. This is the primary interface for
// procedural macro authors.

/// A quasi-quotation builder.
pub struct QuasiQuoter {
    tokens: Vec<Token>,
}

impl QuasiQuoter {
    pub fn new() -> Self {
        QuasiQuoter { tokens: Vec::new() }
    }

    /// Add a literal token.
    pub fn token(mut self, kind: TokenKind) -> Self {
        self.tokens.push(Token::new(kind));
        self
    }

    /// Add an identifier.
    pub fn ident(mut self, name: &str) -> Self {
        self.tokens.push(Token::new(TokenKind::Ident(name.to_string())));
        self
    }

    /// Splice in a token stream (interpolation).
    pub fn splice(mut self, stream: &TokenStream) -> Self {
        self.tokens.extend(stream.tokens.iter().cloned());
        self
    }

    /// Splice in an integer literal.
    pub fn int_lit(mut self, value: i64) -> Self {
        self.tokens.push(Token::new(TokenKind::IntLiteral(value)));
        self
    }

    /// Splice in a string literal.
    pub fn string_lit(mut self, value: &str) -> Self {
        self.tokens.push(Token::new(TokenKind::StringLiteral(value.to_string())));
        self
    }

    /// Repeat a pattern for each item in a list.
    pub fn repeat<T, F>(mut self, items: &[T], separator: Option<TokenKind>, f: F) -> Self
    where
        F: Fn(&T) -> Vec<Token>,
    {
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                if let Some(sep) = &separator {
                    self.tokens.push(Token::new(sep.clone()));
                }
            }
            self.tokens.extend(f(item));
        }
        self
    }

    /// Build the final token stream.
    pub fn build(self) -> TokenStream {
        TokenStream::from_tokens(self.tokens)
    }
}

// ============================================================================
// PART 8: MACRO EXPANDER (orchestrates everything)
// ============================================================================
// The macro expander sits in the compilation pipeline between parsing
// and type checking. It walks the AST, identifies macro invocations,
// expands them (possibly recursively), and produces a fully expanded AST.

/// The unified macro expansion engine.
pub struct MacroExpander {
    pub declarative: DeclarativeMacroEngine,
    pub procedural: ProcMacroRegistry,
    pub comptime: ComptimeEvaluator,
    pub expansion_depth: usize,
    pub max_expansion_depth: usize,
}

impl MacroExpander {
    pub fn new() -> Self {
        MacroExpander {
            declarative: DeclarativeMacroEngine::new(),
            procedural: ProcMacroRegistry::new(),
            comptime: ComptimeEvaluator::new(),
            expansion_depth: 0,
            max_expansion_depth: 128,
        }
    }

    /// Expand a macro invocation. Handles all macro types.
    pub fn expand(&mut self, name: &str, input: &TokenStream,
                  kind: MacroInvocationKind) -> Result<TokenStream, String>
    {
        self.expansion_depth += 1;
        if self.expansion_depth > self.max_expansion_depth {
            self.expansion_depth -= 1;
            return Err(format!("Maximum macro expansion depth ({}) exceeded — possible infinite recursion",
                               self.max_expansion_depth));
        }

        let result = match kind {
            MacroInvocationKind::Declarative => {
                self.declarative.expand(name, input)
            }
            MacroInvocationKind::FunctionLike => {
                self.procedural.invoke_function_like(name, input)
            }
            MacroInvocationKind::Attribute(ref item) => {
                self.procedural.invoke_attribute(name, input, item)
            }
            MacroInvocationKind::Derive => {
                self.procedural.invoke_derive(name, input)
            }
        };

        self.expansion_depth -= 1;
        result
    }
}

#[derive(Debug, Clone)]
pub enum MacroInvocationKind {
    Declarative,
    FunctionLike,
    Attribute(TokenStream),
    Derive,
}

// ============================================================================
// PART 9: TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- Declarative Macro Tests ---

    #[test]
    fn test_declarative_macro_simple_substitution() {
        let mut engine = DeclarativeMacroEngine::new();

        // Define: my_const!($name:ident, $val:literal) => { const $name = $val; }
        engine.define(DeclarativeMacro {
            name: "my_const".to_string(),
            arms: vec![MacroArm {
                pattern: MacroPattern::Sequence(vec![
                    MacroPattern::Fragment { name: "name".into(), spec: FragmentSpec::Ident },
                    MacroPattern::Literal(TokenKind::Comma),
                    MacroPattern::Fragment { name: "val".into(), spec: FragmentSpec::Literal },
                ]),
                template: MacroTemplate::Sequence(vec![
                    MacroTemplate::Literal(TokenKind::Ident("const".into())),
                    MacroTemplate::Variable("name".into()),
                    MacroTemplate::Literal(TokenKind::Assign),
                    MacroTemplate::Variable("val".into()),
                    MacroTemplate::Literal(TokenKind::Semicolon),
                ]),
            }],
            hygiene_ctx: 0,
        });

        // Invoke: my_const!(MAX, 100)
        let input = TokenStream::from_tokens(vec![
            Token::new(TokenKind::Ident("MAX".into())),
            Token::new(TokenKind::Comma),
            Token::new(TokenKind::IntLiteral(100)),
        ]);

        let result = engine.expand("my_const", &input).unwrap();
        let output = result.to_string();
        assert!(output.contains("const"), "Output: {}", output);
        assert!(output.contains("MAX"), "Output: {}", output);
        assert!(output.contains("100"), "Output: {}", output);
    }

    #[test]
    fn test_declarative_macro_repetition() {
        let mut engine = DeclarativeMacroEngine::new();

        // Define: vec_of!($($x:literal),*) => { vec![$($x),*] }
        engine.define(DeclarativeMacro {
            name: "vec_of".to_string(),
            arms: vec![MacroArm {
                pattern: MacroPattern::Repetition {
                    patterns: vec![
                        MacroPattern::Fragment { name: "x".into(), spec: FragmentSpec::Literal },
                    ],
                    separator: Some(TokenKind::Comma),
                    kind: RepKind::ZeroOrMore,
                },
                template: MacroTemplate::Sequence(vec![
                    MacroTemplate::Literal(TokenKind::Ident("vec".into())),
                    MacroTemplate::Literal(TokenKind::Not),
                    MacroTemplate::Literal(TokenKind::OpenBracket),
                    MacroTemplate::Repetition {
                        templates: vec![MacroTemplate::Variable("x".into())],
                        separator: Some(TokenKind::Comma),
                    },
                    MacroTemplate::Literal(TokenKind::CloseBracket),
                ]),
            }],
            hygiene_ctx: 0,
        });

        let input = TokenStream::from_tokens(vec![
            Token::new(TokenKind::IntLiteral(1)),
            Token::new(TokenKind::Comma),
            Token::new(TokenKind::IntLiteral(2)),
            Token::new(TokenKind::Comma),
            Token::new(TokenKind::IntLiteral(3)),
        ]);

        let result = engine.expand("vec_of", &input).unwrap();
        let output = result.to_string();
        assert!(output.contains("1"), "Output: {}", output);
        assert!(output.contains("2"), "Output: {}", output);
        assert!(output.contains("3"), "Output: {}", output);
    }

    #[test]
    fn test_declarative_macro_no_match() {
        let mut engine = DeclarativeMacroEngine::new();
        engine.define(DeclarativeMacro {
            name: "needs_ident".to_string(),
            arms: vec![MacroArm {
                pattern: MacroPattern::Fragment { name: "x".into(), spec: FragmentSpec::Ident },
                template: MacroTemplate::Variable("x".into()),
            }],
            hygiene_ctx: 0,
        });

        // Pass a literal instead of an identifier — should fail
        let input = TokenStream::from_tokens(vec![
            Token::new(TokenKind::IntLiteral(42)),
        ]);

        let result = engine.expand("needs_ident", &input);
        assert!(result.is_err());
    }

    // --- Comptime Tests ---

    #[test]
    fn test_comptime_arithmetic() {
        let eval = ComptimeEvaluator::new();

        // 2 + 3 * 4 (we build the AST manually; precedence is in the parser)
        let expr = ComptimeExpr::BinOp {
            left: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(2))),
            op: ComptimeOp::Add,
            right: Box::new(ComptimeExpr::BinOp {
                left: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(3))),
                op: ComptimeOp::Mul,
                right: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(4))),
            }),
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::Int(14));
    }

    #[test]
    fn test_comptime_conditionals() {
        let eval = ComptimeEvaluator::new();

        let expr = ComptimeExpr::If {
            cond: Box::new(ComptimeExpr::BinOp {
                left: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(5))),
                op: ComptimeOp::Gt,
                right: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(3))),
            }),
            then_: Box::new(ComptimeExpr::Literal(ComptimeValue::String("yes".into()))),
            else_: Box::new(ComptimeExpr::Literal(ComptimeValue::String("no".into()))),
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::String("yes".into()));
    }

    #[test]
    fn test_comptime_let_binding() {
        let eval = ComptimeEvaluator::new();

        // let x = 10; x * x
        let expr = ComptimeExpr::Let {
            name: "x".into(),
            value: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(10))),
            body: Box::new(ComptimeExpr::BinOp {
                left: Box::new(ComptimeExpr::Var("x".into())),
                op: ComptimeOp::Mul,
                right: Box::new(ComptimeExpr::Var("x".into())),
            }),
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::Int(100));
    }

    #[test]
    fn test_comptime_builtin_functions() {
        let eval = ComptimeEvaluator::new();

        let abs_expr = ComptimeExpr::FnCall {
            name: "abs".into(),
            args: vec![ComptimeExpr::Literal(ComptimeValue::Int(-42))],
        };
        assert_eq!(eval.eval(&abs_expr).unwrap(), ComptimeValue::Int(42));

        let max_expr = ComptimeExpr::FnCall {
            name: "max".into(),
            args: vec![
                ComptimeExpr::Literal(ComptimeValue::Int(3)),
                ComptimeExpr::Literal(ComptimeValue::Int(7)),
            ],
        };
        assert_eq!(eval.eval(&max_expr).unwrap(), ComptimeValue::Int(7));

        let len_expr = ComptimeExpr::FnCall {
            name: "len".into(),
            args: vec![ComptimeExpr::Literal(ComptimeValue::String("hello".into()))],
        };
        assert_eq!(eval.eval(&len_expr).unwrap(), ComptimeValue::Int(5));

        let pow_expr = ComptimeExpr::FnCall {
            name: "pow".into(),
            args: vec![
                ComptimeExpr::Literal(ComptimeValue::Int(2)),
                ComptimeExpr::Literal(ComptimeValue::Int(10)),
            ],
        };
        assert_eq!(eval.eval(&pow_expr).unwrap(), ComptimeValue::Int(1024));
    }

    #[test]
    fn test_comptime_for_loop_summation() {
        let eval = ComptimeEvaluator::new();

        // sum = for i in 0..5 { acc + i } starting from 0
        // = 0 + 1 + 2 + 3 + 4 = 10
        let expr = ComptimeExpr::For {
            var: "i".into(),
            start: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(0))),
            end: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(5))),
            body: Box::new(ComptimeExpr::BinOp {
                left: Box::new(ComptimeExpr::Var("__acc".into())),
                op: ComptimeOp::Add,
                right: Box::new(ComptimeExpr::Var("i".into())),
            }),
            acc: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(0))),
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::Int(10));
    }

    #[test]
    fn test_comptime_array_and_index() {
        let eval = ComptimeEvaluator::new();

        let expr = ComptimeExpr::Index {
            array: Box::new(ComptimeExpr::ArrayLiteral(vec![
                ComptimeExpr::Literal(ComptimeValue::Int(10)),
                ComptimeExpr::Literal(ComptimeValue::Int(20)),
                ComptimeExpr::Literal(ComptimeValue::Int(30)),
            ])),
            index: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(1))),
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::Int(20));
    }

    #[test]
    fn test_comptime_struct_field_access() {
        let eval = ComptimeEvaluator::new();

        let expr = ComptimeExpr::Field {
            object: Box::new(ComptimeExpr::Literal(ComptimeValue::Struct {
                name: "Point".into(),
                fields: vec![
                    ("x".into(), ComptimeValue::Int(3)),
                    ("y".into(), ComptimeValue::Int(4)),
                ],
            })),
            field: "y".into(),
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::Int(4));
    }

    #[test]
    fn test_comptime_string_concat() {
        let eval = ComptimeEvaluator::new();

        let expr = ComptimeExpr::FnCall {
            name: "concat".into(),
            args: vec![
                ComptimeExpr::Literal(ComptimeValue::String("hello".into())),
                ComptimeExpr::Literal(ComptimeValue::String(" ".into())),
                ComptimeExpr::Literal(ComptimeValue::String("world".into())),
            ],
        };

        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::String("hello world".into()));
    }

    #[test]
    fn test_comptime_bitwise_ops() {
        let eval = ComptimeEvaluator::new();

        let expr = ComptimeExpr::BinOp {
            left: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(0xFF))),
            op: ComptimeOp::BitAnd,
            right: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(0x0F))),
        };
        assert_eq!(eval.eval(&expr).unwrap(), ComptimeValue::Int(0x0F));

        let shl = ComptimeExpr::BinOp {
            left: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(1))),
            op: ComptimeOp::Shl,
            right: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(8))),
        };
        assert_eq!(eval.eval(&shl).unwrap(), ComptimeValue::Int(256));
    }

    #[test]
    fn test_comptime_division_by_zero() {
        let eval = ComptimeEvaluator::new();
        let expr = ComptimeExpr::BinOp {
            left: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(10))),
            op: ComptimeOp::Div,
            right: Box::new(ComptimeExpr::Literal(ComptimeValue::Int(0))),
        };
        assert!(eval.eval(&expr).is_err());
    }

    // --- SQL DSL Tests ---

    #[test]
    fn test_sql_dsl_valid_select() {
        let dsl = SqlDsl::new();
        let result = dsl.validate("SELECT id, name FROM users WHERE age > 21");
        assert!(result.valid, "Errors: {:?}", result.errors);
        assert_eq!(result.output_type, Some("ResultSet".to_string()));
    }

    #[test]
    fn test_sql_dsl_schema_validation() {
        let mut tables = HashMap::new();
        tables.insert("users".to_string(), vec![
            SqlColumn { name: "id".into(), data_type: "INTEGER".into(), nullable: false },
            SqlColumn { name: "name".into(), data_type: "TEXT".into(), nullable: false },
        ]);
        let dsl = SqlDsl::with_schema(SqlSchema { tables });

        // Valid table
        let r1 = dsl.validate("SELECT * FROM users");
        assert!(r1.valid, "Errors: {:?}", r1.errors);

        // Invalid table
        let r2 = dsl.validate("SELECT * FROM nonexistent");
        assert!(!r2.valid);
        assert!(r2.errors[0].message.contains("does not exist"));
    }

    #[test]
    fn test_sql_dsl_injection_detection() {
        let dsl = SqlDsl::new();
        let result = dsl.validate("SELECT * FROM users; DROP TABLE users");
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.message.contains("injection")));
    }

    #[test]
    fn test_sql_dsl_syntax_errors() {
        let dsl = SqlDsl::new();

        let r1 = dsl.validate("INSERT users VALUES (1)");
        assert!(!r1.valid);
        assert!(r1.errors.iter().any(|e| e.message.contains("INTO")));

        let r2 = dsl.validate("SELECT * FROM users WHERE name = 'unterminated");
        assert!(!r2.valid);
        assert!(r2.errors.iter().any(|e| e.message.contains("Unterminated string")));

        let r3 = dsl.validate("SELECT (a + b FROM users");
        assert!(!r3.valid);
        assert!(r3.errors.iter().any(|e| e.message.contains("unclosed parenthesis")));
    }

    #[test]
    fn test_sql_dsl_select_star_warning() {
        let dsl = SqlDsl::new();
        let result = dsl.validate("SELECT * FROM users");
        assert!(result.valid);
        assert!(result.warnings.iter().any(|w| w.message.contains("SELECT *")));
    }

    // --- Regex DSL Tests ---

    #[test]
    fn test_regex_dsl_valid_patterns() {
        let r1 = RegexDsl::validate(r"\d+");
        assert!(r1.valid, "Errors: {:?}", r1.errors);

        let r2 = RegexDsl::validate(r"[a-zA-Z_]\w*");
        assert!(r2.valid, "Errors: {:?}", r2.errors);

        let r3 = RegexDsl::validate(r"^https?://[\w.-]+(/\S*)?$");
        assert!(r3.valid, "Errors: {:?}", r3.errors);
    }

    #[test]
    fn test_regex_dsl_unbalanced_brackets() {
        let r = RegexDsl::validate("[abc");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.message.contains("Unclosed character class")));
    }

    #[test]
    fn test_regex_dsl_unbalanced_parens() {
        let r = RegexDsl::validate("(abc");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.message.contains("unclosed group")));
    }

    #[test]
    fn test_regex_dsl_trailing_backslash() {
        let r = RegexDsl::validate(r"abc\");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.message.contains("Trailing backslash")));
    }

    #[test]
    fn test_regex_dsl_quantifier_nothing() {
        let r = RegexDsl::validate("*abc");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.message.contains("nothing to repeat")));
    }

    // --- JSON DSL Tests ---

    #[test]
    fn test_json_dsl_valid() {
        let r1 = JsonDsl::validate(r#"{"name": "Alice", "age": 30}"#);
        assert!(r1.valid, "Errors: {:?}", r1.errors);
        assert_eq!(r1.output_type, Some("JsonObject".to_string()));

        let r2 = JsonDsl::validate(r#"[1, 2, 3]"#);
        assert!(r2.valid, "Errors: {:?}", r2.errors);
        assert_eq!(r2.output_type, Some("JsonArray".to_string()));

        let r3 = JsonDsl::validate(r#"{"nested": {"a": [1, true, null]}}"#);
        assert!(r3.valid, "Errors: {:?}", r3.errors);
    }

    #[test]
    fn test_json_dsl_invalid() {
        let r1 = JsonDsl::validate(r#"{"key": }"#);
        assert!(!r1.valid);

        let r2 = JsonDsl::validate(r#"{"unterminated": "value"#);
        assert!(!r2.valid);

        let r3 = JsonDsl::validate(r#"[1, 2,]"#);
        // Trailing comma is invalid in strict JSON
        assert!(!r3.valid);
    }

    #[test]
    fn test_json_dsl_empty() {
        let r = JsonDsl::validate("");
        assert!(!r.valid);
        assert!(r.errors[0].message.contains("Empty JSON"));
    }

    // --- Shell DSL Tests ---

    #[test]
    fn test_shell_dsl_valid_command() {
        let r = ShellDsl::validate("ls -la /home");
        assert!(r.valid, "Errors: {:?}", r.errors);
    }

    #[test]
    fn test_shell_dsl_dangerous_command() {
        let r = ShellDsl::validate("rm -rf /");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.message.contains("Dangerous")));
    }

    #[test]
    fn test_shell_dsl_injection_warning() {
        let r = ShellDsl::validate("echo $USER_INPUT");
        assert!(r.valid); // warning, not error
        assert!(r.warnings.iter().any(|w| w.message.contains("injection")));
    }

    #[test]
    fn test_shell_dsl_unterminated_quote() {
        let r = ShellDsl::validate("echo 'hello");
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.message.contains("Unterminated")));
    }

    // --- Derive Macro Tests ---

    #[test]
    fn test_derive_debug_generation() {
        let input = DeriveInput {
            name: "Point".into(),
            kind: DeriveItemKind::Struct,
            fields: vec![
                DeriveField { name: "x".into(), ty: "f64".into(), attributes: vec![] },
                DeriveField { name: "y".into(), ty: "f64".into(), attributes: vec![] },
            ],
            variants: vec![],
            attributes: vec![],
        };

        let output = derive_debug(&input);
        let code = output.to_string();
        assert!(code.contains("impl"), "Code: {}", code);
        assert!(code.contains("Debug"), "Code: {}", code);
        assert!(code.contains("Point"), "Code: {}", code);
        assert!(code.contains("debug_struct"), "Code: {}", code);
        assert!(code.contains("\"x\""), "Code: {}", code);
        assert!(code.contains("\"y\""), "Code: {}", code);
    }

    #[test]
    fn test_derive_eq_generation() {
        let input = DeriveInput {
            name: "Color".into(),
            kind: DeriveItemKind::Struct,
            fields: vec![
                DeriveField { name: "r".into(), ty: "u8".into(), attributes: vec![] },
                DeriveField { name: "g".into(), ty: "u8".into(), attributes: vec![] },
                DeriveField { name: "b".into(), ty: "u8".into(), attributes: vec![] },
            ],
            variants: vec![],
            attributes: vec![],
        };

        let output = derive_eq(&input);
        let code = output.to_string();
        assert!(code.contains("PartialEq"), "Code: {}", code);
        assert!(code.contains("self.r"), "Code: {}", code);
        assert!(code.contains("other.r"), "Code: {}", code);
        assert!(code.contains("&&"), "Code: {}", code);
    }

    #[test]
    fn test_derive_clone_generation() {
        let input = DeriveInput {
            name: "Config".into(),
            kind: DeriveItemKind::Struct,
            fields: vec![
                DeriveField { name: "name".into(), ty: "String".into(), attributes: vec![] },
                DeriveField { name: "value".into(), ty: "i32".into(), attributes: vec![] },
            ],
            variants: vec![],
            attributes: vec![],
        };

        let output = derive_clone(&input);
        let code = output.to_string();
        assert!(code.contains("Clone"), "Code: {}", code);
        assert!(code.contains("clone()"), "Code: {}", code);
        assert!(code.contains("self.name"), "Code: {}", code);
        assert!(code.contains("self.value"), "Code: {}", code);
    }

    // --- Quasi-Quotation Tests ---

    #[test]
    fn test_quasi_quotation_builder() {
        let stream = QuasiQuoter::new()
            .ident("fn")
            .ident("hello")
            .token(TokenKind::OpenParen)
            .token(TokenKind::CloseParen)
            .token(TokenKind::Arrow)
            .ident("i32")
            .token(TokenKind::OpenBrace)
            .int_lit(42)
            .token(TokenKind::CloseBrace)
            .build();

        let code = stream.to_string();
        assert!(code.contains("fn"), "Code: {}", code);
        assert!(code.contains("hello"), "Code: {}", code);
        assert!(code.contains("42"), "Code: {}", code);
    }

    #[test]
    fn test_quasi_quotation_repetition() {
        let fields = vec!["x", "y", "z"];
        let stream = QuasiQuoter::new()
            .ident("struct")
            .ident("Point")
            .token(TokenKind::OpenBrace)
            .repeat(&fields, Some(TokenKind::Comma), |f| {
                vec![
                    Token::new(TokenKind::Ident(f.to_string())),
                    Token::new(TokenKind::Colon),
                    Token::new(TokenKind::Ident("f64".into())),
                ]
            })
            .token(TokenKind::CloseBrace)
            .build();

        let code = stream.to_string();
        assert!(code.contains("x"), "Code: {}", code);
        assert!(code.contains("y"), "Code: {}", code);
        assert!(code.contains("z"), "Code: {}", code);
    }

    // --- Procedural Macro Tests ---

    #[test]
    fn test_proc_macro_function_like() {
        let mut registry = ProcMacroRegistry::new();

        // A proc macro that wraps its input in a timing block
        fn timing_macro(input: &TokenStream, _: Option<&TokenStream>)
            -> Result<TokenStream, String>
        {
            let mut output = TokenStream::new();
            output.push(Token::new(TokenKind::Ident("let".into())));
            output.push(Token::new(TokenKind::Ident("_start".into())));
            output.push(Token::new(TokenKind::Assign));
            output.push(Token::new(TokenKind::Ident("Instant".into())));
            output.push(Token::new(TokenKind::ColonColon));
            output.push(Token::new(TokenKind::Ident("now".into())));
            output.push(Token::new(TokenKind::OpenParen));
            output.push(Token::new(TokenKind::CloseParen));
            output.push(Token::new(TokenKind::Semicolon));
            output.extend(input);
            Ok(output)
        }

        registry.register(ProcMacro {
            name: "timed".into(),
            kind: ProcMacroKind::FunctionLike,
            handler: timing_macro,
        });

        let input = TokenStream::from_tokens(vec![
            Token::new(TokenKind::Ident("do_work".into())),
            Token::new(TokenKind::OpenParen),
            Token::new(TokenKind::CloseParen),
        ]);

        let result = registry.invoke_function_like("timed", &input).unwrap();
        let code = result.to_string();
        assert!(code.contains("Instant"), "Code: {}", code);
        assert!(code.contains("do_work"), "Code: {}", code);
    }

    // --- Macro Expander Tests ---

    #[test]
    fn test_macro_expander_depth_limit() {
        let mut expander = MacroExpander::new();
        expander.max_expansion_depth = 3;

        // Simulate deep expansion
        expander.expansion_depth = 3;
        let input = TokenStream::new();
        let result = expander.expand("any", &input, MacroInvocationKind::Declarative);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("depth"));
    }

    // --- Token Stream Tests ---

    #[test]
    fn test_token_stream_to_string() {
        let stream = TokenStream::from_tokens(vec![
            Token::new(TokenKind::Ident("let".into())),
            Token::new(TokenKind::Ident("x".into())),
            Token::new(TokenKind::Assign),
            Token::new(TokenKind::IntLiteral(42)),
            Token::new(TokenKind::Semicolon),
        ]);

        let code = stream.to_string();
        assert_eq!(code, "let x = 42;");
    }

    #[test]
    fn test_token_stream_delimiter_spacing() {
        let stream = TokenStream::from_tokens(vec![
            Token::new(TokenKind::Ident("foo".into())),
            Token::new(TokenKind::OpenParen),
            Token::new(TokenKind::IntLiteral(1)),
            Token::new(TokenKind::Comma),
            Token::new(TokenKind::IntLiteral(2)),
            Token::new(TokenKind::CloseParen),
        ]);

        let code = stream.to_string();
        assert_eq!(code, "foo(1, 2)");
    }

    // --- Parse Derive Input Test ---

    #[test]
    fn test_parse_derive_input() {
        let tokens = TokenStream::from_tokens(vec![
            Token::new(TokenKind::Ident("struct".into())),
            Token::new(TokenKind::Ident("Point".into())),
            Token::new(TokenKind::OpenBrace),
            Token::new(TokenKind::Ident("x".into())),
            Token::new(TokenKind::Colon),
            Token::new(TokenKind::Ident("f64".into())),
            Token::new(TokenKind::Comma),
            Token::new(TokenKind::Ident("y".into())),
            Token::new(TokenKind::Colon),
            Token::new(TokenKind::Ident("f64".into())),
            Token::new(TokenKind::CloseBrace),
        ]);

        let input = parse_derive_input(&tokens).unwrap();
        assert_eq!(input.name, "Point");
        assert_eq!(input.kind, DeriveItemKind::Struct);
        assert_eq!(input.fields.len(), 2);
        assert_eq!(input.fields[0].name, "x");
        assert_eq!(input.fields[1].name, "y");
    }
}
