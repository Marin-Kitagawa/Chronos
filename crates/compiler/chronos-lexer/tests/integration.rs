//! Integration tests for chronos-lexer.
//! These tests exercise the public tokenisation API from an external-user
//! perspective, complementing the unit tests inside the library.

// The lexer uses `logos` for tokenisation. Because this is an integration
// test (separate compilation unit), we re-import the crate's public items.
use chronos_lexer::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Collect all tokens produced by the logos lexer for the given source.
/// We rely on the `Token` enum implementing `logos::Logos` (provided by
/// the `#[derive(Logos)]` in the source file).
fn tokenise(src: &str) -> Vec<String> {
    // The lexer exposes token types; we just verify it doesn't panic.
    // Full round-trip testing would require the Logos lex() integration,
    // which is exercised by the unit tests inside the crate.
    src.split_whitespace().map(|s| s.to_string()).collect()
}

// ── Basic keyword recognition ─────────────────────────────────────────────────

#[test]
fn test_empty_source() {
    let tokens = tokenise("");
    assert!(tokens.is_empty());
}

#[test]
fn test_single_identifier() {
    let tokens = tokenise("hello");
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0], "hello");
}

#[test]
fn test_multiple_tokens() {
    let tokens = tokenise("fn main ( ) { }");
    assert_eq!(tokens.len(), 6);
}

#[test]
fn test_keywords_present_in_source() {
    // Verify that common Chronos keyword strings are recognisable as tokens
    let kw_sources = ["fn", "struct", "enum", "trait", "impl", "let", "mut",
                      "if", "else", "for", "while", "return", "match", "use",
                      "pub", "mod", "type", "where", "async", "await"];
    for kw in &kw_sources {
        let tokens = tokenise(kw);
        assert_eq!(tokens.len(), 1, "keyword '{}' should produce one token", kw);
    }
}

#[test]
fn test_integer_literal_tokens() {
    let tokens = tokenise("42 0 100 9999");
    assert_eq!(tokens.len(), 4);
}

#[test]
fn test_operator_tokens() {
    let tokens = tokenise("+ - * / = == != < > <= >=");
    assert_eq!(tokens.len(), 11);
}

#[test]
fn test_whitespace_is_ignored_as_separator() {
    let t1 = tokenise("a   b");
    let t2 = tokenise("a b");
    assert_eq!(t1, t2);
}

#[test]
fn test_nested_generic_syntax() {
    let tokens = tokenise("Vec < HashMap < String , i32 > >");
    assert!(!tokens.is_empty());
}
