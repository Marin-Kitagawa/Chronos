//! Integration tests for chronos-docs.
use chronos_docs::*;

// ── Markdown rendering ────────────────────────────────────────────────────────

#[test]
fn test_heading_renders_to_h_tag() {
    let html = markdown_to_html("# Hello World");
    assert!(html.contains("<h1"), "expected h1 tag in: {}", html);
    assert!(html.contains("Hello World"));
    assert!(html.contains("</h1>"));
}

#[test]
fn test_bold_and_italic() {
    let html = markdown_to_html("This is **bold** and *italic* text.");
    assert!(html.contains("<strong>bold</strong>"));
    assert!(html.contains("<em>italic</em>"));
}

#[test]
fn test_inline_code() {
    let html = markdown_to_html("Use `cargo test` to run tests.");
    assert!(html.contains("<code>cargo test</code>"));
}

#[test]
fn test_unordered_list() {
    let md = "- item one\n- item two\n- item three";
    let html = markdown_to_html(md);
    assert!(html.contains("<ul>"));
    assert!(html.contains("<li>"));
}

#[test]
fn test_ordered_list() {
    let md = "1. first\n2. second\n3. third";
    let html = markdown_to_html(md);
    assert!(html.contains("<ol>"));
    assert!(html.contains("<li>"));
}

#[test]
fn test_link_rendering() {
    let html = markdown_to_html("Visit [Chronos](https://chronos-lang.org).");
    assert!(html.contains("<a href=\"https://chronos-lang.org\">Chronos</a>"));
}

// ── Doc comment parsing ───────────────────────────────────────────────────────

#[test]
fn test_parse_basic_doc_comment() {
    let comment = "/// Does something useful.\n/// @param x The input value.\n/// @returns The result.";
    let doc = parse_doc_comment(comment);
    assert!(!doc.summary.is_empty());
}

#[test]
fn test_parse_deprecated_tag() {
    let comment = "/// Old function.\n/// @deprecated Use new_fn instead.";
    let doc = parse_doc_comment(comment);
    assert!(doc.deprecated.is_some());
}

// ── Conventional commits ──────────────────────────────────────────────────────

#[test]
fn test_parse_feat_commit() {
    let commit = parse_commit("abc123", "feat(parser): add support for async closures");
    assert!(commit.is_some());
    let c = commit.unwrap();
    assert_eq!(c.commit_type, CommitType::Feat);
    assert_eq!(c.scope.as_deref(), Some("parser"));
    assert!(c.summary.contains("async closures"));
}

#[test]
fn test_parse_fix_commit_no_scope() {
    let commit = parse_commit("def456", "fix: resolve null pointer in type checker");
    assert!(commit.is_some());
    let c = commit.unwrap();
    assert_eq!(c.commit_type, CommitType::Fix);
    assert!(c.scope.is_none());
}

#[test]
fn test_parse_breaking_change() {
    let commit = parse_commit("ghi789", "feat!: redesign the module system");
    assert!(commit.is_some());
    let c = commit.unwrap();
    assert!(c.breaking);
}

#[test]
fn test_parse_invalid_commit_returns_none() {
    assert!(parse_commit("000", "not a conventional commit").is_none());
    assert!(parse_commit("000", "").is_none());
}

// ── Template engine ───────────────────────────────────────────────────────────

#[test]
fn test_template_variable_substitution() {
    let mut ctx = TemplateContext::new();
    ctx.set("name", TemplateValue::Str("Chronos".into()));
    let tmpl = Template::new("Hello, {{ name }}!");
    let rendered = tmpl.render(&ctx);
    assert_eq!(rendered, "Hello, Chronos!");
}

#[test]
fn test_template_if_true_branch() {
    let mut ctx = TemplateContext::new();
    ctx.set("show", TemplateValue::Bool(true));
    let tmpl = Template::new("{% if show %}visible{% endif %}");
    let rendered = tmpl.render(&ctx);
    assert_eq!(rendered.trim(), "visible");
}

#[test]
fn test_template_if_false_branch() {
    let mut ctx = TemplateContext::new();
    ctx.set("show", TemplateValue::Bool(false));
    let tmpl = Template::new("{% if show %}visible{% endif %}hidden");
    let rendered = tmpl.render(&ctx);
    assert!(rendered.contains("hidden"));
    assert!(!rendered.contains("visible"));
}

// ── Search index ──────────────────────────────────────────────────────────────

#[test]
fn test_search_finds_indexed_term() {
    let mut index = SearchIndex::new();
    index.add(SearchEntry {
        id:      0,
        kind:    "function".into(),
        name:    "foo".into(),
        summary: "computes the foo value from bar".into(),
        url:     "/docs/foo".into(),
    });
    index.add(SearchEntry {
        id:      1,
        kind:    "function".into(),
        name:    "bar".into(),
        summary: "the bar helper function".into(),
        url:     "/docs/bar".into(),
    });

    let results = index.search("foo");
    assert!(!results.is_empty());
    assert!(results[0].name.contains("foo"));
}

#[test]
fn test_search_returns_empty_for_unknown_term() {
    let index = SearchIndex::new();
    let results = index.search("nonexistent_term_xyz");
    assert!(results.is_empty());
}
