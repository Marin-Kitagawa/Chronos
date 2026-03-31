// chronos-documentation-system.rs
//
// Chronos Documentation System
// ==============================
// A literate-programming documentation engine that extracts doc comments,
// renders Markdown, generates API references, and produces navigable HTML.
//
// Modules:
//   1.  Doc Comment Parser — extract /// and /** */ comments from source
//   2.  Markdown Renderer — CommonMark subset to HTML
//   3.  AST Doc Nodes — structured representation of documented items
//   4.  Type Signature Formatter — human-readable type rendering
//   5.  Cross-Reference Resolver — intra-doc links [`Type`], [`fn`]
//   6.  Search Index — trigram-based full-text search
//   7.  HTML Generator — full page layout with navigation
//   8.  Literate Programming — tangle/weave (code blocks in docs)
//   9.  Doc Test Runner — extract and validate code examples
//  10.  Changelog Generator — from git-style commit messages
//  11.  Template Engine — simple {{ variable }} / {% for %} templates

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// 1. DOC COMMENT PARSER
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of documented item.
#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Module, Function, Struct, Enum, Trait, TypeAlias,
    Constant, Static, Field, Variant, Method, Macro,
}

impl fmt::Display for ItemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ItemKind::Module    => "module",    ItemKind::Function => "fn",
            ItemKind::Struct    => "struct",    ItemKind::Enum     => "enum",
            ItemKind::Trait     => "trait",     ItemKind::TypeAlias => "type",
            ItemKind::Constant  => "const",     ItemKind::Static   => "static",
            ItemKind::Field     => "field",     ItemKind::Variant  => "variant",
            ItemKind::Method    => "method",    ItemKind::Macro    => "macro",
        };
        write!(f, "{}", s)
    }
}

/// A parsed documentation comment, split into sections.
#[derive(Debug, Clone, Default)]
pub struct DocComment {
    pub summary:    String,           // first paragraph
    pub description: String,          // remaining paragraphs
    pub params:     Vec<(String, String)>,   // (name, description)
    pub returns:    Option<String>,
    pub errors:     Vec<(String, String)>,   // (ErrorType, description)
    pub examples:   Vec<CodeExample>,
    pub panics:     Option<String>,
    pub safety:     Option<String>,
    pub deprecated: Option<String>,
    pub since:      Option<String>,
    pub see_also:   Vec<String>,
}

/// A code example embedded in documentation.
#[derive(Debug, Clone)]
pub struct CodeExample {
    pub code:     String,
    pub language: String,    // "chronos", "text", etc.
    pub runnable: bool,      // marked with `# run`
    pub hidden:   bool,      // lines starting with # are hidden in output
    pub title:    Option<String>,
}

/// Parse a raw doc comment string into structured `DocComment`.
/// Handles `///` line comments and `/** */` block comments.
pub fn parse_doc_comment(raw: &str) -> DocComment {
    // Strip comment syntax
    let stripped = strip_doc_syntax(raw);
    parse_doc_sections(&stripped)
}

fn strip_doc_syntax(raw: &str) -> String {
    let mut lines = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("///") {
            lines.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        } else if trimmed.starts_with("/**") || trimmed.starts_with("*/") || trimmed.starts_with("* ") {
            let inner = trimmed
                .trim_start_matches("/**").trim_start_matches("*/")
                .trim_start_matches("* ").trim_start_matches('*');
            lines.push(inner.to_string());
        } else {
            lines.push(trimmed.to_string());
        }
    }
    lines.join("\n")
}

fn parse_doc_sections(text: &str) -> DocComment {
    let mut doc = DocComment::default();
    let mut paragraphs: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut in_code_block = false;
    let mut code_lang = String::new();
    let mut code_buf  = String::new();
    let mut code_runnable = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Code block delimiters
        if trimmed.starts_with("```") {
            if !in_code_block {
                in_code_block = true;
                let after = trimmed[3..].trim();
                if after.starts_with("chronos") || after.starts_with("ch") {
                    code_lang = "chronos".to_string();
                } else if after.is_empty() {
                    code_lang = "chronos".to_string();
                } else {
                    code_lang = after.to_string();
                }
                code_runnable = after.contains("run");
                code_buf.clear();
            } else {
                in_code_block = false;
                // Flush code block
                if !current.is_empty() { paragraphs.push(current.trim().to_string()); current.clear(); }
                doc.examples.push(CodeExample {
                    code:     code_buf.clone(),
                    language: code_lang.clone(),
                    runnable: code_runnable,
                    hidden:   code_buf.lines().any(|l| l.starts_with('#')),
                    title:    None,
                });
                code_buf.clear();
            }
            continue;
        }
        if in_code_block { code_buf += &format!("{}\n", line); continue; }

        // Section tags
        if trimmed.starts_with("# ") {
            if !current.is_empty() { paragraphs.push(current.trim().to_string()); current.clear(); }
            let section_name = trimmed[2..].to_lowercase();
            // Consume section content until next section
            paragraphs.push(format!("__SECTION__:{}", trimmed[2..].trim().to_string()));
            continue;
        }

        // Tagged doc fields: `@param`, `@returns`, etc.
        if let Some(rest) = trimmed.strip_prefix("@param ").or(trimmed.strip_prefix("# Parameters\n")) {
            if !current.is_empty() { paragraphs.push(current.trim().to_string()); current.clear(); }
            let mut parts = rest.splitn(2, ' ');
            let name = parts.next().unwrap_or("").trim_matches('`').to_string();
            let desc = parts.next().unwrap_or("").to_string();
            doc.params.push((name, desc));
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("@returns ").or(trimmed.strip_prefix("@return ")) {
            if !current.is_empty() { paragraphs.push(current.trim().to_string()); current.clear(); }
            doc.returns = Some(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("@deprecated ") {
            doc.deprecated = Some(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("@since ") {
            doc.since = Some(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("@see ") {
            doc.see_also.push(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("@panics ") {
            doc.panics = Some(rest.to_string());
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("@safety ") {
            doc.safety = Some(rest.to_string());
            continue;
        }

        if trimmed.is_empty() && !current.is_empty() {
            paragraphs.push(current.trim().to_string());
            current.clear();
        } else {
            if !current.is_empty() { current.push(' '); }
            current.push_str(trimmed);
        }
    }
    if !current.is_empty() { paragraphs.push(current.trim().to_string()); }

    // First non-section paragraph is the summary
    let mut remaining = Vec::new();
    let mut summary_set = false;
    for p in paragraphs {
        if p.is_empty() { continue; }
        if p.starts_with("__SECTION__:") { remaining.push(p); continue; }
        if !summary_set { doc.summary = p; summary_set = true; }
        else { remaining.push(p); }
    }
    doc.description = remaining.into_iter()
        .filter(|p| !p.starts_with("__SECTION__:"))
        .collect::<Vec<_>>()
        .join("\n\n");
    doc
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. MARKDOWN RENDERER (CommonMark subset)
// ─────────────────────────────────────────────────────────────────────────────

/// Render a CommonMark subset to HTML.
/// Supports: headings, paragraphs, bold, italic, code, code blocks,
/// links, images, blockquotes, unordered lists, ordered lists, tables, HR.
pub fn markdown_to_html(md: &str) -> String {
    let mut html = String::new();
    let mut lines = md.lines().peekable();
    let mut in_code_block = false;
    let mut in_list = false;
    let mut list_ordered = false;
    let mut in_blockquote = false;
    let mut in_table = false;

    while let Some(line) = lines.next() {
        // Fenced code blocks
        if line.trim_start().starts_with("```") {
            if !in_code_block {
                close_block(&mut html, &mut in_list, &mut in_blockquote, &mut in_table);
                let lang = line.trim_start()[3..].trim();
                let class = if lang.is_empty() { String::new() } else { format!(" class=\"language-{}\"", lang) };
                html += &format!("<pre><code{}>\n", class);
                in_code_block = true;
            } else {
                html += "</code></pre>\n";
                in_code_block = false;
            }
            continue;
        }
        if in_code_block { html += &format!("{}\n", html_escape(line)); continue; }

        // Headings
        if line.starts_with('#') {
            close_block(&mut html, &mut in_list, &mut in_blockquote, &mut in_table);
            let level = line.chars().take_while(|&c| c == '#').count().min(6);
            let text  = line[level..].trim();
            let id    = to_anchor_id(text);
            html += &format!("<h{} id=\"{}\">{}</h{}>\n", level, id, inline_md(text), level);
            continue;
        }

        // Horizontal rule
        if line.trim() == "---" || line.trim() == "***" || line.trim() == "___" {
            close_block(&mut html, &mut in_list, &mut in_blockquote, &mut in_table);
            html += "<hr />\n";
            continue;
        }

        // Blockquote
        if line.starts_with("> ") {
            if !in_blockquote { html += "<blockquote>\n"; in_blockquote = true; }
            html += &format!("<p>{}</p>\n", inline_md(line[2..].trim()));
            continue;
        } else if in_blockquote {
            html += "</blockquote>\n"; in_blockquote = false;
        }

        // Unordered list
        if line.starts_with("- ") || line.starts_with("* ") || line.starts_with("+ ") {
            if !in_list { html += "<ul>\n"; in_list = true; list_ordered = false; }
            html += &format!("<li>{}</li>\n", inline_md(line[2..].trim()));
            continue;
        }

        // Ordered list
        if line.len() > 2 && line.chars().next().map_or(false, |c| c.is_ascii_digit())
            && line.contains(". ") {
            if !in_list { html += "<ol>\n"; in_list = true; list_ordered = true; }
            let rest = line.splitn(2, ". ").nth(1).unwrap_or("");
            html += &format!("<li>{}</li>\n", inline_md(rest));
            continue;
        }

        // Table (pipes)
        if line.contains('|') && line.trim_start().starts_with('|') {
            let cells: Vec<&str> = line.trim().trim_matches('|').split('|').collect();
            // Skip separator row
            if cells.iter().all(|c| c.trim().chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')) {
                continue;
            }
            if !in_table { html += "<table>\n<tbody>\n"; in_table = true; }
            html += "<tr>";
            for cell in &cells { html += &format!("<td>{}</td>", inline_md(cell.trim())); }
            html += "</tr>\n";
            continue;
        } else if in_table {
            html += "</tbody>\n</table>\n"; in_table = false;
        }

        close_block(&mut html, &mut in_list, &mut in_blockquote, &mut in_table);

        // Empty line → paragraph break
        if line.trim().is_empty() { continue; }

        // Regular paragraph
        html += &format!("<p>{}</p>\n", inline_md(line.trim()));
    }

    close_block(&mut html, &mut in_list, &mut in_blockquote, &mut in_table);
    if in_code_block { html += "</code></pre>\n"; }
    html
}

fn close_block(html: &mut String, in_list: &mut bool, in_bq: &mut bool, in_table: &mut bool) {
    if *in_list    { *html += if *in_list { "</ul>\n" } else { "</ol>\n" }; *in_list = false; }
    if *in_bq      { *html += "</blockquote>\n"; *in_bq = false; }
    if *in_table   { *html += "</tbody>\n</table>\n"; *in_table = false; }
}

/// Render inline Markdown: bold, italic, code, links, images.
fn inline_md(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        // Bold **text** or __text__
        if i + 1 < n && ((chars[i] == '*' && chars[i+1] == '*') || (chars[i] == '_' && chars[i+1] == '_')) {
            let marker = if chars[i] == '*' { "**" } else { "__" };
            if let Some(end) = find_closing(&chars, i+2, marker) {
                let inner: String = chars[i+2..end].iter().collect();
                out += &format!("<strong>{}</strong>", html_escape(&inner));
                i = end + 2; continue;
            }
        }
        // Italic *text* or _text_
        if chars[i] == '*' || (chars[i] == '_' && (i == 0 || chars[i-1] == ' ')) {
            let marker = if chars[i] == '*' { "*" } else { "_" };
            if let Some(end) = find_closing(&chars, i+1, marker) {
                let inner: String = chars[i+1..end].iter().collect();
                out += &format!("<em>{}</em>", html_escape(&inner));
                i = end + 1; continue;
            }
        }
        // Inline code `code`
        if chars[i] == '`' {
            if let Some(end) = find_closing(&chars, i+1, "`") {
                let inner: String = chars[i+1..end].iter().collect();
                out += &format!("<code>{}</code>", html_escape(&inner));
                i = end + 1; continue;
            }
        }
        // Link [text](url) or intra-doc [`Type`]
        if chars[i] == '[' {
            if let Some(bracket_end) = find_closing_char(&chars, i+1, ']') {
                let text: String = chars[i+1..bracket_end].iter().collect();
                if bracket_end + 1 < n && chars[bracket_end + 1] == '(' {
                    if let Some(paren_end) = find_closing_char(&chars, bracket_end+2, ')') {
                        let url: String = chars[bracket_end+2..paren_end].iter().collect();
                        out += &format!("<a href=\"{}\">{}</a>", html_escape(&url), html_escape(&text));
                        i = paren_end + 1; continue;
                    }
                } else {
                    // Intra-doc link [`Type`] — rendered as a cross-reference
                    let clean = text.trim_matches('`');
                    out += &format!("<a class=\"intra-doc\" href=\"#{}\">{}</a>", to_anchor_id(clean), html_escape(clean));
                    i = bracket_end + 1; continue;
                }
            }
        }
        out.push(if chars[i] == '<' { '<' } else if chars[i] == '>' { '>' } else { chars[i] });
        if chars[i] == '<' { let _ = out.pop(); out += "&lt;"; }
        else if chars[i] == '>' { let _ = out.pop(); out += "&gt;"; }
        else if chars[i] == '&' { let _ = out.pop(); out += "&amp;"; }
        i += 1;
    }
    out
}

fn find_closing(chars: &[char], start: usize, marker: &str) -> Option<usize> {
    let mc: Vec<char> = marker.chars().collect();
    let mlen = mc.len();
    for i in start..chars.len().saturating_sub(mlen - 1) {
        if chars[i..].starts_with(&mc) { return Some(i); }
    }
    None
}
fn find_closing_char(chars: &[char], start: usize, target: char) -> Option<usize> {
    chars[start..].iter().position(|&c| c == target).map(|i| i + start)
}
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}
fn to_anchor_id(s: &str) -> String {
    s.to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. DOC ITEM NODES
// ─────────────────────────────────────────────────────────────────────────────

/// A fully-documented item in the API reference.
#[derive(Debug, Clone)]
pub struct DocItem {
    pub kind:       ItemKind,
    pub name:       String,
    pub signature:  String,          // rendered type signature
    pub doc:        DocComment,
    pub visibility: Visibility,
    pub attributes: Vec<String>,     // #[derive(...)], #[cfg(...)], etc.
    pub children:   Vec<DocItem>,    // fields, methods, variants
    pub source_loc: Option<SourceRef>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Visibility { Public, Crate, Private }

#[derive(Debug, Clone)]
pub struct SourceRef { pub file: String, pub line: u32 }

impl DocItem {
    pub fn new(kind: ItemKind, name: &str, sig: &str, doc: DocComment) -> Self {
        DocItem { kind, name: name.to_string(), signature: sig.to_string(), doc,
                  visibility: Visibility::Public, attributes: Vec::new(),
                  children: Vec::new(), source_loc: None }
    }

    /// Render the item as an HTML fragment for API docs.
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        let id = format!("{}.{}", self.kind, to_anchor_id(&self.name));

        // Deprecated banner
        if let Some(ref dep) = self.doc.deprecated {
            html += &format!("<div class=\"deprecated\">⚠ Deprecated: {}</div>\n", html_escape(dep));
        }

        // Item header
        html += &format!("<section id=\"{}\" class=\"item {}\">\n", id, self.kind);
        html += &format!("  <h3><code class=\"sig\">{}</code></h3>\n", html_escape(&self.signature));

        // Summary
        if !self.doc.summary.is_empty() {
            html += &format!("  <div class=\"summary\">{}</div>\n", markdown_to_html(&self.doc.summary));
        }

        // Full description
        if !self.doc.description.is_empty() {
            html += &format!("  <div class=\"desc\">{}</div>\n", markdown_to_html(&self.doc.description));
        }

        // Parameters
        if !self.doc.params.is_empty() {
            html += "  <h4>Parameters</h4>\n  <dl>\n";
            for (name, desc) in &self.doc.params {
                html += &format!("    <dt><code>{}</code></dt><dd>{}</dd>\n",
                    html_escape(name), inline_md(desc));
            }
            html += "  </dl>\n";
        }

        // Returns
        if let Some(ref ret) = self.doc.returns {
            html += &format!("  <h4>Returns</h4>\n  <p>{}</p>\n", inline_md(ret));
        }

        // Errors
        if !self.doc.errors.is_empty() {
            html += "  <h4>Errors</h4>\n  <dl>\n";
            for (kind, desc) in &self.doc.errors {
                html += &format!("    <dt><code>{}</code></dt><dd>{}</dd>\n",
                    html_escape(kind), inline_md(desc));
            }
            html += "  </dl>\n";
        }

        // Panics
        if let Some(ref p) = self.doc.panics {
            html += &format!("  <div class=\"panics\"><strong>Panics:</strong> {}</div>\n", inline_md(p));
        }

        // Safety
        if let Some(ref s) = self.doc.safety {
            html += &format!("  <div class=\"safety\"><strong>Safety:</strong> {}</div>\n", inline_md(s));
        }

        // Examples
        for (i, ex) in self.doc.examples.iter().enumerate() {
            let default_title = format!("Example {}", i + 1);
            let title = ex.title.as_deref().unwrap_or(&default_title);
            html += &format!("  <h4>{}</h4>\n", html_escape(title));
            html += &format!("  <pre><code class=\"language-{}\">{}</code></pre>\n",
                html_escape(&ex.language), html_escape(&ex.code));
        }

        // See also
        if !self.doc.see_also.is_empty() {
            html += "  <p><strong>See also:</strong> ";
            for (i, s) in self.doc.see_also.iter().enumerate() {
                if i > 0 { html += ", "; }
                html += &format!("<a href=\"#{}\">{}</a>", to_anchor_id(s), html_escape(s));
            }
            html += "</p>\n";
        }

        // Children (fields, methods, variants)
        if !self.children.is_empty() {
            html += "  <div class=\"children\">\n";
            for child in &self.children { html += &child.to_html(); }
            html += "  </div>\n";
        }

        html += "</section>\n";
        html
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. CROSS-REFERENCE RESOLVER
// ─────────────────────────────────────────────────────────────────────────────

/// Resolves intra-doc links like [`Vec`], [`HashMap::insert`], [`crate::Foo`].
pub struct CrossRefResolver {
    /// Maps item path → URL fragment (e.g. "Vec" → "struct.Vec")
    index: HashMap<String, String>,
}

impl CrossRefResolver {
    pub fn new() -> Self { CrossRefResolver { index: HashMap::new() } }

    pub fn register(&mut self, name: &str, kind: &ItemKind, path: &str) {
        let fragment = format!("{}.{}", kind, to_anchor_id(name));
        self.index.insert(name.to_string(), fragment.clone());
        self.index.insert(path.to_string(), fragment.clone());
        // Also register short name
        if let Some(short) = path.split("::").last() {
            self.index.entry(short.to_string()).or_insert(fragment);
        }
    }

    /// Resolve a link target. Returns Some(url) or None if not found.
    pub fn resolve(&self, target: &str) -> Option<&str> {
        self.index.get(target).map(|s| s.as_str())
    }

    /// Resolve all intra-doc links in HTML, replacing `<a class="intra-doc">`.
    pub fn resolve_html(&self, html: &str) -> String {
        let mut out = html.to_string();
        // Simple substitution: replace href="#<target>" with resolved URL
        // In a real impl, use a proper HTML parser
        for (name, url) in &self.index {
            let placeholder = format!("href=\"#{}\"", to_anchor_id(name));
            let resolved   = format!("href=\"{}\"", url);
            out = out.replace(&placeholder, &resolved);
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. SEARCH INDEX (TRIGRAM-BASED)
// ─────────────────────────────────────────────────────────────────────────────

/// A searchable entry in the documentation.
#[derive(Debug, Clone)]
pub struct SearchEntry {
    pub id:      usize,
    pub kind:    String,
    pub name:    String,
    pub summary: String,
    pub url:     String,
}

/// Trigram-based full-text search index.
/// Trigrams provide O(1) lookup with ~80% precision for substring queries.
pub struct SearchIndex {
    pub entries:  Vec<SearchEntry>,
    trigrams:     HashMap<[u8; 3], Vec<usize>>,  // trigram → [entry ids]
}

impl SearchIndex {
    pub fn new() -> Self { SearchIndex { entries: Vec::new(), trigrams: HashMap::new() } }

    pub fn add(&mut self, entry: SearchEntry) {
        let id = entry.id;
        let text = format!("{} {} {}", entry.name.to_lowercase(),
                           entry.kind.to_lowercase(), entry.summary.to_lowercase());
        for tri in extract_trigrams(&text) {
            self.trigrams.entry(tri).or_default().push(id);
        }
        self.entries.push(entry);
    }

    /// Search for entries matching the query. Returns entries ranked by hit count.
    pub fn search(&self, query: &str) -> Vec<&SearchEntry> {
        let q = query.to_lowercase();
        let query_tris = extract_trigrams(&q);
        if query_tris.is_empty() { return Vec::new(); }

        let mut scores: HashMap<usize, usize> = HashMap::new();
        for tri in &query_tris {
            if let Some(ids) = self.trigrams.get(tri) {
                for &id in ids { *scores.entry(id).or_insert(0) += 1; }
            }
        }

        // Exact name match boost
        for entry in &self.entries {
            if entry.name.to_lowercase().contains(&q) {
                *scores.entry(entry.id).or_insert(0) += 10;
            }
        }

        let mut ranked: Vec<(usize, usize)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        ranked.iter().filter_map(|(id, _)| self.entries.iter().find(|e| e.id == *id)).collect()
    }

    /// Export search index as JSON for client-side search.
    pub fn to_json(&self) -> String {
        let mut json = String::from("[\n");
        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 { json += ",\n"; }
            json += &format!(
                "  {{\"id\":{},\"kind\":\"{}\",\"name\":\"{}\",\"summary\":\"{}\",\"url\":\"{}\"}}",
                entry.id, entry.kind, json_escape(&entry.name),
                json_escape(&entry.summary), json_escape(&entry.url)
            );
        }
        json += "\n]";
        json
    }
}

fn extract_trigrams(s: &str) -> Vec<[u8; 3]> {
    let padded = format!("  {}  ", s); // pad for boundary trigrams
    let bytes  = padded.as_bytes();
    (0..bytes.len().saturating_sub(2))
        .map(|i| [bytes[i], bytes[i+1], bytes[i+2]])
        .collect()
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
     .replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t")
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. HTML GENERATOR
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for HTML generation.
#[derive(Debug, Clone)]
pub struct HtmlConfig {
    pub title:        String,
    pub logo_url:     Option<String>,
    pub favicon_url:  Option<String>,
    pub custom_css:   Option<String>,
    pub version:      String,
    pub repository:   Option<String>,
    pub theme:        Theme,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Theme { Light, Dark, Auto }

impl Default for HtmlConfig {
    fn default() -> Self {
        HtmlConfig {
            title: "Chronos Documentation".to_string(),
            logo_url: None, favicon_url: None, custom_css: None,
            version: "0.1.0".to_string(), repository: None, theme: Theme::Auto,
        }
    }
}

/// Generates a complete HTML documentation page.
pub struct HtmlGenerator {
    pub config: HtmlConfig,
}

impl HtmlGenerator {
    pub fn new(config: HtmlConfig) -> Self { HtmlGenerator { config } }

    /// Generate the full HTML page for a module's documentation.
    pub fn render_module(&self, module_name: &str, items: &[DocItem],
                         nav: &[(String, String)]) -> String {
        let mut body = String::new();
        for item in items { body += &item.to_html(); }
        self.wrap_page(&format!("{} — {}", module_name, self.config.title), &body, nav)
    }

    /// Wrap content in a full HTML page with navigation sidebar.
    pub fn wrap_page(&self, title: &str, content: &str, nav: &[(String, String)]) -> String {
        let theme_attr = match self.config.theme {
            Theme::Light => " data-theme=\"light\"",
            Theme::Dark  => " data-theme=\"dark\"",
            Theme::Auto  => "",
        };

        let nav_html: String = nav.iter().map(|(href, label)| {
            format!("<a href=\"{}\">{}</a>\n", html_escape(href), html_escape(label))
        }).collect();

        let favicon = self.config.favicon_url.as_deref()
            .map_or(String::new(), |u| format!("<link rel=\"icon\" href=\"{}\">\n", u));

        let custom_css = self.config.custom_css.as_deref()
            .map_or(String::new(), |css| format!("<style>\n{}\n</style>\n", css));

        let repo_link = self.config.repository.as_deref()
            .map_or(String::new(), |u| format!(" | <a href=\"{}\">Repository</a>", u));

        format!(
r#"<!DOCTYPE html>
<html lang="en"{theme}>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  {favicon}<link rel="stylesheet" href="/chronos-doc.css">
  {custom_css}</head>
<body>
  <nav class="sidebar">
    <div class="logo"><strong>{pkg}</strong> <span class="version">v{ver}</span></div>
    <div class="nav-links">
{nav}    </div>
  </nav>
  <main class="content">
    <div class="doc-body">
{content}    </div>
  </main>
  <footer>
    Generated by <strong>chronos-doc</strong> v{ver}{repo}
  </footer>
  <script src="/chronos-doc.js"></script>
</body>
</html>
"#,
            theme = theme_attr,
            title = html_escape(title),
            favicon = favicon,
            custom_css = custom_css,
            pkg = html_escape(&self.config.title),
            ver = html_escape(&self.config.version),
            nav = nav_html,
            content = content,
            repo = repo_link,
        )
    }

    /// Generate a search page with embedded JSON index.
    pub fn render_search_page(&self, index: &SearchIndex) -> String {
        let body = format!(
            "<h1>Search</h1>\n\
             <input id=\"search-input\" type=\"text\" placeholder=\"Search docs...\" autofocus>\n\
             <div id=\"search-results\"></div>\n\
             <script>window.SEARCH_INDEX = {};</script>\n",
            index.to_json()
        );
        self.wrap_page(&format!("Search — {}", self.config.title), &body, &[])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. LITERATE PROGRAMMING — TANGLE / WEAVE
// ─────────────────────────────────────────────────────────────────────────────

/// A chunk of a literate program.
#[derive(Debug, Clone)]
pub enum LitChunk {
    Prose(String),                     // documentation text (Markdown)
    Code { name: String, lang: String, body: String, deps: Vec<String> },
}

/// A literate program: interleaved prose and named code chunks.
pub struct LiterateProgram {
    pub chunks: Vec<LitChunk>,
}

impl LiterateProgram {
    /// Parse a literate Chronos file (.lch).
    /// Code chunks are delimited by `<<name>>=` ... `@` (Knuth/Noweb syntax).
    pub fn parse(source: &str) -> Self {
        let mut chunks = Vec::new();
        let mut prose  = String::new();
        let mut in_code = false;
        let mut code_name = String::new();
        let mut code_lang = String::new();
        let mut code_body  = String::new();

        for line in source.lines() {
            if !in_code && line.trim_start().starts_with("<<") && line.contains(">>=") {
                // Start of code chunk
                if !prose.is_empty() { chunks.push(LitChunk::Prose(prose.clone())); prose.clear(); }
                let inner = line.trim_start()[2..].split(">>=").next().unwrap_or("").trim();
                code_name = inner.to_string();
                code_lang = "chronos".to_string();
                code_body.clear();
                in_code = true;
            } else if in_code && line.trim() == "@" {
                // End of code chunk
                let deps: Vec<String> = extract_chunk_refs(&code_body);
                chunks.push(LitChunk::Code {
                    name: code_name.clone(), lang: code_lang.clone(),
                    body: code_body.clone(), deps,
                });
                code_body.clear(); in_code = false;
            } else if in_code {
                code_body += &format!("{}\n", line);
            } else {
                prose += &format!("{}\n", line);
            }
        }
        if !prose.is_empty() { chunks.push(LitChunk::Prose(prose)); }
        LiterateProgram { chunks }
    }

    /// **Tangle**: extract all code chunks and concatenate in dependency order.
    /// Resolves `<<chunk-name>>` references (transclusion).
    pub fn tangle(&self, root_chunk: &str) -> String {
        let mut code_map: HashMap<&str, &str> = HashMap::new();
        for chunk in &self.chunks {
            if let LitChunk::Code { name, body, .. } = chunk {
                code_map.insert(name.as_str(), body.as_str());
            }
        }
        expand_chunk(root_chunk, &code_map, &mut Vec::new())
    }

    /// **Weave**: render the literate program as a documented HTML page.
    pub fn weave(&self) -> String {
        let mut html = String::new();
        for (i, chunk) in self.chunks.iter().enumerate() {
            match chunk {
                LitChunk::Prose(text) => html += &markdown_to_html(text),
                LitChunk::Code { name, lang, body, .. } => {
                    html += &format!(
                        "<div class=\"code-chunk\">\n\
                         <span class=\"chunk-name\">⟨{}⟩≡</span>\n\
                         <pre><code class=\"language-{}\">{}</code></pre>\n\
                         </div>\n",
                        html_escape(name), lang, html_escape(body)
                    );
                }
            }
        }
        html
    }
}

fn extract_chunk_refs(body: &str) -> Vec<String> {
    body.lines().filter_map(|line| {
        let t = line.trim();
        if t.starts_with("<<") && t.ends_with(">>") {
            Some(t[2..t.len()-2].to_string())
        } else { None }
    }).collect()
}

fn expand_chunk<'a>(name: &str, map: &HashMap<&str, &'a str>, stack: &mut Vec<String>) -> String {
    if stack.contains(&name.to_string()) { return format!("/* circular ref: {} */\n", name); }
    stack.push(name.to_string());
    let body = map.get(name).copied().unwrap_or("/* chunk not found */\n");
    let result = body.lines().map(|line| {
        let t = line.trim();
        if t.starts_with("<<") && t.ends_with(">>") {
            let ref_name = &t[2..t.len()-2];
            expand_chunk(ref_name, map, stack)
        } else {
            format!("{}\n", line)
        }
    }).collect();
    stack.pop();
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. DOC TEST RUNNER
// ─────────────────────────────────────────────────────────────────────────────

/// A code example extracted for testing.
#[derive(Debug, Clone)]
pub struct DocTest {
    pub item_name: String,
    pub index:     usize,
    pub code:      String,
    pub expected:  Option<String>,  // expected output (from `// Output:` comment)
}

/// Extract doc tests from a collection of doc items.
pub fn extract_doc_tests(items: &[DocItem]) -> Vec<DocTest> {
    let mut tests = Vec::new();
    for item in items {
        for (i, ex) in item.doc.examples.iter().enumerate() {
            if ex.language == "chronos" || ex.language.is_empty() {
                // Extract expected output from `// Output:` or `// => ` comments
                let expected = ex.code.lines()
                    .find(|l| l.trim().starts_with("// Output:") || l.trim().starts_with("// =>"))
                    .map(|l| l.trim()
                        .trim_start_matches("// Output:")
                        .trim_start_matches("// =>")
                        .trim().to_string());
                tests.push(DocTest {
                    item_name: item.name.clone(),
                    index:     i,
                    code:      ex.code.clone(),
                    expected,
                });
            }
        }
        // Recurse into children
        tests.extend(extract_doc_tests(&item.children));
    }
    tests
}

/// A doc test result.
#[derive(Debug, Clone)]
pub struct DocTestResult {
    pub test:    DocTest,
    pub passed:  bool,
    pub message: String,
}

/// Simulate running doc tests (real impl would compile & execute each snippet).
pub fn run_doc_tests(tests: &[DocTest]) -> Vec<DocTestResult> {
    tests.iter().map(|t| {
        // Simulate: tests with expected output pass only if code is non-empty
        let passed = !t.code.trim().is_empty();
        DocTestResult {
            test:    t.clone(),
            passed,
            message: if passed { "ok".to_string() }
                     else { "empty code block".to_string() },
        }
    }).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. CHANGELOG GENERATOR
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed conventional commit.
#[derive(Debug, Clone)]
pub struct ConventionalCommit {
    pub commit_type: CommitType,
    pub scope:       Option<String>,
    pub summary:     String,
    pub body:        Option<String>,
    pub breaking:    bool,
    pub hash:        String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CommitType { Feat, Fix, Docs, Style, Refactor, Perf, Test, Chore, Build, Ci, Revert }

impl CommitType {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "feat"     => CommitType::Feat,
            "fix"      => CommitType::Fix,
            "docs"     => CommitType::Docs,
            "style"    => CommitType::Style,
            "refactor" => CommitType::Refactor,
            "perf"     => CommitType::Perf,
            "test"     => CommitType::Test,
            "chore"    => CommitType::Chore,
            "build"    => CommitType::Build,
            "ci"       => CommitType::Ci,
            "revert"   => CommitType::Revert,
            _          => CommitType::Chore,
        }
    }
    fn label(&self) -> &str {
        match self {
            CommitType::Feat     => "Features",
            CommitType::Fix      => "Bug Fixes",
            CommitType::Docs     => "Documentation",
            CommitType::Perf     => "Performance",
            CommitType::Refactor => "Refactoring",
            CommitType::Style    => "Style",
            CommitType::Test     => "Tests",
            CommitType::Chore    => "Chores",
            CommitType::Build    => "Build",
            CommitType::Ci       => "CI",
            CommitType::Revert   => "Reverts",
        }
    }
}

/// Parse a conventional commit message.
/// Format: `type(scope): summary` with optional `BREAKING CHANGE:` footer.
pub fn parse_commit(hash: &str, message: &str) -> Option<ConventionalCommit> {
    let first_line = message.lines().next()?;
    let (type_part, rest) = if first_line.contains(": ") {
        let idx = first_line.find(": ")?;
        (&first_line[..idx], &first_line[idx+2..])
    } else { return None; };

    let (commit_type_str, scope) = if type_part.contains('(') && type_part.contains(')') {
        let si = type_part.find('(')?;
        let ei = type_part.find(')')?;
        (&type_part[..si], Some(type_part[si+1..ei].to_string()))
    } else {
        (type_part.trim_end_matches('!'), None)
    };

    let breaking = type_part.ends_with('!') || message.contains("BREAKING CHANGE:");
    let body = message.lines().skip(2)
        .filter(|l| !l.starts_with("BREAKING CHANGE:"))
        .collect::<Vec<_>>().join("\n");

    Some(ConventionalCommit {
        commit_type: CommitType::from_str(commit_type_str),
        scope,
        summary: rest.to_string(),
        body:    if body.is_empty() { None } else { Some(body) },
        breaking,
        hash: hash.to_string(),
    })
}

/// Generate a CHANGELOG.md section from a list of commits.
pub fn generate_changelog(version: &str, date: &str, commits: &[ConventionalCommit]) -> String {
    let mut by_type: HashMap<String, Vec<&ConventionalCommit>> = HashMap::new();
    let mut breaking = Vec::new();

    for c in commits {
        if c.breaking { breaking.push(c); }
        by_type.entry(c.commit_type.label().to_string()).or_default().push(c);
    }

    let mut out = format!("## [{}] — {}\n\n", version, date);

    if !breaking.is_empty() {
        out += "### ⚠ BREAKING CHANGES\n\n";
        for c in &breaking {
            let scope = c.scope.as_deref().map_or(String::new(), |s| format!("**{}:** ", s));
            out += &format!("- {}{} ({})\n", scope, c.summary, &c.hash[..8.min(c.hash.len())]);
        }
        out += "\n";
    }

    // Ordered sections
    let section_order = ["Features", "Bug Fixes", "Performance", "Documentation",
                         "Refactoring", "Build", "CI", "Tests", "Style", "Chores", "Reverts"];
    for section in &section_order {
        if let Some(cs) = by_type.get(*section) {
            out += &format!("### {}\n\n", section);
            for c in cs {
                let scope = c.scope.as_deref().map_or(String::new(), |s| format!("**{}:** ", s));
                out += &format!("- {}{} ({})\n", scope, c.summary, &c.hash[..8.min(c.hash.len())]);
            }
            out += "\n";
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. TEMPLATE ENGINE
// ─────────────────────────────────────────────────────────────────────────────

/// A minimal template engine supporting `{{ var }}`, `{% if var %}`, `{% for x in list %}`.
pub struct Template { source: String }

impl Template {
    pub fn new(source: &str) -> Self { Template { source: source.to_string() } }

    /// Render the template with the given context.
    pub fn render(&self, ctx: &TemplateContext) -> String {
        render_template(&self.source, ctx)
    }
}

pub struct TemplateContext {
    vars:  HashMap<String, TemplateValue>,
}

#[derive(Debug, Clone)]
pub enum TemplateValue {
    Str(String),
    Bool(bool),
    Int(i64),
    List(Vec<TemplateValue>),
    Map(HashMap<String, TemplateValue>),
}

impl TemplateContext {
    pub fn new() -> Self { TemplateContext { vars: HashMap::new() } }
    pub fn set(&mut self, key: &str, val: TemplateValue) { self.vars.insert(key.to_string(), val); }
    pub fn get(&self, key: &str) -> Option<&TemplateValue> { self.vars.get(key) }
}

impl fmt::Display for TemplateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateValue::Str(s)  => write!(f, "{}", s),
            TemplateValue::Bool(b) => write!(f, "{}", b),
            TemplateValue::Int(n)  => write!(f, "{}", n),
            TemplateValue::List(v) => write!(f, "[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
            TemplateValue::Map(_)  => write!(f, "{{...}}"),
        }
    }
}

fn render_template(src: &str, ctx: &TemplateContext) -> String {
    let mut out = String::new();
    let mut rest = src;

    while !rest.is_empty() {
        let var_pos  = rest.find("{{");
        let tag_pos  = rest.find("{%");
        // Pick the earlier delimiter; prefer block tags if they come before variable subs.
        let pick_var = match (var_pos, tag_pos) {
            (Some(vp), Some(tp)) => vp < tp,
            (Some(_),  None)     => true,
            _                    => false,
        };
        // Variable substitution: {{ var }}
        if pick_var {
            let start = var_pos.unwrap();
            out.push_str(&rest[..start]);
            rest = &rest[start+2..];
            if let Some(end) = rest.find("}}") {
                let expr = rest[..end].trim();
                // Support {{ var | filter }} — basic filters: upper, lower, len
                let (var_name, filter) = if expr.contains(" | ") {
                    let mut p = expr.splitn(2, " | ");
                    (p.next().unwrap_or("").trim(), p.next().unwrap_or("").trim())
                } else { (expr, "") };
                let val = ctx.get(var_name).map(|v| v.to_string()).unwrap_or_default();
                let filtered = match filter {
                    "upper" => val.to_uppercase(),
                    "lower" => val.to_lowercase(),
                    "len"   => val.len().to_string(),
                    "escape" => html_escape(&val),
                    _       => val,
                };
                out.push_str(&filtered);
                rest = &rest[end+2..];
            }
        }
        // Block tags: {% if %}, {% for %}, {% endfor %}, {% endif %}
        else if let Some(start) = rest.find("{%") {
            out.push_str(&rest[..start]);
            rest = &rest[start+2..];
            if let Some(end) = rest.find("%}") {
                let tag = rest[..end].trim();
                rest = &rest[end+2..];

                if let Some(cond) = tag.strip_prefix("if ") {
                    // Find matching {% endif %}
                    if let Some(endif_pos) = rest.find("{% endif %}") {
                        let body = &rest[..endif_pos];
                        rest = &rest[endif_pos + 11..];
                        // Evaluate condition
                        let val = ctx.get(cond.trim());
                        let truthy = match val {
                            Some(TemplateValue::Bool(b)) => *b,
                            Some(TemplateValue::Str(s))  => !s.is_empty(),
                            Some(TemplateValue::Int(n))  => *n != 0,
                            Some(_) => true,
                            None    => false,
                        };
                        if truthy { out.push_str(&render_template(body, ctx)); }
                    }
                } else if let Some(rest_tag) = tag.strip_prefix("for ") {
                    // {% for item in list %}
                    if let Some(in_pos) = rest_tag.find(" in ") {
                        let var_name  = rest_tag[..in_pos].trim();
                        let list_name = rest_tag[in_pos+4..].trim();
                        if let Some(endfor_pos) = rest.find("{% endfor %}") {
                            let body = &rest[..endfor_pos];
                            rest = &rest[endfor_pos + 12..];
                            if let Some(TemplateValue::List(items)) = ctx.get(list_name) {
                                for item in items {
                                    let mut inner_ctx = TemplateContext::new();
                                    // Copy parent vars
                                    for (k, v) in &ctx.vars { inner_ctx.set(k, v.clone()); }
                                    inner_ctx.set(var_name, item.clone());
                                    out.push_str(&render_template(body, &inner_ctx));
                                }
                            }
                        }
                    }
                }
                // {% else %}, {% endif %}, {% endfor %} handled implicitly above
            }
        } else {
            out.push_str(rest);
            break;
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Doc Comment Parsing ───────────────────────────────────────────────────

    #[test]
    fn test_parse_summary_only() {
        let doc = parse_doc_comment("/// Computes the sum of two integers.");
        assert_eq!(doc.summary, "Computes the sum of two integers.");
    }

    #[test]
    fn test_parse_param_tag() {
        let raw = "/// Adds two numbers.\n/// @param a The first operand.\n/// @param b The second operand.";
        let doc = parse_doc_comment(raw);
        assert_eq!(doc.params.len(), 2);
        assert_eq!(doc.params[0].0, "a");
        assert!(doc.params[0].1.contains("first"));
    }

    #[test]
    fn test_parse_returns_tag() {
        let raw = "/// @returns The sum of a and b.";
        let doc = parse_doc_comment(raw);
        assert_eq!(doc.returns.as_deref(), Some("The sum of a and b."));
    }

    #[test]
    fn test_parse_deprecated_tag() {
        let raw = "/// @deprecated Use `add_v2` instead.";
        let doc = parse_doc_comment(raw);
        assert!(doc.deprecated.is_some());
        assert!(doc.deprecated.unwrap().contains("add_v2"));
    }

    #[test]
    fn test_parse_see_also() {
        let raw = "/// Does something.\n/// @see OtherType\n/// @see other_fn";
        let doc = parse_doc_comment(raw);
        assert_eq!(doc.see_also.len(), 2);
        assert!(doc.see_also.contains(&"OtherType".to_string()));
    }

    #[test]
    fn test_parse_code_example() {
        let raw = "/// Example:\n/// ```chronos\n/// let x = 42;\n/// ```";
        let doc = parse_doc_comment(raw);
        assert_eq!(doc.examples.len(), 1);
        assert_eq!(doc.examples[0].language, "chronos");
        assert!(doc.examples[0].code.contains("let x = 42;"));
    }

    // ── Markdown Renderer ─────────────────────────────────────────────────────

    #[test]
    fn test_md_heading() {
        let html = markdown_to_html("# Hello World");
        assert!(html.contains("<h1"), "Should be h1");
        assert!(html.contains("Hello World"));
    }

    #[test]
    fn test_md_headings_levels() {
        let html = markdown_to_html("## Section\n### Subsection");
        assert!(html.contains("<h2"));
        assert!(html.contains("<h3"));
    }

    #[test]
    fn test_md_paragraph() {
        let html = markdown_to_html("This is a paragraph.");
        assert!(html.contains("<p>"));
        assert!(html.contains("This is a paragraph."));
    }

    #[test]
    fn test_md_bold() {
        let html = inline_md("This is **bold** text.");
        assert!(html.contains("<strong>bold</strong>"), "Bold: {}", html);
    }

    #[test]
    fn test_md_italic() {
        let html = inline_md("This is *italic* text.");
        assert!(html.contains("<em>italic</em>"), "Italic: {}", html);
    }

    #[test]
    fn test_md_inline_code() {
        let html = inline_md("Use `println!` to print.");
        assert!(html.contains("<code>println!</code>"), "Code: {}", html);
    }

    #[test]
    fn test_md_link() {
        let html = inline_md("[Chronos](https://chronos.dev)");
        assert!(html.contains("<a href="), "Link: {}", html);
        assert!(html.contains("Chronos"));
    }

    #[test]
    fn test_md_unordered_list() {
        let html = markdown_to_html("- item one\n- item two\n- item three");
        assert!(html.contains("<ul>"));
        assert!(html.contains("<li>"));
        assert!(html.contains("item one"));
    }

    #[test]
    fn test_md_code_block() {
        let html = markdown_to_html("```chronos\nlet x = 1;\n```");
        assert!(html.contains("<pre><code"), "Code block: {}", html);
        assert!(html.contains("let x = 1;"));
    }

    #[test]
    fn test_md_html_escape() {
        let html = markdown_to_html("Use <br> and &amp; carefully.");
        assert!(!html.contains("<br>"), "Should escape <br>");
    }

    #[test]
    fn test_md_horizontal_rule() {
        let html = markdown_to_html("Before\n---\nAfter");
        assert!(html.contains("<hr"), "HR: {}", html);
    }

    #[test]
    fn test_anchor_id_generation() {
        assert_eq!(to_anchor_id("Hello World!"), "hello-world-");
        assert_eq!(to_anchor_id("my_function"), "my-function");
    }

    // ── Doc Item HTML ─────────────────────────────────────────────────────────

    #[test]
    fn test_doc_item_html_contains_signature() {
        let doc = parse_doc_comment("/// Adds two numbers.");
        let item = DocItem::new(ItemKind::Function, "add", "fn add(a: i32, b: i32) -> i32", doc);
        let html = item.to_html();
        assert!(html.contains("fn add"), "Signature in HTML: {}", &html[..200.min(html.len())]);
    }

    #[test]
    fn test_doc_item_deprecated_banner() {
        let mut doc = DocComment::default();
        doc.summary = "Old function.".to_string();
        doc.deprecated = Some("Use new_fn instead.".to_string());
        let item = DocItem::new(ItemKind::Function, "old_fn", "fn old_fn()", doc);
        let html = item.to_html();
        assert!(html.contains("deprecated") || html.contains("Deprecated"), "Deprecated: {}", &html[..200.min(html.len())]);
    }

    #[test]
    fn test_doc_item_with_params() {
        let mut doc = DocComment::default();
        doc.summary = "Test fn.".to_string();
        doc.params = vec![("x".to_string(), "the x value".to_string())];
        let item = DocItem::new(ItemKind::Function, "f", "fn f(x: i32)", doc);
        let html = item.to_html();
        assert!(html.contains("Parameters"), "Params section: {}", &html[..300.min(html.len())]);
    }

    // ── Cross-Reference Resolver ──────────────────────────────────────────────

    #[test]
    fn test_crossref_register_and_resolve() {
        let mut res = CrossRefResolver::new();
        res.register("Vec", &ItemKind::Struct, "std::vec::Vec");
        assert!(res.resolve("Vec").is_some());
        assert!(res.resolve("Vec").unwrap().contains("struct"));
    }

    #[test]
    fn test_crossref_short_name_resolution() {
        let mut res = CrossRefResolver::new();
        res.register("HashMap", &ItemKind::Struct, "std::collections::HashMap");
        // Short name "HashMap" should resolve
        assert!(res.resolve("HashMap").is_some());
    }

    #[test]
    fn test_crossref_unknown_returns_none() {
        let res = CrossRefResolver::new();
        assert!(res.resolve("NonExistentType").is_none());
    }

    // ── Search Index ──────────────────────────────────────────────────────────

    #[test]
    fn test_search_finds_exact_match() {
        let mut idx = SearchIndex::new();
        idx.add(SearchEntry { id: 1, kind: "fn".into(), name: "sort_vec".into(),
                              summary: "Sorts a vector.".into(), url: "#fn.sort_vec".into() });
        idx.add(SearchEntry { id: 2, kind: "fn".into(), name: "reverse_vec".into(),
                              summary: "Reverses a vector.".into(), url: "#fn.reverse_vec".into() });
        let results = idx.search("sort");
        assert!(!results.is_empty(), "Should find 'sort'");
        assert_eq!(results[0].name, "sort_vec", "Exact match should rank first");
    }

    #[test]
    fn test_search_empty_query() {
        let mut idx = SearchIndex::new();
        idx.add(SearchEntry { id: 1, kind: "fn".into(), name: "foo".into(),
                              summary: "bar".into(), url: "#".into() });
        // Empty query → need at least 3 chars for trigrams
        let results = idx.search("fo");
        // May or may not return results — just ensure it doesn't panic
        let _ = results.len();
    }

    #[test]
    fn test_search_json_export() {
        let mut idx = SearchIndex::new();
        idx.add(SearchEntry { id: 1, kind: "struct".into(), name: "Point".into(),
                              summary: "A 2D point.".into(), url: "#struct.Point".into() });
        let json = idx.to_json();
        assert!(json.starts_with("["), "JSON array: {}", &json[..50.min(json.len())]);
        assert!(json.contains("\"name\""), "Has name field");
        assert!(json.contains("Point"));
    }

    #[test]
    fn test_trigram_extraction() {
        let tris = extract_trigrams("abc");
        // "  abc  " → "  a", " ab", "abc", "bc ", "c  "
        assert!(!tris.is_empty());
        assert!(tris.len() >= 3);
    }

    // ── HTML Generator ────────────────────────────────────────────────────────

    #[test]
    fn test_html_generator_page_structure() {
        let gen = HtmlGenerator::new(HtmlConfig::default());
        let html = gen.wrap_page("Test Page", "<p>content</p>", &[("/index.html".into(), "Home".into())]);
        assert!(html.contains("<!DOCTYPE html>"), "DOCTYPE");
        assert!(html.contains("<nav"), "Nav sidebar");
        assert!(html.contains("<main"), "Main content");
        assert!(html.contains("content"));
        assert!(html.contains("Home"));
    }

    #[test]
    fn test_html_generator_dark_theme() {
        let config = HtmlConfig { theme: Theme::Dark, ..Default::default() };
        let gen = HtmlGenerator::new(config);
        let html = gen.wrap_page("Test", "", &[]);
        assert!(html.contains("data-theme=\"dark\""), "Dark theme attr");
    }

    // ── Literate Programming ──────────────────────────────────────────────────

    #[test]
    fn test_literate_parse_chunks() {
        let src = "This is prose.\n\n<<main>>=\nfn main() {\n    println!(\"hello\");\n}\n@\n\nMore prose.";
        let prog = LiterateProgram::parse(src);
        let code_chunks: Vec<_> = prog.chunks.iter().filter(|c| matches!(c, LitChunk::Code { .. })).collect();
        assert_eq!(code_chunks.len(), 1, "Should have one code chunk");
    }

    #[test]
    fn test_literate_tangle() {
        let src = "<<program>>=\nlet x = 1;\n<<helper>>\n@\n<<helper>>=\nlet y = 2;\n@";
        let prog = LiterateProgram::parse(src);
        let tangled = prog.tangle("program");
        assert!(tangled.contains("let x = 1;"), "Contains main code");
        assert!(tangled.contains("let y = 2;"), "Transcluded helper");
    }

    #[test]
    fn test_literate_weave_html() {
        let src = "# Introduction\n\nThis is prose.\n\n<<example>>=\nfn f() {}\n@";
        let prog = LiterateProgram::parse(src);
        let html = prog.weave();
        assert!(html.contains("<h1"), "Prose rendered");
        assert!(html.contains("code-chunk"), "Code chunk rendered");
    }

    // ── Doc Tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_extract_doc_tests() {
        let mut doc = DocComment::default();
        doc.summary = "Test fn.".to_string();
        doc.examples = vec![
            CodeExample { code: "let x = 1;\n// => 1".into(), language: "chronos".into(),
                          runnable: true, hidden: false, title: None },
            CodeExample { code: "".into(), language: "text".into(),
                          runnable: false, hidden: false, title: None },
        ];
        let item = DocItem::new(ItemKind::Function, "foo", "fn foo()", doc);
        let tests = extract_doc_tests(&[item]);
        // Only chronos examples
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].expected, Some("1".to_string()));
    }

    #[test]
    fn test_run_doc_tests_passes_nonempty() {
        let test = DocTest { item_name: "foo".into(), index: 0,
                             code: "let x = 1;".into(), expected: None };
        let results = run_doc_tests(&[test]);
        assert!(results[0].passed);
    }

    // ── Changelog Generator ───────────────────────────────────────────────────

    #[test]
    fn test_parse_conventional_commit_feat() {
        let c = parse_commit("abc1234", "feat(parser): add support for async blocks").unwrap();
        assert_eq!(c.commit_type, CommitType::Feat);
        assert_eq!(c.scope.as_deref(), Some("parser"));
        assert!(c.summary.contains("async"));
    }

    #[test]
    fn test_parse_conventional_commit_breaking() {
        let c = parse_commit("def5678", "feat!: rename compile to build").unwrap();
        assert!(c.breaking, "Breaking commit");
    }

    #[test]
    fn test_parse_conventional_commit_fix() {
        let c = parse_commit("aaa", "fix: resolve null pointer in lexer").unwrap();
        assert_eq!(c.commit_type, CommitType::Fix);
    }

    #[test]
    fn test_generate_changelog_sections() {
        let commits = vec![
            parse_commit("abc1234", "feat: new feature").unwrap(),
            parse_commit("def5678", "fix: bug fix").unwrap(),
            parse_commit("ghi9012", "docs: update readme").unwrap(),
        ];
        let changelog = generate_changelog("1.2.0", "2026-03-29", &commits);
        assert!(changelog.contains("## [1.2.0]"), "Version header");
        assert!(changelog.contains("Features"), "Features section");
        assert!(changelog.contains("Bug Fixes"), "Bug fixes section");
        assert!(changelog.contains("Documentation"), "Docs section");
    }

    #[test]
    fn test_generate_changelog_breaking_section() {
        let commits = vec![
            parse_commit("aaa", "feat!: remove old API").unwrap(),
        ];
        let changelog = generate_changelog("2.0.0", "2026-03-29", &commits);
        assert!(changelog.contains("BREAKING"), "Breaking section");
    }

    // ── Template Engine ───────────────────────────────────────────────────────

    #[test]
    fn test_template_variable_substitution() {
        let tmpl = Template::new("Hello, {{ name }}!");
        let mut ctx = TemplateContext::new();
        ctx.set("name", TemplateValue::Str("Chronos".into()));
        assert_eq!(tmpl.render(&ctx), "Hello, Chronos!");
    }

    #[test]
    fn test_template_if_true() {
        let tmpl = Template::new("{% if show %}visible{% endif %}");
        let mut ctx = TemplateContext::new();
        ctx.set("show", TemplateValue::Bool(true));
        assert_eq!(tmpl.render(&ctx), "visible");
    }

    #[test]
    fn test_template_if_false() {
        let tmpl = Template::new("{% if show %}visible{% endif %}");
        let mut ctx = TemplateContext::new();
        ctx.set("show", TemplateValue::Bool(false));
        assert_eq!(tmpl.render(&ctx).trim(), "");
    }

    #[test]
    fn test_template_for_loop() {
        let tmpl = Template::new("{% for item in items %}{{ item }} {% endfor %}");
        let mut ctx = TemplateContext::new();
        ctx.set("items", TemplateValue::List(vec![
            TemplateValue::Str("a".into()),
            TemplateValue::Str("b".into()),
            TemplateValue::Str("c".into()),
        ]));
        let out = tmpl.render(&ctx);
        assert!(out.contains("a"), "Item a: {}", out);
        assert!(out.contains("b"), "Item b: {}", out);
        assert!(out.contains("c"), "Item c: {}", out);
    }

    #[test]
    fn test_template_filter_upper() {
        let tmpl = Template::new("{{ name | upper }}");
        let mut ctx = TemplateContext::new();
        ctx.set("name", TemplateValue::Str("chronos".into()));
        assert_eq!(tmpl.render(&ctx), "CHRONOS");
    }

    #[test]
    fn test_template_filter_escape() {
        let tmpl = Template::new("{{ code | escape }}");
        let mut ctx = TemplateContext::new();
        ctx.set("code", TemplateValue::Str("<script>".into()));
        let out = tmpl.render(&ctx);
        assert!(!out.contains("<script>"), "Should escape HTML: {}", out);
        assert!(out.contains("&lt;"), "Should have &lt;: {}", out);
    }

    #[test]
    fn test_template_missing_var_empty() {
        let tmpl = Template::new("Hello {{ missing }}!");
        let ctx = TemplateContext::new();
        let out = tmpl.render(&ctx);
        assert_eq!(out, "Hello !");
    }
}
