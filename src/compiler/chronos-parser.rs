// ============================================================================
// CHRONOS PARSER — Recursive Descent with Pratt Expression Parsing
// ============================================================================
// This parser converts a token stream from the Chronos lexer into a fully
// typed Abstract Syntax Tree (AST). It uses recursive descent for
// declarations and statements, and Pratt parsing (top-down operator
// precedence) for expressions — this combination gives us excellent error
// messages, easy extensibility, and correct precedence handling.
//
// This file references types from the main AST (chronos-lang artifact) and
// the lexer (chronos-lexer artifact).
// ============================================================================

// We assume the AST types from the main artifact are available.
// In a real project, these would be in separate modules with `use` imports.

// use std::collections::HashMap; // provided by parent scope in compiler-core

// =====================================================================
// RE-EXPORTS / TYPE STUBS
// =====================================================================
// In a real Cargo workspace, these would be `use crate::ast::*` and
// `use crate::lexer::*`. Here we reference them conceptually.
// For compilation, copy the lexer Token enum and all AST types from the
// other two artifacts into a shared crate.

// Forward-references to types defined in the AST artifact:
//   ChronosType, Kind, Lifetime, TypeParam, Variance, TypeBound, Effect,
//   Permission, TensorShape, DeviceTarget, DistributionStrategy,
//   Program, Item, ClassDecl, StructDecl, DataClassDecl, TraitDecl,
//   EnumDecl, SealedDecl, TemplateDecl, ImplBlock, FunctionDecl,
//   DegradableFunctionDecl, KernelDecl, AiSkillDecl, AiToolDecl,
//   AiPipelineDecl, ModuleDecl, ImportDecl, VersionAnnotation,
//   Visibility, FieldDecl, FunctionSignature, Parameter, ConstructorDecl,
//   DestructorDecl, Expression, Statement, BinOp, UnaryOp, MatchArm,
//   Pattern, DegradationSchedule, ChronosTimestamp, etc.

// Forward-references to lexer types:
//   Token, SpannedToken

// =====================================================================
// PARSER STATE
// =====================================================================

/// The parser holds a reference to the token stream and a cursor position.
/// It never backtracks — all decisions are made with at most 2 tokens of
/// lookahead, which keeps performance linear in the size of the input.
pub struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
    /// Accumulated parse errors. We collect them all and report at the end,
    /// rather than aborting on the first one.
    errors: Vec<ParseError>,
    /// Feature 3: version annotations that apply to the next item.
    pending_version_annotations: Vec<VersionAnnotation>,
    /// Feature 3: version annotations parsed from //! comments.
    pending_version_comments: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
    pub hint: Option<String>,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[PARSE ERROR] {}:{}: {}", self.line, self.column, self.message)?;
        if let Some(hint) = &self.hint {
            write!(f, "\n  hint: {}", hint)?;
        }
        Ok(())
    }
}

impl Parser {
    pub fn new(tokens: Vec<SpannedToken>) -> Self {
        Self {
            tokens,
            pos: 0,
            errors: Vec::new(),
            pending_version_annotations: Vec::new(),
            pending_version_comments: Vec::new(),
        }
    }

    // =================================================================
    // CORE NAVIGATION METHODS
    // =================================================================

    /// Look at the current token without consuming it.
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos).map(|t| &t.token)
    }

    /// Look at the token N positions ahead.
    fn peek_ahead(&self, n: usize) -> Option<&Token> {
        self.tokens.get(self.pos + n).map(|t| &t.token)
    }

    /// Get the current spanned token (for error locations).
    fn current_spanned(&self) -> Option<&SpannedToken> {
        self.tokens.get(self.pos)
    }

    /// Consume the current token and advance.
    fn advance(&mut self) -> Option<&SpannedToken> {
        let tok = self.tokens.get(self.pos);
        if tok.is_some() { self.pos += 1; }
        tok
    }

    /// Check if we've reached the end of the token stream.
    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    /// Expect a specific token, emit an error if it's not there.
    fn expect(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            let (line, col) = self.current_location();
            self.errors.push(ParseError {
                message: format!(
                    "Expected {:?}, found {:?}",
                    expected,
                    self.peek().cloned().unwrap_or(Token::KwNone) // placeholder
                ),
                line, column: col,
                hint: None,
            });
            false
        }
    }

    /// Expect an identifier and return its name.
    fn expect_identifier(&mut self) -> String {
        if let Some(Token::Identifier(name)) = self.peek().cloned() {
            self.advance();
            name
        } else {
            let (line, col) = self.current_location();
            self.errors.push(ParseError {
                message: format!("Expected identifier, found {:?}", self.peek()),
                line, column: col,
                hint: None,
            });
            "<error>".to_string()
        }
    }

    /// Get the current source location for error reporting.
    fn current_location(&self) -> (usize, usize) {
        self.current_spanned()
            .map(|t| (t.line, t.column))
            .unwrap_or((0, 0))
    }

    /// Check if the current token matches, and consume it if so.
    fn eat(&mut self, expected: &Token) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    // =================================================================
    // TOP-LEVEL: PARSE A COMPLETE PROGRAM
    // =================================================================

    /// Entry point: parse the entire token stream into a Program AST.
    pub fn parse_program(&mut self) -> Program {
        let mut items = Vec::new();

        while !self.is_eof() {
            // Collect version annotation comments before each item.
            self.collect_version_comments();

            if self.is_eof() { break; }

            match self.parse_item() {
                Some(item) => items.push(item),
                None => {
                    // Error recovery: skip the current token and try again.
                    let (line, col) = self.current_location();
                    self.errors.push(ParseError {
                        message: format!(
                            "Unexpected token {:?} at top level", self.peek()
                        ),
                        line, column: col,
                        hint: Some("Expected a declaration (fn, class, struct, etc.)".to_string()),
                    });
                    self.advance(); // Skip to recover
                }
            }
        }

        Program {
            items,
            version_rules: None, // Loaded separately from .rules file
        }
    }

    /// Collect //! version annotation comments that precede a declaration.
    fn collect_version_comments(&mut self) {
        while let Some(Token::VersionComment(comment)) = self.peek() {
            self.pending_version_comments.push(comment.clone());
            self.advance();
        }
    }

    // =================================================================
    // ITEM PARSING — Each declaration type gets its own method
    // =================================================================

    /// Parse a single top-level item.
    fn parse_item(&mut self) -> Option<Item> {
        // First, handle visibility modifiers that precede the actual declaration.
        let visibility = self.parse_visibility();

        // Handle annotations: @deprecated, @device(gpu), etc.
        let annotations = self.parse_annotations();

        // Now dispatch based on the leading keyword.
        match self.peek()? {
            Token::KwFn | Token::KwFun => {
                Some(Item::FunctionDecl(self.parse_function_decl(visibility, annotations)))
            }
            Token::KwAsync => {
                // async fn ...
                self.advance(); // consume 'async'
                let mut f = self.parse_function_decl(visibility, annotations);
                f.signature.is_async = true;
                Some(Item::FunctionDecl(f))
            }
            Token::KwDegradable => {
                Some(Item::DegradableFunctionDecl(self.parse_degradable_function(visibility)))
            }
            Token::KwClass => {
                Some(Item::ClassDecl(self.parse_class_decl(visibility, false, false)))
            }
            Token::KwAbstract => {
                self.advance();
                if self.peek() == Some(&Token::KwClass) {
                    Some(Item::ClassDecl(self.parse_class_decl(visibility, true, false)))
                } else {
                    self.error_here("Expected 'class' after 'abstract'");
                    None
                }
            }
            Token::KwFinal => {
                self.advance();
                Some(Item::ClassDecl(self.parse_class_decl(visibility, false, true)))
            }
            Token::KwData => {
                Some(Item::DataClassDecl(self.parse_data_class_decl(visibility)))
            }
            Token::KwStruct => {
                Some(Item::StructDecl(self.parse_struct_decl(visibility)))
            }
            Token::KwEnum => {
                Some(Item::EnumDecl(self.parse_enum_decl(visibility)))
            }
            Token::KwSealed => {
                Some(Item::SealedDecl(self.parse_sealed_decl(visibility)))
            }
            Token::KwTrait | Token::KwInterface => {
                Some(Item::TraitDecl(self.parse_trait_decl(visibility)))
            }
            Token::KwImpl => {
                Some(Item::ImplBlock(self.parse_impl_block()))
            }
            Token::KwTemplate => {
                Some(Item::TemplateDecl(self.parse_template_decl(visibility)))
            }
            Token::KwType => {
                Some(Item::TypeAlias(self.parse_type_alias(visibility)))
            }
            Token::KwMod | Token::KwModule => {
                Some(Item::ModuleDecl(self.parse_module_decl()))
            }
            Token::KwImport | Token::KwUse => {
                Some(Item::ImportDecl(self.parse_import_decl()))
            }
            Token::KwAi => {
                self.parse_ai_item()
            }
            Token::KwKernel => {
                Some(Item::KernelDecl(self.parse_kernel_decl()))
            }
            _ => None,
        }
    }

    // =================================================================
    // VISIBILITY PARSING
    // =================================================================

    fn parse_visibility(&mut self) -> Visibility {
        match self.peek() {
            Some(Token::KwPub) => { self.advance(); Visibility::Public }
            Some(Token::KwPrivate) => { self.advance(); Visibility::Private }
            Some(Token::KwProtected) => { self.advance(); Visibility::Protected }
            Some(Token::KwInternal) => { self.advance(); Visibility::Internal }
            Some(Token::KwCrate) => { self.advance(); Visibility::Crate }
            _ => Visibility::Private, // Default visibility
        }
    }

    // =================================================================
    // ANNOTATION PARSING
    // =================================================================

    fn parse_annotations(&mut self) -> Vec<Annotation> {
        let mut annotations = Vec::new();
        while let Some(Token::Annotation(name)) = self.peek().cloned() {
            self.advance();
            let args = if self.eat(&Token::LParen) {
                let args = self.parse_expression_list();
                self.expect(&Token::RParen);
                args
            } else {
                Vec::new()
            };
            annotations.push(Annotation { name, args });
        }
        annotations
    }

    // =================================================================
    // FUNCTION DECLARATION
    // =================================================================

    fn parse_function_decl(&mut self, visibility: Visibility, annotations: Vec<Annotation>) -> FunctionDecl {
        // Consume 'fn' or 'fun'
        self.advance();

        let name = self.expect_identifier();

        // Optional type parameters: fn foo<T, U: Display>(...)
        let type_params = self.parse_optional_type_params();

        // Parameter list
        self.expect(&Token::LParen);
        let params = self.parse_param_list();
        self.expect(&Token::RParen);

        // Optional return type: -> Type
        let return_type = if self.eat(&Token::Arrow) {
            self.parse_type()
        } else {
            ChronosType::Void
        };

        // Optional effect annotation: throws IOException, performs IO
        let effects = self.parse_optional_effects();

        // Optional where clause: where T: Display + Clone
        let _where_clauses = self.parse_optional_where_clause();

        // Function body
        let body = if self.peek() == Some(&Token::LBrace) {
            self.parse_block()
        } else {
            // Expression body: fn square(x: i32) -> i32 = x * x;
            if self.eat(&Token::Eq) {
                let expr = self.parse_expression(0);
                self.eat(&Token::Semicolon);
                vec![Statement::Return(Some(expr))]
            } else {
                self.eat(&Token::Semicolon);
                Vec::new()  // Declaration without body (in trait/abstract)
            }
        };

        FunctionDecl {
            signature: FunctionSignature {
                name,
                visibility,
                type_params,
                params,
                return_type,
                effects,
                is_async: false,  // Set by caller if preceded by 'async'
                is_const: false,
                is_inline: false,
                lifetime_params: Vec::new(),
            },
            body,
            annotations,
        }
    }

    // =================================================================
    // DEGRADABLE FUNCTION (Feature 6)
    // =================================================================

    fn parse_degradable_function(&mut self, visibility: Visibility) -> DegradableFunctionDecl {
        self.advance(); // consume 'degradable'

        let function = self.parse_function_decl(visibility, Vec::new());

        // Now parse the degradation schedule.
        // Syntax:
        //   degradable fn foo() -> Bar
        //       expires 2027-01-01
        //       warns 2026-06-01
        //       replaces new_foo
        //   { ... }
        //
        // Or as annotations:
        //   @degradable(expires = "2027-01-01", warns = "2026-06-01")
        //   fn foo() { ... }
        //
        // The inline keyword-based syntax is parsed here.

        let mut warn_after = ChronosTimestamp::new(2099, 12, 31);
        let mut expire_after = ChronosTimestamp::new(2099, 12, 31);
        let mut replacement = None;
        let mut reason = None;

        // Check for trailing degradation clauses (they may appear before the
        // body or as annotations — we already consumed the body above, so
        // in this simplified version we look for them as annotations on the
        // function). A full implementation would parse them between the
        // signature and the body.
        // For demonstration, we look for annotations that were already parsed:
        for ann in &function.annotations {
            match ann.name.as_str() {
                "expires" => {
                    if let Some(Expression::StringLiteral(s)) = ann.args.first() {
                        if let Some(ts) = parse_timestamp(s) {
                            expire_after = ts;
                        }
                    }
                }
                "warns" => {
                    if let Some(Expression::StringLiteral(s)) = ann.args.first() {
                        if let Some(ts) = parse_timestamp(s) {
                            warn_after = ts;
                        }
                    }
                }
                "replaces" => {
                    if let Some(Expression::StringLiteral(s)) = ann.args.first() {
                        replacement = Some(s.clone());
                    }
                }
                "reason" => {
                    if let Some(Expression::StringLiteral(s)) = ann.args.first() {
                        reason = Some(s.clone());
                    }
                }
                _ => {}
            }
        }

        DegradableFunctionDecl {
            function,
            degradation: DegradationSchedule {
                warn_after,
                error_after: None,
                expire_after,
                replacement,
                reason,
                phases: Vec::new(),
            },
        }
    }

    // =================================================================
    // CLASS DECLARATION (Feature 2)
    // =================================================================

    fn parse_class_decl(&mut self, visibility: Visibility, is_abstract: bool, is_final: bool) -> ClassDecl {
        self.advance(); // consume 'class'

        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();

        // Optional superclass: class Dog : Animal
        let superclass = if self.eat(&Token::Colon) {
            Some(self.parse_type())
        } else {
            None
        };

        // Optional interface list: class Dog : Animal, Serializable, Printable
        let mut interfaces = Vec::new();
        while self.eat(&Token::Comma) {
            interfaces.push(self.parse_type());
        }

        // Class body
        self.expect(&Token::LBrace);

        let mut fields = Vec::new();
        let mut methods = Vec::new();
        let mut constructors = Vec::new();
        let mut destructor = None;
        let mut companion = None;

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let member_vis = self.parse_visibility();
            let member_annotations = self.parse_annotations();

            match self.peek().cloned() {
                Some(Token::KwFn) | Some(Token::KwFun) => {
                    methods.push(self.parse_function_decl(member_vis, member_annotations));
                }
                Some(Token::KwNew) | Some(Token::Identifier(_)) => {
                    constructors.push(self.parse_constructor(member_vis));
                }
                Some(Token::KwDrop) | Some(Token::Tilde) => {
                    destructor = Some(self.parse_destructor());
                }
                Some(Token::KwCompanion) => {
                    self.advance();
                    companion = Some(Box::new(self.parse_class_decl(
                        Visibility::Public, false, false
                    )));
                }
                Some(Token::KwLet) | Some(Token::KwVar) | Some(Token::KwVal) => {
                    fields.push(self.parse_field_decl(member_vis));
                }
                _ => {
                    // Try to parse as a field (type name : Type)
                    fields.push(self.parse_field_decl(member_vis));
                }
            }
        }

        self.expect(&Token::RBrace);

        ClassDecl {
            name,
            visibility,
            type_params,
            superclass,
            interfaces,
            fields,
            methods,
            constructors,
            destructor,
            is_abstract,
            is_final,
            companion,
        }
    }

    // =================================================================
    // DATA CLASS (Kotlin-style, Feature 2)
    // =================================================================

    fn parse_data_class_decl(&mut self, visibility: Visibility) -> DataClassDecl {
        self.advance(); // consume 'data'
        self.expect(&Token::KwClass);

        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();

        // Primary constructor parameters become fields.
        self.expect(&Token::LParen);
        let fields = self.parse_field_list();
        self.expect(&Token::RParen);

        // Auto-derive list: data class Point(x: f64, y: f64) derive(Serialize, Debug)
        let auto_derive = if self.eat(&Token::KwDerive) {
            self.parse_derive_list()
        } else {
            vec![AutoDerive::Equals, AutoDerive::Hash, AutoDerive::ToString,
                 AutoDerive::Copy, AutoDerive::Destructure]
        };

        // Optional body
        if self.peek() == Some(&Token::LBrace) {
            self.parse_block(); // Parse and discard for now
        }

        DataClassDecl {
            name,
            visibility,
            type_params,
            fields,
            auto_derive,
        }
    }

    // =================================================================
    // STRUCT DECLARATION (Feature 2)
    // =================================================================

    fn parse_struct_decl(&mut self, visibility: Visibility) -> StructDecl {
        self.advance(); // consume 'struct'

        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();

        self.expect(&Token::LBrace);
        let fields = self.parse_field_list_in_braces();
        // Note: expect RBrace is handled inside parse_field_list_in_braces

        StructDecl {
            name,
            visibility,
            type_params,
            fields,
            is_packed: false,
            is_repr_c: false,
            copy_semantics: CopySemantics::Move,
        }
    }

    // =================================================================
    // ENUM / ADT DECLARATION (Feature 2)
    // =================================================================

    fn parse_enum_decl(&mut self, visibility: Visibility) -> EnumDecl {
        self.advance(); // consume 'enum'

        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();

        self.expect(&Token::LBrace);

        let mut variants = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let variant_name = self.expect_identifier();

            let fields = if self.peek() == Some(&Token::LParen) {
                // Tuple variant: Some(T)
                self.advance();
                let types = self.parse_type_list();
                self.expect(&Token::RParen);
                VariantFields::Tuple(types)
            } else if self.peek() == Some(&Token::LBrace) {
                // Struct variant: Complex { real: f64, imag: f64 }
                self.advance();
                let fields = self.parse_field_list_in_braces();
                VariantFields::Struct(fields)
            } else {
                VariantFields::Unit
            };

            variants.push(EnumVariant { name: variant_name, fields });
            self.eat(&Token::Comma);
        }

        self.expect(&Token::RBrace);

        EnumDecl { name, type_params, variants }
    }

    // =================================================================
    // SEALED HIERARCHY (Scala 3 / Kotlin, Feature 2)
    // =================================================================

    fn parse_sealed_decl(&mut self, visibility: Visibility) -> SealedDecl {
        self.advance(); // consume 'sealed'

        // sealed trait / sealed class
        let _kind = self.peek().cloned();
        self.advance(); // consume 'trait' or 'class'

        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();

        self.expect(&Token::LBrace);

        let mut variants = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let vis = self.parse_visibility();
            match self.peek() {
                Some(Token::KwData) => {
                    variants.push(SealedVariant::DataClass(self.parse_data_class_decl(vis)));
                }
                Some(Token::KwClass) => {
                    variants.push(SealedVariant::SubClass(
                        self.parse_class_decl(vis, false, false)
                    ));
                }
                Some(Token::KwObject) => {
                    self.advance();
                    let obj_name = self.expect_identifier();
                    variants.push(SealedVariant::Singleton(obj_name));
                    self.eat(&Token::Semicolon);
                }
                Some(Token::KwCase) => {
                    self.advance();
                    let case_name = self.expect_identifier();
                    if self.peek() == Some(&Token::LParen) {
                        // case class-like syntax
                        let dc = DataClassDecl {
                            name: case_name,
                            visibility: vis,
                            type_params: Vec::new(),
                            fields: {
                                self.advance(); // (
                                let f = self.parse_field_list();
                                self.expect(&Token::RParen);
                                f
                            },
                            auto_derive: vec![AutoDerive::Equals, AutoDerive::Hash],
                        };
                        variants.push(SealedVariant::DataClass(dc));
                    } else {
                        variants.push(SealedVariant::Singleton(case_name));
                    }
                    self.eat(&Token::Semicolon);
                }
                _ => {
                    self.error_here("Expected variant declaration in sealed type");
                    self.advance();
                }
            }
        }

        self.expect(&Token::RBrace);

        SealedDecl { name, type_params, variants }
    }

    // =================================================================
    // TRAIT / INTERFACE DECLARATION (Feature 2)
    // =================================================================

    fn parse_trait_decl(&mut self, _visibility: Visibility) -> TraitDecl {
        self.advance(); // consume 'trait' or 'interface'

        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();

        // Super traits: trait Printable : Display + Debug
        let super_traits = if self.eat(&Token::Colon) {
            let mut traits = vec![self.parse_type()];
            while self.eat(&Token::Plus) {
                traits.push(self.parse_type());
            }
            traits
        } else {
            Vec::new()
        };

        self.expect(&Token::LBrace);

        let mut methods = Vec::new();
        let mut associated_types = Vec::new();

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            match self.peek() {
                Some(Token::KwType) => {
                    self.advance();
                    let assoc_name = self.expect_identifier();
                    let bounds = if self.eat(&Token::Colon) {
                        vec![TypeBound::Implements(self.expect_identifier())]
                    } else {
                        Vec::new()
                    };
                    let default = if self.eat(&Token::Eq) {
                        Some(self.parse_type())
                    } else {
                        None
                    };
                    self.eat(&Token::Semicolon);
                    associated_types.push(AssociatedType { name: assoc_name, bounds, default });
                }
                Some(Token::KwFn) | Some(Token::KwFun) => {
                    let vis = self.parse_visibility();
                    let sig_fn = self.parse_function_decl(vis, Vec::new());
                    let default_impl = if sig_fn.body.is_empty() {
                        None
                    } else {
                        Some(sig_fn.body.clone())
                    };
                    methods.push(TraitMethod {
                        signature: sig_fn.signature,
                        default_impl,
                    });
                }
                _ => {
                    // Try parsing as method with default visibility
                    let vis = self.parse_visibility();
                    if matches!(self.peek(), Some(Token::KwFn) | Some(Token::KwFun)) {
                        let sig_fn = self.parse_function_decl(vis, Vec::new());
                        methods.push(TraitMethod {
                            signature: sig_fn.signature,
                            default_impl: if sig_fn.body.is_empty() { None } else { Some(sig_fn.body) },
                        });
                    } else {
                        self.error_here("Expected method or associated type in trait body");
                        self.advance();
                    }
                }
            }
        }

        self.expect(&Token::RBrace);

        TraitDecl { name, type_params, super_traits, methods, associated_types }
    }

    // =================================================================
    // IMPL BLOCK (Rust-style)
    // =================================================================

    fn parse_impl_block(&mut self) -> ImplBlock {
        self.advance(); // consume 'impl'

        let type_params = self.parse_optional_type_params();

        // impl TraitName for Type  OR  impl Type
        let first_type = self.parse_type();

        let (trait_name, target_type) = if self.eat(&Token::KwFor) {
            // impl Trait for Type
            let target = self.parse_type();
            let trait_name = if let ChronosType::Named { name, .. } = &first_type {
                Some(name.clone())
            } else {
                None
            };
            (trait_name, target)
        } else {
            (None, first_type)
        };

        self.expect(&Token::LBrace);

        let mut methods = Vec::new();
        let mut associated_types = Vec::new();

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let vis = self.parse_visibility();
            let anns = self.parse_annotations();

            match self.peek() {
                Some(Token::KwType) => {
                    self.advance();
                    let name = self.expect_identifier();
                    self.expect(&Token::Eq);
                    let ty = self.parse_type();
                    self.eat(&Token::Semicolon);
                    associated_types.push((name, ty));
                }
                Some(Token::KwFn) | Some(Token::KwFun) => {
                    methods.push(self.parse_function_decl(vis, anns));
                }
                _ => {
                    self.error_here("Expected method or type alias in impl block");
                    self.advance();
                }
            }
        }

        self.expect(&Token::RBrace);

        ImplBlock { type_params, trait_name, target_type, methods, associated_types }
    }

    // =================================================================
    // TEMPLATE DECLARATION (C++-style, Feature 2)
    // =================================================================

    fn parse_template_decl(&mut self, visibility: Visibility) -> TemplateDecl {
        self.advance(); // consume 'template'

        // template<typename T, int N>
        let params = self.parse_template_params();

        let name = self.expect_identifier();

        // The body can be a class, struct, or function.
        let body = match self.peek() {
            Some(Token::KwClass) => {
                TemplateBody::Class(self.parse_class_decl(visibility.clone(), false, false))
            }
            Some(Token::KwStruct) => {
                TemplateBody::Struct(self.parse_struct_decl(visibility.clone()))
            }
            Some(Token::KwFn) | Some(Token::KwFun) => {
                TemplateBody::Function(self.parse_function_decl(visibility.clone(), Vec::new()))
            }
            _ => {
                self.error_here("Expected class, struct, or fn after template parameters");
                // Recovery: return an empty struct template
                TemplateBody::Struct(StructDecl {
                    name: name.clone(),
                    visibility: visibility.clone(),
                    type_params: Vec::new(),
                    fields: Vec::new(),
                    is_packed: false,
                    is_repr_c: false,
                    copy_semantics: CopySemantics::Move,
                })
            }
        };

        TemplateDecl {
            name,
            params,
            body,
            specializations: Vec::new(),
        }
    }

    fn parse_template_params(&mut self) -> Vec<TemplateParam> {
        let mut params = Vec::new();
        if !self.eat(&Token::Lt) { return params; }

        loop {
            if self.is_eof() || self.peek() == Some(&Token::Gt) { break; }

            match self.peek().cloned() {
                Some(Token::KwType) | Some(Token::Identifier(_)) => {
                    self.advance();
                    let name = self.expect_identifier();
                    let bounds = if self.eat(&Token::Colon) {
                        vec![TypeBound::Implements(self.expect_identifier())]
                    } else {
                        Vec::new()
                    };
                    params.push(TemplateParam::Type { name, bounds });
                }
                _ => {
                    // Value template parameter: template<int N>
                    let ty = self.parse_type();
                    let name = self.expect_identifier();
                    params.push(TemplateParam::Value { name, ty });
                }
            }

            if !self.eat(&Token::Comma) { break; }
        }

        self.expect(&Token::Gt);
        params
    }

    // =================================================================
    // TYPE ALIAS
    // =================================================================

    fn parse_type_alias(&mut self, _visibility: Visibility) -> TypeAliasDecl {
        self.advance(); // consume 'type'
        let name = self.expect_identifier();
        let type_params = self.parse_optional_type_params();
        self.expect(&Token::Eq);
        let aliased = self.parse_type();
        self.eat(&Token::Semicolon);
        TypeAliasDecl { name, type_params, aliased }
    }

    // =================================================================
    // MODULE & IMPORT
    // =================================================================

    fn parse_module_decl(&mut self) -> ModuleDecl {
        self.advance(); // consume 'mod' or 'module'
        let name = self.expect_identifier();

        let items = if self.peek() == Some(&Token::LBrace) {
            self.advance();
            let mut items = Vec::new();
            while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                if let Some(item) = self.parse_item() {
                    items.push(item);
                } else {
                    self.advance();
                }
            }
            self.expect(&Token::RBrace);
            items
        } else {
            self.eat(&Token::Semicolon);
            Vec::new()
        };

        ModuleDecl { name, items }
    }

    fn parse_import_decl(&mut self) -> ImportDecl {
        self.advance(); // consume 'import' or 'use'

        let mut path = vec![self.expect_identifier()];
        while self.eat(&Token::PathSep) || self.eat(&Token::Dot) {
            path.push(self.expect_identifier());
        }

        let selective = if self.eat(&Token::LBrace) {
            let mut names = Vec::new();
            loop {
                names.push(self.expect_identifier());
                if !self.eat(&Token::Comma) { break; }
            }
            self.expect(&Token::RBrace);
            names
        } else {
            Vec::new()
        };

        let alias = if self.eat(&Token::KwAs) {
            Some(self.expect_identifier())
        } else {
            None
        };

        self.eat(&Token::Semicolon);

        ImportDecl { path, alias, selective }
    }

    // =================================================================
    // AI-NATIVE DECLARATIONS (Feature 4)
    // =================================================================

    fn parse_ai_item(&mut self) -> Option<Item> {
        self.advance(); // consume 'ai'

        match self.peek()? {
            Token::KwSkill => {
                self.advance();
                Some(Item::AiSkillDecl(self.parse_ai_skill()))
            }
            Token::KwTool => {
                self.advance();
                Some(Item::AiToolDecl(self.parse_ai_tool()))
            }
            Token::KwPipeline => {
                self.advance();
                Some(Item::AiPipelineDecl(self.parse_ai_pipeline()))
            }
            _ => {
                self.error_here("Expected 'skill', 'tool', or 'pipeline' after 'ai'");
                None
            }
        }
    }

    fn parse_ai_skill(&mut self) -> AiSkillDecl {
        let name = self.expect_identifier();
        self.expect(&Token::LBrace);

        let mut description = String::new();
        let mut input_schema = Vec::new();
        let mut output_schema = AiOutputSchema {
            ty: AiType::Text,
            description: String::new(),
            format: None,
        };
        let mut instructions = Vec::new();
        let mut constraints = Vec::new();
        let mut examples = Vec::new();

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            match self.peek() {
                Some(Token::KwInstruction) => {
                    self.advance();
                    if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
                        self.advance();
                        instructions.push(AiInstruction {
                            kind: AiInstructionKind::System,
                            content: s,
                            priority: instructions.len() as u8,
                        });
                    }
                }
                Some(Token::KwConstraint) => {
                    self.advance();
                    if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
                        self.advance();
                        constraints.push(AiConstraint {
                            description: s,
                            enforcement: ConstraintEnforcement::Runtime,
                        });
                    }
                }
                Some(Token::KwSchema) => {
                    self.advance();
                    // Parse schema block: schema { input: str, output: str }
                    if self.eat(&Token::LBrace) {
                        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                            let param_name = self.expect_identifier();
                            self.expect(&Token::Colon);
                            let param_type = self.parse_ai_type();
                            input_schema.push(AiParam {
                                name: param_name,
                                description: String::new(),
                                ty: param_type,
                                required: true,
                                default: None,
                                validation: None,
                            });
                            self.eat(&Token::Comma);
                        }
                        self.expect(&Token::RBrace);
                    }
                }
                Some(Token::KwExample) => {
                    self.advance();
                    // Parse example block
                    if self.eat(&Token::LBrace) {
                        let mut input = String::new();
                        let mut output = String::new();
                        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                            let key = self.expect_identifier();
                            self.expect(&Token::Colon);
                            if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
                                self.advance();
                                match key.as_str() {
                                    "input" => input = s,
                                    "output" => output = s,
                                    _ => {}
                                }
                            }
                            self.eat(&Token::Comma);
                        }
                        self.expect(&Token::RBrace);
                        examples.push(AiExample { input, output, explanation: None });
                    }
                }
                Some(Token::Identifier(ref s)) if s == "description" => {
                    self.advance();
                    if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
                        self.advance();
                        description = s;
                    }
                }
                _ => { self.advance(); }
            }
        }

        self.expect(&Token::RBrace);

        AiSkillDecl {
            name,
            description,
            input_schema,
            output_schema,
            instructions,
            constraints,
            examples,
            model_requirements: None,
        }
    }

    fn parse_ai_tool(&mut self) -> AiToolDecl {
        let name = self.expect_identifier();
        let description = if let Some(Token::StringLiteral(s)) = self.peek().cloned() {
            self.advance(); s
        } else { String::new() };

        // Parse the implementation function
        let implementation = if matches!(self.peek(), Some(Token::KwFn) | Some(Token::KwFun)) {
            self.parse_function_decl(Visibility::Public, Vec::new())
        } else {
            self.expect(&Token::LBrace);
            while !self.is_eof() && self.peek() != Some(&Token::RBrace) { self.advance(); }
            self.expect(&Token::RBrace);
            FunctionDecl {
                signature: FunctionSignature {
                    name: name.clone(),
                    visibility: Visibility::Public,
                    type_params: Vec::new(),
                    params: Vec::new(),
                    return_type: ChronosType::Void,
                    effects: Vec::new(),
                    is_async: false, is_const: false, is_inline: false,
                    lifetime_params: Vec::new(),
                },
                body: Vec::new(),
                annotations: Vec::new(),
            }
        };

        AiToolDecl {
            name, description,
            parameters: Vec::new(),
            return_type: AiOutputSchema { ty: AiType::Text, description: String::new(), format: None },
            implementation,
            retry_policy: None,
            rate_limit: None,
        }
    }

    fn parse_ai_pipeline(&mut self) -> AiPipelineDecl {
        let name = self.expect_identifier();
        self.expect(&Token::LBrace);

        let mut stages = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let stage_name = self.expect_identifier();
            self.expect(&Token::Arrow);
            let skill_or_tool = self.expect_identifier();
            stages.push(AiPipelineStage {
                name: stage_name,
                skill_or_tool,
                input_mapping: HashMap::new(),
                condition: None,
            });
            self.eat(&Token::Comma);
        }

        self.expect(&Token::RBrace);

        AiPipelineDecl {
            name,
            stages,
            error_handling: AiErrorStrategy::Fail,
        }
    }

    fn parse_ai_type(&mut self) -> AiType {
        match self.peek() {
            Some(Token::TyStr) | Some(Token::TyString) => {
                self.advance(); AiType::Text
            }
            Some(Token::TyI32) | Some(Token::TyI64) | Some(Token::TyF32) | Some(Token::TyF64) => {
                self.advance(); AiType::Number
            }
            Some(Token::TyBool) => {
                self.advance(); AiType::Boolean
            }
            _ => {
                // Default to bridging into the main type system
                let ty = self.parse_type();
                AiType::ChronosType(ty)
            }
        }
    }

    // =================================================================
    // KERNEL DECLARATION (Feature 5 — Mojo-inspired)
    // =================================================================

    fn parse_kernel_decl(&mut self) -> KernelDecl {
        self.advance(); // consume 'kernel'

        // Target device: kernel gpu / kernel tpu / kernel npu
        let target = match self.peek() {
            Some(Token::KwGpu) => { self.advance(); DeviceTarget::Gpu { index: 0 } }
            Some(Token::KwTpu) => { self.advance(); DeviceTarget::Tpu { index: 0 } }
            Some(Token::KwNpu) => { self.advance(); DeviceTarget::Npu { index: 0 } }
            Some(Token::KwCpu) => { self.advance(); DeviceTarget::Cpu }
            _ => DeviceTarget::Auto,
        };

        let name = self.expect_identifier();
        let _type_params = self.parse_optional_type_params();

        // Parameters
        self.expect(&Token::LParen);
        let params = self.parse_param_list().into_iter().map(|p| {
            KernelParam {
                name: p.name,
                ty: p.ty,
                memory_space: MemorySpace::Global,
            }
        }).collect();
        self.expect(&Token::RParen);

        // Return type
        let return_type = if self.eat(&Token::Arrow) {
            self.parse_type()
        } else {
            ChronosType::Void
        };

        // Body
        let body = self.parse_block();

        KernelDecl {
            name,
            target,
            params,
            return_type,
            body,
            launch_config: None,
            memory_annotations: Vec::new(),
        }
    }

    // =================================================================
    // TYPE PARSING (Feature 1: Universal Type System)
    // =================================================================
    // This is where we parse type expressions. The Chronos type grammar
    // handles everything from simple `i32` to complex types like
    // `linear Capability<File, Read>` and `tensor<bf16, [batch, 512, 768]>`.

    fn parse_type(&mut self) -> ChronosType {
        // Check for type qualifiers first.
        match self.peek() {
            Some(Token::KwLinear) => {
                self.advance();
                return ChronosType::Linear(Box::new(self.parse_type()));
            }
            Some(Token::KwAffine) => {
                self.advance();
                return ChronosType::Affine(Box::new(self.parse_type()));
            }
            Some(Token::Ampersand) => {
                self.advance();
                let mutable = self.eat(&Token::KwMut);
                let inner = self.parse_type();
                return ChronosType::Borrowed {
                    inner: Box::new(inner),
                    mutable,
                    lifetime: Lifetime::Inferred,
                };
            }
            _ => {}
        }

        // Parse the base type.
        let base = match self.peek() {
            // Primitive types
            Some(Token::TyVoid) => { self.advance(); ChronosType::Void }
            Some(Token::TyBool) => { self.advance(); ChronosType::Bool }
            Some(Token::TyI8) => { self.advance(); ChronosType::Int8 }
            Some(Token::TyI16) => { self.advance(); ChronosType::Int16 }
            Some(Token::TyI32) => { self.advance(); ChronosType::Int32 }
            Some(Token::TyI64) => { self.advance(); ChronosType::Int64 }
            Some(Token::TyI128) => { self.advance(); ChronosType::Int128 }
            Some(Token::TyU8) => { self.advance(); ChronosType::UInt8 }
            Some(Token::TyU16) => { self.advance(); ChronosType::UInt16 }
            Some(Token::TyU32) => { self.advance(); ChronosType::UInt32 }
            Some(Token::TyU64) => { self.advance(); ChronosType::UInt64 }
            Some(Token::TyU128) => { self.advance(); ChronosType::UInt128 }
            Some(Token::TyF16) => { self.advance(); ChronosType::Float16 }
            Some(Token::TyF32) => { self.advance(); ChronosType::Float32 }
            Some(Token::TyF64) => { self.advance(); ChronosType::Float64 }
            Some(Token::TyF128) => { self.advance(); ChronosType::Float128 }
            Some(Token::TyBf16) => { self.advance(); ChronosType::BFloat16 }
            Some(Token::TyChar) => { self.advance(); ChronosType::Char }
            Some(Token::TyStr) | Some(Token::TyString) => { self.advance(); ChronosType::Str }
            Some(Token::TyInt) => { self.advance(); ChronosType::IntArbitrary }
            Some(Token::TyUInt) => { self.advance(); ChronosType::UIntArbitrary }
            Some(Token::TyNever) => { self.advance(); ChronosType::Never }

            // Tensor type: tensor<bf16, [batch, 512, 768]>
            Some(Token::KwTensor) => {
                self.advance();
                self.parse_tensor_type()
            }

            // Tuple type: (i32, str, bool)
            Some(Token::LParen) => {
                self.advance();
                let types = self.parse_type_list();
                self.expect(&Token::RParen);
                if types.len() == 1 {
                    types.into_iter().next().unwrap() // Parenthesized type
                } else {
                    ChronosType::Tuple(types)
                }
            }

            // Array type: [i32; 10] or slice: [i32]
            Some(Token::LBracket) => {
                self.advance();
                let element = self.parse_type();
                if self.eat(&Token::Semicolon) {
                    // Fixed-size array: [i32; 10]
                    let size = if let Some(Token::IntLiteral(n)) = self.peek() {
                        let n = *n as usize;
                        self.advance();
                        Some(n)
                    } else { None };
                    self.expect(&Token::RBracket);
                    ChronosType::Array { element: Box::new(element), size }
                } else {
                    // Slice: [i32]
                    self.expect(&Token::RBracket);
                    ChronosType::Slice { element: Box::new(element) }
                }
            }

            // Function type: fn(i32, str) -> bool
            Some(Token::KwFn) => {
                self.advance();
                self.expect(&Token::LParen);
                let params = self.parse_type_list();
                self.expect(&Token::RParen);
                let ret = if self.eat(&Token::Arrow) {
                    self.parse_type()
                } else {
                    ChronosType::Void
                };
                ChronosType::Function {
                    params,
                    return_type: Box::new(ret),
                    effects: Vec::new(),
                }
            }

            // Named type (user-defined or generic): Vec<T>, MyStruct, std::io::File
            Some(Token::Identifier(_)) => {
                self.parse_named_type()
            }

            _ => {
                self.error_here("Expected type expression");
                ChronosType::Void
            }
        };

        // Check for postfix type modifiers.
        match self.peek() {
            // Optional type: T?
            Some(Token::Question) => {
                self.advance();
                ChronosType::Optional(Box::new(base))
            }
            _ => base,
        }
    }

    fn parse_named_type(&mut self) -> ChronosType {
        let mut path = vec![self.expect_identifier()];

        // Parse module path: std::collections::HashMap
        while self.peek() == Some(&Token::PathSep) {
            self.advance();
            path.push(self.expect_identifier());
        }

        let name = path.pop().unwrap();

        // Check for type arguments: Vec<T>, HashMap<K, V>
        if self.peek() == Some(&Token::Lt) {
            // Careful: < could be a comparison operator. We use the heuristic
            // that if we just parsed an identifier and see <, it's a type arg.
            self.advance();
            let _type_args = self.parse_type_list();
            self.expect(&Token::Gt);
        }

        ChronosType::Named { name, module_path: path }
    }

    fn parse_tensor_type(&mut self) -> ChronosType {
        // tensor<bf16, [batch, 512, 768]>
        self.expect(&Token::Lt);

        let element = Box::new(self.parse_type());
        self.expect(&Token::Comma);

        // Parse shape: [batch, 512, 768] or [3, 224, 224]
        self.expect(&Token::LBracket);
        let mut shape_parts: Vec<usize> = Vec::new();
        let mut is_symbolic = false;
        let mut static_dims: Vec<usize> = Vec::new();
        let mut symbolic_dims: Vec<String> = Vec::new();

        loop {
            match self.peek() {
                Some(Token::IntLiteral(n)) => {
                    let n = *n as usize;
                    self.advance();
                    static_dims.push(n);
                    symbolic_dims.push(n.to_string());
                }
                Some(Token::Identifier(s)) => {
                    is_symbolic = true;
                    symbolic_dims.push(s.clone());
                    self.advance();
                }
                _ => break,
            }
            if !self.eat(&Token::Comma) { break; }
        }

        self.expect(&Token::RBracket);

        let shape = if is_symbolic {
            TensorShape::Symbolic(symbolic_dims)
        } else {
            TensorShape::Static(static_dims)
        };

        // Optional device: tensor<bf16, [3, 224, 224], gpu>
        let device = if self.eat(&Token::Comma) {
            match self.peek() {
                Some(Token::KwGpu) => { self.advance(); DeviceTarget::Gpu { index: 0 } }
                Some(Token::KwTpu) => { self.advance(); DeviceTarget::Tpu { index: 0 } }
                Some(Token::KwNpu) => { self.advance(); DeviceTarget::Npu { index: 0 } }
                Some(Token::KwCpu) => { self.advance(); DeviceTarget::Cpu }
                _ => DeviceTarget::Auto,
            }
        } else {
            DeviceTarget::Auto
        };

        self.expect(&Token::Gt);

        ChronosType::Tensor { element, shape, device }
    }

    // =================================================================
    // EXPRESSION PARSING — Pratt Parser
    // =================================================================
    // The Pratt parser handles operator precedence elegantly. The key idea:
    // parse_expression(min_bp) parses any expression whose binding power
    // is at least min_bp. Left-recursive operators work because we use a
    // loop that calls parse_expression recursively with a higher min_bp.

    fn parse_expression(&mut self, min_bp: u8) -> Expression {
        // Step 1: Parse the left-hand side (prefix position).
        let mut lhs = self.parse_prefix();

        // Step 2: Loop over infix and postfix operators.
        loop {
            if self.is_eof() { break; }

            // Check for postfix operators first (higher precedence).
            if let Some(bp) = self.peek().and_then(|t| t.postfix_binding_power()) {
                if bp < min_bp { break; }

                let op_token = self.peek().cloned().unwrap();
                self.advance();

                lhs = match op_token {
                    Token::Question => {
                        // Error propagation: expr?
                        Expression::MethodCall {
                            object: Box::new(lhs),
                            method: "try_unwrap".to_string(),
                            args: Vec::new(),
                        }
                    }
                    Token::LParen => {
                        // Function call: expr(args...)
                        let args = if self.peek() != Some(&Token::RParen) {
                            self.parse_expression_list()
                        } else {
                            Vec::new()
                        };
                        self.expect(&Token::RParen);
                        Expression::Call {
                            function: Box::new(lhs),
                            args,
                        }
                    }
                    Token::LBracket => {
                        // Indexing: expr[index]
                        let index = self.parse_expression(0);
                        self.expect(&Token::RBracket);
                        Expression::MethodCall {
                            object: Box::new(lhs),
                            method: "index".to_string(),
                            args: vec![index],
                        }
                    }
                    _ => lhs,
                };
                continue;
            }

            // Check for infix operators.
            if let Some((l_bp, r_bp)) = self.peek().and_then(|t| t.infix_binding_power()) {
                if l_bp < min_bp { break; }

                let op_token = self.peek().cloned().unwrap();
                self.advance();

                // Special cases for non-standard infix operators.
                match &op_token {
                    Token::Dot | Token::SafeDot => {
                        // Member access: expr.field or expr.method(args)
                        let member = self.expect_identifier();

                        if self.peek() == Some(&Token::LParen) {
                            self.advance();
                            let args = if self.peek() != Some(&Token::RParen) {
                                self.parse_expression_list()
                            } else {
                                Vec::new()
                            };
                            self.expect(&Token::RParen);
                            lhs = Expression::MethodCall {
                                object: Box::new(lhs),
                                method: member,
                                args,
                            };
                        } else {
                            lhs = Expression::FieldAccess {
                                object: Box::new(lhs),
                                field: member,
                            };
                        }
                        continue;
                    }
                    Token::PathSep => {
                        // Path access: Type::method
                        let member = self.expect_identifier();
                        lhs = Expression::FieldAccess {
                            object: Box::new(lhs),
                            field: member,
                        };
                        continue;
                    }
                    Token::PipeForward => {
                        // Pipe forward: expr |> fn  →  fn(expr)
                        let rhs = self.parse_expression(r_bp);
                        lhs = Expression::Call {
                            function: Box::new(rhs),
                            args: vec![lhs],
                        };
                        continue;
                    }
                    _ => {}
                }

                let rhs = self.parse_expression(r_bp);
                let op = token_to_binop(&op_token);
                lhs = Expression::BinaryOp {
                    left: Box::new(lhs),
                    op,
                    right: Box::new(rhs),
                };
                continue;
            }

            break;
        }

        lhs
    }

    /// Parse a prefix expression (literal, identifier, unary operator, etc.).
    fn parse_prefix(&mut self) -> Expression {
        match self.peek().cloned() {
            Some(Token::IntLiteral(n)) => {
                self.advance();
                Expression::IntLiteral(n as i64)
            }
            Some(Token::FloatLiteral(f)) => {
                self.advance();
                Expression::FloatLiteral(f)
            }
            Some(Token::StringLiteral(s)) => {
                self.advance();
                Expression::StringLiteral(s)
            }
            Some(Token::CharLiteral(c)) => {
                self.advance();
                Expression::IntLiteral(c as i64)
            }
            Some(Token::KwTrue) => {
                self.advance();
                Expression::BoolLiteral(true)
            }
            Some(Token::KwFalse) => {
                self.advance();
                Expression::BoolLiteral(false)
            }
            Some(Token::Identifier(name)) => {
                self.advance();
                Expression::Identifier(name)
            }
            Some(Token::KwSelf_) | Some(Token::KwThis) => {
                self.advance();
                Expression::Identifier("self".to_string())
            }

            // Unary prefix operators: -x, !x, ~x, &x, *x
            Some(ref tok) if tok.prefix_binding_power().is_some() => {
                let bp = tok.prefix_binding_power().unwrap();
                let op_tok = tok.clone();
                self.advance();
                let operand = self.parse_expression(bp);
                let op = match op_tok {
                    Token::Minus => UnaryOp::Neg,
                    Token::Bang => UnaryOp::Not,
                    Token::Tilde => UnaryOp::BitNot,
                    Token::Ampersand => UnaryOp::Ref,
                    Token::Star => UnaryOp::Deref,
                    _ => UnaryOp::Neg,
                };
                Expression::UnaryOp { op, expr: Box::new(operand) }
            }

            // Parenthesized expression or tuple: (expr) or (a, b, c)
            Some(Token::LParen) => {
                self.advance();
                if self.peek() == Some(&Token::RParen) {
                    self.advance();
                    return Expression::Identifier("unit".to_string()); // Unit value
                }
                let expr = self.parse_expression(0);
                if self.eat(&Token::Comma) {
                    // Tuple literal: (a, b, c)
                    let mut elements = vec![expr];
                    loop {
                        elements.push(self.parse_expression(0));
                        if !self.eat(&Token::Comma) { break; }
                    }
                    self.expect(&Token::RParen);
                    // Represent as nested binary ops for simplicity
                    return elements.into_iter().reduce(|acc, e| {
                        Expression::BinaryOp {
                            left: Box::new(acc),
                            op: BinOp::Add, // Placeholder; real AST would use Tuple
                            right: Box::new(e),
                        }
                    }).unwrap();
                }
                self.expect(&Token::RParen);
                expr
            }

            // If expression
            Some(Token::KwIf) => {
                self.parse_if_expression()
            }

            // Match expression
            Some(Token::KwMatch) | Some(Token::KwWhen) => {
                self.parse_match_expression()
            }

            // Lambda: |params| -> Type { body }  or  { params -> body }
            Some(Token::Pipe) => {
                self.parse_lambda()
            }

            // Block expression: { statements; expr }
            Some(Token::LBrace) => {
                let stmts = self.parse_block();
                Expression::Block(stmts)
            }

            // AI invocation: @@skill_name { input: value }
            Some(Token::AtAt) => {
                self.advance();
                let skill_name = self.expect_identifier();
                let inputs = if self.eat(&Token::LBrace) {
                    let mut map = HashMap::new();
                    while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
                        let key = self.expect_identifier();
                        self.expect(&Token::Colon);
                        let value = self.parse_expression(0);
                        map.insert(key, value);
                        self.eat(&Token::Comma);
                    }
                    self.expect(&Token::RBrace);
                    map
                } else {
                    HashMap::new()
                };
                Expression::AiInvoke { skill_name, inputs }
            }

            _ => {
                self.error_here("Expected expression");
                self.advance();
                Expression::IntLiteral(0) // Error recovery
            }
        }
    }

    fn parse_if_expression(&mut self) -> Expression {
        self.advance(); // consume 'if'
        let condition = self.parse_expression(0);
        let then_branch = Expression::Block(self.parse_block());
        let else_branch = if self.eat(&Token::KwElse) {
            if self.peek() == Some(&Token::KwIf) {
                Some(Box::new(self.parse_if_expression()))
            } else {
                Some(Box::new(Expression::Block(self.parse_block())))
            }
        } else {
            None
        };
        Expression::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch,
        }
    }

    fn parse_match_expression(&mut self) -> Expression {
        self.advance(); // consume 'match' or 'when'
        let scrutinee = self.parse_expression(0);
        self.expect(&Token::LBrace);

        let mut arms = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let pattern = self.parse_pattern();
            let guard = if self.eat(&Token::KwIf) {
                Some(self.parse_expression(0))
            } else {
                None
            };
            self.expect(&Token::FatArrow);
            let body = self.parse_expression(0);
            arms.push(MatchArm { pattern, guard, body });
            self.eat(&Token::Comma);
        }

        self.expect(&Token::RBrace);
        Expression::Match { scrutinee: Box::new(scrutinee), arms }
    }

    fn parse_pattern(&mut self) -> Pattern {
        match self.peek().cloned() {
            Some(Token::Identifier(ref name)) if name == "_" => {
                self.advance();
                Pattern::Wildcard
            }
            Some(Token::Identifier(name)) => {
                self.advance();
                if self.peek() == Some(&Token::LParen) {
                    // Constructor pattern: Some(x)
                    self.advance();
                    let mut fields = Vec::new();
                    while self.peek() != Some(&Token::RParen) {
                        fields.push(self.parse_pattern());
                        self.eat(&Token::Comma);
                    }
                    self.expect(&Token::RParen);
                    Pattern::Constructor { name, fields }
                } else {
                    Pattern::Binding(name)
                }
            }
            Some(Token::IntLiteral(_)) | Some(Token::StringLiteral(_)) |
            Some(Token::KwTrue) | Some(Token::KwFalse) => {
                Pattern::Literal(self.parse_prefix())
            }
            Some(Token::LParen) => {
                self.advance();
                let mut patterns = Vec::new();
                while self.peek() != Some(&Token::RParen) {
                    patterns.push(self.parse_pattern());
                    self.eat(&Token::Comma);
                }
                self.expect(&Token::RParen);
                Pattern::Tuple(patterns)
            }
            _ => {
                self.error_here("Expected pattern");
                self.advance();
                Pattern::Wildcard
            }
        }
    }

    fn parse_lambda(&mut self) -> Expression {
        self.advance(); // consume '|'
        let mut params = Vec::new();
        while self.peek() != Some(&Token::Pipe) && !self.is_eof() {
            let name = self.expect_identifier();
            let ty = if self.eat(&Token::Colon) {
                self.parse_type()
            } else {
                ChronosType::Void // Type inference will fill this in
            };
            params.push(Parameter {
                name, ty, default: None, is_variadic: false,
            });
            self.eat(&Token::Comma);
        }
        self.expect(&Token::Pipe);

        let return_type = if self.eat(&Token::Arrow) {
            Some(self.parse_type())
        } else {
            None
        };

        let body = Box::new(self.parse_expression(0));

        Expression::Lambda { params, body, return_type }
    }

    // =================================================================
    // STATEMENT PARSING
    // =================================================================

    fn parse_block(&mut self) -> Vec<Statement> {
        self.expect(&Token::LBrace);
        let mut stmts = Vec::new();

        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            stmts.push(self.parse_statement());
        }

        self.expect(&Token::RBrace);
        stmts
    }

    fn parse_statement(&mut self) -> Statement {
        match self.peek() {
            Some(Token::KwLet) | Some(Token::KwVal) => self.parse_let_statement(false),
            Some(Token::KwVar) => self.parse_let_statement(true),
            Some(Token::KwReturn) => {
                self.advance();
                let value = if self.peek() != Some(&Token::Semicolon) && self.peek() != Some(&Token::RBrace) {
                    Some(self.parse_expression(0))
                } else {
                    None
                };
                self.eat(&Token::Semicolon);
                Statement::Return(value)
            }
            Some(Token::KwBreak) => {
                self.advance();
                self.eat(&Token::Semicolon);
                Statement::Break
            }
            Some(Token::KwContinue) => {
                self.advance();
                self.eat(&Token::Semicolon);
                Statement::Continue
            }
            Some(Token::KwWhile) => {
                self.advance();
                let condition = self.parse_expression(0);
                let body = self.parse_block();
                Statement::While { condition, body }
            }
            Some(Token::KwFor) => {
                self.advance();
                let binding = self.expect_identifier();
                self.expect(&Token::KwIn);
                let iterator = self.parse_expression(0);
                let body = self.parse_block();
                Statement::For { binding, iterator, body }
            }
            Some(Token::KwDrop) => {
                self.advance();
                let name = self.expect_identifier();
                self.eat(&Token::Semicolon);
                Statement::Drop(name)
            }
            Some(Token::KwDevice) => {
                self.advance();
                let target = match self.peek() {
                    Some(Token::KwGpu) => { self.advance(); DeviceTarget::Gpu { index: 0 } }
                    Some(Token::KwTpu) => { self.advance(); DeviceTarget::Tpu { index: 0 } }
                    Some(Token::KwNpu) => { self.advance(); DeviceTarget::Npu { index: 0 } }
                    _ => DeviceTarget::Auto,
                };
                let body = self.parse_block();
                Statement::DeviceScope { target, body }
            }
            _ => {
                let expr = self.parse_expression(0);
                // Check for assignment: expr = value
                if self.eat(&Token::Eq) {
                    let value = self.parse_expression(0);
                    self.eat(&Token::Semicolon);
                    Statement::Assignment { target: expr, value }
                } else {
                    self.eat(&Token::Semicolon);
                    Statement::ExprStatement(expr)
                }
            }
        }
    }

    fn parse_let_statement(&mut self, mutable: bool) -> Statement {
        self.advance(); // consume 'let', 'var', or 'val'

        let name = self.expect_identifier();
        let ty = if self.eat(&Token::Colon) {
            Some(self.parse_type())
        } else {
            None
        };

        self.expect(&Token::Eq);
        let value = self.parse_expression(0);
        self.eat(&Token::Semicolon);

        Statement::Let { name, ty, value, mutable }
    }

    // =================================================================
    // HELPER PARSERS
    // =================================================================

    fn parse_optional_type_params(&mut self) -> Vec<TypeParam> {
        if self.peek() != Some(&Token::Lt) { return Vec::new(); }
        self.advance();

        let mut params = Vec::new();
        loop {
            if self.is_eof() || self.peek() == Some(&Token::Gt) { break; }

            // Check for variance annotation: +T, -T
            let variance = match self.peek() {
                Some(Token::Plus) => { self.advance(); Variance::Covariant }
                Some(Token::Minus) => { self.advance(); Variance::Contravariant }
                _ => Variance::Invariant,
            };

            let name = self.expect_identifier();
            let bounds = if self.eat(&Token::Colon) {
                let mut bounds = vec![TypeBound::Implements(self.expect_identifier())];
                while self.eat(&Token::Plus) {
                    bounds.push(TypeBound::Implements(self.expect_identifier()));
                }
                bounds
            } else {
                Vec::new()
            };

            let default = if self.eat(&Token::Eq) {
                Some(self.parse_type())
            } else {
                None
            };

            params.push(TypeParam { name, variance, bounds, default });

            if !self.eat(&Token::Comma) { break; }
        }

        self.expect(&Token::Gt);
        params
    }

    fn parse_param_list(&mut self) -> Vec<Parameter> {
        let mut params = Vec::new();
        if self.peek() == Some(&Token::RParen) { return params; }

        loop {
            let is_variadic = self.eat(&Token::Ellipsis);
            let name = self.expect_identifier();
            self.expect(&Token::Colon);
            let ty = self.parse_type();
            let default = if self.eat(&Token::Eq) {
                Some(self.parse_expression(0))
            } else {
                None
            };

            params.push(Parameter { name, ty, default, is_variadic });

            if !self.eat(&Token::Comma) { break; }
        }

        params
    }

    fn parse_field_list(&mut self) -> Vec<FieldDecl> {
        let mut fields = Vec::new();
        loop {
            if self.is_eof() || self.peek() == Some(&Token::RParen) { break; }

            let mutable = self.eat(&Token::KwVar) || self.eat(&Token::KwMut);
            let name = self.expect_identifier();
            self.expect(&Token::Colon);
            let ty = self.parse_type();
            let default = if self.eat(&Token::Eq) {
                Some(self.parse_expression(0))
            } else { None };

            fields.push(FieldDecl {
                name, ty, visibility: Visibility::Public, mutable, default,
            });

            if !self.eat(&Token::Comma) { break; }
        }
        fields
    }

    fn parse_field_list_in_braces(&mut self) -> Vec<FieldDecl> {
        let mut fields = Vec::new();
        while !self.is_eof() && self.peek() != Some(&Token::RBrace) {
            let vis = self.parse_visibility();
            let mutable = self.eat(&Token::KwVar) || self.eat(&Token::KwMut);
            let name = self.expect_identifier();
            self.expect(&Token::Colon);
            let ty = self.parse_type();
            let default = if self.eat(&Token::Eq) {
                Some(self.parse_expression(0))
            } else { None };
            self.eat(&Token::Comma);
            self.eat(&Token::Semicolon);

            fields.push(FieldDecl { name, ty, visibility: vis, mutable, default });
        }
        self.expect(&Token::RBrace);
        fields
    }

    fn parse_field_decl(&mut self, visibility: Visibility) -> FieldDecl {
        let mutable = self.eat(&Token::KwLet) && false  // let = immutable
            || self.eat(&Token::KwVar) || self.eat(&Token::KwVal) && false;
        let name = self.expect_identifier();
        self.expect(&Token::Colon);
        let ty = self.parse_type();
        let default = if self.eat(&Token::Eq) {
            Some(self.parse_expression(0))
        } else { None };
        self.eat(&Token::Semicolon);
        FieldDecl { name, ty, visibility, mutable, default }
    }

    fn parse_constructor(&mut self, visibility: Visibility) -> ConstructorDecl {
        self.advance(); // consume 'new', 'init', or 'constructor'
        self.expect(&Token::LParen);
        let params = self.parse_param_list();
        self.expect(&Token::RParen);
        let body = self.parse_block();
        ConstructorDecl { visibility, params, body, is_primary: true }
    }

    fn parse_destructor(&mut self) -> DestructorDecl {
        self.advance(); // consume 'drop' or '~'
        if self.peek() == Some(&Token::LParen) {
            self.advance();
            self.expect(&Token::RParen);
        }
        let body = self.parse_block();
        DestructorDecl { body }
    }

    fn parse_type_list(&mut self) -> Vec<ChronosType> {
        let mut types = Vec::new();
        loop {
            types.push(self.parse_type());
            if !self.eat(&Token::Comma) { break; }
        }
        types
    }

    fn parse_expression_list(&mut self) -> Vec<Expression> {
        let mut exprs = Vec::new();
        loop {
            exprs.push(self.parse_expression(0));
            if !self.eat(&Token::Comma) { break; }
        }
        exprs
    }

    fn parse_derive_list(&mut self) -> Vec<AutoDerive> {
        let mut derives = Vec::new();
        self.expect(&Token::LParen);
        loop {
            if self.is_eof() || self.peek() == Some(&Token::RParen) { break; }
            let name = self.expect_identifier();
            derives.push(match name.as_str() {
                "Eq" | "Equals" => AutoDerive::Equals,
                "Hash" => AutoDerive::Hash,
                "Debug" | "ToString" => AutoDerive::ToString,
                "Copy" | "Clone" => AutoDerive::Copy,
                "Destructure" => AutoDerive::Destructure,
                "Serialize" | "Ser" => AutoDerive::Serialize,
                "Ord" | "Order" => AutoDerive::Order,
                _ => AutoDerive::Debug,
            });
            if !self.eat(&Token::Comma) { break; }
        }
        self.expect(&Token::RParen);
        derives
    }

    fn parse_optional_effects(&mut self) -> Vec<Effect> {
        // Parse effect annotations like: throws IOException, performs IO
        let mut effects = Vec::new();
        while let Some(Token::Identifier(ref s)) = self.peek() {
            match s.as_str() {
                "throws" => {
                    self.advance();
                    let err_type = self.expect_identifier();
                    effects.push(Effect::Throw(err_type));
                }
                "performs" => {
                    self.advance();
                    let eff_name = self.expect_identifier();
                    effects.push(match eff_name.as_str() {
                        "IO" => Effect::IO,
                        "Alloc" => Effect::Alloc,
                        "Async" => Effect::Async,
                        _ => Effect::IO,
                    });
                }
                _ => break,
            }
        }
        effects
    }

    fn parse_optional_where_clause(&mut self) -> Vec<TypeBound> {
        if !self.eat(&Token::KwWhere) { return Vec::new(); }
        let mut bounds = Vec::new();
        loop {
            let _name = self.expect_identifier();
            self.expect(&Token::Colon);
            bounds.push(TypeBound::Implements(self.expect_identifier()));
            while self.eat(&Token::Plus) {
                bounds.push(TypeBound::Implements(self.expect_identifier()));
            }
            if !self.eat(&Token::Comma) { break; }
        }
        bounds
    }

    // =================================================================
    // ERROR HELPERS
    // =================================================================

    fn error_here(&mut self, msg: &str) {
        let (line, col) = self.current_location();
        self.errors.push(ParseError {
            message: msg.to_string(),
            line, column: col,
            hint: None,
        });
    }

    /// Get all accumulated errors.
    pub fn get_errors(&self) -> &[ParseError] {
        &self.errors
    }
}

// =================================================================
// HELPER: Convert token to binary operator
// =================================================================
fn token_to_binop(token: &Token) -> BinOp {
    match token {
        Token::Plus => BinOp::Add,
        Token::Minus => BinOp::Sub,
        Token::Star => BinOp::Mul,
        Token::Slash => BinOp::Div,
        Token::Percent => BinOp::Mod,
        Token::AndAnd => BinOp::And,
        Token::OrOr => BinOp::Or,
        Token::EqEq => BinOp::Eq,
        Token::NotEq => BinOp::Neq,
        Token::Lt => BinOp::Lt,
        Token::Gt => BinOp::Gt,
        Token::LtEq => BinOp::Lte,
        Token::GtEq => BinOp::Gte,
        Token::Ampersand => BinOp::BitAnd,
        Token::Pipe => BinOp::BitOr,
        Token::Caret => BinOp::Xor,
        Token::Shl => BinOp::Shl,
        Token::Shr => BinOp::Shr,
        Token::At => BinOp::MatMul,
        Token::Eq => BinOp::Assign,
        Token::PlusEq => BinOp::AddAssign,
        Token::MinusEq => BinOp::SubAssign,
        Token::StarEq => BinOp::MulAssign,
        Token::SlashEq => BinOp::DivAssign,
        Token::PercentEq => BinOp::ModAssign,
        _ => BinOp::Add, // Fallback
    }
}

// =================================================================
// HELPER: Parse ISO timestamp from string "YYYY-MM-DD"
// =================================================================
fn parse_timestamp(s: &str) -> Option<ChronosTimestamp> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() >= 3 {
        Some(ChronosTimestamp {
            year: parts[0].parse().ok()?,
            month: parts[1].parse().ok()?,
            day: parts[2].parse().ok()?,
            hour: 0,
            minute: 0,
        })
    } else {
        None
    }
}
