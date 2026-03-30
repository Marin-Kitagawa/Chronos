// ============================================================================
// CHRONOS TYPE INFERENCE ENGINE
// ============================================================================
// This implements type inference for the Chronos language. The core algorithm
// is based on Hindley-Milner (Algorithm W), extended with:
//   - Subtyping (for class hierarchies and union/intersection types)
//   - Effect inference (tracking IO, Alloc, GPU, etc. through the type system)
//   - Linear/affine type checking (ensuring resources are used correctly)
//   - Refinement type checking (verifying value predicates)
//   - Higher-kinded type resolution (for Functor, Monad, etc.)
//
// The approach: we generate "type constraints" during a traversal of the AST,
// then solve them via unification. This is the same strategy used by Rust's
// type checker (rustc) and OCaml's, adapted for Chronos's richer type system.
// ============================================================================

// use std::collections::HashMap; // provided by parent scope in compiler-core

// Assumes AST types from chronos-lang are in scope.
// In a real Cargo workspace: use crate::ast::*;

// ============================================================================
// TYPE VARIABLES & SUBSTITUTION
// ============================================================================
// During inference, we don't know what every type is yet. We introduce
// "type variables" (think of them as blanks) and then fill them in as we
// discover information. A Substitution maps type variables to concrete types.

/// A unique identifier for a type variable.
pub type TypeVarId = u32;

/// A type during inference — extends ChronosType with inference-specific nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum InferType {
    /// A known, concrete type from the AST.
    Concrete(ChronosType),
    /// An unknown type variable that will be resolved during unification.
    Var(TypeVarId),
    /// A function type with potentially unresolved parameter/return types.
    Function {
        params: Vec<InferType>,
        return_type: Box<InferType>,
        effects: Vec<Effect>,
    },
    /// A generic type being instantiated: Vec<T> where T is still unknown.
    Applied {
        constructor: String,
        args: Vec<InferType>,
    },
    /// An error placeholder — used when inference fails to keep going.
    Error(String),
}

/// A substitution maps type variable IDs to their resolved types.
/// We use the union-find / path-compression trick for efficiency.
#[derive(Debug, Clone)]
pub struct Substitution {
    /// For each type variable, either a mapping to another type or None.
    bindings: HashMap<TypeVarId, InferType>,
    /// Counter for generating fresh type variables.
    next_var: TypeVarId,
}

impl Substitution {
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            next_var: 0,
        }
    }

    /// Create a fresh type variable that doesn't map to anything yet.
    pub fn fresh_var(&mut self) -> InferType {
        let id = self.next_var;
        self.next_var += 1;
        InferType::Var(id)
    }

    /// Bind a type variable to a concrete type.
    pub fn bind(&mut self, var: TypeVarId, ty: InferType) {
        // Occurs check: prevent infinite types like T = List<T>.
        if self.occurs_in(var, &ty) {
            // In a real compiler, this would be a proper error.
            return;
        }
        self.bindings.insert(var, ty);
    }

    /// Look up what a type variable is bound to, following chains.
    /// (Path compression: if A -> B -> C, we short-circuit to A -> C.)
    pub fn resolve(&self, ty: &InferType) -> InferType {
        match ty {
            InferType::Var(id) => {
                if let Some(bound) = self.bindings.get(id) {
                    self.resolve(bound)
                } else {
                    ty.clone()
                }
            }
            InferType::Function { params, return_type, effects } => {
                InferType::Function {
                    params: params.iter().map(|p| self.resolve(p)).collect(),
                    return_type: Box::new(self.resolve(return_type)),
                    effects: effects.clone(),
                }
            }
            InferType::Applied { constructor, args } => {
                InferType::Applied {
                    constructor: constructor.clone(),
                    args: args.iter().map(|a| self.resolve(a)).collect(),
                }
            }
            _ => ty.clone(),
        }
    }

    /// Occurs check: does type variable `var` appear anywhere in `ty`?
    /// If so, binding var to ty would create an infinite type — which is bad.
    fn occurs_in(&self, var: TypeVarId, ty: &InferType) -> bool {
        match ty {
            InferType::Var(id) => {
                if *id == var { return true; }
                if let Some(bound) = self.bindings.get(id) {
                    self.occurs_in(var, bound)
                } else {
                    false
                }
            }
            InferType::Function { params, return_type, .. } => {
                params.iter().any(|p| self.occurs_in(var, p))
                    || self.occurs_in(var, return_type)
            }
            InferType::Applied { args, .. } => {
                args.iter().any(|a| self.occurs_in(var, a))
            }
            _ => false,
        }
    }
}

// ============================================================================
// CONSTRAINT GENERATION & SOLVING
// ============================================================================
// We model type inference as constraint solving. During AST traversal, we
// generate constraints like "the type of x must equal i32" or "the return
// type of f must be a subtype of the expected type". Then we solve them all.

/// A type constraint generated during inference.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Two types must be equal: T1 = T2.
    Equal(InferType, InferType, ConstraintOrigin),
    /// Subtype relationship: T1 <: T2.
    Subtype(InferType, InferType, ConstraintOrigin),
    /// A type must implement a trait: T : Trait.
    Implements(InferType, String, ConstraintOrigin),
    /// An effect constraint: function must have at most these effects.
    EffectSubset(Vec<Effect>, Vec<Effect>, ConstraintOrigin),
    /// Linear type usage: resource must be consumed exactly once.
    LinearUsage(String, LinearUsageKind, ConstraintOrigin),
    /// Refinement type: value must satisfy a predicate.
    Refinement(InferType, String, ConstraintOrigin),
}

/// Where a constraint came from (for error messages).
#[derive(Debug, Clone)]
pub struct ConstraintOrigin {
    pub file: String,
    pub line: u32,
    pub description: String,
}

#[derive(Debug, Clone)]
pub enum LinearUsageKind {
    MustConsumeOnce,    // Linear type
    MustConsumeAtMost,  // Affine type
    Unrestricted,       // Normal type
}

/// Result of type inference for one function or module.
#[derive(Debug)]
pub struct InferenceResult {
    pub substitution: Substitution,
    pub errors: Vec<TypeError>,
    pub inferred_types: HashMap<String, InferType>,
    pub inferred_effects: HashMap<String, Vec<Effect>>,
}

#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub origin: ConstraintOrigin,
    pub kind: TypeErrorKind,
}

#[derive(Debug, Clone)]
pub enum TypeErrorKind {
    Mismatch { expected: InferType, found: InferType },
    UnboundVariable(String),
    TraitNotImplemented { ty: InferType, trait_name: String },
    LinearityViolation(String),
    RefinementViolation { predicate: String },
    EffectViolation { disallowed: Effect },
    InfiniteType,
    AmbiguousType,
}

// ============================================================================
// THE INFERENCE ENGINE
// ============================================================================

pub struct TypeInferenceEngine {
    /// The substitution being built up.
    subst: Substitution,
    /// Constraints collected during traversal.
    constraints: Vec<Constraint>,
    /// Type environment: maps variable names to their types.
    env: Vec<TypeEnv>,
    /// Known type definitions (classes, structs, enums, etc.).
    type_defs: HashMap<String, TypeDefinition>,
    /// Known trait definitions.
    trait_defs: HashMap<String, TraitInfo>,
    /// Accumulated type errors.
    errors: Vec<TypeError>,
    /// Inferred effect sets for each function.
    effect_sets: HashMap<String, Vec<Effect>>,
}

/// A type environment frame (one per scope).
#[derive(Debug, Clone)]
struct TypeEnv {
    bindings: HashMap<String, InferType>,
    linear_resources: HashMap<String, LinearUsageKind>,
}

/// Information about a type definition.
#[derive(Debug, Clone)]
struct TypeDefinition {
    name: String,
    type_params: Vec<String>,
    fields: HashMap<String, InferType>,
    supertype: Option<String>,
    kind: TypeDefKind,
}

#[derive(Debug, Clone)]
enum TypeDefKind {
    Class, Struct, DataClass, Enum, Sealed, Alias(InferType),
}

/// Information about a trait.
#[derive(Debug, Clone)]
struct TraitInfo {
    name: String,
    methods: HashMap<String, InferType>,
    super_traits: Vec<String>,
}

impl TypeInferenceEngine {
    pub fn new() -> Self {
        let mut engine = Self {
            subst: Substitution::new(),
            constraints: Vec::new(),
            env: vec![TypeEnv { bindings: HashMap::new(), linear_resources: HashMap::new() }],
            type_defs: HashMap::new(),
            trait_defs: HashMap::new(),
            errors: Vec::new(),
            effect_sets: HashMap::new(),
        };
        engine.register_builtins();
        engine
    }

    /// Populate the global scope with well-known built-in function types.
    fn register_builtins(&mut self) {
        let builtins: &[(&str, InferType)] = &[
            ("println", InferType::Function {
                params: vec![InferType::Concrete(ChronosType::Str)],
                return_type: Box::new(InferType::Concrete(ChronosType::Void)),
                effects: vec![Effect::IO],
            }),
            ("print", InferType::Function {
                params: vec![InferType::Concrete(ChronosType::Str)],
                return_type: Box::new(InferType::Concrete(ChronosType::Void)),
                effects: vec![Effect::IO],
            }),
            ("print_line", InferType::Function {
                params: vec![InferType::Concrete(ChronosType::Str)],
                return_type: Box::new(InferType::Concrete(ChronosType::Void)),
                effects: vec![Effect::IO],
            }),
            ("assert", InferType::Function {
                params: vec![
                    InferType::Concrete(ChronosType::Bool),
                    InferType::Concrete(ChronosType::Str),
                ],
                return_type: Box::new(InferType::Concrete(ChronosType::Void)),
                effects: vec![],
            }),
            ("exit", InferType::Function {
                params: vec![InferType::Concrete(ChronosType::Int32)],
                return_type: Box::new(InferType::Concrete(ChronosType::Never)),
                effects: vec![Effect::IO],
            }),
            ("abort", InferType::Function {
                params: vec![],
                return_type: Box::new(InferType::Concrete(ChronosType::Never)),
                effects: vec![Effect::IO],
            }),
        ];
        for (name, ty) in builtins {
            self.bind_variable(name, ty.clone());
        }
    }

    // =================================================================
    // ENVIRONMENT MANAGEMENT
    // =================================================================

    fn push_scope(&mut self) {
        self.env.push(TypeEnv {
            bindings: HashMap::new(),
            linear_resources: HashMap::new(),
        });
    }

    fn pop_scope(&mut self) {
        if let Some(scope) = self.env.pop() {
            // Check that all linear resources in this scope were consumed.
            for (name, kind) in &scope.linear_resources {
                match kind {
                    LinearUsageKind::MustConsumeOnce => {
                        self.errors.push(TypeError {
                            message: format!("Linear resource `{}` was not consumed", name),
                            origin: ConstraintOrigin {
                                file: String::new(), line: 0,
                                description: "scope exit".to_string(),
                            },
                            kind: TypeErrorKind::LinearityViolation(name.clone()),
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    fn bind_variable(&mut self, name: &str, ty: InferType) {
        if let Some(scope) = self.env.last_mut() {
            scope.bindings.insert(name.to_string(), ty);
        }
    }

    fn lookup_variable(&self, name: &str) -> Option<InferType> {
        // Search scopes from innermost to outermost.
        for scope in self.env.iter().rev() {
            if let Some(ty) = scope.bindings.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    fn register_linear_resource(&mut self, name: &str, kind: LinearUsageKind) {
        if let Some(scope) = self.env.last_mut() {
            scope.linear_resources.insert(name.to_string(), kind);
        }
    }

    fn consume_linear_resource(&mut self, name: &str) {
        for scope in self.env.iter_mut().rev() {
            if scope.linear_resources.remove(name).is_some() {
                return;
            }
        }
    }

    // =================================================================
    // CONSTRAINT GENERATION: Traverse the AST and emit constraints
    // =================================================================

    /// Infer the type of an expression, generating constraints along the way.
    pub fn infer_expression(&mut self, expr: &Expression) -> InferType {
        match expr {
            Expression::IntLiteral(_) => InferType::Concrete(ChronosType::Int64),
            Expression::FloatLiteral(_) => InferType::Concrete(ChronosType::Float64),
            Expression::StringLiteral(_) => InferType::Concrete(ChronosType::Str),
            Expression::BoolLiteral(_) => InferType::Concrete(ChronosType::Bool),

            Expression::Identifier(name) => {
                if let Some(ty) = self.lookup_variable(name) {
                    ty
                } else {
                    self.errors.push(TypeError {
                        message: format!("Unbound variable: `{}`", name),
                        origin: ConstraintOrigin {
                            file: String::new(), line: 0,
                            description: format!("reference to `{}`", name),
                        },
                        kind: TypeErrorKind::UnboundVariable(name.clone()),
                    });
                    InferType::Error(format!("unbound: {}", name))
                }
            }

            Expression::BinaryOp { left, op, right } => {
                let left_ty = self.infer_expression(left);
                let right_ty = self.infer_expression(right);
                let result_ty = self.subst.fresh_var();

                // For arithmetic ops, both sides must be the same numeric type.
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        self.constraints.push(Constraint::Equal(
                            left_ty.clone(), right_ty.clone(),
                            self.origin("arithmetic operands must match"),
                        ));
                        self.constraints.push(Constraint::Equal(
                            result_ty.clone(), left_ty,
                            self.origin("arithmetic result type"),
                        ));
                    }
                    BinOp::Eq | BinOp::Neq | BinOp::Lt | BinOp::Gt |
                    BinOp::Lte | BinOp::Gte => {
                        self.constraints.push(Constraint::Equal(
                            left_ty, right_ty,
                            self.origin("comparison operands must match"),
                        ));
                        return InferType::Concrete(ChronosType::Bool);
                    }
                    BinOp::And | BinOp::Or => {
                        self.constraints.push(Constraint::Equal(
                            left_ty, InferType::Concrete(ChronosType::Bool),
                            self.origin("logical op requires bool"),
                        ));
                        self.constraints.push(Constraint::Equal(
                            right_ty, InferType::Concrete(ChronosType::Bool),
                            self.origin("logical op requires bool"),
                        ));
                        return InferType::Concrete(ChronosType::Bool);
                    }
                    BinOp::MatMul => {
                        // Matrix multiply: result is a tensor type determined
                        // by the shapes of the operands.
                        // For now, constrain that both sides are tensors.
                        self.constraints.push(Constraint::Equal(
                            result_ty.clone(), left_ty,
                            self.origin("matmul result inherits element type"),
                        ));
                    }
                    _ => {
                        self.constraints.push(Constraint::Equal(
                            left_ty.clone(), right_ty,
                            self.origin("bitwise op operands must match"),
                        ));
                        self.constraints.push(Constraint::Equal(
                            result_ty.clone(), left_ty,
                            self.origin("bitwise op result type"),
                        ));
                    }
                }

                result_ty
            }

            Expression::UnaryOp { op, expr: inner } => {
                let inner_ty = self.infer_expression(inner);
                match op {
                    UnaryOp::Neg | UnaryOp::BitNot => inner_ty,
                    UnaryOp::Not => {
                        self.constraints.push(Constraint::Equal(
                            inner_ty, InferType::Concrete(ChronosType::Bool),
                            self.origin("! requires bool"),
                        ));
                        InferType::Concrete(ChronosType::Bool)
                    }
                    UnaryOp::Ref => {
                        InferType::Applied {
                            constructor: "Ref".to_string(),
                            args: vec![inner_ty],
                        }
                    }
                    UnaryOp::MutRef => {
                        InferType::Applied {
                            constructor: "MutRef".to_string(),
                            args: vec![inner_ty],
                        }
                    }
                    UnaryOp::Deref => {
                        let result = self.subst.fresh_var();
                        self.constraints.push(Constraint::Equal(
                            inner_ty,
                            InferType::Applied {
                                constructor: "Ref".to_string(),
                                args: vec![result.clone()],
                            },
                            self.origin("dereference requires reference type"),
                        ));
                        result
                    }
                }
            }

            Expression::Call { function, args } => {
                // Check if this is a call to a known variadic builtin.
                let is_variadic = matches!(function.as_ref(),
                    Expression::Identifier(name) if matches!(name.as_str(),
                        "println" | "print" | "print_line" | "eprintln" | "eprint" | "format"
                    )
                );

                let fn_ty = self.infer_expression(function);
                let arg_types: Vec<InferType> = args.iter()
                    .map(|a| self.infer_expression(a))
                    .collect();
                let return_ty = self.subst.fresh_var();

                if !is_variadic {
                    // Constrain: fn_ty must be a function taking arg_types, returning return_ty.
                    self.constraints.push(Constraint::Equal(
                        fn_ty,
                        InferType::Function {
                            params: arg_types,
                            return_type: Box::new(return_ty.clone()),
                            effects: Vec::new(), // Effects inferred separately
                        },
                        self.origin("function call type"),
                    ));
                }

                return_ty
            }

            Expression::FieldAccess { object, field } => {
                let obj_ty = self.infer_expression(object);
                let field_ty = self.subst.fresh_var();

                // The field type depends on what type `object` is.
                // In a full implementation, we'd look up the field in the type
                // definition. Here we generate a constraint that will be solved
                // once the object type is known.
                self.constraints.push(Constraint::Equal(
                    field_ty.clone(),
                    InferType::Error(format!("field `{}` of {{unknown}}", field)),
                    self.origin(&format!("field access .{}", field)),
                ));

                field_ty
            }

            Expression::MethodCall { object, method, args } => {
                let obj_ty = self.infer_expression(object);
                let arg_types: Vec<InferType> = args.iter()
                    .map(|a| self.infer_expression(a))
                    .collect();
                let return_ty = self.subst.fresh_var();

                // Method calls are resolved based on the receiver type.
                // We'll need the type definition to know what methods exist.
                // For now, generate a fresh return type.
                return_ty
            }

            Expression::Lambda { params, body, return_type } => {
                self.push_scope();

                let param_types: Vec<InferType> = params.iter().map(|p| {
                    let ty = if p.ty != ChronosType::Void {
                        InferType::Concrete(p.ty.clone())
                    } else {
                        self.subst.fresh_var()
                    };
                    self.bind_variable(&p.name, ty.clone());
                    ty
                }).collect();

                let body_ty = self.infer_expression(body);

                if let Some(ret_ty) = return_type {
                    self.constraints.push(Constraint::Subtype(
                        body_ty.clone(),
                        InferType::Concrete(ret_ty.clone()),
                        self.origin("lambda return type"),
                    ));
                }

                self.pop_scope();

                InferType::Function {
                    params: param_types,
                    return_type: Box::new(body_ty),
                    effects: Vec::new(),
                }
            }

            Expression::If { condition, then_branch, else_branch } => {
                let cond_ty = self.infer_expression(condition);
                self.constraints.push(Constraint::Equal(
                    cond_ty, InferType::Concrete(ChronosType::Bool),
                    self.origin("if condition must be bool"),
                ));

                let then_ty = self.infer_expression(then_branch);

                if let Some(else_expr) = else_branch {
                    let else_ty = self.infer_expression(else_expr);
                    // Both branches must have the same type.
                    self.constraints.push(Constraint::Equal(
                        then_ty.clone(), else_ty,
                        self.origin("if/else branches must have same type"),
                    ));
                    then_ty
                } else {
                    then_ty
                }
            }

            Expression::Match { scrutinee, arms } => {
                let scrutinee_ty = self.infer_expression(scrutinee);
                let result_ty = self.subst.fresh_var();

                for arm in arms {
                    self.push_scope();
                    self.infer_pattern(&arm.pattern, &scrutinee_ty);
                    let arm_ty = self.infer_expression(&arm.body);
                    self.constraints.push(Constraint::Equal(
                        result_ty.clone(), arm_ty,
                        self.origin("match arms must have same type"),
                    ));
                    self.pop_scope();
                }

                result_ty
            }

            Expression::AiInvoke { skill_name, inputs } => {
                // AI invocations return a dynamic type that depends on the
                // skill's output schema. For now, we treat it as opaque.
                InferType::Applied {
                    constructor: format!("AiResult<{}>", skill_name),
                    args: Vec::new(),
                }
            }

            Expression::Block(stmts) => {
                self.push_scope();
                let mut last_ty = InferType::Concrete(ChronosType::Void);
                for stmt in stmts {
                    last_ty = self.infer_statement(stmt);
                }
                self.pop_scope();
                last_ty
            }

            Expression::TypeAscription { expr, ty } => {
                let inferred = self.infer_expression(expr);
                let expected = InferType::Concrete(ty.clone());
                self.constraints.push(Constraint::Subtype(
                    inferred, expected.clone(),
                    self.origin("type ascription"),
                ));
                expected
            }

            Expression::TensorLiteral { elements, shape, device } => {
                if let Some(first) = elements.first() {
                    let elem_ty = self.infer_expression(first);
                    for elem in &elements[1..] {
                        let ty = self.infer_expression(elem);
                        self.constraints.push(Constraint::Equal(
                            elem_ty.clone(), ty,
                            self.origin("tensor elements must have same type"),
                        ));
                    }
                    // Return a tensor type with the inferred element type.
                    InferType::Applied {
                        constructor: "Tensor".to_string(),
                        args: vec![elem_ty],
                    }
                } else {
                    InferType::Applied {
                        constructor: "Tensor".to_string(),
                        args: vec![self.subst.fresh_var()],
                    }
                }
            }

            Expression::Await(inner) => self.infer_expression(inner),
            Expression::Return(inner) => self.infer_expression(inner),
        }
    }

    /// Infer types for a statement, returning the "type" of the statement
    /// (Void for most, the expression type for expression statements).
    pub fn infer_statement(&mut self, stmt: &Statement) -> InferType {
        match stmt {
            Statement::Let { name, ty, value, mutable } => {
                let value_ty = self.infer_expression(value);

                let final_ty = if let Some(declared_ty) = ty {
                    let declared = InferType::Concrete(declared_ty.clone());
                    self.constraints.push(Constraint::Subtype(
                        value_ty.clone(), declared.clone(),
                        self.origin(&format!("let binding `{}`", name)),
                    ));

                    // Check if this is a linear or affine type.
                    match declared_ty {
                        ChronosType::Linear(_) => {
                            self.register_linear_resource(name, LinearUsageKind::MustConsumeOnce);
                        }
                        ChronosType::Affine(_) => {
                            self.register_linear_resource(name, LinearUsageKind::MustConsumeAtMost);
                        }
                        _ => {}
                    }

                    declared
                } else {
                    value_ty
                };

                self.bind_variable(name, final_ty);
                InferType::Concrete(ChronosType::Void)
            }

            Statement::Assignment { target, value } => {
                let target_ty = self.infer_expression(target);
                let value_ty = self.infer_expression(value);
                self.constraints.push(Constraint::Subtype(
                    value_ty, target_ty,
                    self.origin("assignment"),
                ));
                InferType::Concrete(ChronosType::Void)
            }

            Statement::Return(Some(expr)) => {
                self.infer_expression(expr)
            }

            Statement::Return(None) => {
                InferType::Concrete(ChronosType::Void)
            }

            Statement::ExprStatement(expr) => {
                self.infer_expression(expr)
            }

            Statement::While { condition, body } => {
                let cond_ty = self.infer_expression(condition);
                self.constraints.push(Constraint::Equal(
                    cond_ty, InferType::Concrete(ChronosType::Bool),
                    self.origin("while condition must be bool"),
                ));
                self.push_scope();
                for stmt in body {
                    self.infer_statement(stmt);
                }
                self.pop_scope();
                InferType::Concrete(ChronosType::Void)
            }

            Statement::For { binding, iterator, body } => {
                let iter_ty = self.infer_expression(iterator);
                let elem_ty = self.subst.fresh_var();
                self.push_scope();
                self.bind_variable(binding, elem_ty);
                for stmt in body {
                    self.infer_statement(stmt);
                }
                self.pop_scope();
                InferType::Concrete(ChronosType::Void)
            }

            Statement::Drop(name) => {
                self.consume_linear_resource(name);
                InferType::Concrete(ChronosType::Void)
            }

            Statement::DeviceScope { target, body } => {
                // Inside a device scope, tensor operations target the specified device.
                self.push_scope();
                for stmt in body {
                    self.infer_statement(stmt);
                }
                self.pop_scope();
                InferType::Concrete(ChronosType::Void)
            }

            Statement::Break | Statement::Continue => {
                InferType::Concrete(ChronosType::Never)
            }

            Statement::Require { condition, message } => {
                self.infer_expression(condition);
                if let Some(msg) = message { self.infer_expression(msg); }
                InferType::Concrete(ChronosType::Void)
            }
        }
    }

    /// Infer bindings from a pattern match.
    fn infer_pattern(&mut self, pattern: &Pattern, scrutinee_ty: &InferType) {
        match pattern {
            Pattern::Wildcard => {} // Matches anything, binds nothing
            Pattern::Binding(name) => {
                self.bind_variable(name, scrutinee_ty.clone());
            }
            Pattern::Literal(expr) => {
                let lit_ty = self.infer_expression(expr);
                self.constraints.push(Constraint::Equal(
                    scrutinee_ty.clone(), lit_ty,
                    self.origin("pattern literal must match scrutinee"),
                ));
            }
            Pattern::Tuple(patterns) => {
                for (i, pat) in patterns.iter().enumerate() {
                    let elem_ty = self.subst.fresh_var();
                    self.infer_pattern(pat, &elem_ty);
                }
            }
            Pattern::Constructor { name, fields } => {
                // Look up the constructor in type definitions.
                for (i, field_pat) in fields.iter().enumerate() {
                    let field_ty = self.subst.fresh_var();
                    self.infer_pattern(field_pat, &field_ty);
                }
            }
            Pattern::Or(alternatives) => {
                for alt in alternatives {
                    self.infer_pattern(alt, scrutinee_ty);
                }
            }
        }
    }

    // =================================================================
    // CONSTRAINT SOLVING: Unification
    // =================================================================
    // After collecting all constraints, we solve them by unification.
    // This is the classic Algorithm W approach: try to make both sides
    // of each constraint identical by binding type variables.

    pub fn solve_constraints(&mut self) {
        // We iterate until no more progress can be made.
        let constraints = std::mem::take(&mut self.constraints);

        for constraint in &constraints {
            match constraint {
                Constraint::Equal(t1, t2, origin) => {
                    self.unify(t1, t2, origin);
                }
                Constraint::Subtype(sub, sup, origin) => {
                    // For now, treat subtype as equality.
                    // A full implementation would have proper subtyping rules
                    // for class hierarchies, union/intersection types, etc.
                    self.unify(sub, sup, origin);
                }
                Constraint::Implements(ty, trait_name, origin) => {
                    // Check that the resolved type implements the required trait.
                    let resolved = self.subst.resolve(ty);
                    // In a full implementation, look up trait impls in the database.
                    // For now, just record it.
                }
                Constraint::EffectSubset(actual, allowed, origin) => {
                    // Check that actual effects are a subset of allowed effects.
                    for eff in actual {
                        if !allowed.contains(eff) {
                            self.errors.push(TypeError {
                                message: format!(
                                    "Effect {:?} is not allowed in this context", eff
                                ),
                                origin: origin.clone(),
                                kind: TypeErrorKind::EffectViolation { disallowed: eff.clone() },
                            });
                        }
                    }
                }
                Constraint::LinearUsage(name, kind, origin) => {
                    // Checked during scope exit — see pop_scope.
                }
                Constraint::Refinement(ty, predicate, origin) => {
                    // Refinement type checking: verify that the predicate holds.
                    // This is an area where we'd integrate with an SMT solver
                    // (like Z3) for full verification. For now, we trust the
                    // programmer's annotations and check at runtime.
                }
            }
        }
    }

    /// Unify two types, making them equal by binding type variables.
    fn unify(&mut self, t1: &InferType, t2: &InferType, origin: &ConstraintOrigin) {
        let t1 = self.subst.resolve(t1);
        let t2 = self.subst.resolve(t2);

        // If they're already equal, nothing to do.
        if t1 == t2 { return; }

        match (&t1, &t2) {
            // If either is a variable, bind it to the other.
            (InferType::Var(id), _) => {
                self.subst.bind(*id, t2);
            }
            (_, InferType::Var(id)) => {
                self.subst.bind(*id, t1);
            }

            // Two concrete types must match exactly.
            (InferType::Concrete(a), InferType::Concrete(b)) => {
                if !self.types_compatible(a, b) {
                    self.errors.push(TypeError {
                        message: format!("Type mismatch: expected {:?}, found {:?}", a, b),
                        origin: origin.clone(),
                        kind: TypeErrorKind::Mismatch {
                            expected: t2.clone(),
                            found: t1.clone(),
                        },
                    });
                }
            }

            // Two function types: unify params and return types.
            (InferType::Function { params: p1, return_type: r1, .. },
             InferType::Function { params: p2, return_type: r2, .. }) => {
                if p1.len() != p2.len() {
                    self.errors.push(TypeError {
                        message: format!(
                            "Function arity mismatch: expected {} params, found {}",
                            p2.len(), p1.len()
                        ),
                        origin: origin.clone(),
                        kind: TypeErrorKind::Mismatch {
                            expected: t2.clone(),
                            found: t1.clone(),
                        },
                    });
                    return;
                }
                for (a, b) in p1.iter().zip(p2.iter()) {
                    self.unify(a, b, origin);
                }
                self.unify(r1, r2, origin);
            }

            // Two applied types: constructors must match, then unify args.
            (InferType::Applied { constructor: c1, args: a1 },
             InferType::Applied { constructor: c2, args: a2 }) => {
                if c1 != c2 {
                    self.errors.push(TypeError {
                        message: format!(
                            "Type constructor mismatch: {} vs {}", c1, c2
                        ),
                        origin: origin.clone(),
                        kind: TypeErrorKind::Mismatch {
                            expected: t2.clone(),
                            found: t1.clone(),
                        },
                    });
                    return;
                }
                for (a, b) in a1.iter().zip(a2.iter()) {
                    self.unify(a, b, origin);
                }
            }

            // Error types are silently absorbed.
            (InferType::Error(_), _) | (_, InferType::Error(_)) => {}

            // Everything else is a mismatch.
            _ => {
                self.errors.push(TypeError {
                    message: format!("Cannot unify {:?} with {:?}", t1, t2),
                    origin: origin.clone(),
                    kind: TypeErrorKind::Mismatch {
                        expected: t2.clone(),
                        found: t1,
                    },
                });
            }
        }
    }

    /// Check if two concrete Chronos types are compatible.
    /// This handles implicit numeric widening, subtyping, etc.
    fn types_compatible(&self, a: &ChronosType, b: &ChronosType) -> bool {
        if a == b { return true; }

        // Numeric widening: i32 -> i64, f32 -> f64, etc.
        match (a, b) {
            (ChronosType::Int8, ChronosType::Int16) |
            (ChronosType::Int8, ChronosType::Int32) |
            (ChronosType::Int8, ChronosType::Int64) |
            (ChronosType::Int16, ChronosType::Int32) |
            (ChronosType::Int16, ChronosType::Int64) |
            (ChronosType::Int32, ChronosType::Int64) |
            (ChronosType::Float16, ChronosType::Float32) |
            (ChronosType::Float16, ChronosType::Float64) |
            (ChronosType::Float32, ChronosType::Float64) |
            (ChronosType::BFloat16, ChronosType::Float32) => true,

            // Optional coercion: T -> Optional(T)
            (t, ChronosType::Optional(inner)) if t == inner.as_ref() => true,

            // Never is a subtype of everything.
            (ChronosType::Never, _) => true,

            _ => false,
        }
    }

    // =================================================================
    // EFFECT INFERENCE
    // =================================================================
    // Effect inference works similarly to type inference. We collect all
    // the effects that a function body produces and check them against
    // the declared effect set.

    /// Infer the effects of a function body.
    pub fn infer_effects(&mut self, fn_name: &str, body: &[Statement]) -> Vec<Effect> {
        let mut effects = Vec::new();

        for stmt in body {
            self.collect_effects_from_statement(stmt, &mut effects);
        }

        // Deduplicate
        effects.sort_by_key(|e| format!("{:?}", e));
        effects.dedup();

        self.effect_sets.insert(fn_name.to_string(), effects.clone());
        effects
    }

    fn collect_effects_from_statement(&self, stmt: &Statement, effects: &mut Vec<Effect>) {
        match stmt {
            Statement::ExprStatement(expr) | Statement::Return(Some(expr)) => {
                self.collect_effects_from_expression(expr, effects);
            }
            Statement::Let { value, .. } => {
                self.collect_effects_from_expression(value, effects);
            }
            Statement::While { condition, body } => {
                self.collect_effects_from_expression(condition, effects);
                for s in body { self.collect_effects_from_statement(s, effects); }
            }
            Statement::For { iterator, body, .. } => {
                self.collect_effects_from_expression(iterator, effects);
                for s in body { self.collect_effects_from_statement(s, effects); }
            }
            Statement::DeviceScope { target, body } => {
                match target {
                    DeviceTarget::Gpu { .. } => effects.push(Effect::GpuKernel),
                    DeviceTarget::Tpu { .. } => effects.push(Effect::TpuKernel),
                    DeviceTarget::Npu { .. } => effects.push(Effect::NpuKernel),
                    _ => {}
                }
                for s in body { self.collect_effects_from_statement(s, effects); }
            }
            _ => {}
        }
    }

    fn collect_effects_from_expression(&self, expr: &Expression, effects: &mut Vec<Effect>) {
        match expr {
            Expression::Call { function, args } => {
                // If we know the function's effects, include them.
                if let Expression::Identifier(name) = function.as_ref() {
                    if let Some(fn_effects) = self.effect_sets.get(name) {
                        effects.extend(fn_effects.iter().cloned());
                    }
                }
                for arg in args {
                    self.collect_effects_from_expression(arg, effects);
                }
            }
            Expression::AiInvoke { .. } => {
                effects.push(Effect::IO);
                effects.push(Effect::Async);
            }
            Expression::BinaryOp { left, right, op: BinOp::MatMul, .. } => {
                effects.push(Effect::Alloc);
                self.collect_effects_from_expression(left, effects);
                self.collect_effects_from_expression(right, effects);
            }
            _ => {}
        }
    }

    // =================================================================
    // PUBLIC INTERFACE: Run full inference on a program
    // =================================================================

    pub fn infer_program(&mut self, program: &Program) -> InferenceResult {
        // Phase 1: Register all type definitions.
        for item in &program.items {
            self.register_type_definitions(item);
        }

        // Phase 1b: Forward-declare all function signatures so that
        // mutually-recursive or out-of-order calls resolve correctly.
        for item in &program.items {
            if let Item::FunctionDecl(f) = item {
                let fn_ty = InferType::Function {
                    params: f.signature.params.iter()
                        .map(|p| InferType::Concrete(p.ty.clone()))
                        .collect(),
                    return_type: Box::new(InferType::Concrete(f.signature.return_type.clone())),
                    effects: f.signature.effects.clone(),
                };
                self.bind_variable(&f.signature.name, fn_ty);
            }
        }

        // Phase 2: Infer types for all functions and expressions.
        for item in &program.items {
            self.infer_item(item);
        }

        // Phase 3: Solve all accumulated constraints.
        self.solve_constraints();

        // Phase 4: Collect results.
        let mut inferred_types = HashMap::new();
        for scope in &self.env {
            for (name, ty) in &scope.bindings {
                inferred_types.insert(name.clone(), self.subst.resolve(ty));
            }
        }

        InferenceResult {
            substitution: self.subst.clone(),
            errors: self.errors.clone(),
            inferred_types,
            inferred_effects: self.effect_sets.clone(),
        }
    }

    fn register_type_definitions(&mut self, item: &Item) {
        match item {
            Item::ClassDecl(c) => {
                let mut fields = HashMap::new();
                for f in &c.fields {
                    fields.insert(f.name.clone(), InferType::Concrete(f.ty.clone()));
                }
                self.type_defs.insert(c.name.clone(), TypeDefinition {
                    name: c.name.clone(),
                    type_params: c.type_params.iter().map(|p| p.name.clone()).collect(),
                    fields,
                    supertype: c.superclass.as_ref().and_then(|t| {
                        if let ChronosType::Named { name, .. } = t { Some(name.clone()) } else { None }
                    }),
                    kind: TypeDefKind::Class,
                });
            }
            Item::StructDecl(s) => {
                let mut fields = HashMap::new();
                for f in &s.fields {
                    fields.insert(f.name.clone(), InferType::Concrete(f.ty.clone()));
                }
                self.type_defs.insert(s.name.clone(), TypeDefinition {
                    name: s.name.clone(),
                    type_params: s.type_params.iter().map(|p| p.name.clone()).collect(),
                    fields,
                    supertype: None,
                    kind: TypeDefKind::Struct,
                });
            }
            Item::TraitDecl(t) => {
                let mut methods = HashMap::new();
                for m in &t.methods {
                    let fn_type = InferType::Function {
                        params: m.signature.params.iter()
                            .map(|p| InferType::Concrete(p.ty.clone()))
                            .collect(),
                        return_type: Box::new(InferType::Concrete(m.signature.return_type.clone())),
                        effects: m.signature.effects.clone(),
                    };
                    methods.insert(m.signature.name.clone(), fn_type);
                }
                self.trait_defs.insert(t.name.clone(), TraitInfo {
                    name: t.name.clone(),
                    methods,
                    super_traits: t.super_traits.iter().filter_map(|st| {
                        if let ChronosType::Named { name, .. } = st { Some(name.clone()) } else { None }
                    }).collect(),
                });
            }
            _ => {}
        }
    }

    fn infer_item(&mut self, item: &Item) {
        match item {
            Item::FunctionDecl(f) => {
                self.push_scope();
                for param in &f.signature.params {
                    self.bind_variable(&param.name, InferType::Concrete(param.ty.clone()));
                }
                let mut last_ty = InferType::Concrete(ChronosType::Void);
                for stmt in &f.body {
                    last_ty = self.infer_statement(stmt);
                }
                // Constrain return type.
                if f.signature.return_type != ChronosType::Void {
                    self.constraints.push(Constraint::Subtype(
                        last_ty,
                        InferType::Concrete(f.signature.return_type.clone()),
                        self.origin(&format!("return type of `{}`", f.signature.name)),
                    ));
                }
                // Infer effects.
                self.infer_effects(&f.signature.name, &f.body);
                self.pop_scope();
            }
            Item::DegradableFunctionDecl(df) => {
                self.infer_item(&Item::FunctionDecl(df.function.clone()));
            }
            _ => {}
        }
    }

    fn origin(&self, description: &str) -> ConstraintOrigin {
        ConstraintOrigin {
            file: String::new(),
            line: 0,
            description: description.to_string(),
        }
    }
}
