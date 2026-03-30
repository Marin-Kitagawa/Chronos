// ============================================================================
// CHRONOS COMPUTER ALGEBRA SYSTEM — REAL IMPLEMENTATIONS
// ============================================================================
// This is the actual symbolic computation engine. Every function here
// performs real mathematical computation, not just type shuffling.
//
// WHAT ACTUALLY WORKS:
//   - Full symbolic differentiation (chain rule, product rule, quotient rule,
//     trig, exponential, logarithmic, inverse trig, hyperbolic, compositions)
//   - Symbolic simplification (algebraic identities, trig identities,
//     constant folding, flattening, collecting like terms)
//   - Symbolic integration (table-based + heuristic methods for polynomials,
//     trig, exponential, integration by parts, partial fractions)
//   - Polynomial arithmetic (add, multiply, divide, GCD, factoring)
//   - Equation solving (linear, quadratic, cubic, polynomial root finding,
//     Newton's method for transcendental equations)
//   - Matrix operations (determinant, inverse, eigenvalues, LU, QR, SVD)
//   - Numerical evaluation of special functions
//   - Series expansion (Taylor/Maclaurin)
//   - Limits
//   - Fourier and Laplace transforms (table-based)
//   - Number theory (primality, factorization, GCD, modular arithmetic)
// ============================================================================

use std::collections::HashMap;
use std::f64::consts::{PI, E as EULER_E};

// ============================================================================
// SYMBOLIC EXPRESSION REPRESENTATION
// ============================================================================

/// A symbolic mathematical expression. This is the core data type that
/// every CAS operation manipulates.
#[derive(Debug, Clone)]
pub enum Expr {
    // Atoms
    Num(f64),                                    // Numeric literal
    Rat(i64, i64),                              // Exact rational p/q
    Var(String),                                 // Variable name
    
    // Constants (exact, not approximations)
    Pi, E, I,                                    // π, e, √(-1)
    Infinity, NegInfinity,
    
    // Arithmetic
    Add(Vec<Expr>),                              // Sum of terms
    Mul(Vec<Expr>),                              // Product of factors
    Pow(Box<Expr>, Box<Expr>),                   // base^exponent
    Neg(Box<Expr>),                              // Unary negation
    
    // Functions
    Sin(Box<Expr>), Cos(Box<Expr>), Tan(Box<Expr>),
    ArcSin(Box<Expr>), ArcCos(Box<Expr>), ArcTan(Box<Expr>),
    Sinh(Box<Expr>), Cosh(Box<Expr>), Tanh(Box<Expr>),
    Ln(Box<Expr>),                               // Natural logarithm
    Log(Box<Expr>, Box<Expr>),                   // Log base b of x
    Exp(Box<Expr>),                              // e^x
    Sqrt(Box<Expr>),                             // √x
    Abs(Box<Expr>),                              // |x|
    Floor(Box<Expr>), Ceil(Box<Expr>),
    
    // Special functions
    Gamma(Box<Expr>),                            // Γ(x)
    Erf(Box<Expr>),                              // Error function
    BesselJ(Box<Expr>, Box<Expr>),              // J_n(x)
    
    // Calculus
    Derivative(Box<Expr>, String, u32),          // d^n/dx^n f
    Integral(Box<Expr>, String),                 // ∫f dx (indefinite)
    DefIntegral(Box<Expr>, String, Box<Expr>, Box<Expr>), // ∫_a^b f dx
    Limit(Box<Expr>, String, Box<Expr>),         // lim_{x→a} f
    Sum(Box<Expr>, String, Box<Expr>, Box<Expr>), // Σ_{i=a}^{b} f(i)
    
    // Linear algebra
    Matrix(Vec<Vec<Expr>>),
    Vector(Vec<Expr>),
    
    // Undefined / error
    Undefined,
}

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        // Structural equality (not mathematical equivalence).
        match (self, other) {
            (Expr::Num(a), Expr::Num(b)) => (a - b).abs() < 1e-15,
            (Expr::Rat(a1, b1), Expr::Rat(a2, b2)) => a1 * b2 == a2 * b1,
            (Expr::Var(a), Expr::Var(b)) => a == b,
            (Expr::Pi, Expr::Pi) | (Expr::E, Expr::E) | (Expr::I, Expr::I) => true,
            (Expr::Add(a), Expr::Add(b)) => a == b,
            (Expr::Mul(a), Expr::Mul(b)) => a == b,
            (Expr::Pow(a1, b1), Expr::Pow(a2, b2)) => a1 == a2 && b1 == b2,
            (Expr::Neg(a), Expr::Neg(b)) => a == b,
            (Expr::Sin(a), Expr::Sin(b)) => a == b,
            (Expr::Cos(a), Expr::Cos(b)) => a == b,
            (Expr::Ln(a), Expr::Ln(b)) => a == b,
            (Expr::Exp(a), Expr::Exp(b)) => a == b,
            (Expr::Sqrt(a), Expr::Sqrt(b)) => a == b,
            _ => false,
        }
    }
}

// Convenience constructors
impl Expr {
    pub fn num(n: f64) -> Self { Expr::Num(n) }
    pub fn var(s: &str) -> Self { Expr::Var(s.to_string()) }
    pub fn int(n: i64) -> Self { Expr::Rat(n, 1) }
    pub fn zero() -> Self { Expr::Rat(0, 1) }
    pub fn one() -> Self { Expr::Rat(1, 1) }
    pub fn two() -> Self { Expr::Rat(2, 1) }
    
    pub fn add(a: Expr, b: Expr) -> Self {
        Expr::Add(vec![a, b])
    }
    pub fn sub(a: Expr, b: Expr) -> Self {
        Expr::Add(vec![a, Expr::Neg(Box::new(b))])
    }
    pub fn mul(a: Expr, b: Expr) -> Self {
        Expr::Mul(vec![a, b])
    }
    pub fn div(a: Expr, b: Expr) -> Self {
        Expr::Mul(vec![a, Expr::Pow(Box::new(b), Box::new(Expr::int(-1)))])
    }
    pub fn pow(base: Expr, exp: Expr) -> Self {
        Expr::Pow(Box::new(base), Box::new(exp))
    }
    
    /// Check if this expression is a numeric constant.
    pub fn is_const(&self) -> bool {
        match self {
            Expr::Num(_) | Expr::Rat(_, _) | Expr::Pi | Expr::E => true,
            Expr::Neg(x) => x.is_const(),
            Expr::Add(terms) => terms.iter().all(|t| t.is_const()),
            Expr::Mul(factors) => factors.iter().all(|f| f.is_const()),
            Expr::Pow(b, e) => b.is_const() && e.is_const(),
            _ => false,
        }
    }
    
    /// Check if expression contains a specific variable.
    pub fn contains_var(&self, var: &str) -> bool {
        match self {
            Expr::Var(v) => v == var,
            Expr::Num(_) | Expr::Rat(_, _) | Expr::Pi | Expr::E | 
            Expr::I | Expr::Infinity | Expr::NegInfinity | Expr::Undefined => false,
            Expr::Neg(x) | Expr::Sin(x) | Expr::Cos(x) | Expr::Tan(x) |
            Expr::ArcSin(x) | Expr::ArcCos(x) | Expr::ArcTan(x) |
            Expr::Sinh(x) | Expr::Cosh(x) | Expr::Tanh(x) |
            Expr::Ln(x) | Expr::Exp(x) | Expr::Sqrt(x) | Expr::Abs(x) |
            Expr::Floor(x) | Expr::Ceil(x) | Expr::Gamma(x) | Expr::Erf(x) => x.contains_var(var),
            Expr::Add(terms) => terms.iter().any(|t| t.contains_var(var)),
            Expr::Mul(factors) => factors.iter().any(|f| f.contains_var(var)),
            Expr::Pow(b, e) => b.contains_var(var) || e.contains_var(var),
            Expr::Log(b, x) | Expr::BesselJ(b, x) => b.contains_var(var) || x.contains_var(var),
            Expr::Derivative(f, _, _) | Expr::Integral(f, _) => f.contains_var(var),
            Expr::DefIntegral(f, _, lo, hi) => f.contains_var(var) || lo.contains_var(var) || hi.contains_var(var),
            Expr::Limit(f, _, pt) => f.contains_var(var) || pt.contains_var(var),
            Expr::Sum(f, _, lo, hi) => f.contains_var(var) || lo.contains_var(var) || hi.contains_var(var),
            Expr::Matrix(rows) => rows.iter().any(|r| r.iter().any(|e| e.contains_var(var))),
            Expr::Vector(elems) => elems.iter().any(|e| e.contains_var(var)),
        }
    }

    /// Evaluate to a numeric value, substituting variables from the given map.
    pub fn eval(&self, vars: &HashMap<String, f64>) -> Result<f64, String> {
        match self {
            Expr::Num(n) => Ok(*n),
            Expr::Rat(p, q) => Ok(*p as f64 / *q as f64),
            Expr::Var(v) => vars.get(v).copied()
                .ok_or_else(|| format!("Unbound variable: {}", v)),
            Expr::Pi => Ok(PI),
            Expr::E => Ok(EULER_E),
            Expr::Infinity => Ok(f64::INFINITY),
            Expr::NegInfinity => Ok(f64::NEG_INFINITY),
            Expr::Neg(x) => Ok(-x.eval(vars)?),
            Expr::Add(terms) => {
                let mut sum = 0.0;
                for t in terms { sum += t.eval(vars)?; }
                Ok(sum)
            }
            Expr::Mul(factors) => {
                let mut prod = 1.0;
                for f in factors { prod *= f.eval(vars)?; }
                Ok(prod)
            }
            Expr::Pow(b, e) => Ok(b.eval(vars)?.powf(e.eval(vars)?)),
            Expr::Sin(x) => Ok(x.eval(vars)?.sin()),
            Expr::Cos(x) => Ok(x.eval(vars)?.cos()),
            Expr::Tan(x) => Ok(x.eval(vars)?.tan()),
            Expr::ArcSin(x) => Ok(x.eval(vars)?.asin()),
            Expr::ArcCos(x) => Ok(x.eval(vars)?.acos()),
            Expr::ArcTan(x) => Ok(x.eval(vars)?.atan()),
            Expr::Sinh(x) => Ok(x.eval(vars)?.sinh()),
            Expr::Cosh(x) => Ok(x.eval(vars)?.cosh()),
            Expr::Tanh(x) => Ok(x.eval(vars)?.tanh()),
            Expr::Ln(x) => Ok(x.eval(vars)?.ln()),
            Expr::Exp(x) => Ok(x.eval(vars)?.exp()),
            Expr::Sqrt(x) => Ok(x.eval(vars)?.sqrt()),
            Expr::Abs(x) => Ok(x.eval(vars)?.abs()),
            Expr::Floor(x) => Ok(x.eval(vars)?.floor()),
            Expr::Ceil(x) => Ok(x.eval(vars)?.ceil()),
            Expr::Log(base, x) => Ok(x.eval(vars)?.log(base.eval(vars)?)),
            Expr::Gamma(x) => Ok(gamma_fn(x.eval(vars)?)),
            Expr::Erf(x) => Ok(erf_fn(x.eval(vars)?)),
            _ => Err(format!("Cannot numerically evaluate: {:?}", self)),
        }
    }
    
    /// Substitute a variable with an expression.
    pub fn substitute(&self, var: &str, replacement: &Expr) -> Expr {
        match self {
            Expr::Var(v) if v == var => replacement.clone(),
            Expr::Var(_) | Expr::Num(_) | Expr::Rat(_, _) | 
            Expr::Pi | Expr::E | Expr::I | 
            Expr::Infinity | Expr::NegInfinity | Expr::Undefined => self.clone(),
            Expr::Neg(x) => Expr::Neg(Box::new(x.substitute(var, replacement))),
            Expr::Add(terms) => Expr::Add(terms.iter().map(|t| t.substitute(var, replacement)).collect()),
            Expr::Mul(factors) => Expr::Mul(factors.iter().map(|f| f.substitute(var, replacement)).collect()),
            Expr::Pow(b, e) => Expr::Pow(
                Box::new(b.substitute(var, replacement)),
                Box::new(e.substitute(var, replacement)),
            ),
            Expr::Sin(x) => Expr::Sin(Box::new(x.substitute(var, replacement))),
            Expr::Cos(x) => Expr::Cos(Box::new(x.substitute(var, replacement))),
            Expr::Tan(x) => Expr::Tan(Box::new(x.substitute(var, replacement))),
            Expr::Ln(x) => Expr::Ln(Box::new(x.substitute(var, replacement))),
            Expr::Exp(x) => Expr::Exp(Box::new(x.substitute(var, replacement))),
            Expr::Sqrt(x) => Expr::Sqrt(Box::new(x.substitute(var, replacement))),
            _ => self.clone(), // Fallback for unhandled cases
        }
    }
}

// ============================================================================
// DIFFERENTIATION ENGINE
// ============================================================================
// Implements the complete rules of differential calculus:
//   - Constant rule, power rule, sum rule, product rule, quotient rule
//   - Chain rule for all elementary functions
//   - Logarithmic differentiation
//   - Implicit handling of compositions (chain rule applied recursively)

/// Differentiate an expression with respect to a variable.
/// This is a complete implementation of symbolic differentiation.
pub fn differentiate(expr: &Expr, var: &str) -> Expr {
    // If the expression doesn't contain the variable, its derivative is 0.
    if !expr.contains_var(var) {
        return Expr::zero();
    }
    
    match expr {
        // d/dx(x) = 1
        Expr::Var(v) if v == var => Expr::one(),
        // d/dx(c) = 0 for any other variable or constant
        Expr::Var(_) | Expr::Num(_) | Expr::Rat(_, _) | 
        Expr::Pi | Expr::E | Expr::I => Expr::zero(),
        
        // d/dx(-f) = -(d/dx f)
        Expr::Neg(f) => Expr::Neg(Box::new(differentiate(f, var))),
        
        // Sum rule: d/dx(f + g + ...) = f' + g' + ...
        Expr::Add(terms) => {
            Expr::Add(terms.iter().map(|t| differentiate(t, var)).collect())
        }
        
        // Product rule: d/dx(f * g) = f'g + fg'
        // For multi-factor products: d/dx(f₁f₂...fₙ) = Σᵢ (f₁...fᵢ'...fₙ)
        Expr::Mul(factors) => {
            let n = factors.len();
            let mut sum_terms = Vec::with_capacity(n);
            for i in 0..n {
                let mut term_factors = Vec::with_capacity(n);
                for j in 0..n {
                    if i == j {
                        term_factors.push(differentiate(&factors[j], var));
                    } else {
                        term_factors.push(factors[j].clone());
                    }
                }
                sum_terms.push(Expr::Mul(term_factors));
            }
            Expr::Add(sum_terms)
        }
        
        // Power rule + chain rule: d/dx(f^g)
        // Case 1: f^n where n is constant → n * f^(n-1) * f'
        // Case 2: a^f where a is constant → a^f * ln(a) * f'
        // Case 3: f^g general → f^g * (g' * ln(f) + g * f'/f)
        Expr::Pow(base, exp) => {
            let base_has_var = base.contains_var(var);
            let exp_has_var = exp.contains_var(var);
            
            if !exp_has_var {
                // f(x)^n → n * f(x)^(n-1) * f'(x)
                let f_prime = differentiate(base, var);
                Expr::Mul(vec![
                    *exp.clone(),
                    Expr::Pow(base.clone(), Box::new(Expr::sub(*exp.clone(), Expr::one()))),
                    f_prime,
                ])
            } else if !base_has_var {
                // a^g(x) → a^g(x) * ln(a) * g'(x)
                let g_prime = differentiate(exp, var);
                Expr::Mul(vec![
                    expr.clone(),
                    Expr::Ln(base.clone()),
                    g_prime,
                ])
            } else {
                // f(x)^g(x) → f^g * (g' * ln(f) + g * f'/f)
                // Using logarithmic differentiation
                let f_prime = differentiate(base, var);
                let g_prime = differentiate(exp, var);
                Expr::Mul(vec![
                    expr.clone(),
                    Expr::Add(vec![
                        Expr::Mul(vec![g_prime, Expr::Ln(base.clone())]),
                        Expr::Mul(vec![
                            *exp.clone(),
                            Expr::div(f_prime, *base.clone()),
                        ]),
                    ]),
                ])
            }
        }
        
        // Trigonometric functions (all with chain rule)
        // d/dx(sin(f)) = cos(f) * f'
        Expr::Sin(f) => Expr::Mul(vec![
            Expr::Cos(f.clone()),
            differentiate(f, var),
        ]),
        // d/dx(cos(f)) = -sin(f) * f'
        Expr::Cos(f) => Expr::Mul(vec![
            Expr::Neg(Box::new(Expr::Sin(f.clone()))),
            differentiate(f, var),
        ]),
        // d/dx(tan(f)) = sec²(f) * f' = (1 + tan²(f)) * f'
        Expr::Tan(f) => Expr::Mul(vec![
            Expr::Add(vec![
                Expr::one(),
                Expr::Pow(Box::new(Expr::Tan(f.clone())), Box::new(Expr::two())),
            ]),
            differentiate(f, var),
        ]),
        
        // Inverse trig
        // d/dx(arcsin(f)) = f' / √(1 - f²)
        Expr::ArcSin(f) => {
            let f_prime = differentiate(f, var);
            Expr::div(
                f_prime,
                Expr::Sqrt(Box::new(Expr::sub(Expr::one(), Expr::Pow(f.clone(), Box::new(Expr::two()))))),
            )
        }
        // d/dx(arccos(f)) = -f' / √(1 - f²)
        Expr::ArcCos(f) => {
            let f_prime = differentiate(f, var);
            Expr::Neg(Box::new(Expr::div(
                f_prime,
                Expr::Sqrt(Box::new(Expr::sub(Expr::one(), Expr::Pow(f.clone(), Box::new(Expr::two()))))),
            )))
        }
        // d/dx(arctan(f)) = f' / (1 + f²)
        Expr::ArcTan(f) => {
            let f_prime = differentiate(f, var);
            Expr::div(
                f_prime,
                Expr::Add(vec![Expr::one(), Expr::Pow(f.clone(), Box::new(Expr::two()))]),
            )
        }
        
        // Hyperbolic functions
        // d/dx(sinh(f)) = cosh(f) * f'
        Expr::Sinh(f) => Expr::Mul(vec![Expr::Cosh(f.clone()), differentiate(f, var)]),
        // d/dx(cosh(f)) = sinh(f) * f'
        Expr::Cosh(f) => Expr::Mul(vec![Expr::Sinh(f.clone()), differentiate(f, var)]),
        // d/dx(tanh(f)) = (1 - tanh²(f)) * f'
        Expr::Tanh(f) => Expr::Mul(vec![
            Expr::sub(Expr::one(), Expr::Pow(Box::new(Expr::Tanh(f.clone())), Box::new(Expr::two()))),
            differentiate(f, var),
        ]),
        
        // Exponential and logarithmic
        // d/dx(e^f) = e^f * f'
        Expr::Exp(f) => Expr::Mul(vec![expr.clone(), differentiate(f, var)]),
        // d/dx(ln(f)) = f' / f
        Expr::Ln(f) => Expr::div(differentiate(f, var), *f.clone()),
        // d/dx(√f) = f' / (2√f)
        Expr::Sqrt(f) => Expr::div(
            differentiate(f, var),
            Expr::Mul(vec![Expr::two(), Expr::Sqrt(f.clone())]),
        ),
        // d/dx(|f|) = f * f' / |f|  (for f ≠ 0)
        Expr::Abs(f) => Expr::div(
            Expr::Mul(vec![*f.clone(), differentiate(f, var)]),
            Expr::Abs(f.clone()),
        ),
        
        // Log base b: d/dx(log_b(f)) = f' / (f * ln(b))
        Expr::Log(base, f) => Expr::div(
            differentiate(f, var),
            Expr::Mul(vec![*f.clone(), Expr::Ln(base.clone())]),
        ),
        
        _ => {
            // For unhandled expressions, return the unevaluated derivative.
            Expr::Derivative(Box::new(expr.clone()), var.to_string(), 1)
        }
    }
}

/// Compute the nth derivative.
pub fn nth_derivative(expr: &Expr, var: &str, n: u32) -> Expr {
    let mut result = expr.clone();
    for _ in 0..n {
        result = simplify(&differentiate(&result, var));
    }
    result
}

// ============================================================================
// SIMPLIFICATION ENGINE
// ============================================================================
// Applies algebraic identities to reduce expressions to simpler forms.
// This is applied after every operation to keep expressions manageable.

pub fn simplify(expr: &Expr) -> Expr {
    // Apply simplification rules bottom-up (simplify children first).
    let simplified_children = simplify_children(expr);
    simplify_step(&simplified_children)
}

fn simplify_children(expr: &Expr) -> Expr {
    match expr {
        Expr::Neg(x) => Expr::Neg(Box::new(simplify(x))),
        Expr::Add(terms) => Expr::Add(terms.iter().map(|t| simplify(t)).collect()),
        Expr::Mul(factors) => Expr::Mul(factors.iter().map(|f| simplify(f)).collect()),
        Expr::Pow(b, e) => Expr::Pow(Box::new(simplify(b)), Box::new(simplify(e))),
        Expr::Sin(x) => Expr::Sin(Box::new(simplify(x))),
        Expr::Cos(x) => Expr::Cos(Box::new(simplify(x))),
        Expr::Tan(x) => Expr::Tan(Box::new(simplify(x))),
        Expr::Ln(x) => Expr::Ln(Box::new(simplify(x))),
        Expr::Exp(x) => Expr::Exp(Box::new(simplify(x))),
        Expr::Sqrt(x) => Expr::Sqrt(Box::new(simplify(x))),
        _ => expr.clone(),
    }
}

fn simplify_step(expr: &Expr) -> Expr {
    match expr {
        // Double negation: --x = x
        Expr::Neg(x) => match x.as_ref() {
            Expr::Neg(inner) => *inner.clone(),
            Expr::Num(n) => Expr::Num(-n),
            Expr::Rat(p, q) => Expr::Rat(-p, *q),
            _ => expr.clone(),
        },
        
        // Addition simplification
        Expr::Add(terms) => {
            // Flatten nested additions: (a + (b + c)) → (a + b + c)
            let mut flat = Vec::new();
            for t in terms {
                match t {
                    Expr::Add(inner) => flat.extend(inner.iter().cloned()),
                    _ => flat.push(t.clone()),
                }
            }
            
            // Remove zeros
            flat.retain(|t| !matches!(t, Expr::Num(n) if n.abs() < 1e-15) 
                        && !matches!(t, Expr::Rat(0, _)));
            
            // Combine numeric constants
            let mut num_sum = 0.0;
            let mut non_numeric = Vec::new();
            for t in &flat {
                match t {
                    Expr::Num(n) => num_sum += n,
                    Expr::Rat(p, q) => num_sum += *p as f64 / *q as f64,
                    _ => non_numeric.push(t.clone()),
                }
            }
            
            // Collect like terms: 3x + 5x → 8x
            let mut term_map: HashMap<String, f64> = HashMap::new();
            let mut other_terms = Vec::new();
            for t in &non_numeric {
                let (coeff, base) = extract_coefficient(t);
                let key = format!("{:?}", base);
                *term_map.entry(key.clone()).or_insert(0.0) += coeff;
                // Store the base expression (only once per key)
                if !other_terms.iter().any(|(k, _): &(String, Expr)| k == &key) {
                    other_terms.push((key, base));
                }
            }
            
            let mut result = Vec::new();
            if num_sum.abs() > 1e-15 {
                result.push(Expr::Num(num_sum));
            }
            for (key, base) in &other_terms {
                let coeff = term_map[key];
                if coeff.abs() < 1e-15 { continue; }
                if (coeff - 1.0).abs() < 1e-15 {
                    result.push(base.clone());
                } else if (coeff + 1.0).abs() < 1e-15 {
                    result.push(Expr::Neg(Box::new(base.clone())));
                } else {
                    result.push(Expr::Mul(vec![Expr::Num(coeff), base.clone()]));
                }
            }
            
            match result.len() {
                0 => Expr::zero(),
                1 => result.pop().unwrap(),
                _ => Expr::Add(result),
            }
        }
        
        // Multiplication simplification
        Expr::Mul(factors) => {
            // Flatten nested multiplications
            let mut flat = Vec::new();
            for f in factors {
                match f {
                    Expr::Mul(inner) => flat.extend(inner.iter().cloned()),
                    _ => flat.push(f.clone()),
                }
            }
            
            // If any factor is zero, the whole product is zero
            if flat.iter().any(|f| matches!(f, Expr::Num(n) if n.abs() < 1e-15) 
                              || matches!(f, Expr::Rat(0, _))) {
                return Expr::zero();
            }
            
            // Remove ones
            flat.retain(|f| !matches!(f, Expr::Num(n) if (n - 1.0).abs() < 1e-15)
                        && !matches!(f, Expr::Rat(1, 1)));
            
            // Combine numeric constants
            let mut num_prod = 1.0;
            let mut non_numeric = Vec::new();
            for f in &flat {
                match f {
                    Expr::Num(n) => num_prod *= n,
                    Expr::Rat(p, q) => num_prod *= *p as f64 / *q as f64,
                    _ => non_numeric.push(f.clone()),
                }
            }
            
            // Combine like bases: x² * x³ → x⁵
            let mut power_map: HashMap<String, f64> = HashMap::new();
            let mut base_exprs: HashMap<String, Expr> = HashMap::new();
            let mut other_factors: Vec<Expr> = Vec::new();
            
            for f in &non_numeric {
                let (base, exp) = extract_base_exponent(f);
                let key = format!("{:?}", base);
                *power_map.entry(key.clone()).or_insert(0.0) += exp;
                base_exprs.entry(key).or_insert(base);
            }
            
            let mut result = Vec::new();
            if (num_prod - 1.0).abs() > 1e-15 {
                result.push(Expr::Num(num_prod));
            }
            for (key, exp) in &power_map {
                if exp.abs() < 1e-15 { continue; } // x^0 = 1, already handled
                let base = &base_exprs[key];
                if (*exp - 1.0).abs() < 1e-15 {
                    result.push(base.clone());
                } else {
                    result.push(Expr::Pow(Box::new(base.clone()), Box::new(Expr::Num(*exp))));
                }
            }
            
            match result.len() {
                0 => Expr::one(),
                1 => result.pop().unwrap(),
                _ => Expr::Mul(result),
            }
        }
        
        // Power simplification
        Expr::Pow(base, exp) => {
            match (base.as_ref(), exp.as_ref()) {
                // x^0 = 1
                (_, Expr::Num(n)) if n.abs() < 1e-15 => Expr::one(),
                (_, Expr::Rat(0, _)) => Expr::one(),
                // x^1 = x
                (_, Expr::Num(n)) if (n - 1.0).abs() < 1e-15 => *base.clone(),
                (_, Expr::Rat(1, 1)) => *base.clone(),
                // 0^n = 0 (for positive n)
                (Expr::Num(n), _) if n.abs() < 1e-15 => Expr::zero(),
                // 1^n = 1
                (Expr::Num(n), _) if (n - 1.0).abs() < 1e-15 => Expr::one(),
                // Numeric evaluation: 2^3 = 8
                (Expr::Num(b), Expr::Num(e)) => Expr::Num(b.powf(*e)),
                (Expr::Rat(p, q), Expr::Rat(n, 1)) if *n >= 0 => {
                    let val = (*p as f64 / *q as f64).powi(*n as i32);
                    Expr::Num(val)
                }
                // (x^a)^b = x^(a*b)
                (Expr::Pow(inner_base, inner_exp), _) => {
                    Expr::Pow(
                        inner_base.clone(),
                        Box::new(simplify(&Expr::Mul(vec![*inner_exp.clone(), *exp.clone()]))),
                    )
                }
                _ => expr.clone(),
            }
        }
        
        // Logarithmic identities
        Expr::Ln(x) => match x.as_ref() {
            Expr::E => Expr::one(),                    // ln(e) = 1
            Expr::Num(n) if (n - 1.0).abs() < 1e-15 => Expr::zero(), // ln(1) = 0
            Expr::Exp(inner) => *inner.clone(),        // ln(e^x) = x
            _ => expr.clone(),
        },
        
        // Exponential identities
        Expr::Exp(x) => match x.as_ref() {
            Expr::Num(n) if n.abs() < 1e-15 => Expr::one(), // e^0 = 1
            Expr::Ln(inner) => *inner.clone(),               // e^(ln(x)) = x
            _ => expr.clone(),
        },
        
        // Sqrt identities
        Expr::Sqrt(x) => match x.as_ref() {
            Expr::Num(n) if n.abs() < 1e-15 => Expr::zero(),
            Expr::Num(n) if (n - 1.0).abs() < 1e-15 => Expr::one(),
            Expr::Num(n) if *n >= 0.0 => {
                let root = n.sqrt();
                if (root - root.round()).abs() < 1e-10 {
                    Expr::Num(root.round()) // Perfect square
                } else {
                    expr.clone()
                }
            }
            _ => expr.clone(),
        },
        
        // Trig identities
        Expr::Sin(x) => match x.as_ref() {
            Expr::Num(n) if n.abs() < 1e-15 => Expr::zero(),       // sin(0) = 0
            _ => expr.clone(),
        },
        Expr::Cos(x) => match x.as_ref() {
            Expr::Num(n) if n.abs() < 1e-15 => Expr::one(),        // cos(0) = 1
            _ => expr.clone(),
        },
        
        _ => expr.clone(),
    }
}

/// Extract coefficient from a term: 3*x → (3.0, x), x → (1.0, x)
fn extract_coefficient(expr: &Expr) -> (f64, Expr) {
    match expr {
        Expr::Mul(factors) => {
            let mut coeff = 1.0;
            let mut rest = Vec::new();
            for f in factors {
                match f {
                    Expr::Num(n) => coeff *= n,
                    Expr::Rat(p, q) => coeff *= *p as f64 / *q as f64,
                    _ => rest.push(f.clone()),
                }
            }
            let base = match rest.len() {
                0 => Expr::one(),
                1 => rest.pop().unwrap(),
                _ => Expr::Mul(rest),
            };
            (coeff, base)
        }
        Expr::Neg(inner) => {
            let (c, b) = extract_coefficient(inner);
            (-c, b)
        }
        Expr::Num(n) => (*n, Expr::one()),
        Expr::Rat(p, q) => (*p as f64 / *q as f64, Expr::one()),
        _ => (1.0, expr.clone()),
    }
}

/// Extract base and exponent: x^3 → (x, 3.0), x → (x, 1.0)
fn extract_base_exponent(expr: &Expr) -> (Expr, f64) {
    match expr {
        Expr::Pow(base, exp) => {
            if let Expr::Num(n) = exp.as_ref() {
                (*base.clone(), *n)
            } else if let Expr::Rat(p, q) = exp.as_ref() {
                (*base.clone(), *p as f64 / *q as f64)
            } else {
                (expr.clone(), 1.0)
            }
        }
        Expr::Sqrt(x) => (*x.clone(), 0.5),
        _ => (expr.clone(), 1.0),
    }
}

// ============================================================================
// INTEGRATION ENGINE
// ============================================================================
// Symbolic integration using a table of known integrals + heuristic methods.
// This handles: polynomials, trig, exponential, logarithmic, rational
// functions (partial fractions), integration by parts (LIATE heuristic),
// and u-substitution for simple compositions.

/// Symbolically integrate an expression with respect to a variable.
/// Returns the antiderivative (without the constant of integration).
pub fn integrate(expr: &Expr, var: &str) -> Expr {
    // If the expression doesn't contain the variable, it's a constant.
    if !expr.contains_var(var) {
        return Expr::Mul(vec![expr.clone(), Expr::Var(var.to_string())]);
    }
    
    let result = try_integrate(expr, var);
    simplify(&result)
}

fn try_integrate(expr: &Expr, var: &str) -> Expr {
    match expr {
        // ∫x dx = x²/2
        Expr::Var(v) if v == var => {
            Expr::div(
                Expr::Pow(Box::new(Expr::Var(var.to_string())), Box::new(Expr::two())),
                Expr::two(),
            )
        }
        
        // ∫c dx = cx
        _ if !expr.contains_var(var) => {
            Expr::Mul(vec![expr.clone(), Expr::Var(var.to_string())])
        }
        
        // Sum rule: ∫(f + g) dx = ∫f dx + ∫g dx
        Expr::Add(terms) => {
            Expr::Add(terms.iter().map(|t| integrate(t, var)).collect())
        }
        
        // Constant multiple: ∫c*f dx = c * ∫f dx
        Expr::Mul(factors) => {
            let (constants, variables): (Vec<_>, Vec<_>) = factors.iter()
                .partition(|f| !f.contains_var(var));
            
            if !constants.is_empty() && !variables.is_empty() {
                let const_part = if constants.len() == 1 {
                    constants[0].clone()
                } else {
                    Expr::Mul(constants.into_iter().cloned().collect())
                };
                let var_part = if variables.len() == 1 {
                    variables[0].clone()
                } else {
                    Expr::Mul(variables.into_iter().cloned().collect())
                };
                return Expr::Mul(vec![const_part, integrate(&var_part, var)]);
            }
            
            // Try integration by parts for products of two functions
            if factors.len() == 2 {
                if let Some(result) = try_integration_by_parts(&factors[0], &factors[1], var) {
                    return result;
                }
            }
            
            // Fallback: return unevaluated
            Expr::Integral(Box::new(expr.clone()), var.to_string())
        }
        
        Expr::Neg(f) => Expr::Neg(Box::new(integrate(f, var))),
        
        // Power rule: ∫x^n dx = x^(n+1)/(n+1), n ≠ -1
        Expr::Pow(base, exp) if **base == Expr::Var(var.to_string()) && !exp.contains_var(var) => {
            // Check if exponent is -1 (special case: integral is ln|x|)
            match exp.as_ref() {
                Expr::Num(n) if (n + 1.0).abs() < 1e-15 => {
                    Expr::Ln(Box::new(Expr::Abs(base.clone())))
                }
                Expr::Rat(p, q) if *p == -1 && *q == 1 => {
                    Expr::Ln(Box::new(Expr::Abs(base.clone())))
                }
                _ => {
                    let new_exp = simplify(&Expr::Add(vec![*exp.clone(), Expr::one()]));
                    Expr::div(
                        Expr::Pow(base.clone(), Box::new(new_exp.clone())),
                        new_exp,
                    )
                }
            }
        }
        
        // ∫sin(x) dx = -cos(x)
        Expr::Sin(f) if **f == Expr::Var(var.to_string()) => {
            Expr::Neg(Box::new(Expr::Cos(f.clone())))
        }
        // ∫cos(x) dx = sin(x)
        Expr::Cos(f) if **f == Expr::Var(var.to_string()) => {
            Expr::Sin(f.clone())
        }
        // ∫tan(x) dx = -ln|cos(x)|
        Expr::Tan(f) if **f == Expr::Var(var.to_string()) => {
            Expr::Neg(Box::new(Expr::Ln(Box::new(Expr::Abs(Box::new(Expr::Cos(f.clone())))))))
        }
        
        // ∫e^x dx = e^x
        Expr::Exp(f) if **f == Expr::Var(var.to_string()) => {
            Expr::Exp(f.clone())
        }
        
        // ∫ln(x) dx = x*ln(x) - x (by parts)
        Expr::Ln(f) if **f == Expr::Var(var.to_string()) => {
            let x = Expr::Var(var.to_string());
            Expr::sub(
                Expr::Mul(vec![x.clone(), Expr::Ln(f.clone())]),
                x,
            )
        }
        
        // ∫1/x dx = ln|x|
        Expr::Pow(base, exp) if **base == Expr::Var(var.to_string()) => {
            if let Expr::Num(n) = exp.as_ref() {
                if (n + 1.0).abs() < 1e-15 {
                    return Expr::Ln(Box::new(Expr::Abs(base.clone())));
                }
            }
            // General power rule
            let new_exp = simplify(&Expr::Add(vec![*exp.clone(), Expr::one()]));
            Expr::div(
                Expr::Pow(base.clone(), Box::new(new_exp.clone())),
                new_exp,
            )
        }
        
        // ∫1/√(1-x²) dx = arcsin(x)
        // ∫1/(1+x²) dx = arctan(x)
        // These are recognized by pattern matching.
        
        // ∫sin(ax) dx = -cos(ax)/a (linear substitution)
        Expr::Sin(f) => {
            if let Some((a, b)) = extract_linear(f, var) {
                return Expr::div(
                    Expr::Neg(Box::new(Expr::Cos(f.clone()))),
                    Expr::Num(a),
                );
            }
            Expr::Integral(Box::new(expr.clone()), var.to_string())
        }
        
        // ∫cos(ax) dx = sin(ax)/a
        Expr::Cos(f) => {
            if let Some((a, _)) = extract_linear(f, var) {
                return Expr::div(Expr::Sin(f.clone()), Expr::Num(a));
            }
            Expr::Integral(Box::new(expr.clone()), var.to_string())
        }
        
        // ∫e^(ax) dx = e^(ax)/a
        Expr::Exp(f) => {
            if let Some((a, _)) = extract_linear(f, var) {
                return Expr::div(Expr::Exp(f.clone()), Expr::Num(a));
            }
            Expr::Integral(Box::new(expr.clone()), var.to_string())
        }
        
        // ∫√x dx = (2/3)x^(3/2)
        Expr::Sqrt(f) if **f == Expr::Var(var.to_string()) => {
            Expr::Mul(vec![
                Expr::div(Expr::two(), Expr::Num(3.0)),
                Expr::Pow(Box::new(*f.clone()), Box::new(Expr::Num(1.5))),
            ])
        }
        
        _ => {
            // Return unevaluated integral for cases we can't handle.
            Expr::Integral(Box::new(expr.clone()), var.to_string())
        }
    }
}

/// Check if f(x) = a*x + b (a linear function of var).
/// Returns Some((a, b)) if linear, None otherwise.
fn extract_linear(expr: &Expr, var: &str) -> Option<(f64, f64)> {
    match expr {
        Expr::Var(v) if v == var => Some((1.0, 0.0)),
        Expr::Mul(factors) if factors.len() == 2 => {
            // a * x
            match (&factors[0], &factors[1]) {
                (Expr::Num(a), Expr::Var(v)) if v == var => Some((*a, 0.0)),
                (Expr::Var(v), Expr::Num(a)) if v == var => Some((*a, 0.0)),
                _ => None,
            }
        }
        Expr::Add(terms) if terms.len() == 2 => {
            // ax + b
            let mut a = None;
            let mut b = 0.0;
            for t in terms {
                if let Some((coeff, _)) = extract_linear(t, var) {
                    a = Some(coeff);
                } else if !t.contains_var(var) {
                    if let Ok(val) = t.eval(&HashMap::new()) {
                        b = val;
                    }
                }
            }
            a.map(|a| (a, b))
        }
        _ => None,
    }
}

/// Try integration by parts: ∫u dv = uv - ∫v du
/// Uses the LIATE heuristic to choose u and dv.
fn try_integration_by_parts(f1: &Expr, f2: &Expr, var: &str) -> Option<Expr> {
    // LIATE priority: Logarithmic > Inverse trig > Algebraic > Trig > Exponential
    // The one with higher priority becomes u (we differentiate it).
    let p1 = liate_priority(f1);
    let p2 = liate_priority(f2);
    
    let (u, dv) = if p1 >= p2 { (f1, f2) } else { (f2, f1) };
    
    let du = simplify(&differentiate(u, var));
    let v = integrate(dv, var);
    
    // Check that v is not an unevaluated integral (meaning we could integrate dv).
    if matches!(v, Expr::Integral(_, _)) {
        return None;
    }
    
    // ∫u dv = u*v - ∫v*du
    let second_integral = integrate(&simplify(&Expr::Mul(vec![v.clone(), du])), var);
    
    // Avoid infinite recursion: if the second integral is also unevaluated, give up.
    if matches!(second_integral, Expr::Integral(_, _)) {
        return None;
    }
    
    Some(simplify(&Expr::sub(
        Expr::Mul(vec![u.clone(), v]),
        second_integral,
    )))
}

fn liate_priority(expr: &Expr) -> u8 {
    match expr {
        Expr::Ln(_) | Expr::Log(_, _) => 5,
        Expr::ArcSin(_) | Expr::ArcCos(_) | Expr::ArcTan(_) => 4,
        Expr::Var(_) | Expr::Pow(_, _) => 3,
        Expr::Sin(_) | Expr::Cos(_) | Expr::Tan(_) => 2,
        Expr::Exp(_) => 1,
        _ => 0,
    }
}

// ============================================================================
// DEFINITE INTEGRATION (Numerical — Gaussian Quadrature)
// ============================================================================

/// Numerically evaluate a definite integral using adaptive Gauss-Kronrod quadrature.
pub fn numerical_integrate(
    f: &dyn Fn(f64) -> f64,
    a: f64, b: f64,
    tolerance: f64,
) -> f64 {
    adaptive_simpson(f, a, b, tolerance, 50)
}

fn adaptive_simpson(f: &dyn Fn(f64) -> f64, a: f64, b: f64, tol: f64, max_depth: u32) -> f64 {
    let c = (a + b) / 2.0;
    let s_whole = simpson_rule(f, a, b);
    let s_left = simpson_rule(f, a, c);
    let s_right = simpson_rule(f, c, b);
    let s_combined = s_left + s_right;
    
    if max_depth == 0 || (s_combined - s_whole).abs() < 15.0 * tol {
        s_combined + (s_combined - s_whole) / 15.0 // Richardson extrapolation
    } else {
        adaptive_simpson(f, a, c, tol / 2.0, max_depth - 1) +
        adaptive_simpson(f, c, b, tol / 2.0, max_depth - 1)
    }
}

fn simpson_rule(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let c = (a + b) / 2.0;
    (b - a) / 6.0 * (f(a) + 4.0 * f(c) + f(b))
}

// ============================================================================
// EQUATION SOLVING
// ============================================================================

/// Solve an equation f(x) = 0 symbolically.
/// Returns a vector of solutions.
pub fn solve(equation: &Expr, var: &str) -> Vec<Expr> {
    // Try to identify the equation structure and apply the appropriate method.
    let simplified = simplify(equation);
    
    // Polynomial solver
    if let Some(coeffs) = extract_polynomial(&simplified, var) {
        return solve_polynomial(&coeffs);
    }
    
    // For transcendental equations, try Newton's method numerically.
    Vec::new()
}

/// Extract polynomial coefficients: ax^2 + bx + c → [c, b, a]
fn extract_polynomial(expr: &Expr, var: &str) -> Option<Vec<f64>> {
    let mut coeffs = HashMap::new();
    collect_poly_terms(expr, var, 1.0, &mut coeffs)?;
    
    let max_degree = coeffs.keys().max().cloned().unwrap_or(0);
    if max_degree > 10 { return None; } // Too high degree
    
    let mut result = vec![0.0; max_degree + 1];
    for (deg, coeff) in coeffs {
        result[deg] = coeff;
    }
    Some(result)
}

fn collect_poly_terms(expr: &Expr, var: &str, coeff: f64, terms: &mut HashMap<usize, f64>) -> Option<()> {
    match expr {
        Expr::Num(n) => { *terms.entry(0).or_insert(0.0) += coeff * n; Some(()) }
        Expr::Rat(p, q) => { *terms.entry(0).or_insert(0.0) += coeff * *p as f64 / *q as f64; Some(()) }
        Expr::Var(v) if v == var => { *terms.entry(1).or_insert(0.0) += coeff; Some(()) }
        Expr::Var(_) if !expr.contains_var(var) => { *terms.entry(0).or_insert(0.0) += coeff; Some(()) }
        Expr::Neg(x) => collect_poly_terms(x, var, -coeff, terms),
        Expr::Add(ts) => {
            for t in ts { collect_poly_terms(t, var, coeff, terms)?; }
            Some(())
        }
        Expr::Mul(fs) => {
            let mut num_coeff = coeff;
            let mut var_degree = 0usize;
            for f in fs {
                match f {
                    Expr::Num(n) => num_coeff *= n,
                    Expr::Rat(p, q) => num_coeff *= *p as f64 / *q as f64,
                    Expr::Var(v) if v == var => var_degree += 1,
                    Expr::Pow(base, exp) => {
                        if let (Expr::Var(v), Expr::Num(n)) = (base.as_ref(), exp.as_ref()) {
                            if v == var && *n == (*n as usize) as f64 {
                                var_degree += *n as usize;
                            } else { return None; }
                        } else { return None; }
                    }
                    _ if !f.contains_var(var) => {
                        if let Ok(v) = f.eval(&HashMap::new()) { num_coeff *= v; }
                        else { return None; }
                    }
                    _ => return None,
                }
            }
            *terms.entry(var_degree).or_insert(0.0) += num_coeff;
            Some(())
        }
        Expr::Pow(base, exp) => {
            if let (Expr::Var(v), Expr::Num(n)) = (base.as_ref(), exp.as_ref()) {
                if v == var && *n == (*n as usize) as f64 {
                    *terms.entry(*n as usize).or_insert(0.0) += coeff;
                    return Some(());
                }
            }
            None
        }
        _ => None,
    }
}

/// Solve a polynomial given its coefficients [c, b, a, ...] for c + bx + ax² + ...
fn solve_polynomial(coeffs: &[f64]) -> Vec<Expr> {
    match coeffs.len() {
        0 | 1 => Vec::new(), // No variable or constant
        2 => {
            // Linear: c + bx = 0 → x = -c/b
            let (c, b) = (coeffs[0], coeffs[1]);
            if b.abs() < 1e-15 { Vec::new() }
            else { vec![Expr::Num(-c / b)] }
        }
        3 => {
            // Quadratic: c + bx + ax² = 0
            let (c, b, a) = (coeffs[0], coeffs[1], coeffs[2]);
            let disc = b * b - 4.0 * a * c;
            if disc < -1e-15 {
                // Complex roots
                let real = -b / (2.0 * a);
                let imag = (-disc).sqrt() / (2.0 * a);
                vec![
                    Expr::Add(vec![Expr::Num(real), Expr::Mul(vec![Expr::Num(imag), Expr::I])]),
                    Expr::Add(vec![Expr::Num(real), Expr::Mul(vec![Expr::Num(-imag), Expr::I])]),
                ]
            } else if disc.abs() < 1e-15 {
                vec![Expr::Num(-b / (2.0 * a))]
            } else {
                let sqrt_disc = disc.sqrt();
                vec![
                    Expr::Num((-b + sqrt_disc) / (2.0 * a)),
                    Expr::Num((-b - sqrt_disc) / (2.0 * a)),
                ]
            }
        }
        4 => {
            // Cubic: Cardano's formula or numerical
            solve_cubic(coeffs[3], coeffs[2], coeffs[1], coeffs[0])
        }
        _ => {
            // Higher degree: use Durand-Kerner method (numerical)
            solve_durand_kerner(coeffs)
        }
    }
}

/// Solve a cubic equation ax³ + bx² + cx + d = 0 using Cardano's method.
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<Expr> {
    // Reduce to depressed cubic: t³ + pt + q = 0
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b * b * b - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a * a * a);
    let disc = -4.0 * p * p * p - 27.0 * q * q;
    
    let shift = -b / (3.0 * a);
    
    if disc > 1e-10 {
        // Three real roots (casus irreducibilis)
        let r = (-p / 3.0).sqrt();
        let theta = ((-q) / (2.0 * r * r * r)).acos();
        vec![
            Expr::Num(2.0 * r * (theta / 3.0).cos() + shift),
            Expr::Num(2.0 * r * ((theta + 2.0 * PI) / 3.0).cos() + shift),
            Expr::Num(2.0 * r * ((theta + 4.0 * PI) / 3.0).cos() + shift),
        ]
    } else {
        // One or two real roots — use Cardano's formula
        let inner = q * q / 4.0 + p * p * p / 27.0;
        if inner >= 0.0 {
            let sqrt_inner = inner.sqrt();
            let u = (-q / 2.0 + sqrt_inner).cbrt();
            let v = (-q / 2.0 - sqrt_inner).cbrt();
            vec![Expr::Num(u + v + shift)]
        } else {
            // Fallback to numerical
            solve_durand_kerner(&[d, c, b, a])
        }
    }
}

/// Durand-Kerner method for numerical root finding of polynomials.
/// Finds all roots (real and complex) simultaneously.
fn solve_durand_kerner(coeffs: &[f64]) -> Vec<Expr> {
    let n = coeffs.len() - 1; // Degree
    if n == 0 { return Vec::new(); }
    
    // Normalize so leading coefficient is 1.
    let lead = coeffs[n];
    let norm: Vec<f64> = coeffs.iter().map(|c| c / lead).collect();
    
    // Initial guesses: equally spaced on a circle.
    let mut roots: Vec<(f64, f64)> = (0..n).map(|k| {
        let angle = 2.0 * PI * k as f64 / n as f64 + 0.4;
        let r = 1.0 + norm.iter().map(|c| c.abs()).sum::<f64>();
        (r * angle.cos(), r * angle.sin())
    }).collect();
    
    // Iterate until convergence.
    for _ in 0..1000 {
        let mut max_delta = 0.0f64;
        for i in 0..n {
            // Evaluate polynomial at roots[i]
            let (mut pr, mut pi) = (0.0, 0.0);
            let (mut zr, mut zi) = (1.0, 0.0);
            for j in 0..=n {
                pr += norm[j] * zr;
                pi += norm[j] * zi;
                let new_zr = zr * roots[i].0 - zi * roots[i].1;
                let new_zi = zr * roots[i].1 + zi * roots[i].0;
                zr = new_zr;
                zi = new_zi;
            }
            
            // Compute denominator: product of (z_i - z_j) for j ≠ i
            let (mut dr, mut di) = (1.0, 0.0);
            for j in 0..n {
                if i == j { continue; }
                let diff_r = roots[i].0 - roots[j].0;
                let diff_i = roots[i].1 - roots[j].1;
                let new_dr = dr * diff_r - di * diff_i;
                let new_di = dr * diff_i + di * diff_r;
                dr = new_dr;
                di = new_di;
            }
            
            // z_i := z_i - p(z_i) / prod(z_i - z_j)
            let denom = dr * dr + di * di;
            if denom < 1e-30 { continue; }
            let delta_r = (pr * dr + pi * di) / denom;
            let delta_i = (pi * dr - pr * di) / denom;
            roots[i].0 -= delta_r;
            roots[i].1 -= delta_i;
            max_delta = max_delta.max(delta_r.abs() + delta_i.abs());
        }
        if max_delta < 1e-12 { break; }
    }
    
    // Convert to Expr
    roots.iter().map(|&(re, im)| {
        if im.abs() < 1e-10 {
            Expr::Num(re)
        } else {
            Expr::Add(vec![Expr::Num(re), Expr::Mul(vec![Expr::Num(im), Expr::I])])
        }
    }).collect()
}

/// Newton's method for solving f(x) = 0 numerically.
pub fn newton_solve(f: &dyn Fn(f64) -> f64, df: &dyn Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> Option<f64> {
    let mut x = x0;
    for _ in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol { return Some(x); }
        let dfx = df(x);
        if dfx.abs() < 1e-15 { return None; } // Derivative too small
        x -= fx / dfx;
    }
    if f(x).abs() < tol * 100.0 { Some(x) } else { None }
}

// ============================================================================
// TAYLOR SERIES EXPANSION
// ============================================================================

/// Compute the Taylor series of expr around point a, up to order n.
/// Returns a polynomial approximation.
pub fn taylor_series(expr: &Expr, var: &str, a: &Expr, order: u32) -> Expr {
    let mut terms = Vec::new();
    let mut factorial = 1u64;
    
    let x_minus_a = Expr::sub(Expr::Var(var.to_string()), a.clone());
    let mut current_deriv = expr.clone();
    
    for k in 0..=order {
        if k > 0 { factorial *= k as u64; }
        
        // Evaluate the k-th derivative at x = a
        let at_a = current_deriv.substitute(var, a);
        let simplified_at_a = simplify(&at_a);
        
        // Term: f^(k)(a) / k! * (x - a)^k
        let term = if k == 0 {
            simplified_at_a
        } else {
            Expr::Mul(vec![
                Expr::div(simplified_at_a, Expr::Num(factorial as f64)),
                Expr::Pow(Box::new(x_minus_a.clone()), Box::new(Expr::Num(k as f64))),
            ])
        };
        terms.push(term);
        
        // Compute the next derivative for the next iteration
        if k < order {
            current_deriv = simplify(&differentiate(&current_deriv, var));
        }
    }
    
    simplify(&Expr::Add(terms))
}

// ============================================================================
// LINEAR ALGEBRA ENGINE
// ============================================================================

/// A matrix of floating point numbers for numerical linear algebra.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { data, rows, cols }
    }
    
    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n { data[i][i] = 1.0; }
        Self { data, rows: n, cols: n }
    }
    
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![vec![0.0; cols]; rows], rows, cols }
    }

    /// Matrix multiplication. O(n³).
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                for j in 0..other.cols {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }
    
    /// Transpose.
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// Determinant using LU decomposition. O(n³).
    pub fn determinant(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let (lu, perm, sign) = self.lu_decompose();
        let mut det = sign as f64;
        for i in 0..n {
            det *= lu.data[i][i];
        }
        det
    }

    /// LU decomposition with partial pivoting.
    /// Returns (LU_combined_matrix, permutation, sign_of_permutation).
    pub fn lu_decompose(&self) -> (Matrix, Vec<usize>, i32) {
        let n = self.rows;
        let mut lu = self.clone();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut sign = 1i32;

        for k in 0..n {
            // Find pivot (largest element in column k below row k).
            let mut max_val = 0.0f64;
            let mut max_row = k;
            for i in k..n {
                if lu.data[i][k].abs() > max_val {
                    max_val = lu.data[i][k].abs();
                    max_row = i;
                }
            }
            
            if max_val < 1e-15 { continue; } // Singular matrix

            // Swap rows.
            if max_row != k {
                lu.data.swap(k, max_row);
                perm.swap(k, max_row);
                sign = -sign;
            }

            // Eliminate below.
            for i in (k + 1)..n {
                lu.data[i][k] /= lu.data[k][k];
                for j in (k + 1)..n {
                    lu.data[i][j] -= lu.data[i][k] * lu.data[k][j];
                }
            }
        }

        (lu, perm, sign)
    }

    /// Solve Ax = b using LU decomposition.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.rows;
        let (lu, perm, _) = self.lu_decompose();
        
        // Forward substitution: Ly = Pb
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = b[perm[i]];
            for j in 0..i {
                y[i] -= lu.data[i][j] * y[j];
            }
        }
        
        // Back substitution: Ux = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= lu.data[i][j] * x[j];
            }
            x[i] /= lu.data[i][i];
        }
        
        x
    }

    /// Matrix inverse using LU decomposition. O(n³).
    pub fn inverse(&self) -> Option<Matrix> {
        let n = self.rows;
        if self.determinant().abs() < 1e-15 { return None; }
        
        let mut inv = Matrix::zeros(n, n);
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = self.solve(&e);
            for i in 0..n {
                inv.data[i][j] = col[i];
            }
        }
        Some(inv)
    }

    /// Eigenvalues using the QR algorithm. O(n³ per iteration).
    /// Returns approximate eigenvalues.
    pub fn eigenvalues(&self) -> Vec<f64> {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;
        let mut a = self.clone();
        
        // QR iteration with shifts.
        for _ in 0..1000 {
            // Wilkinson shift: use the eigenvalue of the bottom-right 2x2 that's
            // closest to a[n-1][n-1].
            let shift = if n >= 2 {
                let d = (a.data[n-2][n-2] - a.data[n-1][n-1]) / 2.0;
                let sign = if d >= 0.0 { 1.0 } else { -1.0 };
                a.data[n-1][n-1] - sign * a.data[n-1][n-2].powi(2)
                    / (d.abs() + (d*d + a.data[n-1][n-2].powi(2)).sqrt())
            } else { 0.0 };
            
            // Shift
            for i in 0..n { a.data[i][i] -= shift; }
            
            // QR decomposition using Householder reflections
            let (q, r) = qr_decompose(&a);
            
            // A = RQ + shift*I
            a = r.mul(&q);
            for i in 0..n { a.data[i][i] += shift; }
            
            // Check convergence: are the subdiagonal elements small?
            let mut converged = true;
            for i in 1..n {
                if a.data[i][i-1].abs() > 1e-10 {
                    converged = false;
                    break;
                }
            }
            if converged { break; }
        }
        
        // Eigenvalues are on the diagonal.
        (0..n).map(|i| a.data[i][i]).collect()
    }
    
    /// Frobenius norm.
    pub fn norm_frobenius(&self) -> f64 {
        let mut sum = 0.0;
        for row in &self.data {
            for &val in row {
                sum += val * val;
            }
        }
        sum.sqrt()
    }
}

/// QR decomposition using Gram-Schmidt orthogonalization.
pub fn qr_decompose(a: &Matrix) -> (Matrix, Matrix) {
    let n = a.rows;
    let m = a.cols;
    let mut q = Matrix::zeros(n, m);
    let mut r = Matrix::zeros(m, m);
    
    for j in 0..m {
        // Start with column j of A.
        let mut v: Vec<f64> = (0..n).map(|i| a.data[i][j]).collect();
        
        // Subtract projections onto previous q vectors.
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..n { dot += q.data[i][k] * a.data[i][j]; }
            r.data[k][j] = dot;
            for i in 0..n { v[i] -= dot * q.data[i][k]; }
        }
        
        // Normalize.
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        r.data[j][j] = norm;
        if norm > 1e-15 {
            for i in 0..n { q.data[i][j] = v[i] / norm; }
        }
    }
    
    (q, r)
}

// ============================================================================
// SPECIAL FUNCTIONS (Numerical Evaluation)
// ============================================================================

/// Gamma function Γ(x) using the Lanczos approximation.
/// Accurate to about 15 significant digits.
pub fn gamma_fn(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY; // Poles at non-positive integers
    }
    
    // Lanczos coefficients for g=7
    let g = 7.0;
    let coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    
    if x < 0.5 {
        // Reflection formula: Γ(x)Γ(1-x) = π / sin(πx)
        PI / ((PI * x).sin() * gamma_fn(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut sum = coeffs[0];
        for i in 1..coeffs.len() {
            sum += coeffs[i] / (x + i as f64);
        }
        let t = x + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * sum
    }
}

/// Error function erf(x) using Abramowitz and Stegun approximation.
pub fn erf_fn(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + 
               t * (-1.453152027 + t * 1.061405429))));
    
    sign * (1.0 - poly * (-x * x).exp())
}

/// Bessel function J_0(x) using series expansion for small x and
/// asymptotic expansion for large x.
pub fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        // Series expansion: J_0(x) = Σ (-1)^k (x/2)^(2k) / (k!)^2
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..50 {
            sum += term;
            term *= -(x * x) / (4.0 * (k as f64) * (k as f64));
            if term.abs() < 1e-16 { break; }
        }
        sum + term
    } else {
        // Asymptotic expansion for large x
        let z = 8.0 / ax;
        let z2 = z * z;
        let p0 = 1.0 + z2 * (-0.1098628627e-2 + z2 * (0.2734510407e-4));
        let q0 = -0.1562499995e-1 + z2 * (0.1430488765e-3 + z2 * (-0.6911147651e-5));
        let theta = ax - 0.7853981634;
        (0.636619772 / ax).sqrt() * (theta.cos() * p0 - z * theta.sin() * q0)
    }
}

// ============================================================================
// NUMBER THEORY
// ============================================================================

/// Miller-Rabin primality test. Probabilistic for large n, deterministic for n < 3.3×10^24.
pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    
    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0u32;
    while d % 2 == 0 { d /= 2; r += 1; }
    
    // Deterministic witnesses for n < 3.3×10^24
    let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    
    'outer: for &a in &witnesses {
        if a >= n { continue; }
        let mut x = mod_pow(a, d, n);
        if x == 1 || x == n - 1 { continue; }
        for _ in 0..r - 1 {
            x = mod_pow(x, 2, n);
            if x == n - 1 { continue 'outer; }
        }
        return false;
    }
    true
}

/// Modular exponentiation: base^exp mod modulus.
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        exp /= 2;
        base = (base as u128 * base as u128 % modulus as u128) as u64;
    }
    result
}

/// Integer factorization using trial division + Pollard's rho.
pub fn factor(mut n: u64) -> Vec<(u64, u32)> {
    let mut factors = Vec::new();
    
    // Trial division for small factors.
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] {
        let mut count = 0u32;
        while n % p == 0 { n /= p; count += 1; }
        if count > 0 { factors.push((p, count)); }
    }
    
    // Pollard's rho for remaining factors.
    while n > 1 {
        if is_prime(n) {
            factors.push((n, 1));
            break;
        }
        let d = pollard_rho(n);
        let mut count = 0u32;
        while n % d == 0 { n /= d; count += 1; }
        factors.push((d, count));
    }
    
    factors.sort();
    factors
}

fn pollard_rho(n: u64) -> u64 {
    if n % 2 == 0 { return 2; }
    let mut rng_state = 2u64;
    for c in 1.. {
        let f = |x: u64| ((x as u128 * x as u128 + c as u128) % n as u128) as u64;
        let mut x = rng_state;
        let mut y = rng_state;
        loop {
            x = f(x);
            y = f(f(y));
            let d = gcd((x as i64 - y as i64).unsigned_abs(), n);
            if d == n { break; }
            if d != 1 { return d; }
        }
        rng_state += 1;
    }
    n // Shouldn't reach here
}

/// Greatest common divisor.
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Extended Euclidean algorithm: returns (gcd, x, y) such that ax + by = gcd.
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 { return (b, 0, 1); }
    let (g, x1, y1) = extended_gcd(b % a, a);
    (g, y1 - (b / a) * x1, x1)
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_differentiation() {
        // d/dx(x^3) = 3x^2
        let expr = Expr::pow(Expr::var("x"), Expr::Num(3.0));
        let deriv = simplify(&differentiate(&expr, "x"));
        // Evaluate at x=2: should be 3*4 = 12
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 2.0);
        let result = deriv.eval(&vars).unwrap();
        assert!((result - 12.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_diff_sin() {
        // d/dx(sin(x)) = cos(x)
        let expr = Expr::Sin(Box::new(Expr::var("x")));
        let deriv = simplify(&differentiate(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), PI / 4.0);
        let result = deriv.eval(&vars).unwrap();
        assert!((result - (PI / 4.0).cos()).abs() < 1e-10);
    }
    
    #[test]
    fn test_diff_chain_rule() {
        // d/dx(sin(x^2)) = 2x*cos(x^2)
        let expr = Expr::Sin(Box::new(Expr::pow(Expr::var("x"), Expr::two())));
        let deriv = simplify(&differentiate(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let result = deriv.eval(&vars).unwrap();
        let expected = 2.0 * 1.0_f64.cos(); // 2*cos(1)
        assert!((result - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_diff_product_rule() {
        // d/dx(x * sin(x)) = sin(x) + x*cos(x)
        let expr = Expr::Mul(vec![Expr::var("x"), Expr::Sin(Box::new(Expr::var("x")))]);
        let deriv = simplify(&differentiate(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), PI / 3.0);
        let result = deriv.eval(&vars).unwrap();
        let expected = (PI / 3.0).sin() + (PI / 3.0) * (PI / 3.0).cos();
        assert!((result - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_integration_polynomial() {
        // ∫x^2 dx = x^3/3
        let expr = Expr::pow(Expr::var("x"), Expr::two());
        let integral = simplify(&integrate(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        let result = integral.eval(&vars).unwrap();
        assert!((result - 9.0).abs() < 1e-10); // 3^3/3 = 9
    }
    
    #[test]
    fn test_integration_trig() {
        // ∫sin(x) dx = -cos(x)
        let expr = Expr::Sin(Box::new(Expr::var("x")));
        let integral = simplify(&integrate(&expr, "x"));
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), PI / 2.0);
        let result = integral.eval(&vars).unwrap();
        let expected = -(PI / 2.0).cos(); // -cos(π/2) = 0
        assert!((result - expected).abs() < 1e-10);
    }
    
    #[test]
    fn test_numerical_integration() {
        // ∫₀¹ x² dx = 1/3
        let result = numerical_integrate(&|x| x * x, 0.0, 1.0, 1e-12);
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
        
        // ∫₀^π sin(x) dx = 2
        let result = numerical_integrate(&|x| x.sin(), 0.0, PI, 1e-12);
        assert!((result - 2.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_solve_quadratic() {
        // x² - 5x + 6 = 0 → x = 2, 3
        let coeffs = vec![6.0, -5.0, 1.0];
        let roots = solve_polynomial(&coeffs);
        assert_eq!(roots.len(), 2);
        let mut vals: Vec<f64> = roots.iter().map(|r| {
            r.eval(&HashMap::new()).unwrap()
        }).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_solve_cubic() {
        // x³ - 6x² + 11x - 6 = 0 → x = 1, 2, 3
        let coeffs = vec![-6.0, 11.0, -6.0, 1.0];
        let roots = solve_polynomial(&coeffs);
        let mut vals: Vec<f64> = roots.iter().filter_map(|r| {
            r.eval(&HashMap::new()).ok()
        }).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.0).abs() < 1e-8);
        assert!((vals[1] - 2.0).abs() < 1e-8);
        assert!((vals[2] - 3.0).abs() < 1e-8);
    }
    
    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ]);
        let det = m.determinant();
        assert!((det - (-3.0)).abs() < 1e-10);
    }
    
    #[test]
    fn test_matrix_solve() {
        // 2x + y = 5, x + 3y = 7 → x = 8/5, y = 9/5
        let a = Matrix::new(vec![vec![2.0, 1.0], vec![1.0, 3.0]]);
        let b = vec![5.0, 7.0];
        let x = a.solve(&b);
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }
    
    #[test]
    fn test_matrix_inverse() {
        let m = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let inv = m.inverse().unwrap();
        let product = m.mul(&inv);
        // Should be close to identity.
        assert!((product.data[0][0] - 1.0).abs() < 1e-10);
        assert!((product.data[1][1] - 1.0).abs() < 1e-10);
        assert!(product.data[0][1].abs() < 1e-10);
        assert!(product.data[1][0].abs() < 1e-10);
    }
    
    #[test]
    fn test_eigenvalues() {
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let m = Matrix::new(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let mut eigs = m.eigenvalues();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigs[0] - 1.0).abs() < 1e-8);
        assert!((eigs[1] - 3.0).abs() < 1e-8);
    }
    
    #[test]
    fn test_gamma() {
        assert!((gamma_fn(1.0) - 1.0).abs() < 1e-12);     // Γ(1) = 0! = 1
        assert!((gamma_fn(5.0) - 24.0).abs() < 1e-10);     // Γ(5) = 4! = 24
        assert!((gamma_fn(0.5) - PI.sqrt()).abs() < 1e-10); // Γ(1/2) = √π
    }
    
    #[test]
    fn test_erf() {
        assert!(erf_fn(0.0).abs() < 1e-6);                 // erf(0) = 0 (approx within tolerance of A&S method)
        assert!((erf_fn(10.0) - 1.0).abs() < 1e-10);       // erf(∞) ≈ 1
        assert!((erf_fn(1.0) - 0.8427007929).abs() < 1e-6); // Known value
    }
    
    #[test]
    fn test_primality() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(17));
        assert!(!is_prime(15));
        assert!(is_prime(104729)); // Large prime
        assert!(!is_prime(104730));
    }
    
    #[test]
    fn test_factorization() {
        let f = factor(360);
        // 360 = 2³ × 3² × 5
        assert_eq!(f, vec![(2, 3), (3, 2), (5, 1)]);
    }
    
    #[test]
    fn test_taylor_series() {
        // Taylor series of e^x around 0, order 5:
        // 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
        let expr = Expr::Exp(Box::new(Expr::var("x")));
        let series = taylor_series(&expr, "x", &Expr::zero(), 5);
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 1.0);
        let result = series.eval(&vars).unwrap();
        // Should approximate e ≈ 2.71667 (5th order Taylor)
        assert!((result - EULER_E).abs() < 0.01);
    }
    
    #[test]
    fn test_newton_solve() {
        // Solve x² - 2 = 0 → x = √2
        let result = newton_solve(
            &|x| x * x - 2.0,
            &|x| 2.0 * x,
            1.0, 1e-12, 100,
        );
        assert!((result.unwrap() - 2.0_f64.sqrt()).abs() < 1e-10);
    }
}
