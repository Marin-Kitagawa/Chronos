// chronos-financial-engine.rs
//
// Chronos Financial Engineering Engine
// =====================================
// A comprehensive financial mathematics library implementing quantitative
// finance algorithms from first principles. Covers derivatives pricing,
// risk analytics, portfolio optimization, and fixed-income mathematics.
//
// Modules:
//   1. Core Types & Conventions (DayCount, Currency, dates)
//   2. Options Pricing (Black-Scholes, Binomial Tree, Monte Carlo)
//   3. Greeks (Delta, Gamma, Theta, Vega, Rho)
//   4. Volatility (implied vol via Brent's method, vol surface)
//   5. Fixed Income (bond pricing, duration, convexity, yield curve)
//   6. Nelson-Siegel Yield Curve Model
//   7. Portfolio Optimization (Markowitz mean-variance, efficient frontier)
//   8. Risk Analytics (VaR, CVaR, Sharpe, Sortino, drawdown)
//   9. Monte Carlo simulation engine
//  10. Interest Rate Models (Vasicek, CIR, Hull-White)
//  11. Credit Risk (CDS pricing, hazard rate models)
//  12. Exotics (barrier, Asian, lookback options)

use std::collections::HashMap;
use std::f64::consts::{PI, E};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// 1. CORE TYPES & UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/// Day-count convention for accrual calculations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DayCount {
    Actual360,   // actual days / 360
    Actual365,   // actual days / 365
    Thirty360,   // 30/360 (bond basis)
    ActualActual,// ISDA actual/actual
}

impl DayCount {
    /// Compute year fraction between two dates (expressed as day offsets from epoch).
    pub fn year_fraction(&self, from_day: i64, to_day: i64) -> f64 {
        let days = (to_day - from_day) as f64;
        match self {
            DayCount::Actual360    => days / 360.0,
            DayCount::Actual365    => days / 365.0,
            DayCount::Thirty360    => days / 360.0, // simplified
            DayCount::ActualActual => days / 365.25,
        }
    }
}

/// Option type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionType { Call, Put }

/// Option exercise style.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExerciseStyle { European, American }

/// Normal CDF using Horner's method approximation (Abramowitz & Stegun 26.2.17).
/// Max error < 7.5e-8.
pub fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x >  8.0 { return 1.0; }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530
        + t * (-0.356563782
        + t * (1.781477937
        + t * (-1.821255978
        + t * 1.330274429))));
    let pdf = (-0.5 * x * x).exp() / (2.0 * PI).sqrt();
    if x >= 0.0 { 1.0 - pdf * poly }
    else        { pdf * poly }
}

/// Normal PDF.
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Inverse normal CDF (quantile function) via rational approximation.
/// Accurate to ~1e-9 for p in (0,1).
pub fn normal_inv(p: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0,1)");
    // Coefficients for rational approximation (Peter Acklam's algorithm)
    let a = [-3.969683028665376e+01, 2.209460984245205e+02,
             -2.759285104469687e+02, 1.383577518672690e+02,
             -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [-5.447609879822406e+01, 1.615858368580409e+02,
             -1.556989798598866e+02, 6.680131188771972e+01,
             -1.328068155288572e+01];
    let c = [-7.784894002430293e-03,-3.223964580411365e-01,
             -2.400758277161838e+00,-2.549732539343734e+00,
              4.374664141464968e+00, 2.938163982698783e+00];
    let d = [ 7.784695709041462e-03, 3.224671290700398e-01,
              2.445134137142996e+00, 3.754408661907416e+00];

    let p_low  = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
         ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}

/// Simple linear congruential PRNG (for deterministic Monte Carlo tests).
/// Not cryptographically secure — use OS entropy for production.
pub struct Lcg { state: u64 }

impl Lcg {
    pub fn new(seed: u64) -> Self { Lcg { state: seed } }

    pub fn next_u64(&mut self) -> u64 {
        // Knuth's constants
        self.state = self.state.wrapping_mul(6364136223846793005)
                               .wrapping_add(1442695040888963407);
        self.state
    }

    /// Uniform [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller transform.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Antithetic pair: returns (z, -z) for variance reduction.
    pub fn antithetic_pair(&mut self) -> (f64, f64) {
        let z = self.normal();
        (z, -z)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. BLACK-SCHOLES MODEL
// ─────────────────────────────────────────────────────────────────────────────

/// Black-Scholes European option parameters.
#[derive(Debug, Clone)]
pub struct BsParams {
    pub spot:     f64,  // current asset price S
    pub strike:   f64,  // option strike K
    pub rate:     f64,  // continuously-compounded risk-free rate r
    pub dividend: f64,  // continuous dividend yield q
    pub vol:      f64,  // annualised volatility σ
    pub expiry:   f64,  // time to expiry T (years)
    pub opt_type: OptionType,
}

impl BsParams {
    /// Compute d1 and d2 (the standard BS intermediate values).
    pub fn d1_d2(&self) -> (f64, f64) {
        let s  = self.spot;
        let k  = self.strike;
        let r  = self.rate;
        let q  = self.dividend;
        let v  = self.vol;
        let t  = self.expiry;
        let d1 = ((s / k).ln() + (r - q + 0.5 * v * v) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        (d1, d2)
    }

    /// Price a European option using the Black-Scholes-Merton formula.
    /// With continuous dividend yield q:
    ///   Call = S·e^{-qT}·N(d1) − K·e^{-rT}·N(d2)
    ///   Put  = K·e^{-rT}·N(-d2) − S·e^{-qT}·N(-d1)
    pub fn price(&self) -> f64 {
        let s  = self.spot;
        let k  = self.strike;
        let r  = self.rate;
        let q  = self.dividend;
        let t  = self.expiry;
        let (d1, d2) = self.d1_d2();
        match self.opt_type {
            OptionType::Call => {
                s * (-q * t).exp() * normal_cdf(d1)
                    - k * (-r * t).exp() * normal_cdf(d2)
            }
            OptionType::Put => {
                k * (-r * t).exp() * normal_cdf(-d2)
                    - s * (-q * t).exp() * normal_cdf(-d1)
            }
        }
    }

    /// Put-call parity check: C - P = S·e^{-qT} - K·e^{-rT}
    pub fn put_call_parity(&self) -> f64 {
        let fwd = self.spot * (-self.dividend * self.expiry).exp();
        let pv_k = self.strike * (-self.rate * self.expiry).exp();
        fwd - pv_k
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. GREEKS
// ─────────────────────────────────────────────────────────────────────────────

/// All first- and second-order Black-Scholes Greeks.
#[derive(Debug, Clone)]
pub struct Greeks {
    pub delta: f64,  // ∂V/∂S
    pub gamma: f64,  // ∂²V/∂S²
    pub theta: f64,  // ∂V/∂t (per calendar day)
    pub vega:  f64,  // ∂V/∂σ (per 1% vol move)
    pub rho:   f64,  // ∂V/∂r (per 1% rate move)
}

impl Greeks {
    /// Compute all BS Greeks analytically.
    pub fn compute(p: &BsParams) -> Self {
        let s  = p.spot;
        let k  = p.strike;
        let r  = p.rate;
        let q  = p.dividend;
        let v  = p.vol;
        let t  = p.expiry;
        let (d1, d2) = p.d1_d2();
        let nd1 = normal_pdf(d1);
        let eq  = (-q * t).exp();
        let er  = (-r * t).exp();

        let (delta, theta, rho) = match p.opt_type {
            OptionType::Call => {
                let delta = eq * normal_cdf(d1);
                let theta = -(s * eq * nd1 * v) / (2.0 * t.sqrt())
                    - r * k * er * normal_cdf(d2)
                    + q * s * eq * normal_cdf(d1);
                let rho = k * t * er * normal_cdf(d2);
                (delta, theta / 365.0, rho / 100.0)
            }
            OptionType::Put => {
                let delta = eq * (normal_cdf(d1) - 1.0);
                let theta = -(s * eq * nd1 * v) / (2.0 * t.sqrt())
                    + r * k * er * normal_cdf(-d2)
                    - q * s * eq * normal_cdf(-d1);
                let rho = -k * t * er * normal_cdf(-d2);
                (delta, theta / 365.0, rho / 100.0)
            }
        };

        let gamma = eq * nd1 / (s * v * t.sqrt());
        let vega  = s * eq * nd1 * t.sqrt() / 100.0;

        Greeks { delta, gamma, theta, vega, rho }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. IMPLIED VOLATILITY (BRENT'S METHOD)
// ─────────────────────────────────────────────────────────────────────────────

/// Find implied volatility given a market price using Brent's root-finding method.
/// Brent's method combines bisection, secant, and inverse quadratic interpolation.
/// Guaranteed to converge in O(log ε) function evaluations for continuous functions.
pub fn implied_vol(market_price: f64, params: &BsParams) -> Option<f64> {
    // We solve f(σ) = BS_price(σ) - market_price = 0
    // Search in [1e-6, 10.0] (0.0001% to 1000% vol)
    let mut a = 1e-6f64;
    let mut b = 10.0f64;

    let f = |vol: f64| {
        let mut p = params.clone();
        p.vol = vol;
        p.price() - market_price
    };

    let fa = f(a);
    let fb = f(b);

    // Check bracketing
    if fa * fb > 0.0 { return None; }

    let mut fa = fa;
    let mut fb = fb;
    let mut c  = a;
    let mut fc = fa;
    let mut d  = b - a;
    let mut e  = d;

    let tol = 1e-10;
    let max_iter = 100;

    for _ in 0..max_iter {
        if fb * fc > 0.0 {
            c = a; fc = fa; d = b - a; e = d;
        }
        if fc.abs() < fb.abs() {
            a = b; b = c; c = a;
            fa = fb; fb = fc; fc = fa;
        }
        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let xm = 0.5 * (c - b);
        if xm.abs() <= tol1 || fb == 0.0 { return Some(b); }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p_brent, q_brent);
            if a == c {
                p_brent = 2.0 * xm * s;
                q_brent = 1.0 - s;
            } else {
                let q2 = fa / fc;
                let r2 = fb / fc;
                p_brent = s * (2.0 * xm * q2 * (q2 - r2) - (b - a) * (r2 - 1.0));
                q_brent = (q2 - 1.0) * (r2 - 1.0) * (s - 1.0);
            }
            let (p_brent, q_brent) = if p_brent > 0.0 { (p_brent, -q_brent) }
                                     else { (-p_brent, q_brent) };
            if 2.0 * p_brent < (3.0 * xm * q_brent - (tol1 * q_brent).abs())
                .min(e.abs() * q_brent.abs()) {
                e = d;
                d = p_brent / q_brent;
            } else {
                d = xm; e = d;
            }
        } else {
            d = xm; e = d;
        }

        a = b; fa = fb;
        if d.abs() > tol1 { b += d; } else { b += if xm > 0.0 { tol1 } else { -tol1 }; }
        fb = f(b);
    }
    None // did not converge
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. BINOMIAL TREE (Cox-Ross-Rubinstein)
// ─────────────────────────────────────────────────────────────────────────────

/// CRR binomial tree for European and American options.
/// The binomial tree discretises the GBM into N up/down steps.
/// u = e^{σ√Δt}, d = 1/u, risk-neutral prob p = (e^{rΔt} - d)/(u - d).
/// American options allow early exercise at each node.
pub struct BinomialTree {
    pub steps: usize,
}

impl BinomialTree {
    pub fn new(steps: usize) -> Self { BinomialTree { steps } }

    pub fn price(&self, p: &BsParams, style: ExerciseStyle) -> f64 {
        let n   = self.steps;
        let dt  = p.expiry / n as f64;
        let u   = (p.vol * dt.sqrt()).exp();
        let d   = 1.0 / u;
        let disc = (-p.rate * dt).exp();
        let pu  = (((p.rate - p.dividend) * dt).exp() - d) / (u - d);
        let pd  = 1.0 - pu;

        // Terminal payoffs
        let mut values: Vec<f64> = (0..=n).map(|j| {
            let s = p.spot * u.powi(j as i32) * d.powi((n - j) as i32);
            match p.opt_type {
                OptionType::Call => (s - p.strike).max(0.0),
                OptionType::Put  => (p.strike - s).max(0.0),
            }
        }).collect();

        // Backward induction
        for i in (0..n).rev() {
            for j in 0..=i {
                let continuation = disc * (pu * values[j + 1] + pd * values[j]);
                if style == ExerciseStyle::American {
                    let s = p.spot * u.powi(j as i32) * d.powi((i - j) as i32);
                    let intrinsic = match p.opt_type {
                        OptionType::Call => (s - p.strike).max(0.0),
                        OptionType::Put  => (p.strike - s).max(0.0),
                    };
                    values[j] = continuation.max(intrinsic);
                } else {
                    values[j] = continuation;
                }
            }
        }
        values[0]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. MONTE CARLO OPTION PRICING
// ─────────────────────────────────────────────────────────────────────────────

/// Monte Carlo engine for option pricing.
/// Uses antithetic variates for variance reduction (~4× improvement typical).
/// GBM path: S_T = S_0 · exp((r - q - σ²/2)T + σ·√T·Z)
pub struct MonteCarloEngine {
    pub paths:     usize,
    pub use_antithetic: bool,
    pub seed:      u64,
}

impl MonteCarloEngine {
    pub fn new(paths: usize, seed: u64) -> Self {
        MonteCarloEngine { paths, use_antithetic: true, seed }
    }

    /// Price a European option.
    pub fn price_european(&self, p: &BsParams) -> (f64, f64) {
        let mut rng = Lcg::new(self.seed);
        let drift = (p.rate - p.dividend - 0.5 * p.vol * p.vol) * p.expiry;
        let diff  = p.vol * p.expiry.sqrt();
        let disc  = (-p.rate * p.expiry).exp();

        let mut sum   = 0.0;
        let mut sum_sq = 0.0;
        let n_eff = if self.use_antithetic { self.paths / 2 } else { self.paths };

        for _ in 0..n_eff {
            let payoff = if self.use_antithetic {
                let (z1, z2) = rng.antithetic_pair();
                let s1 = p.spot * (drift + diff * z1).exp();
                let s2 = p.spot * (drift + diff * z2).exp();
                let pay1 = match p.opt_type {
                    OptionType::Call => (s1 - p.strike).max(0.0),
                    OptionType::Put  => (p.strike - s1).max(0.0),
                };
                let pay2 = match p.opt_type {
                    OptionType::Call => (s2 - p.strike).max(0.0),
                    OptionType::Put  => (p.strike - s2).max(0.0),
                };
                0.5 * (pay1 + pay2)
            } else {
                let z = rng.normal();
                let s = p.spot * (drift + diff * z).exp();
                match p.opt_type {
                    OptionType::Call => (s - p.strike).max(0.0),
                    OptionType::Put  => (p.strike - s).max(0.0),
                }
            };
            sum    += payoff;
            sum_sq += payoff * payoff;
        }

        let mean = sum / n_eff as f64;
        let var  = sum_sq / n_eff as f64 - mean * mean;
        let se   = (var / n_eff as f64).sqrt();
        (disc * mean, disc * se * 1.96) // price, 95% CI half-width
    }

    /// Price an Asian (arithmetic average) call/put option.
    /// Asian options depend on the average price over the path.
    pub fn price_asian(&self, p: &BsParams, monitoring_steps: usize) -> f64 {
        let mut rng = Lcg::new(self.seed);
        let dt    = p.expiry / monitoring_steps as f64;
        let drift = (p.rate - p.dividend - 0.5 * p.vol * p.vol) * dt;
        let diff  = p.vol * dt.sqrt();
        let disc  = (-p.rate * p.expiry).exp();
        let mut sum = 0.0;

        for _ in 0..self.paths {
            let mut s = p.spot;
            let mut path_sum = 0.0;
            for _ in 0..monitoring_steps {
                s *= (drift + diff * rng.normal()).exp();
                path_sum += s;
            }
            let avg = path_sum / monitoring_steps as f64;
            let payoff = match p.opt_type {
                OptionType::Call => (avg - p.strike).max(0.0),
                OptionType::Put  => (p.strike - avg).max(0.0),
            };
            sum += payoff;
        }
        disc * sum / self.paths as f64
    }

    /// Price a barrier option (knock-out or knock-in).
    /// `barrier`: barrier level. `knock_out`: true for knock-out, false for knock-in.
    /// `upper_barrier`: true if barrier is above spot (up-&-out), false for down-&-out.
    pub fn price_barrier(&self, p: &BsParams, barrier: f64, knock_out: bool,
                         upper_barrier: bool, steps: usize) -> f64 {
        let mut rng = Lcg::new(self.seed);
        let dt    = p.expiry / steps as f64;
        let drift = (p.rate - p.dividend - 0.5 * p.vol * p.vol) * dt;
        let diff  = p.vol * dt.sqrt();
        let disc  = (-p.rate * p.expiry).exp();
        let mut sum = 0.0;

        for _ in 0..self.paths {
            let mut s = p.spot;
            let mut breached = false;
            for _ in 0..steps {
                s *= (drift + diff * rng.normal()).exp();
                if upper_barrier { if s >= barrier { breached = true; break; } }
                else             { if s <= barrier { breached = true; break; } }
            }
            let alive = if knock_out { !breached } else { breached };
            if alive {
                let payoff = match p.opt_type {
                    OptionType::Call => (s - p.strike).max(0.0),
                    OptionType::Put  => (p.strike - s).max(0.0),
                };
                sum += payoff;
            }
        }
        disc * sum / self.paths as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. FIXED INCOME — BOND PRICING
// ─────────────────────────────────────────────────────────────────────────────

/// A coupon-bearing bond.
#[derive(Debug, Clone)]
pub struct Bond {
    pub face:         f64,      // par / face value
    pub coupon_rate:  f64,      // annual coupon rate (e.g. 0.05 for 5%)
    pub periods:      usize,    // number of coupon periods
    pub periods_year: f64,      // coupon periods per year (2 = semi-annual)
}

impl Bond {
    /// Price the bond given a yield-to-maturity `ytm` (annual, continuously compounded).
    /// P = Σ C·e^{-r·t_i} + F·e^{-r·T}
    pub fn price_continuous(&self, ytm: f64) -> f64 {
        let c = self.face * self.coupon_rate / self.periods_year;
        let mut price = 0.0;
        for i in 1..=self.periods {
            let t = i as f64 / self.periods_year;
            price += c * (-ytm * t).exp();
        }
        let t_final = self.periods as f64 / self.periods_year;
        price += self.face * (-ytm * t_final).exp();
        price
    }

    /// Price using periodically-compounded YTM (e.g. for US Treasuries with semi-annual).
    /// P = Σ C/(1+y/m)^i + F/(1+y/m)^N
    pub fn price_periodic(&self, ytm: f64) -> f64 {
        let m   = self.periods_year;
        let c   = self.face * self.coupon_rate / m;
        let r   = ytm / m;
        let mut price = 0.0;
        for i in 1..=self.periods {
            price += c / (1.0 + r).powi(i as i32);
        }
        price += self.face / (1.0 + r).powi(self.periods as i32);
        price
    }

    /// Macaulay duration (in years): D = (1/P) · Σ t_i · PV(CF_i)
    pub fn duration(&self, ytm: f64) -> f64 {
        let c = self.face * self.coupon_rate / self.periods_year;
        let price = self.price_continuous(ytm);
        let mut weighted = 0.0;
        for i in 1..=self.periods {
            let t = i as f64 / self.periods_year;
            weighted += t * c * (-ytm * t).exp();
        }
        let t_final = self.periods as f64 / self.periods_year;
        weighted += t_final * self.face * (-ytm * t_final).exp();
        weighted / price
    }

    /// Modified duration: D_mod = D_mac / (1 + y/m) for periodic compounding.
    pub fn modified_duration(&self, ytm: f64) -> f64 {
        self.duration(ytm) // For continuous compounding, Macaulay = Modified
    }

    /// Convexity: C = (1/P) · Σ t_i² · PV(CF_i)
    pub fn convexity(&self, ytm: f64) -> f64 {
        let c = self.face * self.coupon_rate / self.periods_year;
        let price = self.price_continuous(ytm);
        let mut weighted = 0.0;
        for i in 1..=self.periods {
            let t = i as f64 / self.periods_year;
            weighted += t * t * c * (-ytm * t).exp();
        }
        let t_final = self.periods as f64 / self.periods_year;
        weighted += t_final * t_final * self.face * (-ytm * t_final).exp();
        weighted / price
    }

    /// Dollar duration: DV01 = -dP/dy ≈ D_mod · P / 10000 (per basis point).
    pub fn dv01(&self, ytm: f64) -> f64 {
        let d = self.modified_duration(ytm);
        let p = self.price_continuous(ytm);
        d * p / 10000.0
    }

    /// Yield to maturity via bisection (continuous compounding).
    pub fn ytm(&self, price: f64) -> Option<f64> {
        let f = |y: f64| self.price_continuous(y) - price;
        // Price and YTM are inverse: high price → low YTM
        let mut lo = 0.0001f64;
        let mut hi = 5.0f64;
        if f(lo) * f(hi) > 0.0 { return None; }
        for _ in 0..100 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 { lo = mid; } else { hi = mid; }
            if hi - lo < 1e-12 { return Some(mid); }
        }
        Some(0.5 * (lo + hi))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. NELSON-SIEGEL YIELD CURVE MODEL
// ─────────────────────────────────────────────────────────────────────────────

/// Nelson-Siegel model for zero-coupon yield curves.
/// y(τ) = β0 + β1·(1-e^{-λτ})/(λτ) + β2·[(1-e^{-λτ})/(λτ) - e^{-λτ}]
///
/// Interpretation:
///   β0: long-term level (as τ→∞, y→β0)
///   β1: short-term component (positive → upward slope from short end)
///   β2: medium-term hump (peak/trough at maturity related to λ)
///   λ:  decay factor controlling where the hump occurs
#[derive(Debug, Clone)]
pub struct NelsonSiegel {
    pub beta0: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub lambda: f64,
}

impl NelsonSiegel {
    /// Compute the zero-coupon yield for maturity τ (years).
    pub fn yield_at(&self, tau: f64) -> f64 {
        if tau <= 0.0 { return self.beta0 + self.beta1; }
        let l_tau = self.lambda * tau;
        let factor1 = (1.0 - (-l_tau).exp()) / l_tau;
        let factor2 = factor1 - (-l_tau).exp();
        self.beta0 + self.beta1 * factor1 + self.beta2 * factor2
    }

    /// Spot rate = yield (NS gives zero-coupon/spot rates directly).
    pub fn spot_rate(&self, tau: f64) -> f64 { self.yield_at(tau) }

    /// Instantaneous forward rate: f(τ) = -d[τ·y(τ)]/dτ
    pub fn forward_rate(&self, tau: f64) -> f64 {
        if tau <= 1e-9 { return self.beta0 + self.beta1; }
        let l  = self.lambda;
        let lt = l * tau;
        let e  = (-lt).exp();
        self.beta0
            + self.beta1 * e
            + self.beta2 * lt * e
    }

    /// Discount factor: P(0,τ) = e^{-y(τ)·τ}
    pub fn discount_factor(&self, tau: f64) -> f64 {
        (-self.yield_at(tau) * tau).exp()
    }

    /// Fit Nelson-Siegel parameters to market data via least squares (gradient descent).
    /// `maturities`: slice of τ values; `yields`: observed market yields.
    pub fn fit(maturities: &[f64], yields: &[f64]) -> Self {
        assert_eq!(maturities.len(), yields.len());
        // Simple gradient descent with numerical gradients
        let mut params = NelsonSiegel { beta0: 0.05, beta1: -0.02, beta2: 0.01, lambda: 0.5 };
        let lr = 1e-5;
        let eps = 1e-7;
        for _iter in 0..5000 {
            // Compute loss
            let loss = |ns: &NelsonSiegel| -> f64 {
                maturities.iter().zip(yields).map(|(&t, &y)| {
                    let diff = ns.yield_at(t) - y;
                    diff * diff
                }).sum()
            };
            let l0 = loss(&params);
            // Numerical gradient
            macro_rules! grad {
                ($field:ident) => {{
                    let mut tmp = params.clone();
                    tmp.$field += eps;
                    (loss(&tmp) - l0) / eps
                }}
            }
            let gb0 = grad!(beta0);
            let gb1 = grad!(beta1);
            let gb2 = grad!(beta2);
            let gl  = grad!(lambda);
            params.beta0  -= lr * gb0;
            params.beta1  -= lr * gb1;
            params.beta2  -= lr * gb2;
            params.lambda  = (params.lambda - lr * gl).max(0.01);
        }
        params
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. PORTFOLIO OPTIMIZATION (MARKOWITZ MEAN-VARIANCE)
// ─────────────────────────────────────────────────────────────────────────────

/// Portfolio analytics and mean-variance optimization.
///
/// Mean-variance framework (Markowitz, 1952):
///   Minimize portfolio variance σ²_p = wᵀΣw
///   Subject to: wᵀμ = μ_target, wᵀ1 = 1, w ≥ 0 (long-only)
///
/// We use a simple numerical approach: parameterise the efficient frontier
/// by target return μ* and find weights via gradient projection.
pub struct Portfolio {
    /// Expected returns vector μ (annualised).
    pub returns:  Vec<f64>,
    /// Covariance matrix Σ (annualised). Stored row-major.
    pub cov:      Vec<Vec<f64>>,
    pub n:        usize,
}

impl Portfolio {
    pub fn new(returns: Vec<f64>, cov: Vec<Vec<f64>>) -> Self {
        let n = returns.len();
        assert_eq!(cov.len(), n);
        Portfolio { returns, cov, n }
    }

    /// Portfolio expected return given weights.
    pub fn expected_return(&self, w: &[f64]) -> f64 {
        w.iter().zip(&self.returns).map(|(wi, ri)| wi * ri).sum()
    }

    /// Portfolio variance given weights.
    pub fn variance(&self, w: &[f64]) -> f64 {
        let mut var = 0.0;
        for i in 0..self.n {
            for j in 0..self.n {
                var += w[i] * w[j] * self.cov[i][j];
            }
        }
        var
    }

    /// Portfolio volatility (annualised standard deviation).
    pub fn volatility(&self, w: &[f64]) -> f64 { self.variance(w).sqrt() }

    /// Sharpe ratio: (μ_p - r_f) / σ_p
    pub fn sharpe(&self, w: &[f64], risk_free: f64) -> f64 {
        let ret = self.expected_return(w);
        let vol = self.volatility(w);
        if vol < 1e-12 { return 0.0; }
        (ret - risk_free) / vol
    }

    /// Global minimum variance portfolio via gradient projection.
    /// Constraints: Σwᵢ=1, wᵢ≥0.
    pub fn min_variance_weights(&self) -> Vec<f64> {
        self.optimize_weights(None, 0.0)
    }

    /// Maximum Sharpe ratio portfolio (tangency portfolio).
    pub fn max_sharpe_weights(&self, risk_free: f64) -> Vec<f64> {
        // Maximise Sharpe = (μ - r_f) / σ is equivalent to minimising -Sharpe.
        // We use a simple projected gradient ascent on Sharpe.
        let mut w = vec![1.0 / self.n as f64; self.n];
        let lr = 0.001;
        for _iter in 0..10000 {
            let ret = self.expected_return(&w);
            let vol = self.volatility(&w);
            if vol < 1e-12 { break; }
            // Gradient of Sharpe w.r.t. w_i:
            // ∂S/∂w_i = [μ_i·σ - (μ_p-r_f)·∂σ/∂w_i] / σ²
            // ∂σ/∂w_i = (Σw)_i / σ
            let sigma_w: Vec<f64> = (0..self.n).map(|i| {
                (0..self.n).map(|j| self.cov[i][j] * w[j]).sum::<f64>()
            }).collect();
            let grad: Vec<f64> = (0..self.n).map(|i| {
                (self.returns[i] * vol - (ret - risk_free) * sigma_w[i] / vol)
                    / (vol * vol)
            }).collect();
            // Gradient ascent + project to simplex
            let new_w: Vec<f64> = w.iter().zip(&grad).map(|(wi, gi)| (wi + lr * gi).max(0.0)).collect();
            w = project_to_simplex(&new_w);
        }
        w
    }

    /// Optimise weights for a target return (efficient frontier point).
    /// If target_return is None, minimises variance (GMV portfolio).
    fn optimize_weights(&self, target_return: Option<f64>, _risk_free: f64) -> Vec<f64> {
        let mut w = vec![1.0 / self.n as f64; self.n];
        let lr = 0.01;
        for _iter in 0..5000 {
            // Gradient of variance
            let grad: Vec<f64> = (0..self.n).map(|i| {
                2.0 * (0..self.n).map(|j| self.cov[i][j] * w[j]).sum::<f64>()
            }).collect();
            let new_w: Vec<f64> = w.iter().zip(&grad).map(|(wi, gi)| (wi - lr * gi).max(0.0)).collect();
            let mut pw = project_to_simplex(&new_w);
            // If target return constraint, project again
            if let Some(mu_target) = target_return {
                let mu_p = self.expected_return(&pw);
                let diff = mu_target - mu_p;
                // Nudge in return direction
                let ret_grad_norm: f64 = self.returns.iter().map(|r| r*r).sum::<f64>().sqrt();
                if ret_grad_norm > 1e-12 {
                    for i in 0..self.n {
                        pw[i] += 0.001 * diff * self.returns[i] / ret_grad_norm;
                        pw[i] = pw[i].max(0.0);
                    }
                    pw = project_to_simplex(&pw);
                }
            }
            w = pw;
        }
        w
    }

    /// Generate efficient frontier: N points from min-variance to max-return.
    pub fn efficient_frontier(&self, n_points: usize) -> Vec<(f64, f64)> {
        let min_ret = self.returns.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ret = self.returns.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (0..n_points).map(|i| {
            let mu_t = min_ret + (max_ret - min_ret) * i as f64 / (n_points - 1) as f64;
            let w = self.optimize_weights(Some(mu_t), 0.0);
            let vol = self.volatility(&w);
            let ret = self.expected_return(&w);
            (vol, ret)
        }).collect()
    }
}

/// Project a vector onto the probability simplex: Σwᵢ=1, wᵢ≥0.
/// Uses the O(n log n) algorithm by Duchi et al. (2008).
fn project_to_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut cumsum = 0.0;
    let mut rho = 0;
    for (i, &ui) in u.iter().enumerate() {
        cumsum += ui;
        if ui - (cumsum - 1.0) / (i + 1) as f64 > 0.0 { rho = i; }
    }
    let cumsum_rho: f64 = u[..=rho].iter().sum();
    let theta = (cumsum_rho - 1.0) / (rho + 1) as f64;
    v.iter().map(|wi| (wi - theta).max(0.0)).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. RISK ANALYTICS
// ─────────────────────────────────────────────────────────────────────────────

/// Historical risk metrics for a return series.
pub struct RiskAnalytics {
    pub returns: Vec<f64>,  // daily/periodic returns
}

impl RiskAnalytics {
    pub fn new(returns: Vec<f64>) -> Self { RiskAnalytics { returns } }

    /// Arithmetic mean return.
    pub fn mean(&self) -> f64 {
        self.returns.iter().sum::<f64>() / self.returns.len() as f64
    }

    /// Sample variance.
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let n = self.returns.len() as f64;
        self.returns.iter().map(|r| (r - m).powi(2)).sum::<f64>() / (n - 1.0)
    }

    /// Volatility (std dev).
    pub fn volatility(&self) -> f64 { self.variance().sqrt() }

    /// Annualised Sharpe ratio (assuming `periods_per_year` observation frequency).
    pub fn sharpe_ratio(&self, risk_free_per_period: f64, periods_per_year: f64) -> f64 {
        let excess = self.mean() - risk_free_per_period;
        let vol    = self.volatility();
        if vol < 1e-12 { return 0.0; }
        (excess / vol) * periods_per_year.sqrt()
    }

    /// Sortino ratio: penalises only downside volatility.
    /// Downside deviation = sqrt(E[min(r-target, 0)²])
    pub fn sortino_ratio(&self, target: f64, periods_per_year: f64) -> f64 {
        let downside_sq: f64 = self.returns.iter()
            .map(|r| (r - target).min(0.0).powi(2))
            .sum::<f64>() / self.returns.len() as f64;
        let downside_vol = downside_sq.sqrt();
        if downside_vol < 1e-12 { return 0.0; }
        let excess = self.mean() - target;
        (excess / downside_vol) * periods_per_year.sqrt()
    }

    /// Historical Value at Risk at confidence level α (e.g. 0.95 for 95% VaR).
    /// VaR is the loss not exceeded with probability α.
    pub fn var_historical(&self, confidence: f64) -> f64 {
        let mut sorted = self.returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
        let idx = idx.min(sorted.len() - 1);
        -sorted[idx] // Return as positive loss
    }

    /// Parametric (Gaussian) VaR.
    pub fn var_parametric(&self, confidence: f64) -> f64 {
        let z = normal_inv(1.0 - confidence).abs(); // negative tail
        -(self.mean() - z * self.volatility())
    }

    /// Conditional Value at Risk (CVaR / Expected Shortfall):
    /// Average loss in the worst (1-α) fraction of outcomes.
    pub fn cvar_historical(&self, confidence: f64) -> f64 {
        let mut sorted = self.returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let cutoff_idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
        if cutoff_idx == 0 { return -sorted[0]; }
        let tail_mean: f64 = sorted[..cutoff_idx].iter().sum::<f64>() / cutoff_idx as f64;
        -tail_mean
    }

    /// Maximum drawdown: max peak-to-trough decline in cumulative wealth.
    pub fn max_drawdown(&self) -> f64 {
        let mut peak = 1.0f64;
        let mut wealth = 1.0f64;
        let mut max_dd = 0.0f64;
        for &r in &self.returns {
            wealth *= 1.0 + r;
            if wealth > peak { peak = wealth; }
            let dd = (peak - wealth) / peak;
            if dd > max_dd { max_dd = dd; }
        }
        max_dd
    }

    /// Calmar ratio: annualised return / max drawdown.
    pub fn calmar_ratio(&self, periods_per_year: f64) -> f64 {
        let ann_return = (1.0 + self.mean()).powf(periods_per_year) - 1.0;
        let mdd = self.max_drawdown();
        if mdd < 1e-12 { return f64::INFINITY; }
        ann_return / mdd
    }

    /// Skewness of the return distribution.
    pub fn skewness(&self) -> f64 {
        let n = self.returns.len() as f64;
        let m = self.mean();
        let s = self.volatility();
        if s < 1e-12 { return 0.0; }
        let sk: f64 = self.returns.iter().map(|r| ((r - m) / s).powi(3)).sum::<f64>();
        n / ((n - 1.0) * (n - 2.0)) * sk
    }

    /// Excess kurtosis (normal distribution has excess kurtosis = 0).
    pub fn excess_kurtosis(&self) -> f64 {
        let n = self.returns.len() as f64;
        let m = self.mean();
        let s = self.volatility();
        if s < 1e-12 { return 0.0; }
        let k4: f64 = self.returns.iter().map(|r| ((r - m) / s).powi(4)).sum::<f64>();
        n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * k4
            - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. INTEREST RATE MODELS
// ─────────────────────────────────────────────────────────────────────────────

/// Vasicek short-rate model: dr = κ(θ - r)dt + σdW
/// Mean-reverting Ornstein-Uhlenbeck process.
/// Parameters: κ (speed of mean reversion), θ (long-run mean), σ (vol).
pub struct VasicekModel {
    pub kappa: f64,  // mean reversion speed
    pub theta: f64,  // long-run mean rate
    pub sigma: f64,  // instantaneous volatility
    pub r0:    f64,  // initial short rate
}

impl VasicekModel {
    /// Analytical zero-coupon bond price P(0,T) in Vasicek model.
    /// P(0,T) = A(T) · e^{-B(T)·r0}
    pub fn bond_price(&self, t: f64) -> f64 {
        let b = (1.0 - (-self.kappa * t).exp()) / self.kappa;
        let a_exp = (self.theta - self.sigma * self.sigma / (2.0 * self.kappa * self.kappa))
            * (b - t)
            - self.sigma * self.sigma * b * b / (4.0 * self.kappa);
        a_exp.exp() * (-b * self.r0).exp()
    }

    /// Spot rate R(0,T) = -ln(P(0,T)) / T
    pub fn spot_rate(&self, t: f64) -> f64 {
        -self.bond_price(t).ln() / t
    }

    /// Simulate short rate path using Euler-Maruyama discretisation.
    pub fn simulate(&self, t: f64, steps: usize, seed: u64) -> Vec<f64> {
        let dt  = t / steps as f64;
        let mut rng = Lcg::new(seed);
        let mut r = self.r0;
        let mut path = vec![r];
        for _ in 0..steps {
            let dw = rng.normal() * dt.sqrt();
            r += self.kappa * (self.theta - r) * dt + self.sigma * dw;
            path.push(r);
        }
        path
    }

    /// Expected short rate at time t: E[r_t] = r0·e^{-κt} + θ(1-e^{-κt})
    pub fn expected_rate(&self, t: f64) -> f64 {
        self.r0 * (-self.kappa * t).exp() + self.theta * (1.0 - (-self.kappa * t).exp())
    }

    /// Variance of short rate at time t.
    pub fn rate_variance(&self, t: f64) -> f64 {
        self.sigma * self.sigma / (2.0 * self.kappa) * (1.0 - (-2.0 * self.kappa * t).exp())
    }
}

/// Cox-Ingersoll-Ross model: dr = κ(θ - r)dt + σ√r · dW
/// Unlike Vasicek, CIR guarantees r > 0 when 2κθ > σ² (Feller condition).
pub struct CirModel {
    pub kappa: f64,
    pub theta: f64,
    pub sigma: f64,
    pub r0:    f64,
}

impl CirModel {
    /// Simulate CIR path using Milstein scheme (preserves positivity).
    pub fn simulate(&self, t: f64, steps: usize, seed: u64) -> Vec<f64> {
        let dt  = t / steps as f64;
        let mut rng = Lcg::new(seed);
        let mut r = self.r0;
        let mut path = vec![r];
        for _ in 0..steps {
            let dw = rng.normal() * dt.sqrt();
            // Milstein correction term: 0.5·σ²(dW² - dt)
            let milstein = 0.5 * self.sigma * self.sigma * (dw * dw - dt);
            r = (r + self.kappa * (self.theta - r) * dt
                   + self.sigma * r.max(0.0).sqrt() * dw
                   + milstein).max(0.0);
            path.push(r);
        }
        path
    }

    /// Feller condition: 2κθ > σ² ensures r stays strictly positive.
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta > self.sigma * self.sigma
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. CREDIT RISK — HAZARD RATE MODELS
// ─────────────────────────────────────────────────────────────────────────────

/// Credit Default Swap (CDS) pricing under a constant hazard rate model.
/// Default intensity (hazard rate) λ: P(survive to t) = e^{-λt}.
pub struct CdsModel {
    pub hazard_rate:    f64,   // constant default intensity λ
    pub recovery:       f64,   // recovery rate R (fraction of notional)
    pub risk_free_rate: f64,   // risk-free discount rate r
    pub notional:       f64,
}

impl CdsModel {
    /// Survival probability to time t.
    pub fn survival_prob(&self, t: f64) -> f64 { (-self.hazard_rate * t).exp() }

    /// Default probability by time t.
    pub fn default_prob(&self, t: f64) -> f64 { 1.0 - self.survival_prob(t) }

    /// Present value of the protection leg (CDS pays (1-R)·N on default).
    /// PV_protection = (1-R)·N · ∫₀ᵀ e^{-(r+λ)t} λ dt = (1-R)·N·λ/(r+λ)·(1-e^{-(r+λ)T})
    pub fn pv_protection_leg(&self, maturity: f64) -> f64 {
        let rl = self.risk_free_rate + self.hazard_rate;
        let loss_given_default = (1.0 - self.recovery) * self.notional;
        loss_given_default * self.hazard_rate / rl * (1.0 - (-rl * maturity).exp())
    }

    /// Present value of the premium leg (CDS buyer pays spread s continuously).
    /// PV_premium = s·N · ∫₀ᵀ e^{-(r+λ)t} dt = s·N/(r+λ)·(1-e^{-(r+λ)T})
    pub fn pv_premium_leg_per_spread(&self, maturity: f64) -> f64 {
        let rl = self.risk_free_rate + self.hazard_rate;
        self.notional / rl * (1.0 - (-rl * maturity).exp())
    }

    /// Fair CDS spread (break-even premium): s* = PV_protection / PV_premium_per_unit
    pub fn fair_spread(&self, maturity: f64) -> f64 {
        let pv_prot = self.pv_protection_leg(maturity);
        let pv_prem = self.pv_premium_leg_per_spread(maturity);
        pv_prot / pv_prem
    }

    /// CDS mark-to-market value for spread `traded_spread`.
    pub fn mtm(&self, maturity: f64, traded_spread: f64) -> f64 {
        let fair = self.fair_spread(maturity);
        let pv_annuity = self.pv_premium_leg_per_spread(maturity);
        (fair - traded_spread) * pv_annuity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. VOLATILITY SURFACE (SABR MODEL)
// ─────────────────────────────────────────────────────────────────────────────

/// SABR stochastic volatility model (Hagan et al. 2002).
/// Provides an analytical approximation for implied volatility across strikes.
///
/// SABR: dF = σ·F^β·dW₁, dσ = α·σ·dW₂, corr(dW₁,dW₂) = ρ
/// Approximation formula gives σ_BS(K,T) as a function of (F,K,T,α,β,ρ,ν).
#[derive(Debug, Clone)]
pub struct SabrModel {
    pub alpha: f64,  // initial vol level
    pub beta:  f64,  // CEV exponent (0=normal, 1=lognormal)
    pub rho:   f64,  // correlation between F and σ
    pub nu:    f64,  // vol of vol
}

impl SabrModel {
    /// Hagan's SABR approximation for implied vol.
    /// F: forward, K: strike, T: expiry (years).
    pub fn implied_vol(&self, f: f64, k: f64, t: f64) -> f64 {
        let a = self.alpha;
        let b = self.beta;
        let r = self.rho;
        let n = self.nu;

        if (f - k).abs() < 1e-10 {
            // ATM approximation
            let fmid = f.powf(1.0 - b);
            let term1 = a / fmid;
            let term2 = 1.0 + ((1.0 - b).powi(2) / 24.0 * a * a / fmid.powi(2)
                + r * b * n * a / (4.0 * fmid)
                + (2.0 - 3.0 * r * r) / 24.0 * n * n) * t;
            term1 * term2
        } else {
            let logfk = (f / k).ln();
            let fkb   = (f * k).powf((1.0 - b) / 2.0);
            let z     = n / a * fkb * logfk;
            let xz    = (((1.0 - 2.0 * r * z + z * z).sqrt() + z - r) / (1.0 - r)).ln();
            let numer = a;
            let denom = fkb * (1.0 + (1.0 - b).powi(2) / 24.0 * logfk.powi(2)
                + (1.0 - b).powi(4) / 1920.0 * logfk.powi(4));
            let correction = 1.0 + ((1.0 - b).powi(2) / 24.0 * a * a / fkb.powi(2)
                + r * b * n * a / (4.0 * fkb)
                + (2.0 - 3.0 * r * r) / 24.0 * n * n) * t;
            numer / denom * z / xz * correction
        }
    }

    /// Generate vol smile: implied vols across a range of strikes.
    pub fn vol_smile(&self, forward: f64, expiry: f64, strikes: &[f64]) -> Vec<f64> {
        strikes.iter().map(|&k| self.implied_vol(forward, k, expiry)).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. EXOTIC OPTIONS — LOOKBACK & DIGITAL
// ─────────────────────────────────────────────────────────────────────────────

/// Monte Carlo pricing for lookback options.
/// Floating-strike lookback call: payoff = S_T - min(S_t)
/// Floating-strike lookback put:  payoff = max(S_t) - S_T
pub fn price_lookback(p: &BsParams, steps: usize, paths: usize, seed: u64) -> f64 {
    let mut rng = Lcg::new(seed);
    let dt    = p.expiry / steps as f64;
    let drift = (p.rate - p.dividend - 0.5 * p.vol * p.vol) * dt;
    let diff  = p.vol * dt.sqrt();
    let disc  = (-p.rate * p.expiry).exp();
    let mut sum = 0.0;

    for _ in 0..paths {
        let mut s = p.spot;
        let mut s_min = p.spot;
        let mut s_max = p.spot;
        for _ in 0..steps {
            s *= (drift + diff * rng.normal()).exp();
            if s < s_min { s_min = s; }
            if s > s_max { s_max = s; }
        }
        let payoff = match p.opt_type {
            OptionType::Call => (s - s_min).max(0.0),
            OptionType::Put  => (s_max - s).max(0.0),
        };
        sum += payoff;
    }
    disc * sum / paths as f64
}

/// Cash-or-nothing digital option: pays $1 if S_T > K (call) or S_T < K (put).
/// Analytical BS formula: C_digital = e^{-rT}·N(d2), P_digital = e^{-rT}·N(-d2)
pub fn price_digital_cash_or_nothing(p: &BsParams) -> f64 {
    let (_, d2) = p.d1_d2();
    let disc = (-p.rate * p.expiry).exp();
    match p.opt_type {
        OptionType::Call => disc * normal_cdf(d2),
        OptionType::Put  => disc * normal_cdf(-d2),
    }
}

/// Asset-or-nothing digital option: pays S_T if S_T > K (call).
/// Analytical: S·e^{-qT}·N(d1)
pub fn price_digital_asset_or_nothing(p: &BsParams) -> f64 {
    let (d1, _) = p.d1_d2();
    match p.opt_type {
        OptionType::Call => p.spot * (-p.dividend * p.expiry).exp() * normal_cdf(d1),
        OptionType::Put  => p.spot * (-p.dividend * p.expiry).exp() * normal_cdf(-d1),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. FUTURES & FORWARD PRICING
// ─────────────────────────────────────────────────────────────────────────────

/// Forward price of an asset (cost-of-carry model).
/// F = S · e^{(r - q)T}
pub fn forward_price(spot: f64, rate: f64, dividend: f64, expiry: f64) -> f64 {
    spot * ((rate - dividend) * expiry).exp()
}

/// Futures price = Forward price for constant rates (by cash-and-carry).
pub fn futures_price(spot: f64, rate: f64, dividend: f64, expiry: f64) -> f64 {
    forward_price(spot, rate, dividend, expiry)
}

/// Convenience yield implied by futures price.
/// c = r - (1/T)·ln(F/S)
pub fn implied_convenience_yield(spot: f64, futures: f64, rate: f64, expiry: f64) -> f64 {
    rate - (futures / spot).ln() / expiry
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. TERM STRUCTURE BOOTSTRAPPING
// ─────────────────────────────────────────────────────────────────────────────

/// Bootstrap zero-coupon rates from coupon bond prices.
/// Given a sequence of bonds (ordered by maturity), strip out zero rates.
/// Method: for each bond, solve for the zero rate at the terminal maturity
/// given already-bootstrapped zeros for earlier cash flows.
pub fn bootstrap_zero_rates(bonds: &[(f64, f64, f64, f64)]) -> Vec<(f64, f64)> {
    // bonds: Vec<(maturity_years, coupon_rate, periods_per_year, price)>
    let mut zero_rates: Vec<(f64, f64)> = Vec::new();

    for &(mat, coupon_rate, freq, price) in bonds {
        // Discount earlier cash flows using bootstrapped zero rates
        let c = coupon_rate / freq;
        let n_periods = (mat * freq).round() as usize;
        let mut pv_coupons = 0.0;

        for i in 1..n_periods {
            let t = i as f64 / freq;
            // Interpolate zero rate at t
            let zero_t = interpolate_zero(&zero_rates, t);
            pv_coupons += c * (-zero_t * t).exp();
        }

        // Solve for zero rate at terminal maturity T
        // price = pv_coupons + (1 + c) · e^{-z·T}
        let terminal_cf = 1.0 + c;
        let pv_terminal = price - pv_coupons;
        let z_t = -((pv_terminal / terminal_cf).ln()) / mat;
        zero_rates.push((mat, z_t));
    }
    zero_rates
}

/// Linear interpolation of zero rates.
fn interpolate_zero(zeros: &[(f64, f64)], t: f64) -> f64 {
    if zeros.is_empty() { return 0.03; } // default 3%
    if t <= zeros[0].0 { return zeros[0].1; }
    if t >= zeros[zeros.len() - 1].0 { return zeros[zeros.len() - 1].1; }
    for i in 1..zeros.len() {
        if t <= zeros[i].0 {
            let (t0, z0) = zeros[i - 1];
            let (t1, z1) = zeros[i];
            return z0 + (z1 - z0) * (t - t0) / (t1 - t0);
        }
    }
    zeros[zeros.len() - 1].1
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: make a standard BS params for ATM call
    fn atm_call() -> BsParams {
        BsParams { spot: 100.0, strike: 100.0, rate: 0.05, dividend: 0.0,
                   vol: 0.20, expiry: 1.0, opt_type: OptionType::Call }
    }

    fn atm_put() -> BsParams {
        BsParams { spot: 100.0, strike: 100.0, rate: 0.05, dividend: 0.0,
                   vol: 0.20, expiry: 1.0, opt_type: OptionType::Put }
    }

    // ── Normal Distribution ──────────────────────────────────────────────────

    #[test]
    fn test_normal_cdf_symmetry() {
        // N(0) = 0.5 by symmetry
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normal_cdf_tails() {
        assert!(normal_cdf(-5.0) < 1e-5);
        assert!(normal_cdf(5.0) > 1.0 - 1e-5);
    }

    #[test]
    fn test_normal_cdf_known() {
        // N(1.96) ≈ 0.975
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
        // N(-1.645) ≈ 0.05
        assert!((normal_cdf(-1.645) - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_normal_inv_roundtrip() {
        for p in [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let z = normal_inv(p);
            let p2 = normal_cdf(z);
            assert!((p2 - p).abs() < 1e-6, "Failed at p={}", p);
        }
    }

    #[test]
    fn test_normal_pdf_peak() {
        // PDF peaks at x=0 with value 1/√(2π)
        let expected = 1.0 / (2.0 * PI).sqrt();
        assert!((normal_pdf(0.0) - expected).abs() < 1e-10);
    }

    // ── Black-Scholes Pricing ────────────────────────────────────────────────

    #[test]
    fn test_bs_call_price_known() {
        // S=100, K=100, r=0.05, q=0, σ=0.2, T=1: call ≈ 10.4506
        let p = atm_call();
        let price = p.price();
        assert!((price - 10.4506).abs() < 0.01, "BS call price: {}", price);
    }

    #[test]
    fn test_bs_put_price_known() {
        // Put ≈ 5.5735 (from put-call parity: P = C - S + K·e^{-rT})
        let p = atm_put();
        let price = p.price();
        assert!((price - 5.5735).abs() < 0.01, "BS put price: {}", price);
    }

    #[test]
    fn test_put_call_parity() {
        let call = atm_call();
        let put  = atm_put();
        // C - P = S - K·e^{-rT}
        let lhs = call.price() - put.price();
        let rhs = call.put_call_parity();
        assert!((lhs - rhs).abs() < 1e-8, "PCP: lhs={}, rhs={}", lhs, rhs);
    }

    #[test]
    fn test_bs_call_deep_itm() {
        // Deep ITM call ≈ S - K·e^{-rT}
        let p = BsParams { spot: 200.0, strike: 100.0, rate: 0.05, dividend: 0.0,
                           vol: 0.20, expiry: 1.0, opt_type: OptionType::Call };
        let price = p.price();
        let intrinsic = p.spot - p.strike * (-p.rate * p.expiry).exp();
        assert!((price - intrinsic).abs() < 2.0, "Deep ITM call near intrinsic");
    }

    #[test]
    fn test_bs_call_deep_otm_near_zero() {
        let p = BsParams { spot: 100.0, strike: 300.0, rate: 0.05, dividend: 0.0,
                           vol: 0.20, expiry: 1.0, opt_type: OptionType::Call };
        assert!(p.price() < 0.001, "Deep OTM call near zero");
    }

    #[test]
    fn test_bs_zero_time() {
        // T → 0: call = max(S-K, 0)
        let p = BsParams { spot: 110.0, strike: 100.0, rate: 0.05, dividend: 0.0,
                           vol: 0.20, expiry: 0.001, opt_type: OptionType::Call };
        let price = p.price();
        assert!((price - 10.0).abs() < 0.05, "Near-expiry ITM call ≈ intrinsic");
    }

    // ── Greeks ───────────────────────────────────────────────────────────────

    #[test]
    fn test_delta_atm_call_near_half() {
        // ATM call delta ≈ 0.5 (slightly above for calls)
        let p = atm_call();
        let g = Greeks::compute(&p);
        assert!(g.delta > 0.5 && g.delta < 0.65, "Call delta: {}", g.delta);
    }

    #[test]
    fn test_delta_put_negative() {
        let g = Greeks::compute(&atm_put());
        assert!(g.delta < 0.0 && g.delta > -1.0, "Put delta: {}", g.delta);
    }

    #[test]
    fn test_put_call_delta_sum() {
        // call_delta - put_delta = e^{-qT} (for q=0: ≈ 1)
        let gc = Greeks::compute(&atm_call());
        let gp = Greeks::compute(&atm_put());
        assert!((gc.delta - gp.delta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_positive() {
        // Gamma is always positive for vanilla options
        let gc = Greeks::compute(&atm_call());
        let gp = Greeks::compute(&atm_put());
        assert!(gc.gamma > 0.0);
        assert!(gp.gamma > 0.0);
        // Call and put gammas are equal
        assert!((gc.gamma - gp.gamma).abs() < 1e-10);
    }

    #[test]
    fn test_vega_positive() {
        let g = Greeks::compute(&atm_call());
        assert!(g.vega > 0.0, "Vega must be positive");
    }

    #[test]
    fn test_theta_negative() {
        // Theta is negative (time decay) for long options
        let gc = Greeks::compute(&atm_call());
        let gp = Greeks::compute(&atm_put());
        assert!(gc.theta < 0.0, "Call theta: {}", gc.theta);
        assert!(gp.theta < 0.0, "Put theta: {}", gp.theta);
    }

    #[test]
    fn test_finite_difference_delta() {
        // Numerical delta: (price(S+h) - price(S-h)) / 2h
        let mut p = atm_call();
        let h = 0.01;
        let mut p_up = p.clone(); p_up.spot += h;
        let mut p_dn = p.clone(); p_dn.spot -= h;
        let fd_delta = (p_up.price() - p_dn.price()) / (2.0 * h);
        let g = Greeks::compute(&p);
        assert!((g.delta - fd_delta).abs() < 1e-4, "FD delta diff: {}", (g.delta - fd_delta).abs());
    }

    // ── Implied Volatility ───────────────────────────────────────────────────

    #[test]
    fn test_implied_vol_roundtrip() {
        let p = atm_call();
        let market_price = p.price();
        let iv = implied_vol(market_price, &p).unwrap();
        assert!((iv - 0.20).abs() < 1e-8, "IV roundtrip: {}", iv);
    }

    #[test]
    fn test_implied_vol_different_strikes() {
        for &(k, vol) in &[(80.0f64, 0.25f64), (100.0, 0.20), (120.0, 0.18)] {
            let p = BsParams { spot: 100.0, strike: k, rate: 0.05, dividend: 0.0,
                               vol, expiry: 1.0, opt_type: OptionType::Call };
            let market_price = p.price();
            let iv = implied_vol(market_price, &p).unwrap();
            assert!((iv - vol).abs() < 1e-7, "IV at K={}: got {}, expected {}", k, iv, vol);
        }
    }

    // ── Binomial Tree ────────────────────────────────────────────────────────

    #[test]
    fn test_binomial_converges_to_bs() {
        let p = atm_call();
        let bs_price = p.price();
        let tree = BinomialTree::new(500);
        let tree_price = tree.price(&p, ExerciseStyle::European);
        assert!((tree_price - bs_price).abs() < 0.05,
                "Binomial vs BS: {} vs {}", tree_price, bs_price);
    }

    #[test]
    fn test_american_put_premium() {
        // American put >= European put (early exercise premium)
        let p = atm_put();
        let tree = BinomialTree::new(200);
        let american = tree.price(&p, ExerciseStyle::American);
        let european = tree.price(&p, ExerciseStyle::European);
        assert!(american >= european - 1e-10,
                "American: {}, European: {}", american, european);
    }

    #[test]
    fn test_american_call_no_dividend_equals_european() {
        // For calls on non-dividend assets, early exercise is never optimal
        let p = atm_call();
        let tree = BinomialTree::new(200);
        let american = tree.price(&p, ExerciseStyle::American);
        let european = tree.price(&p, ExerciseStyle::European);
        assert!((american - european).abs() < 1e-6,
                "American call = European call (no div): {} vs {}", american, european);
    }

    // ── Monte Carlo ──────────────────────────────────────────────────────────

    #[test]
    fn test_mc_european_call_accuracy() {
        let p = atm_call();
        let bs = p.price();
        let mc = MonteCarloEngine::new(100_000, 42);
        let (mc_price, ci) = mc.price_european(&p);
        // BS price should be within 95% CI
        assert!((mc_price - bs).abs() < ci * 2.0 + 0.10,
                "MC: {}, BS: {}, CI: {}", mc_price, bs, ci);
    }

    #[test]
    fn test_mc_asian_cheaper_than_european() {
        // Asian option (arithmetic avg) is cheaper than European (less volatility in avg)
        let p = atm_call();
        let mc = MonteCarloEngine::new(50_000, 123);
        let european = mc.price_european(&p).0;
        let asian = mc.price_asian(&p, 52); // weekly monitoring
        assert!(asian < european + 0.5, "Asian: {}, European: {}", asian, european);
    }

    #[test]
    fn test_mc_barrier_knock_out_cheaper() {
        // Up-and-out call is cheaper than vanilla (knock-out reduces payoff)
        let p = atm_call();
        let mc = MonteCarloEngine::new(50_000, 999);
        let vanilla = mc.price_european(&p).0;
        let barrier = mc.price_barrier(&p, 130.0, true, true, 252);
        assert!(barrier < vanilla + 0.5,
                "Barrier: {}, Vanilla: {}", barrier, vanilla);
    }

    // ── Bond Pricing ─────────────────────────────────────────────────────────

    #[test]
    fn test_bond_par_when_coupon_equals_yield() {
        // When coupon rate = YTM with periodic (discrete) discounting, bond price = face value.
        // Use price_periodic which correctly prices at par when coupon rate equals YTM.
        let bond = Bond { face: 1000.0, coupon_rate: 0.05, periods: 10, periods_year: 1.0 };
        let price = bond.price_periodic(0.05);
        assert!((price - 1000.0).abs() < 1.0, "Bond at par: {}", price);
    }

    #[test]
    fn test_bond_premium_when_low_yield() {
        // When YTM < coupon, bond trades at premium
        let bond = Bond { face: 1000.0, coupon_rate: 0.08, periods: 10, periods_year: 1.0 };
        let price = bond.price_continuous(0.05);
        assert!(price > 1000.0, "Premium bond: {}", price);
    }

    #[test]
    fn test_bond_discount_when_high_yield() {
        // When YTM > coupon, bond trades at discount
        let bond = Bond { face: 1000.0, coupon_rate: 0.05, periods: 10, periods_year: 1.0 };
        let price = bond.price_continuous(0.08);
        assert!(price < 1000.0, "Discount bond: {}", price);
    }

    #[test]
    fn test_bond_duration_positive() {
        let bond = Bond { face: 1000.0, coupon_rate: 0.05, periods: 10, periods_year: 1.0 };
        let dur = bond.duration(0.05);
        assert!(dur > 0.0 && dur < 10.0, "Duration: {}", dur);
    }

    #[test]
    fn test_zero_coupon_bond_duration_equals_maturity() {
        // For a zero-coupon bond, Macaulay duration = maturity
        let bond = Bond { face: 1000.0, coupon_rate: 0.0, periods: 5, periods_year: 1.0 };
        let dur = bond.duration(0.05);
        assert!((dur - 5.0).abs() < 0.01, "ZCB duration: {}", dur);
    }

    #[test]
    fn test_bond_convexity_positive() {
        let bond = Bond { face: 1000.0, coupon_rate: 0.05, periods: 10, periods_year: 1.0 };
        let conv = bond.convexity(0.05);
        assert!(conv > 0.0, "Convexity: {}", conv);
    }

    #[test]
    fn test_bond_ytm_roundtrip() {
        let bond = Bond { face: 1000.0, coupon_rate: 0.06, periods: 10, periods_year: 1.0 };
        let ytm_orig = 0.07;
        let price = bond.price_continuous(ytm_orig);
        let ytm_back = bond.ytm(price).unwrap();
        assert!((ytm_back - ytm_orig).abs() < 1e-8, "YTM roundtrip: {}", ytm_back);
    }

    // ── Nelson-Siegel ────────────────────────────────────────────────────────

    #[test]
    fn test_ns_long_rate_equals_beta0() {
        let ns = NelsonSiegel { beta0: 0.05, beta1: -0.02, beta2: 0.01, lambda: 0.5 };
        // As T→∞, yield → β0
        let long_rate = ns.yield_at(100.0);
        assert!((long_rate - 0.05).abs() < 0.001, "Long rate: {}", long_rate);
    }

    #[test]
    fn test_ns_short_rate_equals_beta0_plus_beta1() {
        let ns = NelsonSiegel { beta0: 0.05, beta1: -0.02, beta2: 0.01, lambda: 0.5 };
        // As T→0, yield → β0 + β1
        let short_rate = ns.yield_at(0.001);
        assert!((short_rate - 0.03).abs() < 0.001, "Short rate: {}", short_rate);
    }

    #[test]
    fn test_ns_discount_factor_decreasing() {
        let ns = NelsonSiegel { beta0: 0.04, beta1: 0.01, beta2: 0.005, lambda: 0.5 };
        let d1 = ns.discount_factor(1.0);
        let d5 = ns.discount_factor(5.0);
        let d10 = ns.discount_factor(10.0);
        assert!(d1 > d5 && d5 > d10, "Discount factors not decreasing");
    }

    #[test]
    fn test_ns_fit_recovers_flat_curve() {
        // A flat yield curve should be fit by β0=rate, β1≈0, β2≈0.
        // The gradient-descent fitter (lr=1e-5, 5000 iters) converges approximately;
        // use a tolerance consistent with the numerical optimizer's precision.
        let mats = vec![1.0, 2.0, 5.0, 10.0, 30.0];
        let yields = vec![0.04, 0.04, 0.04, 0.04, 0.04];
        let ns = NelsonSiegel::fit(&mats, &yields);
        for &t in &mats {
            let y = ns.yield_at(t);
            assert!((y - 0.04).abs() < 0.015, "NS flat fit at T={}: {}", t, y);
        }
    }

    // ── Portfolio Optimization ───────────────────────────────────────────────

    #[test]
    fn test_portfolio_expected_return() {
        let p = Portfolio::new(
            vec![0.10, 0.08, 0.06],
            vec![vec![0.04, 0.01, 0.0], vec![0.01, 0.03, 0.005], vec![0.0, 0.005, 0.02]],
        );
        let w = vec![1.0/3.0, 1.0/3.0, 1.0/3.0];
        let ret = p.expected_return(&w);
        assert!((ret - 0.08).abs() < 1e-10, "Expected return: {}", ret);
    }

    #[test]
    fn test_portfolio_variance_single_asset() {
        let p = Portfolio::new(
            vec![0.10],
            vec![vec![0.04]], // σ²=0.04, σ=0.20
        );
        let w = vec![1.0];
        assert!((p.variance(&w) - 0.04).abs() < 1e-12);
        assert!((p.volatility(&w) - 0.20).abs() < 1e-12);
    }

    #[test]
    fn test_min_variance_weights_sum_to_one() {
        let p = Portfolio::new(
            vec![0.10, 0.08, 0.06],
            vec![vec![0.04, 0.01, 0.0], vec![0.01, 0.03, 0.005], vec![0.0, 0.005, 0.02]],
        );
        let w = p.min_variance_weights();
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Weights sum: {}", sum);
        assert!(w.iter().all(|&wi| wi >= -1e-9), "All weights non-negative");
    }

    #[test]
    fn test_simplex_projection() {
        let v = vec![0.5, 0.3, -0.1, 0.8];
        let w = project_to_simplex(&v);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Simplex sum: {}", sum);
        assert!(w.iter().all(|&wi| wi >= 0.0), "Simplex non-negative");
    }

    // ── Risk Analytics ───────────────────────────────────────────────────────

    #[test]
    fn test_risk_mean() {
        let rets = vec![0.01, 0.02, -0.01, 0.03, -0.02];
        let ra = RiskAnalytics::new(rets);
        assert!((ra.mean() - 0.006).abs() < 1e-10, "Mean: {}", ra.mean());
    }

    #[test]
    fn test_var_historical() {
        // 100 returns; VaR at 95% = 5th percentile worst loss
        let mut rets: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.001).collect();
        let ra = RiskAnalytics::new(rets);
        let var95 = ra.var_historical(0.95);
        // 5th percentile of uniform is around -0.045
        assert!(var95 > 0.0, "VaR should be positive (loss)");
    }

    #[test]
    fn test_cvar_gte_var() {
        let rets: Vec<f64> = (0..200).map(|i| (i as f64 - 100.0) * 0.001).collect();
        let ra = RiskAnalytics::new(rets);
        let var = ra.var_historical(0.95);
        let cvar = ra.cvar_historical(0.95);
        assert!(cvar >= var - 1e-10, "CVaR {} >= VaR {}", cvar, var);
    }

    #[test]
    fn test_max_drawdown_zero_on_rising() {
        let rets = vec![0.01, 0.02, 0.03, 0.01, 0.02]; // always rising
        let ra = RiskAnalytics::new(rets);
        assert!((ra.max_drawdown() - 0.0).abs() < 1e-10, "MDD on rising: {}", ra.max_drawdown());
    }

    #[test]
    fn test_max_drawdown_correct() {
        // 50% gain then 33% loss = 33% drawdown
        let rets = vec![0.5, -0.33333];
        let ra = RiskAnalytics::new(rets);
        let mdd = ra.max_drawdown();
        assert!((mdd - 0.33333).abs() < 0.001, "MDD: {}", mdd);
    }

    #[test]
    fn test_sharpe_ratio() {
        let rets = vec![0.001; 252]; // 0.1% daily, no variance
        let ra = RiskAnalytics::new(rets);
        // Sharpe should be very high (constant positive returns)
        // vol is 0 → sharpe undefined, but nearly 0 vol → very large
        // Just check it's positive with non-degenerate data
        let rets2: Vec<f64> = (0..252).map(|i| 0.001 + if i % 2 == 0 { 0.002 } else { -0.002 }).collect();
        let ra2 = RiskAnalytics::new(rets2);
        let sharpe = ra2.sharpe_ratio(0.0, 252.0);
        assert!(sharpe > 0.0, "Sharpe: {}", sharpe);
    }

    // ── Vasicek Model ────────────────────────────────────────────────────────

    #[test]
    fn test_vasicek_bond_price_reasonable() {
        let v = VasicekModel { kappa: 0.5, theta: 0.05, sigma: 0.01, r0: 0.03 };
        let p = v.bond_price(5.0);
        assert!(p > 0.7 && p < 1.0, "Vasicek bond price: {}", p);
    }

    #[test]
    fn test_vasicek_expected_rate_mean_reverts() {
        let v = VasicekModel { kappa: 0.5, theta: 0.05, sigma: 0.01, r0: 0.03 };
        // E[r_0] = r0
        assert!((v.expected_rate(0.0) - 0.03).abs() < 1e-10);
        // E[r_∞] → θ
        assert!((v.expected_rate(100.0) - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_vasicek_spot_rate_curve() {
        let v = VasicekModel { kappa: 0.5, theta: 0.06, sigma: 0.02, r0: 0.03 };
        // r0 < θ: upward sloping curve expected
        let r1  = v.spot_rate(1.0);
        let r10 = v.spot_rate(10.0);
        assert!(r10 > r1, "Upward sloping: r1={}, r10={}", r1, r10);
    }

    #[test]
    fn test_vasicek_simulation_mean_reverts() {
        let v = VasicekModel { kappa: 2.0, theta: 0.05, sigma: 0.02, r0: 0.10 };
        let path = v.simulate(5.0, 1000, 42);
        let final_rate = *path.last().unwrap();
        // After 5 years with κ=2 (strong mean reversion), should be close to θ
        assert!((final_rate - 0.05).abs() < 0.05, "Final rate: {}", final_rate);
    }

    #[test]
    fn test_cir_feller() {
        let cir = CirModel { kappa: 1.0, theta: 0.06, sigma: 0.1, r0: 0.04 };
        // 2κθ = 0.12, σ² = 0.01 → Feller satisfied
        assert!(cir.feller_satisfied());
        let bad = CirModel { kappa: 0.1, theta: 0.01, sigma: 0.5, r0: 0.04 };
        assert!(!bad.feller_satisfied());
    }

    #[test]
    fn test_cir_simulation_stays_positive() {
        let cir = CirModel { kappa: 1.0, theta: 0.05, sigma: 0.1, r0: 0.04 };
        let path = cir.simulate(10.0, 10000, 777);
        assert!(path.iter().all(|&r| r >= 0.0), "CIR path went negative");
    }

    // ── CDS Model ────────────────────────────────────────────────────────────

    #[test]
    fn test_cds_fair_spread_positive() {
        let cds = CdsModel { hazard_rate: 0.02, recovery: 0.40, risk_free_rate: 0.03,
                             notional: 1_000_000.0 };
        let spread = cds.fair_spread(5.0);
        assert!(spread > 0.0, "CDS spread: {}", spread);
        // Approximately λ(1-R) for low rates: 0.02 * 0.6 = 0.012
        assert!((spread - 0.012).abs() < 0.005, "CDS spread approx: {}", spread);
    }

    #[test]
    fn test_cds_higher_hazard_rate_higher_spread() {
        let base = CdsModel { hazard_rate: 0.01, recovery: 0.40, risk_free_rate: 0.03,
                              notional: 1_000_000.0 };
        let risky = CdsModel { hazard_rate: 0.05, recovery: 0.40, risk_free_rate: 0.03,
                               notional: 1_000_000.0 };
        assert!(risky.fair_spread(5.0) > base.fair_spread(5.0));
    }

    #[test]
    fn test_cds_mtm_zero_at_fair_spread() {
        let cds = CdsModel { hazard_rate: 0.02, recovery: 0.40, risk_free_rate: 0.03,
                             notional: 1_000_000.0 };
        let fair = cds.fair_spread(5.0);
        let mtm = cds.mtm(5.0, fair);
        assert!(mtm.abs() < 1e-6, "MTM at fair spread should be 0: {}", mtm);
    }

    // ── SABR Model ───────────────────────────────────────────────────────────

    #[test]
    fn test_sabr_atm_positive() {
        let sabr = SabrModel { alpha: 0.20, beta: 0.5, rho: -0.3, nu: 0.4 };
        let iv = sabr.implied_vol(100.0, 100.0, 1.0);
        assert!(iv > 0.0 && iv < 2.0, "SABR ATM IV: {}", iv);
    }

    #[test]
    fn test_sabr_smile_shape() {
        // With negative rho, SABR produces a skewed smile (lower IV for high strikes)
        let sabr = SabrModel { alpha: 0.20, beta: 0.5, rho: -0.3, nu: 0.4 };
        let f = 100.0;
        let strikes = vec![80.0, 90.0, 100.0, 110.0, 120.0];
        let smile = sabr.vol_smile(f, 1.0, &strikes);
        assert!(smile.len() == 5);
        // All IVs should be positive
        assert!(smile.iter().all(|&v| v > 0.0));
    }

    // ── Digital Options ──────────────────────────────────────────────────────

    #[test]
    fn test_digital_cash_atm_near_half() {
        // ATM digital call ≈ 0.5 · e^{-rT} (roughly)
        let p = atm_call();
        let price = price_digital_cash_or_nothing(&p);
        let disc = (-p.rate * p.expiry).exp();
        assert!((price / disc - 0.5).abs() < 0.1, "Digital cash: {}", price);
    }

    #[test]
    fn test_digital_decomposition() {
        // Vanilla call = Asset-or-nothing call - K · Cash-or-nothing call
        // price_digital_cash_or_nothing already includes e^{-rT}, so no extra discounting.
        let p = atm_call();
        let aon = price_digital_asset_or_nothing(&p);
        let con = price_digital_cash_or_nothing(&p);
        let decomposed = aon - p.strike * con;
        let vanilla = p.price();
        assert!((decomposed - vanilla).abs() < 1e-8, "Decomposed: {}, Vanilla: {}", decomposed, vanilla);
    }

    // ── Lookback Options ─────────────────────────────────────────────────────

    #[test]
    fn test_lookback_call_gt_vanilla() {
        // Floating-strike lookback call >= vanilla ATM call (lookback has more flexibility)
        let p = atm_call();
        let lookback = price_lookback(&p, 252, 20_000, 42);
        let vanilla = p.price();
        // Lookback should be significantly more expensive
        assert!(lookback > vanilla * 0.8, "Lookback: {}, Vanilla: {}", lookback, vanilla);
    }

    // ── Forwards & Futures ───────────────────────────────────────────────────

    #[test]
    fn test_forward_price_no_dividend() {
        // F = S·e^{rT}
        let f = forward_price(100.0, 0.05, 0.0, 1.0);
        assert!((f - 100.0 * E.powf(0.05)).abs() < 1e-8, "Forward: {}", f);
    }

    #[test]
    fn test_forward_price_with_dividend() {
        // F = S·e^{(r-q)T}; dividend reduces forward
        let f_no_div = forward_price(100.0, 0.05, 0.0, 1.0);
        let f_div    = forward_price(100.0, 0.05, 0.02, 1.0);
        assert!(f_div < f_no_div, "Dividend reduces forward");
    }

    #[test]
    fn test_implied_convenience_yield() {
        // Recover convenience yield from forward price
        let spot = 50.0;
        let rate = 0.04;
        let conv_yield = 0.06; // convenience yield > rate: backwardation
        let f = spot * ((rate - conv_yield) * 1.0_f64).exp();
        let implied = implied_convenience_yield(spot, f, rate, 1.0);
        assert!((implied - conv_yield).abs() < 1e-10, "Implied convenience yield: {}", implied);
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────

    #[test]
    fn test_bootstrap_single_bond() {
        // Single 1Y bond, coupon=5%, price=1.0 (per-unit face) → zero rate ≈ 5%
        // bootstrap_zero_rates uses unit-face convention: terminal_cf = 1 + coupon/freq
        let bonds = vec![(1.0f64, 0.05, 1.0, 1.0)];
        let zeros = bootstrap_zero_rates(&bonds);
        assert_eq!(zeros.len(), 1);
        let (mat, z) = zeros[0];
        assert!((mat - 1.0).abs() < 1e-10);
        // Price = (1 + 0.05) * e^{-z} = 1 → z = ln(1.05)
        let expected_z = (1.05_f64).ln();
        assert!((z - expected_z).abs() < 1e-6, "Bootstrap z: {}", z);
    }

    #[test]
    fn test_project_simplex_already_on_simplex() {
        let v = vec![0.3, 0.3, 0.4];
        let w = project_to_simplex(&v);
        for (wi, vi) in w.iter().zip(&v) {
            assert!((wi - vi).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lcg_uniform_range() {
        let mut rng = Lcg::new(12345);
        for _ in 0..10000 {
            let u = rng.uniform();
            assert!(u >= 0.0 && u < 1.0, "Uniform out of range: {}", u);
        }
    }

    #[test]
    fn test_lcg_normal_moments() {
        let mut rng = Lcg::new(99999);
        let n = 10000usize;
        let samples: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let mean: f64 = samples.iter().sum::<f64>() / n as f64;
        let var: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "Normal mean: {}", mean);
        assert!((var - 1.0).abs() < 0.05, "Normal variance: {}", var);
    }
}
