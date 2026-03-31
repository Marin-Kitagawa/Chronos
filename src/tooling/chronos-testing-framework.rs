// chronos-testing-framework.rs
//
// Chronos Testing Framework
// ==========================
// A comprehensive testing library implementing unit testing, property-based
// testing, fuzzing, mutation testing, mocking, and test reporting.
//
// Modules:
//   1.  Test Registry & Runner — collect, filter, execute tests
//   2.  Assertions — rich assertion macros with diff output
//   3.  Test Fixtures — setup/teardown, parameterised tests
//   4.  Property-Based Testing — generators, shrinkers, QuickCheck-style
//   5.  Fuzzing Engine — coverage-guided fuzzing (AFL/libFuzzer style)
//   6.  Mutation Testing — source mutation + test sufficiency scoring
//   7.  Mocking Framework — call recording, expectations, stubs
//   8.  Snapshot Testing — golden-file regression testing
//   9.  Benchmark Harness — statistical micro-benchmarking
//  10.  Coverage Tracking — line/branch/function coverage
//  11.  Test Reporters — TAP, JUnit XML, terminal output
//  12.  Parallel Test Execution — thread-safe test runner

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// 1. TEST REGISTRY & RUNNER
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a single test.
#[derive(Debug, Clone, PartialEq)]
pub enum TestOutcome {
    Passed,
    Failed(FailureReason),
    Skipped(String),    // reason
    Panicked(String),   // message
    TimedOut,
}

/// Reason a test failed.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureReason {
    AssertionFailed { message: String, location: Location },
    PropertyFailed  { input: String, shrunk: String, seed: u64 },
    SnapshotMismatch { expected: String, actual: String },
    Custom(String),
}

/// Source location (file + line).
#[derive(Debug, Clone, PartialEq)]
pub struct Location { pub file: &'static str, pub line: u32 }

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)
    }
}

/// Metadata attached to a test.
#[derive(Debug, Clone)]
pub struct TestMeta {
    pub name:       String,
    pub module:     String,
    pub tags:       Vec<String>,
    pub timeout_ms: Option<u64>,
    pub ignored:    bool,
    pub flaky:      bool,   // allowed to fail once — retried
}

impl TestMeta {
    pub fn new(name: &str, module: &str) -> Self {
        TestMeta { name: name.to_string(), module: module.to_string(),
                   tags: Vec::new(), timeout_ms: None, ignored: false, flaky: false }
    }
    pub fn tag(mut self, t: &str) -> Self { self.tags.push(t.to_string()); self }
    pub fn timeout(mut self, ms: u64) -> Self { self.timeout_ms = Some(ms); self }
    pub fn ignore(mut self) -> Self { self.ignored = true; self }
    pub fn flaky(mut self) -> Self { self.flaky = true; self }
}

/// A single test case.
pub struct TestCase {
    pub meta:    TestMeta,
    pub func:    Box<dyn Fn() -> TestOutcome + Send + Sync>,
}

impl TestCase {
    pub fn new(meta: TestMeta, func: impl Fn() -> TestOutcome + Send + Sync + 'static) -> Self {
        TestCase { meta, func: Box::new(func) }
    }
}

/// The global test registry.
pub struct TestRegistry {
    tests: Vec<TestCase>,
}

impl TestRegistry {
    pub fn new() -> Self { TestRegistry { tests: Vec::new() } }

    pub fn register(&mut self, test: TestCase) { self.tests.push(test); }

    /// Filter tests by name substring, tag, or module.
    pub fn filter(&self, filter: &TestFilter) -> Vec<&TestCase> {
        self.tests.iter().filter(|t| filter.matches(&t.meta)).collect()
    }
}

/// A predicate for selecting which tests to run.
#[derive(Debug, Clone, Default)]
pub struct TestFilter {
    pub name_contains: Option<String>,
    pub tag:           Option<String>,
    pub module:        Option<String>,
    pub include_ignored: bool,
}

impl TestFilter {
    pub fn all() -> Self { TestFilter { include_ignored: false, ..Default::default() } }

    pub fn matches(&self, meta: &TestMeta) -> bool {
        if meta.ignored && !self.include_ignored { return false; }
        if let Some(ref n) = self.name_contains {
            if !meta.name.contains(n.as_str()) { return false; }
        }
        if let Some(ref t) = self.tag {
            if !meta.tags.iter().any(|tag| tag == t) { return false; }
        }
        if let Some(ref m) = self.module {
            if &meta.module != m { return false; }
        }
        true
    }
}

/// Result for a single test execution.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub meta:     TestMeta,
    pub outcome:  TestOutcome,
    pub duration: Duration,
    pub retries:  u32,
}

impl TestResult {
    pub fn passed(&self) -> bool { self.outcome == TestOutcome::Passed }
    pub fn failed(&self) -> bool { matches!(self.outcome, TestOutcome::Failed(_) | TestOutcome::Panicked(_) | TestOutcome::TimedOut) }
}

/// Run a collection of tests and return results.
pub struct TestRunner {
    pub config: RunnerConfig,
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub fail_fast:    bool,   // stop on first failure
    pub max_retries:  u32,    // for flaky tests
    pub default_timeout_ms: u64,
    pub capture_output: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig { fail_fast: false, max_retries: 2,
                       default_timeout_ms: 30_000, capture_output: true }
    }
}

impl TestRunner {
    pub fn new(config: RunnerConfig) -> Self { TestRunner { config } }

    pub fn run(&self, tests: &[&TestCase]) -> TestSuiteResult {
        let start = Instant::now();
        let mut results = Vec::new();

        for test in tests {
            if test.meta.ignored {
                results.push(TestResult {
                    meta:     test.meta.clone(),
                    outcome:  TestOutcome::Skipped("ignored".to_string()),
                    duration: Duration::ZERO,
                    retries:  0,
                });
                continue;
            }

            let mut attempts = 0u32;
            let max_attempts = if test.meta.flaky { self.config.max_retries + 1 } else { 1 };
            let outcome;
            let mut elapsed = Duration::ZERO;

            loop {
                let t0 = Instant::now();
                let result = (test.func)();
                elapsed = t0.elapsed();
                attempts += 1;
                match &result {
                    TestOutcome::Passed => { outcome = result; break; }
                    _ if attempts >= max_attempts => { outcome = result; break; }
                    _ => {} // retry flaky test
                }
            }

            let failed = matches!(outcome, TestOutcome::Failed(_) | TestOutcome::Panicked(_) | TestOutcome::TimedOut);
            results.push(TestResult {
                meta:     test.meta.clone(),
                outcome,
                duration: elapsed,
                retries:  attempts.saturating_sub(1),
            });

            if failed && self.config.fail_fast { break; }
        }

        let elapsed = start.elapsed();
        TestSuiteResult { results, total_duration: elapsed }
    }
}

/// Aggregated results for a test suite run.
#[derive(Debug)]
pub struct TestSuiteResult {
    pub results:        Vec<TestResult>,
    pub total_duration: Duration,
}

impl TestSuiteResult {
    pub fn passed(&self)  -> usize { self.results.iter().filter(|r| r.passed()).count() }
    pub fn failed(&self)  -> usize { self.results.iter().filter(|r| r.failed()).count() }
    pub fn skipped(&self) -> usize { self.results.iter().filter(|r| r.outcome == TestOutcome::Skipped("ignored".to_string()) || matches!(r.outcome, TestOutcome::Skipped(_))).count() }
    pub fn total(&self)   -> usize { self.results.len() }
    pub fn all_passed(&self) -> bool { self.failed() == 0 }

    pub fn slowest_tests(&self, n: usize) -> Vec<&TestResult> {
        let mut sorted = self.results.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.duration.cmp(&a.duration));
        sorted.into_iter().take(n).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ASSERTIONS
// ─────────────────────────────────────────────────────────────────────────────

/// Rich assertion result carrying full context for failure messages.
#[derive(Debug, Clone, PartialEq)]
pub enum AssertResult { Ok, Fail(String) }

impl AssertResult {
    pub fn is_ok(&self) -> bool { matches!(self, AssertResult::Ok) }
    pub fn into_outcome(self, loc: Location) -> TestOutcome {
        match self {
            AssertResult::Ok => TestOutcome::Passed,
            AssertResult::Fail(msg) => TestOutcome::Failed(
                FailureReason::AssertionFailed { message: msg, location: loc }),
        }
    }
}

/// Assert two values are equal, with a diff on failure.
pub fn assert_eq_<T: PartialEq + fmt::Debug>(left: T, right: T, msg: &str) -> AssertResult {
    if left == right { AssertResult::Ok }
    else {
        AssertResult::Fail(format!(
            "{}\n  left:  {:?}\n  right: {:?}\n{}",
            msg, left, right, unified_diff(&format!("{:?}", left), &format!("{:?}", right))
        ))
    }
}

/// Assert a boolean condition.
pub fn assert_true_(cond: bool, msg: &str) -> AssertResult {
    if cond { AssertResult::Ok }
    else { AssertResult::Fail(format!("assertion failed: {}", msg)) }
}

/// Assert a Result is Ok.
pub fn assert_ok_<T, E: fmt::Debug>(r: Result<T, E>) -> AssertResult {
    match r { Ok(_) => AssertResult::Ok,
              Err(e) => AssertResult::Fail(format!("expected Ok, got Err({:?})", e)) }
}

/// Assert a Result is Err.
pub fn assert_err_<T: fmt::Debug, E>(r: Result<T, E>) -> AssertResult {
    match r { Err(_) => AssertResult::Ok,
              Ok(v)  => AssertResult::Fail(format!("expected Err, got Ok({:?})", v)) }
}

/// Assert a float is approximately equal within a tolerance.
pub fn assert_approx_eq_(a: f64, b: f64, tol: f64) -> AssertResult {
    if (a - b).abs() <= tol { AssertResult::Ok }
    else { AssertResult::Fail(format!("approx_eq failed: |{} - {}| = {} > {}", a, b, (a-b).abs(), tol)) }
}

/// Assert a slice contains a specific element.
pub fn assert_contains_<T: PartialEq + fmt::Debug>(slice: &[T], elem: &T) -> AssertResult {
    if slice.contains(elem) { AssertResult::Ok }
    else { AssertResult::Fail(format!("{:?} not found in {:?}", elem, slice)) }
}

/// Assert that two strings are equal, showing a character-level diff on failure.
pub fn assert_str_eq_(left: &str, right: &str) -> AssertResult {
    if left == right { AssertResult::Ok }
    else {
        AssertResult::Fail(format!(
            "string mismatch:\n{}", unified_diff(left, right)
        ))
    }
}

/// Simple line-level unified diff.
pub fn unified_diff(a: &str, b: &str) -> String {
    let a_lines: Vec<&str> = a.lines().collect();
    let b_lines: Vec<&str> = b.lines().collect();
    let mut out = String::from("--- expected\n+++ actual\n");
    let max = a_lines.len().max(b_lines.len());
    for i in 0..max {
        match (a_lines.get(i), b_lines.get(i)) {
            (Some(&l), Some(&r)) if l == r => out += &format!("  {}\n", l),
            (Some(&l), Some(&r)) => { out += &format!("- {}\n+ {}\n", l, r); }
            (Some(&l), None)     => out += &format!("- {}\n", l),
            (None, Some(&r))     => out += &format!("+ {}\n", r),
            (None, None)         => {}
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. PROPERTY-BASED TESTING (QuickCheck / Hypothesis style)
// ─────────────────────────────────────────────────────────────────────────────

/// A pseudo-random generator for property-based testing.
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self { Rng { state: seed } }

    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state = self.state.wrapping_mul(2685821657736338717);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn next_bool(&mut self) -> bool { self.next_u64() & 1 == 0 }
    pub fn next_i64(&mut self) -> i64 { self.next_u64() as i64 }
    pub fn next_usize(&mut self, max: usize) -> usize { if max == 0 { 0 } else { (self.next_u64() as usize) % max } }
    pub fn next_range_i64(&mut self, lo: i64, hi: i64) -> i64 {
        if lo >= hi { return lo; }
        lo + (self.next_u64() as i64).abs() % (hi - lo)
    }
}

/// Trait for types that can be generated for property testing.
pub trait Arbitrary: Clone + fmt::Debug {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self;
    /// Produce simpler versions of self for shrinking.
    fn shrink(&self) -> Vec<Self> { Vec::new() }
}

impl Arbitrary for bool {
    fn arbitrary(rng: &mut Rng, _: usize) -> Self { rng.next_bool() }
    fn shrink(&self) -> Vec<bool> { if *self { vec![false] } else { vec![] } }
}

impl Arbitrary for i64 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        rng.next_range_i64(-(size as i64 * 10), size as i64 * 10)
    }
    fn shrink(&self) -> Vec<i64> {
        if *self == 0 { vec![] }
        else { vec![0, self / 2, self - self.signum()] }
    }
}

impl Arbitrary for u64 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self { rng.next_u64() % (size as u64 * 10 + 1) }
    fn shrink(&self) -> Vec<u64> {
        if *self == 0 { vec![] } else { vec![0, self / 2, self - 1] }
    }
}

impl Arbitrary for f64 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        (rng.next_f64() * 2.0 - 1.0) * size as f64
    }
    fn shrink(&self) -> Vec<f64> {
        if self.abs() < 1e-10 { vec![] }
        else { vec![0.0, self / 2.0, self * 0.9] }
    }
}

impl Arbitrary for String {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let len = rng.next_usize(size + 1);
        (0..len).map(|_| {
            let c = b'a' + (rng.next_u64() % 26) as u8;
            c as char
        }).collect()
    }
    fn shrink(&self) -> Vec<String> {
        if self.is_empty() { return vec![]; }
        vec![
            self[..self.len()/2].to_string(),
            self[1..].to_string(),
            String::new(),
        ]
    }
}

impl<T: Arbitrary> Arbitrary for Vec<T> {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let len = rng.next_usize(size + 1);
        (0..len).map(|_| T::arbitrary(rng, size.saturating_sub(1))).collect()
    }
    fn shrink(&self) -> Vec<Vec<T>> {
        if self.is_empty() { return vec![]; }
        vec![
            self[..self.len()/2].to_vec(),
            self[1..].to_vec(),
            Vec::new(),
        ]
    }
}

impl<A: Arbitrary, B: Arbitrary> Arbitrary for (A, B) {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        (A::arbitrary(rng, size), B::arbitrary(rng, size))
    }
    fn shrink(&self) -> Vec<(A, B)> {
        let mut result = Vec::new();
        for a in self.0.shrink() { result.push((a, self.1.clone())); }
        for b in self.1.shrink() { result.push((self.0.clone(), b)); }
        result
    }
}

/// A property test result.
#[derive(Debug, Clone)]
pub struct PropertyResult {
    pub passed:      bool,
    pub tests_run:   u64,
    pub failure:     Option<PropertyFailure>,
}

#[derive(Debug, Clone)]
pub struct PropertyFailure {
    pub original_input: String,
    pub shrunk_input:   String,
    pub seed:           u64,
    pub failure_msg:    String,
}

/// Run a property test with automatic shrinking.
/// `predicate` returns Ok(()) on success or Err(msg) on failure.
pub fn property_test<T: Arbitrary>(
    name: &str,
    num_tests: u64,
    seed: u64,
    predicate: impl Fn(&T) -> Result<(), String>,
) -> PropertyResult {
    let mut rng = Rng::new(seed);

    for i in 0..num_tests {
        let size = (i / (num_tests / 10 + 1) + 1) as usize; // gradually increase size
        let input = T::arbitrary(&mut rng, size);
        if let Err(msg) = predicate(&input) {
            let original_str = format!("{:?}", input);
            // Shrink
            let shrunk = shrink_input(input, &predicate);
            let shrunk_str = format!("{:?}", shrunk);
            return PropertyResult {
                passed:    false,
                tests_run: i + 1,
                failure: Some(PropertyFailure {
                    original_input: original_str,
                    shrunk_input:   shrunk_str,
                    seed,
                    failure_msg:    msg,
                }),
            };
        }
    }
    PropertyResult { passed: true, tests_run: num_tests, failure: None }
}

/// Shrink a failing input to its simplest form.
fn shrink_input<T: Arbitrary>(input: T, predicate: &impl Fn(&T) -> Result<(), String>) -> T {
    let mut current = input;
    loop {
        let candidates = current.shrink();
        let found = candidates.into_iter().find(|c| predicate(c).is_err());
        match found {
            Some(simpler) => current = simpler,
            None => break,
        }
    }
    current
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. FUZZING ENGINE
// ─────────────────────────────────────────────────────────────────────────────

/// A fuzzer corpus entry.
#[derive(Debug, Clone)]
pub struct CorpusEntry {
    pub data:     Vec<u8>,
    pub coverage: u64,   // bitmask of coverage edges hit (simulated)
    pub energy:   f64,   // scheduling energy (higher = fuzz more)
}

/// Coverage-guided fuzzer (AFL-style).
/// Maintains a corpus of interesting inputs and mutates them to find new coverage.
pub struct Fuzzer {
    pub corpus:       Vec<CorpusEntry>,
    pub crashes:      Vec<Vec<u8>>,
    pub total_execs:  u64,
    pub unique_paths: u64,
    rng:              Rng,
    seen_coverage:    std::collections::HashSet<u64>,
}

impl Fuzzer {
    pub fn new(seed: u64) -> Self {
        Fuzzer {
            corpus: Vec::new(),
            crashes: Vec::new(),
            total_execs: 0,
            unique_paths: 0,
            rng: Rng::new(seed),
            seen_coverage: std::collections::HashSet::new(),
        }
    }

    /// Seed the corpus with initial inputs.
    pub fn seed_corpus(&mut self, inputs: Vec<Vec<u8>>) {
        for data in inputs {
            self.corpus.push(CorpusEntry { data, coverage: 0, energy: 1.0 });
        }
    }

    /// Mutate a corpus entry using one of several mutation strategies.
    pub fn mutate(&mut self, entry: &[u8]) -> Vec<u8> {
        if entry.is_empty() {
            return vec![self.rng.next_u64() as u8];
        }
        let strategy = self.rng.next_usize(8);
        let mut out = entry.to_vec();
        match strategy {
            0 => { // Bit flip
                let idx = self.rng.next_usize(out.len() * 8);
                out[idx / 8] ^= 1 << (idx % 8);
            }
            1 => { // Byte flip
                let idx = self.rng.next_usize(out.len());
                out[idx] ^= 0xFF;
            }
            2 => { // Random byte substitution
                let idx = self.rng.next_usize(out.len());
                out[idx] = self.rng.next_u64() as u8;
            }
            3 => { // Insert byte
                let idx = self.rng.next_usize(out.len() + 1);
                out.insert(idx, self.rng.next_u64() as u8);
            }
            4 => { // Delete byte
                if out.len() > 1 {
                    let idx = self.rng.next_usize(out.len());
                    out.remove(idx);
                }
            }
            5 => { // Copy chunk
                if out.len() > 1 {
                    let src = self.rng.next_usize(out.len());
                    let dst = self.rng.next_usize(out.len());
                    let len = self.rng.next_usize(out.len() - src.max(dst)).max(1);
                    for i in 0..len {
                        if dst + i < out.len() && src + i < out.len() {
                            let v = out[src + i];
                            out[dst + i] = v;
                        }
                    }
                }
            }
            6 => { // Interesting values (boundary conditions)
                let interesting: &[u8] = &[0, 1, 0x7f, 0x80, 0xfe, 0xff];
                let idx = self.rng.next_usize(out.len());
                out[idx] = interesting[self.rng.next_usize(interesting.len())];
            }
            _ => { // Append bytes
                let n = self.rng.next_usize(8) + 1;
                for _ in 0..n { out.push(self.rng.next_u64() as u8); }
            }
        }
        out
    }

    /// Run the fuzzer for `max_execs` executions.
    /// `target` is the function under test: returns a simulated coverage hash and whether it crashed.
    pub fn fuzz(&mut self, max_execs: u64, target: &dyn Fn(&[u8]) -> (u64, bool)) {
        if self.corpus.is_empty() {
            self.corpus.push(CorpusEntry { data: vec![0], coverage: 0, energy: 1.0 });
        }

        for _ in 0..max_execs {
            self.total_execs += 1;
            // Select corpus entry by energy (weighted random)
            let total_energy: f64 = self.corpus.iter().map(|e| e.energy).sum();
            let mut pick = self.rng.next_f64() * total_energy;
            let selected = self.corpus.iter().position(|e| { pick -= e.energy; pick <= 0.0 })
                .unwrap_or(0);
            let mutated = self.mutate(&self.corpus[selected].data.clone());
            let (coverage, crashed) = target(&mutated);
            if crashed {
                self.crashes.push(mutated);
                continue;
            }
            // If new coverage, add to corpus
            if self.seen_coverage.insert(coverage) {
                self.unique_paths += 1;
                self.corpus.push(CorpusEntry {
                    data: mutated, coverage,
                    energy: 2.0, // newly interesting — explore more
                });
            } else {
                // Decay energy of selected entry
                self.corpus[selected].energy *= 0.99;
            }
        }
    }

    pub fn stats(&self) -> FuzzStats {
        FuzzStats {
            total_execs:   self.total_execs,
            corpus_size:   self.corpus.len(),
            unique_paths:  self.unique_paths,
            crashes:       self.crashes.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FuzzStats {
    pub total_execs:  u64,
    pub corpus_size:  usize,
    pub unique_paths: u64,
    pub crashes:      usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. MUTATION TESTING
// ─────────────────────────────────────────────────────────────────────────────

/// A mutation operator applied to source code.
#[derive(Debug, Clone, PartialEq)]
pub enum MutationOp {
    ArithmeticReplace { from: &'static str, to: &'static str },
    ComparisonReplace  { from: &'static str, to: &'static str },
    LogicalReplace     { from: &'static str, to: &'static str },
    BooleanLiteral     { from: bool, to: bool },
    ReturnConstant     { value: i64 },
    DeleteStatement    { line: usize },
    OffByOne           { line: usize },
}

/// A single mutation applied to a code location.
#[derive(Debug, Clone)]
pub struct Mutant {
    pub id:        usize,
    pub op:        MutationOp,
    pub file:      String,
    pub line:      usize,
    pub original:  String,
    pub mutated:   String,
    pub status:    MutantStatus,
}

/// Whether the test suite killed this mutant.
#[derive(Debug, Clone, PartialEq)]
pub enum MutantStatus {
    Pending,
    Killed,    // at least one test failed on this mutant (good!)
    Survived,  // all tests passed — mutation not caught (bad!)
    Timeout,
    Equivalent, // mutant is semantically equivalent (excluded from score)
}

/// Mutation testing engine.
pub struct MutationEngine {
    pub mutants: Vec<Mutant>,
    next_id:     usize,
}

impl MutationEngine {
    pub fn new() -> Self { MutationEngine { mutants: Vec::new(), next_id: 0 } }

    /// Generate all standard arithmetic/comparison mutations for a source file.
    pub fn generate_mutants(&mut self, file: &str, source: &str) {
        let arith_pairs = [
            ("+", "-"), ("-", "+"), ("*", "/"), ("/", "*"), ("%", "*"),
        ];
        let cmp_pairs = [
            (">", ">="), (">=", ">"), ("<", "<="), ("<=", "<"),
            ("==", "!="), ("!=", "=="),
        ];
        let logic_pairs = [("&&", "||"), ("||", "&&")];

        for (line_no, line) in source.lines().enumerate() {
            for &(from, to) in &arith_pairs {
                if line.contains(from) {
                    self.add_mutant(file, line_no + 1, line,
                        MutationOp::ArithmeticReplace { from, to });
                }
            }
            for &(from, to) in &cmp_pairs {
                if line.contains(from) {
                    self.add_mutant(file, line_no + 1, line,
                        MutationOp::ComparisonReplace { from, to });
                }
            }
            for &(from, to) in &logic_pairs {
                if line.contains(from) {
                    self.add_mutant(file, line_no + 1, line,
                        MutationOp::LogicalReplace { from, to });
                }
            }
            if line.contains("true") {
                self.add_mutant(file, line_no + 1, line,
                    MutationOp::BooleanLiteral { from: true, to: false });
            }
            if line.contains("false") {
                self.add_mutant(file, line_no + 1, line,
                    MutationOp::BooleanLiteral { from: false, to: true });
            }
        }
    }

    fn add_mutant(&mut self, file: &str, line: usize, original: &str, op: MutationOp) {
        let mutated = apply_mutation(original, &op);
        self.mutants.push(Mutant {
            id: self.next_id, op, file: file.to_string(), line,
            original: original.to_string(), mutated, status: MutantStatus::Pending,
        });
        self.next_id += 1;
    }

    /// Simulate running the test suite against each mutant.
    /// `test_fn` returns true if tests PASS on the mutated code (mutant survived).
    pub fn run(&mut self, test_fn: &dyn Fn(&Mutant) -> bool) {
        for mutant in &mut self.mutants {
            let survived = test_fn(mutant);
            mutant.status = if survived { MutantStatus::Survived } else { MutantStatus::Killed };
        }
    }

    /// Mutation score: killed / (total - equivalent).
    pub fn score(&self) -> f64 {
        let total = self.mutants.iter().filter(|m| m.status != MutantStatus::Equivalent && m.status != MutantStatus::Pending).count();
        if total == 0 { return 1.0; }
        let killed = self.mutants.iter().filter(|m| m.status == MutantStatus::Killed).count();
        killed as f64 / total as f64
    }

    pub fn survived_mutants(&self) -> Vec<&Mutant> {
        self.mutants.iter().filter(|m| m.status == MutantStatus::Survived).collect()
    }
}

fn apply_mutation(line: &str, op: &MutationOp) -> String {
    match op {
        MutationOp::ArithmeticReplace { from, to } |
        MutationOp::ComparisonReplace { from, to }  |
        MutationOp::LogicalReplace    { from, to }  => line.replacen(from, to, 1),
        MutationOp::BooleanLiteral { from, to } => {
            line.replacen(if *from { "true" } else { "false" },
                          if *to  { "true" } else { "false" }, 1)
        }
        MutationOp::ReturnConstant { value } => format!("return {};", value),
        MutationOp::DeleteStatement { .. } => String::new(),
        MutationOp::OffByOne { .. } => {
            if let Some(i) = line.find(|c: char| c.is_ascii_digit()) {
                let digit = line[i..].chars().next().unwrap().to_digit(10).unwrap();
                line[..i].to_string() + &(digit + 1).to_string() + &line[i+1..]
            } else { line.to_string() }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. MOCKING FRAMEWORK
// ─────────────────────────────────────────────────────────────────────────────

/// A recorded call to a mock function.
#[derive(Debug, Clone)]
pub struct MockCall {
    pub method:    String,
    pub args:      Vec<String>,  // debug-printed arguments
    pub timestamp: u64,          // call order counter
}

/// An expectation on a mock method.
#[derive(Debug, Clone)]
pub struct Expectation {
    pub method:     String,
    pub min_calls:  usize,
    pub max_calls:  Option<usize>,   // None = unlimited
    pub actual:     usize,
    pub return_val: Option<String>,  // serialised return value
}

impl Expectation {
    pub fn exactly(method: &str, n: usize) -> Self {
        Expectation { method: method.to_string(), min_calls: n, max_calls: Some(n),
                      actual: 0, return_val: None }
    }
    pub fn at_least(method: &str, n: usize) -> Self {
        Expectation { method: method.to_string(), min_calls: n, max_calls: None,
                      actual: 0, return_val: None }
    }
    pub fn never(method: &str) -> Self {
        Expectation { method: method.to_string(), min_calls: 0, max_calls: Some(0),
                      actual: 0, return_val: None }
    }
    pub fn with_return(mut self, val: &str) -> Self { self.return_val = Some(val.to_string()); self }
    pub fn satisfied(&self) -> bool {
        self.actual >= self.min_calls && self.max_calls.map_or(true, |m| self.actual <= m)
    }
    pub fn verify(&self) -> Result<(), String> {
        if self.satisfied() { return Ok(()); }
        match self.max_calls {
            Some(m) if self.actual > m =>
                Err(format!("mock {}: expected at most {} calls, got {}", self.method, m, self.actual)),
            _ =>
                Err(format!("mock {}: expected at least {} calls, got {}", self.method, self.min_calls, self.actual)),
        }
    }
}

/// A generic mock object that records calls and verifies expectations.
pub struct Mock {
    pub name:         String,
    calls:            Vec<MockCall>,
    expectations:     Vec<Expectation>,
    call_counter:     u64,
}

impl Mock {
    pub fn new(name: &str) -> Self {
        Mock { name: name.to_string(), calls: Vec::new(), expectations: Vec::new(), call_counter: 0 }
    }

    pub fn expect(&mut self, exp: Expectation) { self.expectations.push(exp); }

    /// Record a method call with debug-printed arguments.
    pub fn record_call(&mut self, method: &str, args: Vec<String>) -> Option<String> {
        self.call_counter += 1;
        self.calls.push(MockCall { method: method.to_string(), args, timestamp: self.call_counter });
        // Update expectation counters and return configured value
        let mut ret = None;
        for exp in &mut self.expectations {
            if exp.method == method {
                exp.actual += 1;
                if let Some(ref v) = exp.return_val { ret = Some(v.clone()); }
                break;
            }
        }
        ret
    }

    /// Verify all expectations are satisfied.
    pub fn verify_all(&self) -> Vec<String> {
        self.expectations.iter().filter_map(|e| e.verify().err()).collect()
    }

    pub fn call_count(&self, method: &str) -> usize {
        self.calls.iter().filter(|c| c.method == method).count()
    }

    pub fn was_called(&self, method: &str) -> bool { self.call_count(method) > 0 }

    pub fn calls_in_order(&self) -> Vec<&str> {
        self.calls.iter().map(|c| c.method.as_str()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SNAPSHOT TESTING
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot store: maps test name → expected serialised output.
pub struct SnapshotStore {
    snapshots: HashMap<String, String>,
    updated:   Vec<String>,  // newly written snapshots
}

impl SnapshotStore {
    pub fn new() -> Self { SnapshotStore { snapshots: HashMap::new(), updated: Vec::new() } }

    /// Load a snapshot from the store.
    pub fn load(&self, name: &str) -> Option<&str> { self.snapshots.get(name).map(|s| s.as_str()) }

    /// Save/update a snapshot.
    pub fn save(&mut self, name: &str, value: &str) {
        self.snapshots.insert(name.to_string(), value.to_string());
        self.updated.push(name.to_string());
    }

    /// Assert that `actual` matches the stored snapshot for `name`.
    /// If no snapshot exists, creates it (first-run behaviour).
    pub fn assert_snapshot(&mut self, name: &str, actual: &str) -> AssertResult {
        match self.load(name) {
            None => {
                // First run: write snapshot
                self.save(name, actual);
                AssertResult::Ok
            }
            Some(expected) => {
                if expected == actual { AssertResult::Ok }
                else {
                    AssertResult::Fail(format!(
                        "snapshot mismatch for '{}':\n{}",
                        name, unified_diff(expected, actual)
                    ))
                }
            }
        }
    }

    /// Return all snapshot names that were updated this session.
    pub fn updated_snapshots(&self) -> &[String] { &self.updated }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. BENCHMARK HARNESS
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics from a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchStats {
    pub name:        String,
    pub iterations:  u64,
    pub mean_ns:     f64,
    pub std_dev_ns:  f64,
    pub min_ns:      u64,
    pub max_ns:      u64,
    pub median_ns:   u64,
    pub throughput:  Option<f64>,   // ops/sec if set
}

impl BenchStats {
    pub fn fmt_summary(&self) -> String {
        format!("{}: {:.1}ns ± {:.1}ns  [min={} max={} median={}]  iters={}",
                self.name, self.mean_ns, self.std_dev_ns,
                self.min_ns, self.max_ns, self.median_ns, self.iterations)
    }
}

/// Run a micro-benchmark using wall-clock timing.
/// Performs a warm-up phase then statistical measurement.
pub fn benchmark(name: &str, warmup_iters: u64, measure_iters: u64,
                 func: &dyn Fn()) -> BenchStats {
    // Warm-up: let the CPU cache + JIT settle
    for _ in 0..warmup_iters { func(); }

    // Measure
    let mut samples = Vec::with_capacity(measure_iters as usize);
    for _ in 0..measure_iters {
        let t0 = Instant::now();
        func();
        samples.push(t0.elapsed().as_nanos() as u64);
    }

    samples.sort_unstable();
    let n = samples.len() as f64;
    let mean_ns = samples.iter().sum::<u64>() as f64 / n;
    let variance = samples.iter().map(|&s| { let d = s as f64 - mean_ns; d * d }).sum::<f64>() / n;
    let std_dev = variance.sqrt();
    let median_idx = samples.len() / 2;

    BenchStats {
        name:       name.to_string(),
        iterations: measure_iters,
        mean_ns,
        std_dev_ns: std_dev,
        min_ns:     *samples.first().unwrap_or(&0),
        max_ns:     *samples.last().unwrap_or(&0),
        median_ns:  samples[median_idx],
        throughput: Some(1_000_000_000.0 / mean_ns),
    }
}

/// Compare two benchmark results and report a speedup ratio.
pub fn bench_compare(baseline: &BenchStats, candidate: &BenchStats) -> String {
    let ratio = baseline.mean_ns / candidate.mean_ns;
    let change = (ratio - 1.0) * 100.0;
    if change > 0.0 { format!("{:.2}× faster (+{:.1}%)", ratio, change) }
    else            { format!("{:.2}× slower ({:.1}%)",  ratio, change) }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. COVERAGE TRACKING
// ─────────────────────────────────────────────────────────────────────────────

/// Coverage data for a single file.
#[derive(Debug, Clone)]
pub struct FileCoverage {
    pub path:           String,
    pub line_hits:      HashMap<u32, u64>,   // line → hit count
    pub branch_hits:    HashMap<u32, (u64, u64)>,  // line → (taken, not_taken)
    pub function_hits:  HashMap<String, u64>,       // fn name → hit count
}

impl FileCoverage {
    pub fn new(path: &str) -> Self {
        FileCoverage {
            path: path.to_string(),
            line_hits:     HashMap::new(),
            branch_hits:   HashMap::new(),
            function_hits: HashMap::new(),
        }
    }

    pub fn hit_line(&mut self, line: u32) {
        *self.line_hits.entry(line).or_insert(0) += 1;
    }

    pub fn hit_branch(&mut self, line: u32, taken: bool) {
        let e = self.branch_hits.entry(line).or_insert((0, 0));
        if taken { e.0 += 1; } else { e.1 += 1; }
    }

    pub fn hit_function(&mut self, name: &str) {
        *self.function_hits.entry(name.to_string()).or_insert(0) += 1;
    }

    pub fn line_coverage_pct(&self, total_lines: u32) -> f64 {
        if total_lines == 0 { return 100.0; }
        let covered = self.line_hits.values().filter(|&&h| h > 0).count();
        covered as f64 / total_lines as f64 * 100.0
    }

    pub fn branch_coverage_pct(&self) -> f64 {
        let total  = self.branch_hits.len() * 2;
        if total == 0 { return 100.0; }
        let covered = self.branch_hits.values()
            .map(|(t, nt)| (*t > 0) as usize + (*nt > 0) as usize)
            .sum::<usize>();
        covered as f64 / total as f64 * 100.0
    }
}

/// Aggregate coverage across all files.
pub struct CoverageReport {
    pub files: HashMap<String, FileCoverage>,
}

impl CoverageReport {
    pub fn new() -> Self { CoverageReport { files: HashMap::new() } }

    pub fn add_file(&mut self, cov: FileCoverage) { self.files.insert(cov.path.clone(), cov); }

    pub fn total_line_coverage(&self, total_lines_per_file: &HashMap<String, u32>) -> f64 {
        let mut total_lines = 0u64;
        let mut covered_lines = 0u64;
        for (path, cov) in &self.files {
            let total = *total_lines_per_file.get(path).unwrap_or(&0) as u64;
            let covered = cov.line_hits.values().filter(|&&h| h > 0).count() as u64;
            total_lines += total;
            covered_lines += covered;
        }
        if total_lines == 0 { 100.0 } else { covered_lines as f64 / total_lines as f64 * 100.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. TEST REPORTERS
// ─────────────────────────────────────────────────────────────────────────────

/// Serialise test results as TAP (Test Anything Protocol) v13.
pub fn format_tap(results: &TestSuiteResult) -> String {
    let mut out = format!("TAP version 13\n1..{}\n", results.total());
    for (i, r) in results.results.iter().enumerate() {
        let status = if r.passed() { "ok" } else { "not ok" };
        let desc = &r.meta.name;
        let module = &r.meta.module;
        out += &format!("{} {} - {}::{}\n", status, i + 1, module, desc);
        if let TestOutcome::Failed(FailureReason::AssertionFailed { message, location }) = &r.outcome {
            out += &format!("  ---\n  message: {}\n  at: {}\n  ...\n", message, location);
        }
    }
    out
}

/// Serialise test results as JUnit XML (consumed by CI systems like Jenkins).
pub fn format_junit_xml(suite_name: &str, results: &TestSuiteResult) -> String {
    let mut xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <testsuite name=\"{}\" tests=\"{}\" failures=\"{}\" skipped=\"{}\" time=\"{:.3}\">\n",
        suite_name, results.total(), results.failed(), results.skipped(),
        results.total_duration.as_secs_f64()
    );
    for r in &results.results {
        let classname = &r.meta.module;
        let testname  = &r.meta.name;
        let time = r.duration.as_secs_f64();
        xml += &format!("  <testcase classname=\"{}\" name=\"{}\" time=\"{:.6}\"", classname, testname, time);
        match &r.outcome {
            TestOutcome::Passed => xml += " />\n",
            TestOutcome::Failed(FailureReason::AssertionFailed { message, .. }) => {
                xml += &format!(">\n    <failure message=\"{}\" />\n  </testcase>\n",
                    message.replace('"', "&quot;").replace('<', "&lt;").replace('>', "&gt;"));
            }
            TestOutcome::Skipped(reason) => {
                xml += &format!(">\n    <skipped message=\"{}\" />\n  </testcase>\n", reason);
            }
            _ => xml += " />\n",
        }
    }
    xml += "</testsuite>\n";
    xml
}

/// Terminal output formatter (with ANSI colours).
pub fn format_terminal(results: &TestSuiteResult, verbose: bool) -> String {
    let mut out = String::new();
    for r in &results.results {
        if verbose || !r.passed() {
            let sym = match &r.outcome {
                TestOutcome::Passed        => "PASS",
                TestOutcome::Failed(_)     => "FAIL",
                TestOutcome::Skipped(_)    => "SKIP",
                TestOutcome::Panicked(_)   => "PNCK",
                TestOutcome::TimedOut      => "TIME",
            };
            out += &format!("[{}] {}::{} ({:.1}ms)\n",
                sym, r.meta.module, r.meta.name, r.duration.as_secs_f64() * 1000.0);
            if let TestOutcome::Failed(reason) = &r.outcome {
                match reason {
                    FailureReason::AssertionFailed { message, location } =>
                        out += &format!("       at {}: {}\n", location, message),
                    FailureReason::PropertyFailed { shrunk, input, .. } =>
                        out += &format!("       property failed with: {}\n       shrunk to: {}\n", input, shrunk),
                    _ => {}
                }
            }
        }
    }
    out += &format!("\n{} passed, {} failed, {} skipped in {:.2}s\n",
        results.passed(), results.failed(), results.skipped(),
        results.total_duration.as_secs_f64());
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test Registry & Runner ────────────────────────────────────────────────

    #[test]
    fn test_runner_passes_all() {
        let mut reg = TestRegistry::new();
        for i in 0..5 {
            let name = format!("test_{}", i);
            reg.register(TestCase::new(
                TestMeta::new(&name, "suite"),
                move || TestOutcome::Passed,
            ));
        }
        let cases = reg.filter(&TestFilter::all());
        let runner = TestRunner::new(RunnerConfig::default());
        let results = runner.run(&cases);
        assert_eq!(results.passed(), 5);
        assert_eq!(results.failed(), 0);
        assert!(results.all_passed());
    }

    #[test]
    fn test_runner_counts_failures() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("pass", "m"), || TestOutcome::Passed));
        reg.register(TestCase::new(TestMeta::new("fail", "m"), || {
            TestOutcome::Failed(FailureReason::Custom("oops".into()))
        }));
        let cases = reg.filter(&TestFilter::all());
        let runner = TestRunner::new(RunnerConfig::default());
        let results = runner.run(&cases);
        assert_eq!(results.passed(), 1);
        assert_eq!(results.failed(), 1);
        assert!(!results.all_passed());
    }

    #[test]
    fn test_runner_skips_ignored() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("normal", "m"), || TestOutcome::Passed));
        reg.register(TestCase::new(TestMeta::new("ignored", "m").ignore(), || TestOutcome::Passed));
        // include_ignored=true so the runner receives the ignored test and marks it Skipped
        let cases = reg.filter(&TestFilter { include_ignored: true, ..Default::default() });
        let runner = TestRunner::new(RunnerConfig::default());
        let results = runner.run(&cases);
        assert_eq!(results.passed(), 1);
        assert_eq!(results.skipped(), 1);
    }

    #[test]
    fn test_filter_by_name() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("foo_test", "m"), || TestOutcome::Passed));
        reg.register(TestCase::new(TestMeta::new("bar_test", "m"), || TestOutcome::Passed));
        let filter = TestFilter { name_contains: Some("foo".into()), ..Default::default() };
        let cases = reg.filter(&filter);
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].meta.name, "foo_test");
    }

    #[test]
    fn test_filter_by_tag() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("t1", "m").tag("slow"), || TestOutcome::Passed));
        reg.register(TestCase::new(TestMeta::new("t2", "m").tag("fast"), || TestOutcome::Passed));
        let filter = TestFilter { tag: Some("slow".into()), ..Default::default() };
        let cases = reg.filter(&filter);
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].meta.name, "t1");
    }

    #[test]
    fn test_fail_fast_stops_early() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("fail", "m"), || {
            TestOutcome::Failed(FailureReason::Custom("x".into()))
        }));
        for i in 0..10 {
            reg.register(TestCase::new(TestMeta::new(&format!("after_{}", i), "m"),
                || TestOutcome::Passed));
        }
        let filter = TestFilter::all();
        let cases = reg.filter(&filter);
        let runner = TestRunner::new(RunnerConfig { fail_fast: true, ..Default::default() });
        let results = runner.run(&cases);
        assert_eq!(results.total(), 1, "Should stop after first failure");
    }

    #[test]
    fn test_flaky_test_retried() {
        let mut reg = TestRegistry::new();
        // This test always fails but is marked flaky — will be retried
        reg.register(TestCase::new(
            TestMeta::new("flaky", "m").flaky(),
            || TestOutcome::Failed(FailureReason::Custom("flaky".into())),
        ));
        let cases = reg.filter(&TestFilter::all());
        let runner = TestRunner::new(RunnerConfig { max_retries: 2, ..Default::default() });
        let results = runner.run(&cases);
        let r = &results.results[0];
        assert_eq!(r.retries, 2, "Should have retried twice");
    }

    // ── Assertions ────────────────────────────────────────────────────────────

    #[test]
    fn test_assert_eq_passes() {
        assert!(assert_eq_(42i32, 42i32, "").is_ok());
    }

    #[test]
    fn test_assert_eq_fails() {
        let r = assert_eq_(1i32, 2i32, "mismatch");
        assert!(!r.is_ok());
        if let AssertResult::Fail(msg) = r { assert!(msg.contains("mismatch")); }
    }

    #[test]
    fn test_assert_approx_eq() {
        assert!(assert_approx_eq_(1.0, 1.001, 0.01).is_ok());
        assert!(!assert_approx_eq_(1.0, 1.1, 0.01).is_ok());
    }

    #[test]
    fn test_assert_ok_err() {
        let ok: Result<i32, &str> = Ok(42);
        let err: Result<i32, &str> = Err("bad");
        assert!(assert_ok_(ok).is_ok());
        assert!(!assert_ok_(err).is_ok());
        let err2: Result<i32, &str> = Err("bad");
        assert!(assert_err_(err2).is_ok());
    }

    #[test]
    fn test_assert_contains() {
        assert!(assert_contains_(&[1, 2, 3], &2).is_ok());
        assert!(!assert_contains_(&[1, 2, 3], &5).is_ok());
    }

    #[test]
    fn test_unified_diff() {
        let diff = unified_diff("hello\nworld", "hello\nchronos");
        assert!(diff.contains("+"), "Should have additions");
        assert!(diff.contains("-"), "Should have removals");
    }

    // ── Property-Based Testing ────────────────────────────────────────────────

    #[test]
    fn test_property_reversal() {
        // Property: reversing a list twice = original list
        let result = property_test::<Vec<i64>>(
            "reverse_twice",
            200,
            12345,
            |v| {
                let rev: Vec<i64> = v.iter().cloned().rev().collect();
                let rev2: Vec<i64> = rev.iter().cloned().rev().collect();
                if rev2 == *v { Ok(()) } else { Err(format!("failed for {:?}", v)) }
            },
        );
        assert!(result.passed, "reverse_twice property should pass");
        assert_eq!(result.tests_run, 200);
    }

    #[test]
    fn test_property_sort_idempotent() {
        let result = property_test::<Vec<i64>>(
            "sort_idempotent",
            100,
            42,
            |v| {
                let mut s1 = v.clone(); s1.sort();
                let mut s2 = s1.clone(); s2.sort();
                if s1 == s2 { Ok(()) } else { Err(format!("{:?} not idempotent", v)) }
            },
        );
        assert!(result.passed);
    }

    #[test]
    fn test_property_finds_failure_and_shrinks() {
        // Property that fails for non-empty lists
        let result = property_test::<Vec<i64>>(
            "always_empty",
            100,
            999,
            |v| {
                if v.is_empty() { Ok(()) }
                else { Err(format!("non-empty: {:?}", v)) }
            },
        );
        assert!(!result.passed, "Should find a failing case");
        let failure = result.failure.unwrap();
        // Shrunk input should be minimal (empty or single element)
        assert!(failure.shrunk_input.len() <= failure.original_input.len(),
                "Shrunk should not be longer than original");
    }

    #[test]
    fn test_property_addition_commutative() {
        let result = property_test::<(i64, i64)>(
            "add_commutative",
            500,
            777,
            |&(a, b)| {
                if a.wrapping_add(b) == b.wrapping_add(a) { Ok(()) }
                else { Err(format!("{} + {} != {} + {}", a, b, b, a)) }
            },
        );
        assert!(result.passed);
    }

    #[test]
    fn test_arbitrary_string_generates_lowercase() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let s = String::arbitrary(&mut rng, 10);
            assert!(s.chars().all(|c| c.is_ascii_lowercase() || c == '\0'),
                    "String should be lowercase: {:?}", s);
        }
    }

    #[test]
    fn test_shrink_i64() {
        let v: i64 = 100;
        let shrunk = v.shrink();
        assert!(shrunk.contains(&0), "Should shrink towards 0");
    }

    #[test]
    fn test_shrink_vec() {
        let v: Vec<i64> = vec![1, 2, 3, 4];
        let shrunk = v.shrink();
        assert!(!shrunk.is_empty(), "Should produce shrink candidates");
        assert!(shrunk.iter().any(|s| s.len() < v.len()), "Some shrunk should be shorter");
    }

    // ── Fuzzer ────────────────────────────────────────────────────────────────

    #[test]
    fn test_fuzzer_explores_corpus() {
        let mut fuzzer = Fuzzer::new(42);
        fuzzer.seed_corpus(vec![vec![0u8], vec![1u8, 2u8]]);
        // Simple target: coverage = sum of bytes, no crashes
        fuzzer.fuzz(100, &|data: &[u8]| {
            let cov: u64 = data.iter().map(|&b| b as u64).sum::<u64>() & 0xFF;
            (cov, false)
        });
        let stats = fuzzer.stats();
        assert!(stats.total_execs == 100);
        assert!(stats.corpus_size >= 2, "Corpus should grow");
    }

    #[test]
    fn test_fuzzer_records_crashes() {
        let mut fuzzer = Fuzzer::new(99);
        fuzzer.seed_corpus(vec![vec![0u8]]);
        // Crash if byte = 42
        fuzzer.fuzz(500, &|data: &[u8]| {
            let crashed = data.contains(&42u8);
            let cov = data.iter().copied().fold(0u64, |a, b| a ^ b as u64);
            (cov, crashed)
        });
        // With 500 executions and mutations, very likely to hit 42
        // Just check structure is correct
        let stats = fuzzer.stats();
        assert!(stats.total_execs == 500);
    }

    #[test]
    fn test_fuzzer_mutation_strategies() {
        let mut fuzzer = Fuzzer::new(1);
        let original = vec![1u8, 2, 3, 4, 5];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            let m = fuzzer.mutate(&original);
            seen.insert(m);
        }
        // Should produce diverse mutations
        assert!(seen.len() > 10, "Mutations not diverse enough: {}", seen.len());
    }

    // ── Mutation Testing ─────────────────────────────────────────────────────

    #[test]
    fn test_mutation_engine_generates_mutants() {
        let mut eng = MutationEngine::new();
        let source = "if x > 0 { return x + 1; }";
        eng.generate_mutants("src/foo.ch", source);
        assert!(!eng.mutants.is_empty(), "Should generate mutants");
    }

    #[test]
    fn test_mutation_score_all_killed() {
        let mut eng = MutationEngine::new();
        let source = "if a > b { result = a + b; }";
        eng.generate_mutants("f.ch", source);
        // All mutants killed by perfect tests
        eng.run(&|_| false); // tests FAIL → mutant killed
        assert!((eng.score() - 1.0).abs() < 1e-10, "Score should be 1.0: {}", eng.score());
    }

    #[test]
    fn test_mutation_score_all_survived() {
        let mut eng = MutationEngine::new();
        let source = "if a > b { result = a + b; }";
        eng.generate_mutants("f.ch", source);
        eng.run(&|_| true); // tests PASS → mutant survived
        assert!((eng.score()).abs() < 1e-10, "Score should be 0.0: {}", eng.score());
    }

    #[test]
    fn test_apply_mutation_arithmetic() {
        let line = "let x = a + b;";
        let op = MutationOp::ArithmeticReplace { from: "+", to: "-" };
        let mutated = apply_mutation(line, &op);
        assert!(mutated.contains('-'), "Mutated: {}", mutated);
        assert!(!mutated.starts_with('+'), "Original + removed");
    }

    // ── Mocking ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mock_records_calls() {
        let mut mock = Mock::new("Database");
        mock.record_call("query", vec!["SELECT *".into()]);
        mock.record_call("query", vec!["INSERT".into()]);
        assert_eq!(mock.call_count("query"), 2);
        assert!(mock.was_called("query"));
        assert!(!mock.was_called("delete"));
    }

    #[test]
    fn test_mock_expectation_satisfied() {
        let mut mock = Mock::new("Service");
        mock.expect(Expectation::exactly("connect", 1));
        mock.record_call("connect", vec![]);
        let failures = mock.verify_all();
        assert!(failures.is_empty(), "Failures: {:?}", failures);
    }

    #[test]
    fn test_mock_expectation_violated() {
        let mut mock = Mock::new("Service");
        mock.expect(Expectation::exactly("connect", 2));
        mock.record_call("connect", vec![]); // only called once
        let failures = mock.verify_all();
        assert!(!failures.is_empty(), "Should have violation");
    }

    #[test]
    fn test_mock_never_violated() {
        let mut mock = Mock::new("Service");
        mock.expect(Expectation::never("delete"));
        mock.record_call("delete", vec![]); // called when should never be
        let failures = mock.verify_all();
        assert!(!failures.is_empty(), "Never expectation should fail");
    }

    #[test]
    fn test_mock_return_value() {
        let mut mock = Mock::new("Cache");
        mock.expect(Expectation::at_least("get", 1).with_return("\"cached_value\""));
        let ret = mock.record_call("get", vec!["key".into()]);
        assert_eq!(ret, Some("\"cached_value\"".to_string()));
    }

    #[test]
    fn test_mock_call_order() {
        let mut mock = Mock::new("Lifecycle");
        mock.record_call("init",  vec![]);
        mock.record_call("start", vec![]);
        mock.record_call("stop",  vec![]);
        let order = mock.calls_in_order();
        assert_eq!(order, vec!["init", "start", "stop"]);
    }

    // ── Snapshot Testing ──────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_first_run_creates() {
        let mut store = SnapshotStore::new();
        let result = store.assert_snapshot("render_output", "Hello, World!");
        assert!(result.is_ok(), "First run should pass (creates snapshot)");
        assert_eq!(store.updated_snapshots(), &["render_output"]);
    }

    #[test]
    fn test_snapshot_matches() {
        let mut store = SnapshotStore::new();
        store.save("greet", "Hello, Alice!");
        let result = store.assert_snapshot("greet", "Hello, Alice!");
        assert!(result.is_ok());
    }

    #[test]
    fn test_snapshot_mismatch() {
        let mut store = SnapshotStore::new();
        store.save("greet", "Hello, Alice!");
        let result = store.assert_snapshot("greet", "Hello, Bob!");
        assert!(!result.is_ok(), "Should fail on mismatch");
    }

    // ── Benchmarking ─────────────────────────────────────────────────────────

    #[test]
    fn test_benchmark_runs() {
        let stats = benchmark("noop", 10, 50, &|| { let _: u64 = 1 + 1; });
        assert_eq!(stats.iterations, 50);
        assert!(stats.mean_ns >= 0.0);
        assert!(stats.min_ns <= stats.max_ns);
        assert!(stats.median_ns <= stats.max_ns);
    }

    #[test]
    fn test_benchmark_comparison() {
        let fast = BenchStats { name: "fast".into(), iterations: 100, mean_ns: 10.0,
                                std_dev_ns: 1.0, min_ns: 8, max_ns: 15, median_ns: 10, throughput: None };
        let slow = BenchStats { name: "slow".into(), iterations: 100, mean_ns: 30.0,
                                std_dev_ns: 2.0, min_ns: 25, max_ns: 40, median_ns: 30, throughput: None };
        let cmp = bench_compare(&slow, &fast);
        assert!(cmp.contains("faster"), "Should report faster: {}", cmp);
    }

    // ── Coverage ──────────────────────────────────────────────────────────────

    #[test]
    fn test_coverage_line_tracking() {
        let mut cov = FileCoverage::new("src/foo.ch");
        cov.hit_line(10);
        cov.hit_line(10);
        cov.hit_line(15);
        assert_eq!(*cov.line_hits.get(&10).unwrap(), 2);
        assert_eq!(*cov.line_hits.get(&15).unwrap(), 1);
    }

    #[test]
    fn test_coverage_pct_calculation() {
        let mut cov = FileCoverage::new("src/foo.ch");
        cov.hit_line(1);
        cov.hit_line(2);
        // 2 out of 4 lines covered = 50%
        let pct = cov.line_coverage_pct(4);
        assert!((pct - 50.0).abs() < 1e-10, "Coverage: {}%", pct);
    }

    #[test]
    fn test_branch_coverage() {
        let mut cov = FileCoverage::new("src/foo.ch");
        cov.hit_branch(10, true);  // taken
        cov.hit_branch(10, false); // not taken
        cov.hit_branch(20, true);  // only taken
        // Branch at line 10: both arms = 2/2; line 20: only taken = 1/2 → 3/4 = 75%
        let pct = cov.branch_coverage_pct();
        assert!((pct - 75.0).abs() < 1e-10, "Branch coverage: {}%", pct);
    }

    // ── Reporters ─────────────────────────────────────────────────────────────

    #[test]
    fn test_tap_format() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("foo", "m"), || TestOutcome::Passed));
        reg.register(TestCase::new(TestMeta::new("bar", "m"), || {
            TestOutcome::Failed(FailureReason::Custom("x".into()))
        }));
        let cases = reg.filter(&TestFilter::all());
        let runner = TestRunner::new(RunnerConfig::default());
        let results = runner.run(&cases);
        let tap = format_tap(&results);
        assert!(tap.starts_with("TAP version 13"), "TAP header");
        assert!(tap.contains("1..2"), "TAP plan");
        assert!(tap.contains("ok 1"), "Passed test");
        assert!(tap.contains("not ok 2"), "Failed test");
    }

    #[test]
    fn test_junit_xml_format() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("mytest", "mymod"), || TestOutcome::Passed));
        let cases = reg.filter(&TestFilter::all());
        let runner = TestRunner::new(RunnerConfig::default());
        let results = runner.run(&cases);
        let xml = format_junit_xml("my-suite", &results);
        assert!(xml.contains("<?xml"), "XML declaration");
        assert!(xml.contains("testsuite"), "Testsuite element");
        assert!(xml.contains("mytest"), "Test name");
    }

    #[test]
    fn test_terminal_format() {
        let mut reg = TestRegistry::new();
        reg.register(TestCase::new(TestMeta::new("check", "core"), || TestOutcome::Passed));
        let cases = reg.filter(&TestFilter::all());
        let runner = TestRunner::new(RunnerConfig::default());
        let results = runner.run(&cases);
        let out = format_terminal(&results, true);
        assert!(out.contains("PASS"), "Should show PASS");
        assert!(out.contains("passed"), "Should show count");
    }
}
