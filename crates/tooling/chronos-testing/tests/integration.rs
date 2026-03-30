//! Integration tests for chronos-testing.
use chronos_testing::*;

// ── Test registry ─────────────────────────────────────────────────────────────

#[test]
fn test_registry_run_passing_test() {
    let mut registry = TestRegistry::new();
    registry.register(TestCase::new(
        TestMeta::new("always_passes", "integration"),
        || TestOutcome::Passed,
    ));
    let runner = TestRunner::new(RunnerConfig::default());
    let result = runner.run(&registry.filter(&TestFilter::all()));
    assert_eq!(result.passed(), 1);
    assert_eq!(result.failed(), 0);
}

#[test]
fn test_registry_run_failing_test() {
    let mut registry = TestRegistry::new();
    registry.register(TestCase::new(
        TestMeta::new("always_fails", "integration"),
        || TestOutcome::Failed(FailureReason::AssertionFailed {
            message: "expected 1 == 2".into(),
            location: Location { file: "test.rs", line: 1 },
        }),
    ));
    let runner = TestRunner::new(RunnerConfig::default());
    let result = runner.run(&registry.filter(&TestFilter::all()));
    assert_eq!(result.passed(), 0);
    assert_eq!(result.failed(), 1);
}

// ── TAP output ───────────────────────────────────────────────────────────────

fn make_suite_result(name: &str, outcomes: Vec<(&str, TestOutcome)>) -> TestSuiteResult {
    let results = outcomes
        .into_iter()
        .map(|(test_name, outcome)| TestResult {
            meta:     TestMeta::new(test_name, name),
            outcome,
            duration: std::time::Duration::ZERO,
            retries:  0,
        })
        .collect();
    TestSuiteResult {
        results,
        total_duration: std::time::Duration::ZERO,
    }
}

#[test]
fn test_tap_output_format() {
    let suite = make_suite_result("my_suite", vec![
        ("test_a", TestOutcome::Passed),
        ("test_b", TestOutcome::Passed),
        ("test_c", TestOutcome::Failed(FailureReason::AssertionFailed {
            message: "oops".into(),
            location: Location { file: "test.rs", line: 42 },
        })),
    ]);
    let tap = format_tap(&suite);
    assert!(tap.contains("TAP version 13"));
    assert!(tap.contains("1..3"));
    assert!(tap.contains("ok 1"));
    assert!(tap.contains("not ok 3"));
}

// ── JUnit XML output ─────────────────────────────────────────────────────────

#[test]
fn test_junit_xml_output_format() {
    let suite = make_suite_result("xml_suite", vec![
        ("test_pass", TestOutcome::Passed),
        ("test_skip", TestOutcome::Skipped("ignored".into())),
    ]);
    let xml = format_junit_xml("xml_suite", &suite);
    assert!(xml.contains("<testsuite"));
    assert!(xml.contains("testcase"));
    assert!(xml.contains("xml_suite"));
}

// ── Property-based testing ───────────────────────────────────────────────────

#[test]
fn test_property_commutative_addition() {
    // For i64, a + b == b + a
    let result = property_test::<(i64, i64)>(
        "commutative_add",
        100,
        42,
        |&(a, b)| {
            if a.wrapping_add(b) == b.wrapping_add(a) { Ok(()) }
            else { Err(format!("{} + {} != {} + {}", a, b, b, a)) }
        },
    );
    assert!(result.passed, "addition should be commutative");
}

#[test]
fn test_property_finds_counterexample() {
    // This property is false: n * 2 == n + 1 (only holds for n=1)
    let result = property_test::<i64>(
        "false_property",
        100,
        99,
        |&n| {
            if n.wrapping_mul(2) == n.wrapping_add(1) { Ok(()) }
            else { Err(format!("{} * 2 != {} + 1", n, n)) }
        },
    );
    assert!(!result.passed, "should find a counterexample");
}

// ── Benchmark ────────────────────────────────────────────────────────────────

#[test]
fn test_benchmark_returns_stats() {
    let stats = benchmark("no_op", 10, 100, &|| {
        let _ = 1 + 1;
    });
    assert!(stats.iterations >= 100);
    assert!(stats.mean_ns >= 0.0);
    assert!(stats.std_dev_ns >= 0.0);
}
