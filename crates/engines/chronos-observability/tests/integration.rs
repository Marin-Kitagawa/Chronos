//! Integration tests for chronos-observability.
//! Exercises the three-pillar telemetry pipeline end-to-end.
use chronos_observability::*;

fn make_rng() -> Xorshift128Plus {
    Xorshift128Plus::new(0xDEAD_BEEF, 0xCAFE_1234)
}

// ── Tracing pipeline ──────────────────────────────────────────────────────────

#[test]
fn test_full_trace_lifecycle() {
    let mut rng = make_rng();
    let sampler = AlwaysOnSampler;
    let trace_id = TraceId::generate(&mut rng);
    let span_id  = SpanId::generate(&mut rng);

    let ctx = SpanContext::new(trace_id, span_id, TraceFlags::SAMPLED);
    let mut span = Span::new(ctx, None, "http.request", SpanKind::Server, 1_000);
    span.set_attr("http.method", AttrValue::Str("POST".into()));
    span.set_attr("http.url",    AttrValue::Str("/api/users".into()));
    span.add_event("request_received", 1_100, vec![]);
    span.end(5_000);

    assert!(span.is_ended());
    assert_eq!(span.duration_ns(), Some(4_000));
    assert!(sampler.should_sample(None, &trace_id, "http.request")
        == SamplingDecision::RecordAndSample);
}

#[test]
fn test_parent_child_span_sharing_trace_id() {
    let mut rng = make_rng();
    let tid = TraceId::generate(&mut rng);
    let parent_ctx = SpanContext::new(tid, SpanId::generate(&mut rng), TraceFlags::SAMPLED);
    let child_ctx  = SpanContext::new(tid, SpanId::generate(&mut rng), TraceFlags::SAMPLED);

    let parent = Span::new(parent_ctx, None,           "parent", SpanKind::Server,   0);
    let child  = Span::new(child_ctx,  Some(parent_ctx), "child", SpanKind::Internal, 100);

    assert_eq!(parent.context.trace_id, child.context.trace_id);
    assert_eq!(child.parent_ctx.unwrap().span_id, parent.context.span_id);
}

#[test]
fn test_batch_processor_full_pipeline() {
    let mut proc = BatchSpanProcessor::new(50);
    let mut rng  = make_rng();
    let mut exp  = InMemorySpanExporter::new();

    for i in 0..10 {
        let ctx = SpanContext::new(TraceId::generate(&mut rng), SpanId::generate(&mut rng), TraceFlags::SAMPLED);
        let mut span = Span::new(ctx, None, &format!("op-{}", i), SpanKind::Internal, i * 100);
        span.end(i * 100 + 50);
        proc.on_end(span);
    }

    let batch = proc.flush(10);
    for span in batch { exp.export(span); }
    assert_eq!(exp.len(), 10);
}

// ── Metrics pipeline ──────────────────────────────────────────────────────────

#[test]
fn test_counter_increment_and_prometheus_output() {
    let mut c = Counter::new("http_requests_total", "1", "HTTP request count");
    let get_200 = vec![("method".into(), "GET".into()), ("status".into(), "200".into())];
    let get_404 = vec![("method".into(), "GET".into()), ("status".into(), "404".into())];
    c.add(100.0, &get_200);
    c.add(5.0,   &get_404);
    assert_eq!(c.get(&get_200), 100.0);
    assert_eq!(c.get(&get_404),   5.0);

    let prom = prometheus_counter(&c);
    assert!(prom.contains("# TYPE http_requests_total counter"));
    assert!(prom.contains("100"));
    assert!(prom.contains("5"));
}

#[test]
fn test_histogram_p99_latency() {
    let mut h = Histogram::new("rpc_latency_ms", "ms", "RPC latency",
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]);
    let attrs: Attributes = vec![("service".into(), "auth".into())];

    // Simulate 1000 requests: 900 fast (< 10ms), 99 medium, 1 slow
    for _ in 0..900 { h.record(5.0,    &attrs); }
    for _ in 0..99  { h.record(50.0,   &attrs); }
    for _ in 0..1   { h.record(500.0,  &attrs); }

    let data = h.data_for(&attrs).unwrap();
    assert_eq!(data.count, 1000);

    let p50 = data.quantile(0.50);
    let p99 = data.quantile(0.99);
    assert!(p50 <= 10.0,  "p50={} should be ≤10ms", p50);
    // p99 falls in the (25ms, 50ms] bucket via linear interpolation (~47.7ms)
    assert!(p99 >= 40.0,  "p99={} should be ≥40ms", p99);
}

// ── Logging pipeline ─────────────────────────────────────────────────────────

#[test]
fn test_log_pipeline_with_span_injection() {
    let mut rng = make_rng();
    let ctx = SpanContext::new(TraceId::generate(&mut rng), SpanId::generate(&mut rng), TraceFlags::SAMPLED);

    let record = LogRecord::new(1_700_000_000_000_000_000, Level::Error, "auth::service",
        "authentication failed")
        .with_attr("user_id", "42")
        .with_attr("ip",      "10.0.0.1")
        .inject_span_ctx(&ctx);

    let json = format_json(&record);
    assert!(json.contains("\"level\":\"ERROR\""));
    assert!(json.contains("trace_id"));
    assert!(json.contains("span_id"));
    assert!(json.contains("user_id"));

    let logfmt = format_logfmt(&record);
    assert!(logfmt.contains("level=ERROR"));
}

#[test]
fn test_memory_appender_retains_recent() {
    let filter = LogFilter::new(Level::Debug);
    let mut app = MemoryAppender::new(5);
    for i in 0..8 {
        let r = LogRecord::new(i, Level::Info, "test", &format!("msg {}", i));
        if filter.enabled(&r.target, r.level) {
            app.append(&r, &format_text(&r));
        }
    }
    assert_eq!(app.len(), 5);
    assert_eq!(app.dropped(), 3);
}

// ── W3C context propagation end-to-end ───────────────────────────────────────

#[test]
fn test_http_propagation_roundtrip() {
    let mut rng = make_rng();
    let span_ctx = SpanContext::new(TraceId::generate(&mut rng), SpanId::generate(&mut rng), TraceFlags::SAMPLED);
    let mut bag = Baggage::default();
    bag.set("request.id", "req-abc-123");
    bag.set("tenant",     "acme-corp");

    let outgoing_ctx = Context::default().with_span(span_ctx).with_baggage(bag);
    let mut headers = std::collections::HashMap::new();
    outgoing_ctx.inject(&mut headers);

    // Simulate crossing a service boundary
    let incoming_ctx = Context::extract(&headers);
    assert_eq!(incoming_ctx.span_context.unwrap().trace_id, span_ctx.trace_id);
    assert_eq!(incoming_ctx.baggage.get("tenant"), Some("acme-corp"));
}
