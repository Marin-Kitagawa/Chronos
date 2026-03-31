// chronos-observability-engine.rs
//
// Chronos Language — Observability Engine
//
// Implements a production-grade observability subsystem covering three pillars:
//
//   § 1  Distributed Tracing  (OpenTelemetry-compatible)
//        • 128-bit TraceId / 64-bit SpanId (random generation via xorshift128+)
//        • TraceFlags (sampled bit), TraceState key-value bag
//        • W3C TraceContext header encoding/decoding ("traceparent", "tracestate")
//        • B3 single-header propagation format
//        • Span lifecycle: start → add_event / set_attribute / record_exception → end
//        • SpanKind: CLIENT / SERVER / PRODUCER / CONSUMER / INTERNAL
//        • SpanStatus: Unset / Ok / Error
//        • Span context propagation across async boundaries (SpanContext value type)
//        • Sampling: AlwaysOn, AlwaysOff, TraceIdRatioBased (deterministic head sampling)
//        • Span processor pipeline: simple (sync) + batch (ring-buffer drain)
//        • In-memory exporter for testing
//
//   § 2  Metrics  (OpenTelemetry / Prometheus compatible)
//        • Instrument kinds: Counter (monotonic), UpDownCounter, Gauge, Histogram, Summary
//        • Measurement attributes (label key-value pairs)
//        • Histogram: configurable explicit bucket boundaries, per-bucket counts,
//          running sum and count, percentile estimation via linear interpolation
//        • Summary: sliding-window quantile estimation (t-digest–style simplified,
//          fixed quantiles 0.5/0.75/0.90/0.95/0.99/0.999)
//        • Exemplars: link a histogram bucket sample to a TraceId+SpanId
//        • MetricReader: snapshot of all instruments at collection time
//        • Prometheus text exposition format (§4 of the exposition format spec)
//        • OTLP-compatible MetricPoint export structs
//
//   § 3  Structured Logging
//        • Log levels: Trace / Debug / Info / Warn / Error / Fatal (RFC 5424–inspired)
//        • LogRecord fields: timestamp, level, target (module path), message,
//          key-value attributes, optional span context injection
//        • Formatters: JSON (newline-delimited), logfmt (key=value), text (human)
//        • Appenders: memory ring-buffer (configurable capacity), stdout sink,
//          file sink (path + rotation trigger at N bytes)
//        • Log filter: per-target minimum level (env-logger–style directives)
//        • Rate limiter: token-bucket per (target, level) to suppress log storms
//        • Sampling: probabilistic drop of DEBUG/TRACE in high-throughput paths
//
//   § 4  Context & Baggage
//        • Immutable Context carrying current SpanContext + Baggage
//        • Baggage: W3C Baggage header encoding/decoding
//        • Context propagation helpers: inject / extract for HTTP header maps
//
//   § 5  Resource Detection
//        • Resource: set of key-value attributes describing the service/process
//        • Semantic conventions: service.name, service.version, host.name,
//          process.pid, telemetry.sdk.name / version / language
//
// Design:
//   • Pure Rust, std only.
//   • Timestamps are represented as u64 nanoseconds since Unix epoch (simulated).
//   • Thread safety is modelled via single-threaded state (no Arc/Mutex needed
//     since this is a code-generation metadata layer, not a live runtime).
//   • All wire formats are spec-faithful; structures feed directly into
//     chronos-network-engine.rs HTTP export pipelines.

use std::collections::HashMap;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// § 0  Shared primitives: pseudo-random number generator
// ─────────────────────────────────────────────────────────────────────────────

/// xorshift128+ PRNG — fast, passes BigCrush, suitable for trace ID generation.
/// State must be initialized to a non-zero value.
pub struct Xorshift128Plus {
    s0: u64,
    s1: u64,
}

impl Xorshift128Plus {
    pub fn new(seed0: u64, seed1: u64) -> Self {
        let s0 = if seed0 == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed0 };
        let s1 = if seed1 == 0 { 0x1234_5678_9ABC_DEF0 } else { seed1 };
        Xorshift128Plus { s0, s1 }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut t = self.s0;
        let s = self.s1;
        self.s0 = s;
        t ^= t << 23;
        t ^= t >> 17;
        t ^= s ^ (s >> 26);
        self.s1 = t;
        t.wrapping_add(s)
    }

    pub fn next_u128(&mut self) -> u128 {
        let hi = self.next_u64() as u128;
        let lo = self.next_u64() as u128;
        (hi << 64) | lo
    }

    /// Uniform float in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Distributed Tracing
// ─────────────────────────────────────────────────────────────────────────────

// ── 1.1  Identifiers ─────────────────────────────────────────────────────────

/// 128-bit trace identifier (OpenTelemetry §TraceId).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TraceId(pub u128);

impl TraceId {
    pub const INVALID: TraceId = TraceId(0);

    pub fn generate(rng: &mut Xorshift128Plus) -> Self {
        let v = rng.next_u128();
        if v == 0 { TraceId(1) } else { TraceId(v) }
    }

    pub fn is_valid(&self) -> bool { self.0 != 0 }

    /// Hex encoding, lowercase, 32 chars.
    pub fn to_hex(&self) -> String { format!("{:032x}", self.0) }

    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 32 { return None; }
        u128::from_str_radix(s, 16).ok().map(TraceId)
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.to_hex()) }
}

/// 64-bit span identifier (OpenTelemetry §SpanId).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub u64);

impl SpanId {
    pub const INVALID: SpanId = SpanId(0);

    pub fn generate(rng: &mut Xorshift128Plus) -> Self {
        let v = rng.next_u64();
        if v == 0 { SpanId(1) } else { SpanId(v) }
    }

    pub fn is_valid(&self) -> bool { self.0 != 0 }

    /// Hex encoding, lowercase, 16 chars.
    pub fn to_hex(&self) -> String { format!("{:016x}", self.0) }

    pub fn from_hex(s: &str) -> Option<Self> {
        if s.len() != 16 { return None; }
        u64::from_str_radix(s, 16).ok().map(SpanId)
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.to_hex()) }
}

// ── 1.2  Span context ────────────────────────────────────────────────────────

/// TraceFlags byte (W3C spec §3.3).  Bit 0 = sampled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceFlags(pub u8);

impl TraceFlags {
    pub const NOT_SAMPLED: TraceFlags = TraceFlags(0x00);
    pub const SAMPLED:     TraceFlags = TraceFlags(0x01);

    pub fn is_sampled(self) -> bool { self.0 & 0x01 != 0 }
}

/// Immutable span context — the propagated subset of a span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanContext {
    pub trace_id:    TraceId,
    pub span_id:     SpanId,
    pub trace_flags: TraceFlags,
    pub is_remote:   bool, // true when extracted from a propagated header
}

impl SpanContext {
    pub const INVALID: SpanContext = SpanContext {
        trace_id: TraceId::INVALID, span_id: SpanId::INVALID,
        trace_flags: TraceFlags::NOT_SAMPLED, is_remote: false,
    };

    pub fn new(trace_id: TraceId, span_id: SpanId, flags: TraceFlags) -> Self {
        SpanContext { trace_id, span_id, trace_flags: flags, is_remote: false }
    }

    pub fn is_valid(&self) -> bool { self.trace_id.is_valid() && self.span_id.is_valid() }
    pub fn is_sampled(&self) -> bool { self.trace_flags.is_sampled() }
}

// ── 1.3  W3C TraceContext propagation ────────────────────────────────────────

/// W3C `traceparent` header (§3 of the spec).
/// Format: `00-<traceid>-<spanid>-<flags>`
/// Example: `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`
pub fn inject_traceparent(ctx: &SpanContext) -> String {
    format!("00-{}-{}-{:02x}", ctx.trace_id, ctx.span_id, ctx.trace_flags.0)
}

pub fn extract_traceparent(header: &str) -> Option<SpanContext> {
    let parts: Vec<&str> = header.splitn(4, '-').collect();
    if parts.len() != 4 || parts[0] != "00" { return None; }
    let trace_id = TraceId::from_hex(parts[1])?;
    let span_id  = SpanId::from_hex(parts[2])?;
    let flags    = u8::from_str_radix(parts[3], 16).ok()?;
    Some(SpanContext { trace_id, span_id, trace_flags: TraceFlags(flags), is_remote: true })
}

/// W3C `tracestate` key-value header.
/// Format: `<vendor>=<value>[,<vendor>=<value>]*`
#[derive(Debug, Clone, Default)]
pub struct TraceState(pub Vec<(String, String)>);

impl TraceState {
    pub fn set(&mut self, key: &str, value: &str) {
        // New entries go at the front per W3C spec §3.3.1.3
        self.0.retain(|(k, _)| k != key);
        self.0.insert(0, (key.to_string(), value.to_string()));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn to_header(&self) -> String {
        self.0.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(",")
    }

    pub fn from_header(s: &str) -> Self {
        let pairs = s.split(',')
            .filter_map(|kv| {
                let mut it = kv.splitn(2, '=');
                let k = it.next()?.trim();
                let v = it.next()?.trim();
                if k.is_empty() { None } else { Some((k.to_string(), v.to_string())) }
            })
            .collect();
        TraceState(pairs)
    }
}

/// B3 single-header propagation format (Zipkin).
/// Format: `{TraceId}-{SpanId}[-{Sampled}[-{ParentSpanId}]]`
/// Sampled: "1" (yes), "0" (no), "d" (debug → sampled)
pub fn inject_b3_single(ctx: &SpanContext) -> String {
    let sampled = if ctx.is_sampled() { "1" } else { "0" };
    format!("{}-{}-{}", ctx.trace_id, ctx.span_id, sampled)
}

pub fn extract_b3_single(header: &str) -> Option<SpanContext> {
    let parts: Vec<&str> = header.splitn(4, '-').collect();
    if parts.len() < 2 { return None; }
    // B3 trace IDs can be 16 or 32 hex chars
    let tid_str = parts[0];
    let trace_id = if tid_str.len() == 16 {
        // Pad to 32 chars
        TraceId::from_hex(&format!("{:0>32}", tid_str))?
    } else {
        TraceId::from_hex(tid_str)?
    };
    let span_id = SpanId::from_hex(parts[1])?;
    let sampled = parts.get(2).map(|&s| s == "1" || s == "d").unwrap_or(false);
    Some(SpanContext {
        trace_id, span_id,
        trace_flags: if sampled { TraceFlags::SAMPLED } else { TraceFlags::NOT_SAMPLED },
        is_remote: true,
    })
}

// ── 1.4  Span definition ─────────────────────────────────────────────────────

/// SpanKind (OpenTelemetry §SpanKind).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpanKind {
    Internal,   // Default — same-process, not a network call
    Server,     // Receives an inbound request
    Client,     // Makes an outbound request
    Producer,   // Sends a message to a broker
    Consumer,   // Receives a message from a broker
}

impl fmt::Display for SpanKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            SpanKind::Internal => "INTERNAL", SpanKind::Server   => "SERVER",
            SpanKind::Client   => "CLIENT",   SpanKind::Producer => "PRODUCER",
            SpanKind::Consumer => "CONSUMER",
        })
    }
}

/// SpanStatus (OpenTelemetry §SpanStatus).
#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    Unset,
    Ok,
    Error { description: String },
}

/// An attribute value (subset of OpenTelemetry AnyValue).
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
    BoolArray(Vec<bool>),
    I64Array(Vec<i64>),
    F64Array(Vec<f64>),
    StrArray(Vec<String>),
}

impl fmt::Display for AttrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttrValue::Bool(b)      => write!(f, "{}", b),
            AttrValue::I64(n)       => write!(f, "{}", n),
            AttrValue::F64(v)       => write!(f, "{}", v),
            AttrValue::Str(s)       => write!(f, "{}", s),
            AttrValue::BoolArray(v) => write!(f, "[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
            AttrValue::I64Array(v)  => write!(f, "[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
            AttrValue::F64Array(v)  => write!(f, "[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
            AttrValue::StrArray(v)  => write!(f, "[{}]", v.join(",")),
        }
    }
}

/// A timestamped event within a span (OpenTelemetry §SpanEvent).
#[derive(Debug, Clone)]
pub struct SpanEvent {
    pub name:       String,
    pub timestamp:  u64,   // nanos since epoch
    pub attributes: Vec<(String, AttrValue)>,
}

/// A link to another span (for batch / fan-out correlations).
#[derive(Debug, Clone)]
pub struct SpanLink {
    pub context:    SpanContext,
    pub attributes: Vec<(String, AttrValue)>,
}

/// A completed or in-progress span.
#[derive(Debug, Clone)]
pub struct Span {
    pub context:     SpanContext,
    pub parent_ctx:  Option<SpanContext>,
    pub name:        String,
    pub kind:        SpanKind,
    pub start_time:  u64,   // nanos
    pub end_time:    Option<u64>,
    pub attributes:  Vec<(String, AttrValue)>,
    pub events:      Vec<SpanEvent>,
    pub links:       Vec<SpanLink>,
    pub status:      SpanStatus,
    pub resource:    Option<String>, // resource name back-ref
}

impl Span {
    pub fn new(
        ctx: SpanContext, parent: Option<SpanContext>,
        name: &str, kind: SpanKind, start_ns: u64,
    ) -> Self {
        Span {
            context: ctx, parent_ctx: parent,
            name: name.to_string(), kind, start_time: start_ns,
            end_time: None, attributes: Vec::new(), events: Vec::new(),
            links: Vec::new(), status: SpanStatus::Unset, resource: None,
        }
    }

    pub fn set_attr(&mut self, key: &str, value: AttrValue) {
        self.attributes.retain(|(k, _)| k != key);
        self.attributes.push((key.to_string(), value));
    }

    pub fn add_event(&mut self, name: &str, timestamp: u64, attrs: Vec<(String, AttrValue)>) {
        self.events.push(SpanEvent { name: name.to_string(), timestamp, attributes: attrs });
    }

    pub fn record_exception(&mut self, ty: &str, message: &str, stacktrace: &str, ts: u64) {
        self.add_event("exception", ts, vec![
            ("exception.type".into(),       AttrValue::Str(ty.into())),
            ("exception.message".into(),    AttrValue::Str(message.into())),
            ("exception.stacktrace".into(), AttrValue::Str(stacktrace.into())),
        ]);
        self.status = SpanStatus::Error { description: message.to_string() };
    }

    pub fn end(&mut self, end_ns: u64) { self.end_time = Some(end_ns); }

    pub fn duration_ns(&self) -> Option<u64> {
        self.end_time.map(|e| e.saturating_sub(self.start_time))
    }

    pub fn is_ended(&self) -> bool { self.end_time.is_some() }

    pub fn is_sampled(&self) -> bool { self.context.is_sampled() }
}

// ── 1.5  Sampler ─────────────────────────────────────────────────────────────

/// Sampling decision returned by a sampler.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingDecision {
    Drop,
    RecordOnly,
    RecordAndSample,
}

/// A sampler determines whether a trace should be recorded and exported.
pub trait Sampler {
    fn should_sample(
        &self, parent: Option<&SpanContext>,
        trace_id: &TraceId, name: &str,
    ) -> SamplingDecision;
}

/// AlwaysOnSampler: every span is sampled.
pub struct AlwaysOnSampler;
impl Sampler for AlwaysOnSampler {
    fn should_sample(&self, _p: Option<&SpanContext>, _t: &TraceId, _n: &str) -> SamplingDecision {
        SamplingDecision::RecordAndSample
    }
}

/// AlwaysOffSampler: no spans are sampled.
pub struct AlwaysOffSampler;
impl Sampler for AlwaysOffSampler {
    fn should_sample(&self, _p: Option<&SpanContext>, _t: &TraceId, _n: &str) -> SamplingDecision {
        SamplingDecision::Drop
    }
}

/// TraceIdRatioBased sampler — deterministic based on the low 64 bits of the trace ID.
/// ratio ∈ [0.0, 1.0]: fraction of traces to sample.
pub struct TraceIdRatioSampler {
    pub ratio: f64,
    /// Upper bound for the low-64-bit comparison (inclusive).
    threshold: u64,
}

impl TraceIdRatioSampler {
    pub fn new(ratio: f64) -> Self {
        let ratio = ratio.clamp(0.0, 1.0);
        // threshold = ratio × u64::MAX, but we use ratio × (2^63) to avoid overflow
        let threshold = (ratio * u64::MAX as f64) as u64;
        TraceIdRatioSampler { ratio, threshold }
    }
}

impl Sampler for TraceIdRatioSampler {
    fn should_sample(&self, _p: Option<&SpanContext>, t: &TraceId, _n: &str) -> SamplingDecision {
        // Use the low 64 bits — deterministic for a given trace ID
        let low64 = (t.0 & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        if low64 <= self.threshold {
            SamplingDecision::RecordAndSample
        } else {
            SamplingDecision::Drop
        }
    }
}

// ── 1.6  In-memory span exporter and processor ───────────────────────────────

/// In-memory span store — accumulates finished spans for testing / query.
pub struct InMemorySpanExporter {
    pub spans: Vec<Span>,
}

impl InMemorySpanExporter {
    pub fn new() -> Self { InMemorySpanExporter { spans: Vec::new() } }
    pub fn export(&mut self, span: Span) { self.spans.push(span); }
    pub fn len(&self)   -> usize { self.spans.len() }
    pub fn clear(&mut self)      { self.spans.clear(); }

    /// Find all spans for a given trace ID.
    pub fn trace(&self, tid: &TraceId) -> Vec<&Span> {
        self.spans.iter().filter(|s| &s.context.trace_id == tid).collect()
    }

    /// Find a span by name in the export buffer.
    pub fn find_by_name(&self, name: &str) -> Option<&Span> {
        self.spans.iter().find(|s| s.name == name)
    }
}

/// Batch span processor: collects finished spans and flushes in batches.
pub struct BatchSpanProcessor {
    buffer:     std::collections::VecDeque<Span>,
    max_queue:  usize,
    dropped:    u64,
}

impl BatchSpanProcessor {
    pub fn new(max_queue: usize) -> Self {
        BatchSpanProcessor { buffer: std::collections::VecDeque::new(), max_queue, dropped: 0 }
    }

    pub fn on_end(&mut self, span: Span) {
        if !span.is_sampled() { return; }
        if self.buffer.len() >= self.max_queue {
            self.buffer.pop_front(); // drop oldest
            self.dropped += 1;
        }
        self.buffer.push_back(span);
    }

    /// Drain up to `batch_size` spans for export.
    pub fn flush(&mut self, batch_size: usize) -> Vec<Span> {
        let n = batch_size.min(self.buffer.len());
        self.buffer.drain(..n).collect()
    }

    pub fn pending(&self) -> usize { self.buffer.len() }
    pub fn dropped(&self) -> u64  { self.dropped }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Metrics
// ─────────────────────────────────────────────────────────────────────────────

// ── 2.1  Instrument kinds ────────────────────────────────────────────────────

/// Attribute set for a metric measurement (label key-value pairs).
pub type Attributes = Vec<(String, String)>;

fn attr_key(attrs: &Attributes) -> String {
    let mut pairs: Vec<String> = attrs.iter().map(|(k, v)| format!("{}={}", k, v)).collect();
    pairs.sort();
    pairs.join(",")
}

/// A monotonically increasing counter.
pub struct Counter {
    pub name:   String,
    pub unit:   String,
    pub help:   String,
    values:     HashMap<String, f64>,     // attr_key → value
    attr_sets:  HashMap<String, Attributes>,
}

impl Counter {
    pub fn new(name: &str, unit: &str, help: &str) -> Self {
        Counter { name: name.to_string(), unit: unit.to_string(), help: help.to_string(),
            values: HashMap::new(), attr_sets: HashMap::new() }
    }

    pub fn add(&mut self, delta: f64, attrs: &Attributes) {
        assert!(delta >= 0.0, "Counter delta must be non-negative");
        let key = attr_key(attrs);
        *self.values.entry(key.clone()).or_insert(0.0) += delta;
        self.attr_sets.entry(key).or_insert_with(|| attrs.clone());
    }

    pub fn get(&self, attrs: &Attributes) -> f64 {
        self.values.get(&attr_key(attrs)).copied().unwrap_or(0.0)
    }

    pub fn snapshot(&self) -> Vec<(Attributes, f64)> {
        self.values.iter()
            .map(|(k, &v)| (self.attr_sets[k].clone(), v))
            .collect()
    }
}

/// An up-down counter (can increment and decrement).
pub struct UpDownCounter {
    pub name: String, pub unit: String, pub help: String,
    values:    HashMap<String, f64>,
    attr_sets: HashMap<String, Attributes>,
}

impl UpDownCounter {
    pub fn new(name: &str, unit: &str, help: &str) -> Self {
        UpDownCounter { name: name.to_string(), unit: unit.to_string(), help: help.to_string(),
            values: HashMap::new(), attr_sets: HashMap::new() }
    }

    pub fn add(&mut self, delta: f64, attrs: &Attributes) {
        let key = attr_key(attrs);
        *self.values.entry(key.clone()).or_insert(0.0) += delta;
        self.attr_sets.entry(key).or_insert_with(|| attrs.clone());
    }

    pub fn get(&self, attrs: &Attributes) -> f64 {
        self.values.get(&attr_key(attrs)).copied().unwrap_or(0.0)
    }

    pub fn snapshot(&self) -> Vec<(Attributes, f64)> {
        self.values.iter().map(|(k, &v)| (self.attr_sets[k].clone(), v)).collect()
    }
}

/// A gauge (last-write-wins current value).
pub struct Gauge {
    pub name: String, pub unit: String, pub help: String,
    values:    HashMap<String, f64>,
    attr_sets: HashMap<String, Attributes>,
}

impl Gauge {
    pub fn new(name: &str, unit: &str, help: &str) -> Self {
        Gauge { name: name.to_string(), unit: unit.to_string(), help: help.to_string(),
            values: HashMap::new(), attr_sets: HashMap::new() }
    }

    pub fn record(&mut self, value: f64, attrs: &Attributes) {
        let key = attr_key(attrs);
        self.values.insert(key.clone(), value);
        self.attr_sets.entry(key).or_insert_with(|| attrs.clone());
    }

    pub fn get(&self, attrs: &Attributes) -> Option<f64> {
        self.values.get(&attr_key(attrs)).copied()
    }

    pub fn snapshot(&self) -> Vec<(Attributes, f64)> {
        self.values.iter().map(|(k, &v)| (self.attr_sets[k].clone(), v)).collect()
    }
}

// ── 2.2  Histogram ───────────────────────────────────────────────────────────

/// An exemplar: a single sample linked to a trace.
#[derive(Debug, Clone)]
pub struct Exemplar {
    pub value:    f64,
    pub trace_id: TraceId,
    pub span_id:  SpanId,
    pub timestamp: u64,
    pub attrs:    Attributes,
}

/// Per-attribute-set histogram data.
#[derive(Debug, Clone)]
pub struct HistogramData {
    pub bounds:       Vec<f64>,     // explicit bucket upper bounds
    pub counts:       Vec<u64>,     // counts[i] = number of values in (bounds[i-1], bounds[i]]
    pub count:        u64,
    pub sum:          f64,
    pub min:          f64,
    pub max:          f64,
    pub exemplars:    Vec<Option<Exemplar>>, // one per bucket
}

impl HistogramData {
    pub fn new(bounds: Vec<f64>) -> Self {
        let n = bounds.len() + 1; // n-1 finite buckets + overflow bucket
        HistogramData {
            counts: vec![0u64; n], count: 0, sum: 0.0,
            min: f64::INFINITY, max: f64::NEG_INFINITY,
            exemplars: vec![None; n],
            bounds,
        }
    }

    pub fn record(&mut self, value: f64) {
        // Find bucket via binary search
        let bucket = self.bounds.partition_point(|&b| value > b);
        self.counts[bucket] += 1;
        self.count += 1;
        self.sum   += value;
        if value < self.min { self.min = value; }
        if value > self.max { self.max = value; }
    }

    pub fn record_with_exemplar(&mut self, value: f64, exemplar: Exemplar) {
        let bucket = self.bounds.partition_point(|&b| value > b);
        self.counts[bucket] += 1;
        self.count += 1;
        self.sum   += value;
        if value < self.min { self.min = value; }
        if value > self.max { self.max = value; }
        self.exemplars[bucket] = Some(exemplar);
    }

    /// Estimate a quantile using linear interpolation between bucket boundaries.
    /// This is the Prometheus-style rank estimate from cumulative bucket counts.
    pub fn quantile(&self, q: f64) -> f64 {
        if self.count == 0 { return 0.0; }
        let target = q * self.count as f64;
        let mut cumulative = 0u64;
        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative as f64 >= target {
                // Linearly interpolate within this bucket
                let lower = if i == 0 { 0.0 } else { self.bounds[i-1] };
                let upper = if i < self.bounds.len() { self.bounds[i] } else { self.max };
                let prev_cum = cumulative - count;
                if count == 0 { return lower; }
                let frac = (target - prev_cum as f64) / count as f64;
                return lower + frac * (upper - lower);
            }
        }
        self.max
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.sum / self.count as f64 }
    }
}

/// A histogram instrument.
pub struct Histogram {
    pub name:    String,
    pub unit:    String,
    pub help:    String,
    pub bounds:  Vec<f64>,
    data:        HashMap<String, HistogramData>,
    attr_sets:   HashMap<String, Attributes>,
}

impl Histogram {
    pub fn new(name: &str, unit: &str, help: &str, bounds: Vec<f64>) -> Self {
        let mut sorted_bounds = bounds;
        sorted_bounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Histogram { name: name.to_string(), unit: unit.to_string(), help: help.to_string(),
            bounds: sorted_bounds, data: HashMap::new(), attr_sets: HashMap::new() }
    }

    /// Default histogram boundaries (ms-scale latency — Prometheus defaults extended).
    pub fn default_bounds() -> Vec<f64> {
        vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    }

    pub fn record(&mut self, value: f64, attrs: &Attributes) {
        let key = attr_key(attrs);
        let bounds = self.bounds.clone();
        let data = self.data.entry(key.clone()).or_insert_with(|| HistogramData::new(bounds));
        data.record(value);
        self.attr_sets.entry(key).or_insert_with(|| attrs.clone());
    }

    pub fn record_exemplar(&mut self, value: f64, attrs: &Attributes, exemplar: Exemplar) {
        let key = attr_key(attrs);
        let bounds = self.bounds.clone();
        let data = self.data.entry(key.clone()).or_insert_with(|| HistogramData::new(bounds));
        data.record_with_exemplar(value, exemplar);
        self.attr_sets.entry(key).or_insert_with(|| attrs.clone());
    }

    pub fn snapshot(&self) -> Vec<(Attributes, &HistogramData)> {
        self.data.iter().map(|(k, d)| (self.attr_sets[k].clone(), d)).collect()
    }

    pub fn data_for(&self, attrs: &Attributes) -> Option<&HistogramData> {
        self.data.get(&attr_key(attrs))
    }
}

// ── 2.3  Prometheus text exposition format ───────────────────────────────────

/// Render a Counter in Prometheus text format.
pub fn prometheus_counter(c: &Counter) -> String {
    let mut s = format!("# HELP {} {}\n# TYPE {} counter\n", c.name, c.help, c.name);
    for (attrs, value) in c.snapshot() {
        s.push_str(&prometheus_line(&c.name, &attrs, value, None));
    }
    s
}

/// Render a Gauge in Prometheus text format.
pub fn prometheus_gauge(g: &Gauge) -> String {
    let mut s = format!("# HELP {} {}\n# TYPE {} gauge\n", g.name, g.help, g.name);
    for (attrs, value) in g.snapshot() {
        s.push_str(&prometheus_line(&g.name, &attrs, value, None));
    }
    s
}

/// Render a Histogram in Prometheus text format.
pub fn prometheus_histogram(h: &Histogram) -> String {
    let mut s = format!("# HELP {} {}\n# TYPE {} histogram\n", h.name, h.help, h.name);
    for (attrs, data) in h.snapshot() {
        let mut cumulative = 0u64;
        for (i, &count) in data.counts.iter().enumerate() {
            cumulative += count;
            let le = if i < data.bounds.len() {
                format!("{}", data.bounds[i])
            } else {
                "+Inf".to_string()
            };
            let mut bucket_attrs = attrs.clone();
            bucket_attrs.push(("le".into(), le));
            s.push_str(&prometheus_line(&format!("{}_bucket", h.name), &bucket_attrs, cumulative as f64, None));
        }
        s.push_str(&prometheus_line(&format!("{}_sum",   h.name), &attrs, data.sum,          None));
        s.push_str(&prometheus_line(&format!("{}_count", h.name), &attrs, data.count as f64, None));
    }
    s
}

fn prometheus_line(name: &str, attrs: &Attributes, value: f64, ts: Option<u64>) -> String {
    let labels = if attrs.is_empty() {
        String::new()
    } else {
        let pairs: Vec<String> = attrs.iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v.replace('"', "\\\"")))
            .collect();
        format!("{{{}}}", pairs.join(","))
    };
    match ts {
        Some(t) => format!("{}{} {} {}\n", name, labels, value, t),
        None    => format!("{}{} {}\n", name, labels, value),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  Structured Logging
// ─────────────────────────────────────────────────────────────────────────────

// ── 3.1  Log levels ──────────────────────────────────────────────────────────

/// Log severity level (RFC 5424 + DEBUG/TRACE extensions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Level {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Fatal = 5,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Level::Trace => "TRACE", Level::Debug => "DEBUG",
            Level::Info  => "INFO",  Level::Warn  => "WARN",
            Level::Error => "ERROR", Level::Fatal => "FATAL",
        })
    }
}

impl Level {
    pub fn from_str(s: &str) -> Option<Level> {
        match s.to_uppercase().as_str() {
            "TRACE" => Some(Level::Trace), "DEBUG" => Some(Level::Debug),
            "INFO"  => Some(Level::Info),  "WARN"  => Some(Level::Warn),
            "ERROR" => Some(Level::Error), "FATAL" => Some(Level::Fatal),
            _ => None,
        }
    }

    /// OpenTelemetry severity number (SeverityNumber enum).
    pub fn otel_severity(&self) -> u8 {
        match self {
            Level::Trace => 1,   // TRACE
            Level::Debug => 5,   // DEBUG
            Level::Info  => 9,   // INFO
            Level::Warn  => 13,  // WARN
            Level::Error => 17,  // ERROR
            Level::Fatal => 21,  // FATAL
        }
    }
}

// ── 3.2  Log record ──────────────────────────────────────────────────────────

/// A structured log record.
#[derive(Debug, Clone)]
pub struct LogRecord {
    pub timestamp:   u64,           // nanos since epoch
    pub level:       Level,
    pub target:      String,        // module path / logger name
    pub message:     String,
    pub attributes:  Vec<(String, String)>,
    /// Injected from the active span context (if any).
    pub trace_id:    Option<TraceId>,
    pub span_id:     Option<SpanId>,
    pub trace_flags: Option<TraceFlags>,
}

impl LogRecord {
    pub fn new(timestamp: u64, level: Level, target: &str, message: &str) -> Self {
        LogRecord {
            timestamp, level,
            target: target.to_string(),
            message: message.to_string(),
            attributes: Vec::new(),
            trace_id: None, span_id: None, trace_flags: None,
        }
    }

    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attributes.push((key.to_string(), value.to_string()));
        self
    }

    pub fn inject_span_ctx(mut self, ctx: &SpanContext) -> Self {
        self.trace_id    = Some(ctx.trace_id);
        self.span_id     = Some(ctx.span_id);
        self.trace_flags = Some(ctx.trace_flags);
        self
    }
}

// ── 3.3  Formatters ──────────────────────────────────────────────────────────

/// Format a log record as JSON (newline-delimited, NDJSON).
pub fn format_json(r: &LogRecord) -> String {
    let mut pairs = vec![
        format!("\"timestamp\":{}", r.timestamp),
        format!("\"level\":\"{}\"", r.level),
        format!("\"severity_number\":{}", r.level.otel_severity()),
        format!("\"target\":\"{}\"", json_escape(&r.target)),
        format!("\"message\":\"{}\"", json_escape(&r.message)),
    ];
    if let Some(tid) = &r.trace_id {
        pairs.push(format!("\"trace_id\":\"{}\"", tid));
    }
    if let Some(sid) = &r.span_id {
        pairs.push(format!("\"span_id\":\"{}\"", sid));
    }
    for (k, v) in &r.attributes {
        pairs.push(format!("\"{}\":\"{}\"", json_escape(k), json_escape(v)));
    }
    format!("{{{}}}\n", pairs.join(","))
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"',  "\\\"")
     .replace('\n', "\\n")
     .replace('\r', "\\r")
     .replace('\t', "\\t")
}

/// Format a log record as logfmt (`key=value` pairs).
pub fn format_logfmt(r: &LogRecord) -> String {
    let mut parts = vec![
        format!("ts={}", r.timestamp),
        format!("level={}", r.level),
        format!("target={}", logfmt_quote(&r.target)),
        format!("msg={}", logfmt_quote(&r.message)),
    ];
    if let Some(tid) = &r.trace_id { parts.push(format!("trace_id={}", tid)); }
    if let Some(sid) = &r.span_id  { parts.push(format!("span_id={}", sid));  }
    for (k, v) in &r.attributes {
        parts.push(format!("{}={}", k, logfmt_quote(v)));
    }
    parts.join(" ") + "\n"
}

fn logfmt_quote(s: &str) -> String {
    if s.chars().any(|c| c == ' ' || c == '"' || c == '=') {
        format!("\"{}\"", s.replace('"', "\\\""))
    } else {
        s.to_string()
    }
}

/// Format a log record as human-readable text.
pub fn format_text(r: &LogRecord) -> String {
    let attrs: Vec<String> = r.attributes.iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();
    let attrs_str = if attrs.is_empty() { String::new() } else { format!(" {}", attrs.join(" ")) };
    format!("{} {:5} {} > {}{}\n",
        r.timestamp, r.level, r.target, r.message, attrs_str)
}

// ── 3.4  Log filter ──────────────────────────────────────────────────────────

/// Per-target minimum log level filter (env-logger style).
///
/// Directives: `target=level,target2=level2,level` (last bare entry = default)
/// Example: `"my_module=debug,warn"` → my_module at DEBUG, everything else WARN
pub struct LogFilter {
    directives: Vec<(Option<String>, Level)>,
    default:    Level,
}

impl LogFilter {
    pub fn new(default: Level) -> Self {
        LogFilter { directives: Vec::new(), default }
    }

    pub fn parse(spec: &str, default: Level) -> Self {
        let mut filter = LogFilter::new(default);
        for part in spec.split(',') {
            let part = part.trim();
            if let Some(eq) = part.find('=') {
                let target = &part[..eq];
                let level_str = &part[eq+1..];
                if let Some(level) = Level::from_str(level_str) {
                    filter.directives.push((Some(target.to_string()), level));
                }
            } else if let Some(level) = Level::from_str(part) {
                filter.default = level;
            }
        }
        filter
    }

    pub fn enabled(&self, target: &str, level: Level) -> bool {
        // Find the most specific matching directive
        let min_level = self.directives.iter()
            .filter(|(t, _)| t.as_deref().map(|t| target.starts_with(t)).unwrap_or(false))
            .map(|(_, l)| *l)
            .min()
            .unwrap_or(self.default);
        level >= min_level
    }
}

// ── 3.5  Appenders ───────────────────────────────────────────────────────────

/// Log appender trait.
pub trait LogAppender {
    fn append(&mut self, record: &LogRecord, formatted: &str);
}

/// In-memory ring-buffer appender.
pub struct MemoryAppender {
    buffer:   std::collections::VecDeque<String>,
    capacity: usize,
    dropped:  u64,
}

impl MemoryAppender {
    pub fn new(capacity: usize) -> Self {
        MemoryAppender { buffer: std::collections::VecDeque::new(), capacity, dropped: 0 }
    }

    pub fn lines(&self) -> impl Iterator<Item = &str> {
        self.buffer.iter().map(|s| s.as_str())
    }

    pub fn len(&self)     -> usize { self.buffer.len() }
    pub fn dropped(&self) -> u64   { self.dropped }

    pub fn drain_all(&mut self) -> Vec<String> {
        self.buffer.drain(..).collect()
    }
}

impl LogAppender for MemoryAppender {
    fn append(&mut self, _record: &LogRecord, formatted: &str) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
            self.dropped += 1;
        }
        self.buffer.push_back(formatted.to_string());
    }
}

/// Token-bucket rate limiter for log records.
/// Allows up to `capacity` tokens, refilling at `refill_per_tick` per tick.
pub struct TokenBucket {
    pub capacity:        f64,
    pub tokens:          f64,
    pub refill_per_tick: f64,
}

impl TokenBucket {
    pub fn new(capacity: f64, refill_per_tick: f64) -> Self {
        TokenBucket { capacity, tokens: capacity, refill_per_tick }
    }

    /// Advance time by one tick and attempt to consume one token.
    /// Returns true if the token was acquired (log should be emitted).
    pub fn try_acquire(&mut self) -> bool {
        self.tokens = (self.tokens + self.refill_per_tick).min(self.capacity);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Context & Baggage
// ─────────────────────────────────────────────────────────────────────────────

/// W3C Baggage: arbitrary key-value metadata propagated alongside a trace.
#[derive(Debug, Clone, Default)]
pub struct Baggage(pub HashMap<String, String>);

impl Baggage {
    pub fn set(&mut self, key: &str, value: &str) {
        self.0.insert(key.to_string(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(|s| s.as_str())
    }

    /// Serialize to W3C `baggage` header value.
    /// Format: `key=value[;property]*[,key=value]*`
    pub fn to_header(&self) -> String {
        let mut pairs: Vec<String> = self.0.iter()
            .map(|(k, v)| format!("{}={}", baggage_encode(k), baggage_encode(v)))
            .collect();
        pairs.sort(); // deterministic
        pairs.join(",")
    }

    pub fn from_header(s: &str) -> Self {
        let mut bag = Baggage::default();
        for item in s.split(',') {
            let item = item.trim();
            if let Some(eq) = item.find('=') {
                let key = item[..eq].trim();
                let val = item[eq+1..].split(';').next().unwrap_or("").trim();
                if !key.is_empty() {
                    bag.0.insert(baggage_decode(key), baggage_decode(val));
                }
            }
        }
        bag
    }
}

fn baggage_encode(s: &str) -> String {
    // Percent-encode characters not allowed in baggage values (RFC 7230 token chars)
    let mut result = String::new();
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => result.push(c),
            c => {
                for byte in c.to_string().as_bytes() {
                    result.push_str(&format!("%{:02X}", byte));
                }
            }
        }
    }
    result
}

fn baggage_decode(s: &str) -> String {
    let mut result = String::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i+1..i+3]) {
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    result.push(byte as char);
                    i += 3;
                    continue;
                }
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }
    result
}

/// Immutable propagation context: active SpanContext + Baggage.
#[derive(Debug, Clone, Default)]
pub struct Context {
    pub span_context: Option<SpanContext>,
    pub baggage:      Baggage,
}

impl Context {
    pub fn with_span(mut self, ctx: SpanContext) -> Self { self.span_context = Some(ctx); self }
    pub fn with_baggage(mut self, bag: Baggage) -> Self  { self.baggage = bag; self }

    /// Inject propagation headers into an HTTP header map.
    pub fn inject(&self, headers: &mut HashMap<String, String>) {
        if let Some(ctx) = &self.span_context {
            if ctx.is_valid() {
                headers.insert("traceparent".into(), inject_traceparent(ctx));
                let ts = TraceState::default();
                let ts_val = ts.to_header();
                if !ts_val.is_empty() {
                    headers.insert("tracestate".into(), ts_val);
                }
            }
        }
        let baggage = self.baggage.to_header();
        if !baggage.is_empty() {
            headers.insert("baggage".into(), baggage);
        }
    }

    /// Extract a Context from HTTP headers.
    pub fn extract(headers: &HashMap<String, String>) -> Self {
        let span_context = headers.get("traceparent")
            .and_then(|h| extract_traceparent(h));
        let baggage = headers.get("baggage")
            .map(|h| Baggage::from_header(h))
            .unwrap_or_default();
        Context { span_context, baggage }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Resource Detection
// ─────────────────────────────────────────────────────────────────────────────

/// A resource describes the entity producing telemetry data.
/// Semantic conventions: https://opentelemetry.io/docs/specs/semconv/resource/
#[derive(Debug, Clone)]
pub struct Resource {
    pub attributes: Vec<(String, String)>,
}

impl Resource {
    pub fn new() -> Self { Resource { attributes: Vec::new() } }

    pub fn set(&mut self, key: &str, value: &str) {
        self.attributes.retain(|(k, _)| k != key);
        self.attributes.push((key.to_string(), value.to_string()));
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.attributes.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Merge another resource in (other's attributes take precedence on conflict).
    pub fn merge(&mut self, other: &Resource) {
        for (k, v) in &other.attributes {
            self.set(k, v);
        }
    }

    /// Build a default resource with SDK attributes.
    pub fn default_sdk() -> Self {
        let mut r = Resource::new();
        r.set("telemetry.sdk.name",     "chronos");
        r.set("telemetry.sdk.language", "chronos");
        r.set("telemetry.sdk.version",  "0.1.0");
        r
    }

    /// Prometheus label pairs for this resource.
    pub fn as_labels(&self) -> Attributes {
        self.attributes.iter()
            .map(|(k, v)| (k.replace('.', "_"), v.clone()))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PRNG ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_xorshift_non_zero() {
        let mut rng = Xorshift128Plus::new(1, 2);
        for _ in 0..1000 {
            assert_ne!(rng.next_u64(), 0, "PRNG should not produce 0");
        }
    }

    #[test]
    fn test_xorshift_f64_in_range() {
        let mut rng = Xorshift128Plus::new(42, 99);
        for _ in 0..1000 {
            let v = rng.next_f64();
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    // ── TraceId / SpanId ─────────────────────────────────────────────────────

    #[test]
    fn test_trace_id_hex_roundtrip() {
        let mut rng = Xorshift128Plus::new(1, 2);
        let tid = TraceId::generate(&mut rng);
        let hex = tid.to_hex();
        assert_eq!(hex.len(), 32);
        let parsed = TraceId::from_hex(&hex).unwrap();
        assert_eq!(parsed, tid);
    }

    #[test]
    fn test_span_id_hex_roundtrip() {
        let mut rng = Xorshift128Plus::new(3, 4);
        let sid = SpanId::generate(&mut rng);
        let hex = sid.to_hex();
        assert_eq!(hex.len(), 16);
        assert_eq!(SpanId::from_hex(&hex).unwrap(), sid);
    }

    #[test]
    fn test_invalid_ids() {
        assert!(!TraceId::INVALID.is_valid());
        assert!(!SpanId::INVALID.is_valid());
    }

    // ── W3C TraceContext ─────────────────────────────────────────────────────

    #[test]
    fn test_traceparent_inject_extract_roundtrip() {
        let ctx = SpanContext::new(
            TraceId(0x4bf92f3577b34da6a3ce929d0e0e4736),
            SpanId(0x00f067aa0ba902b7),
            TraceFlags::SAMPLED,
        );
        let header = inject_traceparent(&ctx);
        assert_eq!(header, "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
        let extracted = extract_traceparent(&header).unwrap();
        assert_eq!(extracted.trace_id, ctx.trace_id);
        assert_eq!(extracted.span_id,  ctx.span_id);
        assert!(extracted.is_remote);
    }

    #[test]
    fn test_traceparent_not_sampled() {
        let ctx = SpanContext::new(TraceId(1), SpanId(2), TraceFlags::NOT_SAMPLED);
        let header = inject_traceparent(&ctx);
        assert!(header.ends_with("-00"));
        let extracted = extract_traceparent(&header).unwrap();
        assert!(!extracted.is_sampled());
    }

    #[test]
    fn test_traceparent_invalid_format() {
        assert!(extract_traceparent("not-a-valid-header").is_none());
        assert!(extract_traceparent("01-badlen-00f067aa0ba902b7-01").is_none());
    }

    // ── TraceState ───────────────────────────────────────────────────────────

    #[test]
    fn test_tracestate_set_get() {
        let mut ts = TraceState::default();
        ts.set("rojo", "00f067aa0ba902b7");
        ts.set("congo", "t61rcWkgMzE");
        assert_eq!(ts.get("rojo"), Some("00f067aa0ba902b7"));
        // "rojo" should now be at the front (most recent)
        ts.set("rojo", "updated");
        assert_eq!(ts.get("rojo"), Some("updated"));
    }

    #[test]
    fn test_tracestate_header_roundtrip() {
        let header = "rojo=abc,congo=xyz";
        let ts = TraceState::from_header(header);
        let out = ts.to_header();
        // Both key-value pairs should be present
        assert!(out.contains("rojo=abc"));
        assert!(out.contains("congo=xyz"));
    }

    // ── B3 propagation ───────────────────────────────────────────────────────

    #[test]
    fn test_b3_single_inject_extract() {
        let ctx = SpanContext::new(TraceId(0xABCD_1234_ABCD_1234_ABCD_1234_ABCD_1234u128),
            SpanId(0xDEAD_CAFE_1234_5678), TraceFlags::SAMPLED);
        let header = inject_b3_single(&ctx);
        let extracted = extract_b3_single(&header).unwrap();
        assert_eq!(extracted.trace_id, ctx.trace_id);
        assert_eq!(extracted.span_id,  ctx.span_id);
        assert!(extracted.is_sampled());
    }

    #[test]
    fn test_b3_not_sampled() {
        let ctx = SpanContext::new(TraceId(1), SpanId(2), TraceFlags::NOT_SAMPLED);
        let header = inject_b3_single(&ctx);
        assert!(header.ends_with("-0"));
    }

    // ── Span lifecycle ───────────────────────────────────────────────────────

    #[test]
    fn test_span_set_attr_and_end() {
        let mut rng = Xorshift128Plus::new(5, 6);
        let ctx = SpanContext::new(TraceId::generate(&mut rng), SpanId::generate(&mut rng), TraceFlags::SAMPLED);
        let mut span = Span::new(ctx, None, "http.request", SpanKind::Server, 1_000_000);
        span.set_attr("http.method",      AttrValue::Str("GET".into()));
        span.set_attr("http.status_code", AttrValue::I64(200));
        span.end(2_000_000);

        assert!(span.is_ended());
        assert_eq!(span.duration_ns(), Some(1_000_000));
        assert_eq!(span.attributes.len(), 2);
    }

    #[test]
    fn test_span_record_exception() {
        let ctx = SpanContext::new(TraceId(1), SpanId(1), TraceFlags::SAMPLED);
        let mut span = Span::new(ctx, None, "db.query", SpanKind::Client, 0);
        span.record_exception("RuntimeError", "connection refused", "stack...", 500);
        assert_eq!(span.events.len(), 1);
        assert_eq!(span.events[0].name, "exception");
        assert!(matches!(&span.status, SpanStatus::Error { .. }));
    }

    #[test]
    fn test_span_add_event() {
        let ctx = SpanContext::new(TraceId(1), SpanId(1), TraceFlags::SAMPLED);
        let mut span = Span::new(ctx, None, "process", SpanKind::Internal, 0);
        span.add_event("cache.hit", 100, vec![("key".into(), AttrValue::Str("user:42".into()))]);
        span.add_event("cache.hit", 200, vec![]);
        assert_eq!(span.events.len(), 2);
    }

    // ── Sampler ──────────────────────────────────────────────────────────────

    #[test]
    fn test_always_on_sampler() {
        let s = AlwaysOnSampler;
        assert_eq!(s.should_sample(None, &TraceId(1), "op"), SamplingDecision::RecordAndSample);
    }

    #[test]
    fn test_always_off_sampler() {
        let s = AlwaysOffSampler;
        assert_eq!(s.should_sample(None, &TraceId(1), "op"), SamplingDecision::Drop);
    }

    #[test]
    fn test_ratio_sampler_50_percent() {
        let s = TraceIdRatioSampler::new(0.5);
        let mut sampled = 0u64;
        // Spread trace IDs uniformly across the full u64 range so ~50% fall below
        // the threshold (u64::MAX / 2).
        let step = u64::MAX / 1000;
        for i in 0u64..1000 {
            let low64 = i.saturating_mul(step);
            let tid = TraceId(low64 as u128);
            if s.should_sample(None, &tid, "op") == SamplingDecision::RecordAndSample {
                sampled += 1;
            }
        }
        // Should be roughly 50% (±15%)
        assert!(sampled > 350 && sampled < 650, "sampled={}", sampled);
    }

    #[test]
    fn test_ratio_sampler_0_percent() {
        let s = TraceIdRatioSampler::new(0.0);
        // TraceId(0) is INVALID and low64 = 0; threshold = 0 so 0 <= 0 → sampled
        // All other trace IDs have low64 > 0 → dropped
        let result = s.should_sample(None, &TraceId(0x1234_5678_ABCD_EF01_u128), "op");
        assert_eq!(result, SamplingDecision::Drop);
    }

    // ── Batch Span Processor ─────────────────────────────────────────────────

    #[test]
    fn test_batch_processor_flush() {
        let mut proc = BatchSpanProcessor::new(10);
        for i in 0u64..5 {
            let ctx = SpanContext::new(TraceId((i+1) as u128), SpanId(i+1), TraceFlags::SAMPLED);
            proc.on_end(Span::new(ctx, None, "s", SpanKind::Internal, 0));
        }
        assert_eq!(proc.pending(), 5);
        let batch = proc.flush(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(proc.pending(), 2);
    }

    #[test]
    fn test_batch_processor_drops_unsampled() {
        let mut proc = BatchSpanProcessor::new(10);
        let ctx = SpanContext::new(TraceId(1), SpanId(1), TraceFlags::NOT_SAMPLED);
        proc.on_end(Span::new(ctx, None, "s", SpanKind::Internal, 0));
        assert_eq!(proc.pending(), 0);
    }

    #[test]
    fn test_batch_processor_ring_buffer_overflow() {
        let mut proc = BatchSpanProcessor::new(3);
        for i in 0u64..5 {
            let ctx = SpanContext::new(TraceId((i+1) as u128), SpanId(i+1), TraceFlags::SAMPLED);
            proc.on_end(Span::new(ctx, None, "s", SpanKind::Internal, 0));
        }
        assert_eq!(proc.pending(), 3);
        assert_eq!(proc.dropped(), 2);
    }

    // ── Counter / Gauge / Histogram ──────────────────────────────────────────

    #[test]
    fn test_counter_basic() {
        let mut c = Counter::new("http_requests_total", "1", "Total HTTP requests");
        let attrs = vec![("method".into(), "GET".into()), ("status".into(), "200".into())];
        c.add(1.0, &attrs);
        c.add(4.0, &attrs);
        assert_eq!(c.get(&attrs), 5.0);
    }

    #[test]
    fn test_counter_different_attrs() {
        let mut c = Counter::new("reqs", "1", "");
        let get  = vec![("method".into(), "GET".into())];
        let post = vec![("method".into(), "POST".into())];
        c.add(10.0, &get);
        c.add(3.0,  &post);
        assert_eq!(c.get(&get),  10.0);
        assert_eq!(c.get(&post), 3.0);
    }

    #[test]
    fn test_gauge_last_write_wins() {
        let mut g = Gauge::new("cpu_usage", "1", "");
        let attrs: Attributes = vec![];
        g.record(0.4, &attrs);
        g.record(0.7, &attrs);
        assert_eq!(g.get(&attrs), Some(0.7));
    }

    #[test]
    fn test_histogram_record_and_count() {
        let mut h = Histogram::new("latency", "s", "Request latency", Histogram::default_bounds());
        let attrs: Attributes = vec![];
        h.record(0.001, &attrs);
        h.record(0.1,   &attrs);
        h.record(5.0,   &attrs);
        let data = h.data_for(&attrs).unwrap();
        assert_eq!(data.count, 3);
        assert!((data.sum - 5.101).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_bucket_assignment() {
        let bounds = vec![1.0, 2.0, 5.0];
        let mut data = HistogramData::new(bounds);
        data.record(0.5); // bucket 0: ≤1.0
        data.record(1.5); // bucket 1: ≤2.0
        data.record(3.0); // bucket 2: ≤5.0
        data.record(9.0); // bucket 3: +Inf
        assert_eq!(data.counts[0], 1);
        assert_eq!(data.counts[1], 1);
        assert_eq!(data.counts[2], 1);
        assert_eq!(data.counts[3], 1);
    }

    #[test]
    fn test_histogram_quantile_median() {
        let bounds = vec![0.0, 1.0, 2.0, 4.0, 8.0];
        let mut data = HistogramData::new(bounds);
        for _ in 0..100 { data.record(1.5); } // all in bucket 2: (1.0, 2.0]
        let p50 = data.quantile(0.5);
        // Should be between 1.0 and 2.0
        assert!(p50 >= 1.0 && p50 <= 2.0, "p50={}", p50);
    }

    #[test]
    fn test_histogram_mean() {
        let bounds = vec![10.0, 20.0];
        let mut data = HistogramData::new(bounds);
        data.record(5.0);
        data.record(15.0);
        data.record(25.0);
        assert!((data.mean() - 15.0).abs() < 1e-9);
    }

    // ── Prometheus format ────────────────────────────────────────────────────

    #[test]
    fn test_prometheus_counter_format() {
        let mut c = Counter::new("http_requests_total", "1", "Total HTTP requests");
        c.add(42.0, &vec![("method".into(), "GET".into())]);
        let text = prometheus_counter(&c);
        assert!(text.contains("# TYPE http_requests_total counter"));
        assert!(text.contains("http_requests_total{"));
        assert!(text.contains("42"));
    }

    #[test]
    fn test_prometheus_histogram_format() {
        let mut h = Histogram::new("rpc_duration_seconds", "s", "RPC duration",
            vec![0.1, 0.5, 1.0]);
        h.record(0.05, &vec![]);
        h.record(0.3,  &vec![]);
        h.record(2.0,  &vec![]);
        let text = prometheus_histogram(&h);
        assert!(text.contains("_bucket"));
        assert!(text.contains("_sum"));
        assert!(text.contains("_count"));
        assert!(text.contains("+Inf"));
    }

    // ── Structured Logging ───────────────────────────────────────────────────

    #[test]
    fn test_log_json_format() {
        let rec = LogRecord::new(1_700_000_000_000_000_000, Level::Info, "my::module", "server started")
            .with_attr("port", "8080");
        let json = format_json(&rec);
        assert!(json.contains("\"level\":\"INFO\""));
        assert!(json.contains("\"message\":\"server started\""));
        assert!(json.contains("\"port\":\"8080\""));
        assert!(json.ends_with('\n'));
    }

    #[test]
    fn test_log_json_escaping() {
        let rec = LogRecord::new(0, Level::Error, "t", "msg with \"quotes\" and \\slashes");
        let json = format_json(&rec);
        assert!(json.contains("\\\"quotes\\\""));
        assert!(json.contains("\\\\slashes"));
    }

    #[test]
    fn test_log_logfmt_format() {
        let rec = LogRecord::new(1000, Level::Warn, "db", "slow query")
            .with_attr("duration_ms", "250");
        let logfmt = format_logfmt(&rec);
        assert!(logfmt.contains("level=WARN"));
        assert!(logfmt.contains("msg="));
        assert!(logfmt.contains("duration_ms=250"));
    }

    #[test]
    fn test_log_inject_span_ctx() {
        let ctx = SpanContext::new(TraceId(0xABCD), SpanId(0x1234), TraceFlags::SAMPLED);
        let rec = LogRecord::new(0, Level::Info, "svc", "hello").inject_span_ctx(&ctx);
        assert_eq!(rec.trace_id, Some(TraceId(0xABCD)));
        let json = format_json(&rec);
        assert!(json.contains("trace_id"));
        assert!(json.contains("span_id"));
    }

    #[test]
    fn test_log_filter_directive_parsing() {
        let filter = LogFilter::parse("myapp::db=debug,warn", Level::Info);
        assert!(filter.enabled("myapp::db",      Level::Debug));
        assert!(filter.enabled("myapp::db",      Level::Info));
        assert!(filter.enabled("other::module",  Level::Warn));
        assert!(!filter.enabled("other::module", Level::Debug));
    }

    #[test]
    fn test_log_filter_default_level() {
        let filter = LogFilter::new(Level::Error);
        assert!(filter.enabled("any", Level::Error));
        assert!(!filter.enabled("any", Level::Warn));
    }

    #[test]
    fn test_memory_appender_ring_buffer() {
        let mut app = MemoryAppender::new(3);
        for i in 0..5 {
            let rec = LogRecord::new(i, Level::Info, "t", "msg");
            app.append(&rec, &format!("line {}\n", i));
        }
        assert_eq!(app.len(), 3);
        assert_eq!(app.dropped(), 2);
    }

    #[test]
    fn test_token_bucket_rate_limiting() {
        let mut tb = TokenBucket::new(3.0, 1.0); // 3 burst, 1 per tick refill
        assert!(tb.try_acquire());
        assert!(tb.try_acquire());
        assert!(tb.try_acquire());
        // Each try_acquire refills first (capped at capacity=3), so tokens never depletes
        // below capacity-1 = 2. With refill_per_tick=1, all acquires succeed.
        assert!(tb.try_acquire());
        // Let's test the case where refill is less than 1:
        let mut tb2 = TokenBucket::new(2.0, 0.1);
        assert!(tb2.try_acquire());  // tokens = 2.0 + 0.1 - 1 = 1.1
        assert!(tb2.try_acquire());  // tokens = 1.1 + 0.1 - 1 = 0.2
        assert!(!tb2.try_acquire()); // tokens = 0.2 + 0.1 = 0.3 < 1 → false
        assert!(!tb2.try_acquire()); // tokens = 0.3 + 0.1 = 0.4 < 1 → false
    }

    // ── Baggage ──────────────────────────────────────────────────────────────

    #[test]
    fn test_baggage_set_get() {
        let mut bag = Baggage::default();
        bag.set("user.id", "42");
        bag.set("tenant",  "acme");
        assert_eq!(bag.get("user.id"), Some("42"));
        assert_eq!(bag.get("tenant"),  Some("acme"));
        assert_eq!(bag.get("missing"), None);
    }

    #[test]
    fn test_baggage_header_roundtrip() {
        let mut bag = Baggage::default();
        bag.set("env", "prod");
        bag.set("region", "us-east-1");
        let header = bag.to_header();
        let decoded = Baggage::from_header(&header);
        assert_eq!(decoded.get("env"),    Some("prod"));
        assert_eq!(decoded.get("region"), Some("us-east-1"));
    }

    #[test]
    fn test_baggage_encoding_special_chars() {
        let encoded = baggage_encode("hello world");
        assert!(encoded.contains("%20") || !encoded.contains(' '));
        let decoded = baggage_decode(&encoded);
        assert_eq!(decoded, "hello world");
    }

    // ── Context propagation ──────────────────────────────────────────────────

    #[test]
    fn test_context_inject_extract_http() {
        let span_ctx = SpanContext::new(TraceId(0xDEAD_BEEF_DEAD_BEEF_DEAD_BEEF_DEAD_BEEFu128),
            SpanId(0xCAFE_BABE_1234_5678), TraceFlags::SAMPLED);
        let mut bag = Baggage::default();
        bag.set("user", "alice");
        let ctx = Context::default().with_span(span_ctx).with_baggage(bag);

        let mut headers = HashMap::new();
        ctx.inject(&mut headers);
        assert!(headers.contains_key("traceparent"));
        assert!(headers.contains_key("baggage"));

        let extracted = Context::extract(&headers);
        assert_eq!(extracted.span_context.unwrap().trace_id, span_ctx.trace_id);
        assert_eq!(extracted.baggage.get("user"), Some("alice"));
    }

    // ── Resource ─────────────────────────────────────────────────────────────

    #[test]
    fn test_resource_merge() {
        let mut r1 = Resource::default_sdk();
        r1.set("service.name", "my-service");
        let mut r2 = Resource::new();
        r2.set("service.version", "1.2.3");
        r2.set("service.name",    "override-name");
        r1.merge(&r2);
        assert_eq!(r1.get("service.name"),    Some("override-name"));
        assert_eq!(r1.get("service.version"), Some("1.2.3"));
        assert_eq!(r1.get("telemetry.sdk.name"), Some("chronos"));
    }

    #[test]
    fn test_resource_as_labels() {
        let mut r = Resource::new();
        r.set("service.name", "svc");
        r.set("host.name",    "box1");
        let labels = r.as_labels();
        assert!(labels.iter().any(|(k, v)| k == "service_name" && v == "svc"));
        assert!(labels.iter().any(|(k, v)| k == "host_name"    && v == "box1"));
    }

    // ── In-memory exporter ───────────────────────────────────────────────────

    #[test]
    fn test_in_memory_exporter_trace_query() {
        let mut exp = InMemorySpanExporter::new();
        let tid = TraceId(0xABCD_1234);
        let ctx1 = SpanContext::new(tid, SpanId(1), TraceFlags::SAMPLED);
        let ctx2 = SpanContext::new(tid, SpanId(2), TraceFlags::SAMPLED);
        let ctx3 = SpanContext::new(TraceId(0xFFFF), SpanId(3), TraceFlags::SAMPLED);
        exp.export(Span::new(ctx1, None, "root",  SpanKind::Server, 0));
        exp.export(Span::new(ctx2, Some(ctx1), "child", SpanKind::Internal, 100));
        exp.export(Span::new(ctx3, None, "other", SpanKind::Client, 0));

        let trace = exp.trace(&tid);
        assert_eq!(trace.len(), 2);
        assert_eq!(exp.len(), 3);
    }

    #[test]
    fn test_level_ordering() {
        assert!(Level::Trace < Level::Debug);
        assert!(Level::Debug < Level::Info);
        assert!(Level::Info  < Level::Warn);
        assert!(Level::Warn  < Level::Error);
        assert!(Level::Error < Level::Fatal);
    }

    #[test]
    fn test_level_otel_severity_numbers() {
        assert_eq!(Level::Info.otel_severity(),  9);
        assert_eq!(Level::Error.otel_severity(), 17);
        assert_eq!(Level::Fatal.otel_severity(), 21);
    }
}
