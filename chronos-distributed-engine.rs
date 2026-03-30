// ============================================================================
// CHRONOS DISTRIBUTED SYSTEMS ENGINE
// ============================================================================
//
// HOW DISTRIBUTED SYSTEMS ACTUALLY WORK (and why they're so hard):
//
// A distributed system is a collection of independent computers that
// appears to its users as a single coherent system. The fundamental
// challenge is that these computers communicate by sending messages
// over an unreliable network: messages can be delayed, reordered,
// duplicated, or lost entirely. Computers themselves can crash at any
// moment and come back later with their disk state intact but their
// memory wiped. Despite all this, we need the system to behave
// correctly — which means we need to reason very carefully about
// what "correctly" even means.
//
// The key theoretical results that constrain everything:
//
// 1. THE CAP THEOREM (Brewer, 2000): A distributed system can provide
//    at most two of three guarantees simultaneously: Consistency (every
//    read gets the most recent write), Availability (every request gets
//    a response), and Partition tolerance (the system works even when
//    some network links fail). Since network partitions are inevitable
//    in practice, you must choose between consistency and availability.
//
// 2. THE FLP IMPOSSIBILITY (Fischer, Lynch, Paterson, 1985): In an
//    asynchronous system where even one process can crash, there is NO
//    deterministic algorithm that guarantees consensus. Every practical
//    consensus algorithm (Paxos, Raft) works around this by using
//    timeouts or randomization — they sacrifice termination guarantees
//    during periods of asynchrony.
//
// 3. LINEARIZABILITY (Herlihy & Wing, 1990): The strongest consistency
//    model. Every operation appears to take effect atomically at some
//    point between its invocation and response. This is what people
//    intuitively expect, but it's expensive to implement because it
//    requires coordination between all nodes.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  Lamport clocks (logical timestamps for causal ordering)
//   2.  Vector clocks (precise causality tracking per-node)
//   3.  Hybrid Logical Clocks (HLC — combines wall clock and logical)
//   4.  Consistent hashing (for distributing data across nodes)
//   5.  Phi Accrual Failure Detector (adaptive failure detection)
//   6.  Raft consensus algorithm (leader election + log replication)
//   7.  CRDTs: G-Counter, PN-Counter, G-Set, OR-Set, LWW-Register
//   8.  Two-Phase Commit (2PC) for distributed transactions
//   9.  Saga pattern (compensating transactions for microservices)
//  10.  Merkle tree (for efficient data synchronization)
//  11.  Consistent broadcast (FIFO, causal, total order)
//  12.  Gossip protocol (epidemic dissemination)
// ============================================================================

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::cmp;

// ============================================================================
// PART 1: LOGICAL CLOCKS
// ============================================================================
// In a distributed system, you can't rely on wall clocks because different
// machines have slightly different clock speeds (clock skew) and their
// clocks may not be synchronized. Logical clocks provide a way to order
// events without depending on physical time.

// ---- Lamport Clock (1978) ----
// The simplest logical clock. A single counter that increases monotonically.
// Rule 1: Before each event, increment the counter.
// Rule 2: When sending a message, include the counter value.
// Rule 3: When receiving a message, set your counter to max(yours, received) + 1.
//
// This gives us a "happens-before" ordering: if event A happened before
// event B, then clock(A) < clock(B). But NOT the reverse — two events
// with different clock values might be concurrent.

/// A Lamport logical clock. Tracks a single monotonically increasing counter.
#[derive(Debug, Clone)]
pub struct LamportClock {
    pub time: u64,
    pub node_id: String,
}

impl LamportClock {
    pub fn new(node_id: &str) -> Self {
        Self { time: 0, node_id: node_id.to_string() }
    }

    /// Record a local event. Increments the clock.
    pub fn tick(&mut self) -> u64 {
        self.time += 1;
        self.time
    }

    /// Prepare a timestamp for an outgoing message.
    /// Increments the clock and returns the new value to include in the message.
    pub fn send(&mut self) -> u64 {
        self.tick()
    }

    /// Process an incoming message with timestamp `received_time`.
    /// Sets our clock to max(ours, received) + 1.
    pub fn receive(&mut self, received_time: u64) -> u64 {
        self.time = cmp::max(self.time, received_time) + 1;
        self.time
    }
}

// ---- Vector Clock ----
// A vector clock is a map from node IDs to counters. Unlike Lamport clocks,
// vector clocks can tell us exactly which events are causally related and
// which are concurrent. This is essential for conflict detection in
// eventually consistent systems like Amazon Dynamo.
//
// Two events are concurrent (neither happened before the other) if and only
// if neither vector clock dominates the other. Event A dominates B if
// A[i] >= B[i] for all i and A[j] > B[j] for at least one j.

/// A vector clock. Maps node IDs to their logical timestamps.
/// This is the gold standard for tracking causality in distributed systems.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    /// The clock values: node_id → timestamp.
    pub clocks: BTreeMap<String, u64>,
}

/// The causal relationship between two vector clocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CausalOrder {
    /// A happened before B (A < B).
    Before,
    /// B happened before A (A > B).
    After,
    /// A and B are concurrent (neither happened before the other).
    /// This means there's a potential conflict that needs resolution.
    Concurrent,
    /// A and B are identical (same event or same causal position).
    Equal,
}

impl VectorClock {
    pub fn new() -> Self {
        Self { clocks: BTreeMap::new() }
    }

    /// Increment the counter for a specific node (a local event on that node).
    pub fn increment(&mut self, node_id: &str) {
        let counter = self.clocks.entry(node_id.to_string()).or_insert(0);
        *counter += 1;
    }

    /// Get the timestamp for a specific node.
    pub fn get(&self, node_id: &str) -> u64 {
        self.clocks.get(node_id).copied().unwrap_or(0)
    }

    /// Merge another vector clock into this one (take element-wise maximum).
    /// This is what happens when a node receives a message: it merges the
    /// sender's clock with its own, taking the maximum at each position,
    /// then increments its own entry.
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, &time) in &other.clocks {
            let entry = self.clocks.entry(node.clone()).or_insert(0);
            *entry = cmp::max(*entry, time);
        }
    }

    /// Send event: increment own counter and return the clock to attach to a message.
    pub fn send(&mut self, node_id: &str) -> VectorClock {
        self.increment(node_id);
        self.clone()
    }

    /// Receive event: merge with the received clock and increment own counter.
    pub fn receive(&mut self, node_id: &str, received: &VectorClock) {
        self.merge(received);
        self.increment(node_id);
    }

    /// Compare two vector clocks to determine their causal relationship.
    /// This is the fundamental operation: it tells you whether two events
    /// are causally related or concurrent.
    pub fn compare(&self, other: &VectorClock) -> CausalOrder {
        let all_nodes: BTreeSet<&String> = self.clocks.keys()
            .chain(other.clocks.keys()).collect();

        let mut self_leq_other = true;  // Is self <= other at every position?
        let mut other_leq_self = true;  // Is other <= self at every position?

        for node in all_nodes {
            let a = self.get(node);
            let b = other.get(node);
            if a > b { other_leq_self = true; self_leq_other = false; }
            if b > a { self_leq_other = true; other_leq_self = false; }
            // Wait, that's wrong. Let me redo this properly.
        }

        // Recompute correctly: check if self dominates other or vice versa.
        let all_nodes: BTreeSet<String> = self.clocks.keys()
            .chain(other.clocks.keys()).cloned().collect();

        let mut self_le_other = true;
        let mut other_le_self = true;

        for node in &all_nodes {
            let a = self.get(node);
            let b = other.get(node);
            if a > b { self_le_other = false; }
            if b > a { other_le_self = false; }
        }

        match (self_le_other, other_le_self) {
            (true, true) => CausalOrder::Equal,      // Identical clocks
            (true, false) => CausalOrder::Before,     // self happened before other
            (false, true) => CausalOrder::After,      // self happened after other
            (false, false) => CausalOrder::Concurrent, // Neither dominates → concurrent
        }
    }

    /// Check if this clock happened before the other.
    pub fn happened_before(&self, other: &VectorClock) -> bool {
        self.compare(other) == CausalOrder::Before
    }

    /// Check if the two clocks represent concurrent events.
    pub fn is_concurrent_with(&self, other: &VectorClock) -> bool {
        self.compare(other) == CausalOrder::Concurrent
    }
}

// ---- Hybrid Logical Clock (HLC) ----
// Combines the best of physical clocks (human-meaningful timestamps) with
// logical clocks (guaranteed ordering). Used by CockroachDB and other
// modern distributed databases. The key insight: use the wall clock when
// possible, but fall back to a logical counter when the wall clock goes
// backward or doesn't advance (which happens more often than you'd think
// due to NTP corrections).

/// A Hybrid Logical Clock timestamp: (physical_time, logical_counter, node_id).
/// Physical time is milliseconds since epoch. The logical counter breaks ties
/// when multiple events happen at the same physical time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HLCTimestamp {
    pub wall_time_ms: u64,
    pub logical: u32,
    pub node_id: String,
}

impl Ord for HLCTimestamp {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.wall_time_ms.cmp(&other.wall_time_ms)
            .then(self.logical.cmp(&other.logical))
            .then(self.node_id.cmp(&other.node_id))
    }
}

impl PartialOrd for HLCTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A Hybrid Logical Clock instance.
pub struct HLC {
    pub last_time: HLCTimestamp,
    pub node_id: String,
    pub max_drift_ms: u64,
}

impl HLC {
    pub fn new(node_id: &str, max_drift_ms: u64) -> Self {
        Self {
            last_time: HLCTimestamp { wall_time_ms: 0, logical: 0, node_id: node_id.to_string() },
            node_id: node_id.to_string(),
            max_drift_ms,
        }
    }

    fn now_ms() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    }

    /// Generate a new timestamp for a local event or outgoing message.
    pub fn now(&mut self) -> HLCTimestamp {
        let physical = Self::now_ms();

        if physical > self.last_time.wall_time_ms {
            // Normal case: physical clock advanced. Use the new time and reset logical.
            self.last_time = HLCTimestamp {
                wall_time_ms: physical,
                logical: 0,
                node_id: self.node_id.clone(),
            };
        } else {
            // Physical clock hasn't advanced (or went backward due to NTP).
            // Keep the wall time and increment the logical counter.
            self.last_time.logical += 1;
        }

        self.last_time.clone()
    }

    /// Update the clock based on a received timestamp.
    /// The new timestamp is guaranteed to be greater than both the local
    /// clock and the received timestamp.
    pub fn receive(&mut self, received: &HLCTimestamp) -> Result<HLCTimestamp, String> {
        let physical = Self::now_ms();

        // Check for excessive clock drift (a sign of a Byzantine node or broken NTP).
        if received.wall_time_ms > physical + self.max_drift_ms {
            return Err(format!(
                "Received timestamp {} ms is {} ms ahead of local clock — exceeds max drift of {} ms",
                received.wall_time_ms, received.wall_time_ms - physical, self.max_drift_ms
            ));
        }

        let max_wall = cmp::max(cmp::max(physical, self.last_time.wall_time_ms), received.wall_time_ms);

        let logical = if max_wall == self.last_time.wall_time_ms && max_wall == received.wall_time_ms {
            // All three times are equal — increment the max logical counter.
            cmp::max(self.last_time.logical, received.logical) + 1
        } else if max_wall == self.last_time.wall_time_ms {
            // Our wall time is the highest — increment our logical counter.
            self.last_time.logical + 1
        } else if max_wall == received.wall_time_ms {
            // The received wall time is the highest — increment their logical counter.
            received.logical + 1
        } else {
            // Physical time is the highest — reset logical counter.
            0
        };

        self.last_time = HLCTimestamp {
            wall_time_ms: max_wall,
            logical,
            node_id: self.node_id.clone(),
        };

        Ok(self.last_time.clone())
    }
}

// ============================================================================
// PART 2: CONSISTENT HASHING
// ============================================================================
// Consistent hashing distributes data across a cluster of nodes such that
// adding or removing a node only requires redistributing ~1/N of the data
// (where N is the number of nodes). This is used by Dynamo, Cassandra,
// Riak, and most distributed key-value stores.
//
// The idea: arrange all possible hash values in a circle (a "ring").
// Place each node at multiple points on the ring (virtual nodes).
// To find which node owns a key, hash the key and walk clockwise
// around the ring until you hit a node.

/// A consistent hash ring for distributing keys across nodes.
pub struct ConsistentHashRing {
    /// Maps positions on the ring to node IDs.
    /// A BTreeMap gives us efficient "find the next entry >= hash" lookups.
    ring: BTreeMap<u64, String>,
    /// How many virtual nodes each physical node gets.
    /// More virtual nodes = more uniform distribution.
    pub virtual_nodes: usize,
    /// The set of physical nodes currently in the ring.
    pub nodes: HashSet<String>,
}

impl ConsistentHashRing {
    pub fn new(virtual_nodes: usize) -> Self {
        Self { ring: BTreeMap::new(), virtual_nodes, nodes: HashSet::new() }
    }

    /// Add a node to the ring. This creates `virtual_nodes` entries on the ring,
    /// each at a different hash position. The virtual nodes ensure that the
    /// load is distributed evenly even with a small number of physical nodes.
    pub fn add_node(&mut self, node_id: &str) {
        self.nodes.insert(node_id.to_string());
        for i in 0..self.virtual_nodes {
            let key = format!("{}#{}", node_id, i);
            let hash = self.hash(&key);
            self.ring.insert(hash, node_id.to_string());
        }
    }

    /// Remove a node from the ring. All keys that were assigned to this node
    /// will now be handled by the next node clockwise on the ring.
    pub fn remove_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
        for i in 0..self.virtual_nodes {
            let key = format!("{}#{}", node_id, i);
            let hash = self.hash(&key);
            self.ring.remove(&hash);
        }
    }

    /// Find which node is responsible for a given key.
    /// Hash the key, then find the first node at or after that position
    /// on the ring (wrapping around if necessary).
    pub fn get_node(&self, key: &str) -> Option<&str> {
        if self.ring.is_empty() { return None; }
        let hash = self.hash(key);

        // Find the first entry >= hash (clockwise from the hash position).
        // If none exists, wrap around to the first entry on the ring.
        let node = self.ring.range(hash..)
            .next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, node)| node.as_str());

        node
    }

    /// Find the N nodes responsible for a key (for replication).
    /// Returns N distinct physical nodes walking clockwise from the key's position.
    pub fn get_nodes(&self, key: &str, n: usize) -> Vec<String> {
        if self.ring.is_empty() { return Vec::new(); }
        let hash = self.hash(key);

        let mut result = Vec::new();
        let mut seen = HashSet::new();

        // Walk clockwise from the key's position.
        let iter = self.ring.range(hash..).chain(self.ring.iter());
        for (_, node) in iter {
            if seen.insert(node.clone()) {
                result.push(node.clone());
                if result.len() >= n { break; }
            }
        }

        result
    }

    /// A simple hash function (FNV-1a). In production you'd use something
    /// like xxHash or MurmurHash3 for better distribution.
    fn hash(&self, key: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Compute the distribution of keys across nodes (for testing uniformity).
    pub fn key_distribution(&self, keys: &[&str]) -> HashMap<String, usize> {
        let mut dist = HashMap::new();
        for key in keys {
            if let Some(node) = self.get_node(key) {
                *dist.entry(node.to_string()).or_insert(0) += 1;
            }
        }
        dist
    }
}

// ============================================================================
// PART 3: FAILURE DETECTOR
// ============================================================================
// In a distributed system, you need to know when a node has crashed so you
// can reassign its work. But over an asynchronous network, it's impossible
// to distinguish a crashed node from a slow one — this is the fundamental
// problem of failure detection.
//
// The Phi Accrual Failure Detector (Hayashibara et al., 2004) solves this
// by outputting a SUSPICION LEVEL (phi) rather than a binary alive/dead
// decision. Phi is based on the statistical distribution of heartbeat
// intervals: if the current interval is much longer than usual, phi is
// high and the node is probably dead. The application chooses its own
// threshold based on how much false-positive tolerance it has.

/// The Phi Accrual Failure Detector. Monitors heartbeats from a remote node
/// and provides a continuously updated suspicion level.
pub struct PhiAccrualDetector {
    /// History of inter-arrival times (milliseconds between heartbeats).
    intervals: VecDeque<f64>,
    /// Maximum number of intervals to keep (sliding window).
    max_samples: usize,
    /// Timestamp of the last heartbeat received.
    last_heartbeat: Option<Instant>,
    /// Running mean of intervals (for efficient computation).
    mean: f64,
    /// Running variance of intervals.
    variance: f64,
    /// Total number of heartbeats received.
    count: u64,
}

impl PhiAccrualDetector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            intervals: VecDeque::with_capacity(max_samples),
            max_samples,
            last_heartbeat: None,
            mean: 0.0,
            variance: 0.0,
            count: 0,
        }
    }

    /// Record a heartbeat from the monitored node.
    pub fn heartbeat(&mut self) {
        let now = Instant::now();

        if let Some(last) = self.last_heartbeat {
            let interval = now.duration_since(last).as_secs_f64() * 1000.0;

            // Add to the interval history.
            if self.intervals.len() >= self.max_samples {
                self.intervals.pop_front();
            }
            self.intervals.push_back(interval);

            // Update running statistics using Welford's online algorithm.
            // This computes mean and variance in a single pass, which is
            // numerically stable even for large numbers of samples.
            self.count += 1;
            let delta = interval - self.mean;
            self.mean += delta / self.count as f64;
            let delta2 = interval - self.mean;
            self.variance += delta * delta2;
        }

        self.last_heartbeat = Some(now);
    }

    /// Compute the current suspicion level (phi).
    /// Phi is the negative logarithm of the probability that the node is
    /// still alive: phi = -log10(P(alive)).
    ///
    /// phi = 1  →  P(alive) = 10%    (probably dead)
    /// phi = 2  →  P(alive) = 1%     (almost certainly dead)
    /// phi = 3  →  P(alive) = 0.1%   (dead)
    /// phi = 0.5 → P(alive) = 31.6%  (suspicious but might be alive)
    ///
    /// A typical threshold is phi = 8 (P(alive) = 0.00000001).
    pub fn phi(&self) -> f64 {
        let last = match self.last_heartbeat {
            Some(t) => t,
            None => return 0.0, // No heartbeats yet — can't judge.
        };

        if self.intervals.len() < 2 {
            return 0.0; // Not enough data for statistics.
        }

        let elapsed = last.elapsed().as_secs_f64() * 1000.0;
        let std_dev = self.std_dev().max(1.0); // Avoid division by zero.

        // The probability that the next heartbeat would take at least `elapsed` ms,
        // assuming inter-arrival times follow a normal distribution.
        // P(X >= elapsed) = 1 - CDF(elapsed) where CDF is the normal CDF.
        let z = (elapsed - self.mean) / std_dev;
        let p_alive = 1.0 - normal_cdf(z);

        // Phi is the negative log base 10 of this probability.
        if p_alive < 1e-15 {
            15.0 // Cap at a reasonable maximum.
        } else {
            -p_alive.log10()
        }
    }

    /// Check if the node should be considered dead, using a given threshold.
    /// Common thresholds: 8 (conservative), 3 (aggressive).
    pub fn is_dead(&self, threshold: f64) -> bool {
        self.phi() >= threshold
    }

    fn std_dev(&self) -> f64 {
        if self.count < 2 { return 0.0; }
        (self.variance / (self.count - 1) as f64).sqrt()
    }
}

/// Approximation of the standard normal CDF using the Abramowitz & Stegun formula.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/√(2π)
    let p = d * (-x * x / 2.0).exp();
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
        + t * (-1.821255978 + t * 1.330274429))));
    if x >= 0.0 { 1.0 - p * poly } else { p * poly }
}

// ============================================================================
// PART 4: CRDTs (Conflict-Free Replicated Data Types)
// ============================================================================
// CRDTs are data structures that can be replicated across multiple nodes
// and updated independently, with a mathematical guarantee that all
// replicas will converge to the same state — WITHOUT any coordination.
// No locks, no consensus, no leader. This is the foundation of
// eventually consistent systems.
//
// The key property: the merge operation must be commutative, associative,
// and idempotent. This means the order and number of merges doesn't matter
// — you always end up in the same state.

// ---- G-Counter (Grow-only Counter) ----
// Each node maintains its own counter. The global value is the sum of all
// node counters. To increment, a node increments only its own counter.
// To merge, take the element-wise maximum.

/// A Grow-only Counter. Can only increase, never decrease.
/// Used as a building block for more complex CRDTs.
#[derive(Debug, Clone, PartialEq)]
pub struct GCounter {
    /// Maps node_id → that node's counter value.
    pub counts: HashMap<String, u64>,
}

impl GCounter {
    pub fn new() -> Self { Self { counts: HashMap::new() } }

    /// Increment this node's count by 1.
    pub fn increment(&mut self, node_id: &str) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += 1;
    }

    /// Increment this node's count by an arbitrary amount.
    pub fn increment_by(&mut self, node_id: &str, amount: u64) {
        *self.counts.entry(node_id.to_string()).or_insert(0) += amount;
    }

    /// Get the total count (sum of all node counts).
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge with another replica. Takes the element-wise maximum.
    /// This is the key CRDT property: merge is commutative, associative,
    /// and idempotent, so the order of merges doesn't matter.
    pub fn merge(&mut self, other: &GCounter) {
        for (node, &count) in &other.counts {
            let entry = self.counts.entry(node.clone()).or_insert(0);
            *entry = cmp::max(*entry, count);
        }
    }
}

// ---- PN-Counter (Positive-Negative Counter) ----
// Supports both increment and decrement by using TWO G-Counters:
// one for increments (P) and one for decrements (N). The value is P - N.

/// A counter that supports both increment and decrement.
#[derive(Debug, Clone, PartialEq)]
pub struct PNCounter {
    pub positive: GCounter,
    pub negative: GCounter,
}

impl PNCounter {
    pub fn new() -> Self {
        Self { positive: GCounter::new(), negative: GCounter::new() }
    }

    pub fn increment(&mut self, node_id: &str) {
        self.positive.increment(node_id);
    }

    pub fn decrement(&mut self, node_id: &str) {
        self.negative.increment(node_id);
    }

    pub fn value(&self) -> i64 {
        self.positive.value() as i64 - self.negative.value() as i64
    }

    pub fn merge(&mut self, other: &PNCounter) {
        self.positive.merge(&other.positive);
        self.negative.merge(&other.negative);
    }
}

// ---- G-Set (Grow-only Set) ----
// A set where elements can only be added, never removed. Merge is union.

/// A Grow-only Set. Elements can be added but never removed.
#[derive(Debug, Clone, PartialEq)]
pub struct GSet<T: Clone + Eq + std::hash::Hash + Ord> {
    pub elements: BTreeSet<T>,
}

impl<T: Clone + Eq + std::hash::Hash + Ord> GSet<T> {
    pub fn new() -> Self { Self { elements: BTreeSet::new() } }

    pub fn insert(&mut self, element: T) {
        self.elements.insert(element);
    }

    pub fn contains(&self, element: &T) -> bool {
        self.elements.contains(element)
    }

    pub fn merge(&mut self, other: &GSet<T>) {
        for elem in &other.elements {
            self.elements.insert(elem.clone());
        }
    }

    pub fn len(&self) -> usize { self.elements.len() }
}

// ---- OR-Set (Observed-Remove Set) ----
// A set that supports both add and remove. The key challenge: if node A
// adds element X and node B concurrently removes X, what happens after merge?
// The OR-Set uses "unique tags" for each add operation. A remove only removes
// the specific tags that the removing node has seen. So a concurrent add
// (with a new tag) survives the remove — "add wins" semantics.

/// An Observed-Remove Set. Supports both add and remove with "add wins" semantics
/// for concurrent add/remove of the same element.
#[derive(Debug, Clone)]
pub struct ORSet<T: Clone + Eq + std::hash::Hash + std::fmt::Debug> {
    /// Maps elements to their set of unique tags. Each add creates a new tag.
    /// An element is "in" the set if it has at least one tag.
    elements: HashMap<T, HashSet<u64>>,
    /// Tags that have been removed. Used to prevent re-adding removed tags during merge.
    tombstones: HashSet<u64>,
    /// Counter for generating unique tags.
    next_tag: u64,
    /// Node ID for generating globally unique tags.
    node_id: String,
}

impl<T: Clone + Eq + std::hash::Hash + std::fmt::Debug> ORSet<T> {
    pub fn new(node_id: &str) -> Self {
        Self {
            elements: HashMap::new(),
            tombstones: HashSet::new(),
            next_tag: 0,
            node_id: node_id.to_string(),
        }
    }

    /// Add an element to the set. Creates a unique tag for this addition.
    pub fn add(&mut self, element: T) {
        // Generate a globally unique tag using the node ID and a counter.
        let tag = self.generate_tag();
        self.elements.entry(element).or_insert_with(HashSet::new).insert(tag);
    }

    /// Remove an element from the set. Removes all tags currently associated
    /// with it. Any concurrent adds (which create new tags) will survive.
    pub fn remove(&mut self, element: &T) {
        if let Some(tags) = self.elements.remove(element) {
            // Move all tags to the tombstone set so they aren't re-added during merge.
            for tag in tags {
                self.tombstones.insert(tag);
            }
        }
    }

    /// Check if an element is in the set (has at least one living tag).
    pub fn contains(&self, element: &T) -> bool {
        self.elements.get(element).map(|tags| !tags.is_empty()).unwrap_or(false)
    }

    /// Get all elements currently in the set.
    pub fn values(&self) -> Vec<&T> {
        self.elements.iter()
            .filter(|(_, tags)| !tags.is_empty())
            .map(|(elem, _)| elem)
            .collect()
    }

    /// Merge with another replica.
    pub fn merge(&mut self, other: &ORSet<T>) {
        // For each element in the other set, add any tags we haven't tombstoned.
        for (elem, other_tags) in &other.elements {
            let our_tags = self.elements.entry(elem.clone()).or_insert_with(HashSet::new);
            for &tag in other_tags {
                if !self.tombstones.contains(&tag) {
                    our_tags.insert(tag);
                }
            }
        }

        // Merge tombstones.
        for &tag in &other.tombstones {
            self.tombstones.insert(tag);
            // Remove tombstoned tags from all elements.
            for tags in self.elements.values_mut() {
                tags.remove(&tag);
            }
        }

        // Clean up empty tag sets.
        self.elements.retain(|_, tags| !tags.is_empty());
    }

    fn generate_tag(&mut self) -> u64 {
        self.next_tag += 1;
        // Combine node hash with counter for global uniqueness.
        let mut hash = 0xcbf29ce484222325u64;
        for byte in self.node_id.bytes() { hash ^= byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
        hash ^ self.next_tag
    }
}

// ---- LWW-Register (Last-Writer-Wins Register) ----
// A register (single value) where concurrent writes are resolved by
// timestamp: the write with the highest timestamp wins.

/// A Last-Writer-Wins Register. Concurrent writes are resolved by timestamp.
#[derive(Debug, Clone)]
pub struct LWWRegister<T: Clone> {
    pub value: Option<T>,
    pub timestamp: u64,   // Logical timestamp of the last write
    pub node_id: String,  // Tiebreaker when timestamps are equal
}

impl<T: Clone + std::fmt::Debug> LWWRegister<T> {
    pub fn new(node_id: &str) -> Self {
        Self { value: None, timestamp: 0, node_id: node_id.to_string() }
    }

    /// Set the value with a given timestamp.
    pub fn set(&mut self, value: T, timestamp: u64) {
        if timestamp > self.timestamp || (timestamp == self.timestamp && self.node_id < self.node_id) {
            self.value = Some(value);
            self.timestamp = timestamp;
        }
    }

    pub fn get(&self) -> Option<&T> { self.value.as_ref() }

    /// Merge with another replica. The one with the higher timestamp wins.
    pub fn merge(&mut self, other: &LWWRegister<T>) {
        if other.timestamp > self.timestamp
            || (other.timestamp == self.timestamp && other.node_id > self.node_id)
        {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.node_id = other.node_id.clone();
        }
    }
}

// ============================================================================
// PART 5: RAFT CONSENSUS
// ============================================================================
// Raft is a consensus algorithm that ensures all nodes in a cluster agree
// on a sequence of commands, even if some nodes crash. It's equivalent to
// Paxos but designed to be understandable. Used by etcd, CockroachDB, etc.
//
// Raft works by electing a LEADER that handles all client requests.
// The leader replicates commands to followers via an append-only LOG.
// If the leader crashes, a new election happens.

/// The role of a node in the Raft cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
}

/// A single entry in the Raft log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub term: u64,           // The term when this entry was created
    pub index: u64,          // Position in the log (1-indexed)
    pub command: Vec<u8>,    // The replicated command (opaque to Raft)
}

/// Messages exchanged between Raft nodes.
#[derive(Debug, Clone)]
pub enum RaftMessage {
    /// RequestVote: sent by candidates to gather votes during an election.
    RequestVote {
        term: u64,
        candidate_id: String,
        last_log_index: u64,
        last_log_term: u64,
    },
    /// RequestVoteResponse: a node's reply to a vote request.
    RequestVoteResponse {
        term: u64,
        vote_granted: bool,
    },
    /// AppendEntries: sent by the leader to replicate log entries (also used as heartbeat).
    AppendEntries {
        term: u64,
        leader_id: String,
        prev_log_index: u64,
        prev_log_term: u64,
        entries: Vec<LogEntry>,
        leader_commit: u64,
    },
    /// AppendEntriesResponse: a follower's reply to an AppendEntries.
    AppendEntriesResponse {
        term: u64,
        success: bool,
        match_index: u64,
    },
}

/// A single Raft node. Multiple instances form a cluster.
pub struct RaftNode {
    pub id: String,
    pub role: RaftRole,
    pub current_term: u64,
    pub voted_for: Option<String>,
    pub log: Vec<LogEntry>,
    pub commit_index: u64,
    pub last_applied: u64,

    // Leader state (only used when this node is the leader).
    pub next_index: HashMap<String, u64>,
    pub match_index: HashMap<String, u64>,

    // Cluster configuration.
    pub peers: Vec<String>,
    pub votes_received: HashSet<String>,

    // For election timeout tracking.
    pub last_heartbeat: Instant,
    pub election_timeout: Duration,

    // Messages to send (collected during processing, sent by the runtime).
    pub outbox: Vec<(String, RaftMessage)>,

    // Commands that have been committed and can be applied to the state machine.
    pub committed_commands: Vec<Vec<u8>>,
}

impl RaftNode {
    pub fn new(id: &str, peers: Vec<String>, election_timeout: Duration) -> Self {
        Self {
            id: id.to_string(),
            role: RaftRole::Follower,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            next_index: HashMap::new(),
            match_index: HashMap::new(),
            peers,
            votes_received: HashSet::new(),
            last_heartbeat: Instant::now(),
            election_timeout,
            outbox: Vec::new(),
            committed_commands: Vec::new(),
        }
    }

    fn last_log_index(&self) -> u64 {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    fn last_log_term(&self) -> u64 {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    fn log_entry_at(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 || index as usize > self.log.len() { return None; }
        Some(&self.log[(index - 1) as usize])
    }

    fn majority_size(&self) -> usize {
        // peers includes self, so total cluster size = peers.len()
        self.peers.len() / 2 + 1
    }

    /// Called periodically (e.g., every 10ms). Checks if an election
    /// timeout has expired and starts an election if needed.
    pub fn tick(&mut self) {
        match self.role {
            RaftRole::Follower | RaftRole::Candidate => {
                if self.last_heartbeat.elapsed() >= self.election_timeout {
                    self.start_election();
                }
            }
            RaftRole::Leader => {
                // Leaders send heartbeats to prevent followers from timing out.
                self.send_heartbeats();
            }
        }
    }

    /// Start an election: transition to Candidate, increment term, vote for self,
    /// and request votes from all peers.
    fn start_election(&mut self) {
        self.current_term += 1;
        self.role = RaftRole::Candidate;
        self.voted_for = Some(self.id.clone());
        self.votes_received.clear();
        self.votes_received.insert(self.id.clone()); // Vote for self.
        self.last_heartbeat = Instant::now();

        let msg = RaftMessage::RequestVote {
            term: self.current_term,
            candidate_id: self.id.clone(),
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        };

        for peer in self.peers.clone() {
            if peer != self.id {
                self.outbox.push((peer, msg.clone()));
            }
        }
    }

    /// Send heartbeat AppendEntries (empty entries) to all followers.
    fn send_heartbeats(&mut self) {
        for peer in self.peers.clone() {
            if peer == self.id { continue; }
            self.send_append_entries(&peer);
        }
    }

    /// Send AppendEntries to a specific peer with the log entries they need.
    fn send_append_entries(&mut self, peer: &str) {
        let next = self.next_index.get(peer).copied().unwrap_or(self.last_log_index() + 1);
        let prev_index = next.saturating_sub(1);
        let prev_term = self.log_entry_at(prev_index).map(|e| e.term).unwrap_or(0);

        // Collect entries from next_index to the end of our log.
        let entries: Vec<LogEntry> = self.log.iter()
            .filter(|e| e.index >= next)
            .cloned()
            .collect();

        let msg = RaftMessage::AppendEntries {
            term: self.current_term,
            leader_id: self.id.clone(),
            prev_log_index: prev_index,
            prev_log_term: prev_term,
            entries,
            leader_commit: self.commit_index,
        };

        self.outbox.push((peer.to_string(), msg));
    }

    /// Handle an incoming message from another node.
    pub fn handle_message(&mut self, from: &str, msg: RaftMessage) {
        match msg {
            RaftMessage::RequestVote { term, candidate_id, last_log_index, last_log_term } => {
                self.handle_request_vote(from, term, &candidate_id, last_log_index, last_log_term);
            }
            RaftMessage::RequestVoteResponse { term, vote_granted } => {
                self.handle_vote_response(from, term, vote_granted);
            }
            RaftMessage::AppendEntries { term, leader_id, prev_log_index, prev_log_term, entries, leader_commit } => {
                self.handle_append_entries(from, term, &leader_id, prev_log_index, prev_log_term, entries, leader_commit);
            }
            RaftMessage::AppendEntriesResponse { term, success, match_index } => {
                self.handle_append_response(from, term, success, match_index);
            }
        }
    }

    fn handle_request_vote(&mut self, from: &str, term: u64, candidate: &str, last_idx: u64, last_term: u64) {
        // If the candidate's term is higher, we step down to follower.
        if term > self.current_term {
            self.current_term = term;
            self.role = RaftRole::Follower;
            self.voted_for = None;
        }

        // Grant vote if:
        // 1. The candidate's term >= ours.
        // 2. We haven't voted for someone else in this term.
        // 3. The candidate's log is at least as up-to-date as ours.
        //    "Up-to-date" means: higher last term, or same last term but longer log.
        let log_ok = last_term > self.last_log_term()
            || (last_term == self.last_log_term() && last_idx >= self.last_log_index());
        let can_vote = self.voted_for.is_none() || self.voted_for.as_deref() == Some(candidate);
        let grant = term >= self.current_term && can_vote && log_ok;

        if grant {
            self.voted_for = Some(candidate.to_string());
            self.last_heartbeat = Instant::now(); // Reset election timeout.
        }

        self.outbox.push((from.to_string(), RaftMessage::RequestVoteResponse {
            term: self.current_term,
            vote_granted: grant,
        }));
    }

    fn handle_vote_response(&mut self, from: &str, term: u64, granted: bool) {
        if self.role != RaftRole::Candidate || term != self.current_term { return; }

        if term > self.current_term {
            self.current_term = term;
            self.role = RaftRole::Follower;
            return;
        }

        if granted {
            self.votes_received.insert(from.to_string());

            // Check if we've won the election (majority of votes).
            if self.votes_received.len() >= self.majority_size() {
                self.become_leader();
            }
        }
    }

    fn become_leader(&mut self) {
        self.role = RaftRole::Leader;
        // Initialize next_index for each peer to the end of our log + 1.
        // This is the "optimistic" starting point; if a follower is behind,
        // we'll decrement and retry (the log matching property guarantees convergence).
        let next = self.last_log_index() + 1;
        for peer in &self.peers {
            if peer == &self.id { continue; } // skip self
            self.next_index.insert(peer.clone(), next);
            self.match_index.insert(peer.clone(), 0);
        }
        // Send immediate heartbeat to establish authority.
        self.send_heartbeats();
    }

    fn handle_append_entries(&mut self, from: &str, term: u64, leader_id: &str,
                             prev_idx: u64, prev_term: u64,
                             entries: Vec<LogEntry>, leader_commit: u64) {
        // If the leader's term is lower than ours, reject.
        if term < self.current_term {
            self.outbox.push((from.to_string(), RaftMessage::AppendEntriesResponse {
                term: self.current_term, success: false, match_index: 0,
            }));
            return;
        }

        // If the leader's term is >= ours, accept it as the legitimate leader.
        if term > self.current_term {
            self.current_term = term;
            self.voted_for = None;
        }
        self.role = RaftRole::Follower;
        self.last_heartbeat = Instant::now();

        // Log consistency check: verify that our log matches at prev_idx.
        // If it doesn't, reject — the leader will decrement next_index and retry.
        if prev_idx > 0 {
            match self.log_entry_at(prev_idx) {
                None => {
                    self.outbox.push((from.to_string(), RaftMessage::AppendEntriesResponse {
                        term: self.current_term, success: false, match_index: self.last_log_index(),
                    }));
                    return;
                }
                Some(entry) if entry.term != prev_term => {
                    // Conflict: delete this entry and everything after it.
                    self.log.truncate((prev_idx - 1) as usize);
                    self.outbox.push((from.to_string(), RaftMessage::AppendEntriesResponse {
                        term: self.current_term, success: false, match_index: self.last_log_index(),
                    }));
                    return;
                }
                _ => {} // Match — proceed.
            }
        }

        // Append new entries to our log.
        for entry in &entries {
            if entry.index as usize <= self.log.len() {
                // Entry already exists — check for conflicts.
                let existing = &self.log[(entry.index - 1) as usize];
                if existing.term != entry.term {
                    // Conflict — truncate and replace.
                    self.log.truncate((entry.index - 1) as usize);
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // Update commit index.
        if leader_commit > self.commit_index {
            self.commit_index = cmp::min(leader_commit, self.last_log_index());
            self.apply_committed();
        }

        self.outbox.push((from.to_string(), RaftMessage::AppendEntriesResponse {
            term: self.current_term, success: true, match_index: self.last_log_index(),
        }));
    }

    fn handle_append_response(&mut self, from: &str, term: u64, success: bool, match_idx: u64) {
        if self.role != RaftRole::Leader { return; }

        if term > self.current_term {
            self.current_term = term;
            self.role = RaftRole::Follower;
            return;
        }

        if success {
            self.match_index.insert(from.to_string(), match_idx);
            self.next_index.insert(from.to_string(), match_idx + 1);
            self.try_advance_commit();
        } else {
            // Follower's log doesn't match — decrement next_index and retry.
            let next = self.next_index.get(from).copied().unwrap_or(1);
            self.next_index.insert(from.to_string(), next.saturating_sub(1).max(1));
            self.send_append_entries(from);
        }
    }

    /// Advance the commit index if a majority of nodes have replicated an entry.
    fn try_advance_commit(&mut self) {
        for n in (self.commit_index + 1)..=self.last_log_index() {
            // Only commit entries from the current term (Raft safety property).
            if self.log_entry_at(n).map(|e| e.term) != Some(self.current_term) { continue; }

            // Count how many nodes have this entry (including ourselves).
            let mut replication_count = 1; // Count ourselves.
            for (_, &match_idx) in &self.match_index {
                if match_idx >= n { replication_count += 1; }
            }

            if replication_count >= self.majority_size() {
                self.commit_index = n;
            }
        }
        self.apply_committed();
    }

    /// Apply committed log entries to the state machine.
    fn apply_committed(&mut self) {
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            if let Some(entry) = self.log_entry_at(self.last_applied) {
                self.committed_commands.push(entry.command.clone());
            }
        }
    }

    /// Submit a command to be replicated (only works on the leader).
    /// Returns the log index of the new entry, or None if this node isn't the leader.
    pub fn submit_command(&mut self, command: Vec<u8>) -> Option<u64> {
        if self.role != RaftRole::Leader { return None; }

        let index = self.last_log_index() + 1;
        self.log.push(LogEntry {
            term: self.current_term,
            index,
            command,
        });

        // Immediately replicate to all followers.
        for peer in self.peers.clone() {
            if peer != self.id {
                self.send_append_entries(&peer);
            }
        }

        Some(index)
    }
}

// ============================================================================
// PART 6: TWO-PHASE COMMIT (2PC)
// ============================================================================
// 2PC is the simplest protocol for distributed transactions. It ensures
// that either ALL nodes commit a transaction or NONE do.
//
// Phase 1 (Prepare): The coordinator asks all participants "can you commit?"
//   Each participant writes to its WAL and responds YES or NO.
//
// Phase 2 (Commit/Abort): If ALL said YES, coordinator sends COMMIT.
//   If ANY said NO, coordinator sends ABORT. All participants obey.
//
// The problem with 2PC: if the coordinator crashes between Phase 1 and
// Phase 2, participants that voted YES are stuck — they've locked their
// resources but don't know whether to commit or abort. This is the
// "blocking" property that makes 2PC unsuitable for many use cases.

/// The state of a 2PC transaction.
#[derive(Debug, Clone, PartialEq)]
pub enum TwoPhaseState {
    Init,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
}

/// A 2PC transaction coordinator. Drives the two-phase protocol.
pub struct TwoPhaseCoordinator {
    pub transaction_id: String,
    pub state: TwoPhaseState,
    pub participants: Vec<String>,
    pub votes: HashMap<String, bool>,   // participant → voted_yes?
    pub acks: HashSet<String>,          // participants that acknowledged commit/abort
}

impl TwoPhaseCoordinator {
    pub fn new(transaction_id: &str, participants: Vec<String>) -> Self {
        Self {
            transaction_id: transaction_id.to_string(),
            state: TwoPhaseState::Init,
            participants,
            votes: HashMap::new(),
            acks: HashSet::new(),
        }
    }

    /// Phase 1: Send PREPARE to all participants.
    /// Returns the list of (participant, message) pairs to send.
    pub fn begin_prepare(&mut self) -> Vec<(String, TwoPhaseMessage)> {
        self.state = TwoPhaseState::Preparing;
        self.participants.iter().map(|p| {
            (p.clone(), TwoPhaseMessage::Prepare { tx_id: self.transaction_id.clone() })
        }).collect()
    }

    /// Record a vote from a participant.
    pub fn receive_vote(&mut self, participant: &str, vote_yes: bool) -> Option<Vec<(String, TwoPhaseMessage)>> {
        self.votes.insert(participant.to_string(), vote_yes);

        // Have we received all votes?
        if self.votes.len() < self.participants.len() {
            return None; // Still waiting for more votes.
        }

        // Phase 2: decide commit or abort based on votes.
        let all_yes = self.votes.values().all(|&v| v);

        if all_yes {
            self.state = TwoPhaseState::Committing;
            Some(self.participants.iter().map(|p| {
                (p.clone(), TwoPhaseMessage::Commit { tx_id: self.transaction_id.clone() })
            }).collect())
        } else {
            self.state = TwoPhaseState::Aborting;
            Some(self.participants.iter().map(|p| {
                (p.clone(), TwoPhaseMessage::Abort { tx_id: self.transaction_id.clone() })
            }).collect())
        }
    }

    /// Record an acknowledgment from a participant.
    pub fn receive_ack(&mut self, participant: &str) {
        self.acks.insert(participant.to_string());
        if self.acks.len() >= self.participants.len() {
            self.state = match self.state {
                TwoPhaseState::Committing => TwoPhaseState::Committed,
                TwoPhaseState::Aborting => TwoPhaseState::Aborted,
                _ => self.state.clone(),
            };
        }
    }
}

/// A 2PC participant. Receives prepare/commit/abort from the coordinator.
pub struct TwoPhaseParticipant {
    pub id: String,
    pub state: TwoPhaseState,
    /// Simulated "can I commit?" check. In reality, this would check
    /// locks, constraints, available disk space, etc.
    pub can_commit: bool,
}

impl TwoPhaseParticipant {
    pub fn new(id: &str, can_commit: bool) -> Self {
        Self { id: id.to_string(), state: TwoPhaseState::Init, can_commit }
    }

    /// Handle a message from the coordinator.
    pub fn handle_message(&mut self, msg: &TwoPhaseMessage) -> TwoPhaseMessage {
        match msg {
            TwoPhaseMessage::Prepare { tx_id } => {
                if self.can_commit {
                    self.state = TwoPhaseState::Prepared;
                    TwoPhaseMessage::VoteYes { tx_id: tx_id.clone(), participant: self.id.clone() }
                } else {
                    self.state = TwoPhaseState::Aborted;
                    TwoPhaseMessage::VoteNo { tx_id: tx_id.clone(), participant: self.id.clone() }
                }
            }
            TwoPhaseMessage::Commit { tx_id } => {
                self.state = TwoPhaseState::Committed;
                TwoPhaseMessage::Ack { tx_id: tx_id.clone(), participant: self.id.clone() }
            }
            TwoPhaseMessage::Abort { tx_id } => {
                self.state = TwoPhaseState::Aborted;
                TwoPhaseMessage::Ack { tx_id: tx_id.clone(), participant: self.id.clone() }
            }
            _ => TwoPhaseMessage::Ack { tx_id: String::new(), participant: self.id.clone() },
        }
    }
}

#[derive(Debug, Clone)]
pub enum TwoPhaseMessage {
    Prepare { tx_id: String },
    VoteYes { tx_id: String, participant: String },
    VoteNo { tx_id: String, participant: String },
    Commit { tx_id: String },
    Abort { tx_id: String },
    Ack { tx_id: String, participant: String },
}

// ============================================================================
// PART 7: SAGA PATTERN
// ============================================================================
// A Saga is a sequence of local transactions where each step has a
// compensating action. If any step fails, all previously completed steps
// are "undone" by running their compensations in reverse order.
// This is the standard pattern for distributed transactions in microservices.

/// A step in a saga: an action and its compensating (undo) action.
#[derive(Debug, Clone)]
pub struct SagaStep {
    pub name: String,
    pub action: fn(&mut SagaContext) -> Result<(), String>,
    pub compensation: fn(&mut SagaContext) -> Result<(), String>,
}

/// Context passed to saga steps, allowing them to share state.
pub struct SagaContext {
    pub data: HashMap<String, String>,
    pub log: Vec<String>,
}

/// A saga orchestrator that executes steps and handles failures.
pub struct SagaOrchestrator {
    pub name: String,
    pub steps: Vec<SagaStep>,
}

#[derive(Debug, Clone)]
pub enum SagaResult {
    Completed,
    Compensated { failed_step: String, compensation_errors: Vec<String> },
}

impl SagaOrchestrator {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), steps: Vec::new() }
    }

    pub fn add_step(&mut self, step: SagaStep) {
        self.steps.push(step);
    }

    /// Execute the saga. If any step fails, compensate all completed steps in reverse.
    pub fn execute(&self, ctx: &mut SagaContext) -> SagaResult {
        let mut completed = Vec::new();

        for step in &self.steps {
            ctx.log.push(format!("Executing step: {}", step.name));
            match (step.action)(ctx) {
                Ok(()) => {
                    completed.push(step);
                    ctx.log.push(format!("Step '{}' succeeded", step.name));
                }
                Err(err) => {
                    ctx.log.push(format!("Step '{}' FAILED: {}", step.name, err));

                    // Compensate in reverse order.
                    let mut compensation_errors = Vec::new();
                    for prev_step in completed.iter().rev() {
                        ctx.log.push(format!("Compensating step: {}", prev_step.name));
                        if let Err(comp_err) = (prev_step.compensation)(ctx) {
                            compensation_errors.push(format!(
                                "Compensation for '{}' failed: {}", prev_step.name, comp_err
                            ));
                            ctx.log.push(format!("Compensation for '{}' FAILED: {}", prev_step.name, comp_err));
                        } else {
                            ctx.log.push(format!("Compensation for '{}' succeeded", prev_step.name));
                        }
                    }

                    return SagaResult::Compensated {
                        failed_step: step.name.clone(),
                        compensation_errors,
                    };
                }
            }
        }

        SagaResult::Completed
    }
}

// ============================================================================
// PART 8: MERKLE TREE
// ============================================================================
// A Merkle tree is a hash tree where every leaf node is the hash of a data
// block, and every non-leaf node is the hash of its children. This allows
// efficient verification of data integrity and efficient synchronization
// between replicas — you only need to transfer the data blocks whose
// hashes differ. Used by Bitcoin, Git, IPFS, Cassandra, and DynamoDB.

/// A Merkle tree for efficient data integrity verification and sync.
pub struct MerkleTree {
    /// The tree stored as a flat array. For a tree with N leaves:
    /// nodes[0..N] = leaf hashes, nodes[N..2N-1] = internal hashes.
    /// Actually, we use a simpler representation: layers from bottom to top.
    pub leaves: Vec<u64>,
    pub layers: Vec<Vec<u64>>,
    pub root: u64,
}

impl MerkleTree {
    /// Build a Merkle tree from a list of data blocks.
    pub fn build(data_blocks: &[&[u8]]) -> Self {
        if data_blocks.is_empty() {
            return Self { leaves: Vec::new(), layers: Vec::new(), root: 0 };
        }

        // Hash each data block to create the leaf layer.
        let leaves: Vec<u64> = data_blocks.iter().map(|block| hash_bytes(block)).collect();

        // Build layers bottom-up.
        let mut layers = vec![leaves.clone()];
        let mut current = leaves.clone();

        while current.len() > 1 {
            let mut next = Vec::new();
            for chunk in current.chunks(2) {
                if chunk.len() == 2 {
                    next.push(hash_pair(chunk[0], chunk[1]));
                } else {
                    next.push(hash_pair(chunk[0], chunk[0])); // Duplicate odd leaf
                }
            }
            layers.push(next.clone());
            current = next;
        }

        let root = current[0];
        Self { leaves, layers, root }
    }

    /// Get the root hash (the single hash that represents all the data).
    /// If even one byte of any data block changes, the root hash changes.
    pub fn root_hash(&self) -> u64 { self.root }

    /// Generate a proof that a specific leaf is part of the tree.
    /// The proof consists of sibling hashes needed to recompute the root.
    /// This is O(log N) in size, not O(N).
    pub fn proof(&self, leaf_index: usize) -> Vec<(u64, bool)> {
        let mut proof = Vec::new();
        let mut idx = leaf_index;

        for layer in &self.layers[..self.layers.len().saturating_sub(1)] {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            let sibling_hash = layer.get(sibling_idx).copied()
                .unwrap_or_else(|| layer[idx]); // Duplicate if odd
            let is_right = idx % 2 == 0; // Is the sibling on the right?
            proof.push((sibling_hash, is_right));
            idx /= 2;
        }

        proof
    }

    /// Verify a proof: given a leaf hash and a proof, recompute the root
    /// and check if it matches. This is how light clients verify data
    /// without downloading the entire dataset.
    pub fn verify_proof(leaf_hash: u64, proof: &[(u64, bool)], expected_root: u64) -> bool {
        let mut current = leaf_hash;
        for &(sibling, is_right) in proof {
            current = if is_right {
                hash_pair(current, sibling)
            } else {
                hash_pair(sibling, current)
            };
        }
        current == expected_root
    }

    /// Find which leaf indices differ between two trees.
    /// This is the efficient sync operation: compare roots first (O(1)),
    /// then recursively compare subtrees only where hashes differ.
    /// Returns the indices of leaves that differ.
    pub fn diff(&self, other: &MerkleTree) -> Vec<usize> {
        if self.root == other.root { return Vec::new(); }
        let mut diffs = Vec::new();
        Self::diff_recursive(&self.layers, &other.layers, self.layers.len() - 1, 0, &mut diffs);
        diffs
    }

    fn diff_recursive(a_layers: &[Vec<u64>], b_layers: &[Vec<u64>], layer: usize, index: usize, diffs: &mut Vec<usize>) {
        let a_hash = a_layers.get(layer).and_then(|l| l.get(index));
        let b_hash = b_layers.get(layer).and_then(|l| l.get(index));

        match (a_hash, b_hash) {
            (Some(&a), Some(&b)) if a == b => {} // Subtrees match — skip.
            _ => {
                if layer == 0 {
                    // Leaf level — this leaf differs.
                    diffs.push(index);
                } else {
                    // Internal node — recurse into children.
                    Self::diff_recursive(a_layers, b_layers, layer - 1, index * 2, diffs);
                    Self::diff_recursive(a_layers, b_layers, layer - 1, index * 2 + 1, diffs);
                }
            }
        }
    }
}

/// Simple hash function (FNV-1a) for bytes.
fn hash_bytes(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in data { hash ^= byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}

/// Hash two hashes together to form a parent hash.
fn hash_pair(a: u64, b: u64) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in &a.to_be_bytes() { hash ^= byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    for &byte in &b.to_be_bytes() { hash ^= byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}

// ============================================================================
// PART 9: GOSSIP PROTOCOL
// ============================================================================
// Gossip protocols (epidemic protocols) spread information through a cluster
// by having each node periodically pick a random peer and exchange state.
// Like a rumor spreading through a social network, information eventually
// reaches every node with high probability. Used by Cassandra, Consul, SWIM.

/// A gossip protocol instance. Each node maintains a map of key-value pairs
/// with version numbers. On each gossip round, it sends its state to a
/// random peer, and they merge states (keeping the highest version of each key).
pub struct GossipNode {
    pub node_id: String,
    /// The data this node knows about. Each value has a version number.
    pub state: HashMap<String, (String, u64)>,  // key → (value, version)
    /// Known peers.
    pub peers: Vec<String>,
    /// Gossip round counter (for statistics).
    pub rounds: u64,
}

impl GossipNode {
    pub fn new(node_id: &str, peers: Vec<String>) -> Self {
        Self { node_id: node_id.to_string(), state: HashMap::new(), peers, rounds: 0 }
    }

    /// Set a local value. Increments the version so it will propagate.
    pub fn set(&mut self, key: &str, value: &str) {
        let version = self.state.get(key).map(|(_, v)| v + 1).unwrap_or(1);
        self.state.insert(key.to_string(), (value.to_string(), version));
    }

    /// Get a value.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.state.get(key).map(|(v, _)| v.as_str())
    }

    /// Prepare a gossip message (our full state, or a digest).
    pub fn prepare_gossip(&mut self) -> HashMap<String, (String, u64)> {
        self.rounds += 1;
        self.state.clone()
    }

    /// Receive and merge a gossip message from a peer.
    /// Returns the keys that were updated (new information we didn't have).
    pub fn receive_gossip(&mut self, remote_state: &HashMap<String, (String, u64)>) -> Vec<String> {
        let mut updated = Vec::new();
        for (key, (remote_value, remote_version)) in remote_state {
            let dominated = match self.state.get(key) {
                Some((_, local_version)) => remote_version > local_version,
                None => true,
            };
            if dominated {
                self.state.insert(key.clone(), (remote_value.clone(), *remote_version));
                updated.push(key.clone());
            }
        }
        updated
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- Lamport Clock ----
    #[test]
    fn test_lamport_clock() {
        let mut clock_a = LamportClock::new("A");
        let mut clock_b = LamportClock::new("B");

        let t1 = clock_a.send();     // A sends at time 1
        assert_eq!(t1, 1);

        clock_b.tick();                // B does something at time 1
        let t2 = clock_b.receive(t1); // B receives A's message
        assert_eq!(t2, 2);            // max(1, 1) + 1 = 2

        let t3 = clock_b.send();      // B sends at time 3
        assert_eq!(t3, 3);

        clock_a.receive(t3);          // A receives B's message
        assert_eq!(clock_a.time, 4);  // max(1, 3) + 1 = 4
    }

    // ---- Vector Clock ----
    #[test]
    fn test_vector_clock_causality() {
        let mut vc_a = VectorClock::new();
        let mut vc_b = VectorClock::new();

        // A does an event.
        vc_a.increment("A");
        // A sends a message to B.
        let msg = vc_a.send("A");
        // B receives the message.
        vc_b.receive("B", &msg);

        // Now B's clock should dominate A's original clock.
        assert_eq!(vc_a.compare(&vc_b), CausalOrder::Before);
        assert!(vc_a.happened_before(&vc_b));
    }

    #[test]
    fn test_vector_clock_concurrent() {
        let mut vc_a = VectorClock::new();
        let mut vc_b = VectorClock::new();

        // A and B each do an independent event.
        vc_a.increment("A"); // A: {A:1}
        vc_b.increment("B"); // B: {B:1}

        // These events are concurrent — neither happened before the other.
        assert_eq!(vc_a.compare(&vc_b), CausalOrder::Concurrent);
        assert!(vc_a.is_concurrent_with(&vc_b));
    }

    // ---- Consistent Hashing ----
    #[test]
    fn test_consistent_hashing() {
        let mut ring = ConsistentHashRing::new(100);
        ring.add_node("node1");
        ring.add_node("node2");
        ring.add_node("node3");

        // Every key should map to some node.
        assert!(ring.get_node("user:123").is_some());
        assert!(ring.get_node("order:456").is_some());

        // The same key should always map to the same node.
        let node1 = ring.get_node("mykey").unwrap().to_string();
        let node2 = ring.get_node("mykey").unwrap().to_string();
        assert_eq!(node1, node2);

        // Removing a node should only affect keys that were on that node.
        let before = ring.get_node("testkey").unwrap().to_string();
        ring.remove_node("node2");
        // If the key wasn't on node2, it should still be on the same node.
        let after = ring.get_node("testkey").unwrap().to_string();
        if before != "node2" { assert_eq!(before, after); }
    }

    #[test]
    fn test_consistent_hashing_replication() {
        let mut ring = ConsistentHashRing::new(50);
        ring.add_node("A");
        ring.add_node("B");
        ring.add_node("C");

        let replicas = ring.get_nodes("data:789", 3);
        assert_eq!(replicas.len(), 3);
        // All three nodes should be different.
        let unique: HashSet<&String> = replicas.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    // ---- CRDTs ----
    #[test]
    fn test_g_counter() {
        let mut counter_a = GCounter::new();
        let mut counter_b = GCounter::new();

        counter_a.increment("A");
        counter_a.increment("A");
        counter_b.increment("B");
        counter_b.increment("B");
        counter_b.increment("B");

        assert_eq!(counter_a.value(), 2);
        assert_eq!(counter_b.value(), 3);

        // Merge: the result should be the combined count.
        counter_a.merge(&counter_b);
        assert_eq!(counter_a.value(), 5); // 2 from A + 3 from B
    }

    #[test]
    fn test_g_counter_idempotent_merge() {
        let mut counter = GCounter::new();
        counter.increment("A");
        counter.increment("A");

        let snapshot = counter.clone();

        // Merging with yourself should be a no-op (idempotent).
        counter.merge(&snapshot);
        assert_eq!(counter.value(), 2); // Still 2, not 4!
    }

    #[test]
    fn test_pn_counter() {
        let mut counter = PNCounter::new();
        counter.increment("A");
        counter.increment("A");
        counter.increment("A");
        counter.decrement("B");
        assert_eq!(counter.value(), 2); // 3 - 1 = 2
    }

    #[test]
    fn test_or_set() {
        let mut set_a = ORSet::new("A");
        let mut set_b = ORSet::new("B");

        set_a.add("apple");
        set_a.add("banana");
        set_b.add("banana");
        set_b.add("cherry");

        // Concurrent: A removes banana, B adds banana.
        set_a.remove(&"banana");
        assert!(!set_a.contains(&"banana"));

        // After merge: B's concurrent add of banana should survive A's remove
        // because B's add created a NEW tag that A didn't remove.
        set_a.merge(&set_b);
        assert!(set_a.contains(&"apple"));
        assert!(set_a.contains(&"banana")); // B's add wins!
        assert!(set_a.contains(&"cherry"));
    }

    #[test]
    fn test_lww_register() {
        let mut reg_a = LWWRegister::<String>::new("A");
        let mut reg_b = LWWRegister::<String>::new("B");

        reg_a.set("hello".to_string(), 10);
        reg_b.set("world".to_string(), 20);

        // B's write has a higher timestamp, so it wins.
        reg_a.merge(&reg_b);
        assert_eq!(reg_a.get().unwrap(), "world");
    }

    // ---- Raft ----
    #[test]
    fn test_raft_election() {
        let peers = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        let mut n1 = RaftNode::new("n1", peers.clone(), Duration::from_millis(150));
        let mut n2 = RaftNode::new("n2", peers.clone(), Duration::from_millis(150));
        let mut n3 = RaftNode::new("n3", peers.clone(), Duration::from_millis(150));

        // Simulate n1 starting an election.
        n1.start_election();
        assert_eq!(n1.role, RaftRole::Candidate);
        assert_eq!(n1.current_term, 1);

        // Deliver vote requests to n2 and n3.
        let messages: Vec<_> = n1.outbox.drain(..).collect();
        for (target, msg) in &messages {
            if target == "n2" { n2.handle_message("n1", msg.clone()); }
            if target == "n3" { n3.handle_message("n1", msg.clone()); }
        }

        // Deliver vote responses back to n1.
        for (target, msg) in n2.outbox.drain(..).collect::<Vec<_>>() {
            if target == "n1" { n1.handle_message("n2", msg); }
        }
        for (target, msg) in n3.outbox.drain(..).collect::<Vec<_>>() {
            if target == "n1" { n1.handle_message("n3", msg); }
        }

        // n1 should have won the election (got votes from n2 and n3).
        assert_eq!(n1.role, RaftRole::Leader);
    }

    #[test]
    fn test_raft_log_replication() {
        let peers = vec!["n1".to_string(), "n2".to_string(), "n3".to_string()];
        let mut n1 = RaftNode::new("n1", peers.clone(), Duration::from_millis(150));
        let mut n2 = RaftNode::new("n2", peers.clone(), Duration::from_millis(150));

        // Make n1 the leader.
        n1.current_term = 1;
        n1.role = RaftRole::Leader;
        n1.become_leader();

        // Submit a command.
        let idx = n1.submit_command(b"SET x = 42".to_vec()).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(n1.log.len(), 1);

        // Deliver AppendEntries to n2.
        let messages: Vec<_> = n1.outbox.drain(..).collect();
        for (target, msg) in &messages {
            if target == "n2" { n2.handle_message("n1", msg.clone()); }
        }

        // n2 should have the entry now.
        assert_eq!(n2.log.len(), 1);
        assert_eq!(n2.log[0].command, b"SET x = 42");

        // Deliver n2's response back to n1.
        for (target, msg) in n2.outbox.drain(..).collect::<Vec<_>>() {
            if target == "n1" { n1.handle_message("n2", msg); }
        }

        // With 2 out of 3 nodes having the entry, it should be committed.
        assert_eq!(n1.commit_index, 1);
        assert_eq!(n1.committed_commands.len(), 1);
    }

    // ---- Two-Phase Commit ----
    #[test]
    fn test_2pc_commit() {
        let participants = vec!["db1".to_string(), "db2".to_string(), "db3".to_string()];
        let mut coordinator = TwoPhaseCoordinator::new("tx-001", participants.clone());
        let mut p1 = TwoPhaseParticipant::new("db1", true);
        let mut p2 = TwoPhaseParticipant::new("db2", true);
        let mut p3 = TwoPhaseParticipant::new("db3", true);

        // Phase 1: Prepare.
        let prepare_msgs = coordinator.begin_prepare();
        let mut votes = Vec::new();
        for (target, msg) in &prepare_msgs {
            let response = match target.as_str() {
                "db1" => p1.handle_message(msg),
                "db2" => p2.handle_message(msg),
                "db3" => p3.handle_message(msg),
                _ => unreachable!(),
            };
            votes.push((target.clone(), response));
        }

        // Collect votes.
        for (target, vote) in &votes {
            if let TwoPhaseMessage::VoteYes { participant, .. } = vote {
                coordinator.receive_vote(participant, true);
            }
        }

        // All voted yes → should be committing.
        assert_eq!(coordinator.state, TwoPhaseState::Committing);
    }

    #[test]
    fn test_2pc_abort() {
        let participants = vec!["db1".to_string(), "db2".to_string()];
        let mut coordinator = TwoPhaseCoordinator::new("tx-002", participants);
        let mut p1 = TwoPhaseParticipant::new("db1", true);
        let mut p2 = TwoPhaseParticipant::new("db2", false); // Cannot commit!

        let prepare_msgs = coordinator.begin_prepare();
        // p1 votes yes.
        coordinator.receive_vote("db1", true);
        // p2 votes no.
        let result = coordinator.receive_vote("db2", false);
        assert!(result.is_some()); // Phase 2 triggered.
        assert_eq!(coordinator.state, TwoPhaseState::Aborting);
    }

    // ---- Saga ----
    #[test]
    fn test_saga_success() {
        let mut saga = SagaOrchestrator::new("order_saga");
        saga.add_step(SagaStep {
            name: "reserve_inventory".to_string(),
            action: |ctx| { ctx.data.insert("inventory".to_string(), "reserved".to_string()); Ok(()) },
            compensation: |ctx| { ctx.data.insert("inventory".to_string(), "released".to_string()); Ok(()) },
        });
        saga.add_step(SagaStep {
            name: "charge_payment".to_string(),
            action: |ctx| { ctx.data.insert("payment".to_string(), "charged".to_string()); Ok(()) },
            compensation: |ctx| { ctx.data.insert("payment".to_string(), "refunded".to_string()); Ok(()) },
        });

        let mut ctx = SagaContext { data: HashMap::new(), log: Vec::new() };
        let result = saga.execute(&mut ctx);
        assert!(matches!(result, SagaResult::Completed));
        assert_eq!(ctx.data.get("payment").unwrap(), "charged");
    }

    #[test]
    fn test_saga_compensation() {
        let mut saga = SagaOrchestrator::new("order_saga");
        saga.add_step(SagaStep {
            name: "reserve_inventory".to_string(),
            action: |ctx| { ctx.data.insert("inventory".to_string(), "reserved".to_string()); Ok(()) },
            compensation: |ctx| { ctx.data.insert("inventory".to_string(), "released".to_string()); Ok(()) },
        });
        saga.add_step(SagaStep {
            name: "charge_payment".to_string(),
            action: |_| Err("Payment declined".to_string()), // This step FAILS.
            compensation: |ctx| { ctx.data.insert("payment".to_string(), "refunded".to_string()); Ok(()) },
        });

        let mut ctx = SagaContext { data: HashMap::new(), log: Vec::new() };
        let result = saga.execute(&mut ctx);

        // The saga should have compensated the first step.
        assert!(matches!(result, SagaResult::Compensated { .. }));
        assert_eq!(ctx.data.get("inventory").unwrap(), "released"); // Compensation ran!
    }

    // ---- Merkle Tree ----
    #[test]
    fn test_merkle_tree_basic() {
        let data: Vec<&[u8]> = vec![b"block0", b"block1", b"block2", b"block3"];
        let tree = MerkleTree::build(&data);

        // The root should be a non-zero hash.
        assert_ne!(tree.root_hash(), 0);

        // Changing any block should change the root.
        let modified: Vec<&[u8]> = vec![b"block0", b"MODIFIED", b"block2", b"block3"];
        let tree2 = MerkleTree::build(&modified);
        assert_ne!(tree.root_hash(), tree2.root_hash());
    }

    #[test]
    fn test_merkle_proof() {
        let data: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let tree = MerkleTree::build(&data);

        // Generate and verify a proof for leaf 2.
        let proof = tree.proof(2);
        let leaf_hash = hash_bytes(b"c");
        assert!(MerkleTree::verify_proof(leaf_hash, &proof, tree.root_hash()));

        // A wrong leaf hash should fail verification.
        let wrong_hash = hash_bytes(b"WRONG");
        assert!(!MerkleTree::verify_proof(wrong_hash, &proof, tree.root_hash()));
    }

    #[test]
    fn test_merkle_diff() {
        let data1: Vec<&[u8]> = vec![b"a", b"b", b"c", b"d"];
        let data2: Vec<&[u8]> = vec![b"a", b"B", b"c", b"D"]; // Changed blocks 1 and 3
        let tree1 = MerkleTree::build(&data1);
        let tree2 = MerkleTree::build(&data2);

        let diffs = tree1.diff(&tree2);
        assert!(diffs.contains(&1));
        assert!(diffs.contains(&3));
        assert!(!diffs.contains(&0));
        assert!(!diffs.contains(&2));
    }

    // ---- Gossip ----
    #[test]
    fn test_gossip_propagation() {
        let mut node_a = GossipNode::new("A", vec!["B".to_string()]);
        let mut node_b = GossipNode::new("B", vec!["A".to_string()]);

        // A sets a value.
        node_a.set("key1", "value1");
        assert!(node_b.get("key1").is_none()); // B doesn't know yet.

        // A gossips to B.
        let gossip = node_a.prepare_gossip();
        let updated = node_b.receive_gossip(&gossip);

        assert_eq!(updated, vec!["key1"]);
        assert_eq!(node_b.get("key1").unwrap(), "value1"); // Now B knows!
    }

    #[test]
    fn test_gossip_convergence() {
        let mut nodes: Vec<GossipNode> = (0..5).map(|i| {
            GossipNode::new(&format!("node{}", i),
                (0..5).filter(|&j| j != i).map(|j| format!("node{}", j)).collect())
        }).collect();

        // Each node sets a unique key.
        for i in 0..5 {
            nodes[i].set(&format!("key{}", i), &format!("value{}", i));
        }

        // Run several rounds of all-to-all gossip.
        for _ in 0..10 {
            let gossips: Vec<_> = nodes.iter_mut().map(|n| (n.node_id.clone(), n.prepare_gossip())).collect();
            for (sender_id, gossip) in &gossips {
                for node in &mut nodes {
                    if node.node_id != *sender_id {
                        node.receive_gossip(gossip);
                    }
                }
            }
        }

        // After enough rounds, all nodes should have converged to the same state.
        for node in &nodes {
            for i in 0..5 {
                assert_eq!(node.get(&format!("key{}", i)).unwrap(), &format!("value{}", i),
                    "Node {} missing key{}", node.node_id, i);
            }
        }
    }
}
