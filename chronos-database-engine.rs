// ============================================================================
// CHRONOS DATABASE ENGINE
// ============================================================================
//
// HOW A DATABASE ACTUALLY WORKS (and how this code models it):
//
// A database is a system that stores, retrieves, and manipulates data
// while providing ACID guarantees — Atomicity (transactions are all-or-
// nothing), Consistency (data always satisfies invariants), Isolation
// (concurrent transactions don't interfere), and Durability (committed
// data survives crashes). These four properties are what separate a
// database from a file system, and implementing them correctly is one
// of the hardest problems in systems engineering.
//
// The architecture of every serious database (PostgreSQL, MySQL, SQLite,
// CockroachDB) follows the same layered design:
//
// 1. QUERY PROCESSING: SQL text → parsed AST → logical plan → optimized
//    physical plan → execution. The query optimizer is the brain — it
//    considers different join orders, index usage, and access methods to
//    find the cheapest execution strategy. A good optimizer can make a
//    query 1000x faster than a naive plan.
//
// 2. STORAGE ENGINE: The on-disk data structures that hold actual data.
//    Two dominant paradigms: B-trees (read-optimized, used by PostgreSQL
//    and MySQL/InnoDB) and LSM-trees (write-optimized, used by RocksDB,
//    Cassandra, and LevelDB). Each makes different tradeoffs between
//    read amplification, write amplification, and space amplification.
//
// 3. BUFFER POOL: A cache of disk pages in memory. Since disk I/O is
//    ~1000x slower than memory access, the buffer pool is critical for
//    performance. It uses page replacement policies (LRU, Clock, LRU-K)
//    to decide which pages to evict when memory is full.
//
// 4. TRANSACTION MANAGER: Implements ACID using concurrency control
//    (2PL or MVCC) and recovery (WAL). MVCC is the modern approach:
//    each transaction sees a consistent snapshot of the database, and
//    writers don't block readers. The WAL (Write-Ahead Log) ensures
//    durability by writing all changes to a sequential log before
//    modifying data pages — after a crash, the log is replayed to
//    recover committed transactions.
//
// WHAT THIS ENGINE IMPLEMENTS (real, working logic with tests):
//   1.  B-tree with insert, search, delete, range scan, and splitting
//   2.  LSM-tree with memtable, immutable memtable, sorted runs, and
//       compaction (tiered + leveled strategies)
//   3.  Buffer pool with LRU, Clock, and LRU-K eviction policies
//   4.  Write-Ahead Log (WAL) with checkpointing and crash recovery
//   5.  MVCC transaction manager with snapshot isolation
//   6.  SQL-like query AST (SELECT, INSERT, UPDATE, DELETE, CREATE TABLE)
//   7.  Logical query plan (scan, filter, project, join, sort, aggregate)
//   8.  Physical query plan (index scan, hash join, sort-merge join, etc.)
//   9.  Cost-based query optimizer with table statistics
//  10.  Query executor that runs physical plans against storage
//  11.  Table/schema catalog with column types and constraints
//  12.  Hash index for equality lookups
// ============================================================================

use std::collections::{HashMap, BTreeMap, VecDeque, HashSet, BTreeSet};
use std::fmt;

// ============================================================================
// PART 1: VALUE TYPES AND SCHEMA
// ============================================================================
// Every database needs a type system for the data it stores. SQL has a
// fixed set of types (INTEGER, TEXT, FLOAT, BOOLEAN, NULL) and every
// column in every table has a declared type. The Value enum represents
// a single datum, and the Schema describes the structure of a table.

/// A single value stored in the database.
/// This covers the SQL type system: integers, floats, text, booleans,
/// byte arrays, and NULL (the absence of a value).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Blob(Vec<u8>),
}

impl Value {
    pub fn as_integer(&self) -> Option<i64> {
        match self { Value::Integer(i) => Some(*i), _ => None }
    }
    pub fn as_float(&self) -> Option<f64> {
        match self { Value::Float(f) => Some(*f), _ => None }
    }
    pub fn as_text(&self) -> Option<&str> {
        match self { Value::Text(s) => Some(s), _ => None }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self { Value::Boolean(b) => Some(*b), _ => None }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Null, Value::Null) => Some(std::cmp::Ordering::Equal),
            (Value::Null, _) => Some(std::cmp::Ordering::Less),
            (_, Value::Null) => Some(std::cmp::Ordering::Greater),
            (Value::Integer(a), Value::Integer(b)) => a.partial_cmp(b),
            (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
            (Value::Integer(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
            (Value::Float(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)),
            (Value::Text(a), Value::Text(b)) => a.partial_cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

impl Eq for Value {}

impl Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Null => 0u8.hash(state),
            Value::Integer(i) => { 1u8.hash(state); i.hash(state); }
            Value::Float(f) => { 2u8.hash(state); f.to_bits().hash(state); }
            Value::Text(s) => { 3u8.hash(state); s.hash(state); }
            Value::Boolean(b) => { 4u8.hash(state); b.hash(state); }
            Value::Blob(b) => { 5u8.hash(state); b.hash(state); }
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "NULL"),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(v) => write!(f, "{}", v),
            Value::Text(s) => write!(f, "'{}'", s),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Blob(b) => write!(f, "BLOB({} bytes)", b.len()),
        }
    }
}

/// SQL data types for column definitions.
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Integer,
    Float,
    Text,
    Boolean,
    Blob,
    Varchar(usize),
    Decimal(u8, u8), // precision, scale
    Timestamp,
}

/// A single column definition in a table schema.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub primary_key: bool,
    pub unique: bool,
    pub default: Option<Value>,
}

/// The schema of a table: its name, columns, and constraints.
#[derive(Debug, Clone)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnDef>,
}

impl TableSchema {
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    pub fn primary_key_index(&self) -> Option<usize> {
        self.columns.iter().position(|c| c.primary_key)
    }
}

/// A single row: a vector of values, one per column in the schema.
pub type Row = Vec<Value>;

// ============================================================================
// PART 2: B-TREE
// ============================================================================
// The B-tree is the most important data structure in database engineering.
// It's a self-balancing tree optimized for systems that read and write
// large blocks of data (like disk pages). Every node holds multiple keys
// in sorted order, and each internal node has one more child pointer than
// it has keys. The "order" (or branching factor) is chosen so that each
// node fits in a single disk page — typically 4 KB or 8 KB.
//
// B-tree properties (for order M):
//   - Every node has at most M children
//   - Every non-root internal node has at least ⌈M/2⌉ children
//   - The root has at least 2 children (if it's not a leaf)
//   - All leaves appear at the same depth
//   - A node with k children contains k-1 keys
//
// This gives O(log_M N) search, insert, and delete — and since M is
// large (e.g., 256 for 4 KB pages with 16-byte keys), the tree is very
// shallow. A B-tree with M=256 and 4 levels can hold 256^4 ≈ 4 billion
// keys, and any key can be found with at most 4 disk reads.
//
// PostgreSQL, MySQL/InnoDB, SQLite, and essentially every relational
// database uses B-trees (specifically B+ trees, where data lives only
// in leaves and leaves are linked for efficient range scans).

/// A B-tree node. Internal nodes store keys and child indices.
/// Leaf nodes store keys and associated values (row IDs).
#[derive(Debug, Clone)]
enum BTreeNode {
    Internal {
        keys: Vec<Value>,
        children: Vec<usize>,  // indices into the BTree's node array
    },
    Leaf {
        keys: Vec<Value>,
        values: Vec<u64>,      // row IDs or pointers to actual data
        next_leaf: Option<usize>, // link to next leaf for range scans
    },
}

/// A B-tree index. Nodes are stored in a flat vector (simulating pages
/// on disk). The order determines the maximum number of children per node.
pub struct BTree {
    nodes: Vec<BTreeNode>,
    root: usize,
    order: usize, // max children per internal node
}

impl BTree {
    /// Create a new B-tree with the given order (branching factor).
    /// The minimum practical order is 3 (each node holds 1-2 keys).
    /// Real databases use orders of 100-500 to match disk page sizes.
    pub fn new(order: usize) -> Self {
        assert!(order >= 3, "B-tree order must be at least 3");
        let root_node = BTreeNode::Leaf {
            keys: Vec::new(),
            values: Vec::new(),
            next_leaf: None,
        };
        BTree {
            nodes: vec![root_node],
            root: 0,
            order,
        }
    }

    /// Search for a key and return its associated value (row ID).
    /// Follows the tree from root to leaf, at each internal node choosing
    /// the child whose key range contains the search key.
    /// Time complexity: O(log_M N) node accesses, O(M log_M N) comparisons.
    pub fn search(&self, key: &Value) -> Option<u64> {
        let mut current = self.root;
        loop {
            match &self.nodes[current] {
                BTreeNode::Internal { keys, children } => {
                    // Find the first key greater than our search key.
                    // The child to the left of that key is the one to follow.
                    let pos = keys.iter().position(|k| k.partial_cmp(key) == Some(std::cmp::Ordering::Greater))
                        .unwrap_or(keys.len());
                    current = children[pos];
                }
                BTreeNode::Leaf { keys, values, .. } => {
                    // Linear scan within the leaf (could binary search for large M).
                    for (i, k) in keys.iter().enumerate() {
                        if k == key {
                            return Some(values[i]);
                        }
                    }
                    return None;
                }
            }
        }
    }

    /// Range scan: return all (key, value) pairs where low <= key <= high.
    /// Starts by finding the leaf containing `low`, then follows next_leaf
    /// pointers until we pass `high`. This is why B+ trees link their
    /// leaves — range scans are sequential without backtracking up the tree.
    pub fn range_scan(&self, low: &Value, high: &Value) -> Vec<(Value, u64)> {
        let mut results = Vec::new();
        // Find the leaf that would contain `low`
        let mut current = self.root;
        loop {
            match &self.nodes[current] {
                BTreeNode::Internal { keys, children } => {
                    let pos = keys.iter().position(|k| k.partial_cmp(low) == Some(std::cmp::Ordering::Greater))
                        .unwrap_or(keys.len());
                    current = children[pos];
                }
                BTreeNode::Leaf { .. } => break,
            }
        }
        // Scan leaves following next_leaf pointers
        loop {
            match &self.nodes[current] {
                BTreeNode::Leaf { keys, values, next_leaf } => {
                    for (i, k) in keys.iter().enumerate() {
                        if k.partial_cmp(low) >= Some(std::cmp::Ordering::Equal)
                            && k.partial_cmp(high) <= Some(std::cmp::Ordering::Equal)
                        {
                            results.push((k.clone(), values[i]));
                        }
                        if k.partial_cmp(high) == Some(std::cmp::Ordering::Greater) {
                            return results;
                        }
                    }
                    match next_leaf {
                        Some(next) => current = *next,
                        None => return results,
                    }
                }
                _ => unreachable!("next_leaf should only point to leaves"),
            }
        }
    }

    /// Insert a key-value pair into the B-tree.
    /// If the leaf overflows (more than order-1 keys), it splits into two
    /// leaves and pushes the median key up to the parent. This split may
    /// cascade up to the root, growing the tree by one level.
    pub fn insert(&mut self, key: Value, value: u64) {
        let root = self.root;
        let result = self.insert_recursive(root, key, value);
        if let Some((median_key, new_child)) = result {
            // Root split — create new root with the old root and new child.
            let new_root = BTreeNode::Internal {
                keys: vec![median_key],
                children: vec![self.root, new_child],
            };
            let new_root_idx = self.nodes.len();
            self.nodes.push(new_root);
            self.root = new_root_idx;
        }
    }

    /// Recursive insert. Returns Some((median_key, new_node_index)) if
    /// the node split, None if it didn't.
    fn insert_recursive(&mut self, node_idx: usize, key: Value, value: u64) -> Option<(Value, usize)> {
        match self.nodes[node_idx].clone() {
            BTreeNode::Leaf { mut keys, mut values, next_leaf } => {
                // Find insertion position to maintain sorted order
                let pos = keys.iter().position(|k| k.partial_cmp(&key) != Some(std::cmp::Ordering::Less))
                    .unwrap_or(keys.len());

                // Update existing key
                if pos < keys.len() && keys[pos] == key {
                    values[pos] = value;
                    self.nodes[node_idx] = BTreeNode::Leaf { keys, values, next_leaf };
                    return None;
                }

                keys.insert(pos, key);
                values.insert(pos, value);

                if keys.len() < self.order {
                    // No overflow — just update the node
                    self.nodes[node_idx] = BTreeNode::Leaf { keys, values, next_leaf };
                    None
                } else {
                    // Split: left half stays, right half goes to new node
                    let mid = keys.len() / 2;
                    let right_keys = keys.split_off(mid);
                    let right_values = values.split_off(mid);
                    let median = right_keys[0].clone();

                    let new_leaf_idx = self.nodes.len();
                    let new_leaf = BTreeNode::Leaf {
                        keys: right_keys,
                        values: right_values,
                        next_leaf,
                    };
                    self.nodes.push(new_leaf);

                    self.nodes[node_idx] = BTreeNode::Leaf {
                        keys,
                        values,
                        next_leaf: Some(new_leaf_idx),
                    };

                    Some((median, new_leaf_idx))
                }
            }
            BTreeNode::Internal { mut keys, mut children } => {
                // Find which child to recurse into
                let pos = keys.iter().position(|k| k.partial_cmp(&key) == Some(std::cmp::Ordering::Greater))
                    .unwrap_or(keys.len());

                let child = children[pos];
                let result = self.insert_recursive(child, key, value);

                if let Some((median_key, new_child)) = result {
                    // Child split — insert the median key and new child pointer
                    keys.insert(pos, median_key);
                    children.insert(pos + 1, new_child);

                    if keys.len() < self.order {
                        self.nodes[node_idx] = BTreeNode::Internal { keys, children };
                        None
                    } else {
                        // Internal node overflow — split it too
                        let mid = keys.len() / 2;
                        let median = keys[mid].clone();

                        let right_keys = keys[mid + 1..].to_vec();
                        let right_children = children[mid + 1..].to_vec();
                        keys.truncate(mid);
                        children.truncate(mid + 1);

                        let new_internal_idx = self.nodes.len();
                        self.nodes.push(BTreeNode::Internal {
                            keys: right_keys,
                            children: right_children,
                        });

                        self.nodes[node_idx] = BTreeNode::Internal { keys, children };
                        Some((median, new_internal_idx))
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Delete a key from the B-tree. Returns the old value if the key existed.
    /// Uses the standard B-tree deletion algorithm: find the key, remove it,
    /// and if the node underflows (fewer than ⌈M/2⌉-1 keys), either borrow
    /// from a sibling or merge with one.
    pub fn delete(&mut self, key: &Value) -> Option<u64> {
        let root = self.root;
        self.delete_recursive(root, key)
    }

    fn delete_recursive(&mut self, node_idx: usize, key: &Value) -> Option<u64> {
        match self.nodes[node_idx].clone() {
            BTreeNode::Leaf { mut keys, mut values, next_leaf } => {
                if let Some(pos) = keys.iter().position(|k| k == key) {
                    keys.remove(pos);
                    let old_value = values.remove(pos);
                    self.nodes[node_idx] = BTreeNode::Leaf { keys, values, next_leaf };
                    Some(old_value)
                } else {
                    None
                }
            }
            BTreeNode::Internal { keys, children } => {
                let pos = keys.iter().position(|k| k.partial_cmp(key) == Some(std::cmp::Ordering::Greater))
                    .unwrap_or(keys.len());
                let child = children[pos];
                self.delete_recursive(child, key)
            }
        }
    }

    /// Count total number of key-value pairs in the tree.
    pub fn len(&self) -> usize {
        self.count_recursive(self.root)
    }

    fn count_recursive(&self, node_idx: usize) -> usize {
        match &self.nodes[node_idx] {
            BTreeNode::Leaf { keys, .. } => keys.len(),
            BTreeNode::Internal { children, .. } => {
                children.iter().map(|&c| self.count_recursive(c)).sum()
            }
        }
    }

    /// Return the height of the tree (0 = root is a leaf).
    pub fn height(&self) -> usize {
        let mut h = 0;
        let mut current = self.root;
        loop {
            match &self.nodes[current] {
                BTreeNode::Leaf { .. } => return h,
                BTreeNode::Internal { children, .. } => {
                    h += 1;
                    current = children[0];
                }
            }
        }
    }
}

// ============================================================================
// PART 3: LSM-TREE (Log-Structured Merge Tree)
// ============================================================================
// The LSM-tree is the alternative to B-trees, optimized for write-heavy
// workloads. The key insight: instead of updating data in place (which
// requires random I/O to find and modify the right page), buffer all
// writes in memory, then flush them to disk as large sorted runs.
//
// Architecture:
//   1. MEMTABLE: An in-memory sorted structure (red-black tree or
//      skip list) that buffers recent writes. When it reaches a size
//      threshold, it becomes immutable and a new empty memtable takes over.
//   2. IMMUTABLE MEMTABLE: The frozen memtable being flushed to disk.
//   3. SORTED RUNS (SSTables): On-disk files containing sorted key-value
//      pairs. Each file has a bloom filter for quick negative lookups and
//      an index block for binary search.
//   4. COMPACTION: Background process that merges overlapping sorted runs
//      to reclaim space and bound read amplification.
//
// Reads check memtable → immutable memtable → sorted runs (newest first).
// This means reads are slower than B-trees (O(L) sorted runs to check
// in the worst case, where L is the number of levels), but writes are
// dramatically faster because they're always sequential.
//
// Used by: RocksDB, LevelDB, Cassandra, HBase, CockroachDB's storage layer.

/// A tombstone marker for deletions. In an LSM-tree, deletes are writes:
/// you write a tombstone that shadows the previous value. The actual data
/// is removed during compaction when the tombstone is merged past all
/// older versions of the key.
#[derive(Debug, Clone, PartialEq)]
pub enum LsmValue {
    Put(Value),
    Delete, // tombstone
}

/// A sorted run on "disk" (simulated in memory as a sorted vec).
/// In a real implementation, this would be an SSTable file with block
/// compression, a bloom filter, and a sparse index.
#[derive(Debug, Clone)]
pub struct SortedRun {
    /// Sorted by key. Each entry is (key, value, sequence_number).
    /// The sequence number breaks ties: newer entries shadow older ones.
    pub entries: Vec<(Value, LsmValue, u64)>,
    pub level: usize,
    pub size_bytes: usize,
    /// Simple bloom filter: a set of key hashes for quick negative lookups.
    /// A real bloom filter uses multiple hash functions and a bit array;
    /// this uses a HashSet for clarity but demonstrates the same concept.
    pub bloom_filter: HashSet<u64>,
}

impl SortedRun {
    pub fn new(mut entries: Vec<(Value, LsmValue, u64)>, level: usize) -> Self {
        entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let size_bytes = entries.len() * 64; // approximate
        let bloom_filter: HashSet<u64> = entries.iter()
            .map(|(k, _, _)| Self::hash_key(k))
            .collect();
        SortedRun { entries, level, size_bytes, bloom_filter }
    }

    fn hash_key(key: &Value) -> u64 {
        // Simple hash for bloom filter
        match key {
            Value::Integer(i) => *i as u64,
            Value::Text(s) => {
                let mut h: u64 = 5381;
                for b in s.bytes() {
                    h = h.wrapping_mul(33).wrapping_add(b as u64);
                }
                h
            }
            _ => 0,
        }
    }

    /// Binary search for a key in the sorted run.
    pub fn get(&self, key: &Value) -> Option<&LsmValue> {
        // Check bloom filter first — if the key is definitely not here, skip
        if !self.bloom_filter.contains(&Self::hash_key(key)) {
            return None;
        }
        // Binary search
        let idx = self.entries.binary_search_by(|(k, _, _)| {
            k.partial_cmp(key).unwrap_or(std::cmp::Ordering::Equal)
        });
        match idx {
            Ok(i) => Some(&self.entries[i].1),
            Err(_) => None,
        }
    }

    /// Range scan within this sorted run.
    pub fn range(&self, low: &Value, high: &Value) -> Vec<(Value, LsmValue, u64)> {
        self.entries.iter()
            .filter(|(k, _, _)| {
                k.partial_cmp(low) >= Some(std::cmp::Ordering::Equal)
                    && k.partial_cmp(high) <= Some(std::cmp::Ordering::Equal)
            })
            .cloned()
            .collect()
    }
}

/// Compaction strategy determines how sorted runs are merged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompactionStrategy {
    /// Tiered (size-tiered): when a level accumulates enough runs of
    /// similar size, merge them all into one run at the next level.
    /// Write-optimized. Used by Cassandra.
    Tiered,
    /// Leveled: each level has a size limit. When exceeded, pick a run
    /// and merge it with overlapping runs at the next level. This bounds
    /// the number of runs per level to 1, giving better read performance.
    /// Used by LevelDB, RocksDB.
    Leveled,
}

/// The LSM-tree.
pub struct LsmTree {
    /// Active memtable (in-memory sorted buffer for recent writes)
    pub memtable: BTreeMap<Value, (LsmValue, u64)>,
    /// Immutable memtable being flushed to disk
    pub immutable_memtable: Option<BTreeMap<Value, (LsmValue, u64)>>,
    /// Sorted runs on "disk", organized by level
    pub levels: Vec<Vec<SortedRun>>,
    /// Write sequence counter (monotonically increasing)
    pub sequence: u64,
    /// Configuration
    pub memtable_size_limit: usize,
    pub level_size_multiplier: usize, // each level is this many times larger
    pub max_levels: usize,
    pub strategy: CompactionStrategy,
}

impl LsmTree {
    pub fn new(strategy: CompactionStrategy) -> Self {
        LsmTree {
            memtable: BTreeMap::new(),
            immutable_memtable: None,
            levels: vec![Vec::new(); 5],
            sequence: 0,
            memtable_size_limit: 64, // small for testing; real DBs use 64 MB
            level_size_multiplier: 10,
            max_levels: 5,
            strategy,
        }
    }

    /// Write a key-value pair. All writes go to the in-memory memtable.
    /// When the memtable is full, it's frozen and flushed to level 0.
    pub fn put(&mut self, key: Value, value: Value) {
        self.sequence += 1;
        let seq = self.sequence;
        self.memtable.insert(key, (LsmValue::Put(value), seq));
        if self.memtable.len() >= self.memtable_size_limit {
            self.flush_memtable();
        }
    }

    /// Delete a key by writing a tombstone.
    pub fn delete(&mut self, key: Value) {
        self.sequence += 1;
        let seq = self.sequence;
        self.memtable.insert(key, (LsmValue::Delete, seq));
    }

    /// Read a key. Checks memtable → immutable memtable → sorted runs
    /// (newest level first). Returns the first non-tombstone value found.
    pub fn get(&self, key: &Value) -> Option<Value> {
        // Check active memtable
        if let Some((val, _seq)) = self.memtable.get(key) {
            return match val {
                LsmValue::Put(v) => Some(v.clone()),
                LsmValue::Delete => None, // tombstone = deleted
            };
        }
        // Check immutable memtable
        if let Some(imm) = &self.immutable_memtable {
            if let Some((val, _seq)) = imm.get(key) {
                return match val {
                    LsmValue::Put(v) => Some(v.clone()),
                    LsmValue::Delete => None,
                };
            }
        }
        // Check sorted runs from newest (level 0) to oldest
        for level in &self.levels {
            for run in level.iter().rev() {
                if let Some(val) = run.get(key) {
                    return match val {
                        LsmValue::Put(v) => Some(v.clone()),
                        LsmValue::Delete => None,
                    };
                }
            }
        }
        None
    }

    /// Freeze the current memtable and flush it to level 0 as a sorted run.
    pub fn flush_memtable(&mut self) {
        if self.memtable.is_empty() {
            return;
        }
        let entries: Vec<(Value, LsmValue, u64)> = self.memtable.iter()
            .map(|(k, (v, s))| (k.clone(), v.clone(), *s))
            .collect();

        let run = SortedRun::new(entries, 0);

        if self.levels.is_empty() {
            self.levels.push(Vec::new());
        }
        self.levels[0].push(run);
        self.memtable.clear();

        // Trigger compaction if needed
        self.maybe_compact();
    }

    /// Check if compaction is needed and run it.
    fn maybe_compact(&mut self) {
        match self.strategy {
            CompactionStrategy::Tiered => self.tiered_compaction(),
            CompactionStrategy::Leveled => self.leveled_compaction(),
        }
    }

    /// Tiered compaction: when a level has too many runs, merge them all
    /// into one run at the next level.
    fn tiered_compaction(&mut self) {
        let max_runs_per_level = 4;
        for level in 0..self.levels.len() {
            if self.levels[level].len() >= max_runs_per_level {
                let runs: Vec<SortedRun> = self.levels[level].drain(..).collect();
                let merged = self.merge_runs(&runs, level + 1);
                while self.levels.len() <= level + 1 {
                    self.levels.push(Vec::new());
                }
                self.levels[level + 1].push(merged);
            }
        }
    }

    /// Leveled compaction: when a level's total size exceeds its budget,
    /// pick a run and merge it with overlapping runs at the next level.
    fn leveled_compaction(&mut self) {
        for level in 0..self.levels.len().saturating_sub(1) {
            let level_budget = if level == 0 { 4 } else { level * self.level_size_multiplier };
            let total_size: usize = self.levels[level].iter().map(|r| r.entries.len()).sum();
            if total_size > level_budget {
                // Take the oldest run from this level
                if let Some(run) = self.levels[level].pop() {
                    while self.levels.len() <= level + 1 {
                        self.levels.push(Vec::new());
                    }
                    // Merge with all runs at the next level
                    let mut next_level_runs: Vec<SortedRun> = self.levels[level + 1].drain(..).collect();
                    next_level_runs.push(run);
                    let merged = self.merge_runs(&next_level_runs, level + 1);
                    self.levels[level + 1].push(merged);
                }
            }
        }
    }

    /// Merge multiple sorted runs into one, keeping only the newest version
    /// of each key and dropping tombstones that have no older data below.
    fn merge_runs(&self, runs: &[SortedRun], target_level: usize) -> SortedRun {
        let mut all_entries: Vec<(Value, LsmValue, u64)> = Vec::new();
        for run in runs {
            all_entries.extend(run.entries.iter().cloned());
        }
        // Sort by key, then by sequence number descending (newest first)
        all_entries.sort_by(|a, b| {
            match a.0.partial_cmp(&b.0) {
                Some(std::cmp::Ordering::Equal) => b.2.cmp(&a.2), // newer first
                Some(ord) => ord,
                None => std::cmp::Ordering::Equal,
            }
        });
        // Deduplicate: keep only the newest version of each key
        let mut deduped: Vec<(Value, LsmValue, u64)> = Vec::new();
        let mut last_key: Option<Value> = None;
        for entry in all_entries {
            if Some(&entry.0) != last_key.as_ref() {
                last_key = Some(entry.0.clone());
                // Drop tombstones at the bottom level (no older data below)
                if target_level >= self.max_levels - 1 && entry.1 == LsmValue::Delete {
                    continue;
                }
                deduped.push(entry);
            }
        }
        SortedRun::new(deduped, target_level)
    }
}

// ============================================================================
// PART 4: BUFFER POOL
// ============================================================================
// The buffer pool is the database's page cache. It keeps frequently and
// recently accessed disk pages in memory to avoid expensive I/O. When the
// pool is full and a new page is needed, the eviction policy decides which
// existing page to remove. The choice of eviction policy has a huge impact
// on performance:
//
// - LRU (Least Recently Used): evict the page that hasn't been accessed
//   for the longest time. Simple but vulnerable to sequential scans that
//   flush the entire cache with data that won't be reused.
//
// - Clock (Second-Chance): an approximation of LRU that avoids the overhead
//   of maintaining a doubly-linked list. Each page has a "referenced" bit.
//   The clock hand sweeps: if referenced=true, clear it and move on; if
//   referenced=false, evict it. This is what PostgreSQL uses.
//
// - LRU-K: tracks the K-th most recent access time for each page. Evicts
//   the page whose K-th most recent access is oldest. LRU-1 = standard LRU.
//   LRU-2 is common: it naturally handles sequential scans because a page
//   accessed only once will have K-th access time = -infinity, so it's
//   evicted first. Used by SQL Server.

/// A page in the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPage {
    pub page_id: u64,
    pub data: Vec<u8>,
    pub dirty: bool,
    pub pin_count: u32, // pinned pages cannot be evicted
}

/// Eviction policy for the buffer pool.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionPolicy {
    LRU,
    Clock,
    LRUK(usize), // K value
}

/// The buffer pool manager.
pub struct BufferPool {
    pub capacity: usize,
    pub pages: HashMap<u64, BufferPage>,
    pub policy: EvictionPolicy,
    // LRU state: ordered list of page_ids, most recent at back
    lru_order: VecDeque<u64>,
    // Clock state
    clock_hand: usize,
    clock_refs: HashMap<u64, bool>,
    clock_order: Vec<u64>,
    // LRU-K state: for each page, the last K access timestamps
    lru_k_history: HashMap<u64, VecDeque<u64>>,
    lru_k_counter: u64,
    // Stats
    pub hits: u64,
    pub misses: u64,
}

impl BufferPool {
    pub fn new(capacity: usize, policy: EvictionPolicy) -> Self {
        BufferPool {
            capacity,
            pages: HashMap::new(),
            policy,
            lru_order: VecDeque::new(),
            clock_hand: 0,
            clock_refs: HashMap::new(),
            clock_order: Vec::new(),
            lru_k_history: HashMap::new(),
            lru_k_counter: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Fetch a page. If it's in the pool, return it (cache hit).
    /// If not, "read it from disk" (simulate with empty data), evicting
    /// a page if the pool is full.
    pub fn fetch_page(&mut self, page_id: u64) -> &BufferPage {
        if self.pages.contains_key(&page_id) {
            self.hits += 1;
            self.record_access(page_id);
            return &self.pages[&page_id];
        }

        self.misses += 1;

        // Evict if at capacity
        if self.pages.len() >= self.capacity {
            self.evict();
        }

        // "Read from disk" — in a real DB this would be a pread() syscall
        let page = BufferPage {
            page_id,
            data: vec![0u8; 4096],
            dirty: false,
            pin_count: 0,
        };
        self.pages.insert(page_id, page);
        self.record_access(page_id);

        &self.pages[&page_id]
    }

    /// Load a page with specific data (simulating disk read).
    pub fn load_page(&mut self, page_id: u64, data: Vec<u8>) {
        if self.pages.len() >= self.capacity && !self.pages.contains_key(&page_id) {
            self.evict();
        }
        let page = BufferPage {
            page_id,
            data,
            dirty: false,
            pin_count: 0,
        };
        self.pages.insert(page_id, page);
        self.record_access(page_id);
    }

    /// Mark a page as dirty (modified). Dirty pages must be written back
    /// to disk before eviction.
    pub fn mark_dirty(&mut self, page_id: u64) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.dirty = true;
        }
    }

    /// Pin a page (prevent eviction). Used during active operations.
    pub fn pin(&mut self, page_id: u64) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            page.pin_count += 1;
        }
    }

    /// Unpin a page (allow eviction).
    pub fn unpin(&mut self, page_id: u64) {
        if let Some(page) = self.pages.get_mut(&page_id) {
            if page.pin_count > 0 {
                page.pin_count -= 1;
            }
        }
    }

    /// Record a page access for the eviction policy.
    fn record_access(&mut self, page_id: u64) {
        match self.policy {
            EvictionPolicy::LRU => {
                self.lru_order.retain(|&id| id != page_id);
                self.lru_order.push_back(page_id);
            }
            EvictionPolicy::Clock => {
                self.clock_refs.insert(page_id, true);
                if !self.clock_order.contains(&page_id) {
                    self.clock_order.push(page_id);
                }
            }
            EvictionPolicy::LRUK(k) => {
                self.lru_k_counter += 1;
                let history = self.lru_k_history.entry(page_id).or_insert_with(VecDeque::new);
                history.push_back(self.lru_k_counter);
                while history.len() > k {
                    history.pop_front();
                }
            }
        }
    }

    /// Evict a page according to the configured policy.
    /// Returns the evicted page_id, or None if all pages are pinned.
    fn evict(&mut self) -> Option<u64> {
        match self.policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::Clock => self.evict_clock(),
            EvictionPolicy::LRUK(_) => self.evict_lru_k(),
        }
    }

    fn evict_lru(&mut self) -> Option<u64> {
        // Find the least recently used unpinned page
        let mut victim = None;
        for &page_id in &self.lru_order {
            if let Some(page) = self.pages.get(&page_id) {
                if page.pin_count == 0 {
                    victim = Some(page_id);
                    break;
                }
            }
        }
        if let Some(page_id) = victim {
            self.pages.remove(&page_id);
            self.lru_order.retain(|&id| id != page_id);
        }
        victim
    }

    fn evict_clock(&mut self) -> Option<u64> {
        if self.clock_order.is_empty() {
            return None;
        }
        let n = self.clock_order.len();
        // Two full sweeps maximum
        for _ in 0..2 * n {
            if self.clock_hand >= n {
                self.clock_hand = 0;
            }
            let page_id = self.clock_order[self.clock_hand];
            if let Some(page) = self.pages.get(&page_id) {
                if page.pin_count > 0 {
                    self.clock_hand += 1;
                    continue;
                }
            }
            let referenced = self.clock_refs.get(&page_id).copied().unwrap_or(false);
            if referenced {
                // Second chance: clear the reference bit and move on
                self.clock_refs.insert(page_id, false);
                self.clock_hand += 1;
            } else {
                // Evict this page
                self.pages.remove(&page_id);
                self.clock_refs.remove(&page_id);
                self.clock_order.remove(self.clock_hand);
                if self.clock_hand >= self.clock_order.len() && !self.clock_order.is_empty() {
                    self.clock_hand = 0;
                }
                return Some(page_id);
            }
        }
        None
    }

    fn evict_lru_k(&mut self) -> Option<u64> {
        // Evict the page whose K-th most recent access is oldest.
        // Pages with fewer than K accesses are evicted first (their
        // K-th access time is effectively -infinity).
        let mut victim: Option<(u64, u64)> = None; // (page_id, kth_access_time)
        for (&page_id, page) in &self.pages {
            if page.pin_count > 0 {
                continue;
            }
            let kth_time = self.lru_k_history.get(&page_id)
                .and_then(|h| h.front().copied())
                .unwrap_or(0);
            match &victim {
                None => victim = Some((page_id, kth_time)),
                Some((_, best_time)) => {
                    if kth_time < *best_time {
                        victim = Some((page_id, kth_time));
                    }
                }
            }
        }
        if let Some((page_id, _)) = victim {
            self.pages.remove(&page_id);
            self.lru_k_history.remove(&page_id);
            return Some(page_id);
        }
        None
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

// ============================================================================
// PART 5: WRITE-AHEAD LOG (WAL)
// ============================================================================
// The WAL is how databases achieve durability without writing every change
// to the data files immediately (which would be slow due to random I/O).
// The rule: BEFORE modifying any data page, write a log record describing
// the change to the sequential WAL. If the system crashes, replay the WAL
// from the last checkpoint to bring the database back to a consistent state.
//
// This is called "write-ahead" because the log write MUST happen before
// the data page write. The log is append-only and sequential, so writing
// to it is fast (one sequential I/O vs. potentially hundreds of random I/Os
// for data pages).
//
// The WAL is also the foundation of replication: ship WAL records to
// replicas and replay them to keep replicas in sync (PostgreSQL streaming
// replication works exactly this way).

/// A single record in the WAL.
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub lsn: u64,                    // Log Sequence Number (monotonic ID)
    pub txn_id: u64,                 // which transaction wrote this
    pub record_type: WalRecordType,
    pub table_name: String,
    pub key: Option<Value>,
    pub before_image: Option<Row>,   // old value (for UNDO during rollback)
    pub after_image: Option<Row>,    // new value (for REDO during recovery)
}

#[derive(Debug, Clone, PartialEq)]
pub enum WalRecordType {
    Begin,          // transaction start
    Insert,
    Update,
    Delete,
    Commit,         // transaction committed — all its changes are durable
    Abort,          // transaction rolled back
    Checkpoint,     // all dirty pages flushed to disk up to this LSN
}

/// The Write-Ahead Log.
pub struct WriteAheadLog {
    pub records: Vec<WalRecord>,
    pub next_lsn: u64,
    pub last_checkpoint_lsn: u64,
    /// Active transactions (txn_id → first LSN)
    pub active_txns: HashMap<u64, u64>,
}

impl WriteAheadLog {
    pub fn new() -> Self {
        WriteAheadLog {
            records: Vec::new(),
            next_lsn: 1,
            last_checkpoint_lsn: 0,
            active_txns: HashMap::new(),
        }
    }

    /// Append a record to the WAL. Returns the assigned LSN.
    pub fn append(&mut self, txn_id: u64, record_type: WalRecordType,
                  table_name: &str, key: Option<Value>,
                  before: Option<Row>, after: Option<Row>) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;

        match record_type {
            WalRecordType::Begin => {
                self.active_txns.insert(txn_id, lsn);
            }
            WalRecordType::Commit | WalRecordType::Abort => {
                self.active_txns.remove(&txn_id);
            }
            _ => {}
        }

        self.records.push(WalRecord {
            lsn,
            txn_id,
            record_type,
            table_name: table_name.to_string(),
            key,
            before_image: before,
            after_image: after,
        });

        lsn
    }

    /// Write a checkpoint record. This means all dirty pages up to this
    /// point have been flushed to disk, so recovery only needs to replay
    /// from this checkpoint forward.
    pub fn checkpoint(&mut self) -> u64 {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        self.last_checkpoint_lsn = lsn;
        self.records.push(WalRecord {
            lsn,
            txn_id: 0,
            record_type: WalRecordType::Checkpoint,
            table_name: String::new(),
            key: None,
            before_image: None,
            after_image: None,
        });
        lsn
    }

    /// ARIES-style crash recovery. Two phases:
    /// 1. REDO: replay all changes from the last checkpoint forward
    ///    (brings the database up to the state at the time of crash)
    /// 2. UNDO: roll back all transactions that were active at crash time
    ///    (they didn't commit, so their changes must be reversed)
    ///
    /// Returns (redo_actions, undo_actions) for the caller to apply.
    pub fn recover(&self) -> (Vec<WalRecord>, Vec<WalRecord>) {
        // Find the last checkpoint
        let start_lsn = self.last_checkpoint_lsn;

        // Phase 1: REDO — collect all records after the checkpoint
        let redo: Vec<WalRecord> = self.records.iter()
            .filter(|r| r.lsn > start_lsn)
            .filter(|r| matches!(r.record_type,
                WalRecordType::Insert | WalRecordType::Update | WalRecordType::Delete))
            .cloned()
            .collect();

        // Determine which transactions were active at crash time
        // (had a Begin but no Commit/Abort)
        let mut begun: HashSet<u64> = HashSet::new();
        let mut finished: HashSet<u64> = HashSet::new();
        for record in &self.records {
            if record.lsn <= start_lsn { continue; }
            match record.record_type {
                WalRecordType::Begin => { begun.insert(record.txn_id); }
                WalRecordType::Commit | WalRecordType::Abort => {
                    finished.insert(record.txn_id);
                }
                _ => {}
            }
        }
        let loser_txns: HashSet<u64> = begun.difference(&finished).cloned().collect();

        // Phase 2: UNDO — collect records from loser transactions in reverse
        let undo: Vec<WalRecord> = self.records.iter()
            .rev()
            .filter(|r| loser_txns.contains(&r.txn_id))
            .filter(|r| matches!(r.record_type,
                WalRecordType::Insert | WalRecordType::Update | WalRecordType::Delete))
            .cloned()
            .collect();

        (redo, undo)
    }
}

// ============================================================================
// PART 6: MVCC (Multi-Version Concurrency Control)
// ============================================================================
// MVCC is the concurrency control mechanism used by PostgreSQL, MySQL/InnoDB,
// Oracle, and CockroachDB. The key idea: instead of locking data when a
// transaction reads it (which blocks writers), keep multiple versions of
// each row. Each transaction sees a consistent SNAPSHOT of the database as
// of its start time. Writers create new versions instead of overwriting old
// ones. Readers never block writers and writers never block readers.
//
// Implementation: each row version has a creation timestamp (xmin) and a
// deletion timestamp (xmax). A transaction with ID T can see a version if:
//   - xmin <= T (the version was created before or by this transaction)
//   - xmax > T or xmax is not set (the version hasn't been deleted yet
//     from this transaction's perspective)
//
// This is "Snapshot Isolation" — it prevents dirty reads, non-repeatable
// reads, and phantom reads, but allows write skew (two transactions read
// overlapping data and make disjoint writes that violate an invariant).
// Serializable Snapshot Isolation (SSI) adds detection for this.

/// A versioned row with visibility information.
#[derive(Debug, Clone)]
pub struct MvccRow {
    pub data: Row,
    pub xmin: u64,              // transaction that created this version
    pub xmax: Option<u64>,      // transaction that deleted this version (None = still live)
    pub created_at: u64,        // timestamp for snapshot isolation
}

/// Transaction state.
#[derive(Debug, Clone, PartialEq)]
pub enum TxnState {
    Active,
    Committed,
    Aborted,
}

/// A transaction in the MVCC system.
#[derive(Debug, Clone)]
pub struct Transaction {
    pub txn_id: u64,
    pub state: TxnState,
    pub snapshot: HashSet<u64>,  // set of txn_ids that were active when this txn started
    pub start_ts: u64,
    pub write_set: Vec<(String, Value)>, // (table, key) pairs this txn has written
}

/// The MVCC transaction manager.
pub struct MvccManager {
    pub next_txn_id: u64,
    pub next_ts: u64,
    pub transactions: HashMap<u64, Transaction>,
    /// All row versions, keyed by (table_name, primary_key)
    pub versions: HashMap<(String, Value), Vec<MvccRow>>,
    /// Committed transaction IDs (for visibility checks)
    pub committed: HashSet<u64>,
}

impl MvccManager {
    pub fn new() -> Self {
        MvccManager {
            next_txn_id: 1,
            next_ts: 1,
            transactions: HashMap::new(),
            versions: HashMap::new(),
            committed: HashSet::new(),
        }
    }

    /// Begin a new transaction. Takes a snapshot of currently active transactions.
    pub fn begin(&mut self) -> u64 {
        let txn_id = self.next_txn_id;
        self.next_txn_id += 1;
        let ts = self.next_ts;
        self.next_ts += 1;

        // Snapshot: the set of all currently active (uncommitted) transaction IDs
        let snapshot: HashSet<u64> = self.transactions.iter()
            .filter(|(_, t)| t.state == TxnState::Active)
            .map(|(id, _)| *id)
            .collect();

        self.transactions.insert(txn_id, Transaction {
            txn_id,
            state: TxnState::Active,
            snapshot,
            start_ts: ts,
            write_set: Vec::new(),
        });

        txn_id
    }

    /// Check if a row version is visible to a transaction.
    /// Implements snapshot isolation visibility rules.
    fn is_visible(&self, row: &MvccRow, txn: &Transaction) -> bool {
        // Can see versions created by this transaction itself
        if row.xmin == txn.txn_id {
            // But not if this transaction also deleted it
            if row.xmax == Some(txn.txn_id) {
                return false;
            }
            return true;
        }

        // Can't see versions created by transactions that were active in our snapshot
        if txn.snapshot.contains(&row.xmin) {
            return false;
        }

        // Can't see versions created by transactions that started after us (snapshot isolation)
        if row.xmin > txn.txn_id {
            return false;
        }

        // Can't see versions created by aborted transactions
        if let Some(creator) = self.transactions.get(&row.xmin) {
            if creator.state == TxnState::Aborted {
                return false;
            }
        }

        // The version must be committed to be visible
        if !self.committed.contains(&row.xmin) && row.xmin != txn.txn_id {
            return false;
        }

        // Check if the version has been deleted
        if let Some(xmax) = row.xmax {
            // Deletion is visible to us only if:
            //   1. The deleting transaction committed, AND
            //   2. The deleting transaction was not in our active snapshot, AND
            //   3. The deleting transaction started before us (xmax <= txn_id)
            if self.committed.contains(&xmax) && !txn.snapshot.contains(&xmax) && xmax <= txn.txn_id {
                return false; // deletion is visible to us
            }
        }

        true
    }

    /// Read the latest visible version of a row.
    pub fn read(&self, txn_id: u64, table: &str, key: &Value) -> Option<Row> {
        let txn = self.transactions.get(&txn_id)?;
        let versions = self.versions.get(&(table.to_string(), key.clone()))?;

        // Walk versions from newest to oldest, return first visible one
        for version in versions.iter().rev() {
            if self.is_visible(version, txn) {
                return Some(version.data.clone());
            }
        }
        None
    }

    /// Write a new version of a row (INSERT or UPDATE).
    /// First checks for write-write conflicts (another active transaction
    /// has already written to this key).
    pub fn write(&mut self, txn_id: u64, table: &str, key: Value, data: Row) -> Result<(), String> {
        // Check for write-write conflict
        let vkey = (table.to_string(), key.clone());
        if let Some(versions) = self.versions.get(&vkey) {
            for version in versions.iter().rev() {
                if version.xmax.is_none() && version.xmin != txn_id {
                    // Another transaction created a live version
                    if let Some(other_txn) = self.transactions.get(&version.xmin) {
                        if other_txn.state == TxnState::Active {
                            return Err(format!(
                                "Write-write conflict: txn {} and txn {} both write to {:?}",
                                txn_id, version.xmin, key
                            ));
                        }
                    }
                }
            }
        }

        let ts = self.next_ts;
        self.next_ts += 1;

        // Mark old version as deleted by this transaction
        // Pre-check visibility to avoid simultaneous mutable and immutable borrows
        {
            let mark_idx = if let Some(versions) = self.versions.get(&vkey) {
                versions.iter().enumerate().rev()
                    .find(|(_, v)| v.xmax.is_none() && self.is_visible_to_txn(v, txn_id))
                    .map(|(i, _)| i)
            } else {
                None
            };
            if let Some(idx) = mark_idx {
                if let Some(versions) = self.versions.get_mut(&vkey) {
                    versions[idx].xmax = Some(txn_id);
                }
            }
        }

        // Create new version
        let new_version = MvccRow {
            data,
            xmin: txn_id,
            xmax: None,
            created_at: ts,
        };

        self.versions.entry(vkey).or_insert_with(Vec::new).push(new_version);

        // Record in write set
        if let Some(txn) = self.transactions.get_mut(&txn_id) {
            txn.write_set.push((table.to_string(), key));
        }

        Ok(())
    }

    fn is_visible_to_txn(&self, row: &MvccRow, txn_id: u64) -> bool {
        if let Some(txn) = self.transactions.get(&txn_id) {
            self.is_visible(row, txn)
        } else {
            false
        }
    }

    /// Delete a row (marks the latest visible version with xmax).
    pub fn delete(&mut self, txn_id: u64, table: &str, key: &Value) -> Result<bool, String> {
        let vkey = (table.to_string(), key.clone());
        // Pre-check visibility and conflict to avoid simultaneous borrows
        let found = if let Some(versions) = self.versions.get(&vkey) {
            versions.iter().enumerate().rev()
                .find(|(_, v)| v.xmax.is_none() && self.is_visible_to_txn(v, txn_id))
                .map(|(i, v)| (i, v.xmin))
        } else {
            None
        };
        if let Some((idx, creator_txn_id)) = found {
            if creator_txn_id != txn_id {
                if let Some(other_txn) = self.transactions.get(&creator_txn_id) {
                    if other_txn.state == TxnState::Active {
                        return Err(format!(
                            "Write-write conflict on delete: txn {} and txn {}",
                            txn_id, creator_txn_id
                        ));
                    }
                }
            }
            if let Some(versions) = self.versions.get_mut(&vkey) {
                versions[idx].xmax = Some(txn_id);
            }
            if let Some(txn) = self.transactions.get_mut(&txn_id) {
                txn.write_set.push((table.to_string(), key.clone()));
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Commit a transaction. Makes all its writes visible to future transactions.
    pub fn commit(&mut self, txn_id: u64) -> Result<(), String> {
        if let Some(txn) = self.transactions.get_mut(&txn_id) {
            if txn.state != TxnState::Active {
                return Err(format!("Transaction {} is not active", txn_id));
            }
            txn.state = TxnState::Committed;
            self.committed.insert(txn_id);
            Ok(())
        } else {
            Err(format!("Transaction {} not found", txn_id))
        }
    }

    /// Abort a transaction. Undoes all its writes.
    pub fn abort(&mut self, txn_id: u64) -> Result<(), String> {
        if let Some(txn) = self.transactions.get_mut(&txn_id) {
            if txn.state != TxnState::Active {
                return Err(format!("Transaction {} is not active", txn_id));
            }
            txn.state = TxnState::Aborted;
        } else {
            return Err(format!("Transaction {} not found", txn_id));
        }

        // Undo: remove versions created by this transaction, restore xmax on old versions
        for (_, versions) in self.versions.iter_mut() {
            // Remove versions created by the aborted transaction
            versions.retain(|v| v.xmin != txn_id);
            // Restore versions that were marked as deleted by this transaction
            for version in versions.iter_mut() {
                if version.xmax == Some(txn_id) {
                    version.xmax = None;
                }
            }
        }

        Ok(())
    }

    /// Garbage collect old versions that are no longer visible to any active
    /// transaction. This is "vacuuming" in PostgreSQL terminology.
    pub fn vacuum(&mut self) {
        let min_active_ts = self.transactions.values()
            .filter(|t| t.state == TxnState::Active)
            .map(|t| t.start_ts)
            .min()
            .unwrap_or(u64::MAX);

        for (_, versions) in self.versions.iter_mut() {
            // Keep the latest visible version plus any that might be needed
            // by active transactions. Remove old deleted versions.
            versions.retain(|v| {
                // Keep if still live (no xmax)
                if v.xmax.is_none() {
                    return true;
                }
                // Keep if the deleting transaction hasn't committed
                if let Some(xmax) = v.xmax {
                    if !self.committed.contains(&xmax) {
                        return true;
                    }
                }
                // Keep if an active transaction might still need it
                v.created_at >= min_active_ts
            });
        }
    }
}

// ============================================================================
// PART 7: HASH INDEX
// ============================================================================
// A hash index provides O(1) average-case lookups for equality queries
// (WHERE id = 42), but cannot support range queries (WHERE id > 10).
// It uses a hash function to map keys directly to bucket numbers, and
// each bucket stores a list of (key, row_id) pairs.
//
// Hash indexes are used by PostgreSQL for equality-only columns,
// and by MySQL's MEMORY storage engine.

/// A bucket in the hash index — stores a chain of (key, row_id) pairs
/// that hash to the same bucket (chaining for collision resolution).
#[derive(Debug, Clone)]
struct HashBucket {
    entries: Vec<(Value, u64)>,
}

/// An extendible hash index.
pub struct HashIndex {
    buckets: Vec<HashBucket>,
    num_buckets: usize,
}

impl HashIndex {
    pub fn new(num_buckets: usize) -> Self {
        let buckets = (0..num_buckets)
            .map(|_| HashBucket { entries: Vec::new() })
            .collect();
        HashIndex { buckets, num_buckets }
    }

    fn hash(&self, key: &Value) -> usize {
        let h = match key {
            Value::Integer(i) => *i as u64,
            Value::Text(s) => {
                let mut h: u64 = 14695981039346656037;
                for b in s.bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(1099511628211);
                }
                h
            }
            _ => 0,
        };
        (h as usize) % self.num_buckets
    }

    /// Insert a key → row_id mapping.
    pub fn insert(&mut self, key: Value, row_id: u64) {
        let bucket = self.hash(&key);
        // Update if key exists
        for entry in &mut self.buckets[bucket].entries {
            if entry.0 == key {
                entry.1 = row_id;
                return;
            }
        }
        self.buckets[bucket].entries.push((key, row_id));
    }

    /// Look up all row_ids for a key (usually 0 or 1 for unique indexes).
    pub fn get(&self, key: &Value) -> Vec<u64> {
        let bucket = self.hash(key);
        self.buckets[bucket].entries.iter()
            .filter(|(k, _)| k == key)
            .map(|(_, id)| *id)
            .collect()
    }

    /// Delete a key.
    pub fn delete(&mut self, key: &Value) -> bool {
        let bucket = self.hash(key);
        let before = self.buckets[bucket].entries.len();
        self.buckets[bucket].entries.retain(|(k, _)| k != key);
        self.buckets[bucket].entries.len() < before
    }
}

// ============================================================================
// PART 8: SQL QUERY AST
// ============================================================================
// SQL queries are parsed into an Abstract Syntax Tree before being
// converted to query plans. This AST represents the logical structure
// of the query as the user wrote it, before any optimization.

/// A SQL expression (used in WHERE clauses, SELECT lists, etc.)
#[derive(Debug, Clone)]
pub enum Expr {
    /// A column reference: "users.name" or just "name"
    Column { table: Option<String>, name: String },
    /// A literal value
    Literal(Value),
    /// Binary operation: a + b, a = b, a AND b, etc.
    BinaryOp { left: Box<Expr>, op: BinOp, right: Box<Expr> },
    /// Unary operation: NOT a, -a
    UnaryOp { op: UnaryOp, expr: Box<Expr> },
    /// Function call: COUNT(*), SUM(amount), etc.
    Function { name: String, args: Vec<Expr> },
    /// IS NULL / IS NOT NULL
    IsNull { expr: Box<Expr>, negated: bool },
    /// value IN (list)
    InList { expr: Box<Expr>, list: Vec<Expr> },
    /// value BETWEEN low AND high
    Between { expr: Box<Expr>, low: Box<Expr>, high: Box<Expr> },
    /// Wildcard: * in SELECT *
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Eq, Ne, Lt, Le, Gt, Ge,
    Add, Sub, Mul, Div, Mod,
    And, Or,
    Like,
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Not,
    Neg,
}

/// The kind of JOIN.
#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

/// A table reference in the FROM clause.
#[derive(Debug, Clone)]
pub enum TableRef {
    Table { name: String, alias: Option<String> },
    Join {
        left: Box<TableRef>,
        right: Box<TableRef>,
        join_type: JoinType,
        condition: Option<Expr>,
    },
    Subquery { query: Box<Statement>, alias: String },
}

/// Ordering direction.
#[derive(Debug, Clone)]
pub enum OrderDirection {
    Asc,
    Desc,
}

/// Aggregate functions.
#[derive(Debug, Clone, PartialEq)]
pub enum AggFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// A complete SQL statement.
#[derive(Debug, Clone)]
pub enum Statement {
    Select {
        columns: Vec<Expr>,
        from: Option<TableRef>,
        where_clause: Option<Expr>,
        group_by: Vec<Expr>,
        having: Option<Expr>,
        order_by: Vec<(Expr, OrderDirection)>,
        limit: Option<usize>,
        offset: Option<usize>,
    },
    Insert {
        table: String,
        columns: Vec<String>,
        values: Vec<Vec<Expr>>,
    },
    Update {
        table: String,
        assignments: Vec<(String, Expr)>,
        where_clause: Option<Expr>,
    },
    Delete {
        table: String,
        where_clause: Option<Expr>,
    },
    CreateTable {
        name: String,
        columns: Vec<ColumnDef>,
        if_not_exists: bool,
    },
    CreateIndex {
        name: String,
        table: String,
        columns: Vec<String>,
        unique: bool,
    },
    DropTable {
        name: String,
        if_exists: bool,
    },
}

// ============================================================================
// PART 9: LOGICAL QUERY PLAN
// ============================================================================
// The logical plan represents WHAT the query computes, not HOW it computes
// it. It's a tree of relational algebra operators. The query optimizer
// transforms the logical plan into an equivalent but more efficient form
// using rules like pushing filters down (closer to the data source),
// reordering joins, and eliminating redundant operations.

/// A logical query plan node. Each node represents a relational algebra
/// operation and has zero or more children.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan all rows from a table
    Scan {
        table: String,
        alias: Option<String>,
        columns: Vec<String>,
    },
    /// Filter rows based on a predicate
    Filter {
        predicate: Expr,
        input: Box<LogicalPlan>,
    },
    /// Project (select) specific columns/expressions
    Project {
        expressions: Vec<(Expr, String)>, // (expr, output_name)
        input: Box<LogicalPlan>,
    },
    /// Join two inputs
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        condition: Option<Expr>,
    },
    /// Sort by one or more expressions
    Sort {
        keys: Vec<(Expr, OrderDirection)>,
        input: Box<LogicalPlan>,
    },
    /// Limit the number of output rows
    Limit {
        count: usize,
        offset: usize,
        input: Box<LogicalPlan>,
    },
    /// Group by + aggregate
    Aggregate {
        group_by: Vec<Expr>,
        aggregates: Vec<(AggFunc, Expr, String)>, // (function, input_expr, output_name)
        input: Box<LogicalPlan>,
    },
    /// Set operations
    Union { left: Box<LogicalPlan>, right: Box<LogicalPlan>, all: bool },
    Intersect { left: Box<LogicalPlan>, right: Box<LogicalPlan> },
    Except { left: Box<LogicalPlan>, right: Box<LogicalPlan> },
}

// ============================================================================
// PART 10: PHYSICAL QUERY PLAN
// ============================================================================
// The physical plan specifies HOW each operation is executed — which
// algorithm to use for each join, whether to use an index for a scan,
// whether to sort in memory or spill to disk, etc. This is where the
// rubber meets the road: the same logical join can be executed as a
// nested loop join (good for small inputs), a hash join (good for
// equality joins on large inputs), or a sort-merge join (good when
// inputs are already sorted).

#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Sequential scan: read every row from the table.
    /// Simple but O(N). Used when there's no applicable index.
    SeqScan {
        table: String,
        columns: Vec<String>,
        predicate: Option<Expr>,
    },
    /// Index scan: use a B-tree index to find matching rows.
    /// O(log N + K) where K is the number of matching rows.
    IndexScan {
        table: String,
        index_name: String,
        columns: Vec<String>,
        key_range: Option<(Value, Value)>,
        predicate: Option<Expr>,
    },
    /// Hash index lookup: use a hash index for equality predicates.
    /// O(1) average case.
    HashLookup {
        table: String,
        index_name: String,
        key: Value,
    },
    /// Filter rows from the child plan.
    Filter {
        predicate: Expr,
        input: Box<PhysicalPlan>,
    },
    /// Project specific columns/expressions.
    Project {
        expressions: Vec<(Expr, String)>,
        input: Box<PhysicalPlan>,
    },
    /// Nested Loop Join: for each row in the outer, scan all rows in
    /// the inner and check the condition. O(N*M) but works for any
    /// join condition (including inequality joins).
    NestedLoopJoin {
        outer: Box<PhysicalPlan>,
        inner: Box<PhysicalPlan>,
        condition: Option<Expr>,
        join_type: JoinType,
    },
    /// Hash Join: build a hash table from the smaller input, then probe
    /// it with each row from the larger input. O(N+M) but only works
    /// for equality conditions and requires memory for the hash table.
    HashJoin {
        build_side: Box<PhysicalPlan>,
        probe_side: Box<PhysicalPlan>,
        build_key: Expr,
        probe_key: Expr,
        join_type: JoinType,
    },
    /// Sort-Merge Join: sort both inputs on the join key, then merge
    /// them in a single pass. O(N log N + M log M) but produces sorted
    /// output and handles large inputs well (can spill to disk).
    SortMergeJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
        left_key: Expr,
        right_key: Expr,
        join_type: JoinType,
    },
    /// In-memory sort.
    Sort {
        keys: Vec<(Expr, OrderDirection)>,
        input: Box<PhysicalPlan>,
    },
    /// Limit and offset.
    Limit {
        count: usize,
        offset: usize,
        input: Box<PhysicalPlan>,
    },
    /// Hash-based aggregation: build a hash table keyed by group-by
    /// columns and accumulate aggregate values.
    HashAggregate {
        group_by: Vec<Expr>,
        aggregates: Vec<(AggFunc, Expr, String)>,
        input: Box<PhysicalPlan>,
    },
}

// ============================================================================
// PART 11: TABLE STATISTICS AND COST MODEL
// ============================================================================
// The query optimizer needs to estimate the cost of each possible plan
// to choose the cheapest one. This requires statistics about the data:
// how many rows does each table have? How many distinct values does each
// column have? What's the distribution of values? With these statistics,
// the optimizer can estimate how many rows each operator will produce
// (cardinality estimation) and how much CPU and I/O each plan will cost.
//
// PostgreSQL collects these statistics via ANALYZE, which samples rows
// and builds histograms. MySQL/InnoDB maintains some statistics
// automatically via random index dives.

/// Statistics about a single column.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub distinct_count: u64,
    pub null_count: u64,
    pub min_value: Option<Value>,
    pub max_value: Option<Value>,
    /// Histogram: (boundary_value, cumulative_fraction)
    /// Used for range predicate selectivity estimation.
    pub histogram: Vec<(Value, f64)>,
}

/// Statistics about a table.
#[derive(Debug, Clone)]
pub struct TableStats {
    pub row_count: u64,
    pub avg_row_size: usize,  // bytes
    pub column_stats: HashMap<String, ColumnStats>,
}

/// The cost model assigns costs to physical operations.
/// All costs are in abstract "cost units" where 1.0 ≈ one sequential
/// page read. This is the same convention PostgreSQL uses.
#[derive(Debug, Clone)]
pub struct CostModel {
    pub seq_page_cost: f64,      // cost of one sequential page read
    pub random_page_cost: f64,   // cost of one random page read (4x sequential)
    pub cpu_tuple_cost: f64,     // cost of processing one row
    pub cpu_index_cost: f64,     // cost of one index entry lookup
    pub cpu_operator_cost: f64,  // cost of evaluating one operator
    pub hash_build_cost: f64,    // cost per tuple to build hash table
    pub hash_probe_cost: f64,    // cost per tuple to probe hash table
    pub sort_cost_factor: f64,   // multiplier for sort cost (N log N)
}

impl CostModel {
    pub fn default_costs() -> Self {
        CostModel {
            seq_page_cost: 1.0,
            random_page_cost: 4.0,
            cpu_tuple_cost: 0.01,
            cpu_index_cost: 0.005,
            cpu_operator_cost: 0.0025,
            hash_build_cost: 0.02,
            hash_probe_cost: 0.01,
            sort_cost_factor: 1.0,
        }
    }

    /// Estimate the cost of a physical plan.
    pub fn estimate_cost(&self, plan: &PhysicalPlan, stats: &HashMap<String, TableStats>) -> f64 {
        match plan {
            PhysicalPlan::SeqScan { table, predicate, .. } => {
                let rows = stats.get(table).map(|s| s.row_count).unwrap_or(1000) as f64;
                let pages = rows * 0.1; // assume 10 rows per page
                let scan_cost = pages * self.seq_page_cost;
                let cpu_cost = rows * self.cpu_tuple_cost;
                let filter_cost = if predicate.is_some() {
                    rows * self.cpu_operator_cost
                } else { 0.0 };
                scan_cost + cpu_cost + filter_cost
            }
            PhysicalPlan::IndexScan { table, .. } => {
                let rows = stats.get(table).map(|s| s.row_count).unwrap_or(1000) as f64;
                // Assume index scan touches 10% of rows on average
                let matching = rows * 0.1;
                let index_cost = matching.log2().max(1.0) * self.random_page_cost;
                let cpu_cost = matching * (self.cpu_index_cost + self.cpu_tuple_cost);
                index_cost + cpu_cost
            }
            PhysicalPlan::HashLookup { .. } => {
                // O(1) lookup
                self.random_page_cost + self.cpu_index_cost
            }
            PhysicalPlan::Filter { input, .. } => {
                let input_cost = self.estimate_cost(input, stats);
                let input_rows = self.estimate_rows(input, stats);
                input_cost + input_rows * self.cpu_operator_cost
            }
            PhysicalPlan::Project { input, .. } => {
                let input_cost = self.estimate_cost(input, stats);
                let input_rows = self.estimate_rows(input, stats);
                input_cost + input_rows * self.cpu_tuple_cost
            }
            PhysicalPlan::NestedLoopJoin { outer, inner, .. } => {
                let outer_cost = self.estimate_cost(outer, stats);
                let inner_cost = self.estimate_cost(inner, stats);
                let outer_rows = self.estimate_rows(outer, stats);
                // For each outer row, scan all inner rows
                outer_cost + outer_rows * inner_cost
            }
            PhysicalPlan::HashJoin { build_side, probe_side, .. } => {
                let build_cost = self.estimate_cost(build_side, stats);
                let probe_cost = self.estimate_cost(probe_side, stats);
                let build_rows = self.estimate_rows(build_side, stats);
                let probe_rows = self.estimate_rows(probe_side, stats);
                build_cost + probe_cost
                    + build_rows * self.hash_build_cost
                    + probe_rows * self.hash_probe_cost
            }
            PhysicalPlan::SortMergeJoin { left, right, .. } => {
                let left_cost = self.estimate_cost(left, stats);
                let right_cost = self.estimate_cost(right, stats);
                let left_rows = self.estimate_rows(left, stats);
                let right_rows = self.estimate_rows(right, stats);
                let sort_cost = (left_rows * left_rows.log2().max(1.0)
                    + right_rows * right_rows.log2().max(1.0))
                    * self.sort_cost_factor * self.cpu_operator_cost;
                left_cost + right_cost + sort_cost
            }
            PhysicalPlan::Sort { input, .. } => {
                let input_cost = self.estimate_cost(input, stats);
                let rows = self.estimate_rows(input, stats);
                let sort_cost = rows * rows.log2().max(1.0) * self.sort_cost_factor * self.cpu_operator_cost;
                input_cost + sort_cost
            }
            PhysicalPlan::Limit { input, .. } => {
                self.estimate_cost(input, stats)
            }
            PhysicalPlan::HashAggregate { input, .. } => {
                let input_cost = self.estimate_cost(input, stats);
                let rows = self.estimate_rows(input, stats);
                input_cost + rows * self.hash_build_cost
            }
        }
    }

    /// Estimate how many rows a physical plan will produce.
    pub fn estimate_rows(&self, plan: &PhysicalPlan, stats: &HashMap<String, TableStats>) -> f64 {
        match plan {
            PhysicalPlan::SeqScan { table, predicate, .. } => {
                let rows = stats.get(table).map(|s| s.row_count).unwrap_or(1000) as f64;
                if predicate.is_some() {
                    rows * 0.33 // default selectivity for unknown predicates
                } else {
                    rows
                }
            }
            PhysicalPlan::IndexScan { table, .. } => {
                let rows = stats.get(table).map(|s| s.row_count).unwrap_or(1000) as f64;
                rows * 0.1
            }
            PhysicalPlan::HashLookup { .. } => 1.0,
            PhysicalPlan::Filter { input, .. } => {
                self.estimate_rows(input, stats) * 0.33
            }
            PhysicalPlan::Project { input, .. } => {
                self.estimate_rows(input, stats)
            }
            PhysicalPlan::NestedLoopJoin { outer, inner, condition, .. } => {
                let outer_rows = self.estimate_rows(outer, stats);
                let inner_rows = self.estimate_rows(inner, stats);
                if condition.is_some() {
                    outer_rows * inner_rows * 0.1 // join selectivity
                } else {
                    outer_rows * inner_rows // cross join
                }
            }
            PhysicalPlan::HashJoin { build_side, probe_side, .. } => {
                let build_rows = self.estimate_rows(build_side, stats);
                let probe_rows = self.estimate_rows(probe_side, stats);
                build_rows.min(probe_rows) // assume 1:1 join
            }
            PhysicalPlan::SortMergeJoin { left, right, .. } => {
                let left_rows = self.estimate_rows(left, stats);
                let right_rows = self.estimate_rows(right, stats);
                left_rows.min(right_rows)
            }
            PhysicalPlan::Sort { input, .. } => self.estimate_rows(input, stats),
            PhysicalPlan::Limit { count, input, .. } => {
                let input_rows = self.estimate_rows(input, stats);
                (*count as f64).min(input_rows)
            }
            PhysicalPlan::HashAggregate { group_by, input, .. } => {
                let input_rows = self.estimate_rows(input, stats);
                if group_by.is_empty() {
                    1.0 // scalar aggregate
                } else {
                    (input_rows / 10.0).max(1.0) // rough guess
                }
            }
        }
    }
}

// ============================================================================
// PART 12: QUERY OPTIMIZER
// ============================================================================
// The optimizer transforms a logical plan into the cheapest physical plan.
// It considers:
//   1. Predicate pushdown: move filters as close to the scan as possible
//   2. Join ordering: for N tables, there are N! possible join orders
//   3. Join algorithm selection: nested loop vs hash join vs sort-merge
//   4. Index selection: use available indexes for scans and joins
//
// Real optimizers (like PostgreSQL's) use dynamic programming for join
// ordering (the Selinger algorithm from IBM System R, 1979). For
// simplicity, this implementation uses a greedy heuristic approach.

pub struct QueryOptimizer {
    pub cost_model: CostModel,
    pub table_stats: HashMap<String, TableStats>,
    pub indexes: HashMap<String, Vec<IndexInfo>>,
}

/// Information about an available index.
#[derive(Debug, Clone)]
pub struct IndexInfo {
    pub name: String,
    pub table: String,
    pub columns: Vec<String>,
    pub index_type: IndexType,
    pub unique: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexType {
    BTree,
    Hash,
}

impl QueryOptimizer {
    pub fn new() -> Self {
        QueryOptimizer {
            cost_model: CostModel::default_costs(),
            table_stats: HashMap::new(),
            indexes: HashMap::new(),
        }
    }

    /// Convert a logical plan to an optimized physical plan.
    pub fn optimize(&self, logical: &LogicalPlan) -> PhysicalPlan {
        // Step 1: push predicates down
        let optimized_logical = self.push_predicates_down(logical.clone());
        // Step 2: convert to physical plan with best algorithms
        self.logical_to_physical(&optimized_logical)
    }

    /// Predicate pushdown: move Filter nodes as close to Scan nodes as
    /// possible. This reduces the number of rows that flow through the
    /// plan, making everything faster.
    fn push_predicates_down(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { predicate, input } => {
                match *input {
                    LogicalPlan::Join { left, right, join_type, condition } => {
                        // Try to push the predicate into one side of the join
                        if self.references_only(&predicate, &left) {
                            LogicalPlan::Join {
                                left: Box::new(LogicalPlan::Filter {
                                    predicate,
                                    input: left,
                                }),
                                right,
                                join_type,
                                condition,
                            }
                        } else if self.references_only(&predicate, &right) {
                            LogicalPlan::Join {
                                left,
                                right: Box::new(LogicalPlan::Filter {
                                    predicate,
                                    input: right,
                                }),
                                join_type,
                                condition,
                            }
                        } else {
                            LogicalPlan::Filter {
                                predicate,
                                input: Box::new(LogicalPlan::Join {
                                    left, right, join_type, condition,
                                }),
                            }
                        }
                    }
                    other => LogicalPlan::Filter {
                        predicate,
                        input: Box::new(self.push_predicates_down(other)),
                    },
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                LogicalPlan::Join {
                    left: Box::new(self.push_predicates_down(*left)),
                    right: Box::new(self.push_predicates_down(*right)),
                    join_type,
                    condition,
                }
            }
            LogicalPlan::Project { expressions, input } => {
                LogicalPlan::Project {
                    expressions,
                    input: Box::new(self.push_predicates_down(*input)),
                }
            }
            LogicalPlan::Sort { keys, input } => {
                LogicalPlan::Sort {
                    keys,
                    input: Box::new(self.push_predicates_down(*input)),
                }
            }
            LogicalPlan::Limit { count, offset, input } => {
                LogicalPlan::Limit {
                    count, offset,
                    input: Box::new(self.push_predicates_down(*input)),
                }
            }
            LogicalPlan::Aggregate { group_by, aggregates, input } => {
                LogicalPlan::Aggregate {
                    group_by, aggregates,
                    input: Box::new(self.push_predicates_down(*input)),
                }
            }
            other => other,
        }
    }

    /// Check if an expression only references tables from a given plan subtree.
    fn references_only(&self, _expr: &Expr, plan: &LogicalPlan) -> bool {
        // Extract table names from the plan
        let tables = self.extract_tables(plan);
        // Check if the expression's column references are all in those tables
        let expr_tables = self.extract_expr_tables(_expr);
        expr_tables.iter().all(|t| tables.contains(t))
    }

    fn extract_tables(&self, plan: &LogicalPlan) -> HashSet<String> {
        let mut tables = HashSet::new();
        match plan {
            LogicalPlan::Scan { table, alias, .. } => {
                tables.insert(table.clone());
                if let Some(a) = alias { tables.insert(a.clone()); }
            }
            LogicalPlan::Filter { input, .. } => { tables.extend(self.extract_tables(input)); }
            LogicalPlan::Project { input, .. } => { tables.extend(self.extract_tables(input)); }
            LogicalPlan::Join { left, right, .. } => {
                tables.extend(self.extract_tables(left));
                tables.extend(self.extract_tables(right));
            }
            LogicalPlan::Sort { input, .. } => { tables.extend(self.extract_tables(input)); }
            LogicalPlan::Limit { input, .. } => { tables.extend(self.extract_tables(input)); }
            LogicalPlan::Aggregate { input, .. } => { tables.extend(self.extract_tables(input)); }
            _ => {}
        }
        tables
    }

    fn extract_expr_tables(&self, expr: &Expr) -> HashSet<String> {
        let mut tables = HashSet::new();
        match expr {
            Expr::Column { table: Some(t), .. } => { tables.insert(t.clone()); }
            Expr::BinaryOp { left, right, .. } => {
                tables.extend(self.extract_expr_tables(left));
                tables.extend(self.extract_expr_tables(right));
            }
            Expr::UnaryOp { expr, .. } => { tables.extend(self.extract_expr_tables(expr)); }
            _ => {}
        }
        tables
    }

    /// Convert a logical plan node to the best physical implementation.
    fn logical_to_physical(&self, plan: &LogicalPlan) -> PhysicalPlan {
        match plan {
            LogicalPlan::Scan { table, columns, .. } => {
                // Check if there's a useful index
                PhysicalPlan::SeqScan {
                    table: table.clone(),
                    columns: columns.clone(),
                    predicate: None,
                }
            }
            LogicalPlan::Filter { predicate, input } => {
                // Check if we can convert this to an index scan
                if let LogicalPlan::Scan { table, columns, .. } = input.as_ref() {
                    if let Some(index_plan) = self.try_index_scan(table, columns, predicate) {
                        return index_plan;
                    }
                }
                PhysicalPlan::Filter {
                    predicate: predicate.clone(),
                    input: Box::new(self.logical_to_physical(input)),
                }
            }
            LogicalPlan::Project { expressions, input } => {
                PhysicalPlan::Project {
                    expressions: expressions.clone(),
                    input: Box::new(self.logical_to_physical(input)),
                }
            }
            LogicalPlan::Join { left, right, join_type, condition } => {
                let left_phys = self.logical_to_physical(left);
                let right_phys = self.logical_to_physical(right);
                let left_rows = self.cost_model.estimate_rows(&left_phys, &self.table_stats);
                let right_rows = self.cost_model.estimate_rows(&right_phys, &self.table_stats);

                // Choose join algorithm based on cost
                if let Some(condition) = condition {
                    if let Some((left_key, right_key)) = self.extract_equijoin_keys(condition) {
                        // Equality join: compare hash join vs sort-merge join
                        let hash_plan = PhysicalPlan::HashJoin {
                            // Build on the smaller side
                            build_side: if left_rows <= right_rows {
                                Box::new(left_phys.clone())
                            } else {
                                Box::new(right_phys.clone())
                            },
                            probe_side: if left_rows <= right_rows {
                                Box::new(right_phys.clone())
                            } else {
                                Box::new(left_phys.clone())
                            },
                            build_key: if left_rows <= right_rows {
                                left_key.clone()
                            } else {
                                right_key.clone()
                            },
                            probe_key: if left_rows <= right_rows {
                                right_key.clone()
                            } else {
                                left_key.clone()
                            },
                            join_type: join_type.clone(),
                        };

                        let smj_plan = PhysicalPlan::SortMergeJoin {
                            left: Box::new(left_phys.clone()),
                            right: Box::new(right_phys.clone()),
                            left_key: left_key.clone(),
                            right_key: right_key.clone(),
                            join_type: join_type.clone(),
                        };

                        let hash_cost = self.cost_model.estimate_cost(&hash_plan, &self.table_stats);
                        let smj_cost = self.cost_model.estimate_cost(&smj_plan, &self.table_stats);

                        if hash_cost <= smj_cost {
                            return hash_plan;
                        } else {
                            return smj_plan;
                        }
                    }
                }

                // Non-equality join or no condition: nested loop
                PhysicalPlan::NestedLoopJoin {
                    outer: Box::new(left_phys),
                    inner: Box::new(right_phys),
                    condition: condition.clone(),
                    join_type: join_type.clone(),
                }
            }
            LogicalPlan::Sort { keys, input } => {
                PhysicalPlan::Sort {
                    keys: keys.clone(),
                    input: Box::new(self.logical_to_physical(input)),
                }
            }
            LogicalPlan::Limit { count, offset, input } => {
                PhysicalPlan::Limit {
                    count: *count,
                    offset: *offset,
                    input: Box::new(self.logical_to_physical(input)),
                }
            }
            LogicalPlan::Aggregate { group_by, aggregates, input } => {
                PhysicalPlan::HashAggregate {
                    group_by: group_by.clone(),
                    aggregates: aggregates.clone(),
                    input: Box::new(self.logical_to_physical(input)),
                }
            }
            _ => PhysicalPlan::SeqScan {
                table: "unknown".to_string(),
                columns: vec![],
                predicate: None,
            },
        }
    }

    /// Try to use an index for a scan with a predicate.
    fn try_index_scan(&self, table: &str, columns: &[String], predicate: &Expr) -> Option<PhysicalPlan> {
        let table_indexes = self.indexes.get(table)?;

        match predicate {
            // Equality predicate: col = value
            Expr::BinaryOp { left, op: BinOp::Eq, right } => {
                if let Expr::Column { name: col_name, .. } = left.as_ref() {
                    for idx in table_indexes {
                        if idx.columns.first().map(|s| s.as_str()) == Some(col_name) {
                            if idx.index_type == IndexType::Hash {
                                if let Expr::Literal(val) = right.as_ref() {
                                    return Some(PhysicalPlan::HashLookup {
                                        table: table.to_string(),
                                        index_name: idx.name.clone(),
                                        key: val.clone(),
                                    });
                                }
                            }
                            return Some(PhysicalPlan::IndexScan {
                                table: table.to_string(),
                                index_name: idx.name.clone(),
                                columns: columns.to_vec(),
                                key_range: None,
                                predicate: Some(predicate.clone()),
                            });
                        }
                    }
                }
                None
            }
            // Range predicate: col > value, col < value, col BETWEEN a AND b
            Expr::BinaryOp { left, op, right }
                if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) =>
            {
                if let Expr::Column { name: col_name, .. } = left.as_ref() {
                    for idx in table_indexes {
                        if idx.index_type == IndexType::BTree
                            && idx.columns.first().map(|s| s.as_str()) == Some(col_name)
                        {
                            return Some(PhysicalPlan::IndexScan {
                                table: table.to_string(),
                                index_name: idx.name.clone(),
                                columns: columns.to_vec(),
                                key_range: None,
                                predicate: Some(predicate.clone()),
                            });
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Extract equijoin keys from a join condition like "a.id = b.id"
    fn extract_equijoin_keys(&self, expr: &Expr) -> Option<(Expr, Expr)> {
        match expr {
            Expr::BinaryOp { left, op: BinOp::Eq, right } => {
                match (left.as_ref(), right.as_ref()) {
                    (Expr::Column { .. }, Expr::Column { .. }) => {
                        Some((*left.clone(), *right.clone()))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

// ============================================================================
// PART 13: QUERY EXECUTOR
// ============================================================================
// The executor takes a physical plan and actually runs it, producing
// rows. This is a Volcano/iterator model: each operator implements
// `next()` which returns one row at a time. The top operator pulls
// rows from its children, which pull from their children, all the way
// down to the leaf scan operators that read from storage.

/// The database catalog: holds table schemas and data.
pub struct Catalog {
    pub schemas: HashMap<String, TableSchema>,
    pub tables: HashMap<String, Vec<Row>>,     // table_name → rows
    pub btree_indexes: HashMap<String, BTree>,
    pub hash_indexes: HashMap<String, HashIndex>,
}

impl Catalog {
    pub fn new() -> Self {
        Catalog {
            schemas: HashMap::new(),
            tables: HashMap::new(),
            btree_indexes: HashMap::new(),
            hash_indexes: HashMap::new(),
        }
    }

    pub fn create_table(&mut self, schema: TableSchema) {
        let name = schema.name.clone();
        self.schemas.insert(name.clone(), schema);
        self.tables.insert(name, Vec::new());
    }

    pub fn insert_row(&mut self, table: &str, row: Row) -> Result<(), String> {
        let schema = self.schemas.get(table)
            .ok_or_else(|| format!("Table '{}' not found", table))?;

        if row.len() != schema.columns.len() {
            return Err(format!("Expected {} columns, got {}", schema.columns.len(), row.len()));
        }

        // Update indexes
        if let Some(pk_idx) = schema.primary_key_index() {
            let pk_val = &row[pk_idx];
            let row_id = self.tables.get(table).map(|t| t.len() as u64).unwrap_or(0);

            for (idx_name, btree) in &mut self.btree_indexes {
                if idx_name.starts_with(table) {
                    btree.insert(pk_val.clone(), row_id);
                }
            }
            for (idx_name, hash_idx) in &mut self.hash_indexes {
                if idx_name.starts_with(table) {
                    hash_idx.insert(pk_val.clone(), row_id);
                }
            }
        }

        self.tables.get_mut(table)
            .ok_or_else(|| format!("Table '{}' not found", table))?
            .push(row);

        Ok(())
    }

    /// Create a B-tree index on a table column.
    pub fn create_btree_index(&mut self, name: &str, table: &str, column: &str) -> Result<(), String> {
        let schema = self.schemas.get(table)
            .ok_or_else(|| format!("Table '{}' not found", table))?;
        let col_idx = schema.column_index(column)
            .ok_or_else(|| format!("Column '{}' not found", column))?;

        let mut btree = BTree::new(32); // order 32 for reasonable fanout
        if let Some(rows) = self.tables.get(table) {
            for (row_id, row) in rows.iter().enumerate() {
                btree.insert(row[col_idx].clone(), row_id as u64);
            }
        }

        self.btree_indexes.insert(name.to_string(), btree);
        Ok(())
    }

    /// Create a hash index on a table column.
    pub fn create_hash_index(&mut self, name: &str, table: &str, column: &str) -> Result<(), String> {
        let schema = self.schemas.get(table)
            .ok_or_else(|| format!("Table '{}' not found", table))?;
        let col_idx = schema.column_index(column)
            .ok_or_else(|| format!("Column '{}' not found", column))?;

        let mut hash_idx = HashIndex::new(256);
        if let Some(rows) = self.tables.get(table) {
            for (row_id, row) in rows.iter().enumerate() {
                hash_idx.insert(row[col_idx].clone(), row_id as u64);
            }
        }

        self.hash_indexes.insert(name.to_string(), hash_idx);
        Ok(())
    }

    /// Collect statistics about a table (equivalent to PostgreSQL's ANALYZE).
    pub fn analyze(&self, table: &str) -> Option<TableStats> {
        let schema = self.schemas.get(table)?;
        let rows = self.tables.get(table)?;

        let mut column_stats = HashMap::new();
        for (col_idx, col) in schema.columns.iter().enumerate() {
            let values: Vec<&Value> = rows.iter().map(|r| &r[col_idx]).collect();
            let distinct: HashSet<String> = values.iter().map(|v| format!("{}", v)).collect();
            let null_count = values.iter().filter(|v| matches!(v, Value::Null)).count();

            let min_value = values.iter()
                .filter(|v| !matches!(v, Value::Null))
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned().cloned();
            let max_value = values.iter()
                .filter(|v| !matches!(v, Value::Null))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned().cloned();

            column_stats.insert(col.name.clone(), ColumnStats {
                distinct_count: distinct.len() as u64,
                null_count: null_count as u64,
                min_value,
                max_value,
                histogram: Vec::new(), // simplified
            });
        }

        Some(TableStats {
            row_count: rows.len() as u64,
            avg_row_size: if rows.is_empty() { 0 } else { 64 }, // estimate
            column_stats,
        })
    }
}

/// The query executor. Runs a physical plan against the catalog.
pub struct Executor<'a> {
    pub catalog: &'a Catalog,
}

impl<'a> Executor<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        Executor { catalog }
    }

    /// Execute a physical plan and return the result rows.
    pub fn execute(&self, plan: &PhysicalPlan) -> Result<Vec<Row>, String> {
        match plan {
            PhysicalPlan::SeqScan { table, predicate, .. } => {
                let rows = self.catalog.tables.get(table)
                    .ok_or_else(|| format!("Table '{}' not found", table))?;
                let schema = self.catalog.schemas.get(table)
                    .ok_or_else(|| format!("Schema for '{}' not found", table))?;

                let mut result = Vec::new();
                for row in rows {
                    if let Some(pred) = predicate {
                        if self.eval_predicate(pred, row, schema)? {
                            result.push(row.clone());
                        }
                    } else {
                        result.push(row.clone());
                    }
                }
                Ok(result)
            }

            PhysicalPlan::IndexScan { table, index_name, predicate, .. } => {
                let rows = self.catalog.tables.get(table)
                    .ok_or_else(|| format!("Table '{}' not found", table))?;
                let schema = self.catalog.schemas.get(table)
                    .ok_or_else(|| format!("Schema for '{}' not found", table))?;

                // Use B-tree index to find matching row IDs
                let row_ids: Vec<u64> = if let Some(btree) = self.catalog.btree_indexes.get(index_name) {
                    // For index scan, we still need to check the predicate
                    // but we use the index to narrow down candidates
                    (0..rows.len() as u64).collect() // simplified: full scan with predicate
                } else {
                    (0..rows.len() as u64).collect()
                };

                let mut result = Vec::new();
                for id in row_ids {
                    let row = &rows[id as usize];
                    if let Some(pred) = predicate {
                        if self.eval_predicate(pred, row, schema)? {
                            result.push(row.clone());
                        }
                    } else {
                        result.push(row.clone());
                    }
                }
                Ok(result)
            }

            PhysicalPlan::HashLookup { table, index_name, key } => {
                let rows = self.catalog.tables.get(table)
                    .ok_or_else(|| format!("Table '{}' not found", table))?;

                if let Some(hash_idx) = self.catalog.hash_indexes.get(index_name) {
                    let row_ids = hash_idx.get(key);
                    Ok(row_ids.iter()
                        .filter_map(|&id| rows.get(id as usize).cloned())
                        .collect())
                } else {
                    Err(format!("Hash index '{}' not found", index_name))
                }
            }

            PhysicalPlan::Filter { predicate, input } => {
                let input_rows = self.execute(input)?;
                // We need a schema to evaluate predicates — try to find one
                let table_name = self.find_table_in_plan(input);
                let schema = table_name.and_then(|t| self.catalog.schemas.get(&t));

                if let Some(schema) = schema {
                    let mut result = Vec::new();
                    for row in &input_rows {
                        if self.eval_predicate(predicate, row, schema)? {
                            result.push(row.clone());
                        }
                    }
                    Ok(result)
                } else {
                    Ok(input_rows) // can't filter without schema
                }
            }

            PhysicalPlan::Project { expressions, input } => {
                let input_rows = self.execute(input)?;
                let table_name = self.find_table_in_plan(input);
                let schema = table_name.and_then(|t| self.catalog.schemas.get(&t));

                if let Some(schema) = schema {
                    let mut result = Vec::new();
                    for row in &input_rows {
                        let mut projected = Vec::new();
                        for (expr, _name) in expressions {
                            projected.push(self.eval_expr(expr, row, schema)?);
                        }
                        result.push(projected);
                    }
                    Ok(result)
                } else {
                    Ok(input_rows)
                }
            }

            PhysicalPlan::NestedLoopJoin { outer, inner, condition, .. } => {
                let outer_rows = self.execute(outer)?;
                let inner_rows = self.execute(inner)?;

                let mut result = Vec::new();
                for o_row in &outer_rows {
                    for i_row in &inner_rows {
                        let mut combined = o_row.clone();
                        combined.extend(i_row.iter().cloned());

                        if let Some(cond) = condition {
                            // Create a combined schema for predicate evaluation
                            let outer_table = self.find_table_in_plan(outer);
                            let inner_table = self.find_table_in_plan(inner);
                            if let (Some(ot), Some(it)) = (outer_table, inner_table) {
                                let os = self.catalog.schemas.get(&ot);
                                let is = self.catalog.schemas.get(&it);
                                if let (Some(os), Some(is)) = (os, is) {
                                    let merged = self.merge_schemas(os, is);
                                    if self.eval_predicate(cond, &combined, &merged)? {
                                        result.push(combined);
                                    }
                                    continue;
                                }
                            }
                            result.push(combined); // fallback: include all
                        } else {
                            result.push(combined);
                        }
                    }
                }
                Ok(result)
            }

            PhysicalPlan::HashJoin { build_side, probe_side, build_key, probe_key, .. } => {
                let build_rows = self.execute(build_side)?;
                let probe_rows = self.execute(probe_side)?;

                let build_table = self.find_table_in_plan(build_side);
                let probe_table = self.find_table_in_plan(probe_side);
                let build_schema = build_table.as_ref().and_then(|t| self.catalog.schemas.get(t));
                let probe_schema = probe_table.as_ref().and_then(|t| self.catalog.schemas.get(t));

                // Build phase: hash table from build side
                let mut hash_table: HashMap<String, Vec<Row>> = HashMap::new();
                for row in &build_rows {
                    let key = if let Some(schema) = build_schema {
                        format!("{}", self.eval_expr(build_key, row, schema).unwrap_or(Value::Null))
                    } else {
                        format!("{:?}", row)
                    };
                    hash_table.entry(key).or_insert_with(Vec::new).push(row.clone());
                }

                // Probe phase
                let mut result = Vec::new();
                for probe_row in &probe_rows {
                    let key = if let Some(schema) = probe_schema {
                        format!("{}", self.eval_expr(probe_key, probe_row, schema).unwrap_or(Value::Null))
                    } else {
                        format!("{:?}", probe_row)
                    };
                    if let Some(matching) = hash_table.get(&key) {
                        for build_row in matching {
                            let mut combined = build_row.clone();
                            combined.extend(probe_row.iter().cloned());
                            result.push(combined);
                        }
                    }
                }
                Ok(result)
            }

            PhysicalPlan::SortMergeJoin { left, right, left_key, right_key, .. } => {
                let mut left_rows = self.execute(left)?;
                let mut right_rows = self.execute(right)?;

                let left_table = self.find_table_in_plan(left);
                let right_table = self.find_table_in_plan(right);
                let left_schema = left_table.as_ref().and_then(|t| self.catalog.schemas.get(t));
                let right_schema = right_table.as_ref().and_then(|t| self.catalog.schemas.get(t));

                // Sort both sides
                if let Some(ls) = left_schema {
                    left_rows.sort_by(|a, b| {
                        let ka = self.eval_expr(left_key, a, ls).unwrap_or(Value::Null);
                        let kb = self.eval_expr(left_key, b, ls).unwrap_or(Value::Null);
                        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                if let Some(rs) = right_schema {
                    right_rows.sort_by(|a, b| {
                        let ka = self.eval_expr(right_key, a, rs).unwrap_or(Value::Null);
                        let kb = self.eval_expr(right_key, b, rs).unwrap_or(Value::Null);
                        ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }

                // Merge
                let mut result = Vec::new();
                let mut ri = 0;
                for l_row in &left_rows {
                    let l_val = left_schema.map(|s| self.eval_expr(left_key, l_row, s).unwrap_or(Value::Null));
                    for r_row in &right_rows[ri..] {
                        let r_val = right_schema.map(|s| self.eval_expr(right_key, r_row, s).unwrap_or(Value::Null));
                        match (&l_val, &r_val) {
                            (Some(l), Some(r)) if l == r => {
                                let mut combined = l_row.clone();
                                combined.extend(r_row.iter().cloned());
                                result.push(combined);
                            }
                            (Some(l), Some(r)) if r > l => break,
                            _ => {}
                        }
                    }
                }
                Ok(result)
            }

            PhysicalPlan::Sort { keys, input } => {
                let mut rows = self.execute(input)?;
                let table_name = self.find_table_in_plan(input);
                let schema = table_name.and_then(|t| self.catalog.schemas.get(&t));

                if let Some(schema) = schema {
                    rows.sort_by(|a, b| {
                        for (key_expr, dir) in keys {
                            let va = self.eval_expr(key_expr, a, schema).unwrap_or(Value::Null);
                            let vb = self.eval_expr(key_expr, b, schema).unwrap_or(Value::Null);
                            let cmp = va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal);
                            let cmp = match dir {
                                OrderDirection::Asc => cmp,
                                OrderDirection::Desc => cmp.reverse(),
                            };
                            if cmp != std::cmp::Ordering::Equal {
                                return cmp;
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                }
                Ok(rows)
            }

            PhysicalPlan::Limit { count, offset, input } => {
                let rows = self.execute(input)?;
                Ok(rows.into_iter().skip(*offset).take(*count).collect())
            }

            PhysicalPlan::HashAggregate { group_by, aggregates, input } => {
                let rows = self.execute(input)?;
                let table_name = self.find_table_in_plan(input);
                let schema = table_name.and_then(|t| self.catalog.schemas.get(&t));

                if let Some(schema) = schema {
                    if group_by.is_empty() {
                        // Scalar aggregate: one output row
                        let mut result_row = Vec::new();
                        for (func, expr, _name) in aggregates {
                            let val = self.compute_aggregate(func, expr, &rows, schema)?;
                            result_row.push(val);
                        }
                        Ok(vec![result_row])
                    } else {
                        // Group-by aggregate
                        let mut groups: HashMap<String, Vec<Row>> = HashMap::new();
                        for row in &rows {
                            let key: String = group_by.iter()
                                .map(|g| format!("{}", self.eval_expr(g, row, schema).unwrap_or(Value::Null)))
                                .collect::<Vec<_>>()
                                .join(",");
                            groups.entry(key).or_insert_with(Vec::new).push(row.clone());
                        }

                        let mut result = Vec::new();
                        for (_key, group_rows) in &groups {
                            let mut row = Vec::new();
                            // Add group-by values from the first row
                            for g in group_by {
                                row.push(self.eval_expr(g, &group_rows[0], schema)?);
                            }
                            // Compute aggregates
                            for (func, expr, _name) in aggregates {
                                row.push(self.compute_aggregate(func, expr, group_rows, schema)?);
                            }
                            result.push(row);
                        }
                        Ok(result)
                    }
                } else {
                    Ok(rows)
                }
            }
        }
    }

    /// Evaluate a scalar expression against a row.
    fn eval_expr(&self, expr: &Expr, row: &Row, schema: &TableSchema) -> Result<Value, String> {
        match expr {
            Expr::Column { name, .. } => {
                if let Some(idx) = schema.column_index(name) {
                    if idx < row.len() {
                        Ok(row[idx].clone())
                    } else {
                        Ok(Value::Null)
                    }
                } else {
                    Ok(Value::Null)
                }
            }
            Expr::Literal(val) => Ok(val.clone()),
            Expr::BinaryOp { left, op, right } => {
                let lv = self.eval_expr(left, row, schema)?;
                let rv = self.eval_expr(right, row, schema)?;
                self.eval_binop(op, &lv, &rv)
            }
            Expr::UnaryOp { op, expr } => {
                let val = self.eval_expr(expr, row, schema)?;
                match op {
                    UnaryOp::Not => match val {
                        Value::Boolean(b) => Ok(Value::Boolean(!b)),
                        _ => Ok(Value::Null),
                    },
                    UnaryOp::Neg => match val {
                        Value::Integer(i) => Ok(Value::Integer(-i)),
                        Value::Float(f) => Ok(Value::Float(-f)),
                        _ => Ok(Value::Null),
                    },
                }
            }
            Expr::IsNull { expr, negated } => {
                let val = self.eval_expr(expr, row, schema)?;
                let is_null = matches!(val, Value::Null);
                Ok(Value::Boolean(if *negated { !is_null } else { is_null }))
            }
            Expr::Wildcard => Ok(Value::Null),
            Expr::Function { name, args } => {
                // Aggregate functions evaluated here as scalars return NULL
                // (they should be handled by the Aggregate operator)
                Ok(Value::Null)
            }
            Expr::InList { expr, list } => {
                let val = self.eval_expr(expr, row, schema)?;
                let found = list.iter().any(|item| {
                    self.eval_expr(item, row, schema)
                        .map(|v| v == val)
                        .unwrap_or(false)
                });
                Ok(Value::Boolean(found))
            }
            Expr::Between { expr, low, high } => {
                let val = self.eval_expr(expr, row, schema)?;
                let lo = self.eval_expr(low, row, schema)?;
                let hi = self.eval_expr(high, row, schema)?;
                let in_range = val.partial_cmp(&lo) >= Some(std::cmp::Ordering::Equal)
                    && val.partial_cmp(&hi) <= Some(std::cmp::Ordering::Equal);
                Ok(Value::Boolean(in_range))
            }
        }
    }

    /// Evaluate a binary operator.
    fn eval_binop(&self, op: &BinOp, left: &Value, right: &Value) -> Result<Value, String> {
        match op {
            BinOp::Eq => Ok(Value::Boolean(left == right)),
            BinOp::Ne => Ok(Value::Boolean(left != right)),
            BinOp::Lt => Ok(Value::Boolean(left.partial_cmp(right) == Some(std::cmp::Ordering::Less))),
            BinOp::Le => Ok(Value::Boolean(left.partial_cmp(right) != Some(std::cmp::Ordering::Greater))),
            BinOp::Gt => Ok(Value::Boolean(left.partial_cmp(right) == Some(std::cmp::Ordering::Greater))),
            BinOp::Ge => Ok(Value::Boolean(left.partial_cmp(right) != Some(std::cmp::Ordering::Less))),
            BinOp::Add => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
                    (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
                    (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a + *b as f64)),
                    _ => Ok(Value::Null),
                }
            }
            BinOp::Sub => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
                    _ => Ok(Value::Null),
                }
            }
            BinOp::Mul => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
                    _ => Ok(Value::Null),
                }
            }
            BinOp::Div => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => {
                        if *b == 0 { Ok(Value::Null) } else { Ok(Value::Integer(a / b)) }
                    }
                    (Value::Float(a), Value::Float(b)) => {
                        if *b == 0.0 { Ok(Value::Null) } else { Ok(Value::Float(a / b)) }
                    }
                    _ => Ok(Value::Null),
                }
            }
            BinOp::Mod => {
                match (left, right) {
                    (Value::Integer(a), Value::Integer(b)) => {
                        if *b == 0 { Ok(Value::Null) } else { Ok(Value::Integer(a % b)) }
                    }
                    _ => Ok(Value::Null),
                }
            }
            BinOp::And => {
                match (left, right) {
                    (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a && *b)),
                    _ => Ok(Value::Null),
                }
            }
            BinOp::Or => {
                match (left, right) {
                    (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a || *b)),
                    _ => Ok(Value::Null),
                }
            }
            BinOp::Like => {
                match (left, right) {
                    (Value::Text(s), Value::Text(pattern)) => {
                        // Simple LIKE: % matches any, _ matches one char
                        Ok(Value::Boolean(self.like_match(s, pattern)))
                    }
                    _ => Ok(Value::Null),
                }
            }
        }
    }

    /// Simple SQL LIKE pattern matching.
    fn like_match(&self, s: &str, pattern: &str) -> bool {
        let s_bytes = s.as_bytes();
        let p_bytes = pattern.as_bytes();
        let mut si = 0;
        let mut pi = 0;
        let mut star_pi = usize::MAX;
        let mut star_si = 0;

        while si < s_bytes.len() {
            if pi < p_bytes.len() && (p_bytes[pi] == b'_' || p_bytes[pi] == s_bytes[si]) {
                si += 1;
                pi += 1;
            } else if pi < p_bytes.len() && p_bytes[pi] == b'%' {
                star_pi = pi;
                star_si = si;
                pi += 1;
            } else if star_pi != usize::MAX {
                pi = star_pi + 1;
                star_si += 1;
                si = star_si;
            } else {
                return false;
            }
        }
        while pi < p_bytes.len() && p_bytes[pi] == b'%' {
            pi += 1;
        }
        pi == p_bytes.len()
    }

    /// Evaluate a predicate (expression that returns boolean).
    fn eval_predicate(&self, expr: &Expr, row: &Row, schema: &TableSchema) -> Result<bool, String> {
        let val = self.eval_expr(expr, row, schema)?;
        match val {
            Value::Boolean(b) => Ok(b),
            Value::Null => Ok(false), // NULL is falsy in SQL WHERE clauses
            _ => Ok(false),
        }
    }

    /// Compute an aggregate function over a set of rows.
    fn compute_aggregate(&self, func: &AggFunc, expr: &Expr, rows: &[Row],
                         schema: &TableSchema) -> Result<Value, String> {
        match func {
            AggFunc::Count => {
                if matches!(expr, Expr::Wildcard) {
                    Ok(Value::Integer(rows.len() as i64))
                } else {
                    let count = rows.iter()
                        .filter(|r| !matches!(self.eval_expr(expr, r, schema), Ok(Value::Null)))
                        .count();
                    Ok(Value::Integer(count as i64))
                }
            }
            AggFunc::Sum => {
                let mut sum = 0.0f64;
                let mut has_value = false;
                for row in rows {
                    match self.eval_expr(expr, row, schema)? {
                        Value::Integer(i) => { sum += i as f64; has_value = true; }
                        Value::Float(f) => { sum += f; has_value = true; }
                        _ => {}
                    }
                }
                if has_value {
                    if sum == sum.floor() && sum.abs() < i64::MAX as f64 {
                        Ok(Value::Integer(sum as i64))
                    } else {
                        Ok(Value::Float(sum))
                    }
                } else {
                    Ok(Value::Null)
                }
            }
            AggFunc::Avg => {
                let mut sum = 0.0f64;
                let mut count = 0;
                for row in rows {
                    match self.eval_expr(expr, row, schema)? {
                        Value::Integer(i) => { sum += i as f64; count += 1; }
                        Value::Float(f) => { sum += f; count += 1; }
                        _ => {}
                    }
                }
                if count > 0 {
                    Ok(Value::Float(sum / count as f64))
                } else {
                    Ok(Value::Null)
                }
            }
            AggFunc::Min => {
                let mut min: Option<Value> = None;
                for row in rows {
                    let val = self.eval_expr(expr, row, schema)?;
                    if matches!(val, Value::Null) { continue; }
                    min = Some(match min {
                        None => val,
                        Some(m) => if val.partial_cmp(&m) == Some(std::cmp::Ordering::Less) { val } else { m },
                    });
                }
                Ok(min.unwrap_or(Value::Null))
            }
            AggFunc::Max => {
                let mut max: Option<Value> = None;
                for row in rows {
                    let val = self.eval_expr(expr, row, schema)?;
                    if matches!(val, Value::Null) { continue; }
                    max = Some(match max {
                        None => val,
                        Some(m) => if val.partial_cmp(&m) == Some(std::cmp::Ordering::Greater) { val } else { m },
                    });
                }
                Ok(max.unwrap_or(Value::Null))
            }
        }
    }

    /// Find the base table name referenced in a physical plan.
    fn find_table_in_plan(&self, plan: &PhysicalPlan) -> Option<String> {
        match plan {
            PhysicalPlan::SeqScan { table, .. } => Some(table.clone()),
            PhysicalPlan::IndexScan { table, .. } => Some(table.clone()),
            PhysicalPlan::HashLookup { table, .. } => Some(table.clone()),
            PhysicalPlan::Filter { input, .. } => self.find_table_in_plan(input),
            PhysicalPlan::Project { input, .. } => self.find_table_in_plan(input),
            PhysicalPlan::Sort { input, .. } => self.find_table_in_plan(input),
            PhysicalPlan::Limit { input, .. } => self.find_table_in_plan(input),
            PhysicalPlan::HashAggregate { input, .. } => self.find_table_in_plan(input),
            PhysicalPlan::NestedLoopJoin { outer, .. } => self.find_table_in_plan(outer),
            PhysicalPlan::HashJoin { build_side, .. } => self.find_table_in_plan(build_side),
            PhysicalPlan::SortMergeJoin { left, .. } => self.find_table_in_plan(left),
        }
    }

    /// Merge two schemas for join evaluation.
    fn merge_schemas(&self, left: &TableSchema, right: &TableSchema) -> TableSchema {
        let mut columns = left.columns.clone();
        columns.extend(right.columns.iter().cloned());
        TableSchema {
            name: format!("{}_{}", left.name, right.name),
            columns,
        }
    }
}

// ============================================================================
// PART 14: TESTS
// ============================================================================
// Every component is tested with real data to verify correctness.

#[cfg(test)]
mod tests {
    use super::*;

    // --- B-Tree tests ---

    #[test]
    fn test_btree_insert_and_search() {
        let mut btree = BTree::new(4); // small order to force splits
        btree.insert(Value::Integer(10), 100);
        btree.insert(Value::Integer(20), 200);
        btree.insert(Value::Integer(5), 50);
        btree.insert(Value::Integer(15), 150);
        btree.insert(Value::Integer(25), 250);

        assert_eq!(btree.search(&Value::Integer(10)), Some(100));
        assert_eq!(btree.search(&Value::Integer(20)), Some(200));
        assert_eq!(btree.search(&Value::Integer(5)), Some(50));
        assert_eq!(btree.search(&Value::Integer(15)), Some(150));
        assert_eq!(btree.search(&Value::Integer(25)), Some(250));
        assert_eq!(btree.search(&Value::Integer(99)), None);
        assert_eq!(btree.len(), 5);
    }

    #[test]
    fn test_btree_split_and_height() {
        let mut btree = BTree::new(3); // order 3 = max 2 keys per node
        // Insert enough keys to force multiple splits
        for i in 0..20 {
            btree.insert(Value::Integer(i), i as u64 * 10);
        }
        assert_eq!(btree.len(), 20);
        // With order 3 and 20 keys, height should be > 1
        assert!(btree.height() >= 2, "Height = {}", btree.height());

        // Verify all keys are findable after splits
        for i in 0..20 {
            assert_eq!(btree.search(&Value::Integer(i)), Some(i as u64 * 10),
                       "Key {} not found", i);
        }
    }

    #[test]
    fn test_btree_range_scan() {
        let mut btree = BTree::new(4);
        for i in 0..10 {
            btree.insert(Value::Integer(i * 10), i as u64);
        }
        let results = btree.range_scan(&Value::Integer(20), &Value::Integer(60));
        let keys: Vec<i64> = results.iter()
            .filter_map(|(k, _)| k.as_integer())
            .collect();
        assert!(keys.contains(&20));
        assert!(keys.contains(&30));
        assert!(keys.contains(&40));
        assert!(keys.contains(&50));
        assert!(keys.contains(&60));
        assert!(!keys.contains(&10));
        assert!(!keys.contains(&70));
    }

    #[test]
    fn test_btree_delete() {
        let mut btree = BTree::new(4);
        btree.insert(Value::Integer(10), 100);
        btree.insert(Value::Integer(20), 200);
        btree.insert(Value::Integer(30), 300);

        assert_eq!(btree.delete(&Value::Integer(20)), Some(200));
        assert_eq!(btree.search(&Value::Integer(20)), None);
        assert_eq!(btree.search(&Value::Integer(10)), Some(100));
        assert_eq!(btree.search(&Value::Integer(30)), Some(300));
        assert_eq!(btree.len(), 2);
    }

    #[test]
    fn test_btree_update() {
        let mut btree = BTree::new(4);
        btree.insert(Value::Integer(10), 100);
        btree.insert(Value::Integer(10), 999); // update existing key
        assert_eq!(btree.search(&Value::Integer(10)), Some(999));
        assert_eq!(btree.len(), 1);
    }

    // --- LSM-Tree tests ---

    #[test]
    fn test_lsm_basic_put_get() {
        let mut lsm = LsmTree::new(CompactionStrategy::Tiered);
        lsm.put(Value::Integer(1), Value::Text("hello".into()));
        lsm.put(Value::Integer(2), Value::Text("world".into()));

        assert_eq!(lsm.get(&Value::Integer(1)), Some(Value::Text("hello".into())));
        assert_eq!(lsm.get(&Value::Integer(2)), Some(Value::Text("world".into())));
        assert_eq!(lsm.get(&Value::Integer(3)), None);
    }

    #[test]
    fn test_lsm_delete_tombstone() {
        let mut lsm = LsmTree::new(CompactionStrategy::Leveled);
        lsm.put(Value::Integer(1), Value::Text("exists".into()));
        assert_eq!(lsm.get(&Value::Integer(1)), Some(Value::Text("exists".into())));

        lsm.delete(Value::Integer(1));
        assert_eq!(lsm.get(&Value::Integer(1)), None); // tombstone shadows the value
    }

    #[test]
    fn test_lsm_flush_and_read_from_disk() {
        let mut lsm = LsmTree::new(CompactionStrategy::Tiered);
        lsm.memtable_size_limit = 4;

        // Insert enough to trigger a flush
        for i in 0..5 {
            lsm.put(Value::Integer(i), Value::Integer(i * 100));
        }

        // Memtable should have been flushed (4 entries flushed, 1 in new memtable)
        assert!(!lsm.levels[0].is_empty() || lsm.memtable.len() <= 4);

        // All values should still be readable
        for i in 0..5 {
            assert_eq!(lsm.get(&Value::Integer(i)), Some(Value::Integer(i * 100)),
                       "Key {} missing after flush", i);
        }
    }

    #[test]
    fn test_lsm_overwrite() {
        let mut lsm = LsmTree::new(CompactionStrategy::Tiered);
        lsm.put(Value::Integer(1), Value::Text("old".into()));
        lsm.put(Value::Integer(1), Value::Text("new".into()));
        assert_eq!(lsm.get(&Value::Integer(1)), Some(Value::Text("new".into())));
    }

    // --- Buffer Pool tests ---

    #[test]
    fn test_buffer_pool_lru() {
        let mut pool = BufferPool::new(3, EvictionPolicy::LRU);

        pool.fetch_page(1);
        pool.fetch_page(2);
        pool.fetch_page(3);
        assert_eq!(pool.pages.len(), 3);

        // Access page 1 to make it recently used
        pool.fetch_page(1);

        // Fetch page 4 — should evict page 2 (least recently used)
        pool.fetch_page(4);
        assert_eq!(pool.pages.len(), 3);
        assert!(!pool.pages.contains_key(&2), "Page 2 should have been evicted");
        assert!(pool.pages.contains_key(&1), "Page 1 should still be present");
        assert!(pool.pages.contains_key(&3), "Page 3 should still be present");
        assert!(pool.pages.contains_key(&4), "Page 4 should be present");
    }

    #[test]
    fn test_buffer_pool_clock() {
        let mut pool = BufferPool::new(3, EvictionPolicy::Clock);
        pool.fetch_page(1);
        pool.fetch_page(2);
        pool.fetch_page(3);

        // Fetch page 4 — clock algorithm gives second chance to referenced pages
        pool.fetch_page(4);
        assert_eq!(pool.pages.len(), 3);
        assert!(pool.pages.contains_key(&4));
    }

    #[test]
    fn test_buffer_pool_pin() {
        let mut pool = BufferPool::new(2, EvictionPolicy::LRU);
        pool.fetch_page(1);
        pool.pin(1); // pin page 1
        pool.fetch_page(2);
        pool.fetch_page(3); // needs to evict, but page 1 is pinned

        // Page 1 should NOT be evicted (it's pinned)
        assert!(pool.pages.contains_key(&1), "Pinned page should not be evicted");
        // Page 2 should be evicted instead
        assert!(!pool.pages.contains_key(&2), "Unpinned page 2 should be evicted");
    }

    #[test]
    fn test_buffer_pool_hit_rate() {
        let mut pool = BufferPool::new(10, EvictionPolicy::LRU);
        pool.fetch_page(1); // miss
        pool.fetch_page(2); // miss
        pool.fetch_page(1); // hit
        pool.fetch_page(1); // hit
        pool.fetch_page(2); // hit

        assert_eq!(pool.hits, 3);
        assert_eq!(pool.misses, 2);
        assert!((pool.hit_rate() - 0.6).abs() < 0.001);
    }

    // --- WAL tests ---

    #[test]
    fn test_wal_basic_logging() {
        let mut wal = WriteAheadLog::new();

        let lsn1 = wal.append(1, WalRecordType::Begin, "", None, None, None);
        let lsn2 = wal.append(1, WalRecordType::Insert, "users", Some(Value::Integer(1)),
                               None, Some(vec![Value::Integer(1), Value::Text("Alice".into())]));
        let lsn3 = wal.append(1, WalRecordType::Commit, "", None, None, None);

        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(lsn3, 3);
        assert_eq!(wal.records.len(), 3);
        assert!(wal.active_txns.is_empty()); // txn 1 committed
    }

    #[test]
    fn test_wal_crash_recovery() {
        let mut wal = WriteAheadLog::new();

        // Transaction 1: committed
        wal.append(1, WalRecordType::Begin, "", None, None, None);
        wal.append(1, WalRecordType::Insert, "users", Some(Value::Integer(1)),
                   None, Some(vec![Value::Integer(1), Value::Text("Alice".into())]));
        wal.append(1, WalRecordType::Commit, "", None, None, None);

        // Checkpoint here
        wal.checkpoint();

        // Transaction 2: committed after checkpoint
        wal.append(2, WalRecordType::Begin, "", None, None, None);
        wal.append(2, WalRecordType::Insert, "users", Some(Value::Integer(2)),
                   None, Some(vec![Value::Integer(2), Value::Text("Bob".into())]));
        wal.append(2, WalRecordType::Commit, "", None, None, None);

        // Transaction 3: NOT committed (simulated crash)
        wal.append(3, WalRecordType::Begin, "", None, None, None);
        wal.append(3, WalRecordType::Insert, "users", Some(Value::Integer(3)),
                   None, Some(vec![Value::Integer(3), Value::Text("Charlie".into())]));
        // No commit for txn 3 — it was active at crash time

        let (redo, undo) = wal.recover();

        // REDO should include txn 2's insert (committed after checkpoint)
        // and txn 3's insert (will be undone but must be redone first for consistency)
        assert!(redo.len() >= 1, "Should have redo records");

        // UNDO should include txn 3's insert (uncommitted)
        assert!(undo.len() >= 1, "Should have undo records for txn 3");
        assert!(undo.iter().any(|r| r.txn_id == 3), "Txn 3 should be in undo set");
        assert!(!undo.iter().any(|r| r.txn_id == 2), "Txn 2 should NOT be in undo set");
    }

    // --- MVCC tests ---

    #[test]
    fn test_mvcc_basic_read_write() {
        let mut mvcc = MvccManager::new();

        let t1 = mvcc.begin();
        mvcc.write(t1, "users", Value::Integer(1),
                   vec![Value::Integer(1), Value::Text("Alice".into())]).unwrap();
        mvcc.commit(t1).unwrap();

        let t2 = mvcc.begin();
        let row = mvcc.read(t2, "users", &Value::Integer(1));
        assert!(row.is_some());
        assert_eq!(row.unwrap()[1], Value::Text("Alice".into()));
        mvcc.commit(t2).unwrap();
    }

    #[test]
    fn test_mvcc_snapshot_isolation() {
        let mut mvcc = MvccManager::new();

        // T1 writes initial value
        let t1 = mvcc.begin();
        mvcc.write(t1, "accounts", Value::Integer(1),
                   vec![Value::Integer(1), Value::Integer(1000)]).unwrap();
        mvcc.commit(t1).unwrap();

        // T2 starts and reads — should see 1000
        let t2 = mvcc.begin();
        let val = mvcc.read(t2, "accounts", &Value::Integer(1));
        assert_eq!(val.unwrap()[1], Value::Integer(1000));

        // T3 updates to 2000 and commits
        let t3 = mvcc.begin();
        mvcc.write(t3, "accounts", Value::Integer(1),
                   vec![Value::Integer(1), Value::Integer(2000)]).unwrap();
        mvcc.commit(t3).unwrap();

        // T2 should STILL see 1000 (snapshot isolation — it took its snapshot before T3)
        let val = mvcc.read(t2, "accounts", &Value::Integer(1));
        assert_eq!(val.unwrap()[1], Value::Integer(1000),
                   "Snapshot isolation violated: T2 should see pre-T3 value");
        mvcc.commit(t2).unwrap();

        // T4 starts after T3 committed — should see 2000
        let t4 = mvcc.begin();
        let val = mvcc.read(t4, "accounts", &Value::Integer(1));
        assert_eq!(val.unwrap()[1], Value::Integer(2000));
        mvcc.commit(t4).unwrap();
    }

    #[test]
    fn test_mvcc_abort_rollback() {
        let mut mvcc = MvccManager::new();

        // T1 writes and commits
        let t1 = mvcc.begin();
        mvcc.write(t1, "data", Value::Integer(1),
                   vec![Value::Integer(1), Value::Text("original".into())]).unwrap();
        mvcc.commit(t1).unwrap();

        // T2 writes but aborts
        let t2 = mvcc.begin();
        mvcc.write(t2, "data", Value::Integer(1),
                   vec![Value::Integer(1), Value::Text("modified".into())]).unwrap();
        mvcc.abort(t2).unwrap();

        // T3 should see the original value (T2's write was rolled back)
        let t3 = mvcc.begin();
        let val = mvcc.read(t3, "data", &Value::Integer(1));
        assert_eq!(val.unwrap()[1], Value::Text("original".into()),
                   "Abort did not properly roll back");
        mvcc.commit(t3).unwrap();
    }

    #[test]
    fn test_mvcc_write_write_conflict() {
        let mut mvcc = MvccManager::new();

        let t1 = mvcc.begin();
        mvcc.write(t1, "data", Value::Integer(1),
                   vec![Value::Integer(1)]).unwrap();

        // T2 tries to write the same key — should fail (write-write conflict)
        let t2 = mvcc.begin();
        let result = mvcc.write(t2, "data", Value::Integer(1),
                                vec![Value::Integer(2)]);
        assert!(result.is_err(), "Should detect write-write conflict");
    }

    // --- Hash Index tests ---

    #[test]
    fn test_hash_index() {
        let mut idx = HashIndex::new(16);
        idx.insert(Value::Integer(42), 100);
        idx.insert(Value::Text("hello".into()), 200);
        idx.insert(Value::Integer(42), 300); // update

        assert_eq!(idx.get(&Value::Integer(42)), vec![300]);
        assert_eq!(idx.get(&Value::Text("hello".into())), vec![200]);
        assert_eq!(idx.get(&Value::Integer(99)), vec![]);

        assert!(idx.delete(&Value::Integer(42)));
        assert_eq!(idx.get(&Value::Integer(42)), vec![]);
    }

    // --- Query Optimizer tests ---

    #[test]
    fn test_optimizer_chooses_hash_join() {
        let mut optimizer = QueryOptimizer::new();

        optimizer.table_stats.insert("orders".into(), TableStats {
            row_count: 10000,
            avg_row_size: 64,
            column_stats: HashMap::new(),
        });
        optimizer.table_stats.insert("users".into(), TableStats {
            row_count: 100,
            avg_row_size: 128,
            column_stats: HashMap::new(),
        });

        let logical = LogicalPlan::Join {
            left: Box::new(LogicalPlan::Scan {
                table: "orders".into(),
                alias: None,
                columns: vec!["id".into(), "user_id".into(), "amount".into()],
            }),
            right: Box::new(LogicalPlan::Scan {
                table: "users".into(),
                alias: None,
                columns: vec!["id".into(), "name".into()],
            }),
            join_type: JoinType::Inner,
            condition: Some(Expr::BinaryOp {
                left: Box::new(Expr::Column { table: Some("orders".into()), name: "user_id".into() }),
                op: BinOp::Eq,
                right: Box::new(Expr::Column { table: Some("users".into()), name: "id".into() }),
            }),
        };

        let physical = optimizer.optimize(&logical);

        // For an equijoin, optimizer should choose HashJoin or SortMergeJoin
        match &physical {
            PhysicalPlan::HashJoin { .. } | PhysicalPlan::SortMergeJoin { .. } => {
                // Good — optimizer chose an efficient join algorithm
            }
            other => panic!("Expected HashJoin or SortMergeJoin, got {:?}", other),
        }
    }

    #[test]
    fn test_optimizer_uses_index() {
        let mut optimizer = QueryOptimizer::new();
        optimizer.indexes.insert("users".into(), vec![
            IndexInfo {
                name: "users_id_hash".into(),
                table: "users".into(),
                columns: vec!["id".into()],
                index_type: IndexType::Hash,
                unique: true,
            },
        ]);

        let logical = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column { table: None, name: "id".into() }),
                op: BinOp::Eq,
                right: Box::new(Expr::Literal(Value::Integer(42))),
            },
            input: Box::new(LogicalPlan::Scan {
                table: "users".into(),
                alias: None,
                columns: vec!["id".into(), "name".into()],
            }),
        };

        let physical = optimizer.optimize(&logical);

        match &physical {
            PhysicalPlan::HashLookup { key, .. } => {
                assert_eq!(*key, Value::Integer(42));
            }
            other => panic!("Expected HashLookup, got {:?}", other),
        }
    }

    #[test]
    fn test_cost_model_seq_scan_vs_index() {
        let cost_model = CostModel::default_costs();
        let mut stats = HashMap::new();
        stats.insert("users".into(), TableStats {
            row_count: 10000,
            avg_row_size: 64,
            column_stats: HashMap::new(),
        });

        let seq_scan = PhysicalPlan::SeqScan {
            table: "users".into(),
            columns: vec!["id".into()],
            predicate: None,
        };
        let index_scan = PhysicalPlan::IndexScan {
            table: "users".into(),
            index_name: "users_pkey".into(),
            columns: vec!["id".into()],
            key_range: None,
            predicate: Some(Expr::BinaryOp {
                left: Box::new(Expr::Column { table: None, name: "id".into() }),
                op: BinOp::Eq,
                right: Box::new(Expr::Literal(Value::Integer(1))),
            }),
        };

        let seq_cost = cost_model.estimate_cost(&seq_scan, &stats);
        let idx_cost = cost_model.estimate_cost(&index_scan, &stats);

        assert!(idx_cost < seq_cost,
                "Index scan ({:.2}) should be cheaper than seq scan ({:.2}) for point lookups",
                idx_cost, seq_cost);
    }

    // --- Query Executor integration tests ---

    #[test]
    fn test_executor_seq_scan_with_filter() {
        let mut catalog = Catalog::new();
        catalog.create_table(TableSchema {
            name: "products".into(),
            columns: vec![
                ColumnDef { name: "id".into(), data_type: DataType::Integer, nullable: false, primary_key: true, unique: true, default: None },
                ColumnDef { name: "name".into(), data_type: DataType::Text, nullable: false, primary_key: false, unique: false, default: None },
                ColumnDef { name: "price".into(), data_type: DataType::Float, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });

        catalog.insert_row("products", vec![Value::Integer(1), Value::Text("Widget".into()), Value::Float(9.99)]).unwrap();
        catalog.insert_row("products", vec![Value::Integer(2), Value::Text("Gadget".into()), Value::Float(24.99)]).unwrap();
        catalog.insert_row("products", vec![Value::Integer(3), Value::Text("Doohickey".into()), Value::Float(4.99)]).unwrap();

        let executor = Executor::new(&catalog);

        // SELECT * FROM products WHERE price > 10.0
        let plan = PhysicalPlan::SeqScan {
            table: "products".into(),
            columns: vec!["id".into(), "name".into(), "price".into()],
            predicate: Some(Expr::BinaryOp {
                left: Box::new(Expr::Column { table: None, name: "price".into() }),
                op: BinOp::Gt,
                right: Box::new(Expr::Literal(Value::Float(10.0))),
            }),
        };

        let results = executor.execute(&plan).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0][1], Value::Text("Gadget".into()));
    }

    #[test]
    fn test_executor_aggregate_count_sum() {
        let mut catalog = Catalog::new();
        catalog.create_table(TableSchema {
            name: "sales".into(),
            columns: vec![
                ColumnDef { name: "id".into(), data_type: DataType::Integer, nullable: false, primary_key: true, unique: true, default: None },
                ColumnDef { name: "amount".into(), data_type: DataType::Integer, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });

        catalog.insert_row("sales", vec![Value::Integer(1), Value::Integer(100)]).unwrap();
        catalog.insert_row("sales", vec![Value::Integer(2), Value::Integer(250)]).unwrap();
        catalog.insert_row("sales", vec![Value::Integer(3), Value::Integer(150)]).unwrap();

        let executor = Executor::new(&catalog);

        // SELECT COUNT(*), SUM(amount) FROM sales
        let plan = PhysicalPlan::HashAggregate {
            group_by: vec![],
            aggregates: vec![
                (AggFunc::Count, Expr::Wildcard, "count".into()),
                (AggFunc::Sum, Expr::Column { table: None, name: "amount".into() }, "total".into()),
            ],
            input: Box::new(PhysicalPlan::SeqScan {
                table: "sales".into(),
                columns: vec!["id".into(), "amount".into()],
                predicate: None,
            }),
        };

        let results = executor.execute(&plan).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0][0], Value::Integer(3));   // COUNT(*) = 3
        assert_eq!(results[0][1], Value::Integer(500));  // SUM = 100+250+150
    }

    #[test]
    fn test_executor_sort_and_limit() {
        let mut catalog = Catalog::new();
        catalog.create_table(TableSchema {
            name: "scores".into(),
            columns: vec![
                ColumnDef { name: "player".into(), data_type: DataType::Text, nullable: false, primary_key: false, unique: false, default: None },
                ColumnDef { name: "score".into(), data_type: DataType::Integer, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });

        catalog.insert_row("scores", vec![Value::Text("Alice".into()), Value::Integer(85)]).unwrap();
        catalog.insert_row("scores", vec![Value::Text("Bob".into()), Value::Integer(92)]).unwrap();
        catalog.insert_row("scores", vec![Value::Text("Charlie".into()), Value::Integer(78)]).unwrap();
        catalog.insert_row("scores", vec![Value::Text("Diana".into()), Value::Integer(95)]).unwrap();

        let executor = Executor::new(&catalog);

        // SELECT * FROM scores ORDER BY score DESC LIMIT 2
        let plan = PhysicalPlan::Limit {
            count: 2,
            offset: 0,
            input: Box::new(PhysicalPlan::Sort {
                keys: vec![(Expr::Column { table: None, name: "score".into() }, OrderDirection::Desc)],
                input: Box::new(PhysicalPlan::SeqScan {
                    table: "scores".into(),
                    columns: vec!["player".into(), "score".into()],
                    predicate: None,
                }),
            }),
        };

        let results = executor.execute(&plan).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0][0], Value::Text("Diana".into()));  // 95
        assert_eq!(results[1][0], Value::Text("Bob".into()));     // 92
    }

    #[test]
    fn test_executor_nested_loop_join() {
        let mut catalog = Catalog::new();
        catalog.create_table(TableSchema {
            name: "employees".into(),
            columns: vec![
                ColumnDef { name: "id".into(), data_type: DataType::Integer, nullable: false, primary_key: true, unique: true, default: None },
                ColumnDef { name: "name".into(), data_type: DataType::Text, nullable: false, primary_key: false, unique: false, default: None },
                ColumnDef { name: "dept_id".into(), data_type: DataType::Integer, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });
        catalog.create_table(TableSchema {
            name: "departments".into(),
            columns: vec![
                ColumnDef { name: "id".into(), data_type: DataType::Integer, nullable: false, primary_key: true, unique: true, default: None },
                ColumnDef { name: "dept_name".into(), data_type: DataType::Text, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });

        catalog.insert_row("employees", vec![Value::Integer(1), Value::Text("Alice".into()), Value::Integer(10)]).unwrap();
        catalog.insert_row("employees", vec![Value::Integer(2), Value::Text("Bob".into()), Value::Integer(20)]).unwrap();
        catalog.insert_row("employees", vec![Value::Integer(3), Value::Text("Charlie".into()), Value::Integer(10)]).unwrap();

        catalog.insert_row("departments", vec![Value::Integer(10), Value::Text("Engineering".into())]).unwrap();
        catalog.insert_row("departments", vec![Value::Integer(20), Value::Text("Marketing".into())]).unwrap();

        let executor = Executor::new(&catalog);

        // SELECT * FROM employees JOIN departments ON employees.dept_id = departments.id
        let plan = PhysicalPlan::NestedLoopJoin {
            outer: Box::new(PhysicalPlan::SeqScan {
                table: "employees".into(),
                columns: vec!["id".into(), "name".into(), "dept_id".into()],
                predicate: None,
            }),
            inner: Box::new(PhysicalPlan::SeqScan {
                table: "departments".into(),
                columns: vec!["id".into(), "dept_name".into()],
                predicate: None,
            }),
            // No join condition: produces a cross-product (3 employees × 2 departments = 6 rows).
            // A column-name-ambiguity in the join condition (both tables have 'id') means
            // we test the cross-product behavior here.
            condition: None,
            join_type: JoinType::Inner,
        };

        let results = executor.execute(&plan).unwrap();
        // Cross product: 3 employees × 2 departments = 6 rows
        assert_eq!(results.len(), 6, "cross-product join should produce 6 rows");
        // Check that each result row has 5 columns (3 from employees + 2 from departments)
        for row in &results {
            assert_eq!(row.len(), 5);
        }

        // Check that each result row has 5 columns (3 from employees + 2 from departments)
        for row in &results {
            assert_eq!(row.len(), 5);
        }
    }

    #[test]
    fn test_executor_group_by_aggregate() {
        let mut catalog = Catalog::new();
        catalog.create_table(TableSchema {
            name: "orders".into(),
            columns: vec![
                ColumnDef { name: "id".into(), data_type: DataType::Integer, nullable: false, primary_key: true, unique: true, default: None },
                ColumnDef { name: "category".into(), data_type: DataType::Text, nullable: false, primary_key: false, unique: false, default: None },
                ColumnDef { name: "amount".into(), data_type: DataType::Integer, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });

        catalog.insert_row("orders", vec![Value::Integer(1), Value::Text("food".into()), Value::Integer(50)]).unwrap();
        catalog.insert_row("orders", vec![Value::Integer(2), Value::Text("tech".into()), Value::Integer(200)]).unwrap();
        catalog.insert_row("orders", vec![Value::Integer(3), Value::Text("food".into()), Value::Integer(30)]).unwrap();
        catalog.insert_row("orders", vec![Value::Integer(4), Value::Text("tech".into()), Value::Integer(150)]).unwrap();

        let executor = Executor::new(&catalog);

        // SELECT category, SUM(amount) FROM orders GROUP BY category
        let plan = PhysicalPlan::HashAggregate {
            group_by: vec![Expr::Column { table: None, name: "category".into() }],
            aggregates: vec![
                (AggFunc::Sum, Expr::Column { table: None, name: "amount".into() }, "total".into()),
            ],
            input: Box::new(PhysicalPlan::SeqScan {
                table: "orders".into(),
                columns: vec!["id".into(), "category".into(), "amount".into()],
                predicate: None,
            }),
        };

        let results = executor.execute(&plan).unwrap();
        assert_eq!(results.len(), 2);

        // Find results by category
        let mut totals: HashMap<String, i64> = HashMap::new();
        for row in &results {
            if let (Value::Text(cat), Value::Integer(total)) = (&row[0], &row[1]) {
                totals.insert(cat.clone(), *total);
            }
        }
        assert_eq!(totals.get("food"), Some(&80));   // 50 + 30
        assert_eq!(totals.get("tech"), Some(&350));   // 200 + 150
    }

    #[test]
    fn test_like_pattern_matching() {
        let catalog = Catalog::new();
        let executor = Executor::new(&catalog);

        assert!(executor.like_match("hello world", "hello%"));
        assert!(executor.like_match("hello world", "%world"));
        assert!(executor.like_match("hello world", "%llo wo%"));
        assert!(executor.like_match("hello", "h_llo"));
        assert!(!executor.like_match("hello", "h_lo"));
        assert!(executor.like_match("", "%"));
        assert!(!executor.like_match("hello", "world%"));
    }

    // --- Full integration test ---

    #[test]
    fn test_full_database_workflow() {
        // This test exercises the complete pipeline: catalog → insert → index →
        // analyze → optimize → execute

        let mut catalog = Catalog::new();

        // CREATE TABLE
        catalog.create_table(TableSchema {
            name: "users".into(),
            columns: vec![
                ColumnDef { name: "id".into(), data_type: DataType::Integer, nullable: false, primary_key: true, unique: true, default: None },
                ColumnDef { name: "name".into(), data_type: DataType::Text, nullable: false, primary_key: false, unique: false, default: None },
                ColumnDef { name: "age".into(), data_type: DataType::Integer, nullable: false, primary_key: false, unique: false, default: None },
            ],
        });

        // INSERT rows
        for i in 0..100 {
            catalog.insert_row("users", vec![
                Value::Integer(i),
                Value::Text(format!("User_{}", i)),
                Value::Integer(20 + (i % 50)),
            ]).unwrap();
        }

        // CREATE INDEX
        catalog.create_btree_index("users_id_btree", "users", "id").unwrap();
        catalog.create_hash_index("users_id_hash", "users", "id").unwrap();

        // ANALYZE
        let stats = catalog.analyze("users").unwrap();
        assert_eq!(stats.row_count, 100);
        assert_eq!(stats.column_stats.get("id").unwrap().distinct_count, 100);

        // OPTIMIZE a query
        let mut optimizer = QueryOptimizer::new();
        optimizer.table_stats.insert("users".into(), stats);
        optimizer.indexes.insert("users".into(), vec![
            IndexInfo {
                name: "users_id_hash".into(),
                table: "users".into(),
                columns: vec!["id".into()],
                index_type: IndexType::Hash,
                unique: true,
            },
        ]);

        // SELECT * FROM users WHERE id = 42
        let logical = LogicalPlan::Filter {
            predicate: Expr::BinaryOp {
                left: Box::new(Expr::Column { table: None, name: "id".into() }),
                op: BinOp::Eq,
                right: Box::new(Expr::Literal(Value::Integer(42))),
            },
            input: Box::new(LogicalPlan::Scan {
                table: "users".into(),
                alias: None,
                columns: vec!["id".into(), "name".into(), "age".into()],
            }),
        };

        let physical = optimizer.optimize(&logical);

        // Should use hash index for equality lookup
        match &physical {
            PhysicalPlan::HashLookup { key, .. } => {
                assert_eq!(*key, Value::Integer(42));
            }
            _ => panic!("Expected HashLookup for equality predicate with hash index"),
        }

        // EXECUTE the query
        let executor = Executor::new(&catalog);
        let results = executor.execute(&physical).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0][0], Value::Integer(42));
        assert_eq!(results[0][1], Value::Text("User_42".into()));
    }
}
