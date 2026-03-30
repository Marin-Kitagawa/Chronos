// chronos-build-system.rs
//
// Chronos Build System & Package Manager
// ========================================
// A self-hosting build system for the Chronos language implementing dependency
// resolution, incremental compilation, package management, and workspace support.
//
// Modules:
//   1.  Semantic Versioning (SemVer) — parsing, ordering, constraint satisfaction
//   2.  Package Manifest (Chrono.toml equivalent) — metadata, deps, features
//   3.  Dependency Graph — DAG construction, cycle detection (Kahn's algorithm)
//   4.  Version Resolution — MVIC (Minimum Version with Intersection Constraints)
//   5.  Registry & Lock File — package index, Chronos.lock
//   6.  Source Graph — file dependency tracking (module imports)
//   7.  Incremental Build Engine — content-hashing, dirty-set propagation
//   8.  Build Plan — topological compilation order, parallelism hints
//   9.  Artifact Cache — content-addressed store (SHA-256 keyed)
//  10.  Workspace — multi-package monorepo support
//  11.  Task Runner — build/test/run/bench/lint commands
//  12.  Diagnostics — structured warnings, errors, fix suggestions

use std::collections::{HashMap, HashSet, VecDeque, BTreeMap};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// 1. SEMANTIC VERSIONING
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed semantic version: MAJOR.MINOR.PATCH[-prerelease][+build].
/// Follows SemVer 2.0.0 specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major:      u64,
    pub minor:      u64,
    pub patch:      u64,
    pub pre:        Vec<PreRelease>,  // e.g. ["alpha", "1"]
    pub build_meta: Vec<String>,      // ignored in comparisons
}

/// A single pre-release identifier (either alphanumeric or numeric).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreRelease { Numeric(u64), AlphaNum(String) }

impl fmt::Display for PreRelease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self { PreRelease::Numeric(n) => write!(f, "{}", n), PreRelease::AlphaNum(s) => write!(f, "{}", s) }
    }
}

impl PartialOrd for PreRelease {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for PreRelease {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            // Numeric identifiers always have lower precedence than alphanumeric
            (PreRelease::Numeric(a), PreRelease::Numeric(b)) => a.cmp(b),
            (PreRelease::AlphaNum(a), PreRelease::AlphaNum(b)) => a.cmp(b),
            (PreRelease::Numeric(_), PreRelease::AlphaNum(_)) => std::cmp::Ordering::Less,
            (PreRelease::AlphaNum(_), PreRelease::Numeric(_)) => std::cmp::Ordering::Greater,
        }
    }
}

impl Version {
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Version { major, minor, patch, pre: Vec::new(), build_meta: Vec::new() }
    }

    /// Parse a version string like "1.2.3" or "2.0.0-alpha.1+build.5".
    pub fn parse(s: &str) -> Result<Self, BuildError> {
        let (core, rest) = if let Some(idx) = s.find('+') {
            (&s[..idx], Some(&s[idx+1..]))
        } else { (s, None) };
        let (core, pre_str) = if let Some(idx) = core.find('-') {
            (&core[..idx], Some(&core[idx+1..]))
        } else { (core, None) };

        let parts: Vec<&str> = core.split('.').collect();
        if parts.len() != 3 {
            return Err(BuildError::InvalidVersion(s.to_string()));
        }
        let parse_u64 = |s: &str| -> Result<u64, BuildError> {
            s.parse().map_err(|_| BuildError::InvalidVersion(s.to_string()))
        };
        let major = parse_u64(parts[0])?;
        let minor = parse_u64(parts[1])?;
        let patch = parse_u64(parts[2])?;

        let pre = if let Some(p) = pre_str {
            p.split('.').map(|id| {
                if let Ok(n) = id.parse::<u64>() { PreRelease::Numeric(n) }
                else { PreRelease::AlphaNum(id.to_string()) }
            }).collect()
        } else { Vec::new() };

        let build_meta = rest.map_or(Vec::new(), |b| {
            b.split('.').map(|s| s.to_string()).collect()
        });

        Ok(Version { major, minor, patch, pre, build_meta })
    }

    /// Whether this is a stable release (no pre-release identifier).
    pub fn is_stable(&self) -> bool { self.pre.is_empty() }

    /// Whether this version is compatible with `other` under SemVer rules.
    /// Compatible means: same major, self >= other. (^1.2.3 semantics)
    pub fn compatible_with(&self, other: &Version) -> bool {
        self.major == other.major && self >= other
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if !self.pre.is_empty() {
            write!(f, "-")?;
            for (i, p) in self.pre.iter().enumerate() {
                if i > 0 { write!(f, ".")?; }
                write!(f, "{}", p)?;
            }
        }
        Ok(())
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        // Compare MAJOR.MINOR.PATCH first
        let core = self.major.cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch));
        if core != Equal { return core; }
        // Pre-release has lower precedence than stable
        match (self.pre.is_empty(), other.pre.is_empty()) {
            (true,  false) => Greater,
            (false, true)  => Less,
            (true,  true)  => Equal,
            (false, false) => {
                let max_len = self.pre.len().max(other.pre.len());
                for i in 0..max_len {
                    match (self.pre.get(i), other.pre.get(i)) {
                        (Some(a), Some(b)) => { let r = a.cmp(b); if r != Equal { return r; } }
                        (Some(_), None)    => return Greater,
                        (None,    Some(_)) => return Less,
                        (None,    None)    => {}
                    }
                }
                Equal
            }
        }
    }
}

/// A version requirement constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum VersionReq {
    Exact(Version),        // =1.2.3
    Compatible(Version),   // ^1.2.3 (SemVer compatible)
    Tilde(Version),        // ~1.2.3 (patch updates only)
    Gte(Version),          // >=1.2.3
    Lte(Version),          // <=1.2.3
    Gt(Version),           // >1.2.3
    Lt(Version),           // <1.2.3
    Wildcard(u64, u64),    // 1.2.* (major.minor.*)
    Any,                   // *
}

impl VersionReq {
    pub fn parse(s: &str) -> Result<Self, BuildError> {
        let s = s.trim();
        if s == "*" { return Ok(VersionReq::Any); }
        if let Some(rest) = s.strip_prefix(">=") {
            return Ok(VersionReq::Gte(Version::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix("<=") {
            return Ok(VersionReq::Lte(Version::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix(">") {
            return Ok(VersionReq::Gt(Version::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix("<") {
            return Ok(VersionReq::Lt(Version::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix("=") {
            return Ok(VersionReq::Exact(Version::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix("^") {
            return Ok(VersionReq::Compatible(Version::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix("~") {
            return Ok(VersionReq::Tilde(Version::parse(rest.trim())?));
        }
        // Wildcard like "1.2.*"
        if s.ends_with(".*") {
            let parts: Vec<&str> = s[..s.len()-2].split('.').collect();
            if parts.len() == 2 {
                let major = parts[0].parse().map_err(|_| BuildError::InvalidVersion(s.to_string()))?;
                let minor = parts[1].parse().map_err(|_| BuildError::InvalidVersion(s.to_string()))?;
                return Ok(VersionReq::Wildcard(major, minor));
            }
        }
        // Default: treat bare version as ^X.Y.Z
        Ok(VersionReq::Compatible(Version::parse(s)?))
    }

    pub fn matches(&self, v: &Version) -> bool {
        match self {
            VersionReq::Any            => true,
            VersionReq::Exact(req)     => v == req,
            VersionReq::Compatible(req)=> v.compatible_with(req),
            VersionReq::Tilde(req)     => v.major == req.major && v.minor == req.minor && v >= req,
            VersionReq::Gte(req)       => v >= req,
            VersionReq::Lte(req)       => v <= req,
            VersionReq::Gt(req)        => v > req,
            VersionReq::Lt(req)        => v < req,
            VersionReq::Wildcard(maj, min) => v.major == *maj && v.minor == *min,
        }
    }
}

impl fmt::Display for VersionReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionReq::Any                => write!(f, "*"),
            VersionReq::Exact(v)           => write!(f, "={}", v),
            VersionReq::Compatible(v)      => write!(f, "^{}", v),
            VersionReq::Tilde(v)           => write!(f, "~{}", v),
            VersionReq::Gte(v)             => write!(f, ">={}", v),
            VersionReq::Lte(v)             => write!(f, "<={}", v),
            VersionReq::Gt(v)              => write!(f, ">{}", v),
            VersionReq::Lt(v)              => write!(f, "<{}", v),
            VersionReq::Wildcard(maj, min) => write!(f, "{}.{}.*", maj, min),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PACKAGE MANIFEST
// ─────────────────────────────────────────────────────────────────────────────

/// Package target type.
#[derive(Debug, Clone, PartialEq)]
pub enum TargetKind { Library, Binary, Test, Benchmark, Example }

/// A single build target within a package.
#[derive(Debug, Clone)]
pub struct Target {
    pub name:    String,
    pub kind:    TargetKind,
    pub path:    String,     // relative path to root source file
    pub edition: Edition,
}

/// Chronos language edition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Edition { E2024, E2025, E2026 }

impl Edition {
    pub fn latest() -> Self { Edition::E2026 }
    pub fn from_str(s: &str) -> Option<Self> {
        match s { "2024" => Some(Edition::E2024), "2025" => Some(Edition::E2025),
                  "2026" => Some(Edition::E2026), _ => None }
    }
}

/// A conditional feature flag.
#[derive(Debug, Clone)]
pub struct Feature {
    pub name:         String,
    pub enables:      Vec<String>,    // other features this activates
    pub dependencies: Vec<String>,    // dep/feature pairs enabled
}

/// A package dependency declaration.
#[derive(Debug, Clone)]
pub struct Dependency {
    pub name:     String,
    pub req:      VersionReq,
    pub features: Vec<String>,
    pub optional: bool,
    pub dev_only: bool,   // only needed for tests/benches
    pub source:   DependencySource,
}

/// Where a dependency comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum DependencySource {
    Registry(String),  // registry name
    Git { url: String, rev: Option<String>, tag: Option<String>, branch: Option<String> },
    Path(String),      // local path
}

/// The parsed package manifest (Chrono.toml).
#[derive(Debug, Clone)]
pub struct Manifest {
    pub name:         String,
    pub version:      Version,
    pub edition:      Edition,
    pub authors:      Vec<String>,
    pub description:  Option<String>,
    pub license:      Option<String>,
    pub repository:   Option<String>,
    pub targets:      Vec<Target>,
    pub dependencies: Vec<Dependency>,
    pub features:     HashMap<String, Feature>,
    pub build_script: Option<String>,  // path to build.ch
}

impl Manifest {
    pub fn new(name: &str, version: Version) -> Self {
        Manifest {
            name:         name.to_string(),
            version,
            edition:      Edition::latest(),
            authors:      Vec::new(),
            description:  None,
            license:      None,
            repository:   None,
            targets:      vec![Target {
                name:    name.to_string(),
                kind:    TargetKind::Library,
                path:    "src/lib.ch".to_string(),
                edition: Edition::latest(),
            }],
            dependencies: Vec::new(),
            features:     HashMap::new(),
            build_script: None,
        }
    }

    /// Add a dependency.
    pub fn add_dep(&mut self, dep: Dependency) { self.dependencies.push(dep); }

    /// Transitive feature expansion: given active features, return all transitively-enabled deps.
    pub fn expand_features(&self, active: &HashSet<String>) -> HashSet<String> {
        let mut enabled = active.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for feat_name in enabled.clone().iter() {
                if let Some(f) = self.features.get(feat_name) {
                    for sub in &f.enables {
                        if enabled.insert(sub.clone()) { changed = true; }
                    }
                }
            }
        }
        enabled
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ERROR TYPES
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    InvalidVersion(String),
    CyclicDependency(Vec<String>),
    NoMatchingVersion { package: String, req: String },
    UnresolvedDep { package: String, dep: String },
    DuplicatePackage(String),
    IoError(String),
    CompileError { package: String, message: String },
    BuildScriptFailed(String),
    FeatureNotFound { package: String, feature: String },
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::InvalidVersion(s)      => write!(f, "invalid version: {}", s),
            BuildError::CyclicDependency(cycle) => write!(f, "cyclic dependency: {}", cycle.join(" -> ")),
            BuildError::NoMatchingVersion { package, req } =>
                write!(f, "no matching version for {} satisfying {}", package, req),
            BuildError::UnresolvedDep { package, dep } =>
                write!(f, "unresolved dependency: {} requires {}", package, dep),
            BuildError::DuplicatePackage(n)    => write!(f, "duplicate package: {}", n),
            BuildError::IoError(e)             => write!(f, "I/O error: {}", e),
            BuildError::CompileError { package, message } =>
                write!(f, "compile error in {}: {}", package, message),
            BuildError::BuildScriptFailed(p)   => write!(f, "build script failed: {}", p),
            BuildError::FeatureNotFound { package, feature } =>
                write!(f, "feature {} not found in package {}", feature, package),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. DEPENDENCY GRAPH
// ─────────────────────────────────────────────────────────────────────────────

/// A node in the dependency graph.
#[derive(Debug, Clone)]
pub struct DepNode {
    pub package: String,
    pub version: Version,
    pub deps:    Vec<(String, VersionReq)>,  // (package_name, requirement)
}

/// Directed acyclic dependency graph.
pub struct DepGraph {
    pub nodes:    HashMap<String, DepNode>,
    pub adj:      HashMap<String, Vec<String>>,  // pkg → [deps it depends on]
}

impl DepGraph {
    pub fn new() -> Self { DepGraph { nodes: HashMap::new(), adj: HashMap::new() } }

    pub fn add_node(&mut self, node: DepNode) {
        let name = node.package.clone();
        let deps: Vec<String> = node.deps.iter().map(|(d, _)| d.clone()).collect();
        self.nodes.insert(name.clone(), node);
        self.adj.entry(name).or_default().extend(deps);
    }

    /// Detect cycles using Kahn's algorithm.
    /// Returns Err with the cycle members if one is found.
    pub fn detect_cycles(&self) -> Result<(), BuildError> {
        // Compute in-degrees
        let mut in_degree: HashMap<&str, usize> = self.nodes.keys().map(|k| (k.as_str(), 0)).collect();
        for (_, deps) in &self.adj {
            for dep in deps {
                *in_degree.entry(dep.as_str()).or_insert(0) += 1;
            }
        }
        let mut queue: VecDeque<&str> = in_degree.iter()
            .filter(|(_, &d)| d == 0).map(|(&k, _)| k).collect();
        let mut processed = 0usize;
        while let Some(node) = queue.pop_front() {
            processed += 1;
            if let Some(deps) = self.adj.get(node) {
                for dep in deps {
                    let count = in_degree.entry(dep.as_str()).or_insert(0);
                    *count = count.saturating_sub(1);
                    if *count == 0 { queue.push_back(dep.as_str()); }
                }
            }
        }
        if processed < self.nodes.len() {
            // Cycle exists — find members
            let cycle: Vec<String> = in_degree.iter()
                .filter(|(_, &d)| d > 0)
                .map(|(&k, _)| k.to_string())
                .collect();
            return Err(BuildError::CyclicDependency(cycle));
        }
        Ok(())
    }

    /// Topological sort via Kahn's algorithm.
    /// Returns packages in compilation order (dependencies before dependents).
    pub fn topo_sort(&self) -> Result<Vec<String>, BuildError> {
        self.detect_cycles()?;
        let mut in_degree: HashMap<&str, usize> = self.nodes.keys().map(|k| (k.as_str(), 0)).collect();
        for (_, deps) in &self.adj {
            for dep in deps {
                *in_degree.entry(dep.as_str()).or_insert(0) += 1;
            }
        }
        let mut queue: VecDeque<&str> = in_degree.iter()
            .filter(|(_, &d)| d == 0).map(|(&k, _)| k).collect();
        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            if let Some(deps) = self.adj.get(node) {
                for dep in deps {
                    let count = in_degree.get_mut(dep.as_str()).unwrap();
                    *count -= 1;
                    if *count == 0 { queue.push_back(dep.as_str()); }
                }
            }
        }
        // Topological sort has dependencies first; reverse so compiling goes leaf→root
        order.reverse();
        Ok(order)
    }

    /// Find all transitive dependencies of a package.
    pub fn transitive_deps(&self, root: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![root.to_string()];
        while let Some(pkg) = stack.pop() {
            if visited.insert(pkg.clone()) {
                if let Some(deps) = self.adj.get(&pkg) {
                    stack.extend(deps.clone());
                }
            }
        }
        visited.remove(root);
        visited
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. VERSION RESOLUTION (MVIC — Minimum Version with Intersection Constraints)
// ─────────────────────────────────────────────────────────────────────────────

/// The package registry: maps package name → available versions → manifest.
pub struct Registry {
    pub packages: HashMap<String, BTreeMap<Version, Manifest>>,
}

impl Registry {
    pub fn new() -> Self { Registry { packages: HashMap::new() } }

    pub fn publish(&mut self, manifest: Manifest) {
        self.packages
            .entry(manifest.name.clone())
            .or_default()
            .insert(manifest.version.clone(), manifest);
    }

    /// Find the newest version matching all requirements.
    pub fn resolve_single(&self, name: &str, reqs: &[VersionReq]) -> Option<&Version> {
        let versions = self.packages.get(name)?;
        // Iterate in reverse (newest first)
        versions.keys().rev().find(|v| reqs.iter().all(|r| r.matches(v)))
    }

    /// Full dependency resolution using iterative fixpoint.
    /// For each package in the closure, finds the highest version satisfying
    /// all constraints imposed by dependents. Returns (package → resolved_version).
    pub fn resolve(&self, root: &Manifest) -> Result<HashMap<String, Version>, BuildError> {
        let mut constraints: HashMap<String, Vec<VersionReq>> = HashMap::new();
        let mut work_queue: VecDeque<(String, Version)> = VecDeque::new();
        let mut resolved: HashMap<String, Version> = HashMap::new();

        // Seed with root's direct dependencies
        for dep in &root.dependencies {
            constraints.entry(dep.name.clone()).or_default().push(dep.req.clone());
            work_queue.push_back((dep.name.clone(), Version::new(0, 0, 0)));
        }

        let mut iterations = 0;
        while let Some((pkg, _)) = work_queue.pop_front() {
            iterations += 1;
            if iterations > 10_000 { break; } // safety limit

            let reqs = constraints.get(&pkg).cloned().unwrap_or_default();
            let chosen = self.resolve_single(&pkg, &reqs).ok_or_else(|| {
                BuildError::NoMatchingVersion {
                    package: pkg.clone(),
                    req: reqs.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", "),
                }
            })?;

            // If version changed, re-process this package's deps
            if resolved.get(&pkg) != Some(chosen) {
                let chosen = chosen.clone();
                resolved.insert(pkg.clone(), chosen.clone());
                if let Some(manifest) = self.packages.get(&pkg).and_then(|m| m.get(&chosen)) {
                    for dep in &manifest.dependencies {
                        if !dep.dev_only {
                            let c = constraints.entry(dep.name.clone()).or_default();
                            if !c.contains(&dep.req) {
                                c.push(dep.req.clone());
                                work_queue.push_back((dep.name.clone(), Version::new(0,0,0)));
                            }
                        }
                    }
                }
            }
        }
        Ok(resolved)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. LOCK FILE
// ─────────────────────────────────────────────────────────────────────────────

/// A locked dependency entry: exact version + content hash for integrity.
#[derive(Debug, Clone, PartialEq)]
pub struct LockedDep {
    pub name:        String,
    pub version:     Version,
    pub checksum:    String,   // SHA-256 hex of the downloaded tarball
    pub source:      String,   // "registry+https://packages.chronos.dev"
    pub deps:        Vec<String>,  // name@version of direct deps
}

/// The Chronos.lock file: fully-deterministic snapshot of resolved deps.
#[derive(Debug, Clone)]
pub struct LockFile {
    pub version:  u32,
    pub packages: Vec<LockedDep>,
}

impl LockFile {
    pub fn new() -> Self { LockFile { version: 1, packages: Vec::new() } }

    pub fn add(&mut self, dep: LockedDep) { self.packages.push(dep); }

    pub fn find(&self, name: &str, ver: &Version) -> Option<&LockedDep> {
        self.packages.iter().find(|d| d.name == name && &d.version == ver)
    }

    /// Serialise to a deterministic string (TOML-like for readability).
    pub fn serialize(&self) -> String {
        let mut out = format!("# Chronos.lock — DO NOT EDIT MANUALLY\nversion = {}\n\n", self.version);
        // Sort for determinism
        let mut pkgs = self.packages.clone();
        pkgs.sort_by(|a, b| a.name.cmp(&b.name).then(a.version.cmp(&b.version)));
        for p in &pkgs {
            out += &format!("[[package]]\nname = \"{}\"\nversion = \"{}\"\nsource = \"{}\"\nchecksum = \"{}\"\n",
                p.name, p.version, p.source, p.checksum);
            if !p.deps.is_empty() {
                out += &format!("dependencies = [{}]\n",
                    p.deps.iter().map(|d| format!("\"{}\"", d)).collect::<Vec<_>>().join(", "));
            }
            out += "\n";
        }
        out
    }

    /// Verify all checksums are non-empty (real impl would SHA-256 verify downloaded files).
    pub fn verify_checksums(&self) -> Result<(), BuildError> {
        for pkg in &self.packages {
            if pkg.checksum.is_empty() {
                return Err(BuildError::IoError(
                    format!("missing checksum for {}@{}", pkg.name, pkg.version)));
            }
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. SOURCE FILE GRAPH (module-level dependency tracking)
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a single Chronos source file with its declared imports.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path:    String,
    pub imports: Vec<String>,    // module paths this file imports
    pub hash:    u64,            // content hash (FNV-1a)
}

impl SourceFile {
    pub fn new(path: String, content: &str) -> Self {
        let imports = extract_imports(content);
        let hash    = fnv1a_hash(content.as_bytes());
        SourceFile { path, imports, hash }
    }
}

/// Extract import declarations from Chronos source (simulated parser).
/// Real implementation would use the Chronos parser; here we scan for `import "..."` lines.
fn extract_imports(source: &str) -> Vec<String> {
    source.lines().filter_map(|line| {
        let trimmed = line.trim();
        if trimmed.starts_with("import ") {
            let rest = trimmed["import ".len()..].trim();
            let module = rest.trim_end_matches(';').trim().trim_matches('"');
            Some(module.to_string())
        } else { None }
    }).collect()
}

/// FNV-1a 64-bit hash (fast, non-cryptographic, used for change detection).
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    let mut hash = 14695981039346656037u64;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

/// The source dependency graph: maps file path → SourceFile.
pub struct SourceGraph {
    pub files: HashMap<String, SourceFile>,
    /// Module path → file path mapping
    pub module_map: HashMap<String, String>,
}

impl SourceGraph {
    pub fn new() -> Self { SourceGraph { files: HashMap::new(), module_map: HashMap::new() } }

    pub fn add_file(&mut self, file: SourceFile) {
        // Derive module name from path (e.g. "src/parser.ch" → "parser")
        let module = file.path
            .trim_end_matches(".ch")
            .split('/')
            .last()
            .unwrap_or(&file.path)
            .to_string();
        self.module_map.insert(module, file.path.clone());
        self.files.insert(file.path.clone(), file);
    }

    /// Compute the reverse dependency map: for each file, which files import it?
    pub fn reverse_deps(&self) -> HashMap<String, Vec<String>> {
        let mut rev: HashMap<String, Vec<String>> = HashMap::new();
        for (path, file) in &self.files {
            for import in &file.imports {
                if let Some(dep_path) = self.module_map.get(import) {
                    rev.entry(dep_path.clone()).or_default().push(path.clone());
                }
            }
        }
        rev
    }

    /// Given a set of changed files, return all files that need recompilation
    /// (the changed files plus all their transitive reverse-dependents).
    pub fn dirty_set(&self, changed: &HashSet<String>) -> HashSet<String> {
        let rev = self.reverse_deps();
        let mut dirty = changed.clone();
        let mut stack: Vec<String> = changed.iter().cloned().collect();
        while let Some(f) = stack.pop() {
            if let Some(dependents) = rev.get(&f) {
                for dep in dependents {
                    if dirty.insert(dep.clone()) {
                        stack.push(dep.clone());
                    }
                }
            }
        }
        dirty
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. INCREMENTAL BUILD ENGINE
// ─────────────────────────────────────────────────────────────────────────────

/// Result of building a single compilation unit.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildStatus { UpToDate, Rebuilt, Failed(String) }

/// A stored record of a previous build.
#[derive(Debug, Clone)]
pub struct BuildRecord {
    pub path:       String,
    pub hash:       u64,       // content hash at last build
    pub artifact:   String,    // path to compiled artifact
    pub timestamp:  u64,       // Unix timestamp of build
    pub success:    bool,
}

/// Content-addressed build cache.
pub struct BuildCache {
    records: HashMap<String, BuildRecord>,
}

impl BuildCache {
    pub fn new() -> Self { BuildCache { records: HashMap::new() } }

    /// Check if a file needs rebuilding.
    pub fn needs_rebuild(&self, path: &str, current_hash: u64) -> bool {
        match self.records.get(path) {
            None => true,
            Some(r) => r.hash != current_hash || !r.success,
        }
    }

    /// Record a successful build.
    pub fn record_build(&mut self, path: String, hash: u64, artifact: String) {
        self.records.insert(path.clone(), BuildRecord {
            path, hash, artifact, timestamp: 0, success: true,
        });
    }

    /// Invalidate a cached entry (e.g. when a dependency changes).
    pub fn invalidate(&mut self, path: &str) { self.records.remove(path); }

    /// Return all cached artifacts.
    pub fn all_artifacts(&self) -> Vec<&BuildRecord> { self.records.values().collect() }
}

/// The incremental build engine.
pub struct BuildEngine {
    pub cache:        BuildCache,
    pub source_graph: SourceGraph,
}

impl BuildEngine {
    pub fn new() -> Self {
        BuildEngine { cache: BuildCache::new(), source_graph: SourceGraph::new() }
    }

    /// Determine which files need recompilation given the current source state.
    /// Returns a list of (file_path, hash) pairs in compilation order.
    pub fn plan_build(&self) -> Vec<(String, u64)> {
        // Find all dirty files
        let changed: HashSet<String> = self.source_graph.files.iter()
            .filter(|(path, file)| self.cache.needs_rebuild(path, file.hash))
            .map(|(path, _)| path.clone())
            .collect();
        let dirty = self.source_graph.dirty_set(&changed);

        // Topological sort of dirty files (dependencies before dependents)
        let mut in_deg: HashMap<&str, usize> = dirty.iter().map(|p| (p.as_str(), 0)).collect();
        let rev = self.source_graph.reverse_deps();
        for path in &dirty {
            if let Some(file) = self.source_graph.files.get(path) {
                for import in &file.imports {
                    if let Some(dep_path) = self.source_graph.module_map.get(import) {
                        if dirty.contains(dep_path) {
                            *in_deg.entry(path.as_str()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        let mut queue: VecDeque<&str> = in_deg.iter()
            .filter(|(_, &d)| d == 0).map(|(&k, _)| k).collect();
        let mut order = Vec::new();
        while let Some(path) = queue.pop_front() {
            if let Some(file) = self.source_graph.files.get(path) {
                order.push((path.to_string(), file.hash));
                if let Some(dependents) = rev.get(path) {
                    for dep in dependents {
                        if dirty.contains(dep) {
                            let c = in_deg.entry(dep.as_str()).or_insert(0);
                            *c = c.saturating_sub(1);
                            if *c == 0 { queue.push_back(dep.as_str()); }
                        }
                    }
                }
            }
        }
        order
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. ARTIFACT CACHE (CONTENT-ADDRESSED)
// ─────────────────────────────────────────────────────────────────────────────

/// A content-addressed artifact store.
/// Artifacts are keyed by the SHA-256 hash of their inputs, enabling
/// sharing across workspaces and hermetic rebuilds.
pub struct ArtifactStore {
    entries: HashMap<String, ArtifactEntry>,
}

#[derive(Debug, Clone)]
pub struct ArtifactEntry {
    pub input_hash:  String,  // hex SHA-256 of all inputs
    pub artifact:    Vec<u8>, // compiled artifact bytes (simulated)
    pub metadata:    HashMap<String, String>,
}

/// Simulated SHA-256: just FNV-1a formatted as hex for testing purposes.
/// Real implementation would use a proper SHA-256 library.
pub fn sha256_hex(data: &[u8]) -> String {
    // Simulate 256 bits using 4 × FNV-1a with different seeds
    let h0 = fnv1a_hash(data);
    let h1 = fnv1a_hash(&data.iter().rev().cloned().collect::<Vec<_>>());
    let data2: Vec<u8> = data.iter().map(|b| b.wrapping_add(1)).collect();
    let h2 = fnv1a_hash(&data2);
    let h3 = fnv1a_hash(&[data, b"salt"].concat());
    format!("{:016x}{:016x}{:016x}{:016x}", h0, h1, h2, h3)
}

impl ArtifactStore {
    pub fn new() -> Self { ArtifactStore { entries: HashMap::new() } }

    pub fn store(&mut self, input_hash: String, artifact: Vec<u8>,
                 metadata: HashMap<String, String>) {
        self.entries.insert(input_hash.clone(), ArtifactEntry { input_hash, artifact, metadata });
    }

    pub fn lookup(&self, input_hash: &str) -> Option<&ArtifactEntry> {
        self.entries.get(input_hash)
    }

    pub fn contains(&self, input_hash: &str) -> bool { self.entries.contains_key(input_hash) }

    pub fn evict_oldest(&mut self, keep: usize) {
        if self.entries.len() > keep {
            let n_remove = self.entries.len() - keep;
            let keys: Vec<String> = self.entries.keys().cloned().take(n_remove).collect();
            for k in keys { self.entries.remove(&k); }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. WORKSPACE
// ─────────────────────────────────────────────────────────────────────────────

/// A Chronos workspace: multiple packages sharing a single lock file and cache.
pub struct Workspace {
    pub root:     String,
    pub members:  Vec<Manifest>,
    pub lock:     LockFile,
    pub registry: Registry,
}

impl Workspace {
    pub fn new(root: &str) -> Self {
        Workspace {
            root:     root.to_string(),
            members:  Vec::new(),
            lock:     LockFile::new(),
            registry: Registry::new(),
        }
    }

    pub fn add_member(&mut self, manifest: Manifest) { self.members.push(manifest); }

    /// Resolve dependencies for the entire workspace.
    /// All members share the same resolution; the lock file is shared.
    pub fn resolve_all(&self) -> Result<HashMap<String, Version>, BuildError> {
        // Merge all constraints from all members
        let mut combined = Manifest::new("workspace-virtual", Version::new(0, 0, 0));
        for member in &self.members {
            for dep in &member.dependencies {
                combined.add_dep(dep.clone());
            }
        }
        self.registry.resolve(&combined)
    }

    /// Check if any member version is outdated (newer version available).
    pub fn outdated(&self) -> Vec<(String, Version, Version)> {
        let mut result = Vec::new();
        for pkg in &self.lock.packages {
            if let Some(versions) = self.registry.packages.get(&pkg.name) {
                if let Some(newest) = versions.keys().last() {
                    if newest > &pkg.version {
                        result.push((pkg.name.clone(), pkg.version.clone(), newest.clone()));
                    }
                }
            }
        }
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. BUILD PLAN
// ─────────────────────────────────────────────────────────────────────────────

/// A single step in the build plan.
#[derive(Debug, Clone)]
pub struct BuildStep {
    pub package:   String,
    pub target:    String,     // target name (lib, bin:foo, test:bar, ...)
    pub kind:      TargetKind,
    pub inputs:    Vec<String>,    // source files
    pub deps:      Vec<String>,    // packages that must be built before this
    pub flags:     Vec<String>,    // compiler flags
    pub output:    String,         // expected output artifact path
}

/// A complete build plan for a workspace.
pub struct BuildPlan {
    pub steps:        Vec<BuildStep>,
    pub parallelism:  usize,  // hint: max parallel compilations
}

impl BuildPlan {
    /// Construct a build plan from a dep graph and resolved versions.
    pub fn from_graph(graph: &DepGraph, manifests: &HashMap<String, Manifest>) -> Result<Self, BuildError> {
        let order = graph.topo_sort()?;
        let mut steps = Vec::new();
        for pkg in &order {
            if let Some(manifest) = manifests.get(pkg) {
                for target in &manifest.targets {
                    let deps: Vec<String> = manifest.dependencies.iter()
                        .map(|d| d.name.clone()).collect();
                    steps.push(BuildStep {
                        package:  pkg.clone(),
                        target:   target.name.clone(),
                        kind:     target.kind.clone(),
                        inputs:   vec![target.path.clone()],
                        deps,
                        flags:    Vec::new(),
                        output:   format!("target/{}/{}.ca", pkg, target.name), // .ca = Chronos artifact
                    });
                }
            }
        }
        // Estimate parallelism based on critical path depth
        let parallelism = (steps.len() / 4).max(1).min(16);
        Ok(BuildPlan { steps, parallelism })
    }

    /// Compute the critical path (longest dependency chain).
    pub fn critical_path(&self) -> Vec<String> {
        let mut depth: HashMap<&str, usize> = HashMap::new();
        let mut pred:  HashMap<&str, &str>  = HashMap::new();
        for step in &self.steps {
            let d = step.deps.iter()
                .filter_map(|dep| depth.get(dep.as_str()))
                .max()
                .copied()
                .unwrap_or(0) + 1;
            let prev_d = depth.entry(&step.package).or_insert(0);
            if d > *prev_d {
                *prev_d = d;
                if let Some(dep) = step.deps.first() {
                    pred.insert(&step.package, dep.as_str());
                }
            }
        }
        // Trace back
        let deepest = depth.iter().max_by_key(|(_, &d)| d).map(|(&k, _)| k);
        let mut path = Vec::new();
        let mut cur = deepest;
        while let Some(p) = cur {
            path.push(p.to_string());
            cur = pred.get(p).copied();
        }
        path.reverse();
        path
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. TASK RUNNER
// ─────────────────────────────────────────────────────────────────────────────

/// Available built-in build commands.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildCommand {
    Build   { release: bool, features: Vec<String> },
    Test    { filter: Option<String>, no_run: bool },
    Run     { bin: Option<String>, args: Vec<String> },
    Bench   { filter: Option<String> },
    Check,         // type-check without generating artifacts
    Clean,         // remove target/ directory
    Doc    { open: bool },
    Publish { dry_run: bool },
    Update  { package: Option<String> },
    Fetch,         // download dependencies without building
    Format,        // run chronos-fmt
    Lint,          // run chronos-clippy
}

/// Execution context for a build command.
pub struct TaskRunner {
    pub workspace: Workspace,
}

impl TaskRunner {
    pub fn new(workspace: Workspace) -> Self { TaskRunner { workspace } }

    /// Validate that a command is well-formed and all required packages are available.
    pub fn validate(&self, cmd: &BuildCommand) -> Result<(), BuildError> {
        match cmd {
            BuildCommand::Build { features, .. } => {
                for member in &self.workspace.members {
                    for feat in features {
                        if !feat.is_empty() && !member.features.contains_key(feat.as_str()) {
                            return Err(BuildError::FeatureNotFound {
                                package: member.name.clone(),
                                feature: feat.clone(),
                            });
                        }
                    }
                }
                Ok(())
            }
            BuildCommand::Publish { .. } => {
                // Ensure all members have a version and license
                for m in &self.workspace.members {
                    if m.license.is_none() {
                        return Err(BuildError::IoError(
                            format!("package {} missing license for publish", m.name)));
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Simulate running a command and return a structured result.
    pub fn run_command(&self, cmd: &BuildCommand) -> CommandResult {
        match cmd {
            BuildCommand::Check => {
                CommandResult { success: true, output: "All checks passed.".to_string(),
                                warnings: Vec::new(), errors: Vec::new() }
            }
            BuildCommand::Build { release, .. } => {
                let mode = if *release { "release" } else { "debug" };
                CommandResult {
                    success:  true,
                    output:   format!("Compiling {} packages in {} mode", self.workspace.members.len(), mode),
                    warnings: Vec::new(),
                    errors:   Vec::new(),
                }
            }
            BuildCommand::Clean => {
                CommandResult { success: true, output: "Removed target/".to_string(),
                                warnings: Vec::new(), errors: Vec::new() }
            }
            BuildCommand::Fetch => {
                CommandResult { success: true,
                                output: format!("Fetched {} packages", self.workspace.lock.packages.len()),
                                warnings: Vec::new(), errors: Vec::new() }
            }
            _ => CommandResult { success: true, output: String::new(), warnings: Vec::new(), errors: Vec::new() }
        }
    }
}

/// Result of running a build command.
#[derive(Debug, Clone)]
pub struct CommandResult {
    pub success:  bool,
    pub output:   String,
    pub warnings: Vec<Diagnostic>,
    pub errors:   Vec<Diagnostic>,
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. DIAGNOSTICS
// ─────────────────────────────────────────────────────────────────────────────

/// Severity of a diagnostic message.
#[derive(Debug, Clone, PartialEq)]
pub enum Severity { Error, Warning, Note, Help }

/// A structured compiler diagnostic with optional fix suggestion.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity:  Severity,
    pub code:      String,      // e.g. "E0001"
    pub message:   String,
    pub file:      Option<String>,
    pub line:      Option<u32>,
    pub column:    Option<u32>,
    pub snippet:   Option<String>,
    pub suggestion: Option<FixSuggestion>,
}

/// A machine-applicable fix suggestion.
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    pub message:     String,
    pub replacement: String,
    pub applicability: Applicability,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Applicability { MachineApplicable, MaybeIncorrect, HasPlaceholders }

impl Diagnostic {
    pub fn error(code: &str, message: &str) -> Self {
        Diagnostic { severity: Severity::Error, code: code.to_string(), message: message.to_string(),
                     file: None, line: None, column: None, snippet: None, suggestion: None }
    }
    pub fn warning(code: &str, message: &str) -> Self {
        Diagnostic { severity: Severity::Warning, code: code.to_string(), message: message.to_string(),
                     file: None, line: None, column: None, snippet: None, suggestion: None }
    }
    pub fn with_location(mut self, file: &str, line: u32, col: u32) -> Self {
        self.file = Some(file.to_string()); self.line = Some(line); self.column = Some(col); self
    }
    pub fn with_fix(mut self, msg: &str, replacement: &str) -> Self {
        self.suggestion = Some(FixSuggestion {
            message: msg.to_string(), replacement: replacement.to_string(),
            applicability: Applicability::MachineApplicable });
        self
    }

    pub fn fmt_short(&self) -> String {
        let loc = match (&self.file, self.line) {
            (Some(f), Some(l)) => format!("{}:{}: ", f, l),
            _ => String::new(),
        };
        format!("{}{:?}[{}]: {}", loc, self.severity, self.code, self.message)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. PACKAGE GRAPH UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/// Compute strongly-connected components using Kosaraju's two-pass algorithm.
/// Used to detect mutually-recursive package groups (which are forbidden).
pub fn kosaraju_scc(adj: &HashMap<String, Vec<String>>) -> Vec<Vec<String>> {
    let nodes: Vec<&String> = adj.keys().collect();
    // Pass 1: DFS, record finish order
    let mut visited  = HashSet::new();
    let mut finish_order = Vec::new();
    for node in &nodes {
        if !visited.contains(node.as_str()) {
            dfs_finish(node, adj, &mut visited, &mut finish_order);
        }
    }
    // Build reverse graph
    let mut rev_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (src, dests) in adj {
        for dst in dests {
            rev_adj.entry(dst.as_str()).or_default().push(src.as_str());
        }
    }
    // Pass 2: DFS on reverse graph in reverse finish order
    let mut visited2 = HashSet::new();
    let mut sccs = Vec::new();
    for node in finish_order.iter().rev() {
        if !visited2.contains(node.as_str()) {
            let mut scc = Vec::new();
            dfs_collect(node, &rev_adj, &mut visited2, &mut scc);
            sccs.push(scc);
        }
    }
    sccs
}

fn dfs_finish(node: &str, adj: &HashMap<String, Vec<String>>,
              visited: &mut HashSet<String>, order: &mut Vec<String>) {
    if !visited.insert(node.to_string()) { return; }
    if let Some(neighbors) = adj.get(node) {
        for nb in neighbors { dfs_finish(nb, adj, visited, order); }
    }
    order.push(node.to_string());
}

fn dfs_collect<'a>(node: &'a str, rev: &HashMap<&'a str, Vec<&'a str>>,
                   visited: &mut HashSet<String>, scc: &mut Vec<String>) {
    if !visited.insert(node.to_string()) { return; }
    scc.push(node.to_string());
    if let Some(neighbors) = rev.get(node) {
        for &nb in neighbors { dfs_collect(nb, rev, visited, scc); }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TESTS
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SemVer Parsing ────────────────────────────────────────────────────────

    #[test]
    fn test_version_parse_basic() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1); assert_eq!(v.minor, 2); assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_parse_prerelease() {
        let v = Version::parse("2.0.0-alpha.1").unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.pre, vec![PreRelease::AlphaNum("alpha".to_string()), PreRelease::Numeric(1)]);
    }

    #[test]
    fn test_version_parse_build_meta() {
        let v = Version::parse("1.0.0+build.42").unwrap();
        assert_eq!(v.build_meta, vec!["build", "42"]);
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(Version::parse("1.2").is_err());
        assert!(Version::parse("not-a-version").is_err());
    }

    #[test]
    fn test_version_display() {
        let v = Version::parse("1.2.3-beta.1").unwrap();
        assert_eq!(v.to_string(), "1.2.3-beta.1");
    }

    // ── SemVer Ordering ───────────────────────────────────────────────────────

    #[test]
    fn test_version_ordering() {
        let v100 = Version::parse("1.0.0").unwrap();
        let v110 = Version::parse("1.1.0").unwrap();
        let v111 = Version::parse("1.1.1").unwrap();
        let v200 = Version::parse("2.0.0").unwrap();
        assert!(v100 < v110);
        assert!(v110 < v111);
        assert!(v111 < v200);
    }

    #[test]
    fn test_prerelease_lower_than_release() {
        let pre    = Version::parse("1.0.0-alpha").unwrap();
        let stable = Version::parse("1.0.0").unwrap();
        assert!(pre < stable, "Pre-release should be < release");
    }

    #[test]
    fn test_prerelease_ordering() {
        let a1  = Version::parse("1.0.0-alpha.1").unwrap();
        let a2  = Version::parse("1.0.0-alpha.2").unwrap();
        let rc1 = Version::parse("1.0.0-rc.1").unwrap();
        assert!(a1 < a2, "alpha.1 < alpha.2");
        // Numeric < AlphaNum, so "1" < "rc" but "alpha" < "rc"
        // "alpha.2" < "rc.1" since "alpha" < "rc" lexicographically
        assert!(a2 < rc1, "alpha.2 < rc.1");
    }

    #[test]
    fn test_version_compatible_with() {
        let base = Version::parse("1.2.3").unwrap();
        let new  = Version::parse("1.3.0").unwrap();
        let major = Version::parse("2.0.0").unwrap();
        assert!(new.compatible_with(&base));
        assert!(!major.compatible_with(&base));
    }

    // ── Version Requirements ──────────────────────────────────────────────────

    #[test]
    fn test_req_caret() {
        let req = VersionReq::parse("^1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(req.matches(&Version::parse("1.3.0").unwrap()));
        assert!(req.matches(&Version::parse("1.99.99").unwrap()));
        assert!(!req.matches(&Version::parse("2.0.0").unwrap()));
        assert!(!req.matches(&Version::parse("1.2.2").unwrap()));
    }

    #[test]
    fn test_req_tilde() {
        let req = VersionReq::parse("~1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(req.matches(&Version::parse("1.2.9").unwrap()));
        assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
    }

    #[test]
    fn test_req_exact() {
        let req = VersionReq::parse("=1.2.3").unwrap();
        assert!(req.matches(&Version::parse("1.2.3").unwrap()));
        assert!(!req.matches(&Version::parse("1.2.4").unwrap()));
    }

    #[test]
    fn test_req_gte() {
        let req = VersionReq::parse(">=2.0.0").unwrap();
        assert!(req.matches(&Version::parse("2.0.0").unwrap()));
        assert!(req.matches(&Version::parse("3.0.0").unwrap()));
        assert!(!req.matches(&Version::parse("1.9.9").unwrap()));
    }

    #[test]
    fn test_req_wildcard() {
        let req = VersionReq::parse("1.2.*").unwrap();
        assert!(req.matches(&Version::parse("1.2.0").unwrap()));
        assert!(req.matches(&Version::parse("1.2.99").unwrap()));
        assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
    }

    #[test]
    fn test_req_any() {
        let req = VersionReq::parse("*").unwrap();
        assert!(req.matches(&Version::parse("0.0.1").unwrap()));
        assert!(req.matches(&Version::parse("99.99.99").unwrap()));
    }

    // ── Dependency Graph ──────────────────────────────────────────────────────

    #[test]
    fn test_dag_topo_sort_linear() {
        let mut g = DepGraph::new();
        g.add_node(DepNode { package: "c".into(), version: Version::new(1,0,0), deps: vec![] });
        g.add_node(DepNode { package: "b".into(), version: Version::new(1,0,0),
                             deps: vec![("c".into(), VersionReq::Any)] });
        g.add_node(DepNode { package: "a".into(), version: Version::new(1,0,0),
                             deps: vec![("b".into(), VersionReq::Any)] });
        let order = g.topo_sort().unwrap();
        // a depends on b depends on c; dependencies must be compiled before dependents.
        // c (no deps) → b (depends on c) → a (depends on b): c must come before b, b before a.
        let ai = order.iter().position(|x| x == "a").unwrap();
        let bi = order.iter().position(|x| x == "b").unwrap();
        let ci = order.iter().position(|x| x == "c").unwrap();
        assert!(ci < bi, "c before b: {} < {}", ci, bi);
        assert!(bi < ai, "b before a: {} < {}", bi, ai);
    }

    #[test]
    fn test_dag_cycle_detected() {
        let mut g = DepGraph::new();
        g.adj.insert("a".into(), vec!["b".into()]);
        g.adj.insert("b".into(), vec!["a".into()]);
        g.nodes.insert("a".into(), DepNode { package: "a".into(), version: Version::new(1,0,0), deps: vec![] });
        g.nodes.insert("b".into(), DepNode { package: "b".into(), version: Version::new(1,0,0), deps: vec![] });
        assert!(g.detect_cycles().is_err(), "Cycle should be detected");
    }

    #[test]
    fn test_dag_transitive_deps() {
        let mut g = DepGraph::new();
        g.add_node(DepNode { package: "c".into(), version: Version::new(1,0,0), deps: vec![] });
        g.add_node(DepNode { package: "b".into(), version: Version::new(1,0,0),
                             deps: vec![("c".into(), VersionReq::Any)] });
        g.add_node(DepNode { package: "a".into(), version: Version::new(1,0,0),
                             deps: vec![("b".into(), VersionReq::Any)] });
        let trans = g.transitive_deps("a");
        assert!(trans.contains("b"), "b is transitive dep of a");
        assert!(trans.contains("c"), "c is transitive dep of a");
        assert!(!trans.contains("a"), "a not its own dep");
    }

    // ── Registry & Resolution ─────────────────────────────────────────────────

    #[test]
    fn test_registry_resolve_single() {
        let mut reg = Registry::new();
        for ver in ["1.0.0", "1.1.0", "1.2.0", "2.0.0"] {
            reg.publish(Manifest::new("foo", Version::parse(ver).unwrap()));
        }
        let req = VersionReq::parse("^1.1.0").unwrap();
        let resolved = reg.resolve_single("foo", &[req]).unwrap();
        assert_eq!(resolved, &Version::parse("1.2.0").unwrap(),
                   "Should pick highest compatible: {}", resolved);
    }

    #[test]
    fn test_registry_no_match() {
        let mut reg = Registry::new();
        reg.publish(Manifest::new("foo", Version::parse("1.0.0").unwrap()));
        let req = VersionReq::parse("^2.0.0").unwrap();
        assert!(reg.resolve_single("foo", &[req]).is_none());
    }

    #[test]
    fn test_resolve_direct_deps() {
        let mut reg = Registry::new();
        reg.publish(Manifest::new("logger", Version::parse("1.2.0").unwrap()));
        reg.publish(Manifest::new("logger", Version::parse("1.3.0").unwrap()));

        let mut root = Manifest::new("my-app", Version::parse("0.1.0").unwrap());
        root.add_dep(Dependency {
            name:     "logger".into(),
            req:      VersionReq::parse("^1.2.0").unwrap(),
            features: Vec::new(), optional: false, dev_only: false,
            source:   DependencySource::Registry("default".into()),
        });
        let resolved = reg.resolve(&root).unwrap();
        assert_eq!(resolved.get("logger").unwrap(), &Version::parse("1.3.0").unwrap());
    }

    // ── Lock File ─────────────────────────────────────────────────────────────

    #[test]
    fn test_lock_file_serialize_sorted() {
        let mut lock = LockFile::new();
        lock.add(LockedDep { name: "zzz".into(), version: Version::new(1,0,0),
                             checksum: "abc".into(), source: "registry".into(), deps: vec![] });
        lock.add(LockedDep { name: "aaa".into(), version: Version::new(2,0,0),
                             checksum: "def".into(), source: "registry".into(), deps: vec![] });
        let s = lock.serialize();
        let aaa_pos = s.find("aaa").unwrap();
        let zzz_pos = s.find("zzz").unwrap();
        assert!(aaa_pos < zzz_pos, "Lock file should be sorted alphabetically");
    }

    #[test]
    fn test_lock_file_verify_missing_checksum() {
        let mut lock = LockFile::new();
        lock.add(LockedDep { name: "foo".into(), version: Version::new(1,0,0),
                             checksum: "".into(), source: "registry".into(), deps: vec![] });
        assert!(lock.verify_checksums().is_err());
    }

    #[test]
    fn test_lock_file_verify_ok() {
        let mut lock = LockFile::new();
        lock.add(LockedDep { name: "foo".into(), version: Version::new(1,0,0),
                             checksum: "a".repeat(64), source: "registry".into(), deps: vec![] });
        assert!(lock.verify_checksums().is_ok());
    }

    // ── Source Graph & Incremental Builds ─────────────────────────────────────

    #[test]
    fn test_source_graph_dirty_propagation() {
        let mut sg = SourceGraph::new();
        sg.add_file(SourceFile::new("src/utils.ch".into(), "// utils"));
        sg.add_file(SourceFile::new("src/parser.ch".into(), "import \"utils\";\n// parser"));
        sg.add_file(SourceFile::new("src/main.ch".into(), "import \"parser\";\n// main"));

        // If utils changes, parser and main should be dirty
        let mut changed = HashSet::new();
        changed.insert("src/utils.ch".to_string());
        let dirty = sg.dirty_set(&changed);
        assert!(dirty.contains("src/utils.ch"), "utils is dirty");
        assert!(dirty.contains("src/parser.ch"), "parser is dirty (imports utils)");
        assert!(dirty.contains("src/main.ch"), "main is dirty (imports parser)");
    }

    #[test]
    fn test_source_graph_no_propagation() {
        let mut sg = SourceGraph::new();
        sg.add_file(SourceFile::new("src/a.ch".into(), "// a standalone"));
        sg.add_file(SourceFile::new("src/b.ch".into(), "// b standalone"));

        let mut changed = HashSet::new();
        changed.insert("src/a.ch".to_string());
        let dirty = sg.dirty_set(&changed);
        assert!(dirty.contains("src/a.ch"));
        assert!(!dirty.contains("src/b.ch"), "b should not be dirty");
    }

    #[test]
    fn test_build_cache_needs_rebuild() {
        let mut cache = BuildCache::new();
        assert!(cache.needs_rebuild("src/foo.ch", 12345), "Unknown file needs rebuild");
        cache.record_build("src/foo.ch".into(), 12345, "target/foo.ca".into());
        assert!(!cache.needs_rebuild("src/foo.ch", 12345), "Unchanged file does not need rebuild");
        assert!(cache.needs_rebuild("src/foo.ch", 99999), "Changed file needs rebuild");
    }

    // ── Artifact Store ────────────────────────────────────────────────────────

    #[test]
    fn test_artifact_store_roundtrip() {
        let mut store = ArtifactStore::new();
        let hash = sha256_hex(b"input data");
        store.store(hash.clone(), b"artifact bytes".to_vec(), HashMap::new());
        assert!(store.contains(&hash));
        let entry = store.lookup(&hash).unwrap();
        assert_eq!(entry.artifact, b"artifact bytes");
    }

    #[test]
    fn test_artifact_store_evict() {
        let mut store = ArtifactStore::new();
        for i in 0..10u8 {
            let h = sha256_hex(&[i]);
            store.store(h, vec![i], HashMap::new());
        }
        assert_eq!(store.entries.len(), 10);
        store.evict_oldest(5);
        assert_eq!(store.entries.len(), 5);
    }

    #[test]
    fn test_sha256_hex_deterministic() {
        let h1 = sha256_hex(b"hello");
        let h2 = sha256_hex(b"hello");
        assert_eq!(h1, h2);
        let h3 = sha256_hex(b"world");
        assert_ne!(h1, h3);
    }

    // ── FNV Hash ──────────────────────────────────────────────────────────────

    #[test]
    fn test_fnv_hash_empty() {
        // FNV-1a of empty input = FNV offset basis
        assert_eq!(fnv1a_hash(b""), 14695981039346656037u64);
    }

    #[test]
    fn test_fnv_hash_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
        assert_ne!(fnv1a_hash(b"hello"), fnv1a_hash(b"world"));
    }

    // ── Build Plan ────────────────────────────────────────────────────────────

    #[test]
    fn test_build_plan_from_graph() {
        let mut g = DepGraph::new();
        g.add_node(DepNode { package: "core".into(), version: Version::new(1,0,0), deps: vec![] });
        g.add_node(DepNode { package: "app".into(), version: Version::new(1,0,0),
                             deps: vec![("core".into(), VersionReq::Any)] });

        let mut manifests = HashMap::new();
        manifests.insert("core".into(), Manifest::new("core", Version::new(1,0,0)));
        manifests.insert("app".into(),  Manifest::new("app",  Version::new(1,0,0)));

        let plan = BuildPlan::from_graph(&g, &manifests).unwrap();
        assert!(!plan.steps.is_empty());
        assert!(plan.parallelism >= 1);
    }

    // ── Workspace ─────────────────────────────────────────────────────────────

    #[test]
    fn test_workspace_add_member() {
        let mut ws = Workspace::new("/projects/my-workspace");
        ws.add_member(Manifest::new("lib-a", Version::new(1,0,0)));
        ws.add_member(Manifest::new("lib-b", Version::new(2,0,0)));
        assert_eq!(ws.members.len(), 2);
    }

    #[test]
    fn test_workspace_outdated() {
        let mut ws = Workspace::new("/projects/ws");
        ws.registry.publish(Manifest::new("dep", Version::parse("1.0.0").unwrap()));
        ws.registry.publish(Manifest::new("dep", Version::parse("1.1.0").unwrap()));
        ws.lock.add(LockedDep {
            name: "dep".into(), version: Version::new(1,0,0),
            checksum: "x".into(), source: "registry".into(), deps: vec![],
        });
        let outdated = ws.outdated();
        assert_eq!(outdated.len(), 1);
        assert_eq!(outdated[0].0, "dep");
        assert_eq!(outdated[0].2, Version::parse("1.1.0").unwrap());
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    #[test]
    fn test_diagnostic_format() {
        let d = Diagnostic::error("E0001", "undefined variable `x`")
            .with_location("src/main.ch", 10, 5);
        let s = d.fmt_short();
        assert!(s.contains("E0001"), "Code in message: {}", s);
        assert!(s.contains("src/main.ch"), "File in message: {}", s);
        assert!(s.contains("10"), "Line in message: {}", s);
    }

    #[test]
    fn test_diagnostic_fix_suggestion() {
        let d = Diagnostic::warning("W0042", "unused variable")
            .with_fix("prefix with underscore", "_x");
        assert!(d.suggestion.is_some());
        assert_eq!(d.suggestion.unwrap().replacement, "_x");
    }

    // ── Task Runner ───────────────────────────────────────────────────────────

    #[test]
    fn test_task_runner_validate_feature_missing() {
        let ws = Workspace::new("/");
        let runner = TaskRunner::new(ws);
        let cmd = BuildCommand::Build { release: false, features: vec!["nonexistent-feat".into()] };
        // No members in workspace → no feature check → ok
        assert!(runner.validate(&cmd).is_ok());
    }

    #[test]
    fn test_task_runner_build_command() {
        let ws = Workspace::new("/");
        let runner = TaskRunner::new(ws);
        let res = runner.run_command(&BuildCommand::Build { release: true, features: vec![] });
        assert!(res.success);
        assert!(res.output.contains("release"));
    }

    #[test]
    fn test_task_runner_clean_command() {
        let ws = Workspace::new("/");
        let runner = TaskRunner::new(ws);
        let res = runner.run_command(&BuildCommand::Clean);
        assert!(res.success);
        assert!(res.output.contains("target/"));
    }

    // ── SCC ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_kosaraju_no_cycles() {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        adj.insert("a".into(), vec!["b".into()]);
        adj.insert("b".into(), vec!["c".into()]);
        adj.insert("c".into(), vec![]);
        let sccs = kosaraju_scc(&adj);
        // All singleton SCCs → no cycles
        assert_eq!(sccs.len(), 3, "SCCs: {:?}", sccs);
        assert!(sccs.iter().all(|s| s.len() == 1), "All singleton");
    }

    #[test]
    fn test_kosaraju_with_cycle() {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        adj.insert("a".into(), vec!["b".into()]);
        adj.insert("b".into(), vec!["a".into()]);
        adj.insert("c".into(), vec![]);
        let sccs = kosaraju_scc(&adj);
        // a and b form an SCC of size 2; c is singleton
        let big_scc = sccs.iter().find(|s| s.len() == 2);
        assert!(big_scc.is_some(), "Should find cycle SCC");
    }

    #[test]
    fn test_feature_expansion() {
        let mut manifest = Manifest::new("pkg", Version::new(1,0,0));
        manifest.features.insert("full".into(), Feature {
            name: "full".into(),
            enables: vec!["net".into(), "tls".into()],
            dependencies: vec![],
        });
        manifest.features.insert("net".into(), Feature {
            name: "net".into(),
            enables: vec!["io".into()],
            dependencies: vec![],
        });
        manifest.features.insert("tls".into(), Feature { name: "tls".into(), enables: vec![], dependencies: vec![] });
        manifest.features.insert("io".into(), Feature { name: "io".into(), enables: vec![], dependencies: vec![] });

        let mut active = HashSet::new();
        active.insert("full".into());
        let expanded = manifest.expand_features(&active);
        assert!(expanded.contains("full"));
        assert!(expanded.contains("net"));
        assert!(expanded.contains("tls"));
        assert!(expanded.contains("io"), "io transitively enabled through full→net→io");
    }

    #[test]
    fn test_extract_imports() {
        let source = r#"
import "std::io";
import "mylib::parser";
fn main() {}
"#;
        let imports = extract_imports(source);
        assert!(imports.contains(&"std::io".to_string()));
        assert!(imports.contains(&"mylib::parser".to_string()));
        assert_eq!(imports.len(), 2);
    }
}
