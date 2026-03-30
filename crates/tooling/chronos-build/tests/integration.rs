//! Integration tests for chronos-build.
use chronos_build::*;

#[test]
fn test_version_parse_and_display() {
    let v = Version::parse("1.2.3").expect("valid semver");
    assert_eq!(v.major, 1);
    assert_eq!(v.minor, 2);
    assert_eq!(v.patch, 3);
    assert_eq!(v.to_string(), "1.2.3");
}

#[test]
fn test_version_ordering() {
    let v1 = Version::parse("1.0.0").unwrap();
    let v2 = Version::parse("2.0.0").unwrap();
    let v3 = Version::parse("1.1.0").unwrap();
    assert!(v1 < v2);
    assert!(v1 < v3);
    assert!(v3 < v2);
}

#[test]
fn test_prerelease_is_less_than_release() {
    let pre     = Version::parse("1.0.0-alpha.1").unwrap();
    let release = Version::parse("1.0.0").unwrap();
    assert!(pre < release);
}

#[test]
fn test_caret_requirement_matches_compatible() {
    let req = VersionReq::parse("^1.2.3").unwrap();
    assert!(req.matches(&Version::parse("1.2.3").unwrap()));
    assert!(req.matches(&Version::parse("1.9.0").unwrap()));
    assert!(!req.matches(&Version::parse("2.0.0").unwrap()));
    assert!(!req.matches(&Version::parse("1.2.2").unwrap()));
}

#[test]
fn test_tilde_requirement_matches_patch_only() {
    let req = VersionReq::parse("~1.2.3").unwrap();
    assert!(req.matches(&Version::parse("1.2.3").unwrap()));
    assert!(req.matches(&Version::parse("1.2.9").unwrap()));
    assert!(!req.matches(&Version::parse("1.3.0").unwrap()));
}

#[test]
fn test_dep_graph_topo_sort_order() {
    let mut graph = DepGraph::new();
    graph.add_node(DepNode {
        package: "core".into(),
        version: Version::new(1, 0, 0),
        deps: vec![],
    });
    graph.add_node(DepNode {
        package: "utils".into(),
        version: Version::new(1, 0, 0),
        deps: vec![("core".into(), VersionReq::Any)],
    });
    graph.add_node(DepNode {
        package: "app".into(),
        version: Version::new(1, 0, 0),
        deps: vec![("utils".into(), VersionReq::Any)],
    });

    let order = graph.topo_sort().expect("no cycles");
    let pos_core  = order.iter().position(|n| n == "core").unwrap();
    let pos_utils = order.iter().position(|n| n == "utils").unwrap();
    let pos_app   = order.iter().position(|n| n == "app").unwrap();
    assert!(pos_core < pos_utils, "core must come before utils");
    assert!(pos_utils < pos_app,  "utils must come before app");
}

#[test]
fn test_dep_graph_detects_cycle() {
    let mut graph = DepGraph::new();
    graph.add_node(DepNode {
        package: "a".into(),
        version: Version::new(1, 0, 0),
        deps: vec![("b".into(), VersionReq::Any)],
    });
    graph.add_node(DepNode {
        package: "b".into(),
        version: Version::new(1, 0, 0),
        deps: vec![("a".into(), VersionReq::Any)],
    });
    assert!(graph.detect_cycles().is_err());
}

#[test]
fn test_fnv1a_hash_deterministic() {
    let h1 = fnv1a_hash(b"hello world");
    let h2 = fnv1a_hash(b"hello world");
    assert_eq!(h1, h2);
    assert_ne!(h1, fnv1a_hash(b"different"));
}

#[test]
fn test_build_cache_hit_and_miss() {
    let mut cache = BuildCache::new();
    let hash = fnv1a_hash(b"source content");
    assert!(cache.needs_rebuild("my_module", hash));
    cache.record_build("my_module".into(), hash, "/tmp/my_module.o".into());
    assert!(!cache.needs_rebuild("my_module", hash));
    assert!(cache.needs_rebuild("my_module", hash + 1)); // different hash
}
