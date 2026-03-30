//! Integration tests for chronos-graph.
use chronos_graph::*;

#[test]
fn test_graph_can_be_constructed() {
    let mut g = Graph::new(2, false);
    g.add_edge(0, 1, 1.0);
    assert_eq!(g.n, 2);
    assert_eq!(g.edges.len(), 1);
}

#[test]
fn test_bfs_visits_all_reachable() {
    let mut g = Graph::new(3, false);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    let (visited, _, _) = bfs(&g, 0);
    assert!(visited.contains(&0));
    assert!(visited.contains(&1));
    assert!(visited.contains(&2));
}

#[test]
fn test_dijkstra_shortest_path() {
    let mut g = Graph::new(3, true);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 2.0);
    g.add_edge(0, 2, 10.0);
    let (dist, _) = dijkstra(&g, 0);
    // 0→1→2 = 3.0, cheaper than 0→2 = 10.0
    assert!(dist[2] <= 3.0 + 1e-9);
}

#[test]
fn test_disconnected_graph() {
    let mut g = Graph::new(2, false);
    // no edge between 0 and 1
    let (visited, dist, _) = bfs(&g, 0);
    assert!(visited.contains(&0));
    assert!(!visited.contains(&1));
    assert_eq!(dist[1], -1); // unreachable
}

#[test]
fn test_self_loop_does_not_panic() {
    let mut g = Graph::new(1, false);
    g.add_edge(0, 0, 0.0);
    let _ = bfs(&g, 0);
}
