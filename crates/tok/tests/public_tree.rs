//! The public-tree arrays against the Python sizes and against a full walk of the rules engine.

use std::collections::HashSet;

use tok::rules::{GameState, Phase};
use tok::tree::{self, Layer, Sizes};

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("data/reference.json")).expect("reference.json parses")
}

/// (player, own beads, announced totals) of an announcement information set.
type Info = (usize, u8, Vec<u8>);

#[derive(Default)]
struct Walk {
    nodes: usize,
    wins: [usize; 2],
    draws: usize,
    infos: HashSet<Info>,
}

impl Walk {
    fn visit(&mut self, state: &GameState) {
        self.nodes += 1;
        match state.returns() {
            Some([0, 0]) => self.draws += 1,
            Some([payoff, _]) => self.wins[usize::from(payoff < 0)] += 1,
            None => {
                if let Phase::Guess(p) = state.phase() {
                    let own = state.beads()[p as usize].expect("announcers have beads");
                    let totals = state.history().iter().map(|miss| miss.total).collect();
                    self.infos.insert((p as usize, own, totals));
                }
                for action in state.legal_actions() {
                    self.visit(&state.apply(action).expect("legal action"));
                }
            }
        }
    }
}

fn walk(horizon: usize) -> Walk {
    let mut walk = Walk::default();
    walk.visit(&GameState::new(horizon).expect("positive horizon"));
    walk
}

fn array_infos(layers: &[Layer]) -> HashSet<Info> {
    let mut out = HashSet::new();
    for (d, layer) in layers.iter().enumerate() {
        let p = layer.actor;
        for (&node, &hand) in layer.slot_node[p].iter().zip(&layer.slot_hand[p]) {
            let (mut totals, mut node) = (vec![0; d], node as usize);
            for e in (1..=d).rev() {
                totals[e - 1] = layers[e].total[node];
                node = layers[e].parent[node] as usize;
            }
            out.insert((p, hand, totals));
        }
    }
    out
}

#[test]
fn sizes_match_python() {
    let reference = reference();
    for k in 1..=3 {
        let want = &reference["sizes"][k.to_string()];
        let count = |name: &str| want[name].as_u64().expect("a count") as usize;
        let pair = |name: &str| [0, 1].map(|p| want[name][p].as_u64().expect("a count") as usize);
        let sizes = tree::sizes(&tree::build(k).expect("positive horizon"));
        assert_eq!(
            sizes,
            Sizes {
                nodes: count("nodes"),
                hit_leaves: count("hit_leaves"),
                draw_leaves: count("draw_leaves"),
                public_nodes: count("public_nodes"),
                information_sets: pair("information_sets"),
                sequences: pair("sequences"),
            }
        );
    }
}

#[test]
fn d1_matches_the_object_tree_constants() {
    // kNodes, kLeaves, kP0Wins, kP1Wins, kDraws and the information sets of tree.py.
    let layers = tree::build(1).expect("positive horizon");
    let sizes = tree::sizes(&layers);
    assert_eq!(
        (sizes.nodes, sizes.hit_leaves + sizes.draw_leaves),
        (10_111, 9_100)
    );
    assert_eq!(
        (layers[0].pairs(), layers[1].pairs(), sizes.draw_leaves),
        (100, 900, 8_100)
    );
    assert_eq!(sizes.information_sets, [11, 189]);
    assert!(tree::build(0).is_err());
}

#[test]
fn arrays_match_a_walk_of_the_rules() {
    for k in 1..=2 {
        let layers = tree::build(k).expect("positive horizon");
        let (sizes, walk) = (tree::sizes(&layers), walk(k));
        assert_eq!(walk.nodes, sizes.nodes);
        assert_eq!(walk.wins[0] + walk.wins[1], sizes.hit_leaves);
        assert_eq!(
            walk.wins[0],
            layers.iter().step_by(2).map(Layer::pairs).sum::<usize>()
        );
        assert_eq!(walk.draws, sizes.draw_leaves);
        let infos = array_infos(&layers);
        assert_eq!(infos, walk.infos);
        let by_player = [0, 1].map(|p| 1 + infos.iter().filter(|info| info.0 == p).count());
        assert_eq!(by_player, sizes.information_sets);
    }
}
