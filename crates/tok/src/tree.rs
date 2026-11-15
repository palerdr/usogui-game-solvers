//! The game tree of `D_K` as arrays, organized by public transcript.
//!
//! Public nodes are sorted by (parent node, announced total), slots by (public node, own beads),
//! and alive pairs by public node. Column a of an actor slot announces own beads + a + 1.

use crate::{Error, HANDS};

/// Marks an action whose announcement hits every alive pair or ends the game.
pub(crate) const NONE: u32 = u32::MAX;

/// One decision layer: the public nodes, alive pairs and slots of one turn of one player.
#[derive(Debug)]
pub struct Layer {
    /// The player who acts on this layer.
    pub actor: usize,
    /// Public node -> parent public node one layer up (0 on the root layer).
    pub parent: Vec<u32>,
    /// Public node -> the announcement that led to it (0 on the root layer).
    pub total: Vec<u8>,
    /// Alive pair -> beads (x0, x1).
    pub(crate) pair_hand: [Vec<u8>; 2],
    /// Alive pair -> slot of each player.
    pub(crate) pair_slot: [Vec<u32>; 2],
    /// Slot -> public node, per player.
    pub slot_node: [Vec<u32>; 2],
    /// Slot -> own beads, per player.
    pub slot_hand: [Vec<u8>; 2],
    /// Slot -> the same player's slot one layer up (empty on the root layer).
    pub(crate) slot_parent: [Vec<u32>; 2],
    /// Actor slot x action -> the actor's slot one layer down, or [`NONE`].
    pub(crate) child_slot: Vec<u32>,
}

impl Layer {
    fn new(
        actor: usize,
        parent: Vec<u32>,
        total: Vec<u8>,
        pair_node: &[u32],
        pair_hand: [Vec<u8>; 2],
    ) -> Self {
        let [(n0, h0, s0), (n1, h1, s1)] = pair_hand.each_ref().map(|hand| slots(pair_node, hand));
        let child_slot = vec![NONE; HANDS * [n0.len(), n1.len()][actor]];
        Self {
            actor,
            parent,
            total,
            pair_hand,
            pair_slot: [s0, s1],
            slot_node: [n0, n1],
            slot_hand: [h0, h1],
            slot_parent: [Vec::new(), Vec::new()],
            child_slot,
        }
    }

    /// The number of alive pairs, which is the number of full-tree decision nodes here.
    #[must_use]
    pub fn pairs(&self) -> usize {
        self.pair_slot[0].len()
    }

    /// Player q's number of slots.
    #[must_use]
    pub fn slots(&self, q: usize) -> usize {
        self.slot_node[q].len()
    }

    /// The action of q's announcement one layer up that led to q's slot s. Valid when q acted
    /// on the layer above.
    #[must_use]
    pub(crate) fn arrival(&self, q: usize, s: usize) -> usize {
        usize::from(self.total[self.slot_node[q][s] as usize] - self.slot_hand[q][s] - 1)
    }

    /// Flat (actor slot, action) index of the announcement that hits alive pair k.
    #[must_use]
    pub(crate) fn hit(&self, k: usize) -> usize {
        let p = self.actor;
        HANDS * self.pair_slot[p][k] as usize + usize::from(self.pair_hand[1 - p][k]) - 1
    }
}

/// Slots of one player from pairs sorted by node: (slot node, slot hand, pair slot).
fn slots(pair_node: &[u32], hand: &[u8]) -> (Vec<u32>, Vec<u8>, Vec<u32>) {
    let (mut slot_node, mut slot_hand) = (Vec::new(), Vec::new());
    let mut pair_slot = Vec::with_capacity(hand.len());
    let mut start = 0;
    for run in pair_node.chunk_by(|a, b| a == b) {
        let hands = &hand[start..start + run.len()];
        let mut id = [NONE; HANDS];
        for &h in hands {
            id[usize::from(h) - 1] = 0;
        }
        for (h, slot) in (1..).zip(&mut id) {
            if *slot == 0 {
                *slot = slot_node.len() as u32;
                slot_node.push(run[0]);
                slot_hand.push(h);
            }
        }
        pair_slot.extend(hands.iter().map(|&h| id[usize::from(h) - 1]));
        start += run.len();
    }
    (slot_node, slot_hand, pair_slot)
}

/// The 2 * horizon decision layers of `D_horizon`.
///
/// # Errors
///
/// [`Error::Horizon`] when `horizon` is 0.
pub fn build(horizon: usize) -> Result<Vec<Layer>, Error> {
    if horizon == 0 {
        return Err(Error::Horizon);
    }
    let grid = 1..=HANDS as u8;
    let x0 = grid.clone().flat_map(|x| [x; HANDS]).collect();
    let x1 = grid.cycle().take(HANDS * HANDS).collect();
    let mut pair_node = vec![0; HANDS * HANDS];
    let mut layers = vec![Layer::new(0, vec![0], vec![0], &pair_node, [x0, x1])];
    for d in 1..2 * horizon {
        let prev = &layers[d - 1];
        let p = prev.actor;
        let (mut parent, mut total, mut node, mut from) = (vec![], vec![], vec![], vec![]);
        let mut start = 0;
        for run in pair_node.chunk_by(|a, b| a == b) {
            let range = start..start + run.len();
            for g in 2..=2 * HANDS as u8 {
                let before = from.len();
                for k in range.clone() {
                    let own = prev.pair_hand[p][k];
                    let sum = prev.pair_hand[0][k] + prev.pair_hand[1][k];
                    if own < g && g <= own + HANDS as u8 && g != sum {
                        node.push(parent.len() as u32);
                        from.push(k);
                    }
                }
                if from.len() > before {
                    parent.push(run[0]);
                    total.push(g);
                }
            }
            start = range.end;
        }
        let hand = [0, 1].map(|q| from.iter().map(|&k| prev.pair_hand[q][k]).collect());
        let mut layer = Layer::new(1 - p, parent, total, &node, hand);
        for q in 0..2 {
            let mut up = vec![0; layer.slots(q)];
            for (&s, &k) in layer.pair_slot[q].iter().zip(&from) {
                up[s as usize] = prev.pair_slot[q][k];
            }
            layer.slot_parent[q] = up;
        }
        for (s, &k) in layer.slot_parent[p].iter().enumerate() {
            layers[d - 1].child_slot[HANDS * k as usize + layer.arrival(p, s)] = s as u32;
        }
        pair_node = node;
        layers.push(layer);
    }
    Ok(layers)
}

/// The deepest layer on which player i acts.
#[must_use]
pub fn last_layer(layers: &[Layer], i: usize) -> usize {
    layers.len() - 2 + i
}

/// Counts of the full extensive-form tree that the layers represent.
#[derive(Debug, PartialEq, Eq)]
pub struct Sizes {
    /// All nodes: the root, the 10 P1 bead picks, decisions and leaves.
    pub nodes: usize,
    /// Leaves where an announcement hits; one per decision node.
    pub hit_leaves: usize,
    /// Leaves where the last announcement misses.
    pub draw_leaves: usize,
    /// Public nodes over all layers.
    pub public_nodes: usize,
    /// Information sets of each player, the bead pick included.
    pub information_sets: [usize; 2],
    /// Sequences of each player, the empty sequence included.
    pub sequences: [usize; 2],
}

/// Count the full extensive-form tree that `layers` represent.
#[must_use]
pub fn sizes(layers: &[Layer]) -> Sizes {
    let decisions: usize = layers.iter().map(Layer::pairs).sum();
    let draws = layers.last().map_or(0, Layer::pairs) * (HANDS - 1);
    let mut information_sets = [1, 1];
    for layer in layers {
        information_sets[layer.actor] += layer.slots(layer.actor);
    }
    Sizes {
        nodes: 1 + HANDS + 2 * decisions + draws,
        hit_leaves: decisions,
        draw_leaves: draws,
        public_nodes: layers.iter().map(|layer| layer.parent.len()).sum(),
        information_sets,
        sequences: information_sets.map(|n| 1 + HANDS + HANDS * (n - 1)),
    }
}
