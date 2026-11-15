//! The layer-by-layer parallel build of the value table and its audit (DTH.md 4.8-4.9).

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use rayon::prelude::*;

use crate::{Error, MAX_LAYER, MAX_PHI, N, Profiles, RECHECK_SAMPLES, WIN, recheck, strided};

/// The `.npy` header of a C-order little-endian float64 `(N, N + 1)` array, as `numpy.save`
/// writes it: 128 bytes.
pub(crate) fn npy_header() -> Vec<u8> {
    let dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({N}, {}), }}",
        N + 1
    );
    let pad = 64 - (11 + dict.len()) % 64;
    let mut header = b"\x93NUMPY\x01\x00".to_vec();
    header.extend(((dict.len() + pad + 1) as u16).to_le_bytes());
    header.extend(format!("{dict}{}\n", " ".repeat(pad)).bytes());
    header
}

/// The value table under construction: `V[pc, pd]` at `pc * (N + 1) + pd`, NaN until solved, and
/// the column `WIN` at -1. Every class of a layer writes only its own cell and reads only finished
/// layers, so relaxed atomics share it across threads at the cost of plain loads and stores.
pub struct Values(Vec<AtomicU64>);

impl Values {
    fn new() -> Self {
        let cell = |i: usize| {
            if i % (N + 1) == WIN {
                -1.0_f64
            } else {
                f64::NAN
            }
        };
        Self(
            (0..N * (N + 1))
                .into_par_iter()
                .map(|i| AtomicU64::new(cell(i).to_bits()))
                .collect(),
        )
    }

    /// The entry at flat index `i`.
    #[must_use]
    pub fn get(&self, i: usize) -> f64 {
        f64::from_bits(self.0[i].load(Relaxed))
    }

    /// Write the table as `V.npy`, creating its directory.
    ///
    /// # Errors
    ///
    /// When the file cannot be written.
    pub fn write(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
        out.write_all(&npy_header())?;
        for v in &self.0 {
            out.write_all(&v.load(Relaxed).to_le_bytes())?;
        }
        out.flush()
    }
}

/// The rectangles `bucket[a] × bucket[P − a]` (Checker profiles, Dropper profiles) whose classes
/// make up layer `P`.
fn rectangles(profiles: &Profiles, layer: usize) -> impl Iterator<Item = (&[usize], &[usize])> {
    (layer.saturating_sub(MAX_PHI)..=layer.min(MAX_PHI))
        .map(move |a| (&profiles.bucket[a][..], &profiles.bucket[layer - a][..]))
}

/// Solve the layers `MAX_LAYER` down to `stop_at`, each in parallel, and return the table and the
/// classes routed to each rung (pure, support, LP). `on_layer(P, classes)` reports progress.
///
/// # Errors
///
/// [`Error::Uncertified`] when no rung certifies a class; nothing uncertified is ever stored.
pub fn build(
    profiles: &Profiles,
    stop_at: usize,
    mut on_layer: impl FnMut(usize, u64),
) -> Result<(Values, [u64; 3]), Error> {
    let values = Values::new();
    let mut routed = [0; 3];
    for layer in (stop_at..=MAX_LAYER).rev() {
        let rects: Vec<_> = rectangles(profiles, layer).collect();
        let counts = rects
            .par_iter()
            .flat_map(|&(pcs, pds)| pds.par_iter().map(move |&pd| (pcs, pd)))
            .map(|(pcs, pd)| {
                let mut counts = [0; 3];
                for &pc in pcs {
                    let cert = profiles.solve(pc, pd, |i| values.get(i))?;
                    values.0[pc * (N + 1) + pd].store(cert.value().to_bits(), Relaxed);
                    counts[cert.rung as usize] += 1;
                }
                Ok::<_, Error>(counts)
            })
            .try_reduce(|| [0; 3], |a, b| Ok(std::array::from_fn(|k| a[k] + b[k])))?;
        routed = std::array::from_fn(|k| routed[k] + counts[k]);
        on_layer(layer, counts.iter().sum());
    }
    Ok((values, routed))
}

/// Audit a finished table: every class solved and in `[-1, 1]`, and 1,200 strided classes
/// re-derived from their children. Returns the worst re-derivation difference.
///
/// # Errors
///
/// [`Error::Unsolved`] or a recheck failure.
pub fn finalize(values: &Values, profiles: &Profiles) -> Result<f64, Error> {
    (0..N).into_par_iter().try_for_each(|pc| {
        let bad = |v: f64| v.is_nan() || v.abs() > 1.0 + 1e-9;
        match (0..N).find(|&pd| bad(values.get(pc * (N + 1) + pd))) {
            Some(pd) => Err(Error::Unsolved(pc, pd)),
            None => Ok(()),
        }
    })?;
    strided(RECHECK_SAMPLES).try_fold(0.0, |worst: f64, c| {
        Ok(worst.max(recheck(c, |i| values.get(i), profiles)?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CLASSES;

    #[test]
    fn layer_census() {
        let t = Profiles::build();
        let count: Vec<usize> = (0..=MAX_LAYER)
            .map(|l| rectangles(&t, l).map(|(c, d)| c.len() * d.len()).sum())
            .collect();
        assert_eq!(count.iter().sum::<usize>(), CLASSES);
        assert_eq!(count[..601].iter().sum::<usize>(), 283_297_201);
        assert_eq!(count[601..841].iter().sum::<usize>(), 6_015_600);
        assert_eq!(count[841..].iter().sum::<usize>(), 61_320);
        assert_eq!(
            count.iter().enumerate().max_by_key(|&(_, n)| n),
            Some((374, &1_678_715))
        );
        for layer in [0, 374, 600, 901, 1200] {
            for (pcs, pds) in rectangles(&t, layer) {
                assert!(
                    pcs.iter()
                        .all(|&pc| pds.iter().all(|&pd| t.phi[pc] + t.phi[pd] == layer))
                );
            }
        }
    }

    #[test]
    fn npy_header_is_numpy_s() {
        let header = npy_header();
        assert_eq!(header.len(), 128);
        assert!(header.starts_with(b"\x93NUMPY\x01\x00\x76\x00{'descr': '<f8', 'fortran_order': False, 'shape': (17011, 17012), }"));
        assert!(header.ends_with(b"  \n"));
    }
}
