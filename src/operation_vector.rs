#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;
#[cfg(feature = "std")]
use crate::replay_pool::global_replay_pool;
use crate::symbol_slab::SymbolSlab;
#[cfg(feature = "serde_support")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
#[cfg_attr(feature = "serde_support", derive(Serialize, Deserialize))]
#[allow(clippy::upper_case_acronyms)]
pub enum SymbolOps {
    AddAssign {
        dest: usize,
        src: usize,
    },
    MulAssign {
        dest: usize,
        scalar: Octet,
    },
    FMA {
        dest: usize,
        src: usize,
        scalar: Octet,
    },
    Reorder {
        order: Vec<usize>,
    },
}

#[inline(always)]
pub fn perform_op(op: &SymbolOps, symbols: &mut SymbolSlab) {
    match op {
        SymbolOps::AddAssign { dest, src } => {
            symbols.add_assign(*dest, *src);
        }
        SymbolOps::MulAssign { dest, scalar } => {
            symbols.mulassign_scalar(*dest, scalar);
        }
        SymbolOps::FMA { dest, src, scalar } => {
            symbols.fma(*dest, *src, scalar);
        }
        SymbolOps::Reorder { order } => {
            symbols.reorder(order);
        }
    }
}

#[inline(always)]
fn perform_symbol_ops_sequential(ops: &[SymbolOps], symbols: &mut SymbolSlab) {
    for op in ops {
        perform_op(op, symbols);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplayOp {
    AddAssign { dest: u32, src: u32 },
    MulAssign { dest: u32, scalar: Octet },
    FMA { dest: u32, src: u32, scalar: Octet },
    Reorder { order: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplaySchedule {
    ops: Vec<ReplayOp>,
}

impl ReplaySchedule {
    #[inline]
    pub fn from_symbol_ops(ops: &[SymbolOps]) -> Self {
        let mut replay_ops = Vec::with_capacity(ops.len());
        for op in ops {
            match op {
                SymbolOps::AddAssign { dest, src } => replay_ops.push(ReplayOp::AddAssign {
                    dest: u32::try_from(*dest).expect("symbol index overflow"),
                    src: u32::try_from(*src).expect("symbol index overflow"),
                }),
                SymbolOps::MulAssign { dest, scalar } => replay_ops.push(ReplayOp::MulAssign {
                    dest: u32::try_from(*dest).expect("symbol index overflow"),
                    scalar: scalar.clone(),
                }),
                SymbolOps::FMA { dest, src, scalar } => replay_ops.push(ReplayOp::FMA {
                    dest: u32::try_from(*dest).expect("symbol index overflow"),
                    src: u32::try_from(*src).expect("symbol index overflow"),
                    scalar: scalar.clone(),
                }),
                SymbolOps::Reorder { order } => replay_ops.push(ReplayOp::Reorder {
                    order: order.clone(),
                }),
            }
        }

        ReplaySchedule { ops: replay_ops }
    }

    #[inline]
    fn as_slice(&self) -> &[ReplayOp] {
        &self.ops
    }
}

#[inline(always)]
fn perform_replay_op(op: &ReplayOp, symbols: &mut SymbolSlab) {
    match op {
        ReplayOp::AddAssign { dest, src } => {
            symbols.add_assign(*dest as usize, *src as usize);
        }
        ReplayOp::MulAssign { dest, scalar } => {
            symbols.mulassign_scalar(*dest as usize, scalar);
        }
        ReplayOp::FMA { dest, src, scalar } => {
            symbols.fma(*dest as usize, *src as usize, scalar);
        }
        ReplayOp::Reorder { order } => {
            symbols.reorder(order);
        }
    }
}

fn perform_replay_schedule_sequential(schedule: &ReplaySchedule, symbols: &mut SymbolSlab) {
    for op in schedule.as_slice() {
        perform_replay_op(op, symbols);
    }
}

#[cfg(feature = "std")]
fn should_parallelize(op_count: usize, symbol_count: usize, symbol_size: usize) -> bool {
    // Keep all but very large workloads on the sequential fast path; replay setup and
    // synchronization overhead dominate medium-size blocks.
    op_count >= 30_000 && symbol_count >= 4_000 && symbol_size >= 1_024
}

#[cfg(feature = "std")]
fn perform_replay_schedule_parallel(schedule: &ReplaySchedule, symbols: &mut SymbolSlab) -> bool {
    let pool = global_replay_pool();
    let available_parallelism = pool.max_workers();
    if available_parallelism <= 1 {
        return false;
    }

    let ops = schedule.as_slice();
    let reorder_pos = ops
        .iter()
        .position(|op| matches!(op, ReplayOp::Reorder { .. }));
    if let Some(pos) = reorder_pos {
        if pos + 1 != ops.len() {
            return false;
        }
    }
    let replay_ops = &ops[..reorder_pos.unwrap_or(ops.len())];
    if replay_ops.iter().any(|op| {
        matches!(op, ReplayOp::AddAssign { dest, src } | ReplayOp::FMA { dest, src, .. } if dest == src)
    }) {
        return false;
    }

    let symbol_count = symbols.len();
    let symbol_size = symbols.symbol_size();
    if !should_parallelize(replay_ops.len(), symbol_count, symbol_size) {
        return false;
    }

    let max_workers_by_chunk = symbol_size.div_ceil(256).max(1);
    let max_workers_by_ops = replay_ops.len().div_ceil(12_000).max(1);
    let worker_count = available_parallelism
        .min(max_workers_by_chunk)
        .min(max_workers_by_ops);
    if worker_count <= 1 {
        return false;
    }

    pool.replay_ops_chunked(replay_ops, symbols, worker_count);

    if let Some(pos) = reorder_pos {
        if let ReplayOp::Reorder { order } = &ops[pos] {
            symbols.reorder(order);
        }
    }
    true
}

pub fn perform_replay_schedule_with_parallel_hint(
    schedule: &ReplaySchedule,
    symbols: &mut SymbolSlab,
    parallel_hint: bool,
) {
    #[cfg(feature = "std")]
    if parallel_hint && perform_replay_schedule_parallel(schedule, symbols) {
        return;
    }
    perform_replay_schedule_sequential(schedule, symbols);
}

pub fn perform_ops_with_parallel_hint(
    ops: &[SymbolOps],
    symbols: &mut SymbolSlab,
    parallel_hint: bool,
) {
    if !parallel_hint {
        perform_symbol_ops_sequential(ops, symbols);
        return;
    }

    let schedule = ReplaySchedule::from_symbol_ops(ops);
    perform_replay_schedule_with_parallel_hint(&schedule, symbols, true);
}

#[allow(dead_code)]
pub fn perform_ops(ops: &[SymbolOps], symbols: &mut SymbolSlab) {
    perform_ops_with_parallel_hint(ops, symbols, true);
}
#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use rand::{Rng, seq::SliceRandom};
    use std::vec::Vec;

    use crate::octet::Octet;
    use crate::operation_vector::{ReplayOp, ReplaySchedule, SymbolOps, perform_op, perform_ops};
    use crate::replay_pool::global_replay_pool;
    use crate::symbol::Symbol;
    use crate::symbol_slab::SymbolSlab;

    #[test]
    fn test_add() {
        let symbol_size = 1316;
        let mut raw0: Vec<u8> = vec![0; symbol_size];
        let mut raw1: Vec<u8> = vec![0; symbol_size];
        for b in raw0.iter_mut() {
            *b = rand::rng().random();
        }
        for b in raw1.iter_mut() {
            *b = rand::rng().random();
        }
        let expected: Vec<u8> = raw0.iter().zip(raw1.iter()).map(|(a, b)| a ^ b).collect();

        let mut slab =
            SymbolSlab::from_symbols(vec![Symbol::new(raw0), Symbol::new(raw1)], symbol_size);
        perform_op(&SymbolOps::AddAssign { dest: 0, src: 1 }, &mut slab);
        assert_eq!(expected, slab.get(0));
    }

    #[test]
    fn test_add_mul() {
        let symbol_size = 1316;
        let mut raw0: Vec<u8> = vec![0; symbol_size];
        let mut raw1: Vec<u8> = vec![0; symbol_size];
        for b in raw0.iter_mut() {
            *b = rand::rng().random();
        }
        for b in raw1.iter_mut() {
            *b = rand::rng().random();
        }
        let value = 173;
        let expected: Vec<u8> = raw0
            .iter()
            .zip(raw1.iter())
            .map(|(d0, d1)| *d0 ^ (Octet::new(*d1) * Octet::new(value)).byte())
            .collect();

        let mut slab =
            SymbolSlab::from_symbols(vec![Symbol::new(raw0), Symbol::new(raw1)], symbol_size);
        perform_op(
            &SymbolOps::FMA {
                dest: 0,
                src: 1,
                scalar: Octet::new(value),
            },
            &mut slab,
        );
        assert_eq!(expected, slab.get(0));
    }

    #[test]
    fn test_mul() {
        let symbol_size = 1316;
        let mut raw0: Vec<u8> = vec![0; symbol_size];
        for b in raw0.iter_mut() {
            *b = rand::rng().random();
        }
        let value = 215;
        let expected: Vec<u8> = raw0
            .iter()
            .map(|d0| (Octet::new(*d0) * Octet::new(value)).byte())
            .collect();

        let mut slab = SymbolSlab::from_symbols(vec![Symbol::new(raw0)], symbol_size);
        perform_op(
            &SymbolOps::MulAssign {
                dest: 0,
                scalar: Octet::new(value),
            },
            &mut slab,
        );
        assert_eq!(expected, slab.get(0));
    }

    #[test]
    fn test_reorder() {
        let rows = 10;
        let symbol_size = 10;
        let symbols: Vec<Symbol> = (0..rows)
            .map(|i| Symbol::new(vec![i as u8; symbol_size]))
            .collect();
        let mut slab = SymbolSlab::from_symbols(symbols, symbol_size);

        assert_eq!(slab.get(0)[0], 0);
        assert_eq!(slab.get(1)[0], 1);
        assert_eq!(slab.get(2)[0], 2);
        assert_eq!(slab.get(9)[0], 9);

        perform_op(
            &SymbolOps::Reorder {
                order: vec![9, 7, 5, 3, 1, 8, 0, 6, 2, 4],
            },
            &mut slab,
        );
        assert_eq!(slab.get(0)[0], 9);
        assert_eq!(slab.get(1)[0], 7);
        assert_eq!(slab.get(2)[0], 5);
        assert_eq!(slab.get(3)[0], 3);
        assert_eq!(slab.get(4)[0], 1);
        assert_eq!(slab.get(5)[0], 8);
        assert_eq!(slab.get(6)[0], 0);
        assert_eq!(slab.get(7)[0], 6);
        assert_eq!(slab.get(8)[0], 2);
        assert_eq!(slab.get(9)[0], 4);
    }
    #[test]
    fn test_perform_ops_matches_scalar_executor() {
        let symbol_count = 96;
        let symbol_size = 1024;
        let mut symbols = Vec::with_capacity(symbol_count);
        for _ in 0..symbol_count {
            let mut row = vec![0u8; symbol_size];
            for b in row.iter_mut() {
                *b = rand::rng().random();
            }
            symbols.push(Symbol::new(row));
        }

        let mut expected = SymbolSlab::from_symbols(symbols.clone(), symbol_size);
        let mut actual = SymbolSlab::from_symbols(symbols, symbol_size);

        let mut ops = Vec::with_capacity(1_501);
        for i in 0..1_500 {
            let dest = rand::rng().random_range(0..symbol_count);
            let mut src = rand::rng().random_range(0..symbol_count);
            while src == dest {
                src = rand::rng().random_range(0..symbol_count);
            }
            match i % 3 {
                0 => ops.push(SymbolOps::AddAssign { dest, src }),
                1 => ops.push(SymbolOps::MulAssign {
                    dest,
                    scalar: Octet::new(rand::rng().random_range(1..=255)),
                }),
                _ => ops.push(SymbolOps::FMA {
                    dest,
                    src,
                    scalar: Octet::new(rand::rng().random_range(2..=255)),
                }),
            }
        }

        let mut reorder: Vec<usize> = (0..symbol_count).collect();
        reorder.shuffle(&mut rand::rng());
        ops.push(SymbolOps::Reorder { order: reorder });

        for op in ops.iter() {
            perform_op(op, &mut expected);
        }
        perform_ops(&ops, &mut actual);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_replay_pool_chunked_matches_scalar_executor() {
        let pool = global_replay_pool();
        let worker_count = pool.max_workers().min(4);
        if worker_count <= 1 {
            return;
        }

        let symbol_count = 256;
        let symbol_size = 640;
        let mut symbols = Vec::with_capacity(symbol_count);
        for _ in 0..symbol_count {
            let mut row = vec![0u8; symbol_size];
            for b in row.iter_mut() {
                *b = rand::rng().random();
            }
            symbols.push(Symbol::new(row));
        }

        let mut expected = SymbolSlab::from_symbols(symbols.clone(), symbol_size);
        let mut actual = SymbolSlab::from_symbols(symbols, symbol_size);

        let mut ops = Vec::with_capacity(6_001);
        for i in 0..6_000 {
            let dest = rand::rng().random_range(0..symbol_count);
            let mut src = rand::rng().random_range(0..symbol_count);
            while src == dest {
                src = rand::rng().random_range(0..symbol_count);
            }
            match i % 3 {
                0 => ops.push(SymbolOps::AddAssign { dest, src }),
                1 => ops.push(SymbolOps::MulAssign {
                    dest,
                    scalar: Octet::new(rand::rng().random_range(1..=255)),
                }),
                _ => ops.push(SymbolOps::FMA {
                    dest,
                    src,
                    scalar: Octet::new(rand::rng().random_range(2..=255)),
                }),
            }
        }

        let mut reorder: Vec<usize> = (0..symbol_count).collect();
        reorder.shuffle(&mut rand::rng());
        ops.push(SymbolOps::Reorder { order: reorder });

        for op in ops.iter() {
            perform_op(op, &mut expected);
        }

        let schedule = ReplaySchedule::from_symbol_ops(&ops);
        let schedule_ops = schedule.as_slice();
        let reorder_pos = schedule_ops
            .iter()
            .position(|op| matches!(op, ReplayOp::Reorder { .. }))
            .expect("reorder op missing");
        pool.replay_ops_chunked(&schedule_ops[..reorder_pos], &mut actual, worker_count);
        if let ReplayOp::Reorder { order } = &schedule_ops[reorder_pos] {
            actual.reorder(order);
        }

        assert_eq!(expected, actual);
    }
}
