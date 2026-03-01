#[cfg(feature = "std")]
use std::{thread, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::octet::Octet;
use crate::octets::{add_assign, fused_addassign_mul_scalar, mulassign_scalar};
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

fn perform_ops_sequential(ops: &[SymbolOps], symbols: &mut SymbolSlab) {
    for op in ops {
        perform_op(op, symbols);
    }
}

#[cfg(feature = "std")]
fn should_parallelize(op_count: usize, symbol_count: usize, symbol_size: usize) -> bool {
    // Keep small jobs on the sequential fast path.
    op_count >= 1_024 && symbol_count >= 512 && symbol_size >= 512
}

#[cfg(feature = "std")]
unsafe fn replay_ops_chunk(
    data_addr: usize,
    symbol_count: usize,
    symbol_size: usize,
    ops: &[SymbolOps],
    chunk_start: usize,
    chunk_end: usize,
) {
    let width = chunk_end - chunk_start;
    if width == 0 {
        return;
    }
    let data_ptr = data_addr as *mut u8;

    for op in ops {
        match op {
            SymbolOps::AddAssign { dest, src } => {
                debug_assert!(*dest < symbol_count);
                debug_assert!(*src < symbol_count);
                debug_assert_ne!(*dest, *src);
                let dest_offset = *dest * symbol_size + chunk_start;
                let src_offset = *src * symbol_size + chunk_start;
                let (dest_slice, src_slice) = unsafe {
                    (
                        core::slice::from_raw_parts_mut(data_ptr.add(dest_offset), width),
                        core::slice::from_raw_parts(data_ptr.add(src_offset), width),
                    )
                };
                add_assign(dest_slice, src_slice);
            }
            SymbolOps::MulAssign { dest, scalar } => {
                debug_assert!(*dest < symbol_count);
                let dest_offset = *dest * symbol_size + chunk_start;
                let dest_slice =
                    unsafe { core::slice::from_raw_parts_mut(data_ptr.add(dest_offset), width) };
                mulassign_scalar(dest_slice, scalar);
            }
            SymbolOps::FMA { dest, src, scalar } => {
                debug_assert!(*dest < symbol_count);
                debug_assert!(*src < symbol_count);
                debug_assert_ne!(*dest, *src);
                let dest_offset = *dest * symbol_size + chunk_start;
                let src_offset = *src * symbol_size + chunk_start;
                let (dest_slice, src_slice) = unsafe {
                    (
                        core::slice::from_raw_parts_mut(data_ptr.add(dest_offset), width),
                        core::slice::from_raw_parts(data_ptr.add(src_offset), width),
                    )
                };
                fused_addassign_mul_scalar(dest_slice, src_slice, scalar);
            }
            SymbolOps::Reorder { .. } => unreachable!(),
        }
    }
}

#[cfg(feature = "std")]
fn perform_ops_parallel(ops: &[SymbolOps], symbols: &mut SymbolSlab) -> bool {
    let available_parallelism = thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    if available_parallelism <= 1 {
        return false;
    }

    let reorder_pos = ops
        .iter()
        .position(|op| matches!(op, SymbolOps::Reorder { .. }));
    if let Some(pos) = reorder_pos {
        if pos + 1 != ops.len() {
            return false;
        }
    }
    let replay_ops = &ops[..reorder_pos.unwrap_or(ops.len())];
    if replay_ops
        .iter()
        .any(|op| matches!(op, SymbolOps::AddAssign { dest, src } | SymbolOps::FMA { dest, src, .. } if dest == src))
    {
        return false;
    }

    let symbol_count = symbols.len();
    let symbol_size = symbols.symbol_size();
    if !should_parallelize(replay_ops.len(), symbol_count, symbol_size) {
        return false;
    }

    let worker_count = available_parallelism
        .min(symbol_size.max(1))
        .min((symbol_size / 64).max(1));
    if worker_count <= 1 {
        return false;
    }

    let chunk_size = symbol_size.div_ceil(worker_count);
    let data_addr = symbols.data_ptr_mut() as usize;

    thread::scope(|scope| {
        for worker_idx in 0..worker_count {
            let chunk_start = worker_idx * chunk_size;
            if chunk_start >= symbol_size {
                continue;
            }
            let chunk_end = (chunk_start + chunk_size).min(symbol_size);
            scope.spawn(move || unsafe {
                replay_ops_chunk(
                    data_addr,
                    symbol_count,
                    symbol_size,
                    replay_ops,
                    chunk_start,
                    chunk_end,
                );
            });
        }
    });

    if let Some(pos) = reorder_pos {
        if let SymbolOps::Reorder { order } = &ops[pos] {
            symbols.reorder(order);
        }
    }
    true
}

pub fn perform_ops(ops: &[SymbolOps], symbols: &mut SymbolSlab) {
    #[cfg(feature = "std")]
    if perform_ops_parallel(ops, symbols) {
        return;
    }
    perform_ops_sequential(ops, symbols);
}
#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use rand::{Rng, seq::SliceRandom};
    use std::vec::Vec;

    use crate::octet::Octet;
    use crate::operation_vector::{SymbolOps, perform_op, perform_ops};
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
}
