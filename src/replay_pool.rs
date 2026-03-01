use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::vec::Vec;

use crate::octets::{add_assign, fused_addassign_mul_scalar, mulassign_scalar};
use crate::operation_vector::ReplayOp;
use crate::symbol_slab::SymbolSlab;

#[derive(Clone, Copy)]
struct ReplayJob {
    data_addr: usize,
    symbol_count: usize,
    symbol_size: usize,
    ops_addr: usize,
    ops_len: usize,
    chunk_size: usize,
    active_workers: usize,
}

#[derive(Default)]
struct ReplayPoolState {
    generation: u64,
    remaining: usize,
    shutdown: bool,
    job: Option<ReplayJob>,
}

struct ReplayPoolShared {
    state: Mutex<ReplayPoolState>,
    work_cv: Condvar,
    done_cv: Condvar,
}

pub(crate) struct ReplayPool {
    shared: Arc<ReplayPoolShared>,
    worker_count: usize,
    _workers: Vec<JoinHandle<()>>,
}

impl ReplayPool {
    fn new(worker_count: usize) -> ReplayPool {
        assert!(worker_count > 0);

        let shared = Arc::new(ReplayPoolShared {
            state: Mutex::new(ReplayPoolState::default()),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
        });

        let mut workers = Vec::with_capacity(worker_count);
        for worker_idx in 0..worker_count {
            let shared = Arc::clone(&shared);
            workers.push(
                thread::Builder::new()
                    .spawn(move || worker_main(shared, worker_idx))
                    .expect("failed to spawn replay pool worker"),
            );
        }

        ReplayPool {
            shared,
            worker_count,
            _workers: workers,
        }
    }

    #[inline]
    pub(crate) fn max_workers(&self) -> usize {
        self.worker_count
    }

    pub(crate) fn replay_ops_chunked(
        &self,
        ops: &[ReplayOp],
        symbols: &mut SymbolSlab,
        worker_count: usize,
    ) {
        debug_assert!(worker_count > 0);
        debug_assert!(worker_count <= self.worker_count);

        let symbol_size = symbols.symbol_size();
        let chunk_size = symbol_size.div_ceil(worker_count);
        let job = ReplayJob {
            data_addr: symbols.data_ptr_mut() as usize,
            symbol_count: symbols.len(),
            symbol_size,
            ops_addr: ops.as_ptr() as usize,
            ops_len: ops.len(),
            chunk_size,
            active_workers: worker_count,
        };

        let mut state = self
            .shared
            .state
            .lock()
            .expect("replay pool mutex poisoned while dispatching work");
        state.job = Some(job);
        state.remaining = worker_count;
        state.generation = state.generation.wrapping_add(1);
        let expected_generation = state.generation;
        self.shared.work_cv.notify_all();

        while state.generation == expected_generation && state.remaining > 0 {
            state = self
                .shared
                .done_cv
                .wait(state)
                .expect("replay pool mutex poisoned while awaiting completion");
        }
    }
}

impl Drop for ReplayPool {
    fn drop(&mut self) {
        {
            let mut state = self
                .shared
                .state
                .lock()
                .expect("replay pool mutex poisoned during shutdown");
            state.shutdown = true;
        }
        self.shared.work_cv.notify_all();

        for worker in self._workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[inline]
pub(crate) fn global_replay_pool() -> &'static ReplayPool {
    static POOL: OnceLock<ReplayPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let worker_count = thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
            .min(16)
            .max(1);
        ReplayPool::new(worker_count)
    })
}

fn worker_main(shared: Arc<ReplayPoolShared>, worker_idx: usize) {
    let mut last_generation = 0u64;

    loop {
        let (job, generation) = {
            let mut state = shared
                .state
                .lock()
                .expect("replay pool mutex poisoned while waiting for work");

            while !state.shutdown && state.generation == last_generation {
                state = shared
                    .work_cv
                    .wait(state)
                    .expect("replay pool mutex poisoned while parked");
            }

            if state.shutdown {
                return;
            }

            last_generation = state.generation;
            (
                state
                    .job
                    .expect("replay pool worker observed missing job after wake"),
                state.generation,
            )
        };

        if worker_idx < job.active_workers {
            let chunk_start = worker_idx * job.chunk_size;
            if chunk_start < job.symbol_size {
                let chunk_end = (chunk_start + job.chunk_size).min(job.symbol_size);
                unsafe {
                    replay_ops_chunk(job, chunk_start, chunk_end);
                }
            }

            let mut state = shared
                .state
                .lock()
                .expect("replay pool mutex poisoned while finishing work");
            if state.generation == generation {
                state.remaining = state.remaining.saturating_sub(1);
                if state.remaining == 0 {
                    shared.done_cv.notify_one();
                }
            }
        }
    }
}

unsafe fn replay_ops_chunk(job: ReplayJob, chunk_start: usize, chunk_end: usize) {
    let width = chunk_end - chunk_start;
    if width == 0 {
        return;
    }

    let data_ptr = job.data_addr as *mut u8;
    let ops = unsafe { core::slice::from_raw_parts(job.ops_addr as *const ReplayOp, job.ops_len) };

    for op in ops {
        match op {
            ReplayOp::AddAssign { dest, src } => {
                let dest = *dest as usize;
                let src = *src as usize;
                debug_assert!(dest < job.symbol_count);
                debug_assert!(src < job.symbol_count);
                debug_assert_ne!(dest, src);
                let dest_offset = dest * job.symbol_size + chunk_start;
                let src_offset = src * job.symbol_size + chunk_start;
                let (dest_slice, src_slice) = unsafe {
                    (
                        core::slice::from_raw_parts_mut(data_ptr.add(dest_offset), width),
                        core::slice::from_raw_parts(data_ptr.add(src_offset), width),
                    )
                };
                add_assign(dest_slice, src_slice);
            }
            ReplayOp::MulAssign { dest, scalar } => {
                let dest = *dest as usize;
                debug_assert!(dest < job.symbol_count);
                let dest_offset = dest * job.symbol_size + chunk_start;
                let dest_slice =
                    unsafe { core::slice::from_raw_parts_mut(data_ptr.add(dest_offset), width) };
                mulassign_scalar(dest_slice, scalar);
            }
            ReplayOp::FMA { dest, src, scalar } => {
                let dest = *dest as usize;
                let src = *src as usize;
                debug_assert!(dest < job.symbol_count);
                debug_assert!(src < job.symbol_count);
                debug_assert_ne!(dest, src);
                let dest_offset = dest * job.symbol_size + chunk_start;
                let src_offset = src * job.symbol_size + chunk_start;
                let (dest_slice, src_slice) = unsafe {
                    (
                        core::slice::from_raw_parts_mut(data_ptr.add(dest_offset), width),
                        core::slice::from_raw_parts(data_ptr.add(src_offset), width),
                    )
                };
                fused_addassign_mul_scalar(dest_slice, src_slice, scalar);
            }
            ReplayOp::Reorder { .. } => unreachable!(),
        }
    }
}
