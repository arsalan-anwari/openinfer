use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{self, Receiver};

use anyhow::{anyhow, Result};
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

use crate::runtime::executor::run_block;
use crate::runtime::yield_await::YieldSnapshot;
use crate::runtime::state::{RuntimeState, SharedTensor};

#[allow(dead_code)]
pub struct ConsumerResult {
    pub block_name: String,
    pub yielded: HashMap<String, SharedTensor>,
    pub mutated: HashSet<String>,
}

pub struct AsyncScheduler {
    pool: ThreadPool,
    async_blocks: Vec<String>,
    pending: Vec<Receiver<Result<ConsumerResult>>>,
}

impl AsyncScheduler {
    pub fn new(async_blocks: Vec<String>) -> Result<Self> {
        let pool = ThreadPoolBuilder::new()
            .build()
            .map_err(|err| anyhow!("failed to build async threadpool: {}", err))?;
        Ok(Self {
            pool,
            async_blocks,
            pending: Vec::new(),
        })
    }

    pub fn on_yield(&mut self, entry_state: &RuntimeState, snapshot: &YieldSnapshot) -> Result<()> {
        if !self.pending.is_empty() {
            return Err(anyhow!("await required before next yield"));
        }
        if self.async_blocks.is_empty() {
            return Ok(());
        }
        let vars = snapshot.vars.keys().cloned().collect::<Vec<_>>();
        for block_name in &self.async_blocks {
            let (tx, rx) = mpsc::channel();
            let block_name = block_name.clone();
            let vars = vars.clone();
            let mut block_state = entry_state.fork_with_dynamic(snapshot.vars.clone());
            self.pool.spawn(move || {
                crate::trace!(
                    "async.start block={} thread={:?}",
                    block_name,
                    std::thread::current().id()
                );
                let result = run_block(&mut block_state, &block_name)
                    .and_then(|_| collect_yields(&block_state, &vars))
                    .map(|(yielded, mutated)| ConsumerResult {
                        block_name: block_name.clone(),
                        yielded,
                        mutated,
                    });
                crate::trace!(
                    "async.end block={} thread={:?}",
                    block_name,
                    std::thread::current().id()
                );
                let _ = tx.send(result);
            });
            self.pending.push(rx);
        }
        Ok(())
    }

    pub fn on_await(&mut self, entry_state: &mut RuntimeState, _vars: &[String]) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let pending = std::mem::take(&mut self.pending);
        let mut merged = HashMap::new();
        let mut merged_mutated: HashMap<String, bool> = HashMap::new();
        for rx in pending {
            let result = rx
                .recv()
                .map_err(|_| anyhow!("consumer block exited unexpectedly"))??;
            for (name, value) in result.yielded {
                let is_mutated = result.mutated.contains(&name);
                if let Some(existing_mutated) = merged_mutated.get(&name) {
                    if *existing_mutated && is_mutated {
                        return Err(anyhow!("multiple blocks mutated {}", name));
                    }
                    if *existing_mutated && !is_mutated {
                        continue;
                    }
                    if !*existing_mutated && is_mutated {
                        merged.insert(name.clone(), value.clone());
                        merged_mutated.insert(name.clone(), true);
                        entry_state.insert_dynamic_shared(&name, value)?;
                        continue;
                    }
                    continue;
                }
                entry_state.insert_dynamic_shared(&name, value.clone())?;
                merged.insert(name.clone(), value);
                merged_mutated.insert(name, is_mutated);
            }
        }
        Ok(())
    }
}

fn collect_yields(
    state: &RuntimeState,
    vars: &[String],
) -> Result<(HashMap<String, SharedTensor>, HashSet<String>)> {
    let mut yielded = HashMap::new();
    let mut mutated = HashSet::new();
    for var in vars {
        if let Some(value) = state.dynamic_shared(var) {
            yielded.insert(var.clone(), value);
            if state.was_mutated(var) {
                mutated.insert(var.clone());
            }
        }
    }
    Ok((yielded, mutated))
}
