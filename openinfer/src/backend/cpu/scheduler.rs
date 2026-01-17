use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use anyhow::{anyhow, Result};

use crate::graph::{Graph, NodeKind};
use crate::model_loader::ModelLoader;
use crate::simulator::executor::{BlockExit, Executor, ExecutorSnapshot};
use crate::simulator::Device;
use crate::backend::TensorStorage;

pub struct CpuScheduler {
    inner: Arc<SchedulerInner>,
}

struct SchedulerInner {
    state: Mutex<SchedulerState>,
    cv: Condvar,
    job_txs: Vec<mpsc::Sender<WorkerCommand>>,
    result_rx: Mutex<mpsc::Receiver<WorkerResult>>,
}

struct SchedulerState {
    available_vars: HashSet<String>,
    running_blocks: HashSet<String>,
    var_active_blocks: HashMap<String, HashSet<String>>,
    block_awaits: HashMap<String, Vec<String>>,
    block_writes: HashMap<String, Vec<String>>,
    updates: HashMap<String, TensorStorage>,
    error: Option<anyhow::Error>,
    next_worker: usize,
}

struct WorkerJob {
    block: String,
    snapshot: ExecutorSnapshot,
    writer_vars: Vec<String>,
    model: Arc<ModelLoader>,
    device: Device,
    graph: Graph,
    trace_enabled: bool,
    timer_enabled: bool,
    inplace_enabled: bool,
}

enum WorkerCommand {
    Run(WorkerJob),
}

struct WorkerResult {
    block: String,
    updates: Result<HashMap<String, TensorStorage>>,
}

impl CpuScheduler {
    pub fn maybe_new(
        device: Device,
        graph: &Graph,
        _trace_enabled: bool,
        _timer_enabled: bool,
        _inplace_enabled: bool,
    ) -> Result<Option<Arc<Self>>> {
        if !matches!(device, Device::Cpu | Device::CpuAvx | Device::CpuAvx2) {
            return Ok(None);
        }
        if !graph_has_yield_await(graph) {
            return Ok(None);
        }
        let block_awaits = collect_block_awaits(graph);
        let worker_count = block_awaits
            .values()
            .filter(|vars| !vars.is_empty())
            .count();
        let block_writes = collect_block_writes(graph, &block_awaits);
        let mut job_txs = Vec::with_capacity(worker_count);
        let mut job_rxs = Vec::with_capacity(worker_count);
        for _ in 0..worker_count {
            let (tx, rx) = mpsc::channel();
            job_txs.push(tx);
            job_rxs.push(rx);
        }
        let (result_tx, result_rx) = mpsc::channel();
        let inner = Arc::new(SchedulerInner {
            state: Mutex::new(SchedulerState {
                available_vars: HashSet::new(),
                running_blocks: HashSet::new(),
                var_active_blocks: HashMap::new(),
                block_awaits,
                block_writes,
                updates: HashMap::new(),
                error: None,
                next_worker: 0,
            }),
            cv: Condvar::new(),
            job_txs,
            result_rx: Mutex::new(result_rx),
        });
        let scheduler = Arc::new(Self { inner });
        for rx in job_rxs {
            let tx = result_tx.clone();
            let scheduler_clone = scheduler.clone();
            thread::spawn(move || worker_loop(rx, tx, scheduler_clone));
        }
        Ok(Some(scheduler))
    }

    pub fn reset(&self) {
        let mut state = self.inner.state.lock().expect("scheduler state mutex poisoned");
        state.available_vars.clear();
        state.running_blocks.clear();
        state.var_active_blocks.clear();
        state.updates.clear();
        state.error = None;
        state.next_worker = 0;
    }

    pub fn schedule_yield(
        &self,
        vars: &[String],
        snapshot: ExecutorSnapshot,
        model: Arc<ModelLoader>,
        device: Device,
        graph: Graph,
        trace_enabled: bool,
        timer_enabled: bool,
        inplace_enabled: bool,
    ) -> Result<()> {
        let mut state = self.inner.state.lock().expect("scheduler state mutex poisoned");
        if !state.available_vars.is_empty() {
            return Err(anyhow!("yield already active; await before yielding again"));
        }
        for var in vars {
            state.available_vars.insert(var.clone());
        }
        for (block, awaits) in state.block_awaits.clone() {
            if awaits.is_empty() {
                continue;
            }
            if block == "entry" {
                continue;
            }
            if state.running_blocks.contains(&block) {
                continue;
            }
            if !awaits.iter().all(|var| state.available_vars.contains(var)) {
                continue;
            }
            state.running_blocks.insert(block.clone());
            for var in &awaits {
                state
                    .var_active_blocks
                    .entry(var.clone())
                    .or_default()
                    .insert(block.clone());
            }
            let writer_vars = state
                .block_writes
                .get(&block)
                .cloned()
                .unwrap_or_default();
            let job = WorkerJob {
                block,
                snapshot: snapshot.clone(),
                writer_vars,
                model: model.clone(),
                device,
                graph: graph.clone(),
                trace_enabled,
                timer_enabled,
                inplace_enabled,
            };
            if self.inner.job_txs.is_empty() {
                continue;
            }
            let worker_idx = state.next_worker % self.inner.job_txs.len();
            state.next_worker = state.next_worker.wrapping_add(1);
            self.inner.job_txs[worker_idx]
                .send(WorkerCommand::Run(job))
                .map_err(|_| anyhow!("failed to dispatch worker job"))?;
        }
        self.inner.cv.notify_all();
        Ok(())
    }

    pub fn await_vars(&self, vars: &[String]) -> Result<HashMap<String, TensorStorage>> {
        let mut state = self.inner.state.lock().expect("scheduler state mutex poisoned");
        if vars
            .iter()
            .any(|var| !state.available_vars.contains(var))
        {
            return Err(anyhow!("await called before variable was yielded"));
        }
        loop {
            self.drain_results_locked(&mut state);
            if let Some(err) = state.error.take() {
                return Err(err);
            }
            let done = vars.iter().all(|var| {
                state
                    .var_active_blocks
                    .get(var)
                    .map(|blocks| blocks.is_empty())
                    .unwrap_or(true)
            });
            if done {
                break;
            }
            state = self.inner.cv.wait(state).expect("scheduler state mutex poisoned");
        }
        let mut updates = HashMap::new();
        for var in vars {
            state.available_vars.remove(var);
            if let Some(value) = state.updates.remove(var) {
                updates.insert(var.clone(), value);
            }
        }
        Ok(updates)
    }

    pub fn wait_for_vars(&self, vars: &[String]) -> Result<()> {
        let mut state = self.inner.state.lock().expect("scheduler state mutex poisoned");
        loop {
            self.drain_results_locked(&mut state);
            if let Some(err) = state.error.take() {
                return Err(err);
            }
            let ready = vars
                .iter()
                .all(|var| state.available_vars.contains(var));
            if ready {
                break;
            }
            state = self.inner.cv.wait(state).expect("scheduler state mutex poisoned");
        }
        Ok(())
    }

    fn drain_results_locked(&self, state: &mut SchedulerState) {
        let rx = self.inner.result_rx.lock().expect("result rx mutex poisoned");
        while let Ok(result) = rx.try_recv() {
            match result.updates {
                Ok(values) => {
                    for (name, value) in values {
                        state.updates.insert(name, value);
                    }
                }
                Err(err) => {
                    if state.error.is_none() {
                        state.error = Some(err);
                    }
                }
            }
            if let Some(awaits) = state.block_awaits.get(&result.block) {
                for var in awaits {
                    if let Some(blocks) = state.var_active_blocks.get_mut(var) {
                        blocks.remove(&result.block);
                    }
                }
            }
            state.running_blocks.remove(&result.block);
            self.inner.cv.notify_all();
        }
    }
}

impl Clone for CpuScheduler {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

fn worker_loop(
    rx: mpsc::Receiver<WorkerCommand>,
    result_tx: mpsc::Sender<WorkerResult>,
    scheduler: Arc<CpuScheduler>,
) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            WorkerCommand::Run(job) => {
                let result = run_worker_job(job, scheduler.clone());
                let _ = result_tx.send(result);
                scheduler.inner.cv.notify_all();
            }
        }
    }
}

fn run_worker_job(job: WorkerJob, scheduler: Arc<CpuScheduler>) -> WorkerResult {
    let mut updates = HashMap::new();
    let result = Executor::from_snapshot(
        job.model,
        job.device,
        job.graph,
        job.trace_enabled,
        job.timer_enabled,
        job.inplace_enabled,
        job.snapshot,
        Some(scheduler),
    )
    .and_then(|mut exec| {
        let exit = exec.run_block(&job.block)?;
        match exit {
            BlockExit::Yield(vars) => {
                let _ = vars;
                for var in job.writer_vars {
                    if let Ok(value) = exec.get_tensor_for_scheduler(&var) {
                        updates.insert(var, value);
                    }
                }
                Ok(updates)
            }
            BlockExit::Return => {
                for var in job.writer_vars {
                    if let Ok(value) = exec.get_tensor_for_scheduler(&var) {
                        updates.insert(var, value);
                    }
                }
                Ok(updates)
            }
        }
    });
    WorkerResult {
        block: job.block,
        updates: result,
    }
}

fn graph_has_yield_await(graph: &Graph) -> bool {
    graph.blocks.values().any(|block| block_has_yield_await(&block.nodes))
}

fn block_has_yield_await(nodes: &[crate::graph::Node]) -> bool {
    for node in nodes {
        match &node.kind {
            NodeKind::Yield { .. } | NodeKind::Await { .. } => return true,
            NodeKind::Loop { body, .. } => {
                if block_has_yield_await(body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn collect_block_awaits(graph: &Graph) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for (name, block) in &graph.blocks {
        let vars = block_await_vars(&block.nodes);
        out.insert(name.clone(), vars);
    }
    out
}

fn collect_block_writes(
    graph: &Graph,
    block_awaits: &HashMap<String, Vec<String>>,
) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for (name, block) in &graph.blocks {
        let awaits = block_awaits.get(name).cloned().unwrap_or_default();
        if awaits.is_empty() {
            continue;
        }
        let writes = block_write_vars(&block.nodes);
        let filtered = writes
            .into_iter()
            .filter(|var| awaits.contains(var))
            .collect::<Vec<_>>();
        out.insert(name.clone(), filtered);
    }
    out
}

fn block_await_vars(nodes: &[crate::graph::Node]) -> Vec<String> {
    let mut vars = Vec::new();
    for node in nodes {
        match &node.kind {
            NodeKind::Await { vars: awaited } => {
                for var in awaited {
                    if !vars.contains(var) {
                        vars.push(var.clone());
                    }
                }
            }
            NodeKind::Loop { body, .. } => {
                for var in block_await_vars(body) {
                    if !vars.contains(&var) {
                        vars.push(var);
                    }
                }
            }
            _ => {}
        }
    }
    vars
}

fn block_write_vars(nodes: &[crate::graph::Node]) -> Vec<String> {
    let mut vars = Vec::new();
    for node in nodes {
        match &node.kind {
            NodeKind::Assign { name, .. } => {
                if !vars.contains(name) {
                    vars.push(name.clone());
                }
            }
            NodeKind::Op { output, .. } => {
                if !vars.contains(output) {
                    vars.push(output.clone());
                }
            }
            NodeKind::CacheRead { dst, .. } => {
                if !vars.contains(dst) {
                    vars.push(dst.clone());
                }
            }
            NodeKind::Loop { body, .. } => {
                for var in block_write_vars(body) {
                    if !vars.contains(&var) {
                        vars.push(var);
                    }
                }
            }
            _ => {}
        }
    }
    vars
}
