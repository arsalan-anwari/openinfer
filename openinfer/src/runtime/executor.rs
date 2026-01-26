use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::graph::{Graph, Node, NodeKind};
use crate::runtime::engine::{handle_node, NodeEffect};
use crate::runtime::yield_await::handle_await;
use crate::runtime::AsyncScheduler;
use crate::runtime::model_loader::ModelLoader;
use crate::runtime::state::RuntimeState;
use crate::runtime::trace::{TraceEvent, TraceEventKind};
use crate::runtime::value_eval::{tensor_to_bool, tensor_to_i64};
use crate::tensor::{Tensor, TensorElement, TensorValue};

pub trait Fetchable: Sized {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self>;
}

impl Fetchable for TensorValue {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
        state.get_tensor(name)
    }
}

impl<T: TensorElement> Fetchable for Tensor<T> {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
        state.fetch_typed(name)
    }
}

macro_rules! impl_fetch_int {
    ($t:ty) => {
        impl Fetchable for $t {
            fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
                let tensor: TensorValue = state.get_tensor(name)?;
                Ok(tensor_to_i64(&tensor)? as $t)
            }
        }
    };
}

macro_rules! impl_fetch_float {
    ($t:ty) => {
        impl Fetchable for $t {
            fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
                let tensor: TensorValue = state.get_tensor(name)?;
                Ok(tensor_to_f64(&tensor)? as $t)
            }
        }
    };
}

impl_fetch_float!(f32);
impl_fetch_float!(f64);
impl_fetch_int!(i8);
impl_fetch_int!(i16);
impl_fetch_int!(i32);
impl_fetch_int!(i64);
impl_fetch_int!(u8);
impl_fetch_int!(u16);
impl_fetch_int!(u32);
impl_fetch_int!(u64);

impl Fetchable for bool {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
        let tensor: TensorValue = state.get_tensor(name)?;
        tensor_to_bool(&tensor)
    }
}

pub struct Executor {
    state: RuntimeState,
}

impl Executor {
    pub fn new(
        model: Arc<ModelLoader>,
        graph: Graph,
        trace_enabled: bool,
        timer_enabled: bool,
    ) -> Result<Self> {
        Ok(Self {
            state: RuntimeState::new(model, graph, trace_enabled, timer_enabled)?,
        })
    }

    pub fn insert_dynamic<T: Into<TensorValue>>(&mut self, name: &str, data: T) -> Result<()> {
        self.state.insert_dynamic(name, data.into())
    }

    pub fn fetch<T: Fetchable>(&mut self, name: &str) -> Result<T> {
        T::fetch(&mut self.state, name)
    }

    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<Tensor<T>> {
        self.state.fetch_typed(name)
    }

    pub fn fetch_raw(&mut self, name: &str) -> Result<TensorValue> {
        self.state.get_tensor(name)
    }

    pub fn step(&mut self) -> Result<Option<TraceEvent>> {
        let trace_enabled = self.state.trace_enabled();
        let mut iter = self.iterate();
        let mut last_event = None;
        while let Some(step) = iter.next() {
            let step = step?;
            if trace_enabled {
                log_trace_event(&step.event);
            }
            last_event = Some(step.event);
        }
        Ok(last_event)
    }

    pub fn trace(&self) -> Vec<TraceEvent> {
        self.state.trace()
    }

    pub fn iterate(&mut self) -> ExecutorIter<'_> {
        let graph = self.state.graph();
        let (entry_block, nodes) = if let Some(block) = graph.blocks.get("entry") {
            (block.name.clone(), block.nodes.clone())
        } else if let Some((_, block)) = graph.blocks.iter().next() {
            (block.name.clone(), block.nodes.clone())
        } else {
            panic!("graph has no blocks");
        };
        let mut frames = Vec::new();
        frames.push(Frame::Block(BlockFrame {
            block_name: entry_block.clone(),
            nodes: VecDeque::from(nodes),
        }));
        let async_blocks = build_async_blocks(graph, &entry_block);
        let scheduler =
            AsyncScheduler::new(async_blocks).expect("failed to initialize async scheduler");
        ExecutorIter {
            exec: self as *mut Executor,
            frames,
            entry_block,
            scheduler,
            marker: PhantomData,
        }
    }
}

fn log_trace_event(event: &TraceEvent) {
    crate::log!(
        "{} {} [{}] -- {} -- ({})",
        event.node_index,
        event.node_uuid,
        event.block_name,
        event.node_desc,
        event.micros
    );
}

pub struct ExecutorIter<'a> {
    exec: *mut Executor,
    frames: Vec<Frame>,
    entry_block: String,
    scheduler: AsyncScheduler,
    marker: PhantomData<&'a mut Executor>,
}

impl<'a> Iterator for ExecutorIter<'a> {
    type Item = Result<TraceStep<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            let exec = &mut *self.exec;
            loop {
                if self.frames.is_empty() {
                    return None;
                }
                let frame = self.frames.last_mut().expect("frame stack is not empty");
                let (block_name, node) = match frame {
                    Frame::Block(block) => match block.nodes.pop_front() {
                        Some(node) => (block.block_name.clone(), node),
                        None => {
                            self.frames.pop();
                            continue;
                        }
                    },
                    Frame::Loop(loop_frame) => {
                        if loop_frame.counter >= loop_frame.end {
                            let index = loop_frame.index.clone();
                            self.frames.pop();
                            exec.state.clear_loop_var(&index);
                            continue;
                        }
                        if let Some(node) = loop_frame.nodes.pop_front() {
                            (loop_frame.block_name.clone(), node)
                        } else {
                            loop_frame.counter += 1;
                            loop_frame.nodes = VecDeque::from(loop_frame.nodes_template.clone());
                            exec.state.set_loop_var(&loop_frame.index, loop_frame.counter);
                            continue;
                        }
                    }
                };

                let result = handle_node(&mut exec.state, &block_name, &node);
                let (effect, event) = match result {
                    Ok(value) => value,
                    Err(err) => return Some(Err(err)),
                };
                match effect {
                    NodeEffect::Continue => {
                        return Some(Ok(TraceStep {
                            event,
                            exec: self.exec,
                            marker: PhantomData,
                        }));
                    }
                    NodeEffect::Return => {
                        self.frames.pop();
                        return Some(Ok(TraceStep {
                            event,
                            exec: self.exec,
                            marker: PhantomData,
                        }));
                    }
                    NodeEffect::PushBlock(block_name) => {
                        let nodes = match exec.state.graph().blocks.get(&block_name) {
                            Some(block) => block.nodes.clone(),
                            None => {
                                return Some(Err(anyhow!(
                                    "unknown block {} for branch target",
                                    block_name
                                )))
                            }
                        };
                        self.frames.push(Frame::Block(BlockFrame {
                            block_name,
                            nodes: VecDeque::from(nodes),
                        }));
                        return Some(Ok(TraceStep {
                            event,
                            exec: self.exec,
                            marker: PhantomData,
                        }));
                    }
                    NodeEffect::PushLoop(frame) => {
                        exec.state.set_loop_var(&frame.index, frame.current);
                        let nodes_template = frame.body;
                        let nodes = VecDeque::from(nodes_template.clone());
                        self.frames.push(Frame::Loop(LoopFrameState {
                            index: frame.index,
                            counter: frame.current,
                            end: frame.end,
                            nodes_template,
                            block_name: block_name.clone(),
                            nodes,
                        }));
                        return Some(Ok(TraceStep {
                            event,
                            exec: self.exec,
                            marker: PhantomData,
                        }));
                    }
                    NodeEffect::Yield(snapshot) => {
                        if block_name == self.entry_block {
                            if let Err(err) = self.scheduler.on_yield(&exec.state, &snapshot) {
                                return Some(Err(err));
                            }
                        } else {
                            self.frames.pop();
                        }
                        return Some(Ok(TraceStep {
                            event,
                            exec: self.exec,
                            marker: PhantomData,
                        }));
                    }
                    NodeEffect::Await(vars) => {
                        if block_name == self.entry_block {
                            if let Err(err) = self.scheduler.on_await(&mut exec.state, &vars) {
                                return Some(Err(err));
                            }
                            if let Err(err) = handle_await(&mut exec.state, &vars) {
                                return Some(Err(err));
                            }
                        }
                        return Some(Ok(TraceStep {
                            event,
                            exec: self.exec,
                            marker: PhantomData,
                        }));
                    }
                }
            }
        }
    }
}

enum Frame {
    Block(BlockFrame),
    Loop(LoopFrameState),
}

struct BlockFrame {
    block_name: String,
    nodes: VecDeque<Node>,
}

struct LoopFrameState {
    block_name: String,
    index: String,
    counter: i64,
    end: i64,
    nodes_template: Vec<Node>,
    nodes: VecDeque<Node>,
}

pub struct TraceStep<'a> {
    pub event: TraceEvent,
    exec: *mut Executor,
    marker: PhantomData<&'a mut Executor>,
}

impl<'a> std::ops::Deref for TraceStep<'a> {
    type Target = Executor;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.exec }
    }
}

impl<'a> std::ops::DerefMut for TraceStep<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.exec }
    }
}

fn tensor_to_f64(value: &TensorValue) -> Result<f64> {
    if value.len() != 1 {
        return Err(anyhow!("expected scalar value"));
    }
    match value {
        TensorValue::I8(t) => Ok(t.data[0] as f64),
        TensorValue::I16(t) => Ok(t.data[0] as f64),
        TensorValue::I32(t) => Ok(t.data[0] as f64),
        TensorValue::I64(t) => Ok(t.data[0] as f64),
        TensorValue::U8(t) => Ok(t.data[0] as f64),
        TensorValue::U16(t) => Ok(t.data[0] as f64),
        TensorValue::U32(t) => Ok(t.data[0] as f64),
        TensorValue::U64(t) => Ok(t.data[0] as f64),
        TensorValue::Bool(t) => Ok(if t.data[0] { 1.0 } else { 0.0 }),
        TensorValue::F16(t) => Ok(t.data[0].to_f32() as f64),
        TensorValue::BF16(t) => Ok(t.data[0].to_f32() as f64),
        TensorValue::F8E5M2(t) => Ok(t.data[0].to_f32() as f64),
        TensorValue::F32(t) => Ok(t.data[0] as f64),
        TensorValue::F64(t) => Ok(t.data[0]),
        TensorValue::Bitset(t) => Ok(t.data[0].bits as f64),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("packed scalars are not supported")),
    }
}

fn build_async_blocks(graph: &Graph, entry_block: &str) -> Vec<String> {
    let mut branch_targets = HashSet::new();
    for block in graph.blocks.values() {
        for node in &block.nodes {
            if let NodeKind::Branch {
                then_block,
                else_block,
                ..
            } = &node.kind
            {
                branch_targets.insert(then_block.clone());
                if let Some(else_block) = else_block {
                    branch_targets.insert(else_block.clone());
                }
            }
        }
    }

    let mut names = graph.blocks.keys().cloned().collect::<Vec<_>>();
    names.sort();
    let mut pending = Vec::new();
    for name in names {
        if name == entry_block || branch_targets.contains(&name) {
            continue;
        }
        if let Some(block) = graph.blocks.get(&name) {
            pending.push(block.name.clone());
        }
    }
    pending
}

pub(crate) fn run_block(state: &mut RuntimeState, block_name: &str) -> Result<()> {
    let nodes = match state.graph().blocks.get(block_name) {
        Some(block) => block.nodes.clone(),
        None => return Err(anyhow!("unknown block {}", block_name)),
    };
    let mut frames = Vec::new();
    frames.push(Frame::Block(BlockFrame {
        block_name: block_name.to_string(),
        nodes: VecDeque::from(nodes),
    }));
    loop {
        if frames.is_empty() {
            return Ok(());
        }
        let frame = frames.last_mut().expect("frame stack is not empty");
        let (active_block, node) = match frame {
            Frame::Block(block) => match block.nodes.pop_front() {
                Some(node) => (block.block_name.clone(), node),
                None => {
                    frames.pop();
                    continue;
                }
            },
            Frame::Loop(loop_frame) => {
                if loop_frame.counter >= loop_frame.end {
                    let index = loop_frame.index.clone();
                    frames.pop();
                    state.clear_loop_var(&index);
                    continue;
                }
                if let Some(node) = loop_frame.nodes.pop_front() {
                    (loop_frame.block_name.clone(), node)
                } else {
                    loop_frame.counter += 1;
                    loop_frame.nodes = VecDeque::from(loop_frame.nodes_template.clone());
                    state.set_loop_var(&loop_frame.index, loop_frame.counter);
                    continue;
                }
            }
        };
        let (effect, _event) = handle_node(state, &active_block, &node)?;
        match effect {
            NodeEffect::Continue => {}
            NodeEffect::Return => {
                frames.pop();
            }
            NodeEffect::PushBlock(next_block) => {
                let nodes = match state.graph().blocks.get(&next_block) {
                    Some(block) => block.nodes.clone(),
                    None => return Err(anyhow!("unknown block {} for branch target", next_block)),
                };
                frames.push(Frame::Block(BlockFrame {
                    block_name: next_block,
                    nodes: VecDeque::from(nodes),
                }));
            }
            NodeEffect::PushLoop(frame) => {
                state.set_loop_var(&frame.index, frame.current);
                let nodes_template = frame.body;
                let nodes = VecDeque::from(nodes_template.clone());
                frames.push(Frame::Loop(LoopFrameState {
                    index: frame.index,
                    counter: frame.current,
                    end: frame.end,
                    nodes_template,
                    block_name: active_block.clone(),
                    nodes,
                }));
            }
            NodeEffect::Yield(_) => {
                frames.pop();
            }
            NodeEffect::Await(_) => {}
        }
    }
}
