use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::graph::Graph;
use crate::runtime::cache::CacheStore;
use crate::runtime::model_loader::ModelLoader;
use crate::runtime::trace::TraceEvent;
use crate::simulator::Device;
use crate::tensor::DType;

#[derive(Debug)]
pub struct RuntimeShared {
    pub(crate) model: Arc<ModelLoader>,
    pub(crate) graph: Graph,
    pub(crate) device: Device,
    pub(crate) base_var_shapes: HashMap<String, Vec<usize>>,
    pub(crate) base_var_dtypes: HashMap<String, DType>,
    pub(crate) trace_events: Mutex<Vec<TraceEvent>>,
    pub(crate) trace_enabled: bool,
    pub(crate) timer_enabled: bool,
    pub(crate) cache: Mutex<CacheStore>,
}

#[derive(Debug, Clone)]
pub struct TraceTiming {
    pub micros: String,
    pub micros_parts: [u64; 3],
}
