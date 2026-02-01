use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};

use crate::graph::MemoryKind;
use crate::tensor::{DType, Tensor, TensorElement, TensorValue};

use super::{RuntimeState, SharedTensor};

impl RuntimeState {
    pub fn dynamic_shared(&self, name: &str) -> Option<SharedTensor> {
        self.dynamic.get(name).cloned()
    }

    pub fn insert_dynamic(&mut self, name: &str, value: TensorValue) -> Result<()> {
        if !self.shared.graph.vars.contains_key(name) {
            if let Some((base, _)) = name.split_once('[') {
                if !self.shared.graph.vars.contains_key(base) {
                    return Err(anyhow!("unknown variable: {}", name));
                }
            } else {
                return Err(anyhow!("unknown variable: {}", name));
            }
        }
        self.dynamic
            .insert(name.to_string(), Arc::new(Mutex::new(value)));
        Ok(())
    }

    pub fn insert_dynamic_shared(&mut self, name: &str, value: SharedTensor) -> Result<()> {
        if !self.shared.graph.vars.contains_key(name) {
            if let Some((base, _)) = name.split_once('[') {
                if !self.shared.graph.vars.contains_key(base) {
                    return Err(anyhow!("unknown variable: {}", name));
                }
            } else {
                return Err(anyhow!("unknown variable: {}", name));
            }
        }
        self.dynamic.insert(name.to_string(), value);
        Ok(())
    }

    pub fn set_local(&mut self, name: &str, value: TensorValue) {
        self.locals
            .insert(name.to_string(), Arc::new(Mutex::new(value)));
    }

    pub fn set_local_shared(&mut self, name: &str, value: SharedTensor) {
        self.locals.insert(name.to_string(), value);
    }

    pub fn set_loop_var(&mut self, name: &str, value: i64) {
        self.loop_vars.insert(name.to_string(), value);
    }

    pub fn clear_loop_var(&mut self, name: &str) {
        self.loop_vars.remove(name);
    }

    pub fn fetch_typed<T: TensorElement>(&mut self, name: &str) -> Result<Tensor<T>> {
        let value = self.get_tensor(name)?;
        T::from_value(&value).ok_or_else(|| anyhow!("dtype mismatch for fetched tensor {}", name))
    }

    pub fn get_tensor(&mut self, name: &str) -> Result<TensorValue> {
        if let Some(value) = self.dynamic.get(name) {
            return value
                .lock()
                .map(|guard| guard.clone())
                .map_err(|_| anyhow!("dynamic tensor lock poisoned"));
        }
        if let Some(value) = self.locals.get(name) {
            return value
                .lock()
                .map(|guard| guard.clone())
                .map_err(|_| anyhow!("local tensor lock poisoned"));
        }
        if let Some(value) = self
            .shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .get_persistent(name)
        {
            return Ok(value);
        }
        let decl = self
            .shared
            .graph
            .vars
            .get(name)
            .cloned()
            .or_else(|| {
                name.split_once('[')
                    .and_then(|(base, _)| self.shared.graph.vars.get(base).cloned())
            })
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;

        if let Some(info) = self.shared.model.var_info(name) {
            if info.has_data {
                let value = self.shared.model.load_tensor(name)?;
                if decl.kind == MemoryKind::Persistent {
                    self.shared
                        .cache
                        .lock()
                        .expect("cache lock poisoned")
                        .set_persistent(name, value.clone());
                } else {
                    self.locals
                        .insert(name.to_string(), Arc::new(Mutex::new(value.clone())));
                }
                return Ok(value);
            }
        }

        if let Some(value) = self.shared.model.load_metadata_tensor(name)? {
            if decl.kind == MemoryKind::Persistent {
                self.shared
                    .cache
                    .lock()
                    .expect("cache lock poisoned")
                    .set_persistent(name, value.clone());
            } else {
                self.locals
                    .insert(name.to_string(), Arc::new(Mutex::new(value.clone())));
            }
            return Ok(value);
        }

        if decl.kind == MemoryKind::Persistent {
            return self
                .shared
                .cache
                .lock()
                .expect("cache lock poisoned")
                .get_or_init_persistent(&decl.name, &decl, &self.shared.model);
        }

        let shape = self.shared.model.resolve_shape(&decl.dims)?;
        let value = if let Some(init) = &decl.init {
            init.to_tensor_value(decl.dtype, &shape)?
        } else {
            TensorValue::zeros(decl.dtype, &shape)
        };
        self.locals
            .insert(decl.name.clone(), Arc::new(Mutex::new(value.clone())));
        Ok(value)
    }

    pub fn get_tensor_shared(&mut self, name: &str) -> Result<SharedTensor> {
        if let Some(value) = self.dynamic.get(name) {
            return Ok(Arc::clone(value));
        }
        if let Some(value) = self.locals.get(name) {
            return Ok(Arc::clone(value));
        }
        let value = self.get_tensor(name)?;
        Ok(Arc::new(Mutex::new(value)))
    }

    pub fn transfer_var(&mut self, src: &str, dst: &str) -> Result<()> {
        let value = self.get_tensor_shared(src)?;
        self.set_local_shared(dst, value);
        self.mark_mutated(dst);
        Ok(())
    }

    pub fn register_assign(&mut self, name: &str, dtype: DType, dims: &[String]) -> Result<()> {
        let shape = self.shared.model.resolve_shape(dims)?;
        self.var_shapes.insert(name.to_string(), shape.clone());
        self.var_dtypes.insert(name.to_string(), dtype);
        self.temps.insert(name.to_string());
        if !self.locals.contains_key(name) {
            let value = TensorValue::zeros(dtype, &shape);
            self.locals
                .insert(name.to_string(), Arc::new(Mutex::new(value)));
        }
        Ok(())
    }

    pub fn mark_mutated(&mut self, name: &str) {
        self.mutated.insert(name.to_string());
    }

    pub fn was_mutated(&self, name: &str) -> bool {
        self.mutated.contains(name)
    }
}
