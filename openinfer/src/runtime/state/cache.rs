use anyhow::{anyhow, Result};

use crate::graph::CacheAccess;

use super::RuntimeState;

impl RuntimeState {
    pub fn cache_read(&mut self, src: &CacheAccess, dst: &str) -> Result<()> {
        let decl = self
            .shared
            .graph
            .vars
            .get(&src.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", src.base))?;
        let value = self
            .shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .read(
                src,
                &decl,
                &self.shared.graph,
                &self.shared.model,
                &self.loop_vars,
            )?;
        self.locals
            .insert(dst.to_string(), std::sync::Arc::new(std::sync::Mutex::new(value)));
        Ok(())
    }

    pub fn cache_write(&mut self, src: &str, dst: &CacheAccess) -> Result<()> {
        let decl = self
            .shared
            .graph
            .vars
            .get(&dst.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", dst.base))?;
        let value = self.get_tensor(src)?;
        self.shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .write(
                &value,
                dst,
                &decl,
                &self.shared.graph,
                &self.shared.model,
                &self.loop_vars,
            )
    }

    pub fn cache_bump(&mut self, target: &str, amount: i64, decrement: bool) -> Result<()> {
        self.shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .bump(target, amount, decrement, &self.shared.graph, &self.shared.model)
    }

    pub fn cache_reset(&mut self, target: &CacheAccess) -> Result<()> {
        let decl = self
            .shared
            .graph
            .vars
            .get(&target.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target.base))?;
        self.shared
            .cache
            .lock()
            .expect("cache lock poisoned")
            .reset(
                target,
                &decl,
                &self.shared.graph,
                &self.shared.model,
                &self.loop_vars,
            )
    }
}
