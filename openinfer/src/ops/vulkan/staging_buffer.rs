use std::mem;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;

#[derive(Default)]
pub struct StagingBuffers {
    pub input: Vec<u8>,
    pub output: Vec<u8>,
}

static STAGING: OnceLock<Mutex<StagingBuffers>> = OnceLock::new();

fn staging() -> &'static Mutex<StagingBuffers> {
    STAGING.get_or_init(|| Mutex::new(StagingBuffers::default()))
}

pub fn take_staging() -> Result<StagingBuffers> {
    let mut guard = staging()
        .lock()
        .map_err(|_| anyhow::anyhow!("staging buffer lock poisoned"))?;
    let input = mem::take(&mut guard.input);
    let output = mem::take(&mut guard.output);
    Ok(StagingBuffers { input, output })
}

pub fn return_staging(buffers: StagingBuffers) -> Result<()> {
    let mut guard = staging()
        .lock()
        .map_err(|_| anyhow::anyhow!("staging buffer lock poisoned"))?;
    guard.input = buffers.input;
    guard.output = buffers.output;
    Ok(())
}
