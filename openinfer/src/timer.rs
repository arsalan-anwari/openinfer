use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Default)]
struct TimerState {
    starts: Vec<Option<Instant>>,
    durations: Vec<u128>,
    enabled: Vec<bool>,
}

pub struct Timer;

impl Timer {
    fn state() -> &'static Mutex<TimerState> {
        static INSTANCE: OnceLock<Mutex<TimerState>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(TimerState::default()))
    }

    fn ensure_slot(state: &mut TimerState, thread_id: usize) {
        if state.starts.len() <= thread_id {
            state
                .starts
                .resize_with(thread_id + 1, || None);
        }
        if state.durations.len() <= thread_id {
            state.durations.resize(thread_id + 1, 0);
        }
        if state.enabled.len() <= thread_id {
            state.enabled.resize(thread_id + 1, false);
        }
    }

    pub fn set_enabled(thread_id: usize, enabled: bool) {
        let mut state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        Self::ensure_slot(&mut state, thread_id);
        state.enabled[thread_id] = enabled;
        if !enabled {
            state.starts[thread_id] = None;
            state.durations[thread_id] = 0;
        }
    }

    pub fn start(thread_id: usize) {
        let mut state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        Self::ensure_slot(&mut state, thread_id);
        if !state.enabled[thread_id] {
            return;
        }
        state.starts[thread_id] = Some(Instant::now());
    }

    pub fn stop(thread_id: usize) {
        let mut state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        Self::ensure_slot(&mut state, thread_id);
        if !state.enabled[thread_id] {
            return;
        }
        let elapsed = state.starts[thread_id]
            .take()
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or(0);
        state.durations[thread_id] = elapsed;
    }

    pub fn record(thread_id: usize, duration_ns: u128) {
        let mut state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        Self::ensure_slot(&mut state, thread_id);
        if !state.enabled[thread_id] {
            return;
        }
        state.starts[thread_id] = None;
        state.durations[thread_id] = duration_ns;
    }

    pub fn elapsed(thread_id: usize) -> Option<u128> {
        let state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        if thread_id >= state.enabled.len() || !state.enabled[thread_id] {
            return Some(0);
        }
        Some(state.durations.get(thread_id).copied().unwrap_or(0))
    }
}
