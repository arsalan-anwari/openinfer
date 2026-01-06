use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Default)]
struct TimerState {
    starts: Vec<Option<Instant>>,
    durations: Vec<u128>,
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
    }

    pub fn start(thread_id: u32) {
        let thread_id = thread_id as usize;
        let mut state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        Self::ensure_slot(&mut state, thread_id);
        state.starts[thread_id] = Some(Instant::now());
    }

    pub fn stop(thread_id: u32) {
        let thread_id = thread_id as usize;
        let mut state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        Self::ensure_slot(&mut state, thread_id);
        let elapsed = state.starts[thread_id]
            .take()
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or(0);
        state.durations[thread_id] = elapsed;
    }

    pub fn elapsed(thread_id: u32) -> Option<u128> {
        let thread_id = thread_id as usize;
        let state = Self::state()
            .lock()
            .expect("timer state mutex poisoned");
        state.durations.get(thread_id).copied()
    }
}
