use anyhow::Result;

use super::Executor;

#[derive(Debug, Clone)]
pub(super) struct LoopFrame {
    pub(super) index: String,
    pub(super) end: usize,
    pub(super) current: usize,
    pub(super) body: Vec<crate::graph::Node>,
    pub(super) pos: usize,
    pub(super) prev_value: Option<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct BlockFrame {
    pub(super) name: String,
    pub(super) nodes: Vec<crate::graph::Node>,
    pub(super) pos: usize,
}

#[derive(Debug, Clone)]
pub(super) enum ExecFrame {
    Block(BlockFrame),
    Loop(LoopFrame),
}

impl Executor<'_> {
    pub(super) fn next_node_from_frames(&mut self) -> Result<Option<crate::graph::Node>> {
        loop {
            let Some(frame) = self.frames.last_mut() else {
                return Ok(None);
            };
            match frame {
                ExecFrame::Block(block) => {
                    if block.pos >= block.nodes.len() {
                        self.frames.pop();
                        continue;
                    }
                    let node = block.nodes[block.pos].clone();
                    block.pos += 1;
                    return Ok(Some(node));
                }
                ExecFrame::Loop(loop_frame) => {
                    if loop_frame.pos >= loop_frame.body.len() {
                        loop_frame.current = loop_frame.current.saturating_add(1);
                        if loop_frame.current < loop_frame.end {
                            loop_frame.pos = 0;
                            self.loop_vars
                                .insert(loop_frame.index.clone(), loop_frame.current);
                            continue;
                        }
                        let index = loop_frame.index.clone();
                        let prev_value = loop_frame.prev_value;
                        self.frames.pop();
                        if let Some(prev) = prev_value {
                            self.loop_vars.insert(index, prev);
                        } else {
                            self.loop_vars.remove(&index);
                        }
                        continue;
                    }
                    let node = loop_frame.body[loop_frame.pos].clone();
                    loop_frame.pos += 1;
                    return Ok(Some(node));
                }
            }
        }
    }

    pub(super) fn push_loop_frame(
        &mut self,
        index: String,
        start: String,
        end: String,
        body: Vec<crate::graph::Node>,
    ) -> Result<()> {
        let start_val = self.model.resolve_dim_value(&start)?;
        let end_val = self.model.resolve_dim_value(&end)?;
        if start_val >= end_val {
            return Ok(());
        }
        let prev_value = self.loop_vars.insert(index.clone(), start_val);
        self.frames.push(ExecFrame::Loop(LoopFrame {
            index,
            end: end_val,
            current: start_val,
            body,
            pos: 0,
            prev_value,
        }));
        Ok(())
    }
}
