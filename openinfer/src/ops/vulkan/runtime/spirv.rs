use anyhow::{anyhow, Result};

use crate::vk_trace;

use super::VulkanRuntime;

impl VulkanRuntime {
    pub(super) fn select_entrypoint<'a>(
        &self,
        entrypoints: &'a [String],
        target: &'a str,
    ) -> &'a str {
        if entrypoints.iter().any(|name| name == target) {
            return target;
        }
        if entrypoints.iter().any(|name| name == "main") {
            return "main";
        }
        entrypoints.first().map(|s| s.as_str()).unwrap_or("main")
    }

    pub(super) fn spirv_entrypoints(&self, words: &[u32]) -> Vec<String> {
        const OP_ENTRY_POINT: u16 = 15;
        if words.len() < 5 {
            return Vec::new();
        }
        let mut offset = 5usize;
        let mut names = Vec::new();
        while offset < words.len() {
            let word = words[offset];
            let word_count = (word >> 16) as usize;
            let opcode = (word & 0xFFFF) as u16;
            if word_count == 0 {
                break;
            }
            if opcode == OP_ENTRY_POINT {
                let operands = &words[(offset + 1)..(offset + word_count)];
                if operands.len() >= 3 {
                    let (name, _) = Self::read_spv_string(&operands[2..]);
                    if !name.is_empty() {
                        names.push(name);
                    }
                }
            }
            offset = offset.saturating_add(word_count);
        }
        names
    }

    pub(super) fn validate_spirv(&self, bytes: &[u8], target: &str) -> Result<()> {
        if bytes.len() < 20 || bytes.len() % 4 != 0 {
            return Err(anyhow!(
                "invalid SPIR-V size for {} ({} bytes)",
                target,
                bytes.len()
            ));
        }
        let words = bytemuck::cast_slice::<u8, u32>(bytes);
        let magic = words[0];
        let version = words[1];
        let bound = words[3];
        let schema = words[4];
        vk_trace!(
            "spv header target={} magic=0x{:08x} version=0x{:08x} bound={} schema={}",
            target,
            magic,
            version,
            bound,
            schema
        );
        if magic != 0x0723_0203 {
            return Err(anyhow!(
                "invalid SPIR-V magic for {} (0x{:08x})",
                target,
                magic
            ));
        }
        Ok(())
    }

    fn read_spv_string(words: &[u32]) -> (String, usize) {
        let mut bytes = Vec::new();
        for (word_index, word) in words.iter().enumerate() {
            let raw = word.to_le_bytes();
            for &b in &raw {
                if b == 0 {
                    let consumed = word_index + 1;
                    let string = String::from_utf8_lossy(&bytes).to_string();
                    return (string, consumed);
                }
                bytes.push(b);
            }
        }
        (String::from_utf8_lossy(&bytes).to_string(), words.len())
    }
}
