use super::{CacheAccess, CacheIndexExpr, CacheIndexValue, NodeKind};

pub fn describe_node(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Assign { name, .. } => format!("assign {}", name),
        NodeKind::Op {
            op,
            attrs: _,
            inputs,
            output,
        } => {
            format!("op {}({}) >> {}", op, inputs.join(","), output)
        }
        NodeKind::CacheRead { src, dst } => {
            let access = format_cache_access(src);
            format!("cache.read {} >> {}", access, dst)
        }
        NodeKind::CacheWrite { src, dst } => {
            let access = format_cache_access(dst);
            format!("cache.write {} >> {}", src, access)
        }
        NodeKind::CacheIncrement { target, amount } => {
            if *amount == 1 {
                format!("cache.increment {}", target)
            } else {
                format!("cache.increment {} {}", amount, target)
            }
        }
        NodeKind::CacheDecrement { target, amount } => {
            if *amount == 1 {
                format!("cache.decrement {}", target)
            } else {
                format!("cache.decrement {} {}", amount, target)
            }
        }
        NodeKind::CacheReset { target } => {
            let access = format_cache_access(target);
            format!("cache.reset {}", access)
        }
        NodeKind::Yield { vars } => format!("yield {}", vars.join(", ")),
        NodeKind::Await { vars } => format!("await {}", vars.join(", ")),
        NodeKind::Loop {
            name,
            index,
            start,
            end,
            ..
        } => format!("loop {} ({} in {}..{})", name, index, start, end),
        NodeKind::Branch {
            cond,
            then_block,
            else_block,
        } => match (cond, else_block) {
            (Some(cond), Some(else_block)) => {
                format!("branch {} {} {}", cond, then_block, else_block)
            }
            _ => format!("branch {}", then_block),
        },
        NodeKind::Barrier => "barrier".to_string(),
        NodeKind::Dep { after, before } => format!("dep after({}) before({})", after, before),
        NodeKind::Return => "return".to_string(),
        NodeKind::Transfer { src, dst } => format!("transfer {} >> {}", src, dst),
    }
}

fn format_cache_access(access: &CacheAccess) -> String {
    if !access.bracketed {
        return access.base.clone();
    }
    if access.indices.is_empty() {
        return format!("{}[]", access.base);
    }
    let rendered = access
        .indices
        .iter()
        .map(|index| match index {
            CacheIndexExpr::Single(value) => format_cache_value(value),
            CacheIndexExpr::Slice { start, end } => {
                let start = start.as_ref().map(format_cache_value);
                let end = end.as_ref().map(format_cache_value);
                match (start, end) {
                    (Some(start), Some(end)) => format!("{}..{}", start, end),
                    (Some(start), None) => format!("{}..", start),
                    (None, Some(end)) => format!("..{}", end),
                    (None, None) => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{}]", access.base, rendered)
}

fn format_cache_value(value: &CacheIndexValue) -> String {
    match value {
        CacheIndexValue::Ident(name) => name.clone(),
        CacheIndexValue::Lit(value) => value.to_string(),
    }
}
