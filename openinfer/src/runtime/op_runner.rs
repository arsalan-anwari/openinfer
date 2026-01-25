use anyhow::Result;

use crate::graph::{AttrValue, OpAttrs};
use crate::registry::op_def;
use crate::runtime::tensor_store::TensorRef;

pub fn exec_op(
    op: &str,
    attrs: &OpAttrs,
    inputs: &[&TensorRef],
    output: Option<&TensorRef>,
) -> Result<()> {
    let def = op_def(op);
    let is_inplace = def
        .map(|def| def.supports_inplace)
        .unwrap_or(false)
        && inputs
            .iter()
            .any(|input| input.name == output.map(|out| out.name.clone()).unwrap_or_default());
    let is_accumulate = attrs.items.iter().any(|attr| attr.name == "acc");
    let is_broadcast = def
        .map(|def| def.supports_broadcast)
        .unwrap_or(false)
        && inputs
            .windows(2)
            .any(|pair| pair[0].shape != pair[1].shape);

    let input_desc = inputs
        .iter()
        .map(|tensor| tensor.describe())
        .collect::<Vec<_>>()
        .join(", ");
    let output_desc = output
        .map(|tensor| tensor.describe())
        .unwrap_or_else(|| "<missing>".to_string());
    let attr_desc = format_attrs(attrs);

    println!(
        "op={} inputs=[{}] output={} attrs={} broadcast={} accumulate={} inplace={}",
        op, input_desc, output_desc, attr_desc, is_broadcast, is_accumulate, is_inplace
    );
    Ok(())
}

fn format_attrs(attrs: &OpAttrs) -> String {
    if attrs.items.is_empty() {
        return "[]".to_string();
    }
    let rendered = attrs
        .items
        .iter()
        .map(|attr| format!("{}={}", attr.name, format_attr_value(&attr.value)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{}]", rendered)
}

fn format_attr_value(value: &AttrValue) -> String {
    match value {
        AttrValue::Float(val) => val.to_string(),
        AttrValue::Double(val) => val.to_string(),
        AttrValue::Int(val) => val.to_string(),
        AttrValue::UInt(val) => val.to_string(),
        AttrValue::Bool(val) => val.to_string(),
        AttrValue::Var(name) => name.clone(),
        AttrValue::DType(dtype) => format!("{:?}", dtype),
    }
}
