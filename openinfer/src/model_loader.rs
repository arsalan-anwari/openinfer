use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};

use crate::tensor::{DType, Tensor, TensorValue};
use crate::types::VarInfo;

#[derive(Debug, Clone)]
pub struct ModelLoader {
    path: PathBuf,
    sizes: HashMap<String, usize>,
    vars: HashMap<String, VarInfo>,
}

impl ModelLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let txt = fs::read_to_string(&path).with_context(|| "read model file")?;
        let mut sizes = HashMap::new();
        let mut vars = HashMap::new();
        let mut offset = 0usize;

        for line in txt.lines() {
            let line_trim = line.trim();
            if line_trim.is_empty() || line_trim.starts_with('#') {
                offset += line.len() + 1;
                continue;
            }

            if let Some(pos) = line_trim.find(":=") {
                let name = line_trim[..pos].trim();
                let val_str = line_trim[pos + 2..].trim();
                let val = val_str
                    .parse::<usize>()
                    .with_context(|| format!("parse size for {}", name))?;
                sizes.insert(name.to_string(), val);
                offset += line.len() + 1;
                continue;
            }

            if let Some(pos) = line_trim.find(':') {
                let name = line_trim[..pos].trim();
                let rest = line_trim[pos + 1..].trim();
                let (dtype, dims) = parse_dtype_dims(rest)?;
                let value_range = if let (Some(start_brace), Some(end_brace)) =
                    (line.find('{'), line.find('}'))
                {
                    let start = offset + start_brace + 1;
                    let end = offset + end_brace;
                    Some((start, end))
                } else {
                    None
                };
                vars.insert(
                    name.to_string(),
                    VarInfo {
                        name: name.to_string(),
                        dtype,
                        dims,
                        value_range,
                    },
                );
                offset += line.len() + 1;
                continue;
            }

            offset += line.len() + 1;
        }

        Ok(Self { path, sizes, vars })
    }

    pub fn size_of(&self, name: &str) -> Result<usize> {
        self.sizes
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("unknown size: {}", name))
    }

    pub fn resolve_len(&self, dims: &[String]) -> Result<usize> {
        let mut total = 1usize;
        for dim in dims {
            if let Ok(val) = dim.parse::<usize>() {
                total = total.saturating_mul(val);
            } else {
                let v = self.size_of(dim)?;
                total = total.saturating_mul(v);
            }
        }
        Ok(total)
    }

    pub fn var_info(&self, name: &str) -> Option<&VarInfo> {
        self.vars.get(name)
    }

    pub fn load_tensor(&self, name: &str) -> Result<TensorValue> {
        let info = self
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
        let range = info
            .value_range
            .ok_or_else(|| anyhow!("no data found for {}", name))?;
        let txt = fs::read_to_string(&self.path).with_context(|| "read model file")?;
        let slice = txt
            .get(range.0..range.1)
            .ok_or_else(|| anyhow!("invalid range for {}", name))?;
        let data = parse_tensor_values(info.dtype, slice)
            .with_context(|| format!("parse tensor values for {}", name))?;
        Ok(data)
    }
}

fn parse_tensor_values(dtype: DType, slice: &str) -> Result<TensorValue> {
    fn parse_numeric<T>(slice: &str, label: &str) -> Result<Vec<T>>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Display,
    {
        let mut out = Vec::new();
        for part in slice.split(',') {
            let val = part.trim();
            if val.is_empty() {
                continue;
            }
            let parsed = val.parse::<T>().map_err(|err| {
                anyhow!("parse {} value {}: {}", label, val, err)
            })?;
            out.push(parsed);
        }
        Ok(out)
    }

    match dtype {
        DType::F32 => Ok(TensorValue::F32(Tensor::new(parse_numeric::<f32>(
            slice, "f32",
        )?))),
        DType::F64 => Ok(TensorValue::F64(Tensor::new(parse_numeric::<f64>(
            slice, "f64",
        )?))),
        DType::I32 => Ok(TensorValue::I32(Tensor::new(parse_numeric::<i32>(
            slice, "i32",
        )?))),
        DType::I64 => Ok(TensorValue::I64(Tensor::new(parse_numeric::<i64>(
            slice, "i64",
        )?))),
        DType::Bool => {
            let mut out = Vec::new();
            for part in slice.split(',') {
                let val = part.trim();
                if val.is_empty() {
                    continue;
                }
                let parsed = match val {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => {
                        return Err(anyhow!("invalid bool value {}", val));
                    }
                };
                out.push(parsed);
            }
            Ok(TensorValue::Bool(Tensor::new(out)))
        }
    }
}

fn parse_dtype_dims(text: &str) -> Result<(DType, Vec<String>)> {
    let mut rest = text;
    let dtype_end = rest
        .find(|c: char| c == '[' || c.is_whitespace() || c == '=')
        .unwrap_or(rest.len());
    let dtype_str = &rest[..dtype_end];
    let dtype = DType::from_ident(dtype_str.trim())?;
    rest = &rest[dtype_end..];

    let mut dims = Vec::new();
    if let Some(start) = rest.find('[') {
        if let Some(end) = rest.find(']') {
            let dim_str = &rest[start + 1..end];
            dims = dim_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }

    Ok((dtype, dims))
}
