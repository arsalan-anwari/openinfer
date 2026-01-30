use syn::{Ident, LitBool, LitFloat, LitInt, LitStr};

pub(crate) struct GraphDsl {
    pub(crate) sections: Vec<Section>,
}

pub(crate) enum Section {
    Memory(MemorySection),
    Block(BlockSection),
}

pub(crate) struct MemorySection {
    pub(crate) kind: MemoryKindToken,
    pub(crate) vars: Vec<VarDecl>,
}

pub(crate) enum MemoryKindToken {
    Dynamic,
    Volatile,
    Constant,
    Persistent,
}

pub(crate) struct VarDecl {
    pub(crate) name: Ident,
    pub(crate) dtype: Ident,
    pub(crate) dims: Vec<Dim>,
    pub(crate) init: Option<InitValue>,
    pub(crate) ref_name: Option<LitStr>,
    pub(crate) pattern: Option<LitStr>,
    pub(crate) table_indices: Vec<Ident>,
    pub(crate) table: bool,
    pub(crate) auto_dim: Vec<Ident>,
    pub(crate) fixed: Vec<(Ident, LitInt)>,
}

pub(crate) enum Dim {
    Ident(Ident),
    Lit(LitInt),
    Mul { left: DimAtom, right: DimAtom },
}

pub(crate) enum DimAtom {
    Ident(Ident),
    Lit(LitInt),
}

pub(crate) enum InitValue {
    Float { lit: LitFloat, negative: bool },
    Int { lit: LitInt, negative: bool },
    Bool { lit: LitBool },
}

pub(crate) struct BlockSection {
    pub(crate) name: Ident,
    pub(crate) nodes: Vec<Node>,
}

pub(crate) enum Node {
    Assign(AssignNode),
    Op(OpNode),
    Branch(BranchNode),
    Barrier,
    Dep(DepNode),
    CacheRead(CacheReadNode),
    CacheWrite(CacheWriteNode),
    CacheInc(CacheIncNode),
    CacheDec(CacheDecNode),
    CacheReset(CacheResetNode),
    Transfer(TransferNode),
    Loop(LoopNode),
    Yield(YieldNode),
    Await(AwaitNode),
    Return,
}

pub(crate) struct AssignNode {
    pub(crate) name: Ident,
    pub(crate) dtype: Ident,
    pub(crate) dims: Vec<Dim>,
}

pub(crate) struct OpNode {
    pub(crate) name: Ident,
    pub(crate) inputs: Vec<VarRef>,
    pub(crate) settings: Vec<OpSetting>,
    pub(crate) output: Ident,
}

pub(crate) struct BranchNode {
    pub(crate) cond: Option<Ident>,
    pub(crate) then_block: Ident,
    pub(crate) else_block: Option<Ident>,
}

pub(crate) struct DepNode {
    pub(crate) after: Ident,
    pub(crate) before: Ident,
}

pub(crate) struct LoopNode {
    pub(crate) name: Ident,
    pub(crate) index: Ident,
    pub(crate) start: RangeValue,
    pub(crate) end: RangeValue,
    pub(crate) body: Vec<Node>,
}

pub(crate) struct CacheReadNode {
    pub(crate) src: CacheAccess,
    pub(crate) dst: VarRef,
}

pub(crate) struct CacheWriteNode {
    pub(crate) src: VarRef,
    pub(crate) dst: CacheAccess,
}

pub(crate) struct CacheIncNode {
    pub(crate) target: Ident,
    pub(crate) amount: i64,
}

pub(crate) struct CacheDecNode {
    pub(crate) target: Ident,
    pub(crate) amount: i64,
}

pub(crate) struct CacheResetNode {
    pub(crate) target: CacheAccess,
}

pub(crate) struct TransferNode {
    pub(crate) src: VarRef,
    pub(crate) dst: VarRef,
}

pub(crate) struct YieldNode {
    pub(crate) vars: Vec<Ident>,
}

pub(crate) struct AwaitNode {
    pub(crate) vars: Vec<Ident>,
}

#[derive(Clone)]
pub(crate) struct OpSetting {
    pub(crate) name: Ident,
    pub(crate) value: OpAttrValue,
}

#[derive(Clone)]
#[allow(dead_code)]
pub(crate) enum OpAttrValue {
    Float(f32),
    Double(f64),
    Int(i64),
    Bool(bool),
    String(String),
    IntList(Vec<i64>),
    Var(Ident),
}

pub(crate) enum OpArg {
    Input(VarRef),
    Setting(OpSetting),
}

pub(crate) struct VarRef {
    pub(crate) name: Ident,
    pub(crate) indices: Vec<IndexExpr>,
}

pub(crate) struct CacheAccess {
    pub(crate) name: Ident,
    pub(crate) indices: Vec<CacheIndexExpr>,
    pub(crate) bracketed: bool,
}

pub(crate) enum CacheIndexExpr {
    Single(CacheIndexValue),
    Slice {
        start: Option<CacheIndexValue>,
        end: Option<CacheIndexValue>,
    },
}

pub(crate) enum CacheIndexValue {
    Ident(Ident),
    Lit(i64),
}

pub(crate) enum IndexExpr {
    Ident(Ident),
    Lit(LitInt),
}

pub(crate) enum RangeValue {
    Ident(Ident),
    Lit(LitInt),
}
