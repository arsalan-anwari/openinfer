use crate::parsers::op::parse_op_attr_value;
use crate::types::{
    CacheIndexExpr, Dim, DimAtom, InitValue, MemoryKindToken, Node, OpAttrValue, RangeValue,
    Section,
};
use syn::parse::{Parser};
use syn::parse_str;

fn parse_graph(src: &str) -> GraphDsl {
    parse_str::<GraphDsl>(src).expect("parse graph")
}

#[test]
fn parses_full_graph_sections_and_nodes() {
    let graph = parse_graph(
        r#"
        dynamic {
            x: f32[B, D];
            y: i32[4, B*D];
        }

        volatile {
            tmp: f32[B, D] @init(0.0);
            flag: bool @init(true);
        }

        constant {
            w: f32[D, D] @ref("w.0") @pattern("gauss");
            bias: f32[D] @pattern("zero");
        }

        persistent {
            state(i, j): f32[D] @table @auto_dim(i, j) @fixed(i=2, j=3);
        }

        block entry {
            assign t0: f32[B, D];
            op add(x, w, alpha=0.5, clamp_max=6.0, axes=[1, 2], tag="relu", flag=true) >> t0;
            cache.write t0 >> state[0, 1];
            cache.read state[0, 1] >> t0;
            cache.increment 2 state;
            cache.decrement state;
            cache.reset state[0, 1];
            branch cond_block then_block else_block;
            barrier;
            dep after(block_a) before(block_b);
            loop i (idx in 0..4) {
                yield t0;
                await t0;
                transfer t0 >> tmp;
                return;
            }
            return;
        }
        "#,
    );

    assert_eq!(graph.sections.len(), 5);
    match &graph.sections[0] {
        Section::Memory(section) => {
            assert!(matches!(section.kind, MemoryKindToken::Dynamic));
            assert_eq!(section.vars.len(), 2);
        }
        _ => panic!("expected memory section"),
    }

    let block = match &graph.sections[4] {
        Section::Block(block) => block,
        _ => panic!("expected block section"),
    };
    assert_eq!(block.nodes.len(), 12);
    assert!(matches!(block.nodes[0], Node::Assign(_)));
    assert!(matches!(block.nodes[1], Node::Op(_)));
    assert!(matches!(block.nodes[2], Node::CacheWrite(_)));
    assert!(matches!(block.nodes[3], Node::CacheRead(_)));
    assert!(matches!(block.nodes[4], Node::CacheInc(_)));
    assert!(matches!(block.nodes[5], Node::CacheDec(_)));
    assert!(matches!(block.nodes[6], Node::CacheReset(_)));
    assert!(matches!(block.nodes[7], Node::Branch(_)));
    assert!(matches!(block.nodes[8], Node::Barrier));
    assert!(matches!(block.nodes[9], Node::Dep(_)));
    assert!(matches!(block.nodes[10], Node::Loop(_)));
    assert!(matches!(block.nodes[11], Node::Return));
}

#[test]
fn parses_dims_with_mul_atoms() {
    let graph = parse_graph(
        r#"
        dynamic {
            x: f32[B*D, 4, K*2];
        }

        block entry { return; }
        "#,
    );
    let dims = match &graph.sections[0] {
        Section::Memory(section) => &section.vars[0].dims,
        _ => panic!("expected memory section"),
    };
    assert!(matches!(dims[0], Dim::Mul { .. }));
    assert!(matches!(dims[1], Dim::Lit(_)));
    assert!(matches!(dims[2], Dim::Mul { .. }));
    if let Dim::Mul { left, right } = &dims[0] {
        assert!(matches!(left, DimAtom::Ident(_)));
        assert!(matches!(right, DimAtom::Ident(_)));
    }
}

#[test]
fn parses_init_values() {
    let graph = parse_graph(
        r#"
        volatile {
            a: f32 @init(1.25);
            b: i32 @init(-3);
            c: bool @init(false);
        }
        block entry { return; }
        "#,
    );
    let vars = match &graph.sections[0] {
        Section::Memory(section) => &section.vars,
        _ => panic!("expected memory section"),
    };
    assert!(matches!(vars[0].init, Some(InitValue::Float { .. })));
    assert!(matches!(vars[1].init, Some(InitValue::Int { negative: true, .. })));
    assert!(matches!(vars[2].init, Some(InitValue::Bool { .. })));
}

#[test]
fn parses_op_attr_values() {
    let val = parse_op_attr_value.parse_str("3.5").unwrap();
    assert!(matches!(val, OpAttrValue::Double(_)));
    let val = parse_op_attr_value.parse_str("-7").unwrap();
    assert!(matches!(val, OpAttrValue::Int(-7)));
    let val = parse_op_attr_value.parse_str("true").unwrap();
    assert!(matches!(val, OpAttrValue::Bool(true)));
    let val = parse_op_attr_value.parse_str("\"tag\"").unwrap();
    assert!(matches!(val, OpAttrValue::String(_)));
    let val = parse_op_attr_value.parse_str("[1, -2, 3]").unwrap();
    assert!(matches!(val, OpAttrValue::IntList(_)));
}

#[test]
fn parses_cache_slices() {
    let graph = parse_graph(
        r#"
        persistent {
            table(i, j): f32 @table;
            out: f32;
        }

        block entry {
            cache.read table[1, .., 2.., ..3, 4..5] >> out;
            return;
        }
        "#,
    );
    let block = match &graph.sections[1] {
        Section::Block(block) => block,
        _ => panic!("expected block"),
    };
    match &block.nodes[0] {
        Node::CacheRead(node) => {
            assert!(node.src.bracketed);
            assert_eq!(node.src.indices.len(), 5);
            assert!(matches!(node.src.indices[1], CacheIndexExpr::Slice { .. }));
        }
        _ => panic!("expected cache read"),
    }
}

#[test]
fn parse_errors_for_invalid_syntax() {
    let err = parse_str::<GraphDsl>("foo { }")
        .err()
        .expect("expected parse error");
    assert!(err.to_string().contains("expected memory section or block"));

    let err = parse_str::<GraphDsl>(
        r#"
        volatile { x: f32 @init(true) @init(false); }
        block entry { return; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err.to_string().contains("duplicate @init attribute"));

    let err = parse_str::<GraphDsl>(
        r#"
        dynamic { x: f32[B*]; }
        block entry { return; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err
        .to_string()
        .contains("expected identifier or integer for dimension expression"));

    let err = parse_str::<GraphDsl>(
        r#"
        dynamic { x: f32[B]; }
        block entry { op add(x, alpha=1, y) >> x; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err.to_string().contains("positional args must come before settings"));

    let err = parse_str::<GraphDsl>(
        r#"
        persistent { table(i): f32 @table; }
        block entry { cache.read table[-i] >> x; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err.to_string().contains("unexpected '-' before identifier"));

    let err = parse_str::<GraphDsl>(
        r#"
        dynamic { x: f32; }
        block entry { branch cond ok; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err
        .to_string()
        .contains("branch expects condition and two target blocks"));

    let err = parse_str::<GraphDsl>(
        r#"
        dynamic { x: f32; }
        block entry { loop i (idx in ..3) { return; } }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err
        .to_string()
        .contains("expected identifier or integer for loop range"));

    let err = parse_str::<GraphDsl>(
        r#"
        volatile { x: f32 @auto_dim(); }
        block entry { return; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err
        .to_string()
        .contains("@auto_dim requires at least one index"));

    let err = parse_str::<GraphDsl>(
        r#"
        dynamic { x: f32; }
        block entry { cache.foo x; }
        "#,
    )
    .err()
    .expect("expected parse error");
    assert!(err.to_string().contains("unsupported cache operation"));
}

#[test]
fn parse_range_values() {
    let graph = parse_graph(
        r#"
        dynamic { x: f32; }
        block entry {
            loop i (idx in 1..N) { return; }
        }
        "#,
    );
    let block = match &graph.sections[1] {
        Section::Block(block) => block,
        _ => panic!("expected block"),
    };
    match &block.nodes[0] {
        Node::Loop(node) => {
            assert!(matches!(node.start, RangeValue::Lit(_)));
            assert!(matches!(node.end, RangeValue::Ident(_)));
        }
        _ => panic!("expected loop"),
    }
}
