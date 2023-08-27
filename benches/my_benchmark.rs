use criterion::{criterion_group, criterion_main, Criterion};
use markov_decision::*;

fn criterion_benchmark(c: &mut Criterion) {
    todo!()
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);