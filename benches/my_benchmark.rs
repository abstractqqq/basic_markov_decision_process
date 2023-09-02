
use criterion::{criterion_group, criterion_main, Criterion};
use markov_decision::examples::grid_world::{
    GridWorld,
    Movements
};
use markov_decision::markov_decision_process::MarkovDecisionProcess;

fn criterion_benchmark(c: &mut Criterion) {
    let grid_world = GridWorld::default();
    grid_world.print_world();
    let mut mdp = MarkovDecisionProcess::new(
        grid_world,
        0,
        0.9
    );

    c.bench_function("value iteration", |b| b.iter(|| 
        mdp.value_iteration(0.01)
    ));

    c.bench_function("policy iteration", |b| b.iter(|| 
        mdp.policy_iteration(0.01)
    ));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);