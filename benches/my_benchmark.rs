use criterion::{criterion_group, criterion_main, Criterion};
use markov_decision::*;

//------------------------------------------------------ Benchmarks --------------------------------------------------
fn run_policy_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<usize>{
    mdp_problem.policy_iteration(1e-6)
}

fn run_value_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<usize>{
    mdp_problem.value_iteration(1e-6)
}

fn criterion_benchmark(c: &mut Criterion) {
    let states = (1..=100).collect();
    let actions = vec![0,1,2];
    let tram = TramSpace::new(states, 1, 100);
    let mdp_problem = MDP::new(tram, actions, 1.);
    c.bench_function("run_policy_iteration", |b| b.iter(|| run_policy_iteration(&mdp_problem)));
    c.bench_function("run_value_iteration", |b| b.iter(|| run_value_iteration(&mdp_problem)));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);