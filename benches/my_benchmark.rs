use criterion::{criterion_group, criterion_main, Criterion};
use markov_decision::*;

// Can probably relax the epsilon by a bit when running policy iteration.
fn run_policy_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<Policy>{
    mdp_problem.policy_iteration(1e-1)
}

fn run_par_policy_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<Policy>{
    mdp_problem.par_policy_iteration(1e-1)
}

fn run_value_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<Policy>{
    mdp_problem.value_iteration(1e-1)
}

fn run_par_value_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<Policy>{
    mdp_problem.par_value_iteration(1e-1)
}



fn criterion_benchmark(c: &mut Criterion) {
    let states = (1..=1000).collect();
    // let actions = vec![0,1,2];
    let tram = TramSpace::new(states, 1, 1000);
    let mdp_problem = MDP::new(tram, 0, 1.);
    c.bench_function("run_policy_iteration", |b| b.iter(|| run_policy_iteration(&mdp_problem)));
    c.bench_function("run_par_policy_iteration", |b| b.iter(|| run_par_policy_iteration(&mdp_problem)));
    c.bench_function("run_value_iteration", |b| b.iter(|| run_value_iteration(&mdp_problem)));
    c.bench_function("run_par_value_iteration", |b| b.iter(|| run_par_value_iteration(&mdp_problem)));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);