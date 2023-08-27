mod markov_decision_process;
mod examples;

use examples::grid_world::{
    GridWorld,
    Movements
};
use markov_decision_process::MarkovDecisionProcess;

fn main() {

    let grid_world = GridWorld::default();
    grid_world.print_world();
    let mdp = MarkovDecisionProcess::new(
        &grid_world,
        0,
        0.9
    );

    let policy = mdp.value_iteration(0.01).into_iter()
        .map(|x| Movements::from_usize(x)).collect::<Vec<Movements>>();
    println!("{:?}", policy);
    grid_world.print_policy(policy);
}