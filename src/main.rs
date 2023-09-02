mod markov_decision_process;
mod examples;

use examples::grid_world::{
    GridWorld,
    Movements
};
use markov_decision_process::{MarkovDecisionProcess, MDPSolver};

fn main() {

    let grid_world = GridWorld::default();
    grid_world.print_world();
    let mut mdp = MarkovDecisionProcess::new(
        grid_world,
        0,
        0.9
    );

    let policy1 = mdp.solve(MDPSolver::POLICY_ITER, 0.01);
    let values = mdp.get_learned_values();
    // let policy2 = mdp.solve(MDPSolver::VALUE_ITER, 0.01);

    let policy1 = policy1.into_iter()
        .map(|x| Movements::from_usize(x)).collect::<Vec<Movements>>();

    // let policy = mdp.policy_iteration(0.01).into_iter()
    //     .map(|x| Movements::from_usize(x)).collect::<Vec<Movements>>();

    println!("{:?}", policy1);
    let grid_world = mdp.get_state_space();
    grid_world.print_on_states(policy1);
    grid_world.print_on_states(values);
}