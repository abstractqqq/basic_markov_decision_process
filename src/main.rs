mod markov_decision_process;
mod examples;

use examples::grid_world::{
    GridWorld,
    Movements
};
use examples::dien::{
    DieN
};
use markov_decision::markov_decision_process::Policy;
use markov_decision_process::{MarkovDecisionProcess, MDPSolver};

fn main() {

    // let grid_world = GridWorld::default();
    // grid_world.print_world();
    // let mut mdp = MarkovDecisionProcess::new(
    //     grid_world,
    //     0,
    //     0.9
    // );

    // let mut max_diff = f64::MAX;
    // let mut iter: u32 = 0;
    // while max_diff > 0.05 {
    //     iter += 1;
    //     max_diff = mdp.update_value();
    //     let values = mdp.get_learned_values();
    //     let grid_world = mdp.get_state_space();

    //     println!("\nIteration: {}", iter);
    //     grid_world.print_on_states(values);
    //     println!("L1 distance {}, Convergence threshold: {}", max_diff, 0.05);
    // }

    // let policy1 = mdp.solve(MDPSolver::VALUE_ITER, 0.05);
    // let policy1 = policy1.into_iter()
    //     .map(|x| Movements::from_usize(x)).collect::<Vec<Movements>>();

    // let grid_world = mdp.get_state_space();
    // grid_world.print_on_states(policy1);



    let dice_world = DieN::new(vec![1,1,1,0,0,0]);
    let mut mdp2 = MarkovDecisionProcess::new(
        dice_world,
        0,
        1.0
    );
    let policy2 = mdp2.solve(MDPSolver::VALUE_ITER, 0.01);
    let values2 = mdp2.get_learned_values();
    println!("DieN best policy: {:?}", policy2);
    println!("DieN expected winning if start from state: {:?}", values2);

}