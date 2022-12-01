mod lib;
use lib::*;

fn main() {

    let states = (1..=10).collect();
    let actions = vec![0,1,2];
    let action_mapping:Vec<Actions> = vec![Actions::NONE, Actions::WALK, Actions::TRAM];
    let tram = TramSpace::new(states, 1, 10);
    let mdp_problem = MDP::new(tram, actions, 1.);
    let optimal_policy:Vec<usize> = mdp_problem.policy_iteration(1e-6);
    let optimal_policy_text:Vec<&Actions> = optimal_policy.into_iter().map(|p| &action_mapping[p]).collect();
    println!("Optimal policy is (By Policy Iteration): {:?}", optimal_policy_text);
}
