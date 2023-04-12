use markov_decision::*;

fn main() {

    let states = (1..=10).collect();
    // let actions = vec![0,1,2];
    // let action_mapping:Vec<Actions> = vec![Actions::NONE, Actions::WALK, Actions::TRAM];
    let tram = TramSpace::new(states, 1, 10);
    let mdp_problem = MDP::new(tram, 0, 1.);
    let optimal_policy:Vec<Policy> = mdp_problem.value_iteration(1e-1);
    let optimal_policy2:Vec<Policy> = mdp_problem.par_value_iteration(1e-1);
    let optimal_policy3:Vec<Policy> = mdp_problem.policy_iteration(1e-1);
    let optimal_policy4:Vec<Policy> = mdp_problem.par_policy_iteration(1e-1);
    // let optimal_policy_text:Vec<&Actions> = optimal_policy.into_iter().map(|p| &action_mapping[p]).collect();
    println!("Optimal policy is (By value Iteration): {:?}", optimal_policy);
    println!("Optimal policy is (By par value Iteration): {:?}", optimal_policy2);
    println!("Optimal policy is (By policy Iteration): {:?}", optimal_policy3);
    println!("Optimal policy is (By par policy Iteration): {:?}", optimal_policy4);
}
