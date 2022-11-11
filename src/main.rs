use std::collections::{HashSet, HashMap};
use itertools::Itertools;

mod lib;
use lib::*;

/// TramSpace
/// States: 1, 2, 3, 4, 5, 6, 7, ...
/// Idx:    0, 1, 2, 3, 4, 5, 6, ...
/// 
#[derive(Debug, PartialEq)]
pub enum Actions {
    NONE,
    WALK,
    TRAM
}

// TramSpace is specific to the Tram Problem and nothing else.
pub struct TramSpace{
    states: Vec<usize>,
    start: usize,
    end: usize // one terminal state in this problem. 
}

impl TramSpace {
    fn new(all_states:Vec<usize>, start:usize, end:usize) -> TramSpace {
        
        let mut true_end = end;
        if end <= start {
            println!{"Warning: Found end <= start. Forcing end = start + 10."};
            true_end = start + 10;
        }
        TramSpace {states: all_states, start: start, end: true_end}
    }

}

impl StateSpace for TramSpace {

    fn get_all_states(&self) -> &Vec<usize> {
        &self.states 
    }

    fn get_state_count(&self) -> usize {
        self.states.len()
    }
    
    fn get_state_idx(&self, s:usize) -> usize {
        s - self.start
    }

    fn get_state_from_idx(&self, i:usize) -> usize {
        i + self.start
    }
    
    fn is_terminal_state(&self, s:usize) -> bool {
        s == self.end
    }
    
    fn get_actions_at_state(&self, s:usize) -> Vec<usize> {
        // Action to usize mapping : 0 <-> NONE, 1 <-> WALK, 2 <-> TRAM
        // this function should be specific to the game
        let mut actions:Vec<usize> = Vec::new();
        if s + 1 <= self.end {
            actions.push(1);
        }
        if s * 2 <= self.end {
            actions.push(2);
        }
        actions 
    }

    ///To do: Better way to assign distribution.
    /// 
    ///s: state (index)
    ///a: action (index)
    ///returns (next state, probability, reward)
    fn get_prob_reward(&self, s:usize, a:usize) -> Vec<(usize, f32, f32)> {
        let mut output:Vec<(usize, f32, f32)> = Vec::new();
        match a {
            1 => output.push((s + 1, 1.0, -1.)),
            2 => {
                output.push((s * 2, 0.9, -2.));
                output.push((s, 0.1, -2.));
            },
            _ => {}
        }
        output 
    }


}

//-------------------------------------------------------------------------
#[derive(Debug, PartialEq)]
enum Moves {
    NONE,
    UP,
    LEFT,
    DOWN,
    RIGHT
}

struct MountainSpace{
    grid: Vec<(usize, usize)>, // states will be the indices of the grid flattened.
    x_max: usize,
    y_max: usize,
    states:Vec<usize>, // redundant, just to make StateSpace implementation cleaner.
    reward_map:HashMap<usize, f32> // state : reward
}

// work in progress

impl MountainSpace {

    ///Given x_max, y_max, generate a grid [0..=x_max] x [0..y_max], danger spots, specify starting point and destination
    fn new(x_max:usize, y_max:usize, reward:Vec<(usize, usize, f32)>) 
    -> MountainSpace{
        //[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 0), ...
        let grid:Vec<(usize, usize)> = (0..=x_max).cartesian_product(0..=y_max).collect();
        // the indices here will become the states
        let mut reward_map:HashMap<usize, f32> = HashMap::with_capacity(reward.len());
        for (x,y, reward) in reward {
            reward_map.insert(y*(x_max + 1) + x, reward);
        }
        let n = grid.len();
        MountainSpace {
            grid: grid,
            x_max: x_max,
            y_max: y_max,
            states: (0..n).collect(),
            reward_map: reward_map
        }
    }

    fn index_to_grid(&self, idx:usize) -> (usize, usize) {
        self.grid[idx]
    }


}

// impl StateSpace for MountainSpace {

//     fn get_all_states(&self) -> &Vec<usize> {&self.states}

//     fn get_state_count(&self) -> usize {self.states.len()}
    
//     fn get_state_idx(&self, s:usize) -> usize {s}

//     fn get_state_from_idx(&self, i:usize) -> usize {i}
    
//     // s is state index
//     fn is_terminal_state(&self, s:usize) -> bool {
//         self.reward_map.contains_key(&s)
//     }

//     fn get_actions_at_state(&self, s:usize) -> Vec<usize>{
//         // UP -> 1
//         // LEFT -> 2
//         // DOWN -> 3
//         // RIGHT -> 4
//         let mut actions:HashSet<usize> = HashSet::from_iter(1..=4);
//         let (x,y) = self.index_to_grid(s);
        
//         if let Some(r) = self.reward_map.get(&s) {
//             return vec![0];
//         }
//         if x == 0 {
//             actions.remove(&2);
//         }
//         if x == self.x_max{
//             actions.remove(&4);
//         }
//         if y == 0 {
//             actions.remove(&1);
//         }
//         if y == self.y_max {
//             actions.remove(&3);
//         }

//         actions.into_iter().collect_vec()
//     }

//     // fn get_prob_reward(&self, s:usize, a:usize) -> Vec<(usize, f32, f32)> {
//     //     match a {
//     //         1 => {
//     //         }
//     //     }
//     // }

// }


fn main() {

    // let states = (1..=20).collect();
    // let actions = vec![0,1,2];
    // let action_mapping:Vec<Actions> = vec![Actions::NONE, Actions::WALK, Actions::TRAM];
    // let tram = TramSpace::new(states, 1, 20);
    // let mdp_problem = MDP::new(tram, actions, 1.);
    // let optimal_policy:Vec<usize> = mdp_problem.policy_iteration(1e-6);
    // let optimal_policy_text:Vec<&Actions> = optimal_policy.into_iter().map(|p| &action_mapping[p]).collect();
    // println!("Optimal policy is (By Policy Iteration): {:?}", optimal_policy_text);
    // let optimal_policy2:Vec<usize> = mdp_problem.value_iteration(1e-6);
    // let optimal_policy_text2:Vec<&Actions> = optimal_policy2.into_iter().map(|p| &action_mapping[p]).collect();
    // println!("Optimal policy is (By Value Iteration): {:?}", optimal_policy_text2);

    println!("Hello World.");


}
