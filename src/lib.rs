use rayon::prelude::*;
use std::marker::Sync;

///Treating actions and states as usize for now. Probably this is the best way.
///Both actions and states better start from 0 and be continuous.
/// 
/// todo: 
/// (1). Transition(s,a), distributions
/// (2). R(s,a), rewards
/// Now it's all in a get_prob_reward function which is not the best.

pub type Policy = usize;
pub type Action = usize;
pub type State = usize; 
pub trait StateSpace {
    fn get_actions_at_state(&self, s:State) -> Vec<Action>;
    fn get_prob_reward(&self, s:State, a:Action) -> Vec<(State, f64, f64)>;
    fn get_all_states(&self) -> &Vec<State>;
    fn get_state_count(&self) -> usize;
    fn get_idx_from_state(&self, s:State) -> usize;
    fn get_state_from_idx(&self, i:usize) -> State;
    fn is_terminal_state(&self, s:usize) -> bool;
}

pub struct MDP<SP: StateSpace + std::marker::Sync> {
    state_space: SP, 
    default_action: Action,
    gamma: f64
}

impl <SP: StateSpace + std::marker::Sync> MDP<SP> {
    pub fn new(state_space:SP, default:Action, gamma:f64) -> MDP<SP> {
        // kind of stupid here..
        MDP{state_space: state_space, default_action:default, gamma: gamma}
    }

    pub fn value_iteration(&self, epislon:f64) -> Vec<Policy> {
        //Regular version
        let mut v:Vec<f64> = vec![0.; self.state_space.get_state_count()];
        loop {
            let max_diff:f64 = self.state_space.get_all_states()
            .iter()
            .fold(0., |accum:f64, s:&State|{
                if self.state_space.is_terminal_state(*s){
                    accum
                } else {
                    let idx:usize = self.state_space.get_idx_from_state(*s);
                    let old_v:f64 = v[idx];
                    // Side effect
                    v[idx] = self.state_space.get_actions_at_state(*s)
                                .iter()
                                .fold(f64::MIN, |acc:f64, action:&Action| 
                                    acc.max(self.q(&v, *s, *action))
                                );
                    accum.max((old_v - v[idx]).abs())
                }
            });
            if max_diff < epislon {break}
        }

        self.state_space.get_all_states().iter()
        .map(|s:&State| {
            if self.state_space.is_terminal_state(*s) {
                self.default_action
            } else {
                self.better_action(&v, *s).1
            }
        }).collect()
    }

    pub fn par_value_iteration(&self, epislon:f64) -> Vec<Policy> {
        //Parallel version
        let n:usize = self.state_space.get_state_count();
        let mut v:Vec<f64> = vec![0.; n];
        // let mut v:Vec<f64> = vec![0.; self.state_space.get_state_count()];
        loop {
            let mut next_v:Vec<f64> = Vec::with_capacity(n);
            // Compute next v first, in parallel
            (0..n).into_par_iter().map(|i:usize| {
                let s:State = self.state_space.get_state_from_idx(i);
                if self.state_space.is_terminal_state(s) {
                    0.
                } else {
                    self.state_space.get_actions_at_state(s).iter()
                        .fold(f64::MIN, |acc:f64, action:&Action| 
                            acc.max(self.q(&v, s, *action))
                        )
                }
            }).collect_into_vec(&mut next_v);
            // Find max diff (L1 norm), in parallel
            let max_diff:f64 = next_v.par_iter().zip(&v).fold(|| 0.,
                |diff:f64, (new, old):(&f64,&f64)| diff.max((old-new).abs())
            ).max_by(|x,y| x.total_cmp(y)).unwrap();
            
            // update v
            v = next_v;
            if max_diff < epislon {
                break
            }
        }
        // Given optimal v, create optimal policy in parallel.
        let mut optimal_policy:Vec<Policy> = Vec::with_capacity(n);
        self.state_space.get_all_states().par_iter().map(|s:&State|{
            if self.state_space.is_terminal_state(*s) {
                self.default_action
            } else {
                self.better_action(&v, *s).1
            }
        }).collect_into_vec(&mut optimal_policy);
        optimal_policy

    }

    pub fn policy_iteration(&self, epislon:f64) -> Vec<Policy> {
        // let mapping_prep = mdp.state_space.get_all_states();
        let state_count:usize = self.state_space.get_state_count();
        let mut pi:Vec<Policy> = vec![self.default_action; state_count];
        let mut v:Vec<f64> = vec![0.; state_count];
        loop {
            loop { // value of a policy
                let max_diff:f64 = self.state_space.get_all_states()
                    .iter()
                    .fold(0., |acc, s|{
                        if self.state_space.is_terminal_state(*s){
                            acc
                        } else {
                            let idx:usize = self.state_space.get_idx_from_state(*s);
                            let old_v:f64 = v[idx];
                            // side effect, notice here we refer to the action given by pi
                            v[idx] = self.q(&v, *s, pi[idx]);
                            acc.max((old_v - v[idx]).abs())
                        }
                });
                // element-wise check max |difference|, which is equivalent to L1 between two vectors.
                // check convergence, e.g. L1(V - OLD_V) < epsilon
                if max_diff < epislon {break} // end of loop
            };

            let mut stable:bool = true;
            pi.iter_mut().enumerate().for_each(|(idx, p)| {
                let s:usize = self.state_space.get_state_from_idx(idx);
                if self.state_space.is_terminal_state(s){
                    *p = self.default_action;
                } else {
                    let old_action:Action = *p;
                    let new_action:Action = self.better_action(&v, s).1;
                    *p = new_action;
                    if stable {stable = new_action == old_action;} // side effect
                }
            });
            if stable {break pi} // break if stable, and return pi
        }
    }

    pub fn par_policy_iteration(&self, epislon:f64) -> Vec<Policy> {
        let state_count:usize = self.state_space.get_state_count();
        let mut pi:Vec<Policy> = vec![self.default_action; state_count]; // !!!todo. Action should supply a default/initialization/NONE scheme
        let mut v:Vec<f64> = vec![0.; state_count];

        loop {
            loop {
                let mut next_v:Vec<f64> = Vec::with_capacity(state_count);
                // Compute next v first, in parallel
                (0..state_count).into_par_iter().map(|i:usize| {
                    let s:State = self.state_space.get_state_from_idx(i);
                    if self.state_space.is_terminal_state(s) {
                        0.
                    } else {
                        self.q(&v, s, pi[i])
                    }
                }).collect_into_vec(&mut next_v);
                
                let max_diff:f64 = next_v.par_iter().zip(&v).fold(|| 0.,
                    |diff:f64, (new, old):(&f64,&f64)| diff.max((old-new).abs())
                ).max_by(|x,y| x.total_cmp(y)).unwrap();

                v = next_v;
                if max_diff < epislon {break}
            }

            // need this for future comparison. This version is slightly faster.
            let old_pi:Vec<Policy> = pi.clone();
            self.state_space.get_all_states().par_iter()
            .map(|s:&State| {
                if self.state_space.is_terminal_state(*s) {
                    self.default_action
                } else {
                    self.better_action(&v, *s).1
                }
            }).collect_into_vec(&mut pi);
            // Not the "best" comparison.
            if old_pi == pi { // means stable
                break pi
            }

            // let mut stable:bool = true;
            // pi.iter_mut().enumerate().for_each(|(idx, p)| {
            //     let s:usize = self.state_space.get_state_from_idx(idx);
            //     if self.state_space.is_terminal_state(s){
            //         *p = self.default_action;
            //     } else {
            //         let old_action:Action = *p;
            //         let new_action:Action = self.better_action(&v, s).1;
            //         *p = new_action;
            //         if stable {stable = new_action == old_action;} // side effect
            //     }
            // });
            // if stable {break pi} // break if stable, and return pi
        }

    }


    #[inline]
    fn q(&self, current_values:&Vec<f64>, state:State, a:Action) -> f64 {
        // I think performance can be improved more by asking get_prob_reward to return a generator instead.
        // But I won't implement that for simplicity.

        self.state_space
        .get_prob_reward(state, a)
        .into_iter()
        .fold(0., 
            |acc:f64, (st, p, r):(State, f64, f64)| acc + p*(r + self.gamma * current_values[self.state_space.get_idx_from_state(st)])
        )
    }

    #[inline]
    ///Given v, returns the better action for the current state.
    fn better_action(&self, v:&Vec<f64>, s:State) -> (f64, Action) {
        self.state_space.get_actions_at_state(s)
        .into_iter()
        .fold((f64::MIN, self.default_action), |accum:(f64, Action), a:Action| {
            let value_given_a:f64 = self.q(v, s, a);
            if value_given_a > accum.0 {
                (value_given_a, a)
            } else {
                accum // (value given action, action)
            }
        })
    }
}


// EXAMPLES 

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
    states: Vec<State>,
    start: State,
    end: State // one terminal state in this problem. 
}

impl TramSpace {
    pub fn new(all_states:Vec<State>, start:State, end:State) -> TramSpace {
        
        let mut true_end:State = end;
        if end <= start {
            println!{"Warning: Found end <= start. Forcing end = start + 10."};
            true_end = start + 10;
        }
        TramSpace {states: all_states, start: start, end: true_end}
    }
}

impl StateSpace for TramSpace {

    #[inline]
    fn get_all_states(&self) -> &Vec<usize> {
        &self.states 
    }

    #[inline]
    fn get_state_count(&self) -> usize {
        self.states.len()
    }
    
    #[inline]
    fn get_idx_from_state(&self, s:State) -> usize {
        s - self.start
    }

    #[inline]
    fn get_state_from_idx(&self, i:usize) -> State {
        i + self.start
    }
    
    #[inline]
    fn is_terminal_state(&self, s:State) -> bool {
        s == self.end
    }
    
    fn get_actions_at_state(&self, s:State) -> Vec<Action> {
        // Action to usize mapping : 0 <-> NONE, 1 <-> WALK, 2 <-> TRAM
        // this function should be specific to the game
        let mut actions:Vec<Action> = Vec::new();
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
    ///s: state (State)
    ///a: action (Action)
    ///returns (next State, probability, reward)
    fn get_prob_reward(&self, s:State, a:Action) -> Vec<(State, f64, f64)> {
        let mut output:Vec<(State, f64, f64)> = Vec::new();
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
