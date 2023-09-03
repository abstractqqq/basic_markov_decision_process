// use ndarray::{Array1,ArrayView1, parallel::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator}};

use std::{cell::RefCell, slice::Iter};

pub type Action = usize; // see State.
pub type State = usize; // 
pub type Policy = Vec<Action>;

#[derive(Debug)]
pub enum MDPSolver {
    VALUE_ITER,
    POLICY_ITER
}

impl Default for MDPSolver {
    fn default() -> Self {
        MDPSolver::POLICY_ITER
    }
}

impl From<&str> for MDPSolver {
    fn from(value: &str) -> Self {
        match value {
            "policy_iteration" | "policy" => MDPSolver::POLICY_ITER,
            "value_iteration" | "value" => MDPSolver::VALUE_ITER,
            _ => MDPSolver::default()
        }
    }
}

pub trait StateSpace {
    fn get_actions_at_state(&self, s:&State) -> Vec<Action>;
    // return type: next_state, prob, reward
    fn get_future_rewards(&self, s:&State, a:&Action) -> Vec<(State, f64, f64)>;
    // Assume that get_all_states returns all states in the right order, state 0 is at index 0 and so on..
    fn get_all_states(&self) -> Iter<'_, State>; // can this be abstracted away as a trait?
    fn len(&self) -> usize;
    fn is_terminal_state(&self, s:&State) -> bool;
}

pub struct MarkovDecisionProcess<S: StateSpace + std::marker::Sync> {
    state_space: S, 
    default_action: Action,
    gamma: f64,
    learned_values:RefCell<Vec<f64>>
}

impl <'a, S: StateSpace + std::marker::Sync> MarkovDecisionProcess<S> {

    pub fn new(state_space:S, default:Action, gamma:f64) -> MarkovDecisionProcess<S> {
        let values = vec![0.;state_space.len()];
        MarkovDecisionProcess {
            state_space: state_space, 
            default_action:default,
            gamma: gamma,
            learned_values: RefCell::new(values)
        }
    }

    pub fn reset_values(&mut self) {
        *self.learned_values.borrow_mut() = vec![0.;self.state_space.len()];
    }
    
    #[inline]
    fn best_action(&self, current_values:&Vec<f64>, s:&State) -> (f64, Action) {
        //Given v, returns the better action for the current state.
        self.state_space.get_actions_at_state(s)
        .into_iter()
        .fold((f64::MIN, self.default_action), |accum:(f64, Action), a:Action| {
            let value_given_a:f64 = self.q(current_values, s, &a);
            if value_given_a > accum.0 {
                (value_given_a, a)
            } else { // (value given action, action)
                accum
            }
        })
    }

    #[inline]
    fn q(&self, current_values:&Vec<f64>, state:&State, action:&Action) -> f64 {
        // first get all possible (next_state, r) for this 
        self.state_space.get_future_rewards(state, action)
        .into_iter()
        .fold(0., |acc:f64, (next, p, r)| 
            acc + p * (r + self.gamma * current_values[next])
        )
    }

    pub fn update_value(&self) -> f64 {

        let new_values:Vec<f64> = self.state_space.get_all_states()
        .map(|s| {
            // find the action that has the highest Q value.
            // Terminal states will have f64::Min as its value.
            self.state_space.get_actions_at_state(s)
            .iter()
            .fold(f64::MIN, |acc:f64, action:&Action| 
                acc.max(self.q(&self.learned_values.borrow(), s, action))
            ) 
        }).collect::<Vec<f64>>();
        
        let mut learned_mut = self.learned_values.borrow_mut();
        let max_diff: f64 = new_values.into_iter().enumerate().fold(
            0., |acc:f64, (i, v)| {
                if self.state_space.is_terminal_state(&i) {
                    acc
                } else {
                    let abs_diff: f64 = (v - learned_mut[i]).abs();
                    learned_mut[i] = v; // side effect
                    acc.max(abs_diff)
                }
            }
        );
        max_diff
    }

    pub fn value_iteration(&mut self, epislon:f64) -> Policy {

        self.reset_values();
        loop {
            // let mut new_values:Vec<f64> = Vec::with_capacity(self.state_space.len());
            // Non-parallel version is faster for small games. This is expected.
            let new_values:Vec<f64> = self.state_space.get_all_states()
            .map(|s| {
                // find the action that has the highest Q value.
                // Terminal states will have f64::Min as its value.
                self.state_space.get_actions_at_state(s)
                .iter()
                .fold(f64::MIN, |acc:f64, action:&Action| 
                    acc.max(self.q(&self.learned_values.borrow(), s, action))
                ) 
            }).collect::<Vec<f64>>();
            
            let mut learned_mut = self.learned_values.borrow_mut();
            let max_diff: f64 = new_values.into_iter().enumerate().fold(
                0., |acc:f64, (i, v)| {
                    if self.state_space.is_terminal_state(&i) {
                        acc
                    } else {
                        let abs_diff: f64 = (v - learned_mut[i]).abs();
                        learned_mut[i] = v; // side effect
                        acc.max(abs_diff)
                    }
                }
            );
            if max_diff < epislon {
                break
            }
        }

        self.state_space.get_all_states()
        .map(|s| {
            if self.state_space.is_terminal_state(s) {
                self.default_action
            } else {
                self.best_action(&self.learned_values.borrow(), s).1
            }
        }).collect::<Vec<Action>>()
    }

    pub fn policy_iteration(&mut self, epsilon:f64) -> Policy {

        self.reset_values();
        let state_count:usize = self.state_space.len();
        let mut pi:Vec<Action> = vec![self.default_action; state_count];
        loop {
            loop {
                let mut learned_mut = self.learned_values.borrow_mut();
                let max_diff:f64 = self.state_space.get_all_states()
                    .fold(0., |acc, s|{
                        if self.state_space.is_terminal_state(s){
                            acc
                        } else {
                            let old_v:f64 = learned_mut[*s];
                            // side effect, notice here we refer to the action given by pi
                            let new_val: f64 = self.q(&learned_mut, s, &pi[*s]);
                            learned_mut[*s] = new_val;
                            acc.max((old_v - new_val).abs())
                        }
                });
                if max_diff < epsilon {break}
            }

            let mut stable:bool = true;
            pi.iter_mut().enumerate().for_each(|(i, p): (usize, &mut usize)| {
                if !self.state_space.is_terminal_state(&i){
                    let old_action:Action = *p;
                    let new_action:Action = self.best_action(&self.learned_values.borrow(), &i).1;
                    *p = new_action;
                    if stable {
                        stable = new_action == old_action; // side effect
                    } 
                }
            });
            if stable {
                return pi;
            }
        }
    }

    pub fn get_state_space(&self) -> &S {
        &self.state_space
    }

    pub fn get_learned_values(&self) -> Vec<f64> {
        let values = self.learned_values.clone();
        values.take()
    }

    pub fn show_learned_values(&self) {
        println!("{:?}", self.learned_values);
    }

    pub fn solve(&mut self, method:MDPSolver, epsilon:f64) -> Policy {
        self.reset_values();
        println!("\nSolve by {:?}", method);
        match method {
            MDPSolver::POLICY_ITER => {
                self.policy_iteration(epsilon)
            }
            MDPSolver::VALUE_ITER => {
                self.value_iteration(epsilon)
            }
        }
    }


}
