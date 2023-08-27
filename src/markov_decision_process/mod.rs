use ndarray::{Array1,ArrayView1, parallel::prelude::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator}};


pub type Action = usize; // see State.
pub type State = usize; // 
pub type Policy = Vec<Action>;

pub trait StateSpace {
    fn get_actions_at_state(&self, s:&State) -> Vec<Action>;
    // return type: next_state, prob, reward
    fn get_future_rewards(&self, s:&State, a:&Action) -> Vec<(State, f64, f64)>;
    fn get_all_states(&self) -> Vec<&State>; // Assume that get_all_states returns all states in the right order
    fn len(&self) -> usize;
    fn is_terminal_state(&self, s:&State) -> bool;
    // fn get_idx_from_state(&self, s:&State) -> usize; State 0 in get_all_states has index 0
    // fn get_state_from_idx(&self, i:usize) -> State;
}

pub struct MarkovDecisionProcess<'a, S: StateSpace + std::marker::Sync> {
    state_space: &'a S, 
    default_action: Action,
    gamma: f64
}

impl <'a, S: StateSpace + std::marker::Sync> MarkovDecisionProcess<'a, S> {

    pub fn new(state_space:&'a S, default:Action, gamma:f64) -> MarkovDecisionProcess<'a, S> {
        MarkovDecisionProcess {
            state_space: state_space, 
            default_action:default,
            gamma: gamma
        }
    }
    
    #[inline]
    fn best_action(&self, current_values:ArrayView1<f64>, s:&State) -> (f64, Action) {
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
    fn q(&self, current_values:ArrayView1<f64>, state:&State, action:&Action) -> f64 {
        // first get all possible (next_state, r) for this 
        self.state_space.get_future_rewards(state, action)
        .into_iter()
        .fold(0., |acc:f64, (next, p, r)| 
            acc + p * (r + self.gamma * current_values[next])
        )
    }

    pub fn value_iteration(&self, epislon:f64) -> Policy {

        let mut values:Array1<f64> = Array1::from_elem(self.state_space.len(), 0.);
        
        loop {
            let mut new_values:Vec<f64> = Vec::with_capacity(self.state_space.len());
            self.state_space.get_all_states()
            .par_iter()
            .map(|&s| {
                // find the action that has the highest Q value.
                // Terminal states will have f64::Min as its value.
                self.state_space.get_actions_at_state(s)
                .iter()
                .fold(f64::MIN, |acc:f64, action:&Action| 
                    acc.max(self.q(values.view(), s, action))
                ) 
            }).collect_into_vec(&mut new_values);

            let max_diff: f64 = new_values.into_iter().enumerate().fold(
                0., |acc:f64, (i, v)| {
                    if self.state_space.is_terminal_state(&i) {
                        acc
                    } else {
                        let abs_diff: f64 = (v-values[i]).abs();
                        values[i] = v;
                        acc.max(abs_diff)
                    }
                }
            );
            if max_diff < epislon {
                break
            }
        }

        self.state_space.get_all_states()
        .iter()
        .map(|&s| {
            if self.state_space.is_terminal_state(s) {
                self.default_action
            } else {
                self.best_action(values.view(), s).1
            }
        }).collect::<Vec<Action>>()
    }
}
