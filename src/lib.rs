///Treating actions and states as usize for now. Probably this is the best way.
///Both actions and states better start from 0 and be continuous.
/// 
/// todo: 
/// (1). Transition(s,a), distributions
/// (2). R(s,a), rewards
/// Now it's all in a get_prob_reward function which is not the best.
pub trait StateSpace {

    fn get_actions_at_state(&self, s:usize) -> Vec<usize>;
    fn get_prob_reward(&self, s:usize, a:usize) -> Vec<(usize, f32, f32)>;
    fn get_all_states(&self) -> &Vec<usize>;
    fn get_state_count(&self) -> usize;
    fn get_state_idx(&self, s:usize) -> usize;
    fn get_state_from_idx(&self, i:usize) -> usize;
    fn is_terminal_state(&self, s:usize) -> bool;
}

pub struct MDP<SP: StateSpace> {
    state_space: SP, 
    actions: Vec<usize>, // iterable structure of usizes, see todo below.
    gamma: f32
}

impl <SP: StateSpace> MDP<SP>
{
    pub fn new(state_space:SP, actions:Vec<usize>, gamma:f32) -> MDP<SP> {
        MDP{state_space: state_space, actions: actions, gamma: gamma}
    }

    pub fn value_iteration(&self, epislon:f32) -> Vec<usize> {
        let mut v:Vec<f32> = vec![0.;self.state_space.get_state_count()];
        loop {
            let max_diff:f32 = self.state_space.get_all_states()
            .iter()
            .fold(0., |accum:f32, s:&usize|{
                if self.state_space.is_terminal_state(*s){
                    accum
                } else {
                    let idx:usize = self.state_space.get_state_idx(*s);
                    let old_v:f32 = v[idx];
                    v[idx] = self.state_space.get_actions_at_state(*s)
                                .iter()
                                .fold(f32::MIN, |acc:f32, action:&usize| 
                                    {acc.max(self.q(&v, *s, *action))}
                                );
                    accum.max((old_v - v[idx]).abs())
                }
            });
            if max_diff < epislon {break}
        }

        self.state_space.get_all_states().iter()
        .map(|s| {
            if self.state_space.is_terminal_state(*s) {
                0
            } else {
                self.better_action(&v, *s).1
            }
        })
        .collect()
    }

    /// 
    /// Generally faster than value iteration.
    /// 
    pub fn policy_iteration(&self, epislon:f32) -> Vec<usize> {
        // let mapping_prep = mdp.state_space.get_all_states();
        // 0th action is used as default.
        let state_count:usize = self.state_space.get_state_count();
        let mut pi:Vec<usize> = vec![self.actions[0]; state_count]; // !!!todo. Action should supply a default/initialization/NONE scheme
        let mut v:Vec<f32> = vec![0.; state_count];
        loop {
            loop { // value of a policy
                let max_diff:f32 = self.state_space.get_all_states()
                    .iter()
                    .fold(0., |acc, s|{
                        if self.state_space.is_terminal_state(*s){
                            acc
                        } else {
                            let idx:usize = self.state_space.get_state_idx(*s);
                            let old_v:f32 = v[idx]; 
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
                let s = self.state_space.get_state_from_idx(idx);
                if self.state_space.is_terminal_state(s){
                    *p = 0; // 0 is None
                } else {
                    let old_action = *p;
                    let better_pair = self.better_action(&v,s);
                    *p = better_pair.1;
                    if stable {
                        stable = better_pair.1 == old_action;
                    } // side effect
                }
            });
            if stable {break} // break if stable, and return pi
        }
        pi // return pi
    }

    fn q(&self, current_values:&Vec<f32>, state:usize, a:usize) -> f32 {
        // self.state_space
        // .get_prob_reward(state, a)
        // .into_iter()
        // .fold(0., 
        //     |acc:f32, (st, p, r)| acc + p*(r + self.gamma * current_values[self.state_space.get_state_idx(st)])
        // )
        self.state_space
        .get_prob_reward(state, a)
        .into_iter()
        .map(|(st, p, r)| p*(r + self.gamma * current_values[self.state_space.get_state_idx(st)]))
        .sum::<f32>()
    }

    ///
    ///Given v, returns the better action for the current state.
    /// 
    fn better_action(&self, v:&Vec<f32>, s:usize) -> (f32, usize) {
        self.state_space.get_actions_at_state(s)
        .iter()
        .fold((f32::MIN, 0), |accum, a| {
            let value_given_a = self.q(v, s, *a);
            if value_given_a > accum.0 {
                (value_given_a, *a)
            } else {
                accum
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
    states: Vec<usize>,
    start: usize,
    end: usize // one terminal state in this problem. 
}

impl TramSpace {
    pub fn new(all_states:Vec<usize>, start:usize, end:usize) -> TramSpace {
        
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
