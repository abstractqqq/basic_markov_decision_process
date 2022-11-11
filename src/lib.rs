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

impl <SP: StateSpace> MDP<SP>{
    pub fn new(state_space:SP, actions:Vec<usize>, gamma:f32) -> MDP<SP> {
        MDP{state_space: state_space, actions: actions, gamma: gamma}
    }

    pub fn value_iteration(&self, epislon:f32) -> Vec<usize> {
        let mut v:Vec<f32> = vec![0.;self.state_space.get_state_count()];
        loop {
            let max_diff:f32 = self.state_space.get_all_states()
            .into_iter()
            .fold(0., |accum, s|{
                if self.state_space.is_terminal_state(*s){
                    accum
                } else {
                    let idx:usize = self.state_space.get_state_idx(*s);
                    let old_v:f32 = v[idx];
                    v[idx] = self.state_space.get_actions_at_state(*s)
                                .into_iter()
                                .fold(f32::MIN, |acc:f32, action:usize| 
                                    {acc.max(self.q(&v, *s, action))}
                                );
                    accum.max((old_v - v[idx]).abs())
                }
            });
            if max_diff < epislon {break}
        }
        // return value
        self.state_space.get_all_states().into_iter()
        .map(|s|{
            if self.state_space.is_terminal_state(*s){
                0 // see to do below.
            } else {
                // Given the optimized V. Compute Q(s,a) and find the max action for every state. Choose that action.
                // Error handling?
                self.better_action(&v, *s).unwrap().1
            }
        }).collect()
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
            let mut stable:bool = loop { // value of a policy
                let max_diff:f32 = self.state_space.get_all_states()
                    .into_iter()
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
                // This approach might be faster because I am not copying vectors and doing non-optimized vector L1 norm computations.
                // Purely scalar computation. 

                // check convergence, e.g. L1(V - OLD_V) < epsilon
                if max_diff < epislon {break true}
            };
    
            // update policy in place, with a side effect because we are also changing stable.
            pi.iter_mut().enumerate().for_each(|(idx, p)| {
                let s = self.state_space.get_state_from_idx(idx);
                if self.state_space.is_terminal_state(s){
                    *p = 0;
                } else {
                    let old_action = *p;
                    let better_pair = self.better_action(&v,s);
                    if let Some(pair) = better_pair {
                        *p = pair.1;
                        if stable {stable = pair.1 == old_action;} // side effect
                    }
                }
            });
            if stable {break pi} // break if stable, and return pi
        }
    }

    fn q(&self, current_values:&Vec<f32>, state:usize, a:usize) -> f32 {
        self.state_space
        .get_prob_reward(state, a)
        .into_iter()
        .fold(0., 
            |acc:f32, (st, p, r)| acc + p*(r + self.gamma * current_values[self.state_space.get_state_idx(st)])
        )
    }

    ///
    ///Given v, returns the better action for the current state.
    ///returns Optional<(better Q value, usize for the action)>. It is optional because reduce may fail.
    /// 
    fn better_action(&self, v:&Vec<f32>, s:usize) -> Option<(f32, usize)> {
        self.state_space.get_actions_at_state(s)
        .into_iter()
        .map(|a| (self.q(v, s, a), a)) // create the (v, a) pair
        .reduce(|acc, v_a| if v_a.0 > acc.0 {v_a} else {acc}) // find the action that maximizes Q(s,a)
    }
}
