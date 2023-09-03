use crate::markov_decision_process::{
    Action,
    State,
    StateSpace
};

const DIEN_ACTIONS:[usize;2] = [0,1]; // 0: QUIT, 1: ROLL

pub struct DieN {
    n: usize,
    bad_prob: f64,
    good_numbers: Vec<u32>,
    stop_at: usize,
    states:Vec<State>
}

impl DieN {
    pub fn new(config:Vec<u32>) -> Self {
        // Configuration. Bad ones are encoded as 1.
        let n = config.len();
        let mut good_numbers: Vec<u32> = Vec::new();
        let mut good_sum:u32 = 0;
        let mut bad_count:u32 = 0;
        for (i,x) in config.into_iter().enumerate() {
            if x == 1 {
                bad_count +=1;
            } else {
                good_numbers.push(i as u32);
                good_sum += i as u32;
            }
        }

        let bad_prob: f64 = bad_count as f64 / n as f64;
        let rewards_sum:u32 = good_sum + good_numbers.len() as u32;
        let upper_bound = ((rewards_sum as f64/n as f64)/bad_prob).ceil() as usize;

        DieN {
            n: n,
            bad_prob: bad_prob,
            good_numbers: good_numbers,
            stop_at: upper_bound,
            states: (0..=upper_bound).collect::<Vec<State>>()
        }
    }
}

impl StateSpace for DieN {
    fn get_actions_at_state(&self, s:&State) -> Vec<Action> {
        if self.is_terminal_state(s) {
            vec![0]
        } else {
            Vec::from_iter(DIEN_ACTIONS) // 0: QUIT, 1: ROLL
        }
    }

    fn get_future_rewards(&self, s:&State, a:&Action) -> Vec<(State, f64, f64)> {
        if (s >= &self.stop_at) | (a == &0) {
            vec![(self.stop_at, 1.0, 0.)]
        } else {
            let mut out:Vec<(State, f64, f64)> = Vec::with_capacity(self.good_numbers.len() + 1);
            out.push((self.stop_at, self.bad_prob, -(*s as f64)));
            let p:f64 = 1./self.n as f64;
            for i in &self.good_numbers {
                let next: usize = self.stop_at.min(s + (i+1) as usize);
                out.push((next, p, (i+1) as f64));
            }
            out
        }
    }

    fn get_all_states(&self) -> std::slice::Iter<'_, usize> {
        self.states.iter()
    }

    fn len(&self) -> usize {
        self.n
    }

    fn is_terminal_state(&self, s:&State) -> bool {
        s >= &self.stop_at
    }
}
