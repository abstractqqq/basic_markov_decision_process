use criterion::{criterion_group, criterion_main, Criterion};
use markov_decision::*;


// ----------------------------------------------------- Tram Game ---------------------------------------------------
#[derive(Debug, PartialEq)]
enum Actions {
    NONE,
    WALK,
    TRAM
}

// TramSpace is specific to the Tram Problem and nothing else.
struct TramSpace{
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

// ----------------------------------------------------- Volcano Game ------------------------------------------------




//------------------------------------------------------ Benchmarks --------------------------------------------------
fn run_policy_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<usize>{
    mdp_problem.policy_iteration(1e-6)
}

fn run_value_iteration(mdp_problem:&MDP<TramSpace>) -> Vec<usize>{
    mdp_problem.value_iteration(1e-6)
}

fn criterion_benchmark(c: &mut Criterion) {
    let states = (1..=100).collect();
    let actions = vec![0,1,2];
    let tram = TramSpace::new(states, 1, 100);
    let mdp_problem = MDP::new(tram, actions, 1.);
    c.bench_function("run_policy_iteration", |b| b.iter(|| run_policy_iteration(&mdp_problem)));
    c.bench_function("run_value_iteration", |b| b.iter(|| run_value_iteration(&mdp_problem)));
}


criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);