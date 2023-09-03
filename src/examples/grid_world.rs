use std::collections::HashMap;
use crate::markov_decision_process::{
    Action,
    State,
    StateSpace
};

#[derive(Debug)]
pub enum Movements {
    UP,
    LEFT,
    DOWN,
    RIGHT,
    NOTHING
}

impl ToString for Movements {
    fn to_string(&self) -> String {
        match self {
            Movements::NOTHING => "-".to_owned(),
            Movements::UP => "↑".to_owned(),
            Movements::DOWN => "↓".to_owned(),
            Movements::LEFT => "←".to_owned(),
            Movements::RIGHT => "→".to_owned(),
        }
    }
}

impl Movements {
    pub fn from_usize(u: usize) -> Movements {
        match u {
            1 => Movements::UP,
            2 => Movements::LEFT, 
            3 => Movements::DOWN,
            4 => Movements::RIGHT,
            _ => Movements::NOTHING,
        }
    }
}

#[derive(Clone)]
pub struct GridWorld {
    width: usize,
    height: usize,
    unreachable: Vec<(usize, usize)>,
    terminal:Vec<(usize, usize)>,
    default_reward: f64,
    special_reward: HashMap<(usize, usize), f64>,
    all_states: Vec<usize>,
}

impl Default for GridWorld {
    // default is the one illustrated in the lecture
    fn default() -> Self {
        GridWorld { 
            width: 4,
            height: 3, 
            unreachable: vec![(1,1)], 
            terminal: vec![(3,0), (3,1)], 
            default_reward: -0.04, 
            special_reward: HashMap::from_iter([((3,0), 1.0), ((3,1), -1.0)]),
            all_states: (0..12).collect(),
        }
    }
}

impl StateSpace for GridWorld {

    fn get_actions_at_state(&self, s:&State) -> Vec<Action> {
        let coord = self.get_coord_from_idx(s);
        if self.unreachable.contains(&coord) | self.special_reward.contains_key(&coord) {
            return Vec::new()
        }        
        vec![1,2,3,4]
    }

    fn get_future_rewards(&self, s:&State, a:&Action) -> Vec<(State, f64, f64)> {

        let coord = self.get_coord_from_idx(s);
        let (x, y) = coord;
        
        if a == &1 {
            let (s1, r1) = self.move_up(x, y);
            let (s2, r2) = self.move_left(x, y);
            let (s3, r3) = self.move_right(x, y);
            return vec![(s1, 0.8, r1), (s2, 0.1, r2), (s3, 0.1, r3)]
        } else if a == &2 {
            let (s1, r1) = self.move_left(x, y);
            let (s2, r2) = self.move_up(x, y);
            let (s3, r3) = self.move_down(x, y);
            return vec![(s1, 0.8, r1), (s2, 0.1, r2), (s3, 0.1, r3)]
        } else if a == &3 {
            let (s1, r1) = self.move_down(x, y);
            let (s2, r2) = self.move_left(x, y);
            let (s3, r3) = self.move_right(x, y);
            return vec![(s1, 0.8, r1), (s2, 0.1, r2), (s3, 0.1, r3)]
        } else if a == &4 {
            let (s1, r1) = self.move_right(x, y);
            let (s2, r2) = self.move_up(x, y);
            let (s3, r3) = self.move_down(x, y);
            return vec![(s1, 0.8, r1), (s2, 0.1, r2), (s3, 0.1, r3)]
        }
        Vec::new()
    }

    fn get_all_states(&self) -> std::slice::Iter<'_, usize> {
        self.all_states.iter()
    }

    fn len(&self) -> usize {
        self.width * self.height
    }

    fn is_terminal_state(&self, s:&State) -> bool {
        // In this game, special reward state = terminal state
        let coord = self.get_coord_from_idx(s);
        self.unreachable.contains(&coord) | self.special_reward.contains_key(&coord)
    }
}

impl GridWorld {

    fn move_up(&self, x:usize, y:usize) -> (State, f64) {
        let mut next = if y > 0 {
            (x, y-1)
        } else {
            (x, y)
        };
        if self.unreachable.contains(&next) {
            next = (x,y);
        }
        let r: f64 = *self.special_reward.get(&next).unwrap_or(&self.default_reward);
        let s: usize = self.get_idx_from_coord(next.0, next.1);
        (s, r)
    }

    fn move_down(&self, x:usize, y:usize) -> (State, f64) {
        let mut next = if y < self.height - 1 {
            (x, y+1)
        } else {
            (x, y)
        };
        if self.unreachable.contains(&next) {
            next = (x, y);
        }
        let r: f64 = *self.special_reward.get(&next).unwrap_or(&self.default_reward);
        let s: usize = self.get_idx_from_coord(next.0, next.1);
        (s, r)
    }

    fn move_left(&self, x:usize, y:usize) -> (State, f64) {
        let mut next = if x > 0 {
            (x-1, y)
        } else {
            (x, y)
        };
        if self.unreachable.contains(&next) {
            next = (x, y);
        }
        let r: f64 = *self.special_reward.get(&next).unwrap_or(&self.default_reward);
        let s: usize = self.get_idx_from_coord(next.0, next.1);
        (s, r)
    }

    fn move_right(&self, x:usize, y:usize) -> (State, f64) {
        let mut next = if x < self.width - 1 {
            (x+1, y)
        } else {
            (x, y)
        };
        if self.unreachable.contains(&next) {
            next = (x, y);
        }
        let r: f64 = *self.special_reward.get(&next).unwrap_or(&self.default_reward);
        let s: usize = self.get_idx_from_coord(next.0, next.1);
        (s, r)
    }

    fn get_idx_from_coord(&self, x:usize, y:usize) -> usize {
        y * self.width + x
    }

    fn get_coord_from_idx(&self, idx: &usize) -> (usize, usize) {
        let col: usize = idx / self.width;
        let row = idx -  self.width * col;
        (row, col)
    }

    pub fn print_world(&self) {
        println!("\nGame world initial set up:\n` XXX ` means unreachable\nNumbers represent the rewards.\n");
        println!("Top left is (0, 0)");
        let terminal_states = self.terminal.iter().map(|(a,b)| {
            let mut s = String::new();
            s.push('(');
            s.push_str(&a.to_string());
            s.push(',');
            s.push_str(&b.to_string());
            s.push(')');
            s
        }).collect::<Vec<String>>();
        println!("Terminal States are : {}", terminal_states.join(","));

        let x_offset:usize = 2;
        let y_offset:usize = 1;
        let mut grid = String::new();
        for y in 0..self.height + 2*y_offset {
            for x in 0..self.width + 2*x_offset {
                if (y < y_offset) | (y >= y_offset + self.height) {
                    // top boundary
                    if (x < x_offset) | (x >= x_offset + self.width) {
                        grid.push('o');
                    } else {
                        grid.push_str("ooooooo");
                    }
                } else {
                    if (x < x_offset) | (x >= x_offset + self.width) {
                        grid.push('o'); // left right boundary
                    } else {
                        let mapped_coord = (x-x_offset,y-y_offset);
                        if self.unreachable.contains(&mapped_coord) {
                            grid.push_str("  XXX  ");
                        } else {
                            let reward = self.special_reward.get(&mapped_coord).unwrap_or(&self.default_reward);
                            let mut reward_str = reward.to_string();
                            reward_str.truncate(5);
                            let padded = format!("{:^7}", reward_str);
                            grid.push_str(&padded);
                        }
                    }

                }
            }
            grid.push('\n');
        }
        println!("{}", grid);
    }

    pub fn print_on_states<T>(&self, to_print:Vec<T>)
    where T: ToString
    {
        let x_offset:usize = 2;
        let y_offset:usize = 1;
        let mut grid = String::new();
        for y in 0..self.height + 2*y_offset {
            for x in 0..self.width + 2*x_offset {
                if (y < y_offset) | (y >= y_offset + self.height) {
                    // top boundary
                    if (x < x_offset) | (x >= x_offset + self.width) {
                        grid.push('o');
                    } else {
                        grid.push_str("ooooooo");
                    }
                } else {
                    if (x < x_offset) | (x >= x_offset + self.width) {
                        grid.push('o'); // left right boundary
                    } else {
                        let mapped_coord = (x-x_offset,y-y_offset);
                        if self.unreachable.contains(&mapped_coord) {
                            grid.push_str("  XXX  ");
                        } else {
                            let st: usize = self.get_idx_from_coord(mapped_coord.0, mapped_coord.1);
                            let mut m: String = to_print.get(st).unwrap().to_string();
                            if m.len() > 6 {
                                m.truncate(6);
                            }
                            let padded = format!("{:^7}", m);
                            grid.push_str(&padded);
                        }
                    }
                }
            }
            grid.push('\n');
        }
        println!("{}", grid);
    }

}
