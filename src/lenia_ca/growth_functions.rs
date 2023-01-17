#![allow(dead_code)]
#![allow(unused_variables)]

/// A collection of common growth functions.

use super::*;

/// Standard unimodal, "gaussian bump" lenia growth function.
/// 
/// ### Parameters
/// * `num` - The input to evaluate against the growth function.
/// 
/// * `param[0]` - **mu**: The position of the mean / highest point of the growth function.
/// 
/// * `param[1]` - **sigma**: Standard deviation of the gaussian bump. 
/// 
/// ### Returns
/// A `f64` in range `[-1.0..1,0]`. 
pub fn standard_lenia(num: f64, params: &[f64]) -> f64 {
    (2.0 * super::sample_normal(num, params[0], params[1])) - 1.0
}

/// Conway's "Game of life" growth function. `Rulestring: B3/S23`
pub fn conway_game_of_life(num: f64, params: &[f64]) -> f64 {
    let index = (num * 9.0).round() as usize;
    if index == 2 { 0.0 }
    else if index == 3 { 1.0 }
    else {-1.0 }
}

/// Returns `num`. Use this growth function if you would like to not use a growth function, 
/// but merely explore the dynamics of iterative application of kernels. 
pub fn pass(num: f64, params: &[f64]) -> f64 {
    num
}
