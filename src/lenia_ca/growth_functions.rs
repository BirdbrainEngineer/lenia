#![allow(dead_code)]
#![allow(unused_variables)]

/// A collection of common growth functions.

/// Standard unimodal, "gaussian bump" lenia growth function.
/// 
/// ### Parameters
/// * `num` - The input to evaluate against the growth function.
/// 
/// * `params[0]` - **mu**: The position of the mean / highest point of the growth function.
/// 
/// * `params[1]` - **sigma**: Standard deviation of the gaussian bump. 
/// 
/// ### Returns
/// A `f64` in range `[-1.0..1,0]`. 
pub fn standard_lenia(num: f64, params: &[f64]) -> f64 {
    (2.0 * super::sample_normal(num, params[0], params[1])) - 1.0
}

/// Standard unimodal, "gaussian bump" lenia growth function but inverted.
/// 
/// ### Parameters
/// * `num` - The input to evaluate against the growth function.
/// 
/// * `params[0]` - **mu**: The position of the mean / lowest point of the growth function.
/// 
/// * `params[1]` - **sigma**: Standard deviation of the gaussian bump. 
/// 
/// ### Returns
/// A `f64` in range `[-1.0..1,0]`. 
pub fn standard_lenia_inverted(num: f64, params: &[f64]) -> f64 {
    (2.0 * -super::sample_normal(num, params[0], params[1])) + 1.0
}

/// Multimodal "gaussian bumps" growth function. 
/// 
/// ### Parameters
/// * `num` - The input to evaluate against the growth function.
/// 
/// * `params[even number]` - **mu**: The position of the means / the centers of the gaussian bumps.
/// 
/// * `params[odd number]` - **sigma**: Standard deviations of the gaussian bumps. Each sigma corresponds
/// to the mu defined by the previous `params` index.
pub fn multimodal_normal(num: f64, params: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in (0..params.len()).step_by(2) {
        sum += super::sample_normal(num, params[i], params[i + 1]);
    }
    (sum * 2.0) - 1.0
}

/// Multimodal "gaussian bumps" growth function but inverted.
/// 
/// ### Parameters
/// * `num` - The input to evaluate against the growth function.
/// 
/// * `params[even number]` - **mu**: The position of the means / the centers of the gaussian bumps.
/// 
/// * `params[odd number]` - **sigma**: Standard deviations of the gaussian bumps. Each sigma corresponds
/// to the mu defined by the previous `params` index.
pub fn multimodal_normal_inverted(num: f64, params: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in (0..params.len()).step_by(2) {
        sum += super::sample_normal(num, params[i], params[i + 1]);
    }
    (-sum * 2.0) + 1.0
}

/// Samples from a precalculated distribution. The distribution is made of evenly spaced points from
/// `0.0` to `1.0`. In the likely event of the sample falling between 2 points in the distribution, 
/// the result will be interpolated linearly between the two points.
/// 
/// ### Parameters
/// * `num` - The input to evaluate against the growth function.
/// 
/// * `params[0..n]` - Distribution to sample from
pub fn precalculated_linear(num: f64, params: &[f64]) -> f64 {
    let index = num * params.len() as f64;
    if index.abs() as usize >= (params.len() - 1) { return params[params.len() - 1] }
    let a = params[index.abs().floor() as usize];
    let b = params[index.abs().ceil() as usize];
    let dx = index - index.floor();
    let dy = b - a;
    a + (dx * dy)
}

pub fn precalculated_linear_fullrange(num: f64, params: &[f64]) -> f64 {
    let index = ((num + 1.0) * params.len() as f64) * 0.5;
    if index.abs() as usize >= (params.len() - 1) { return params[params.len() - 1] }
    let a = params[index.abs().floor() as usize];
    let b = params[index.abs().ceil() as usize];
    let dx = index - index.floor();
    let dy = b - a;
    a + (dx * dy)
}

/// Conway's "Game of life" growth function. `Rulestring: B3/S23`
pub fn conway_game_of_life(num: f64, params: &[f64]) -> f64 {
    let index = (num * 9.0).round() as usize;
    if index == 2 { 0.0 }
    else if index == 3 { 1.0 }
    else {-1.0 }
}

/// Returns `num`. Use this growth function if you would like to not use a growth function, 
/// but merely explore the dynamics of iterative application of kernels. `num` gets multiplied by `params[0]`
pub fn pass(num: f64, params: &[f64]) -> f64 {
    num * params[0]
}

pub struct Distributions {

}

impl Distributions {
    pub fn geometric_normals(
        mu0: f64, 
        sigma0: f64, 
        peak0: f64, 
        ratio_mu: f64, 
        ratio_sigma: f64, 
        ratio_peak: f64, 
        num_peaks: usize, 
        distribution_length: usize) -> Vec<f64> 
    {
        let mut distribution = vec![0.0; distribution_length];
        let delta = 1.0 / distribution_length as f64;
        let mut mu = mu0;
        let mut sigma = sigma0;
        let mut peak = peak0;
        for _ in 0..(num_peaks - 1) {
            for i in 0..distribution_length {
                distribution[i] += peak * super::sample_normal(i as f64 * delta, mu, sigma);
            }
            mu *= ratio_mu;
            sigma *= ratio_sigma;
            peak *= ratio_peak;
        }
        for i in 0..distribution_length {
            distribution[i] += peak * super::sample_normal(i as f64 * delta, mu, sigma);
            distribution[i] = (distribution[i] * 2.0) - 1.0;
        }
        distribution
    }

    pub fn multi_gaussian(
        means: &[f64],
        sigmas: &[f64],
        peaks: &[f64],
        distribution_length: usize ) -> Vec<f64> 
    {
        let mut distribution = vec![0.0; distribution_length];
        for i in 0..distribution_length {
            let mut sum = 0.0;
            let num = i as f64 * (1.0 / distribution_length as f64);
            for j in 0..means.len() {
                sum += peaks[j] * super::sample_normal(num, means[j], sigmas[j]);
            }
            distribution[i] = sum * 2.0;
            if distribution[i] >= 0.0 { distribution[i] -= 1.0; }
            else { distribution[i] += 1.0; }
        }
        distribution
    }
}