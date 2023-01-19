#![allow(dead_code)]
#![allow(unused_variables)]

use std::ops::IndexMut;

/// Collection of generators for some of the common Lenia kernels. 

use super::*;
use ndarray::IxDyn;

/// Creates the kernel base for a gaussian donut in 2d. 
/// The mean (position of the highest value) is placed at `0.5`
/// in the range `[0.0..1.0]`, where `0.0` is the center of the kernel and `1.0` the outer edge.
/// 
/// 
/// ### Arguments
/// 
/// * `diameter` - The diameter of the whole kernel. The kernel is guaranteed to be square in shape,
/// but any values outside the radius (`diameter / 2`) are set to `0.0`.
/// 
/// * `stddev` - Standard deviation to use. 
/// 
/// ### Returns
/// A 2d array (`ndarray::ArrayD`) of `f64` values.
pub fn gaussian_donut_2d(diameter: usize, stddev: f64) -> ndarray::ArrayD<f64> {
    let radius = diameter as f64 / 2.0;
    let normalizer = 1.0 / radius;
    let mut out = ndarray::ArrayD::zeros(IxDyn(&[diameter, diameter]));
    let x0 = radius;
    let y0 = radius;
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            let x1 = i as f64;
            let y1 = j as f64;
            let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
            if dist <= radius { 
                out[[i, j]] = super::sample_normal(dist * normalizer, 0.5, stddev);
            }
            else { 
                out[[i, j]] = 0.0
            }
        }
    }
    out
}

/// Create the base for a kernel made from multiple concentric gaussian "donuts" in 2d.
/// Each donut/ring is a single index in the list of parameters. 
/// 
/// ### Arguments
/// 
/// * `diameter` - The diameter of the whole kernel. The kernel is guaranteed to be square in shape.
/// but any values outside the radius (`diameter / 2`) are set to `0.0`.
/// 
/// * `means` - The placement of the peak values of individual rings. 
/// Should be in range `[0.0..1.0]`, where `0.0` is the center point of the kernel and
/// `1.0` is the outer edge of the circular kernel. 
/// 
/// * `peaks` - The maximum value that each individual ring can create. 
/// Can be any positive real number but will later be normalized compared to other rings.
/// 
/// * `stddevs` - The standard deviations of each individual ring.
/// 
/// ### Returns
/// 2d array (`ndarray::ArrayD`) of `f64` values.
pub fn multi_gaussian_donut_2d(diameter: usize, means: &[f64], peaks: &[f64], stddevs: &[f64]) -> ndarray::ArrayD<f64> {
    if means.len() != peaks.len() || means.len() != stddevs.len() {
        panic!("Function \"multi_gaussian_donut_2d\" expects each mean parameter to be accompanied by a peak and stddev parameter!");
    }
    let radius = diameter as f64 / 2.0;
    let normalizer = 1.0 / radius;
    let mut out = ndarray::ArrayD::zeros(IxDyn(&[diameter, diameter]));
    let x0 = radius;
    let y0 = radius;
    for i in 0..out.shape()[0] {
        for j in 0..out.shape()[1] {
            let x1 = i as f64;
            let y1 = j as f64;
            let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
            if dist <= radius { 
                let mut sum = 0.0;
                for i in 0..means.len() {
                    sum += super::sample_normal(dist * normalizer, means[i], stddevs[i]) * peaks[i].abs();
                }
                out[[i, j]] = sum;
            }
            else { 
                out[[i, j]] = 0.0
            }
        }
    }
    out
}

/// Creates the kernel base for a gaussian donut in n-dimensions. 
/// The mean (position of the highest value) is placed at `0.5`
/// in the range `[0.0..1.0]`, where `0.0` is the center of the kernel and `1.0` the outer edge.
/// 
/// ### Arguments
/// 
/// * `diameter` - The diameter of the kernel in every axis. 
/// Any values outside the radius (`diameter / 2`) are set to `0.0`.
/// 
/// * `stddev` - Standard deviation to use. 
/// 
/// ### Returns
/// An n-dimensional array (`ndarray::ArrayD`) of `f64` values.
pub fn gaussian_donut_nd(diameter: usize, dimensions: usize, stddev: f64) -> ndarray::ArrayD<f64> {
    let radius = diameter as f64 / 2.0;
    let normalizer = 1.0 / radius;
    let center = vec![radius; dimensions];
    let mut shape: Vec<usize> = Vec::new();
    let mut index: Vec<f64> = Vec::new();
    for i in 0..dimensions {
        shape.push(diameter);
        index.push(0.0);
    }
    let out = ndarray::ArrayD::from_shape_fn(
        shape, 
        |index_info| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&index, &center);
            if dist > radius {
                0.0
            }
            else {
                sample_normal(dist * normalizer, 0.5, stddev)
            }
        }
    );
    out
}

/// Create the base for a kernel made from multiple radial gaussian "hyper-donuts" in n dimensions.
/// Each donut/ring is a single index in the list of parameters. 
/// 
/// ### Arguments
/// 
/// * `diameter` - The diameter of the whole kernel in each axis.
/// Any values outside the radius (`diameter / 2`) are set to `0.0`.
/// 
/// * `means` - The placement of the peak values of individual donuts
/// Should be in range `[0.0..1.0]`, where `0.0` is the center point of the kernel and
/// `1.0` is the outer surface of the hypersphere. 
/// 
/// * `peaks` - The maximum value that each individual donut can create. 
/// Can be any positive real number but will later be normalized compared to other donuts.
/// 
/// * `stddevs` - The standard deviations of each individual donut.
/// 
/// ### Returns
/// An n-dimensional array (`ndarray::ArrayD`) of `f64` values.
pub fn multi_gaussian_donut_nd(diameter: usize, dimensions: usize, means: &[f64], peaks: &[f64], stddevs: &[f64]) -> ndarray::ArrayD<f64> {
    let radius = diameter as f64 / 2.0;
    let normalizer = 1.0 / radius;
    let center = vec![radius; dimensions];
    let mut shape: Vec<usize> = Vec::new();
    let mut index: Vec<f64> = Vec::new();
    for i in 0..dimensions {
        shape.push(diameter);
        index.push(0.0);
    }
    let out = ndarray::ArrayD::from_shape_fn(
        shape, 
        |index_info| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&index, &center);
            if dist > radius {
                0.0
            }
            else {
                let mut sum = 0.0;
                for i in 0..means.len() {
                    sum += super::sample_normal(dist * normalizer, means[i], stddevs[i]) * peaks[i].abs();
                }
                sum
            }
        }
    );
    out
}

/// Moore neighborhood with radius of 1. 
/// 
/// This is the kernel to use for Conway's game of life. 
pub fn moore1() -> ndarray::ArrayD<f64> {
    let mut out = ndarray::ArrayD::from_elem(vec![3 as usize, 3], 1.0);
    out[[1, 1]] = 0.0;
    out
}

/// Gives a kernel the size of 1 unit containing `0.0`, but with as many dimensions as `shape`.
pub fn empty(shape: &[usize]) -> ndarray::ArrayD<f64> {
    let unit_shape: Vec<usize> = Vec::new();
    for _ in shape {
        unit_shape.push(1);
    }
    ndarray::ArrayD::<f64>::zeros(unit_shape)
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut out: f64 = 0.0;
    for i in 0..a.len() {
        out += (a[i] - b[i]) * (a[i] - b[i]);
    }
    out.sqrt()
}