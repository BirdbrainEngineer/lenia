#![allow(dead_code)]
#![allow(unused_variables)]

/// A collection of initial randomness generators for the simulator.

use super::*;
use rand::{Rng, random};
use rayon::prelude::IntoParallelRefMutIterator;

/// Generates an n-dimensional field with the shape `shape`, filled with `constant`.
pub fn constant(shape: &[usize], constant: f64) -> ndarray::ArrayD<f64> {
    ndarray::ArrayD::from_shape_fn(shape,|_| { constant })
}

/// Generates an n-dimensional field with uniformly random values.
/// 
/// ### Parameters
/// 
/// * `shape` - Shape of the field to generate and fill.
/// 
/// * `scaler` - Random values are generated in the range `[0.0..scaler]` if `discrete` is `false`.
/// If `discrete` is `true` then controls the probability of generating a `1.0` as opposed to `0.0`.
/// 
/// * `discrete` - If `true` then generates only values `0.0` and `1.0`.
pub fn random_uniform(shape: &[usize], scaler: f64, discrete: bool) -> ndarray::ArrayD<f64> {
    let mut generator = rand::thread_rng();
    ndarray::ArrayD::from_shape_fn(shape, |_| {
        if discrete { 
            ((scaler + generator.gen::<f64>()).floor()).clamp(0.0, 1.0) 
        }
        else { 
            scaler * generator.gen::<f64>() 
        }
    })
}

/// Generates an n-dimensional field with uniformly random values in an n-dimensional hypercube
/// placed at the center of the field.
/// 
/// ### Parameters
/// 
/// * `shape` - Shape of the field to generate.
/// 
/// * `radius` - Radius (the apothem) of the hypercube to fill with random values.
/// 
/// * `scaler` - Random values are generated in the range `[0.0..scaler]` if `discrete` is `false`.
/// If `discrete` is `true` then controls the probability of generating a `1.0` as opposed to `0.0`.
/// 
/// * `discrete` - If `true` then generates only values `0.0` and `1.0`.
pub fn random_hypercubic(shape: &[usize], radius: usize, scaler: f64, discrete: bool) -> ndarray::ArrayD<f64> {
    let mut out = ndarray::ArrayD::from_elem(shape, 0.0);
    let mut cube_dims: Vec<usize> = Vec::new();
    for dim in shape {
        if *dim < (radius * 2) { cube_dims.push(*dim); }
        else { cube_dims.push(radius * 2); }
    }
    let cube = random_uniform(&cube_dims, scaler, discrete);
    cube.assign_to(out.slice_each_axis_mut(
        |a| ndarray::Slice {
            start: (a.len/2 - cube.shape()[a.axis.index()]/2) as isize,
            end: Some((
                a.len/2
                + cube.shape()[a.axis.index()]/2 
            ) as isize),
            step: 1
        }
    ));
    out
}

/// Generates an n-dimensional field with multiple n-dimensional hypercubes, filled with 
/// uniformly random values. Individual hypercubes are placed at random locations in the field but not
/// overlapping an edge of the field. 
/// 
/// Overlapping hypercubes will add together in the final result.
/// 
/// ### Parameters
/// 
/// * `shape` - Shape of the field to generate.
/// 
/// * `radius` - Radius (the apothem) of the hypercubes to fill with random values.
/// 
/// * `patches` - The number of hypercubes filled with random values to generate.
/// 
/// * `scaler` - Random values are generated in the range `[0.0..scaler]` if `discrete` is `false`.
/// If `discrete` is `true` then controls the probability of generating a `1.0` as opposed to `0.0`.
/// 
/// * `discrete` - If `true` then generates only values `0.0` and `1.0`.
pub fn random_hypercubic_patches(shape: &[usize], radius: usize, patches:usize, scaler:f64, discrete: bool) -> ndarray::ArrayD<f64> {
    let mut generator = rand::thread_rng();
    let mut buf: Vec<ndarray::ArrayD<f64>> = Vec::new();
    let mut cube_dims: Vec<usize> = Vec:: new();
    for dim in shape {
        if *dim < (radius * 2) {
            cube_dims.push(*dim - 2);
        }
        else {
            cube_dims.push(radius);
        }
    }
    for _ in 0..patches {
        let mut patch = ndarray::ArrayD::from_elem(shape, 0.0);
        let randomness;
        randomness = random_uniform(&cube_dims, scaler, discrete);
        
        randomness.assign_to(patch.slice_each_axis_mut(
            |a| {
                let randindex = generator.gen_range(0..(a.len - cube_dims[a.axis.index()]));
                ndarray::Slice {
                    start: randindex as isize,
                    end: Some((randindex + cube_dims[a.axis.index()]) as isize),
                    step: 1,
                }
            }
        ));
        buf.push(patch);
    }
    let mut out = ndarray::ArrayD::from_elem(shape, 0.0);
    for i in 0..patches {
        out.zip_mut_with(&buf[i], 
            |a, b| { 
                *a = (*a + *b).clamp(0.0, 1.0);
            }
        )
    }
    out
}

/// Generates an n-dimensional field with multiple n-dimensional hyperspheres, filled with 
/// uniformly random values. Individual hyperspheres are placed at random locations in the field, 
/// but not overlapping an edge of the field.
/// 
/// Overlapping hypercubes will add together in the final result.
/// 
/// ### Parameters
/// 
/// * `shape` - Shape of the field to generate.
/// 
/// * `radius` - Radius of the hyperspheres to fill with random values.
/// 
/// * `patches` - The number of hyperspheres filled with random values to generate.
/// 
/// * `scaler` - Random values are generated in the range `[0.0..scaler]` if `discrete` is `false`.
/// If `discrete` is `true` then controls the probability of generating a `1.0` as opposed to `0.0`.
/// 
/// * `discrete` - If `true` then generates only values `0.0` and `1.0`.
pub fn random_hyperspheres(shape: &[usize], radius: usize, spheres:usize, scaler:f64, discrete: bool) -> ndarray::ArrayD<f64> {
    let mut generator = rand::thread_rng();
    let mut buf: Vec<ndarray::ArrayD<f64>> = Vec::new();
    let mut cube_dims: Vec<usize> = Vec:: new();
    for dim in shape {
        if *dim < (radius * 2) {
            cube_dims.push(*dim - 2);
        }
        else {
            cube_dims.push(radius);
        }
    }
    for _ in 0..spheres {
        let mut patch = ndarray::ArrayD::from_elem(shape, 0.0);
        let mut randomness;
        randomness = random_uniform(&cube_dims, scaler, discrete);

        let center = vec![randomness.shape()[0] as f64 / 2.0; randomness.shape().len()];
        let mut index: Vec<f64> = vec![0.0; randomness.shape().len()];
        randomness.indexed_iter_mut().for_each(|(index_info, a)| {
            for i in 0..index.len() {
                index[i] = index_info[i] as f64;
            }
            let dist = euclidean_dist(&center, &index);
            if dist > center[0] { *a = 0.0; }
            else { *a = *a; }
        });
        
        randomness.assign_to(patch.slice_each_axis_mut(
            |a| {
                let randindex = generator.gen_range(0..(a.len - cube_dims[a.axis.index()]));
                ndarray::Slice {
                    start: randindex as isize,
                    end: Some((randindex + cube_dims[a.axis.index()]) as isize),
                    step: 1,
                }
            }
        ));
        buf.push(patch);
    }
    let mut out = ndarray::ArrayD::from_elem(shape, 0.0);
    for i in 0..spheres {
        out.zip_mut_with(&buf[i], 
            |a, b| { 
                *a = (*a + *b).clamp(0.0, 1.0);
            }
        )
    }
    out
}

/// Euclidean distance between points `a` and `b`. 
fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut out: f64 = 0.0;
    for i in 0..a.len() {
        out += (a[i] - b[i]) * (a[i] - b[i]);
    }
    out.sqrt()
}
