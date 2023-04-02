#![allow(dead_code)]
#![allow(unused_variables)]

/// A collection of initial randomness generators for the simulator.

use super::*;
use rand::Rng;

/// Generates an n-dimensional field filled with `constant`.
pub fn constant(shape: &[usize], constant: f64) -> ndarray::ArrayD<f64> {
    ndarray::ArrayD::from_shape_fn(shape,|_| { constant })
}

/// Generates an n-dimensional field with uniformly random values.
/// 
/// ### Arguments
/// 
/// * `shape` - Shape of the field to generate and fill.
/// 
/// * `scaler` - Random values are generated in the range `[0.0..scaler]` if `discrete` is `false`.
/// If `discrete` is `true` then controls the probability of generating a `1.0` as opposed to `0.0`.
/// 
/// * `discrete` - If `true` then generates only values `0.0` and `1.0`.
/// 
/// ### Returns
/// n-dimensional array (`ndarray::ArrayD`) of `f64` values, with the shape defined by `shape` parameter.
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
/// ### Arguments
/// 
/// * `shape` - Shape of the field to generate.
/// 
/// * `radius` - Radius (the apothem) of the hypercube to fill with random values.
/// 
/// * `scaler` - Random values are generated in the range `[0.0..scaler]` if `discrete` is `false`.
/// If `discrete` is `true` then controls the probability of generating a `1.0` as opposed to `0.0`.
/// 
/// * `discrete` - If `true` then generates only values `0.0` and `1.0`.
/// 
/// ### Returns
/// n-dimensional array (`ndarray::ArrayD`) of `f64` values, with the shape defined by `shape` parameter.
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
/// uniformly random values, placed at random locations in the field. 
/// 
/// Overlapping hypercubes will add together in the final result.
/// 
/// ### Arguments
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
/// 
/// ### Returns
/// n-dimensional array (`ndarray::ArrayD`) of `f64` values, with the shape defined by `shape` parameter.
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
        random_uniform(&cube_dims, scaler, discrete).assign_to(patch.slice_each_axis_mut(
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
