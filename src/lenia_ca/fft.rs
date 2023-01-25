#![allow(dead_code)]
#![allow(unused_variables)]

/// Original code from Rust crate: ofuton = "0.1.0"
/// 
/// Copyright (c) 2018 Takafumi Hirata
/// 
/// Distributed under the MIT license (https://opensource.org/licenses/MIT)
/// 
/// Code brought up to date to work with ndarray = "0.15.6" and rustfft = "6.1.0" by Birdbrain
/// 
/// Preplanning functionality in the form of PreplannedFFT and PreplannedFFTND 
/// structs added by Birdbrain. 

use std::{fmt, sync::Arc};
use rustfft::{Fft, FftNum, FftPlanner, FftDirection};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Zero};
use ndarray::{Dimension, Array, Array2};


/// `PreplannedFFT` struct is a container for all the relevant data for a determined length
/// 1D Fast-Fourier-Transform. 
pub struct PreplannedFFT{
    fft: Arc<dyn Fft<f64>>,
    scratch_space: Vec<Complex<f64>>,
    inverse: bool,
    length: usize,
}

impl fmt::Debug for PreplannedFFT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreplannedFFT")
         .field("fft", &"Arc<dyn rustfft::Fft<f64>>")
         .field("scratch_space", &self.scratch_space)
         .field("inverse", &self.inverse)
         .field("length", &self.length)
         .finish()
    }
}

impl PreplannedFFT{
    /// Preplan fast-fourier-transform for arrays with the length of the prototype array.
    /// 
    /// ### Arguments
    /// 
    /// * `prototype` - Array the length of which determines the length of data that the
    /// fft will be able to work on
    /// 
    /// * `inverse` - If `true` then the fft will compute inverse fourier transforms instead.
    /// 
    /// ### Returns
    /// Instance capable of computing fourier transforms on certain length data.
    pub fn preplan_by_prototype(prototype: &[Complex<f64>], inverse: bool) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(
            prototype.len(), 
            if inverse == true {
                FftDirection::Inverse
            } 
            else {
                FftDirection::Forward
            }
        );
        let scratch_space_length = fft.get_outofplace_scratch_len();

        PreplannedFFT { 
            fft: fft, 
            scratch_space: vec![Zero::zero(); scratch_space_length], 
            inverse: inverse, 
            length: prototype.len(),
        }
    }

    /// Preplan fast-fourier-transform for arrays of a certain length.
    /// 
    /// ### Arguments
    /// 
    /// * `length` - The array length that the fft will be able to transform.
    /// 
    /// * `inverse` - If `true` then the fft will compute inverse fourier transforms instead.
    /// 
    /// ### Returns
    /// Instance capable of computing fourier transforms on certain length data.
    pub fn preplan_by_length(length: usize, inverse: bool) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(
            length, 
            if inverse == true {
                FftDirection::Inverse
            } 
            else {
                FftDirection::Forward
            }
        );
        let scratch_space_length = fft.get_outofplace_scratch_len();

        PreplannedFFT { 
            fft: fft, 
            scratch_space: vec![Zero::zero(); scratch_space_length], 
            inverse: inverse, 
            length: length,
        }
    }

    /// Computes the forward- or inverse fourier transform using a fast-fourier-transform algorithm.
    /// 
    /// ### Arguments
    /// 
    /// * `input` - Reference to the array to be fourier transformed. **The input array 
    /// will have junk data in it after the transform!**
    /// 
    /// * `output` - Reference to the array that will contain the transformed data.
    pub fn transform(&mut self, input: &mut [Complex<f64>], output: &mut [Complex<f64>]) {
        self.fft.process_outofplace_with_scratch(input, output, &mut self.scratch_space);
        if self.inverse {
            let inverse_len = 1.0 / input.len() as f64;
            for v in output.iter_mut() {
                v.re *= inverse_len;
                v.im *= inverse_len;
            }
        }
    }

    /// Returns `true` if the fft computes inverse fourier transforms, otherwise outputs `false`.
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Returns the array length that the fft can transform.
    pub fn length(&self) -> usize {
        self.length
    }
}


#[derive(Debug)]
/// `PreplannedFFTND` struct is for preplanning fast-fourier-transforms that work on 
/// n-dimensional arrays.
pub struct PreplannedFFTND {
    inverse: bool,
    shape: Vec<usize>,
    fft_instances: Vec<PreplannedFFT>,
}

impl PreplannedFFTND {
    /// Preplan fast-fourier-transform for arrays with the shape of the prototype array.
    /// 
    /// ### Arguments
    /// 
    /// * `prototype` - Array the shape of which determines the shape of data that the
    /// fft will be able to work on
    /// 
    /// * `inverse` - If `true` then the fft will compute inverse fourier transforms instead.
    /// 
    /// ### Returns
    /// Instance capable of computing fourier transforms on certain shaped data.
    pub fn preplan_by_prototype(prototype: &ndarray::ArrayD<Complex<f64>>, inverse: bool) -> Self {
        let mut shape = Vec::new();
        let mut ffts = Vec::new();
        
        for dim in prototype.shape() {
            shape.push(*dim);
            let dimension_prototype = vec![Zero::zero(); *dim];
            ffts.push(PreplannedFFT::preplan_by_prototype(&dimension_prototype, inverse));
        }

        PreplannedFFTND { 
            inverse: inverse, 
            shape: shape, 
            fft_instances: ffts,
        }
    }

    /// Preplan fast-fourier-transform for arrays of a certain shape.
    /// 
    /// ### Arguments
    /// 
    /// * `shape` - The shape of the array that the fft will be able to transform.
    /// 
    /// * `inverse` - If `true` then the fft will compute inverse fourier transforms instead.
    /// 
    /// ### Returns
    /// Instance capable of computing fourier transforms on certain shaped data.
    pub fn preplan_by_shape(shape: &[usize], inverse: bool) -> Self {
        let mut fft_shape: Vec<usize> = Vec::new();
        let mut ffts = Vec::new();
        
        for dim in shape {
            fft_shape.push(*dim);
            ffts.push(PreplannedFFT::preplan_by_length(*dim, inverse));
        }

        PreplannedFFTND { 
            inverse: inverse, 
            shape: fft_shape,
            fft_instances: ffts,
        }
    }

    /// Computes the forward- or inverse fourier transform using a fast-fourier-transform algorithm.
    /// 
    /// ### Arguments
    /// 
    /// * `input` - Reference to the array to be fourier transformed. **The input array 
    /// will have junk data in it after the transform!**
    /// 
    /// * `output` - Reference to the array that will contain the transformed data.
    pub fn transform<D: Dimension>(&mut self, input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axes: &[usize]) {
        let len = axes.len();
        for i in 0..len {
            let axis = axes[i];
            planned_mutate_lane(&mut self.fft_instances[axis], input, output, transform_lane, axis);
            if i < len - 1 {
                let mut outrows = output.rows_mut().into_iter();
                for mut row in input.rows_mut() {
                    let mut outrow = outrows.next().unwrap();
                    row.as_slice_mut().unwrap().copy_from_slice(outrow.as_slice_mut().unwrap());
                }
            }
        }
    }

    /// Outputs `true` if the fft computes inverse fourier transforms, otherwise outputs `false`.
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Returns the array length that the fft can transform.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

fn transform_lane(fft: &mut PreplannedFFT, input: &mut [Complex<f64>], output: &mut [Complex<f64>]) {
    fft.transform(input, output);
}

fn planned_mutate_lane<T: Zero + Clone, D: Dimension>(fft: &mut PreplannedFFT, input: &mut Array<T, D>, output: &mut Array<T, D>, f: fn(&mut PreplannedFFT, &mut [T], &mut [T]) -> (), axis: usize) {
    if axis > 0 {
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
        {
            let mut outrows = output.rows_mut().into_iter();
            for row in input.rows_mut() {
                let mut outrow = outrows.next().unwrap();
                let mut vec = row.to_vec();
                let mut out = vec![Zero::zero(); outrow.len()];
                f(fft, &mut vec, &mut out);
                for i in 0..outrow.len() {
                    outrow[i] = out[i].clone();
                }
            }
        }
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
    } else {
        let mut outrows = output.rows_mut().into_iter();
        for mut row in input.rows_mut() {
            let mut outrow = outrows.next().unwrap();
            f(fft, &mut row.as_slice_mut().unwrap(), &mut outrow.as_slice_mut().unwrap());
        }
    }
}



fn _fft<T: FftNum>(input: &mut [Complex<T>], output: &mut [Complex<T>], inverse: bool) {
    let mut planner = FftPlanner::new();
    let len = input.len();
    let fft = planner.plan_fft(len, if inverse == true {FftDirection::Inverse} else {FftDirection::Forward});
    let scratch_space_size = fft.get_outofplace_scratch_len();
    let mut scratch_space = vec![Zero::zero(); scratch_space_size];
    fft.process_outofplace_with_scratch(input, output, &mut scratch_space);
}

pub fn fft<T: FftNum>(input: &mut [Complex<T>], output: &mut [Complex<T>]) {
    _fft(input, output, false);
}

pub fn ifft<T: FftNum + From<u32>>(input: &mut [Complex<T>], output: &mut [Complex<T>]) {
    _fft(input, output, true);
    for v in output.iter_mut() {
        *v = v.unscale(T::from(input.len() as u32));
    }
}

pub fn fft2(input: &mut Array2<Complex<f64>>, output: &mut Array2<Complex<f64>>) {
    fftnd(input, output, &[0,1]);
}

pub fn ifft2(input: &mut Array2<Complex<f64>>, output: &mut Array2<Complex<f64>>) {
    ifftnd(input, output, &[1,0]);
}

pub fn fftn<D: Dimension>(input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axis: usize) {
    _fftn(input, output, axis, false);
}

pub fn ifftn<D: Dimension>(input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axis: usize) {
    _fftn(input, output, axis, true);
}

fn _fftn<D: Dimension>(input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axis: usize, inverse: bool) {
    if inverse {
        mutate_lane(input, output, ifft, axis)
    } else {
        mutate_lane(input, output, fft, axis)
    }
}

pub fn fftnd<D: Dimension>(input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axes: &[usize]) {
    _fftnd(input, output, axes, false);
}

pub fn ifftnd<D: Dimension>(input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axes: &[usize]) {
    _fftnd(input, output, axes, true);
}

fn _fftnd<D: Dimension>(input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axes: &[usize], inverse: bool) {
    let len = axes.len();
    for i in 0..len {
        let axis = axes[i];
        _fftn(input, output, axis, inverse);
        if i < len - 1 {
            let mut outrows = output.rows_mut().into_iter();
            for mut row in input.rows_mut() {
                let mut outrow = outrows.next().unwrap();
                row.as_slice_mut().unwrap().copy_from_slice(outrow.as_slice_mut().unwrap());
            }
        }
    }
}

fn mutate_lane<T: Zero + Clone, D: Dimension>(input: &mut Array<T, D>, output: &mut Array<T, D>, f: fn(&mut [T], &mut [T]) -> (), axis: usize) {
    if axis > 0 {
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
        {
            let mut outrows = output.rows_mut().into_iter();
            for row in input.rows_mut() {
                let mut outrow = outrows.next().unwrap();
                let mut vec = row.to_vec();
                let mut out = vec![Zero::zero(); outrow.len()];
                f(&mut vec, &mut out);
                for i in 0..outrow.len() {
                    outrow[i] = out[i].clone();
                }
            }
        }
        input.swap_axes(0, axis);
        output.swap_axes(0, axis);
    } else {
        let mut outrows = output.rows_mut().into_iter();
        for mut row in input.rows_mut() {
            let mut outrow = outrows.next().unwrap();
            f(&mut row.as_slice_mut().unwrap(), &mut outrow.as_slice_mut().unwrap());
        }
    }
}