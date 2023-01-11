// Original code from Rust crate: ofuton = "0.1.0"
// Copyright (c) 2018 Takafumi Hirata
// Distributed under the MIT license (https://opensource.org/licenses/MIT)
// Code brought up to date to work with ndarray = "0.15.6" and rustfft = "6.1.0" by Birdbrain
// Preplanning functionality in the form of PreplannedFFT and PreplannedFFTND structs added by Birdbrain. 

#![allow(dead_code)]
use std::sync::Arc;
use rustfft::{Fft, FftNum, FftPlanner, FftDirection};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Zero};
use ndarray::{Dimension, Array, Array2};

pub struct PreplannedFFT{
    fft: Arc<dyn Fft<f64>>,
    scratch_space: Vec<Complex<f64>>,
    inverse: bool,
    length: usize,
}

impl PreplannedFFT{
    pub fn preplan(prototype: &[Complex<f64>], inverse: bool) -> Self {
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

    pub fn transform(&mut self,input: &mut [Complex<f64>], output: &mut [Complex<f64>]) {
        self.fft.process_outofplace_with_scratch(input, output, &mut self.scratch_space);
    }

    pub fn inverse(&self) -> bool {
        self.inverse
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

pub struct PreplannedFFTND {
    inverse: bool,
    shape: Vec<usize>,
    fft_instances: Vec<PreplannedFFT>,
}

impl PreplannedFFTND {
    pub fn preplan(prototype: &ndarray::ArrayD<Complex<f64>>, inverse: bool) -> Self {
        let mut shape = Vec::new();
        let mut ffts = Vec::new();
        
        for dim in prototype.shape() {
            shape.push(*dim);
            let dimension_prototype = vec![Zero::zero(); *dim];
            ffts.push(PreplannedFFT::preplan(&dimension_prototype, inverse));
        }

        PreplannedFFTND { 
            inverse: inverse, 
            shape: shape, 
            fft_instances: ffts,
        }
    }

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

    pub fn transform_single_dim<D: Dimension>(&mut self, input: &mut Array<Complex<f64>, D>, output: &mut Array<Complex<f64>, D>, axis: usize) {
        planned_mutate_lane(&mut self.fft_instances[axis], input, output, transform_lane, axis);
    }

    pub fn inverse(&self) -> bool {
        self.inverse
    }

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
                    outrow[i] = out.remove(0);
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
                    outrow[i] = out.remove(0);
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