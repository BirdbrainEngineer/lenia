#![allow(dead_code)]
#![allow(unused_variables)]

use std::{fmt, sync::Arc};
use rustfft::{Fft, FftNum, FftPlanner, FftDirection};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Zero};
use ndarray::{Dimension, Array, Array2, AssignElem};


pub struct PlannedFFT {
    fft: Arc<dyn Fft<f64>>,
    scratch_space: Vec<Complex<f64>>,
}

impl fmt::Debug for PlannedFFT {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreplannedFFT")
         .field("scratch_space", &format!("Vec<Complex<f64>>, len: {}", self.scratch_space.len()))
         .field("fft", &format!("Arc<dyn rustfft::Fft<f64>> => len: {}, direction: {}", self.fft.len(), self.fft.fft_direction()))
         .finish()
    }
}

impl PlannedFFT {
    pub fn new(length: usize, inverse: bool) -> Self {
        let mut planner = FftPlanner::new();
        let direction: FftDirection;
        match inverse {
            true => { direction = FftDirection::Inverse; }
            false => { direction = FftDirection::Forward; }
        }
        let fft = planner.plan_fft(
            length, 
            direction,
        );
        let scratch_space: Vec<Complex<f64>> = Vec::from_iter(std::iter::repeat(Complex::new(0.0, 0.0)).take(fft.get_inplace_scratch_len()));

        PlannedFFT {
            fft: fft,
            scratch_space: scratch_space,
        }
    }

    pub fn inverse(&self) -> bool {
        match self.fft.fft_direction() {
            FftDirection::Forward => { return false; }
            FftDirection::Inverse => { return true; }
        }
    }

    pub fn length(&self) -> usize {
        self.fft.len()
    }

    pub fn transform(&mut self, data: &mut [Complex<f64>]) {
        self.fft.process_with_scratch(data, &mut self.scratch_space);
        if self.inverse() {     // I fekin' forgot this AGAIN...
            let inverse_len = 1.0 / data.len() as f64;
            for v in data.iter_mut() {
                v.re *= inverse_len;
                v.im *= inverse_len;
            }
        }
    }
}

#[derive(Debug)]
pub struct PlannedFFTND {
    shape: Vec<usize>,
    fft_instances: Vec<PlannedFFT>,
    inverse: bool
}

impl PlannedFFTND {
    pub fn new(shape: &[usize], inverse: bool) -> Self {
        if shape.is_empty() { panic!("PlannedFFTND::new() - Provided shape was empty! Needs at least 1 dimension!"); }
        /*let base_dim = shape[0];
        for dim in shape {
            if *dim != base_dim { panic!("PlannedFFTND::new() - Dimensions not the same length. Differing dimensions not implemented."); }
        }*/
        let mut ffts: Vec<PlannedFFT> = Vec::with_capacity(shape.len());
        for dim in shape {
            ffts.push(PlannedFFT::new(*dim, inverse));
        }
        PlannedFFTND {
            shape: shape.to_vec(),
            fft_instances: ffts,
            inverse: inverse,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn inverse(&self) -> bool {
        self.inverse
    }

    pub fn transform(&mut self, data: &mut ndarray::ArrayD<Complex<f64>>) {
        if data.shape() != self.shape { panic!("PlannedFFTND::transform() - shape of the data to be transformed does not agree with the shape that the fft can work on!"); }
        let mut axis_iterator: Vec<usize> = Vec::with_capacity(self.shape.len());
        if self.inverse() {
            for i in (0..self.shape.len()).rev() {
                axis_iterator.push(i);
            }
        }
        else {
            for i in 0..self.shape.len() {
                axis_iterator.push(i);
            }
        }
        for axis in axis_iterator {
            for mut lane in data.lanes_mut(ndarray::Axis(axis)) {
                let mut buf = lane.to_vec();
                self.fft_instances[axis].transform(&mut buf);
                for i in 0..lane.len() {
                    lane[i] = buf[i];
                }
            }
        }
    }
}
