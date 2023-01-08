// Original code from Rust crate: ofuton = "0.1.0" 
// crate licensed under:
// "https://www.apache.org/licenses/LICENSE-2.0"
// and
// "https://opensource.org/licenses/MIT"
// Code brought up to date to work with ndarray = "0.15.6" and rustfft = "6.1.0" by Birdbrain
// Up to date code retains its former licenses.

#![allow(dead_code)]
use rustfft::{FftNum, FftPlanner, FftDirection};
use rustfft::num_complex::Complex;
use rustfft::num_traits::{Zero};
use ndarray::{Dimension, Array, Array2};

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
            for mut row in input.rows_mut() {
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