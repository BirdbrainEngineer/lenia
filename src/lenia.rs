use ndarray::{self, IxDyn, Axis, s, Slice};
use num_complex::Complex;
use crate::fft;

//Some utility functions and options for implementing default Lenia
pub mod utils {
    //Collection of some common kernel generators
    pub mod kernels {
        use probability::distribution::{Gaussian, Continuous};
        use ndarray::IxDyn;

        //functions
        //Creates the kernel base for a gaussian donut in 2d. The peak is placed at radius/2 distance from center. This is one of the default kernels for 2d Lenia.
        pub fn gaussian_donut_2d(diameter: usize) -> ndarray::ArrayD<f64> {
            let radius = diameter as f64 / 2.0;
            let distribution = Gaussian::new(diameter as f64 / 4.0, diameter as f64 / 12.0);
            let scaler = 1.0 / distribution.density(diameter as f64 / 4.0);
            let mut out = ndarray::ArrayD::zeros(IxDyn(&[diameter, diameter]));
            let x0 = radius;
            let y0 = radius;
            for i in 0..out.shape()[0] {
                for j in 0..out.shape()[1] {
                    let x1 = i as f64;
                    let y1 = j as f64;
                    let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
                    if dist <= radius { 
                        out[[i, j]] = distribution.density(dist) * scaler; 
                    }
                    else { 
                        out[[i, j]] = 0.0
                    }
                }
            }
            out
        }
    }

    pub mod growth_functions {
        pub fn standard_lenia(num: f64) -> f64 {
            1.0 // pass implementation
        }

        pub fn standard_lenia_precomputed(num: f64) -> f64 {
            1.0 //pass implementation
        }

        pub fn game_of_life(num: f64) -> f64 {
            if num < 0.222222222 || num >= 0.444444444 { return -1.0 }
            else if num < 0.333333333 { return 0.0 }
            else { return 1.0 }
        }
    }
}

pub struct Simulator<L: Lenia> {
    sim: L,
    pub kernel: Kernel,
}

impl<L: Lenia> Simulator<L> {
    pub fn new(simulation_type: L) -> Self where L: Lenia {
        Simulator{
            sim: simulation_type,
            kernel: Kernel::new(utils::kernels::gaussian_donut_2d(100), &[100, 100]),
        }
    }

    pub fn iterate(&self) {
        self.sim.iterate();
    }
}

pub trait Lenia {
    fn add_channels(&self) {}
    fn add_affine_channel(&self) {}
    fn set_kernel(&self);
    fn set_growth(&self);
    fn set_weights(&self);
    fn set_dt(&self);
    fn iterate(&self);
}

pub struct StandardLenia {

}

impl StandardLenia {
    pub fn new() -> Self {
        StandardLenia{}
    }
}

impl Lenia for StandardLenia {
    fn iterate(&self) {
        
    }

    fn set_kernel(&self) {

    }

    fn set_growth(&self) {
        
    }

    fn set_weights(&self) {
        
    }

    fn set_dt(&self) {
        
    }
}


struct Channel {
    field: ndarray::ArrayD<Complex<f64>>,
}

struct ConvolutionChannel {
    field: ndarray::ArrayD<Complex<f64>>,
    kernel: Kernel,
    growth: fn(f64) -> f64,
}


pub struct Kernel {
    pub base: ndarray::ArrayD<f64>,
    normalized: ndarray::ArrayD<f64>,
    pub shifted: ndarray::ArrayD<f64>,
    transformed: ndarray::ArrayD<Complex<f64>>,
}

impl Kernel {
    // Creates a new Kernel struct from an n-dimensional ndarray array.
    // Creates optimized representations for convoluton during simulation.
    pub fn new(kernel: ndarray::ArrayD<f64>, channel_shape: &[usize]) -> Self {
        let mut normalized_kernel = kernel.clone();
        let mut shifted_and_fft = ndarray::ArrayD::from_elem(channel_shape, Complex::new(0.0, 0.0));
        
        // Check for coherence in dimensionality and that the kernel is not
        // larger than the channel it is used to convolute with.
        if normalized_kernel.shape().len() != shifted_and_fft.shape().len() { 
            panic!{"Supplied kernel dimensionality does not match the supplied channel dimensionality!"}; 
        }
        for (i, dim) in normalized_kernel.shape().iter().enumerate() {
            if *dim > shifted_and_fft.shape()[i] { 
                panic!{"Supplied kernel is larger than the channel it acts on!"} 
            }
        }

        // Normalize the kernel 
        let scaler = 1.0 / normalized_kernel.sum();
        for elem in &mut normalized_kernel {
            *elem *= scaler;
        }

        // Create a shifted kernel
        //let mut shifted_buffer = ndarray::ArrayD::from_elem(channel_shape, 0.0);
        let mut shifted = normalized_kernel.clone();
        for (i, axis) in normalized_kernel.shape().iter().enumerate() {
            let mut shifted_buffer = shifted.clone();
            shifted.slice_axis(
                    Axis(i), 
                    Slice{
                        start: -(*axis as isize / 2), 
                        end: None, 
                        step: 1,
                    }
                )
                .assign_to(shifted_buffer.slice_axis_mut(
                    Axis(i),
                    Slice { 
                        start: 0, 
                        end: Some(*axis as isize / 2), 
                        step: 1,
                    }
                )
            );
            shifted.slice_axis(
                    Axis(i), 
                    Slice{
                        start: 0, 
                        end: Some(*axis as isize / 2), 
                        step: 1,
                    }
                )
                .assign_to(shifted_buffer.slice_axis_mut(
                    Axis(i),
                    Slice { 
                        start: -(*axis as isize / 2), 
                        end: None, 
                        step: 1,
                    }
                )
            );
            shifted = shifted_buffer;
        }

        // Create the discrete-fourier-transformed representation of the kernel for fft-convolving. 
        let mut scratch_space = shifted_and_fft.clone();

        let mut axes: Vec<usize> = Vec::new();
        for (i, _) in shifted_and_fft.shape().iter().enumerate() {
            axes.push(i);
        }
        fft::fftnd(&mut scratch_space, &mut shifted_and_fft, &axes);

        Kernel{
            base: kernel,
            normalized: normalized_kernel,
            shifted: shifted,
            transformed: shifted_and_fft,
        }
    }
}



