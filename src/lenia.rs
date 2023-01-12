#![allow(dead_code)]
#![allow(unused_variables)]

use ndarray::{self, Axis, Slice, Order};
use num_complex::Complex;
use crate::fft::{self, PreplannedFFTND};

// Some utility functions and options for implementing default Lenia
pub mod utils {
    // Collection of some common kernel generators
    pub mod kernels {
        use ndarray::IxDyn;

        /// Creates the kernel base for a gaussian donut in 2d. 
        /// The peak is placed at radius/2 distance from center.
        /// stddev of ~1.0/3.35 gives a good kernel for standard Lenia.
        /// This is one of the default kernels for 2d Lenia.
        pub fn gaussian_donut_2d(diameter: usize, stddev: f64) -> ndarray::ArrayD<f64> {
            let radius = diameter as f64 / 2.0;
            let mut out = ndarray::ArrayD::zeros(IxDyn(&[diameter, diameter]));
            let x0 = radius;
            let y0 = radius;
            for i in 0..out.shape()[0] {
                for j in 0..out.shape()[1] {
                    let x1 = i as f64;
                    let y1 = j as f64;
                    let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
                    if dist <= radius { 
                        out[[i, j]] = super::sample_normal(dist, diameter as f64 * 0.25, diameter as f64 * 0.25 * stddev);
                    }
                    else { 
                        out[[i, j]] = 0.0
                    }
                }
            }
            out
        }
        /// Create the base for a kernel with multiple concentric gaussian donuts. 
        /// diameter: The diameter of the whole kernel; The kernel is guaranteed to be scuare in shape.
        /// means: The placement of the peak values of individual rings. Must be in range [0.0..1.0], where 0.0 is the center point of the kernel.
        /// peaks: The maximum value that each individual ring can create. Can be any 
        pub fn multi_gaussian_donut_2d(diameter: usize, means: &[f64], peaks: &[f64], stddevs: &[f64]) -> ndarray::ArrayD<f64> {
            if means.len() != peaks.len() || means.len() != stddevs.len() {
                panic!("Function \"multi_gaussian_donut_2d\" expects each mean parameter to be accompanied by a peak and stddev parameter!");
            }
            let radius = diameter as f64 / 2.0;
            let inverse_radius = 1.0 / radius;
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
                            sum += super::sample_normal(dist * inverse_radius, means[i], stddevs[i]) * peaks[i];
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

        pub fn game_of_life() -> ndarray::ArrayD<f64> {
            let mut out = ndarray::ArrayD::from_elem(vec![3 as usize, 3], 1.0);
            out[[1, 1]] = 0.0;
            out
        }
    }

    pub mod growth_functions {
        // Standard unimodal, gaussian lenia growth function.
        // param[0]: mu (the mean / the highest point of the growth function)
        // param[1]: stddev (standard deviation)
        pub fn standard_lenia(num: f64, params: &[f64]) -> f64 {
            (2.0 * super::sample_normal(num, params[0], params[1])) - 1.0
        }

        pub fn game_of_life(num: f64, params: &[f64]) -> f64 {
            let index = (num * 9.0).round() as usize;
            if index == 2 { 0.0 }
            else if index == 3 { 1.0 }
            else {-1.0 }
        }

        pub fn constant(num: f64, params: &[f64]) -> f64 {
            params[0]
        }
    }

    pub mod seeders {
        use rand::Rng;

        fn generate_random_hypercube(shape: &[usize], scaler: f64, discrete: bool) -> ndarray::ArrayD<f64> {
            let mut generator = rand::thread_rng();
            ndarray::ArrayD::from_shape_simple_fn(shape, 
                || {
                    if discrete { (scaler * generator.gen::<f64>()).round() }
                    else { scaler * generator.gen::<f64>() }
                }
            )
        }

        pub fn random_uniform(shape: &[usize], scaler: f64, discrete: bool) -> ndarray::ArrayD<f64> {
            let mut generator = rand::thread_rng();
            ndarray::ArrayD::from_shape_fn(shape, |_| {
                if discrete { (scaler * generator.gen::<f64>()).round() }
                    else { scaler * generator.gen::<f64>() }
            })
        }

        pub fn random_gaussian_2d(shape: &[usize], stddev: f64, scaler: f64, discrete: bool) -> ndarray::ArrayD<f64> {
            let mut generator = rand::thread_rng();
            if shape.len() != 2 { 
                panic!("Seeder \"random_gaussian_2d\" expects 2 dimensions but was given {}",shape.len()); 
            }
            let mut out = ndarray::ArrayD::from_shape_simple_fn(shape, 
                || {
                    generator.gen::<f64>()
                }
            );
            let x0 = shape[0] as f64 / 2.0;
            let y0 = shape[1] as f64 / 2.0;
            for x in 0..shape[0] {
                for y in 0..shape[1] {
                    let x1 = x as f64;
                    let y1 = y as f64;
                    let dist = ((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)).sqrt();
                    out[[x, y]] = scaler * out[[x, y]] * super::sample_normal(dist, 0.0, stddev);
                    if discrete { out[[x, y]] = out[[x, y]].round(); }
                }
            }
            out
        }

        pub fn random_hypercubic(shape: &[usize], radius: usize, scaler: f64, discrete: bool) -> ndarray::ArrayD<f64> {
            let mut out = ndarray::ArrayD::from_elem(shape, 0.0);
            let mut cube_dims: Vec<usize> = Vec::new();
            for dim in shape {
                if *dim < (radius * 2) { cube_dims.push(*dim); }
                else { cube_dims.push(radius * 2); }
            }
            let cube = generate_random_hypercube(&cube_dims, scaler, discrete);
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
                generate_random_hypercube(&cube_dims, scaler, discrete).assign_to(patch.slice_each_axis_mut(
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
            for i in (0..patches).rev() {
                out.zip_mut_with(&buf[i], 
                    |a, b| { 
                        *a = (*a + *b).clamp(0.0, 1.0);
                        if discrete { *a = a.round(); }
                    }
                )
            }
            out
        }
    }

    // Samples the normal distribution where the peak (at x = mu) is 1.
    // Do not use for gaussian probability density!
    fn sample_normal(x: f64, mu: f64, stddev: f64) -> f64 {
        (-(((x - mu) * (x - mu))/(2.0 * (stddev * stddev)))).exp()
    }
}

pub struct Simulator<L: Lenia> {
    sim: L,
    channel_shape: Vec<usize>,
}

impl<L: Lenia> Simulator<L> {
    pub fn new(channel_shape: Vec<usize>) -> Self {
        Simulator{
            sim: L::new(&channel_shape),
            channel_shape: channel_shape,
        }
    }

    pub fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, convolution_channel: usize) {
        self.sim.set_kernel(kernel, convolution_channel);
    }

    pub fn set_growth_function(&mut self, f: fn(f64, &[f64]) -> f64, growth_parameters: Vec<f64>, convolution_channel: usize) {
        self.sim.set_growth(f, growth_parameters, convolution_channel);
    }

    pub fn set_dt(&mut self, dt: f64) {
        self.sim.set_dt(dt);
    }

    pub fn iterate(&mut self) {
        self.sim.iterate();
    }

    pub fn fill_channel(&mut self, data: &ndarray::ArrayD<f64>, channel: usize) {
        self.sim.set_channel(data, channel);
    }

    pub fn get_frame(&self, channel: usize, display_axes: &[usize; 2], dimensions: &[usize]) -> ndarray::Array2<f64> {
        let data_ref = self.sim.get_data_asref(channel);
        return data_ref.slice_each_axis(
            |a|{
                if a.axis.index() == display_axes[0] || a.axis.index() == display_axes[1] {
                    return Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
                else {
                    return Slice {
                        start: dimensions[a.axis.index()] as isize,
                        end: Some(dimensions[a.axis.index() + 1] as isize),
                        step: 1,
                    }
                }
            }
        ).to_shape(
            ((
                data_ref.shape()[display_axes[0]],
                data_ref.shape()[display_axes[1]]
            ), Order::RowMajor)
        ).unwrap()
        .mapv(|el| { el.re });
    }

    pub fn get_dt(&self) -> f64 {
        self.sim.get_dt()
    }

    pub fn get_channel_shape(&self) -> &[usize] {
        &self.channel_shape
    }
}

pub trait Lenia {
    fn new(shape: &[usize]) -> Self;
    fn initialize_channels(&mut self, num_channels: usize);
    fn add_conv_channels(&mut self, num_conv_channels: usize);
    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize);
    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize);
    fn set_weights(&mut self, new_weights: &[f64], conv_channel: usize);
    fn set_dt(&mut self, new_dt: f64);
    fn set_channel(&mut self, data: &ndarray::ArrayD<f64>, channel: usize);
    fn get_data(&self, channel: usize) -> ndarray::ArrayD<f64>;
    fn get_data_asref(&self, channel: usize) -> &ndarray::ArrayD<Complex<f64>>;
    fn get_dt(&self) -> f64;
    fn iterate(&mut self);
}

pub struct StandardLenia {
    dt: f64,
    channel: Vec<Channel>,
    buffer: Vec<ndarray::ArrayD<Complex<f64>>>,
    conv_channel: Vec<ConvolutionChannel>,
    forward_fft_instances: Vec<fft::PreplannedFFTND>,
    inverse_fft_instances: Vec<fft::PreplannedFFTND>,
}

impl Lenia for StandardLenia {
    fn new(shape: &[usize]) -> Self {
        if shape.len() < 2 || shape.len() > 2 { 
            panic!("Expected 2 dimensions for Standard Lenia! Found {}", shape.len()); 
        }
        let mut proper_dims: Vec<usize> = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if *dim < 25 {
                println!("Dimension {} is extremely small ({} pixels). Resized dimension to 25 pixels.", i, *dim);
                proper_dims.push(25);
            }
            else {
                proper_dims.push(*dim);
            }
        }
        let kernel = Kernel::from(utils::kernels::gaussian_donut_2d(26, 1.0/3.35), &proper_dims);
        let conv_field = ndarray::ArrayD::from_elem(proper_dims, Complex::new(0.0, 0.0));
        let conv_channel = ConvolutionChannel {
            input_channel: 0,
            input_buffer: conv_field.clone(),
            kernel: kernel,
            field: conv_field,
            growth: utils::growth_functions::standard_lenia,
            growth_params: vec![0.15, 0.017],
        };

        let channel = Channel {
            field: conv_channel.field.clone(),
            inverse_weight_sums: vec![1.0],
        };
        
        StandardLenia{
            forward_fft_instances: vec![fft::PreplannedFFTND::preplan(&channel.field, false)],
            inverse_fft_instances: vec![fft::PreplannedFFTND::preplan(&channel.field, true)],
            dt: 0.1,
            channel: vec![channel],
            buffer: vec![conv_channel.field.clone()],
            conv_channel: vec![conv_channel],
        }
    }

    fn iterate(&mut self) {
        self.conv_channel[0].input_buffer.zip_mut_with(&self.channel[0].field, 
            |a, b| {
                a.re = b.re;
                a.im = 0.0;
            }
        );
        self.forward_fft_instances[0].transform(
            &mut self.conv_channel[0].input_buffer, 
            &mut self.buffer[0], 
            &[0, 1]
        );
        self.buffer[0].zip_mut_with(&self.conv_channel[0].kernel.transformed, 
            |a, b| {    // Complex multiplication without cloning
                let ac = a.re * b.re;
                let bd = a.im * b.im;
                let real = ac - bd;
                a.im = ((a.re + a.im) * (b.re + b.im)) - real;
                a.re = real;
            }
        );
        self.inverse_fft_instances[0].transform(
            &mut self.buffer[0], 
            &mut self.conv_channel[0].field, 
            &[1, 0]
        );
        self.channel[0].field.zip_mut_with(&self.conv_channel[0].field, 
            |a, b| {
                a.re = (a.re + ((self.conv_channel[0].growth)(b.re, &self.conv_channel[0].growth_params) * self.dt)).clamp(0.0, 1.0);
                a.im = 0.0;
            }
        );
    }

    fn initialize_channels(&mut self, num_channels: usize) {
        println!("Initializing channels not available for Standard Lenia! Try using Extended Lenia instead.");
    }

    fn add_conv_channels(&mut self, num_conv_channels: usize) {
        println!("Adding convolution channels not available for Standard Lenia! Try using Extended Lenia instead.");
    }

    fn set_weights(&mut self, new_weights: &[f64], conv_channel: usize) {
        println!("Convolution output weights are not available for Standard Lenia! Try usingh Extended Lenia instead.");
    }

    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize) {
        self.conv_channel[0].kernel = Kernel::from(kernel, self.channel[0].field.shape());
    }

    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize) {
        self.conv_channel[0].growth = f;
        self.conv_channel[0].growth_params = growth_params;
    }

    fn set_dt(&mut self, new_dt: f64) {
        self.dt = new_dt;
    }

    fn get_data(&self, channel: usize) -> ndarray::ArrayD<f64> {
        self.channel[0].field.mapv(|a| { a.re })
    }

    fn get_data_asref(&self, channel: usize) -> &ndarray::ArrayD<Complex<f64>> {
        &self.channel[0].field
    }

    fn set_channel(&mut self, data: &ndarray::ArrayD<f64>, channel: usize) {
        self.channel[0].field.zip_mut_with(data, 
            |a, b| {
                a.re = *b;
                a.im = 0.0;
            }
        );
    }

    fn get_dt(&self) -> f64 {
        self.dt
    }
}


struct Channel {
    pub field: ndarray::ArrayD<Complex<f64>>,
    pub inverse_weight_sums: Vec<f64>,
}

struct ConvolutionChannel {
    pub input_channel: usize,
    pub input_buffer: ndarray::ArrayD<Complex<f64>>,
    pub field: ndarray::ArrayD<Complex<f64>>,
    pub kernel: Kernel,
    pub growth: fn(f64, &[f64]) -> f64,
    pub growth_params: Vec<f64>,
}


struct Kernel {
    pub base: ndarray::ArrayD<f64>,
    pub normalized: ndarray::ArrayD<f64>,
    pub transformed: ndarray::ArrayD<Complex<f64>>,
}

impl Kernel {
    // Creates a new Kernel struct from an n-dimensional ndarray array.
    // Creates optimized representations for convolution
    pub fn from(kernel: ndarray::ArrayD<f64>, channel_shape: &[usize]) -> Self {
        let mut normalized_kernel = kernel.clone();
        let mut shifted_and_fft = ndarray::ArrayD::from_elem(channel_shape, Complex::new(0.0, 0.0));
        
        // Check for coherence in dimensionality and that the kernel is not
        // larger than the channel it is used to convolve with.
        if normalized_kernel.shape().len() != shifted_and_fft.shape().len() { 
            panic!("Supplied kernel dimensionality does not match the supplied channel dimensionality!
                \nkernel: {} dimensional vs. channel: {} dimensional",
                normalized_kernel.shape().len(), shifted_and_fft.shape().len()
            ); 
        }
        for (i, dim) in normalized_kernel.shape().iter().enumerate() {
            if *dim > shifted_and_fft.shape()[i] { 
                panic!("Supplied kernel is larger than the channel it acts on in axis {}!", i);
            }
        }

        // Normalize the kernel 
        let scaler = 1.0 / normalized_kernel.sum();
        for elem in &mut normalized_kernel {
            *elem *= scaler;
        }

        // Expand the kernel to match the size of the channel shape
        let mut shifted = ndarray::ArrayD::from_elem(channel_shape, 0.0);

        normalized_kernel.assign_to(shifted.slice_each_axis_mut(
            |a| Slice {
                start: (a.len/2 - normalized_kernel.shape()[a.axis.index()]/2) as isize,
                end: Some((
                    a.len/2
                    + normalized_kernel.shape()[a.axis.index()]/2 
                    + normalized_kernel.shape()[a.axis.index()]%2
                ) as isize),
                step: 1
            }
        ));
        
        // Shift the kernel into the corner
        for (i, axis) in channel_shape.iter().enumerate() {
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
        let mut scratch_space = shifted.mapv(|elem| {Complex::new(elem, 0.0)});
        let mut axes: Vec<usize> = Vec::new();
        for (i, _) in shifted_and_fft.shape().iter().enumerate() {
            axes.push(i);
        }
        fft::fftnd(&mut scratch_space, &mut shifted_and_fft, &axes);

        // Create the kernel
        Kernel{
            base: kernel,
            normalized: normalized_kernel,
            transformed: shifted_and_fft,
        }
    }
}



