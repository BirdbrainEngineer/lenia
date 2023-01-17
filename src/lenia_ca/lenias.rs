use ndarray;
use num_complex::Complex;
use super::*;

/// `StandardLenia2D` struct implements the standard Lenia system with a 2d field and 
/// pre-set parameters to facilitate the creation of the
/// ***Orbium unicaudatus*** glider - hallmark of the Lenia system.
/// 
/// This version of Lenia does not allow for adding extra channels nor convolution channels. 
/// In addition, channel weights are not available for this version of Lenia.
/// 
/// Changeable parameters include the timestep a.k.a. integration step **dt**, 
/// the **growth function**, and the **kernel** given that the kernel is 2-dimensional. 
pub struct StandardLenia2D {
    dt: f64,
    channel: Vec<Channel>,
    shape: Vec<usize>,
    convolution_buffer: Vec<ndarray::ArrayD<Complex<f64>>>,
    conv_channel: Vec<ConvolutionChannel>,
    forward_fft_instances: Vec<fft::PreplannedFFTND>,
    inverse_fft_instances: Vec<fft::PreplannedFFTND>,
}

impl StandardLenia2D {
    const STANDARD_LENIA_DEFAULT_KERNEL_SIZE: usize = 28;
}

impl Lenia for StandardLenia2D {
    /// Create and initialize a new instance of "Standard Lenia" in 2D. This version of Lenia
    /// can have only a single channel and a single convolution channel. It also does not
    /// support any weights, as it can be encoded within the `dt` parameter. 
    /// 
    /// The size of either dimension may not be smaller than 28 pixels. 
    /// 
    /// By default the kernel, growth function and dt parameter are set such that when 
    /// simulating, the simulation is capable of producing the ***Orbium unicaudatus*** glider.
    /// 
    /// ### Arguments
    /// 
    /// * `shape` - Reference to the shape that the channels in the `Lenia` instance shall have.
    /// 
    /// ### Returns
    /// 
    /// A `StandardLenia2D` instance, which implements `Lenia` trait.
    /// 
    /// ### Panics
    /// 
    /// * If the length of `shape` is not `2`.
    /// 
    /// * If either of the axis lengths in `shape` are `<28`.
    fn new(shape: &[usize]) -> Self {
        if shape.len() < 2 || shape.len() > 2 { 
            panic!("Expected 2 dimensions for 2D Standard Lenia! Found {}", shape.len()); 
        }
        let mut proper_dims: Vec<usize> = Vec::new();
        for (i, dim) in shape.iter().enumerate() {
            if *dim < StandardLenia2D::STANDARD_LENIA_DEFAULT_KERNEL_SIZE {
                panic!("Axis {} is extremely small ({} pixels). Minimum size of each axis for 2D Standard Lenia is {} pixels.", i, *dim, StandardLenia2D::STANDARD_LENIA_DEFAULT_KERNEL_SIZE);
            }
            else {
                proper_dims.push(*dim);
            }
        }
        let kernel = Kernel::from(
            kernels::gaussian_donut_2d(
                StandardLenia2D::STANDARD_LENIA_DEFAULT_KERNEL_SIZE, 
                1.0/6.7
            ), 
            &proper_dims
        );
        let conv_field = ndarray::ArrayD::from_elem(proper_dims, Complex::new(0.0, 0.0));
        let conv_channel = ConvolutionChannel {
            input_channel: 0,
            input_buffer: conv_field.clone(),
            kernel: kernel,
            field: conv_field,
            growth: growth_functions::standard_lenia,
            growth_params: vec![0.15, 0.017],
        };

        let channel = Channel {
            field: conv_channel.field.clone(),
            inverse_weight_sums: vec![1.0],
        };

        let mut channel_shape = Vec::new();
        for dim in shape {
            channel_shape.push(*dim);
        }
        
        StandardLenia2D{
            forward_fft_instances: vec![fft::PreplannedFFTND::preplan_by_prototype(&channel.field, false)],
            inverse_fft_instances: vec![fft::PreplannedFFTND::preplan_by_prototype(&channel.field, true)],
            dt: 0.1,
            channel: vec![channel],
            shape: channel_shape,
            convolution_buffer: vec![conv_channel.field.clone()],
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
            &mut self.convolution_buffer[0], 
            &[0, 1]
        );
        self.convolution_buffer[0].zip_mut_with(&self.conv_channel[0].kernel.transformed, 
            |a, b| {    // Complex multiplication without cloning
                let ac = a.re * b.re;
                let bd = a.im * b.im;
                let real = ac - bd;
                a.im = ((a.re + a.im) * (b.re + b.im)) - real;
                a.re = real;
            }
        );
        self.inverse_fft_instances[0].transform(
            &mut self.convolution_buffer[0], 
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

    fn set_channels(&mut self, num_channels: usize) {
        println!("Initializing channels not available for Standard Lenia! Try using Extended Lenia instead.");
    }

    fn set_conv_channels(&mut self, num_conv_channels: usize) {
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

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn get_data_as_ref(&self, channel: usize) -> &ndarray::ArrayD<Complex<f64>> {
        &self.channel[0].field
    }

    fn get_data_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<Complex<f64>> {
        &mut self.channel[0].field
    }

    fn dt(&self) -> f64 {
        self.dt
    }

    fn channels(&self) -> usize {
        1 as usize
    }

    fn conv_channels(&self) -> usize {
        1 as usize
    }
}
