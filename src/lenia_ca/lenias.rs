use glium::buffer;
use ndarray;
use num_complex::Complex;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use super::*;
use super::fft::{PreplannedFFTND, PlannedFFTND};

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
    channel: Channel,
    shape: Vec<usize>,
    convolution_buffer: ndarray::ArrayD<Complex<f64>>,
    conv_channel: ConvolutionChannel,
    forward_fft_instance: fft::PreplannedFFTND,
    inverse_fft_instance: fft::PreplannedFFTND,
}

impl StandardLenia2D {
    const DEFAULT_KERNEL_SIZE: usize = 28;
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
            panic!("StandardLenia2D(new): Expected 2 dimensions for 2D Standard Lenia! Found {}.", shape.len()); 
        }
        for (i, dim) in shape.iter().enumerate() {
            if *dim < Self::DEFAULT_KERNEL_SIZE {
                panic!("StandardLenia2D(new): Axis {} is extremely small ({} pixels). Minimum size of each axis for 2D Standard Lenia is {} pixels.", i, *dim, Self::DEFAULT_KERNEL_SIZE);
            }
        }
        let kernel = Kernel::from(
            kernels::gaussian_donut_2d(
                Self::DEFAULT_KERNEL_SIZE, 
                1.0/6.7
            ), 
            shape
        );
        let conv_field = ndarray::ArrayD::from_elem(shape, Complex::new(0.0, 0.0));
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
            weights: vec![1.0],
            weight_sum_reciprocal: 1.0,
        };

        let mut channel_shape = Vec::new();
        for dim in shape {
            channel_shape.push(*dim);
        }
        
        StandardLenia2D{
            forward_fft_instance: fft::PreplannedFFTND::preplan_by_prototype(&channel.field, false),
            inverse_fft_instance: fft::PreplannedFFTND::preplan_by_prototype(&channel.field, true),
            dt: 0.1,
            channel: channel,
            shape: channel_shape,
            convolution_buffer: conv_channel.field.clone(),
            conv_channel: conv_channel,
        }
    }

    fn iterate(&mut self) {
        self.conv_channel.input_buffer.zip_mut_with(&self.channel.field, 
            |a, b| {
                a.re = b.re;
                a.im = 0.0;
            }
        );
        self.forward_fft_instance.transform(
            &mut self.conv_channel.input_buffer, 
            &mut self.convolution_buffer, 
            &[0, 1]
        );
        self.convolution_buffer.zip_mut_with(&self.conv_channel.kernel.transformed, 
            |a, b| {
                // Complex multiplication without cloning
                let real = (a.re * b.re) - (a.im * b.im);
                a.im = ((a.re + a.im) * (b.re + b.im)) - real;
                a.re = real;
            }
        );
        self.inverse_fft_instance.transform(
            &mut self.convolution_buffer, 
            &mut self.conv_channel.field, 
            &[1, 0]
        );
        self.channel.field.zip_mut_with(&self.conv_channel.field, 
            |a, b| {
                a.re = (a.re + ((self.conv_channel.growth)(b.re, &self.conv_channel.growth_params) * self.dt)).clamp(0.0, 1.0);
                a.im = 0.0;
            }
        );
    }

    fn set_channels(&mut self, num_channels: usize) {
        println!("Changing the number of channels is not available for Standard Lenia! Try using Extended Lenia instead.");
    }

    fn set_conv_channels(&mut self, num_conv_channels: usize) {
        println!("Changing the number of channels is not available for Standard Lenia! Try using Extended Lenia instead.");
    }

    fn set_source_channel(&mut self, conv_channel: usize, src_channel: usize) {
        println!("Adding or changing source channels is not available for Standard Lenia! Try using Extended Lenia instead.");
    }

    fn set_weights(&mut self, new_weights: &[f64], conv_channel: usize) {
        println!("Adding or changing convolution output weights is not available for Standard Lenia! Try usingh Extended Lenia instead.");
    }

    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize) {
        self.conv_channel.kernel = Kernel::from(kernel, self.channel.field.shape());
    }

    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize) {
        self.conv_channel.growth = f;
        self.conv_channel.growth_params = growth_params;
    }

    fn set_dt(&mut self, new_dt: f64) {
        self.dt = new_dt;
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn get_data_as_ref(&self, channel: usize) -> &ndarray::ArrayD<Complex<f64>> {
        &self.channel.field
    }

    fn get_data_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<Complex<f64>> {
        &mut self.channel.field
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

    fn weights(&self, channel: usize) -> &[f64] {
        &self.channel.weights
    }
}



/// `ExtendedLenia` struct implements the extended Lenia system, with support for n-dimensional
/// channels, multiple channels, multiple kernels & growth functions (convolution channels) 
/// and individual weights for convolution channels. 
pub struct ExtendedLenia {
    dt: f64,
    channels: Vec<Channel>,
    shape: Vec<usize>,
    buffers: Vec<ndarray::ArrayD<Complex<f64>>>,
    conv_channels: Vec<ConvolutionChannel>,
    forward_fft_instances: Vec<fft::PlannedFFTND>,
    inverse_fft_instances: Vec<fft::PlannedFFTND>,
}

impl ExtendedLenia {
    const DEFAULT_KERNEL_SIZE: usize = 28;
}

impl Lenia for ExtendedLenia {
    /// Create and initialize a new instance of "ExtendedLenia`. 
    fn new(shape: &[usize]) -> Self {
        for (i, dim) in shape.iter().enumerate() {
            if *dim < Self::DEFAULT_KERNEL_SIZE {
                panic!("Axis {} is extremely small ({} pixels). Minimum size of each axis for 2D Standard Lenia is {} pixels.", i, *dim, Self::DEFAULT_KERNEL_SIZE);
            }
        }
        let kernel = Kernel::from(
            kernels::gaussian_donut_nd(
                Self::DEFAULT_KERNEL_SIZE, 
                shape.len(),
                1.0/6.7
            ), 
            shape
        );
        let conv_field = ndarray::ArrayD::from_elem(shape, Complex::new(0.0, 0.0));
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
            weights: vec![1.0],
            weight_sum_reciprocal: 1.0,
        };

        let mut channel_shape = Vec::new();
        for dim in shape {
            channel_shape.push(*dim);
        }
        
        ExtendedLenia{
            forward_fft_instances: vec![fft::PlannedFFTND::new(&channel_shape, false)],
            inverse_fft_instances: vec![fft::PlannedFFTND::new(&channel_shape, true)],
            dt: 0.1,
            channels: vec![channel],
            shape: channel_shape,
            buffers: vec![conv_channel.field.clone()],
            conv_channels: vec![conv_channel],
        }
    }

    // This is a very long and complex function, sorry.
    // It uses concurrency to calculate multiple convolutions at the same time, as well as
    // apply weights and sum the results. 
    fn iterate(&mut self) {
        let mut axes: Vec<usize> = Vec::with_capacity(self.shape.len());
        let mut inverse_axes: Vec<usize> = Vec::with_capacity(self.shape.len());
        for i in 0..self.shape.len() {
            axes.push(i);
        }
        for i in (0..self.shape.len()).rev() {
            inverse_axes.push(i);
        }

        //Create mutexes and rwlocks
        let mut sources: Vec<usize> = Vec::with_capacity(self.conv_channels.len());
        let mut channel_rwlocks: Vec<Arc<RwLock<Channel>>> = Vec::with_capacity(self.channels.len());
        let mut convolution_mutexes: Vec<Arc<Mutex<ConvolutionChannel>>> = Vec::with_capacity(self.conv_channels.len());
        let mut buffer_mutexes: Vec<Arc<Mutex<ndarray::ArrayD<Complex<f64>>>>> = Vec::with_capacity(self.buffers.len());
        let mut forward_fft_mutexes: Vec<Arc<Mutex<PlannedFFTND>>> = Vec::with_capacity(self.forward_fft_instances.len());
        let mut inverse_fft_mutexes: Vec<Arc<Mutex<PlannedFFTND>>> = Vec::with_capacity(self.inverse_fft_instances.len());

        for _ in 0..self.channels.len() {
            channel_rwlocks.push(Arc::new(RwLock::new(self.channels.remove(0))));
        }
        for _ in 0..self.conv_channels.len() {
            sources.push(self.conv_channels[0].input_channel);
            convolution_mutexes.push(Arc::new(Mutex::new(self.conv_channels.remove(0))));
        }
        for _ in 0..self.buffers.len() {
            buffer_mutexes.push(Arc::new(Mutex::new(self.buffers.remove(0))));
        }
        for _ in 0..self.forward_fft_instances.len() {
            forward_fft_mutexes.push(Arc::new(Mutex::new(self.forward_fft_instances.remove(0))));
            inverse_fft_mutexes.push(Arc::new(Mutex::new(self.inverse_fft_instances.remove(0))));
        }

        // Concurrent convolutions
        let mut convolution_handles = Vec::with_capacity(convolution_mutexes.len());

        for i in 0..convolution_mutexes.len() {
            // Set up and aquire locks on data
            let axes_clone = axes.clone();
            let inverse_axes_clone = inverse_axes.clone();
            let source_lock = Arc::clone(&channel_rwlocks[sources[i]]);
            let buffer_lock = Arc::clone(&buffer_mutexes[i]);
            let convolution_channel_lock = Arc::clone(&convolution_mutexes[i]);
            let forward_fft_lock = Arc::clone(&forward_fft_mutexes[i]);
            let inverse_fft_lock = Arc::clone(&inverse_fft_mutexes[i]);

            convolution_handles.push(thread::spawn(move || {
                let mut convolution_channel = convolution_channel_lock.lock().unwrap();
                let input = source_lock.read().unwrap();
                let mut buffer = buffer_lock.lock().unwrap();
                let mut forward_fft = forward_fft_lock.lock().unwrap();
                let mut inverse_fft = inverse_fft_lock.lock().unwrap();
                // Get data from source channel
                buffer.zip_mut_with(&input.field, 
                    |a, b| {
                        a.re = b.re;
                        a.im = 0.0;
                    }
                );
                // Forward fft the input data
                forward_fft.transform(&mut buffer);
                // Fourier-transform convolute
                buffer.zip_mut_with(&convolution_channel.kernel.transformed, 
                    |a, b| {    // Complex multiplication without cloning
                        let real = (a.re * b.re) - (a.im * b.im);
                        a.im = ((a.re + a.im) * (b.re + b.im)) - real;
                        a.re = real;
                    }
                );
                // Inverse fft to get convolution result
                inverse_fft.transform(&mut buffer);

                convolution_channel.field.zip_mut_with(&buffer, 
                    |a, b| {
                        a.re = b.re;
                        a.im = 0.0;
                    }
                );
                // Apply growth function
                let growth_func = (convolution_channel.growth, &convolution_channel.growth_params.clone());
                convolution_channel.field.map_inplace(|a| {
                    a.re = (growth_func.0)(a.re, &growth_func.1);
                    a.im = 0.0;
                });
            }));
        }

        let mut summing_handles = Vec::with_capacity(channel_rwlocks.len());

        for handle in convolution_handles {
            handle.join().unwrap();
        }

        // Collapse convolution channel mutexes back into a single owned vector
        let mut convolution_channels: Vec<ConvolutionChannel> = Vec::with_capacity(convolution_mutexes.len());
        for i in 0..convolution_mutexes.len() {
            let data = convolution_mutexes.remove(0);
            convolution_channels.push(Arc::try_unwrap(data).unwrap().into_inner().unwrap());
        }

        // Concurrent summing of results
        // Make and aquire locks
        let convoluted_results_rwlock = Arc::new(RwLock::new(convolution_channels));

        for i in 0..channel_rwlocks.len() {
            let dt = self.dt.clone();
            let channel_lock = Arc::clone(&channel_rwlocks[i]);
            let buffer_lock = Arc::clone(&buffer_mutexes[i]);
            let convoluted_results_lock = Arc::clone(&convoluted_results_rwlock);
            
            // Thread code
            summing_handles.push(thread::spawn(move || {
                let mut channel = channel_lock.write().unwrap();
                let mut buffer = buffer_lock.lock().unwrap();
                let convoluted_results = convoluted_results_lock.read().unwrap();

                // Add together all of the weighted results from convolutions
                for i in 0..channel.weights.len() {
                    buffer.zip_mut_with(&convoluted_results[i].field, 
                        |a, b| {
                            if i == 0 { a.re = 0.0; }
                            a.re += b.re * channel.weights[i];
                        }
                    );
                }
                // Apply weighted average and dt, store result in channel
                let weighted_average_reciprocal = channel.weight_sum_reciprocal;
                channel.field.zip_mut_with(&buffer, 
                    |a, b| {
                        a.re = (a.re + (b.re * weighted_average_reciprocal * dt)).clamp(0.0, 1.0);
                        a.im = 0.0;
                    }
                );
            }));
        }

        for _ in 0..forward_fft_mutexes.len() {
            self.forward_fft_instances.push(Arc::try_unwrap(forward_fft_mutexes.remove(0)).unwrap().into_inner().unwrap());
        }
        for _ in 0..inverse_fft_mutexes.len() {
            self.inverse_fft_instances.push(Arc::try_unwrap(inverse_fft_mutexes.remove(0)).unwrap().into_inner().unwrap());
        }

        for handle in summing_handles {
            handle.join().unwrap();
        }

        // Return ownership of all data back to Lenia instance
        self.conv_channels = Arc::try_unwrap(convoluted_results_rwlock).unwrap().into_inner().unwrap();
        for _ in 0..channel_rwlocks.len() {
            self.channels.push(Arc::try_unwrap(channel_rwlocks.remove(0)).unwrap().into_inner().unwrap());
        }
        for _ in 0..buffer_mutexes.len() {
            self.buffers.push(Arc::try_unwrap(buffer_mutexes.remove(0)).unwrap().into_inner().unwrap());
        }
        
    }

    fn set_channels(&mut self, num_channels: usize) {
        if num_channels <= self.channels.len() {
            for i in (num_channels..self.channels.len()).rev() {
                self.channels.remove(i);
                if i >= self.conv_channels.len() {
                    self.buffers.remove(i);
                }
            }
        }
        else {
            if num_channels > self.buffers.len() {
                for _ in self.buffers.len()..num_channels {
                    self.buffers.push(self.buffers[0].clone());
                }
            }
            let mut weights_prototype: Vec<f64> = Vec::new();
            for _ in &self.channels[0].weights {
                weights_prototype.push(0.0);
            }
            for _ in self.channels.len()..num_channels {
                self.channels.push(
                    Channel { 
                        field: ndarray::ArrayD::from_elem(self.shape.clone(), Complex{re: 0.0, im: 0.0}),
                        weights: weights_prototype.clone(),
                        weight_sum_reciprocal: 0.0,
                    }
                );
            }
        }
    }

    fn set_conv_channels(&mut self, num_conv_channels: usize) {
        if num_conv_channels <= self.conv_channels.len() {
            for i in (num_conv_channels..self.conv_channels.len()).rev() {
                self.conv_channels.remove(i);
                self.forward_fft_instances.remove(i);
                self.inverse_fft_instances.remove(i);
                if i >= self.channels.len() {
                    self.buffers.remove(i);
                }
            }
            for channel in &mut self.channels {
                for i in (num_conv_channels..channel.weights.len()).rev() {
                    channel.weights.remove(i);
                }
                let sum: f64 = channel.weights.iter().sum();
                channel.weight_sum_reciprocal = 1.0 / sum;
            }
        }
        else {
            if num_conv_channels > self.buffers.len() {
                for _ in self.buffers.len()..num_conv_channels {
                    self.buffers.push(self.buffers[0].clone());
                }
            }
            for i in self.conv_channels.len()..num_conv_channels {
                self.conv_channels.push(
                    ConvolutionChannel { 
                        input_channel: 0, 
                        input_buffer: self.buffers[0].clone(), 
                        field: self.conv_channels[0].field.clone(), 
                        kernel: Kernel::from(kernels::empty(&self.shape), &self.shape), 
                        growth: growth_functions::pass, 
                        growth_params: vec![0.0],
                    }
                );
                self.forward_fft_instances.push(fft::PlannedFFTND::new(&self.shape, false));
                self.inverse_fft_instances.push(fft::PlannedFFTND::new(&self.shape, true));
            }
            for channel in &mut self.channels {
                for _ in channel.weights.len()..num_conv_channels {
                    channel.weights.push(0.0);
                }
                let sum: f64 = channel.weights.iter().sum();
                channel.weight_sum_reciprocal = 1.0 / sum;
            }
        }
    }

    fn set_weights(&mut self, new_weights: &[f64], channel: usize) {
        let mut weights: Vec<f64>;
        if new_weights.len() < self.conv_channels.len() {
            weights = new_weights.clone().to_vec();
            for _ in new_weights.len()..self.conv_channels.len() {
                weights.push(0.0);
            }
        }
        else {
            weights = Vec::with_capacity(new_weights.len());
            for i in 0..self.conv_channels.len() {
                weights.push(new_weights[i]);
            }
        }
        let sum: f64 = weights.iter().sum();
        self.channels[channel].weights = weights;
        self.channels[channel].weight_sum_reciprocal = 1.0 / sum;
    }

    fn set_source_channel(&mut self, conv_channel: usize, src_channel: usize) {
        self.conv_channels[conv_channel].input_channel = src_channel;
    } 

    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize) {
        self.conv_channels[conv_channel].kernel = Kernel::from(kernel, &self.shape);
    } 

    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize) {
        self.conv_channels[conv_channel].growth = f;
        self.conv_channels[conv_channel].growth_params = growth_params;
    } 

    fn set_dt(&mut self, new_dt: f64) {
        self.dt = new_dt;
    } 

    fn shape(&self) -> &[usize] {
        &self.shape
    } 

    fn get_data_as_ref(&self, channel: usize) -> &ndarray::ArrayD<Complex<f64>> {
        &self.channels[channel].field
    } 

    fn get_data_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<Complex<f64>> {
        &mut self.channels[channel].field
    } 

    fn dt(&self) -> f64 {
        self.dt
    } 

    fn channels(&self) -> usize {
        self.channels.len()
    } 

    fn conv_channels(&self) -> usize {
        self.conv_channels.len()
    } 

    fn weights(&self, channel: usize) -> &[f64] {
        &self.channels[channel].weights
    } 
}


