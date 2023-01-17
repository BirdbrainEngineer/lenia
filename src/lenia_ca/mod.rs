#![allow(dead_code)]
#![allow(unused_variables)]

use ndarray::{self, Axis, Slice, Order};
use num_complex::Complex;
mod fft;
pub mod lenias;
pub mod kernels;
pub mod growth_functions;
pub mod seeders;

/// Samples the normal distribution where the peak (at `x = mu`) is 1.
/// This is not suitable for use as a gaussian probability density function!
/// 
/// ### Arguments
/// 
/// * `x` - Point of the normal distribution to sample.
/// 
/// * `mu` - The mean (point of the highest value/peak) of the normal distribution.
/// 
/// * `stddev` - Standard deviation of the normal distribution. 
/// 
/// ### Returns
/// `f64` value of the defined gaussian distribution evaluated at x. 
fn sample_normal(x: f64, mu: f64, stddev: f64) -> f64 {
    (-(((x - mu) * (x - mu))/(2.0 * (stddev * stddev)))).exp()
}

/// Container type for a `Lenia` implementation. Also contains the conversions to- and from 
/// complex numbers normally used in the internals of `Lenia` instances. 
pub struct Simulator<L: Lenia> {
    sim: L,
}

impl<L: Lenia> Simulator<L> {
    /// Initialize a simulator. 
    /// 
    /// Barring wanting to change the type of the `Lenia` instance used by the `Simulator`, 
    /// this should ever need to be called only once during the lifetime of your 
    /// Lenia simulation program.
    /// 
    /// ### Arguments
    /// 
    /// * `channel_shape` - The shape (number of dimensions and their lengths) of the
    /// channels for the `Lenia` instance. 
    /// 
    /// ### Returns
    /// A new instance of a `Simulator`. 
    pub fn new(channel_shape: Vec<usize>) -> Self {
        Simulator{
            sim: L::new(&channel_shape),
        }
    }

    /// Re-initializes the `Lenia` instance, losing **all** of the previous changes, such as
    /// kernel changes, channel additions or any other parameter changes from the defaults
    /// of the specific `Lenia` instance implementation. 
    /// 
    /// Call this if the shape of the channels needs to be changed, or a major restructuring of
    /// channels and/or convolution channels is wanted.
    /// 
    /// ### Arguments
    /// 
    /// * `channel_shape` - The shape (number of dimensions and their lengths) of the
    /// channels for the `Lenia` instance. 
    pub fn remake(&mut self, channel_shape: Vec<usize>) {
        self.sim = L::new(&channel_shape);
    }

    /// Set the kernel of the specified convolution channel. 
    /// 
    /// ### Arguments
    /// 
    /// * `kernel` - n-dimensional `f64` array (`ndarray::ArrayD`), where the number of 
    /// dimensions / axes must match the number of dimensions / axes of the channels of the
    /// `Lenia` instance. 
    /// 
    /// * `convolution_channel` - The convolution channel to which the new kernel is to be assigned.
    pub fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, convolution_channel: usize) {
        self.sim.set_kernel(kernel, convolution_channel);
    }

    /// Set the growth function and its parameters of the specified convolution channel.
    /// 
    /// ### Arguments
    /// 
    /// * `f` - Growth function to use.
    /// 
    /// * `growth_parameters` - The parameters passed to the growth function. 
    /// 
    /// * `convolution_channel` - The convoltution channel to which the new growth function and
    /// parameters are to be assigned.
    pub fn set_growth_function(&mut self, f: fn(f64, &[f64]) -> f64, growth_parameters: Vec<f64>, convolution_channel: usize) {
        self.sim.set_growth(f, growth_parameters, convolution_channel);
    }

    /// Set the integration step (a.k.a. timestep) parameter `dt` of the `Lenia` instance.
    /// 
    /// ### Arguments
    /// 
    /// * `dt` - The new dt value for the `Lenia` instance to use.
    pub fn set_dt(&mut self, dt: f64) {
        self.sim.set_dt(dt);
    }

    /// Performs a single iteration of the `Lenia` instance. Channels are updated with
    /// the resulting new state of the simulation. 
    pub fn iterate(&mut self) {
        self.sim.iterate();
    }

    /// Fills a channel with user data. The shapes of the `data` and the channels in the
    /// `Lenia` instance must be the same. 
    /// 
    /// ### Arguments
    /// 
    /// * `data` - Reference to the n-dimensional array (`ndarray::ArrayD`) of `f64` values
    /// from which to fill the channel's data.
    /// 
    /// * `channel` - Index of the channel to fill. 
    pub fn fill_channel(&mut self, data: &ndarray::ArrayD<f64>, channel: usize) {
        let channel_data = self.sim.get_data_as_mut_ref(channel);
        channel_data.zip_mut_with(data, 
            |a, b| {
                a.re = *b;
                a.im = 0.0;
            }
        );
    }

    /// Retrieve an owned copy ("deep-copy") of the specified channel's data.
    /// 
    /// ### Arguments
    /// 
    /// * `channel` - Index of the channel to copy data from.
    /// 
    /// ### Returns
    /// An array (`ndarray::ArrayD`) of the real components of the channel's data.
    pub fn get_channel_data_f64(&self, channel: usize) -> ndarray::ArrayD<f64> {
        let out = self.sim.get_data_as_ref(channel).map(
            |a| {
                a.re
            }
        );
        out
    }

    /// Retrieve a referenced copy ("shallow-copy") of the specified channel's data. 
    /// Since "deep-copying" `f64` takes about the same amount of time and memory 
    /// on most modern machines as making a copy of references,
    /// then the use case for this is nearly nonexistent. 
    pub fn get_channel_data_as_ref(&self, channel: usize) -> ndarray::ArrayD<&f64> {
        let out = self.sim.get_data_as_ref(channel).map(
            |a| {
                &a.re
            }
        );
        out
    }

    /// Retrieve the real component of a channel's data as mutable references. This allows for
    /// modification of the channel's current data directly. 
    /// 
    /// ### Arguments
    /// 
    /// * `channel` - Index of the channel from which the mutable references are to be taken.
    /// 
    /// ### Returns
    /// An n-dimensional array (`ndarray::ArrayD`) of references to the channel's data's real components.
    pub fn get_channel_data_as_mut_ref(&mut self, channel: usize) -> ndarray::ArrayD<&mut f64> {
        let out = self.sim.get_data_as_mut_ref(channel).map_mut(
            |a| {
                &mut a.re
            }
        );
        out
    }

    /// Receives a 2d array (`ndarray::Array2`) of `f64` values of a 2d slice of a channel's data. 
    /// Use this to simply get a 2d frame for rendering. 
    /// 
    /// ### Arguments
    /// 
    /// * `channel` - Index of the channel to retrieve data from.
    /// 
    /// * `display_axes` - Indexes of the axes to retrieve
    /// 
    /// * `dimensions` - Which indexes in any other axes the 2d slice is taken from. 
    /// The entries for axes selected in `display_axes` can be any number, and will be disregarded. 
    /// 
    /// ### Returns
    /// 2d array (`ndarray::Array2`) filled with `f64` values where the dimension lengths are 
    /// the same as the axis lengths of the axes chosen with `display axes`.
    pub fn get_frame(&self, channel: usize, display_axes: &[usize; 2], dimensions: &[usize]) -> ndarray::Array2<f64> {
        let data_ref = self.sim.get_data_as_ref(channel);
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
        .mapv(|el| { el.re })
    }

    /// Get the current integration step (a.k.a. timestep) parameter `dt` of the `Lenia` instance.
    pub fn dt(&self) -> f64 {
        self.sim.dt()
    }

    /// Get the shape of the channels and convolution channels of the `Lenia` instance.
    pub fn channel_shape(&self) -> &[usize] {
        self.sim.shape()
    }

    /// Get the number of channels initialized in the `Lenia` instance.
    pub fn channels(&self) -> usize {
        self.sim.channels()
    }

    /// Get the number of convolution channels initialized in the `Lenia` instance.
    pub fn convolution_channels(&self) -> usize {
        self.sim.conv_channels()
    }
}

pub trait Lenia {
    /// Creates a new `Lenia` instance. 
    fn new(shape: &[usize]) -> Self;
    /// Sets the number of channels in the `Lenia` instance. 
    /// 
    /// *If the new number of channels is less than before then the user is responsible for re-making
    /// the convolution channels or deleting invalidated convolution channels and
    /// make sure that no convolution channel tries to convolute a non-existent channel!*
    /// 
    /// For the above reason, the `Simulator` struct does not implement a function to
    /// reduce the number of channels, but rather asks to remake the whole `Lenia` instance
    /// when reducing the number of channels is wanted.
    fn set_channels(&mut self, num_channels: usize);
    /// Sets the number of convolution channels in the `Lenia` instance. 
    /// 
    /// Any convolution channels
    /// that have an index larger than the new number of channels **will be dropped**. Conversely,
    /// no convolution channels get invalidated if the new number of convolution channels is
    /// greater than the previous number of convolution channels. 
    fn set_conv_channels(&mut self, num_conv_channels: usize);
    /// Sets the convolution kernel for a convolution channel.
    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize);
    /// Sets the growth function for a convolution channel.
    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize);
    /// Sets the weights of a convolution channel output. 
    fn set_weights(&mut self, new_weights: &[f64], conv_channel: usize);
    /// Sets the dt parameter of the `Lenia` instance. 
    fn set_dt(&mut self, new_dt: f64);
    /// Returns a reference ("shallow-copy") to a channel's current data. 
    fn get_data_as_ref(&self, channel: usize) -> &ndarray::ArrayD<Complex<f64>>;
    /// Returns a mutable reference ("shallow copy") to a channel's current data.
    fn get_data_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<Complex<f64>>;
    /// Returns the shape of the channels and convolution channels (as reference). 
    fn shape(&self) -> &[usize];
    /// Returns the current `dt` parameter of the `Lenia` instance.
    fn dt(&self) -> f64;
    /// Returns the number of channels in the `Lenia` instance.
    fn channels(&self) -> usize;
    /// Returns the number of convolution channels in the `Lenia` instance.
    fn conv_channels(&self) -> usize;
    /// Calculates the next state of the `Lenia` instance, and updates the data in channels accordingly.
    fn iterate(&mut self);
}

/// The `Channel` struct is a wrapper for holding the data of a single channel in a 
/// `Lenia` simulation. The struct also implements functionality for applying the
/// weights of the convolution channels and summing up the final result in any
/// Lenia system more complex than the Standard Lenia. 
pub struct Channel {
    pub field: ndarray::ArrayD<Complex<f64>>,
    pub inverse_weight_sums: Vec<f64>,
}

/// The `ConvolutionChannel` struct holds relevant data for the convolution step of the
/// Lenia simulation. This includes the kernel, the intermittent convolution step, and the 
/// growth function.
pub struct ConvolutionChannel {
    pub input_channel: usize,
    pub input_buffer: ndarray::ArrayD<Complex<f64>>,
    pub field: ndarray::ArrayD<Complex<f64>>,
    pub kernel: Kernel,
    pub growth: fn(f64, &[f64]) -> f64,
    pub growth_params: Vec<f64>,
}

/// The `Kernel` struct holds the data of a specific kernel to be used for convolution in
/// the Lenia simulation. It also implements the necessary conversions to normalize a
/// kernel and prepare it for convolution using fast-fourier-transform. 
pub struct Kernel {
    pub base: ndarray::ArrayD<f64>,
    pub normalized: ndarray::ArrayD<f64>,
    pub transformed: ndarray::ArrayD<Complex<f64>>,
}

impl Kernel {
    /// Creates a new Kernel struct from an n-dimensional array (`ndarray::ArrayD`) of `f64` values.
    /// 
    /// Creates the normalized version of the kernel.
    /// 
    /// Creates a version of the kernel that has been transformed using discrete-fourier-transform, 
    /// and shifted, for future use in fast-fourier-transform based convolution. 
    /// 
    /// ### Arguments
    /// 
    /// * `kernel` - Array (`ndarray::ArrayD`) of `f64` values that define the weights of the kernel.
    /// 
    /// * `channel_shape` - Shape of the channel the kernel is supposed to act on.
    /// 
    /// ### Returns
    /// 
    /// A `Kernel` instance containing the base kernel, normalized version of the base kernel and 
    /// a fourier-transformed kernel representation. 
    /// 
    /// ### Panics
    /// 
    /// * If the number of axes of the `kernel` and `channel_shape` are not equal.
    /// 
    /// * If any of the corresponding axis lengths in `kernel` are greater than in `channel_shape`.
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



