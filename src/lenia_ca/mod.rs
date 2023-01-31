#![allow(dead_code)]
#![allow(unused_variables)]
#[cfg(target_has_atomic = "ptr")]

use std::fmt;
use ndarray::{self, Axis, Slice, Order, IxDyn};
use num_complex::Complex;
use png;
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
pub fn sample_normal(x: f64, mu: f64, stddev: f64) -> f64 {
    (-(((x - mu) * (x - mu))/(2.0 * (stddev * stddev)))).exp()
}

pub fn store_frame_as_png(frame: &ndarray::ArrayD<f64>, frame_number: usize, folder_path: &str) {
    if frame.shape().is_empty() { panic!("lenia_ca::store_frame() - Can not store an empty frame!") }

    let path_base = format!("{}{}{}",
        if folder_path.is_empty() { &"./" } else { folder_path.clone() }, 
        if folder_path.chars().last().unwrap() != '/' && folder_path.chars().last().unwrap() != '\\' { &"/" } else { &"" },
        frame_number
    );
    let data;
    if frame.shape().len() == 1 {
        data = frame.to_shape((ndarray::IxDyn(&[frame.shape()[0], 1]), Order::RowMajor)).unwrap().mapv(|el| { el.clone() } );
    }
    else {
        data = frame.clone();
    }

    std::thread::spawn(move || {
        let mut indexes: Vec<usize> = vec![0; data.shape().len()];
        nested_png_export(path_base, &data, &mut indexes, 0);
    });
}

fn nested_png_export(path: String, data: &ndarray::ArrayD<f64>, indexes: &mut Vec<usize>, current_axis: usize) {
    if current_axis == (indexes.len() - 2) {
        let file_path = format!("{}.png", &path);
        println!("{}", &file_path);
        let file = std::fs::File::create(file_path).unwrap();
        let buf_writer = std::io::BufWriter::new(file);
        let mut encoder = png::Encoder::new(
            buf_writer, 
            data.shape()[data.shape().len()-2] as u32, 
            data.shape()[data.shape().len()-1] as u32
        );
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let image_data = data.slice_each_axis(
            |a|{
                if a.axis.index() == (indexes.len() - 2) || a.axis.index() == (indexes.len() - 1) {
                    return Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }
                }
                else {
                    return Slice {
                        start: indexes[a.axis.index()] as isize,
                        end: Some((indexes[a.axis.index()] + 1) as isize),
                        step: 1,
                    }
                }
            }
        )
        .to_shape(((data.shape()[data.shape().len() - 2] * data.shape()[data.shape().len() - 1]), Order::ColumnMajor))
        .unwrap()
        .mapv(|el| { (el * 255.0) as u8 });
        writer.write_image_data(image_data.as_slice().unwrap());
    }
    else {
        for i in 0..data.shape()[current_axis] {
            indexes[current_axis] = i;
            nested_png_export(
                format!("{}_{}", &path, i), 
                data,
                indexes,
                current_axis + 1
            );
        }
    }
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
    pub fn new(channel_shape: &[usize]) -> Self {
        Simulator{
            sim: L::new(channel_shape),
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

    /// Set the number of channels in the `Lenia` instance. 
    /// 
    /// **In case the number of channels
    /// is less than the current number of channels, it is up to the user to make sure that
    /// no convolution channel tries to use a dropped channel as its source!**
    /// 
    /// All values in newly created channels will be set to `0.0`.
    /// 
    /// The weights from all convolution channels into any newly created channels will start
    /// off at `0.0`.
    /// 
    /// ### Arguments
    /// 
    /// * `channels` - The number of channels the `Lenia` instance should have.
    /// 
    /// ### Panics
    /// 
    /// If `channels` is `0.0`.
    pub fn set_channels(&mut self, channels: usize) {
        if channels == 0 {
            panic!("Simulator::set_channels: Attempting to set the number of channels to 0. This is not allowed.");
        }
        if channels == self.sim.channels() { return; }
        self.sim.set_channels(channels);
    }

    /// Set the number of convolution channels in the `Lenia` instance. 
    /// 
    /// If the new number of 
    /// convolution channels is less than currently, then any convolution channels with an index
    /// higher than the new number of channels will be dropped, and their corresponding contribution
    /// to weighted averages for summing purged. 
    /// 
    /// If the new number of convolution channels is greater than currently then any new 
    /// convolution channels will need to have their kernels and growth functions set. In addition,
    /// weights for the new convolution channels will default to `0.0`. 
    /// 
    /// ### Arguments
    /// 
    /// * `convolution_channels` - The number of convolution channels the `Lenia` instance should have.
    /// 
    /// ### Panics
    /// 
    /// If `convolution_channels` is `0.0`.
    pub fn set_convolution_channels(&mut self, convolution_channels: usize) {
        if convolution_channels == 0 {
            panic!("Simulator::set_convolution_channels: Attempting to set the number of convolution channels to 0. This is not allowed.");
        }
        if convolution_channels == self.convolution_channels() { return; }
        self.sim.set_conv_channels(convolution_channels);
    }

    /// Set the source channel a given convolution channel should act on. 
    /// 
    /// ### Arguments
    /// 
    /// * `convolution_channel` - The convolution channel which will have its source changed.
    /// 
    /// * `source_channel` - The channel that the convolution channel should use as its source
    /// for convoluting.
    /// 
    /// ### Panics
    /// 
    /// * If the specified `convolution_channel` does not exist.
    /// 
    /// * If the specified `source_channel` does not exist.
    pub fn set_convolution_channel_source(&mut self, convolution_channel: usize, source_channel: usize) {
        if convolution_channel >= self.sim.conv_channels() {
            panic!("Simulator::set_convolution_channel_source: Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        if source_channel >= self.sim.channels() {
            panic!("Simulator::set_convolution_channel_source: Specified channel (index {}) does not exist. Current number of channels: {}.", source_channel, self.sim.channels());
        }
        self.sim.set_source_channel(convolution_channel, source_channel);
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
    /// 
    /// ### Panics
    /// 
    /// If the specified `convolution_channel` does not exist. 
    pub fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, convolution_channel: usize) {
        if convolution_channel >= self.sim.conv_channels() {
            panic!("Simulator::set_kernel: Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
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
    /// 
    /// ### Panics
    /// 
    /// If the specified `convolution_channel` does not exist. 
    pub fn set_growth_function(&mut self, f: fn(f64, &[f64]) -> f64, growth_parameters: Vec<f64>, convolution_channel: usize) {
        if convolution_channel >= self.sim.conv_channels() {
            panic!("Simulator::set_growth_function: Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        self.sim.set_growth(f, growth_parameters, convolution_channel);
    }

    /// Set the convolution channel input weights for a specific channel.
    /// 
    /// * If the length of weights is greater than the number of convolution channels, 
    /// then the weights with indexes not corresponding to a convolution channel
    /// will be disregarded.
    /// 
    /// * If the length of weights is less than the number of convolution channels, 
    /// then the weights will default to `0.0`.
    /// 
    /// ### Parameters
    /// 
    /// * `channel` - The channel, which the new weights will be assigned to.
    /// 
    /// * `weights` - The weights to assign. Index in the array corresponds to
    /// the index of the convoution channel. 
    pub fn set_weights(&mut self, channel: usize, weights: &[f64]) {
        self.sim.set_weights(weights, channel);
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
    ///
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn fill_channel(&mut self, data: &ndarray::ArrayD<f64>, channel: usize) {
        if channel >= self.sim.channels() {
            panic!("Simulator::fill_channel: Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        let channel_data = self.sim.get_channel_as_mut_ref(channel);
        channel_data.zip_mut_with(data, 
            |a, b| {
                *a = *b;
            }
        );
    }

    /// Retrieve a referenced to the specified channel's data. 
    /// 
    /// ### Arguments
    /// 
    /// * `channel` - Index of the channel to get a reference from.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn get_channel_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_channel_data_as_ref: Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        self.sim.get_channel_as_ref(channel)
    }

    /// Retrieve a mutable reference to the specified channel's data. This allows for modification
    /// of the channel's data directly. 
    /// 
    /// ### Arguments
    /// 
    /// * `channel` - Index of the channel from which the mutable reference to data is to be taken.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn get_channel_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_channel_data_as_ref() - Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        self.sim.get_channel_as_mut_ref(channel)
    }
    
    /// Retrieve a reference to the specified channel's "deltas". Deltas are the amounts added onto the 
    /// previous iteration's result to get the current iteration's result. 
    /// 
    /// Note that `dt` parameter has not been applied for this field, and the values are in the range
    /// `[-1.0..1.0]`.
    /// 
    /// ### Arguments
    /// 
    /// * `channel` - Index of the channel from which the reference to data is to be taken.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn get_deltas_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_deltas_as_ref() - Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        self.sim.get_deltas_as_ref(channel)
    }

    /// Retrieve a reference to the specified convolution channel's convolution result. 
    /// 
    /// ### Arguments
    /// 
    /// * `convolution_channel` - Index of the convolution channel from which to
    /// produce the `f64` `ndarray`. 
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist.
    pub fn get_convoluted(&self, convolution_channel: usize) -> ndarray::ArrayD<f64> {
        if convolution_channel >= self.sim.channels() {
            panic!("Simulator::get_convoluted() - Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        self.sim.get_convoluted_as_ref(convolution_channel).map(|a| { a.re })
    }

    /// Retrieve a reference to the specified convolution channel's convolution results with
    /// the groth function applied.
    /// 
    /// ### Arguments
    /// 
    /// * `convolution_channel` - Index of the convolution channel from which the
    /// reference to data is to be taken.
    /// 
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist.
    pub fn get_grown_as_ref(&self, convolution_channel: usize) -> &ndarray::ArrayD<f64> {
        if convolution_channel >= self.sim.channels() {
            panic!("Simulator::get_grown_as_ref() - Specified convolution channel (index {}) does not exist. Current number of convolution channels: {}.", convolution_channel, self.sim.conv_channels());
        }
        self.sim.get_grown_as_ref(convolution_channel)
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
    /// ### Panics
    /// 
    /// If the specified `channel` does not exist. 
    pub fn get_frame(&self, channel: usize, display_axes: &[usize; 2], dimensions: &[usize]) -> ndarray::Array2<f64> {
        if channel >= self.sim.channels() {
            panic!("Simulator::get_frame: Specified channel (index {}) does not exist. Current number of channels: {}.", channel, self.sim.channels());
        }
        let data_ref = self.sim.get_channel_as_ref(channel);
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
                        end: Some((dimensions[a.axis.index()] + 1) as isize),
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
        .mapv(|el| { el })
    }

    /// Get the current integration step (a.k.a. timestep) parameter `dt` of the `Lenia` instance.
    pub fn dt(&self) -> f64 {
        self.sim.dt()
    }

    /// Get the shape of the channels and convolution channels of the `Lenia` instance.
    pub fn shape(&self) -> &[usize] {
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
    /// **If the new number of channels is fewer than currently then the user is responsible for re-making
    /// the convolution channels or deleting invalidated convolution channels and
    /// make sure that no convolution channel tries to convolute a non-existent channel!**
    /// 
    /// For the above reason, the `Simulator` struct does not implement a function to
    /// reduce the number of channels, but rather asks to remake the whole `Lenia` instance
    /// when reducing the number of channels is wanted.
    fn set_channels(&mut self, num_channels: usize);
    /// Sets the number of convolution channels in the `Lenia` instance. 
    /// 
    /// * Any convolution channels
    /// that have an index larger than the new number of channels **will be dropped**. Conversely,
    /// no convolution channels get invalidated if the new number of convolution channels is
    /// greater than the previous number of convolution channels. 
    /// 
    /// * Any newly initialized convolution channels will have to have their kernels and
    /// growth functions added. By default all channels will use a weight of `0.0` for the new
    /// channels. 
    fn set_conv_channels(&mut self, num_conv_channels: usize);
    /// Sets the source channel for a convolution channel.
    fn set_source_channel(&mut self, conv_channel: usize, src_channel: usize);
    /// Sets the convolution kernel for a convolution channel.
    fn set_kernel(&mut self, kernel: ndarray::ArrayD<f64>, conv_channel: usize);
    /// Sets the growth function for a convolution channel.
    fn set_growth(&mut self, f: fn(f64, &[f64]) -> f64, growth_params: Vec<f64>, conv_channel: usize);
    /// Sets the weights for input into a channel from convolution channels for summing. 
    /// 
    /// * If the length of `new weights` is less than the number of convolution channels then
    /// the weight for the uncovered convolution channels defaults to `0.0`. 
    /// 
    /// * If the length of `new weights` is greater than the number of convolution channels then
    /// the excess weights will be disregarded, and their effect for the weighted average on the 
    /// channel is not taken into account.
    fn set_weights(&mut self, new_weights: &[f64], channel: usize);
    /// Sets the dt parameter of the `Lenia` instance. 
    fn set_dt(&mut self, new_dt: f64);
    /// Returns a reference to a channel's current data. 
    fn get_channel_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64>;
    /// Returns a mutable reference to a channel's current data.
    fn get_channel_as_mut_ref(&mut self, channel: usize) -> &mut ndarray::ArrayD<f64>;
    /// Returns a reference to the convolution result.
    fn get_convoluted_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<Complex<f64>>;
    /// Returns a reference to the field with growth function applied.
    fn get_grown_as_ref(&self, conv_channel: usize) -> &ndarray::ArrayD<f64>;
    /// Returns a reference to the results to be added to previous channel state. Lacks `dt` scaling.
    fn get_deltas_as_ref(&self, channel: usize) -> &ndarray::ArrayD<f64>;
    /// Returns the shape of the channels and convolution channels (as reference). 
    fn shape(&self) -> &[usize];
    /// Returns the current `dt` parameter of the `Lenia` instance.
    fn dt(&self) -> f64;
    /// Returns the number of channels in the `Lenia` instance.
    fn channels(&self) -> usize;
    /// Returns the number of convolution channels in the `Lenia` instance.
    fn conv_channels(&self) -> usize;
    /// Returns the weights of the specified channel.
    fn weights(&self, channel: usize) -> &[f64];
    /// Calculates the next state of the `Lenia` instance, and updates the data in channels accordingly.
    fn iterate(&mut self);
}


#[derive(Clone, Debug)]
/// The `Channel` struct is a wrapper for holding the data of a single channel in a 
/// `Lenia` simulation. The struct also implements functionality for applying the
/// weights of the convolution channels and summing up the final result in any
/// Lenia system more complex than the Standard Lenia. 
pub struct Channel {
    pub field: ndarray::ArrayD<f64>,
    pub weights: Vec<f64>,
    pub weight_sum_reciprocal: f64,
}


#[derive(Clone)]
/// The `ConvolutionChannel` struct holds relevant data for the convolution step of the
/// Lenia simulation. This includes the kernel, the intermittent convolution step, and the 
/// growth function.
pub struct ConvolutionChannel {
    pub input_channel: usize,
    pub field: ndarray::ArrayD<f64>,
    pub kernel: Kernel,
    pub growth: fn(f64, &[f64]) -> f64,
    pub growth_params: Vec<f64>,
}

impl fmt::Debug for ConvolutionChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConvolutionChannel")
         .field("input_channel", &self.input_channel)
         .field("field", &self.field)
         .field("kernel", &self.kernel)
         .field("growth", &"fn(f64, &[f64]) -> f64")
         .field("growth_params", &self.growth_params)
         .finish()
    }
}


#[derive(Clone, Debug)]
/// The `Kernel` struct holds the data of a specific kernel to be used for convolution in
/// the Lenia simulation. It also implements the necessary conversions to normalize a
/// kernel and prepare it for convolution using fast-fourier-transform. 
pub struct Kernel {
    pub base: ndarray::ArrayD<f64>,
    pub normalized: ndarray::ArrayD<f64>,
    pub shifted: ndarray::ArrayD<f64>,
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
        let shifted_and_fft = ndarray::ArrayD::from_elem(channel_shape, Complex::new(0.0, 0.0));
        
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
        let shifted_stored = shifted.clone();

        // Create the discrete-fourier-transformed representation of the kernel for fft-convolving. 
        let mut shifted_and_fft = shifted.mapv(|elem| {Complex::new(elem, 0.0)});

        let mut fft_instance = fft::PlannedFFTND::new(&channel_shape, false);
        fft_instance.transform(&mut shifted_and_fft);

        // Create the kernel
        Kernel{
            base: kernel,
            normalized: normalized_kernel,
            shifted: shifted_stored,
            transformed: shifted_and_fft,
        }
    }
}



