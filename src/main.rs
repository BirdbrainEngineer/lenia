mod lenia_ca;
mod keyboardhandler;
use std::sync::mpsc::channel;

use ndarray::{ArrayD};
use lenia_ca::{growth_functions, kernels, lenias::*, Channel, ChannelMode};
use pixel_canvas::{Canvas, Color, input};
use probability::distribution;
use rayon::{prelude::*, iter::empty};

const X_SIDE_LEN: usize = 1920/2;
const Y_SIDE_LEN: usize = 1080/2;
const Z_SIDE_LEN: usize = 150;
const SCALE: usize = 2;
const STEPS: usize = 2;
const GAIN: f64 = 5000.0;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let inv_scale = 1.0 / SCALE as f64;
    let randomness_sizer = 0.5;
    let mut simulating = false;
    let mut checking_kernel = true;
    let mut checking_transformed = false;
    let mut checking_deltas = false;
    let mut z_depth = Z_SIDE_LEN/2;
    let kernel_diameter = 80;
    let mut kernel_z_depth = kernel_diameter / 2;
    let channel_shape = vec![X_SIDE_LEN, Y_SIDE_LEN];
    let view_axis = [0, 1];
    let fill_channels = vec![true, true, true, false, false, false, false, false];
    let randomness_scalers = vec![0.75, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let randomness_sizes = vec![110, 100, 140, 50, 50, 50, 50, 50];
    let randomness_patches = vec![7, 7, 7, 1, 1, 1, 1, 1, 1];
    let view_channels: Vec<i32> = vec![1, 2, 3];
    let dt = 0.04;

    let mut lenia_simulator = lenia_ca::Simulator::<ExpandedLenia>::new(&channel_shape);

    //let input_kernel = lenia_ca::load_from_png("/home/clanga/Documents/Programming/lenia_simulator/inputs/kernel_ac.png");

    lenia_simulator.set_channels(1);
    lenia_simulator.set_convolution_channels(1);
    

    /*lenia_simulator.set_channels(4);
    lenia_simulator.set_channel_mode(0, lenia_ca::ChannelMode::Positive);
    lenia_simulator.set_channel_mode(1, lenia_ca::ChannelMode::Positive);
    lenia_simulator.set_channel_mode(2, lenia_ca::ChannelMode::Positive);
    lenia_simulator.set_channel_mode(3, lenia_ca::ChannelMode::Positive);

    lenia_simulator.set_convolution_channels(4);
    lenia_simulator.set_convolution_channel_source(0, 0);
    lenia_simulator.set_convolution_channel_source(1, 1);
    lenia_simulator.set_convolution_channel_source(2, 2);
    lenia_simulator.set_convolution_channel_source(3, 3);


    lenia_simulator.set_weights(0, &vec![1.0, 0.25, 0.5, 0.2]);
    lenia_simulator.set_weights(1, &vec![0.5, 1.0, 0.25, 0.2]);
    lenia_simulator.set_weights(2, &vec![0.25, 0.5, 1.0, 0.2]);
    lenia_simulator.set_weights(3, &vec![1.0, 1.0, 1.0, 0.0]);

    let mut growth_top = lenia_ca::growth_functions::Distributions::multi_gaussian(
        &vec![0.12, 0.27], 
        &vec![0.007, 0.033], 
        &vec![1.0, 1.0], 
        10000
    );
    let mut growth_main = Vec::new();
    /*for num in growth_top.iter().rev() {
        growth_main.push(-*num);
    }*/
    for i in (0..10000).rev() {
        growth_main.push(-(i as f64 / 10000.0));
    }
    growth_main.append(&mut growth_top);

    let mut diffuse_growth: Vec<f64> = Vec::new();
    for i in -5000..5000 {
        diffuse_growth.push(-(i as f64 / 5000.0));
    }
    /*for (i, num) in growth_main.iter().enumerate() {
        if i % 20 == 0 {println!("{} : {}", (i as f64 / 10000.0) - 1.0, *num);}
    }*/
    let pass = lenia_ca::kernels::pass(lenia_simulator.shape());
    let inverse_distance = lenia_ca::kernels::inverse_distance_bump(kernel_diameter / 4, lenia_simulator.shape().len(), 1.0, 2.0);
    let donut = lenia_ca::kernels::gaussian_donut_2d(kernel_diameter, 1.0/6.7);
    let multidonut = kernels::multi_gaussian_donut_nd(
        kernel_diameter, 
        2, 
        &vec![0.25, 0.5, 0.75], 
        &vec![1.0, 0.5, 0.75], 
        &vec![1.0/19.0, 1.0/19.0, 1.0/19.0],
    );
    let multidonut2 = kernels::multi_gaussian_donut_nd(
        kernel_diameter + 40, 
        2, 
        &vec![0.15, 0.27, 0.44, 0.6, 0.75], 
        &vec![0.25, 0.88, 0.5, 0.7, 1.0], 
        &vec![1.0/17.0, 1.0/28.0, 1.0/27.0, 1.0/32.0, 1.0/23.0],
    );
    let multidonut3 = kernels::multi_gaussian_donut_nd(
        kernel_diameter - 20, 
        2, 
        &vec![0.15, 0.44, 0.75], 
        &vec![0.5, 1.0, 0.5], 
        &vec![1.0/19.0, 1.0/17.0, 1.0/17.0],
    );
    let diffuse = lenia_ca::kernels::multi_gaussian_donut_2d(kernel_diameter / 2, &vec![0.0], &vec![1.0], &vec![0.4]);
    let donut_propagation = lenia_ca::kernels::gaussian_donut_2d(kernel_diameter / 4, 1.0/6.7);
    lenia_simulator.set_kernel(multidonut.clone(), 0);
    lenia_simulator.set_kernel(multidonut3.clone(), 1);
    lenia_simulator.set_kernel(multidonut2.clone(), 2);
    lenia_simulator.set_kernel(pass, 3);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::precalculated_linear_fullrange, growth_main.clone(), 0);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.22, 0.036], 1);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::multimodal_normal, vec![0.08, 0.0022, 0.37, 0.078, 0.61, 0.02], 2);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::pass, vec![-1.0], 3);
    */     

    /*lenia_simulator.set_convolution_channels(7);
    lenia_simulator.set_channels(3);
    lenia_simulator.set_convolution_channel_source(1, 0);
    lenia_simulator.set_convolution_channel_source(2, 1);
    lenia_simulator.set_convolution_channel_source(3, 1);
    lenia_simulator.set_convolution_channel_source(4, 0);
    lenia_simulator.set_convolution_channel_source(5, 1);
    lenia_simulator.set_convolution_channel_source(6, 2);
    lenia_simulator.set_weights(0, &vec![0.85, 0.1, 0.1, 0.3, 0.6, 0.2, 0.5]);
    lenia_simulator.set_weights(1, &vec![0.1, 0.2, 0.85, 0.35, 0.3, 0.45, 0.3]);
    lenia_simulator.set_weights(2, &vec![0.1, 0.2, 0.1, 0.35, 0.8, 0.8, 1.5]);

    lenia_simulator.set_kernel(lenia_ca::kernels::gaussian_donut_2d(kernel_diameter, 1.0/6.3), 6);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia_inverted, vec![0.135, 0.025], 6);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.11, 0.0295], 1);
    lenia_simulator.set_kernel(lenia_ca::kernels::gaussian_donut_2d(kernel_diameter + (kernel_diameter / 2), 1.0/5.5), 1);

    let diffuse = lenia_ca::kernels::multi_gaussian_donut_2d(kernel_diameter * 2, &vec![0.0], &vec![1.0], &vec![0.3]);
    lenia_simulator.set_kernel(diffuse.clone(), 4);
    lenia_simulator.set_kernel(diffuse.clone(), 5);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::multimodal_normal, vec![0.11, 0.011, 0.5, 0.07], 4);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::multimodal_normal, vec![0.2, 0.014, 0.5, 0.07], 5);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.157, 0.0237], 2);
    lenia_simulator.set_kernel(lenia_ca::kernels::multi_gaussian_donut_2d(
        kernel_diameter, 
        &vec![0.2, 0.68, 0.72, 0.8], 
        &vec![0.62, 1.0, -0.7, 0.6],
        &vec![1.0/11.2, 1.0/8.8, 1.0/18.7, 1.0/12.9]), 
    2);

    let exponential_kernel_2 = lenia_ca::kernels::exponential_donuts(
        kernel_diameter + 34, 
        channel_shape.len(), 
        &vec![0.2, 0.32, 0.56, 0.8],
        &vec![1.0, -0.75, 0.9, -0.3],
        &vec![8.5, 14.4, 8.1, 5.6]
    );
    lenia_simulator.set_kernel(exponential_kernel_2.clone(), 3);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::multimodal_normal, vec![0.12, 0.0191, 0.27, 0.0293, 0.71, 0.0189], 3);

    let exponential_kernel = lenia_ca::kernels::exponential_donuts(
        kernel_diameter - 8, 
        channel_shape.len(), 
        &vec![0.2, 0.66, 0.56],
        &vec![1.0, 0.75, -0.7],
        &vec![10.5, 8.1, 15.2]
    );
    lenia_simulator.set_kernel(exponential_kernel.clone(), 0);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.189, 0.0298], 0);
    */

    /*let mut distribution: Vec<f64> = vec![];
    let distribution_len: usize = 10000;
    let distribution_delta = 1.0 / distribution_len as f64;
    for i in 0..distribution_len {
        distribution.push(2.0 * lenia_ca::sample_normal(i as f64 * distribution_delta, 0.15, 0.017) - 1.0);
    }*/

    /*lenia_simulator.set_channels(3);
    lenia_simulator.set_convolution_channels(5); 
    lenia_simulator.set_convolution_channel_source(1, 1);
    lenia_simulator.set_weights(0, &[0.7, 0.3, -0.5, -0.3, 0.2]);
    lenia_simulator.set_weights(1, &[0.3, 0.6, -0.6, 0.1, -0.4]);
    lenia_simulator.set_weights(2, &[0.4, 0.4, 0.3, -0.1, 0.1]);

    let distribution = lenia_ca::growth_functions::Distributions::geometric_normal(
        0.12, 0.031, 1.0, 1.0565, 0.85, -0.93, 7, 10000
    );

    lenia_simulator.set_growth_function(growth_functions::precalculated_linear, distribution, 0);
    let kernel_0 = kernels::multi_gaussian_donut_nd(
        kernel_diameter, 
        2, 
        &vec![0.75, 0.5, 0.25], 
        &vec![0.47, 1.0, 0.89], 
        &vec![1.0/12.3, 1.0/15.5, 1.0/10.7],
    );
    lenia_simulator.set_kernel(kernel_0.clone(), 0);

    let distribution2 = lenia_ca::growth_functions::Distributions::geometric_normal(
        0.32, 0.021, 0.65, 1.08, 0.9, 1.058, 5, 10000
    );

    lenia_simulator.set_growth_function(growth_functions::precalculated_linear, distribution2, 1);
    let kernel_1 = kernels::multi_gaussian_donut_nd(
        kernel_diameter - 10, 
        2, 
        &vec![0.7, 0.1], 
        &vec![0.95, 0.4], 
        &vec![1.0/12.3, 1.0/8.5],
    );
    lenia_simulator.set_kernel(kernel_1.clone(), 1);

    let exponential_kernel = lenia_ca::kernels::exponential_donuts(
        kernel_diameter, 
        channel_shape.len(), 
        &vec![0.5],
        &vec![1.0],
        &vec![6.0]
    );
    lenia_simulator.set_kernel(exponential_kernel.clone(), 2);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia_inverted, vec![0.22, 0.088], 2);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.3, 0.035], 3);
    let kernel_2 = kernels::multi_gaussian_donut_nd(
        kernel_diameter -18, 
        2, 
        &vec![0.8, 0.4], 
        &vec![0.5, 1.0], 
        &vec![1.0/16.3, 1.0/5.5],
    );
    lenia_simulator.set_kernel(kernel_2.clone(), 3);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.21, 0.017], 4);
    let kernel_3 = kernels::multi_gaussian_donut_nd(
        kernel_diameter + 14, 
        2, 
        &vec![0.4, 0.55, 0.45], 
        &vec![0.95, 0.7, -0.8], 
        &vec![1.0/12.3, 1.0/3.5, 1.0/18.0],
    );
    lenia_simulator.set_kernel(kernel_3.clone(), 4);
    */
    lenia_simulator.set_dt(dt);
    /*

    lenia_simulator.set_channels(3);
    lenia_simulator.set_convolution_channels(3); 
    lenia_simulator.set_convolution_channel_source(1, 1);
    //lenia_simulator.set_convolution_channel_source(2, 2);
    lenia_simulator.set_weights(0, &[1.0, 0.33, 0.5]);
    lenia_simulator.set_weights(1, &[0.33, 1.0, 0.5]);
    lenia_simulator.set_weights(2, &[0.2, 0.2, 0.6]);

    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.185, 0.02], 1);
    let kernel_1 = kernels::multi_gaussian_donut_2d(
       kernel_diameter + 12, 
       &vec![0.25, 0.75], 
       &vec![0.97, 0.45], 
       &vec![0.15, 0.2]
    );
    lenia_simulator.set_kernel(kernel_1.clone(), 1);

    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.14, 0.02], 2);
    let kernel_2 = kernels::multi_gaussian_donut_2d(
       kernel_diameter - 12, 
       &vec![0.0], 
       &vec![1.0], 
       &vec![0.3]
    );
    lenia_simulator.set_kernel(kernel_2.clone(), 2);
    
    lenia_simulator.set_dt(0.05);
    */
    //let kernel_3d_clone = lenia_ca::Kernel::from(lenia_ca::kernels::gaussian_donut_2d(kernel_diameter, 1.0/6.7), &channel_shape);
    let kernel_3d_clone = lenia_ca::Kernel::from(multidonut2.clone(), &channel_shape);
    
    // Extended lenia test 2
    /*lenia_simulator.set_channels(3);
    lenia_simulator.set_convolution_channels(4);
    lenia_simulator.set_convolution_channel_source(0, 0);
    lenia_simulator.set_convolution_channel_source(1, 1);
    lenia_simulator.set_convolution_channel_source(2, 2);
    lenia_simulator.set_convolution_channel_source(3, 0);
    lenia_simulator.set_weights(0, &[0.6, 0.05, 0.35, -0.2]);
    lenia_simulator.set_weights(1, &[0.4, 0.7, 0.25, -0.2]);
    lenia_simulator.set_weights(2, &[0.1, 0.3, 0.6, 0.35]);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.13, 0.011], 0);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.18, 0.0299], 1);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.112, 0.02], 2);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.2, 0.027], 3);
    lenia_simulator.set_kernel(kernels::gaussian_donut_2d(46, 1.0/10.5), 2);
    let first_conv_kernel = kernels::multi_gaussian_donut_2d(
        kernel_diameter + (kernel_diameter as f64 * 0.15) as usize, 
        &vec![0.3, 0.65], 
        &vec![0.95, 0.4], 
        &vec![0.09, 0.155]
    );
    let second_conv_kernel = kernels::multi_gaussian_donut_2d(
        kernel_diameter - 8, 
        &vec![0.33, 0.7, 0.55], 
        &vec![0.45, 0.9, 0.7], 
        &vec![0.15, 0.06, 0.09]
    );
    let third_conv_kernel = kernels::multi_gaussian_donut_2d(
        kernel_diameter, 
        &vec![0.2, 0.4, 0.8], 
        &vec![0.8, 0.4, 0.2], 
        &vec![0.1, 0.1, 0.1]
    );
    lenia_simulator.set_kernel(first_conv_kernel, 0);
    lenia_simulator.set_kernel(second_conv_kernel, 1);
    lenia_simulator.set_kernel(third_conv_kernel, 3);
    */

    //Orbium unicaudatus (using StandardLenia2D)
    //kernel_for_render = kernels::gaussian_donut_2d(48, 1.0/6.7);

    //Tricircum torquens (using StandardLenia2D)

    /*kernel_for_render = kernels::multi_gaussian_donut_nd(
        kernel_diameter, 
        2, 
        &vec![0.25, 0.75], 
        &vec![0.97, 0.45], 
        &vec![0.075, 0.08]
    );*/

    //lenia_simulator.set_kernel(kernel_into_sim, 0);

    lenia_simulator.fill_channel(
        &lenia_ca::seeders::random_hypercubic_patches(
            lenia_simulator.shape(), 
            kernel_diameter, 
            1, 
            0.45, 
            false
        ), 
        0
    );

    /*for i in 0..STEPS {
        lenia_simulator.iterate();
    }*/
    /*lenia_ca::export_frame_as_png(
        lenia_ca::BitDepth::Eight,
        lenia_simulator.get_channel_as_ref(0), 
        &"0", 
        &r"./output/density"
    ).join().unwrap();
    */
    let mut frames: Vec<ndarray::Array2<f64>> = Vec::with_capacity(3);
    for _ in 0..3 {
        frames.push(ndarray::Array2::zeros([lenia_simulator.shape()[view_axis[0]], lenia_simulator.shape()[view_axis[1]]]));
    }
    let empty_frame = frames[0].clone();
    
    let canvas = Canvas::new(X_SIDE_LEN * SCALE, Y_SIDE_LEN * SCALE)
        .title("Lenia")
        .state(keyboardhandler::KeyboardState::new())
        .input(keyboardhandler::KeyboardState::handle_input);

    canvas.render(move |keyboardstate, image| {

        match keyboardstate.character {
            'r' => {
                let mut data;
                for i in 0..lenia_simulator.channels() {
                    if fill_channels[i] {
                        data = lenia_ca::seeders::random_hypercubic_patches(
                            lenia_simulator.shape(), 
                            randomness_sizes[i], 
                            randomness_patches[i], 
                            randomness_scalers[i], 
                            false
                        );
                        lenia_simulator.fill_channel(&data, i);
                    }
                    else {
                        lenia_simulator.fill_channel(&lenia_ca::seeders::constant(lenia_simulator.shape(), 0.0), i);
                    }
                }
            }
            'n' => { lenia_simulator.iterate(); }
            's' => { simulating = !simulating; }
            'k' => { checking_kernel = !checking_kernel; }
            '+' => { lenia_simulator.set_dt(lenia_simulator.dt() * 1.25); }
            '-' => { lenia_simulator.set_dt(lenia_simulator.dt() * 1.0/1.25); }
            '8' => { if z_depth != (Z_SIDE_LEN-1) { z_depth += 1 }; if kernel_z_depth != (kernel_diameter-1) { kernel_z_depth += 1 }; println!("Depth: {}", z_depth); }
            '2' => { if z_depth != 0 { z_depth -= 1 }; if kernel_z_depth != 0 { kernel_z_depth -= 1 }; println!("Depth: {}", z_depth); }
            't' => { checking_transformed = !checking_transformed; }
            'c' => { lenia_ca::export_frame_as_png(
                    lenia_ca::BitDepth::Eight,
                    lenia_simulator.get_channel_as_ref(0), 
                    &"0", 
                    &r"./output/density");
                lenia_ca::export_frame_as_png(
                    lenia_ca::BitDepth::Eight,
                    lenia_simulator.get_channel_as_ref(0), 
                    &"0", 
                    &r"./output/delta/"); 
            }
            'm' => {
                for i in 0..lenia_simulator.channels() {
                    let mut new_channel = lenia_simulator.get_channel_as_ref(i).clone();
                    new_channel.invert_axis(ndarray::Axis(0));
                    lenia_simulator.fill_channel(&new_channel, i);
                }
            }
            'd' => {
                checking_deltas = !checking_deltas;
            }
            _ => {}
        }
        keyboardstate.character = '\0';

        let width = image.width() as usize;
        if checking_kernel {
            if checking_transformed {
                for (y, row) in image.chunks_mut(width).enumerate() {
                    for (x, pixel) in row.iter_mut().enumerate() {
                        if kernel_3d_clone.transformed.shape().len() == 2 {
                            let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                            pixel.r = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1]].re * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                            pixel.g = 127;//(kernel_3d_clone.shifted[[coords.0, coords.1]] * 255.0) as u8;
                            pixel.b = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1]].im * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                        }
                        else {
                            let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                            pixel.r = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1, z_depth]].re * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                            pixel.g = 127;
                            //pixel.g = (kernel_3d_clone.shifted[[coords.0, coords.1, 0]] * 255.0 * GAIN) as u8;
                            pixel.b = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1, z_depth]].im * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                        }
                    }
                }
            }
            else {
                for (y, row) in image.chunks_mut(width).enumerate() {
                    for (x, pixel) in row.iter_mut().enumerate() {
                        let x_index = ((x as f64 / (SCALE * X_SIDE_LEN) as f64) * kernel_3d_clone.base.shape()[0] as f64) as usize;
                        let y_index = ((y as f64 / (SCALE * Y_SIDE_LEN) as f64) * kernel_3d_clone.base.shape()[1] as f64) as usize;
                        if kernel_3d_clone.base.shape().len() == 2 {
                            pixel.r = (kernel_3d_clone.base[[x_index, y_index]] * 255.0) as u8;
                        } 
                        else {
                            pixel.r = (kernel_3d_clone.base[[x_index, y_index, kernel_z_depth]] * 255.0) as u8;
                        }
                        pixel.g = pixel.r;
                        pixel.b = pixel.r;
                    }
                }
            }
        }
        else {
            for (i, channel) in view_channels.iter().enumerate() {
                let negative;
                let mut view_channel = if *channel < 0 { 
                        negative = true; 
                        channel.abs() as usize 
                    } 
                    else { 
                        negative = false; channel.abs() as usize 
                    };
                view_channel = if view_channel > lenia_simulator.channels() { 
                        0
                    } 
                    else { 
                        view_channel 
                    };
                if view_channel == 0 {
                    frames[i] = empty_frame.clone();
                }
                else {
                    if checking_deltas {
                        lenia_ca::get_frame(&mut frames[i], lenia_simulator.get_deltas_as_ref(view_channel - 1), &view_axis, &[0, 0, z_depth]);
                        frames[i].par_iter_mut().for_each(|a| { *a = (*a * 0.5) + 0.5 });
                    }
                    else {
                        lenia_ca::get_frame(&mut frames[i], lenia_simulator.get_channel_as_ref(view_channel - 1), &view_axis, &[0, 0, z_depth]);
                        if negative { frames[i].par_iter_mut().for_each(|a| { *a *= -1.0; }) }
                    }
                }
            }

            for (y, row) in image.chunks_mut(width).enumerate() {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                    unsafe {
                        pixel.r = (frames[0].uget([coords.0, coords.1]) * 255.0) as u8;
                        pixel.g = ((frames[1].uget([coords.0, coords.1]) * 255.0) as u16 + 0 as u16).clamp(0, 255) as u8;
                        pixel.b = ((frames[2].uget([coords.0, coords.1]) * 255.0) as u16 + 0 as u16).clamp(0, 255) as u8;
                    }
                }
            }
            if simulating {
                lenia_simulator.iterate();
            }
        }
    });
    
}
