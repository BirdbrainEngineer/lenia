mod lenia_ca;
mod keyboardhandler;
use std::{sync::mpsc::channel, ops::IndexMut, thread::JoinHandle};

use ndarray::{ArrayD, IntoNdProducer};
use std::{ops::Range, thread, sync::{Arc, Mutex}};
use lenia_ca::{growth_functions, kernels, lenias::*, Channel};
use pixel_canvas::{Canvas, Color, input};
use probability::distribution;
use rayon::{prelude::*, iter::empty};
use rand::*;

// Make sure to compile as release and run the release version - error checks make the program slow otherwise.
// *If you have a beefy and modern cpu then 1280x720 should run at or close to 60fps with 1 channel and kernel 
// per physical core.
// *Reduce X_SIDE_LEN and Y_SIDE_LEN if the simulation feels sluggish.
// *SCALE magnifies the simulated board for render. Eg. if X and Y side lengths are
// 1000x500, and SCALE is 2, then the created window for rendering will be 2000x1000 pixels. If you make the 
// created window extremely large (larger than your display max resolution) then your OS and system will have a stroke. 

const X_SIDE_LEN: usize = 1280;
const Y_SIDE_LEN: usize = 720;
//const X_SIDE_LEN: usize = 1920/2;
//const Y_SIDE_LEN: usize = 1080/2;
const Z_SIDE_LEN: usize = 50;
const W_SIDE_LEN: usize = 50;
const SCALE: usize = 5;
const GAIN: f64 = 5000.0;
const MU_RANGE: Range<f64> = 0.1..0.6;
const KERNEL_RANGE: Range<f64> = 0.01..1.0;
const SIGMA_LOW: f64 = 0.01;
const SIGMA_HIGH: f64 = 0.2;
const GROWTH_LOW: f64 = 0.05;
const GROWTH_HIGH: f64 = 0.9;
const RADIUS_RANGE: Range<f64> = 0.5..1.0;

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let inv_scale = 1.0 / SCALE as f64;
    let mut simulating = false;
    let mut checking_kernel = true;
    let mut checking_transformed = false;
    let mut checking_deltas = false;
    let mut z_depth = Z_SIDE_LEN/2;
    let kernel_radius = 90;
    let mut kernel_z_depth = kernel_radius / 2;
    let channel_shape = vec![X_SIDE_LEN, Y_SIDE_LEN];
    let view_axis = [0, 1];
    let fill_channels = vec![true, true, false, false, false, false, false, false];
    let randomness_scalers = vec![0.7, 0.618, 0.618, 0.618, 0.618, 1.0, 1.0, 1.0];
    let randomness_sizes = vec![22, 25, 120, 120, 120, 50, 50, 50];
    let randomness_patches = vec![7, 4, 20, 20, 20, 1, 1, 1, 1];
    let discrete_randomness = false;
    let view_channels: Vec<i32> = vec![1, 1, 2];
    let dt = 1.0;
    let render_rate = 1;
    let skip_frames = 1;
    let capture_deltas = true;

    let num_channels: usize = 3;
    let num_convolutions: usize = 12;
    let sigma_base = 0.15;
    let max_rings = 2.0;
    let maximum_adjust = 0.25;
    let mut rules: Vec<Vec<f64>> = Vec::new();

    let mut lenia_simulator = lenia_ca::Simulator::<ExpandedLenia>::new(&channel_shape);
    lenia_simulator.set_kernel(kernels::smoothlife(15, 2, 0.8), 0);
    lenia_simulator.set_growth_function(growth_functions::smooth_life, vec![0.27, 0.36, 0.26, 0.46], 0);
    lenia_simulator.set_channels(2);
    lenia_simulator.set_convolution_channels(3);
    lenia_simulator.set_convolution_channel_source(0, 0);
    lenia_simulator.set_convolution_channel_source(1, 1);
    lenia_simulator.set_convolution_channel_source(2, 1);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.15, 0.021], 0);
    lenia_simulator.set_growth_function(growth_functions::polynomial, vec![0.25, 0.03, 4.0], 1);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.07, 0.025], 2);
    lenia_simulator.set_kernel(kernels::gaussian_donut_2d(13, 1.0/6.7), 0);
    lenia_simulator.set_kernel(kernels::polynomial_nd(24, 2, &vec![4.0, 1.0, 0.33]), 1);
    lenia_simulator.set_kernel(kernels::multi_gaussian_donut_2d(20, &vec![0.75], &vec![1.0], &vec![0.1]), 2);
    lenia_simulator.set_weights(0, &vec![0.6, 0.0, 0.4]);
    lenia_simulator.set_weights(1, &vec![0.0, 1.0, 0.0]);
    lenia_simulator.set_dt(0.1);

    let mut rules = new_lenia(&mut lenia_simulator, kernel_radius, num_channels, num_convolutions, sigma_base, max_rings, dt);
    
    let mut frames: Vec<ndarray::Array2<f64>> = Vec::new();
    for i_ in 0..3 {
        frames.push(ndarray::Array2::zeros([lenia_simulator.shape()[view_axis[0]], lenia_simulator.shape()[view_axis[1]]]));
    }
    let empty_frame = frames[0].clone();
    let mut frame_counter = 0;
    let mut capture_counter = 0;
    let mut frame_index = 0;
    let mut continuous_capture = false;
    let mut kernel_index: usize = 0;
    
    let canvas = Canvas::new(X_SIDE_LEN * SCALE, Y_SIDE_LEN * SCALE)
        .title("Lenia")
        .state(keyboardhandler::KeyboardState::new())
        .input(keyboardhandler::KeyboardState::handle_input);


    // More important key bindings
    // k - toggles between viewing the kernels or simulation
    // r - randomly seeds the simulation board based on constants earlier in the code
    // s - toggles continuous simulating
    // i - performs a single iteration of the simulation
    //
    // If using the code unchanged then the following are also important
    // n - Changes the currently used rulesets completely
    // m - Uses the currently set ruleset as basis and tweaks the ruleset slightly for a slightly different result
    // , - Permanently tweaks the rulesets slightly from the currently used ruleset
    canvas.render(move |keyboardstate, image| {

        if simulating && continuous_capture {
            if capture_counter >= skip_frames {
                let mut density_handles: Vec<JoinHandle<()>> = Vec::new();
                let mut delta_handles: Vec<JoinHandle<()>> = Vec::new();
                for i in 0..lenia_simulator.channels() {
                    density_handles.push(lenia_ca::export_frame_as_png(
                        lenia_ca::BitDepth::Eight,
                        lenia_simulator.get_channel_as_ref(i), 
                        format!("{}_{}", i, frame_index).as_str(), 
                        &r"./output/density"));
                    if capture_deltas {
                        delta_handles.push(lenia_ca::export_frame_as_png(
                            lenia_ca::BitDepth::Eight,
                            lenia_simulator.get_deltas_as_ref(i), 
                            format!("{}_{}", i, frame_index).as_str(), 
                            &r"./output/delta"));
                    }
                }
                
                frame_index += 1;
                for handle in density_handles {
                    handle.join().unwrap();
                }
                if capture_deltas {
                    for handle in delta_handles {
                        handle.join().unwrap();
                    }
                }
                capture_counter = 0;
            }
            else {
                capture_counter += 1;
            }
        }

        match keyboardstate.character {
            'r' => {
                let mut data;
                for i in 0..lenia_simulator.channels() {
                    if fill_channels[i] {
                        data = lenia_ca::seeders::random_hyperspheres(
                            lenia_simulator.shape(), 
                            randomness_sizes[i], 
                            randomness_patches[i], 
                            randomness_scalers[i], 
                            discrete_randomness,
                        );
                        lenia_simulator.fill_channel(&data, i);
                    }
                    else {
                        lenia_simulator.fill_channel(&lenia_ca::seeders::constant(lenia_simulator.shape(), 0.0), i);
                    }
                }
            }
            ',' => {
                cumulate_lenia(&mut lenia_simulator, &mut rules, maximum_adjust, kernel_radius, sigma_base);
            }
            'n' => { 
                rules = new_lenia(&mut lenia_simulator, kernel_radius, num_channels, num_convolutions, sigma_base, max_rings, dt);
            }
            'm' => { adjust_lenia(&mut lenia_simulator, &rules, maximum_adjust, kernel_radius, sigma_base); }
            'i' => { lenia_simulator.iterate(); }
            's' => { simulating = !simulating; }
            'k' => { checking_kernel = !checking_kernel; }
            'x' => { kernel_index += if kernel_index + 1 == lenia_simulator.convolution_channels() {0} else {1};}
            'z' => { kernel_index -= if kernel_index == 0 {0} else {1};}
            '+' => { lenia_simulator.set_dt(lenia_simulator.dt() * 1.25); }
            '-' => { lenia_simulator.set_dt(lenia_simulator.dt() * 1.0/1.25); }
            '8' => { if z_depth != (Z_SIDE_LEN-1) { z_depth += 1 }; if kernel_z_depth != (kernel_radius-1) { kernel_z_depth += 1 }; println!("Depth: {}", z_depth); }
            '2' => { if z_depth != 0 { z_depth -= 1 }; if kernel_z_depth != 0 { kernel_z_depth -= 1 }; println!("Depth: {}", z_depth); }
            't' => { checking_transformed = !checking_transformed; }
            'c' => { lenia_ca::export_frame_as_png(
                    lenia_ca::BitDepth::Eight,
                    lenia_simulator.get_channel_as_ref(0), 
                    &"0", 
                    &r"./output/density/");
                lenia_ca::export_frame_as_png(
                    lenia_ca::BitDepth::Eight,
                    lenia_simulator.get_deltas_as_ref(0), 
                    &"0", 
                    &r"./output/delta/"); 
                if channel_shape.len() == 4 && lenia_simulator.channels() > 1 {
                    lenia_ca::export_frame_as_png(
                        lenia_ca::BitDepth::Eight,
                        lenia_simulator.get_channel_as_ref(1), 
                        &"0", 
                        &r"./output/density2");
                    lenia_ca::export_frame_as_png(
                        lenia_ca::BitDepth::Eight,
                        lenia_simulator.get_deltas_as_ref(1), 
                        &"0", 
                        &r"./output/delta2/"); 
                }
            }
            'f' => {
                continuous_capture = !continuous_capture;
                capture_counter = 0;
            }
            'l' => {
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
        let kernel_3d_clone = lenia_simulator.get_kernel_as_ref(kernel_index);
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
                        else if kernel_3d_clone.transformed.shape().len() == 3 {
                            let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                            pixel.r = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1, z_depth]].re * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                            pixel.g = 127;
                            //pixel.g = (kernel_3d_clone.shifted[[coords.0, coords.1, 0]] * 255.0 * GAIN) as u8;
                            pixel.b = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1, z_depth]].im * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                        }
                        else {
                            let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                            pixel.r = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1, z_depth, z_depth]].re * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                            pixel.g = 127;
                            //pixel.g = (kernel_3d_clone.shifted[[coords.0, coords.1, 0]] * 255.0 * GAIN) as u8;
                            pixel.b = (127 + ((kernel_3d_clone.transformed[[coords.0, coords.1, z_depth, z_depth]].im * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                        }
                    }
                }
            }
            else {
                for (y, row) in image.chunks_mut(width).enumerate() {
                    row.into_par_iter().enumerate().for_each(|(x, pixel)| {
                        let x_index = ((x as f64 / (SCALE * X_SIDE_LEN) as f64) * kernel_3d_clone.base.shape()[0] as f64) as usize;
                        let y_index = ((y as f64 / (SCALE * Y_SIDE_LEN) as f64) * kernel_3d_clone.base.shape()[1] as f64) as usize;
                        if kernel_3d_clone.base.shape().len() == 2 {
                            pixel.r = (kernel_3d_clone.base[[x_index, y_index]] * 255.0) as u8;
                        } 
                        else if kernel_3d_clone.transformed.shape().len() == 3 {
                            pixel.r = (kernel_3d_clone.base[[x_index, y_index, kernel_z_depth]] * 255.0) as u8;
                        }
                        else {
                            pixel.r = (kernel_3d_clone.base[[x_index, y_index, kernel_z_depth, kernel_z_depth]] * 255.0) as u8;
                        }
                        pixel.g = pixel.r;
                        pixel.b = pixel.r;
                    });
                }
            }
        }
        else {
            let channels = view_channels.clone();
            frames.par_iter_mut().enumerate().for_each(|(i, frame)| {
                let negative;
                let mut view_channel = if channels[i] < 0 { 
                        negative = true; 
                        channels[i].abs() as usize 
                    } 
                    else { 
                        negative = false; channels[i].abs() as usize 
                    };
                view_channel = if view_channel > lenia_simulator.channels() { 
                        0
                    } 
                    else { 
                        view_channel 
                    };
                if view_channel == 0 {
                    *frame = empty_frame.clone();
                }
                else {
                    if checking_deltas {
                        lenia_ca::get_frame(lenia_simulator.get_deltas_as_ref(view_channel - 1), frame, &view_axis, &[0, 0, z_depth, z_depth]);
                        frame.par_iter_mut().for_each(|a| { *a = (*a * 0.5) + 0.5 });
                    }
                    else {
                        lenia_ca::get_frame(lenia_simulator.get_channel_as_ref(view_channel - 1), frame, &view_axis, &[0, 0, z_depth, z_depth]);
                        if negative { frame.par_iter_mut().for_each(|a| { *a *= -1.0; }) }
                    }
                }
            });

            for (y, row) in image.chunks_mut(width).enumerate() {
                row.par_iter_mut().enumerate().for_each(|(x, pixel)| {
                    let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                    unsafe {
                        pixel.r = (frames[0].uget([coords.0, coords.1]) * 255.0) as u8;
                        pixel.g = ((frames[1].uget([coords.0, coords.1]) * 255.0) as u16 + 0 as u16).clamp(0, 255) as u8;
                        pixel.b = ((frames[2].uget([coords.0, coords.1]) * 255.0) as u16 + 0 as u16).clamp(0, 255) as u8;
                    }
                });
            }
            if simulating {
                frame_counter += 1;
                if frame_counter >= render_rate {
                    lenia_simulator.iterate();
                    frame_counter = 0;
                }  
            }
        }
    });
}

fn new_lenia(
    lenia_simulator: &mut lenia_ca::Simulator<ExpandedLenia>,
    kernel_radius: usize, 
    num_channels: usize, 
    num_convolutions: usize, 
    sigma_base: f64, 
    max_rings: f64, 
    dt: f64
) 
-> Vec<Vec<f64>> {
    let mut rulevec: Vec<Vec<f64>> = Vec::with_capacity(10);

    let mut growth_mu: Vec<f64> = Vec::new();
    let mut growth_sigma: Vec<f64> = Vec::new();
    let mut growth_amount: Vec<f64> = Vec::new();
    let mut kernel_radii: Vec<f64> = Vec::new();
    let mut kernels0: Vec<f64> = Vec::new();
    let mut kernels1: Vec<f64> = Vec::new();
    let mut kernels2: Vec<f64> = Vec::new();
    let mut kernel_rings: Vec<f64> = Vec::new();
    let mut sources: Vec<f64> = Vec::new();
    let mut destinations: Vec<f64> = Vec::new();

    let mut init_randomizer = rand::thread_rng();

    for i in 0..num_convolutions {
        growth_mu.push(init_randomizer.gen_range(MU_RANGE));
        growth_sigma.push(init_randomizer.gen_range(SIGMA_LOW..(SIGMA_HIGH - (0.01 * (num_channels as f64)))));
        growth_amount.push(init_randomizer.gen_range(GROWTH_LOW..(GROWTH_HIGH - (0.01 * (num_channels as f64)))));
        kernel_radii.push(init_randomizer.gen_range(RADIUS_RANGE));
        kernels0.push(init_randomizer.gen_range(KERNEL_RANGE));
        kernels1.push(init_randomizer.gen_range(KERNEL_RANGE));
        kernels2.push(init_randomizer.gen_range(KERNEL_RANGE));
        kernel_rings.push((init_randomizer.gen_range(0.0..=max_rings) as f64).ceil());
        if i < num_channels {
            sources.push(i as f64);
            destinations.push(i as f64);
        }
        else {
            sources.push(init_randomizer.gen_range(0..num_channels) as f64);
            destinations.push(init_randomizer.gen_range(0..num_channels) as f64);
        }
        let chooser = init_randomizer.gen_range(1..=(kernel_rings[i] as usize));
        match chooser {
            1 => {kernels0[i] = 1.0;}
            2 => {kernels1[i] = 1.0;}
            3 => {kernels2[i] = 1.0;}
            _ => {panic!("Something went wrong!");}
        }
    }

    println!("New ruleset:");
    println!("growth_mu = {}", format_vec_f64(&growth_mu, 5));
    rulevec.push(growth_mu.clone());
    println!("growth_sigma = {}", format_vec_f64(&growth_sigma, 5));
    rulevec.push(growth_sigma.clone());
    println!("growth_amount = {}", format_vec_f64(&growth_amount, 5));
    rulevec.push(growth_amount.clone());
    println!("kernel_radii = {}", format_vec_f64(&kernel_radii, 3));
    rulevec.push(kernel_radii.clone());
    println!("kernels0 = {}", format_vec_f64(&kernels0, 3));
    rulevec.push(kernels0.clone());
    println!("kernels1 = {}", format_vec_f64(&kernels1, 3));
    rulevec.push(kernels1.clone());
    println!("kernels2 = {}", format_vec_f64(&kernels2, 3));
    rulevec.push(kernels2.clone());
    println!("kernel_rings = {}", format_vec_f64(&kernel_rings, 1));
    rulevec.push(kernel_rings.clone());
    println!("sources = {}", format_vec_f64(&sources, 1));
    rulevec.push(sources.clone());
    println!("destinations = {}", format_vec_f64(&destinations, 1));
    rulevec.push(destinations.clone());
    
    lenia_simulator.set_channels(num_channels);
    lenia_simulator.set_convolution_channels(num_convolutions);
    for i in 0..lenia_simulator.convolution_channels() {
        lenia_simulator.set_convolution_channel_source(i, sources[i] as usize);
        //let kernel_parameters = if kernel_rings[i] == 1 {vec![4.0, kernels0[i]]} else {vec![4.0, kernels0[i], kernels1[i]]};
        let kernel_parameters: (Vec<f64>, Vec<f64>, Vec<f64>) = if kernel_rings[i] == 1.0 {
                (vec![0.5 * kernel_radii[i]], vec![kernels0[i]], vec![sigma_base * kernel_radii[i]])
            } 
            else if kernel_rings[i] == 2.0 {
                (vec![0.25 * kernel_radii[i], 0.75 * kernel_radii[i]], vec![kernels0[i], kernels1[i]], vec![(sigma_base * 0.5) * kernel_radii[i]; 2])
            }
            else {
                (vec![(0.333333 * 0.5) * kernel_radii[i], 0.5 * kernel_radii[i], (0.666666 + (0.333333 * 0.5)) * kernel_radii[i]], 
                    vec![kernels0[i], kernels1[i], kernels2[i]], 
                    vec![(sigma_base * 0.333333) * kernel_radii[i]; 3])
            };
        lenia_simulator.set_kernel(
            /*kernels::polynomial_nd(
                (kernel_radii[i] * kernel_radius as f64) as usize, 
                channel_shape.len(), 
                &kernel_parameters
            )*/
            kernels::multi_gaussian_donut_2d(kernel_radius, &kernel_parameters.0, &kernel_parameters.1, &kernel_parameters.2), 
            i
        );
        lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![growth_mu[i], growth_sigma[i] * 0.98], i);
    }
    for i in 0..lenia_simulator.channels() {
        let mut weights: Vec<f64> = Vec::with_capacity(lenia_simulator.convolution_channels());
        for j in 0..lenia_simulator.convolution_channels() {
            weights.push(if destinations[j] as usize == i {growth_amount[i]} else {0.0});
        }
        lenia_simulator.set_weights(i, &weights);
    }
    lenia_simulator.set_dt(dt);

    rulevec
}

fn cumulate_lenia(
    lenia_simulator: &mut lenia_ca::Simulator<ExpandedLenia>, 
    rules: &mut Vec<Vec<f64>>, 
    maximum_adjust: f64, 
    kernel_radius: usize, 
    sigma_base: f64
) {
    let mut randomizer = rand::thread_rng();
    
    for i in 0..lenia_simulator.convolution_channels() {
        rules[0][i] += (randomizer.gen_range(MU_RANGE) - rules[0][i]) * maximum_adjust;
        rules[1][i] += (randomizer.gen_range(SIGMA_LOW..SIGMA_HIGH) - rules[1][i]) * maximum_adjust;
        rules[2][i] += (randomizer.gen_range(GROWTH_LOW..GROWTH_HIGH) - rules[2][i]) * maximum_adjust;
        rules[3][i] += (randomizer.gen_range(RADIUS_RANGE) - rules[3][i]) * maximum_adjust;
        rules[4][i] += if rules[4][i] == 1.0 { 0.0 } else { ( randomizer.gen_range(KERNEL_RANGE) - rules[4][i]) * maximum_adjust };
        rules[5][i] += if rules[5][i] == 1.0 { 0.0 } else { ( randomizer.gen_range(KERNEL_RANGE) - rules[5][i]) * maximum_adjust };
        rules[6][i] += if rules[6][i] == 1.0 { 0.0 } else { ( randomizer.gen_range(KERNEL_RANGE) - rules[6][i]) * maximum_adjust };
    }

    println!("Cumulated ruleset:");
    println!("growth_mu = {}", format_vec_f64(&rules[0], 5));
    println!("growth_sigma = {}", format_vec_f64(&rules[1], 5));
    println!("growth_amount = {}", format_vec_f64(&rules[2], 5));
    println!("kernel_radii = {}", format_vec_f64(&rules[3], 3));
    println!("kernels0 = {}", format_vec_f64(&rules[4], 3));
    println!("kernels1 = {}", format_vec_f64(&rules[5], 3));
    println!("kernels2 = {}", format_vec_f64(&rules[6], 3));
    println!("kernel_rings = {}", format_vec_f64(&rules[7], 1));
    println!("sources = {}", format_vec_f64(&rules[8], 1));
    println!("destinations = {}", format_vec_f64(&rules[9], 1));

    for i in 0..lenia_simulator.convolution_channels() {
        lenia_simulator.set_convolution_channel_source(i, rules[8][i] as usize);
        //let kernel_parameters = if kernel_rings[i] == 1 {vec![4.0, kernels0[i]]} else {vec![4.0, kernels0[i], kernels1[i]]};
        let kernel_parameters: (Vec<f64>, Vec<f64>, Vec<f64>) = if rules[7][i] == 1.0 {
                (vec![0.5 * rules[3][i]], vec![rules[4][i]], vec![sigma_base * rules[3][i]])
            } 
            else if rules[7][i] == 2.0 {
                (vec![0.25 * rules[3][i], 0.75 * rules[3][i]], vec![rules[4][i], rules[5][i]], vec![(sigma_base * 0.5) * rules[3][i]; 2])
            }
            else {
                (vec![(0.333333 * 0.5) * rules[3][i], 0.5 * rules[3][i], (0.666666 + (0.333333 * 0.5)) * rules[3][i]], 
                    vec![rules[4][i], rules[5][i], rules[6][i]], 
                    vec![(sigma_base * 0.333333) * rules[3][i]; 3])
            };
        lenia_simulator.set_kernel(
            /*kernels::polynomial_nd(
                (kernel_radii[i] * kernel_radius as f64) as usize, 
                channel_shape.len(), 
                &kernel_parameters
            )*/
            kernels::multi_gaussian_donut_2d(kernel_radius, &kernel_parameters.0, &kernel_parameters.1, &kernel_parameters.2), 
            i
        );
        lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![rules[0][i], rules[1][i] * 0.98], i);
    }
    for i in 0..lenia_simulator.channels() {
        let mut weights: Vec<f64> = Vec::with_capacity(lenia_simulator.convolution_channels());
        for j in 0..lenia_simulator.convolution_channels() {
            weights.push(if rules[9][j] as usize == i {rules[2][i]} else {0.0});
        }
        lenia_simulator.set_weights(i, &weights);
    }
}

fn adjust_lenia(lenia_simulator: &mut lenia_ca::Simulator<ExpandedLenia>, rulevec: &Vec<Vec<f64>>, maximum_adjust: f64, kernel_radius: usize, sigma_base: f64) {
    let mut randomizer = rand::thread_rng();
    
    let mut growth_mu: Vec<f64> = rulevec[0].clone();
    let mut growth_sigma: Vec<f64> = rulevec[1].clone();
    let mut growth_amount: Vec<f64> = rulevec[2].clone();
    let mut kernel_radii: Vec<f64> = rulevec[3].clone();
    let mut kernels0: Vec<f64> = rulevec[4].clone();
    let mut kernels1: Vec<f64> = rulevec[5].clone();
    let mut kernels2: Vec<f64> = rulevec[6].clone();
    let kernel_rings: Vec<f64> = rulevec[7].clone();
    let sources: Vec<f64> = rulevec[8].clone();
    let destinations: Vec<f64> = rulevec[9].clone();
    
    for i in 0..lenia_simulator.convolution_channels() {
        growth_mu[i] += growth_mu[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust);
        growth_sigma[i] += growth_sigma[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust);
        growth_amount[i] += growth_amount[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust);
        kernel_radii[i] += kernel_radii[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust);
        kernels0[i] += if kernels0[i] == 1.0 { 0.0 } else { kernels0[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust) };
        kernels1[i] += if kernels1[i] == 1.0 { 0.0 } else { kernels1[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust) };
        kernels2[i] += if kernels2[i] == 1.0 { 0.0 } else { kernels2[i] * randomizer.gen_range(-maximum_adjust..=maximum_adjust) };
    }

    println!("Adjusted ruleset:");
    println!("growth_mu = {}", format_vec_f64(&growth_mu, 5));
    println!("growth_sigma = {}", format_vec_f64(&growth_sigma, 5));
    println!("growth_amount = {}", format_vec_f64(&growth_amount, 5));
    println!("kernel_radii = {}", format_vec_f64(&kernel_radii, 3));
    println!("kernels0 = {}", format_vec_f64(&kernels0, 3));
    println!("kernels1 = {}", format_vec_f64(&kernels1, 3));
    println!("kernels2 = {}", format_vec_f64(&kernels2, 3));
    println!("kernel_rings = {}", format_vec_f64(&kernel_rings, 1));
    println!("sources = {}", format_vec_f64(&sources, 1));
    println!("destinations = {}", format_vec_f64(&destinations, 1));

    for i in 0..lenia_simulator.convolution_channels() {
        lenia_simulator.set_convolution_channel_source(i, sources[i] as usize);
        //let kernel_parameters = if kernel_rings[i] == 1 {vec![4.0, kernels0[i]]} else {vec![4.0, kernels0[i], kernels1[i]]};
        let kernel_parameters: (Vec<f64>, Vec<f64>, Vec<f64>) = if kernel_rings[i] == 1.0 {
                (vec![0.5 * kernel_radii[i]], vec![kernels0[i]], vec![sigma_base * kernel_radii[i]])
            } 
            else if kernel_rings[i] == 2.0 {
                (vec![0.25 * kernel_radii[i], 0.75 * kernel_radii[i]], vec![kernels0[i], kernels1[i]], vec![(sigma_base * 0.5) * kernel_radii[i]; 2])
            }
            else {
                (vec![(0.333333 * 0.5) * kernel_radii[i], 0.5 * kernel_radii[i], (0.666666 + (0.333333 * 0.5)) * kernel_radii[i]], 
                    vec![kernels0[i], kernels1[i], kernels2[i]], 
                    vec![(sigma_base * 0.333333) * kernel_radii[i]; 3])
            };
        lenia_simulator.set_kernel(
            /*kernels::polynomial_nd(
                (kernel_radii[i] * kernel_radius as f64) as usize, 
                channel_shape.len(), 
                &kernel_parameters
            )*/
            kernels::multi_gaussian_donut_2d(kernel_radius, &kernel_parameters.0, &kernel_parameters.1, &kernel_parameters.2), 
            i
        );
        lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![growth_mu[i], growth_sigma[i] * 0.98], i);
    }
    for i in 0..lenia_simulator.channels() {
        let mut weights: Vec<f64> = Vec::with_capacity(lenia_simulator.convolution_channels());
        for j in 0..lenia_simulator.convolution_channels() {
            weights.push(if destinations[j] as usize == i {growth_amount[i]} else {0.0});
        }
        lenia_simulator.set_weights(i, &weights);
    }
}

fn format_vec_f64(v: &Vec<f64>, decimal_digits: usize) -> String {
    let formatted_nums: Vec<String> = v.iter().map(|num| format!("{:.1$}", num, decimal_digits)).collect();
    let mut numbers = formatted_nums.join(", ");
    numbers.insert(0, '[');
    numbers + "]"
}
