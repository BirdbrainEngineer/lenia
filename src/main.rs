mod lenia_ca;
mod keyboardhandler;
use std::sync::mpsc::channel;

use ndarray::{ArrayD};
use lenia_ca::{growth_functions, kernels, lenias::*};
use pixel_canvas::{Canvas, Color, input};
use probability::distribution;

const X_SIDE_LEN: usize = 1920/4;
const Y_SIDE_LEN: usize = 1080/4;
const Z_SIDE_LEN: usize = 150;
const SCALE: usize = 4;
const STEPS: usize = 2;
const GAIN: f64 = 5000.0;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let inv_scale = 1.0 / SCALE as f64;
    let mut simulating = false;
    let mut checking_kernel = true;
    let mut checking_transformed = false;
    let mut z_depth = Z_SIDE_LEN/2;
    let kernel_diameter = 80;
    let mut kernel_z_depth = kernel_diameter / 2;
    let channel_shape = vec![X_SIDE_LEN, Y_SIDE_LEN];

    let mut lenia_simulator = lenia_ca::Simulator::<ExpandedLenia>::new(&channel_shape);

    /*let mut distribution: Vec<f64> = vec![];
    let distribution_len: usize = 10000;
    let distribution_delta = 1.0 / distribution_len as f64;
    for i in 0..distribution_len {
        distribution.push(2.0 * lenia_ca::sample_normal(i as f64 * distribution_delta, 0.15, 0.017) - 1.0);
    }*/

    lenia_simulator.set_channels(3);
    lenia_simulator.set_convolution_channels(5); 
    lenia_simulator.set_convolution_channel_source(1, 1);
    lenia_simulator.set_weights(0, &[0.5, 0.3, -0.5, 0.3, 0.2]);
    lenia_simulator.set_weights(1, &[0.3, 0.6, -0.6, 0.1, 0.4]);
    lenia_simulator.set_weights(2, &[0.4, 0.4, 0.3, 0.1, 0.1]);

    let distribution = lenia_ca::growth_functions::Distributions::geometric_normal(
        0.12, 0.027, 1.0, 1.0565, 0.85, -0.93, 7, 10000
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
        0.32, 0.026, 0.65, 1.08, 0.9, 1.058, 4, 10000
    );

    lenia_simulator.set_growth_function(growth_functions::precalculated_linear, distribution2, 1);
    let kernel_1 = kernels::multi_gaussian_donut_nd(
        kernel_diameter - 11, 
        2, 
        &vec![0.7, 0.1], 
        &vec![0.95, 0.4], 
        &vec![1.0/12.3, 1.0/8.5],
    );
    lenia_simulator.set_kernel(kernel_1.clone(), 1);

    lenia_simulator.set_kernel(lenia_ca::kernels::gaussian_donut_2d(kernel_diameter - 15, 1.0/6.7), 2);
    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia_inverted, vec![0.2, 0.08], 2);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.3, 0.031], 3);
    let kernel_2 = kernels::multi_gaussian_donut_nd(
        kernel_diameter + 15, 
        2, 
        &vec![0.8, 0.4], 
        &vec![0.5, 1.0], 
        &vec![1.0/16.3, 1.0/5.5],
    );
    lenia_simulator.set_kernel(kernel_2.clone(), 3);

    lenia_simulator.set_growth_function(lenia_ca::growth_functions::standard_lenia, vec![0.21, 0.023], 4);
    let kernel_3 = kernels::multi_gaussian_donut_nd(
        kernel_diameter - 30, 
        2, 
        &vec![0.4, 0.55, 0.45], 
        &vec![0.95, 0.7, -0.8], 
        &vec![1.0/12.3, 1.0/3.5, 1.0/18.0],
    );
    lenia_simulator.set_kernel(kernel_3.clone(), 4);

    lenia_simulator.set_dt(0.1);
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
    let kernel_3d_clone = lenia_ca::Kernel::from(lenia_ca::kernels::gaussian_donut_2d(kernel_diameter, 1.0/6.7), &channel_shape);
    //let kernel_3d_clone = lenia_ca::Kernel::from(kernel_0.clone(), &channel_shape);
    
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
    
    let canvas = Canvas::new(X_SIDE_LEN * SCALE, Y_SIDE_LEN * SCALE)
        .title("Lenia")
        .state(keyboardhandler::KeyboardState::new())
        .input(keyboardhandler::KeyboardState::handle_input);

    canvas.render(move |keyboardstate, image| {

        match keyboardstate.character {
            'r' => {
                let num_patches = X_SIDE_LEN / kernel_diameter;
                let scaler = 0.75;
                lenia_simulator.fill_channel(
                    &lenia_ca::seeders::random_hypercubic_patches(
                        lenia_simulator.shape(), 
                        kernel_diameter, 
                        num_patches, 
                        scaler, 
                        false
                    ), 
                    0
                );
                if lenia_simulator.channels() >= 2 {
                    lenia_simulator.fill_channel(
                        &lenia_ca::seeders::random_hypercubic_patches(
                            lenia_simulator.shape(), 
                            kernel_diameter, 
                            num_patches, 
                            scaler, 
                            false
                        ), 
                        1
                    );
                }
                if lenia_simulator.channels() >= 3 {
                    lenia_simulator.fill_channel(
                        &lenia_ca::seeders::random_hypercubic_patches(
                            lenia_simulator.shape(), 
                            kernel_diameter, 
                            num_patches, 
                            scaler, 
                            false
                        ), 
                        2
                    );
                }
            }
            'n' => { lenia_simulator.iterate(); }
            's' => { simulating = !simulating; }
            'k' => { checking_kernel = !checking_kernel; }
            '+' => { lenia_simulator.set_dt(lenia_simulator.dt() * 1.5); }
            '-' => { lenia_simulator.set_dt(lenia_simulator.dt() * 0.75); }
            'u' => { if z_depth != (Z_SIDE_LEN-1) { z_depth += 1 }; if kernel_z_depth != (kernel_diameter-1) { kernel_z_depth += 1 }; println!("Depth: {}", z_depth); }
            'd' => { if z_depth != 0 { z_depth -= 1 }; if kernel_z_depth != 0 { kernel_z_depth -= 1 }; println!("Depth: {}", z_depth); }
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
            _ => {}
        }
        keyboardstate.character = '\0';

        //let r_frame = lenia_simulator.get_data_as_ref(0);
        //let g_frame = lenia_simulator.get_data_as_ref(1);
        //let b_frame = lenia_simulator.get_data_as_ref(2);

        //let r_frame = lenia_simulator.get_data_as_f64(0);
        //let g_frame = lenia_simulator.get_data_as_f64(1);
        //let b_frame = lenia_simulator.get_data_as_f64(2);

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
            let r_frame = lenia_simulator.get_frame(0, &[0, 1], &[0, 0, z_depth]);
            let mut g_frame = ndarray::Array2::zeros([1, 1]);
            if lenia_simulator.channels() >= 2 {g_frame = lenia_simulator.get_frame(1, &[0, 1], &[0, 0]);}
            let mut b_frame = ndarray::Array2::zeros([1, 1]);
            if lenia_simulator.channels() >= 3 {b_frame = lenia_simulator.get_frame(2, &[0, 1], &[0, 0]);}

            for (y, row) in image.chunks_mut(width).enumerate() {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                    unsafe {
                        pixel.r = (r_frame.uget([coords.0, coords.1]) * 255.0) as u8;
                        if lenia_simulator.channels() >= 2 {pixel.g = (g_frame.uget([coords.0, coords.1]) * 255.0) as u8;}
                        else {pixel.g = pixel.r;}
                        if lenia_simulator.channels() >= 3 {pixel.b = (b_frame.uget([coords.0, coords.1]) * 255.0) as u8;}
                        else {pixel.b = pixel.r;}
                    }
                    //pixel.r = (r_frame[[coords.0, coords.1]] * 255.0) as u8;
                    //pixel.g = (g_frame[[coords.0, coords.1]] * 255.0) as u8;
                    //pixel.b = (b_frame[[coords.0, coords.1]] * 255.0) as u8;
                }
            }
            if simulating {
                lenia_simulator.iterate();
            }
        }
    });
    
}
