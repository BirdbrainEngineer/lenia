mod lenia_ca;
mod keyboardhandler;
use ndarray::{ArrayD};
use lenia_ca::{growth_functions, kernels, lenias::*};
use pixel_canvas::{Canvas, Color, input};

const X_SIDE_LEN: usize = 150;
const Y_SIDE_LEN: usize = 150;
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
    let kernel_diameter = 32;
    let mut kernel_z_depth = kernel_diameter / 2;
    let channel_shape = vec![X_SIDE_LEN, Y_SIDE_LEN, Z_SIDE_LEN];

    let mut lenia_simulator = lenia_ca::Simulator::<ExpandedLenia>::new(&channel_shape);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.13, 0.0171], 0);
    let kernel_3d = kernels::multi_gaussian_donut_nd(
        kernel_diameter, 
        3, 
        &vec![0.75, 0.37], 
        &vec![0.45, 0.95], 
        &vec![1.0/12.3, 1.0/10.7],
    );
    //lenia_ca::export_frame_as_png(lenia_ca::BitDepth::Sixteen, &kernel_3d, "hiqualitykernel", "./output/");
    //let loaded_kernel = lenia_ca::load_from_png("./output/hiqualitykernel.png");
    //lenia_ca::export_frame_as_png(lenia_ca::BitDepth::Eight, &loaded_kernel, "kernel", "./output/");
    //let kernel_3d_clone = lenia_ca::Kernel::from(kernel_3d.clone(), &channel_shape);
    let kernel_3d_clone = lenia_ca::Kernel::from(kernel_3d.clone(), &channel_shape);
    //let kernel_3d = loaded_kernel;
    lenia_simulator.set_kernel(kernel_3d, 0);
    lenia_simulator.set_dt(0.05);

    // //Test extended lenia 1 - creates an evolving blob, that eventually evolves into a sporadic glider
    // // 2d channel shape
    // lenia_simulator.set_channels(2);
    // lenia_simulator.set_convolution_channels(2);
    // lenia_simulator.set_convolution_channel_source(1, 1);
    // lenia_simulator.set_weights(0, &[0.55, 0.45]);
    // lenia_simulator.set_weights(1, &[0.3, 0.7]);
    // lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.185, 0.025], 1);
    // let second_conv_kernel = kernels::multi_gaussian_donut_2d(
    //    kernel_diameter, 
    //    &vec![0.25, 0.75], 
    //    &vec![0.97, 0.45], 
    //    &vec![0.15, 0.2]
    // );
    // lenia_simulator.set_kernel(second_conv_kernel, 1);

    
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

    for i in 0..STEPS {
        lenia_simulator.iterate();
    }
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
                lenia_simulator.fill_channel(
                    &lenia_ca::seeders::random_hypercubic_patches(
                        lenia_simulator.shape(), 
                        kernel_diameter, 
                        1, 
                        0.45, 
                        false
                    ), 
                    0
                );/*
                lenia_simulator.fill_channel(
                    &lenia_ca::seeders::random_hypercubic_patches(
                        lenia_simulator.shape(), 
                        kernel_diameter, 
                        20, 
                        0.45, 
                        false
                    ), 
                    1
                );
                lenia_simulator.fill_channel(
                    &lenia_ca::seeders::random_hypercubic_patches(
                        lenia_simulator.shape(), 
                        kernel_diameter, 
                        20, 
                        0.45, 
                        false
                    ), 
                    2
                );*/
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
            //let g_frame = lenia_simulator.get_frame(1, &[0, 1], &[0, 0]);
            //let b_frame = lenia_simulator.get_frame(2, &[0, 1], &[0, 0]);

            for (y, row) in image.chunks_mut(width).enumerate() {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                    unsafe {
                        pixel.r = (r_frame.uget([coords.0, coords.1]) * 255.0) as u8;
                        pixel.g = pixel.r;
                        pixel.b = pixel.r;
                        //pixel.g = (g_frame.uget([coords.0, coords.1]) * 255.0) as u8;
                        //pixel.b = (b_frame.uget([coords.0, coords.1]) * 255.0) as u8;
                        //pixel.b = ((pixel.r as u32 + pixel.g as u32) / 2) as u8;
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
