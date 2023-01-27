mod lenia_ca;
mod keyboardhandler;
use ndarray::{ArrayD};
use lenia_ca::{growth_functions, kernels, lenias::*};
use pixel_canvas::{Canvas, Color, input};

const SIDE_LEN: usize = 192/2;
const SCALE: usize = 4;
const STEPS: usize = 5000;
const GAIN: f64 = 5000.0;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let inv_scale = 1.0 / SCALE as f64;
    let mut simulating = false;
    let mut checking_transformed = false;
    let mut z_depth = SIDE_LEN/2;
    let kernel_diameter = 30;
    let mut kernel_z_depth = kernel_diameter / 2;
    let channel_shape = vec![SIDE_LEN, SIDE_LEN, SIDE_LEN];

    let mut lenia_simulator = lenia_ca::Simulator::<ExtendedLenia>::new(&channel_shape);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.15, 0.07], 0);
    let kernel_3d = kernels::multi_gaussian_donut_nd(
        kernel_diameter, 
        3, 
        &vec![0.5], 
        &vec![1.0], 
        &vec![1.0/6.7]
    );
    //let kernel_3d = kernels::gaussian_donut_nd(kernel_diameter, 3, 1.0/10.5);
    let kernel_3d_clone = lenia_ca::Kernel::from(kernel_3d.clone(), &channel_shape);

    lenia_simulator.set_kernel(kernel_3d, 0);

    // Test extended lenia 1 - creates an evolving blob, that eventually evolves into a glider
    //lenia_simulator.set_channels(2);
    //lenia_simulator.set_convolution_channels(2);
    //lenia_simulator.set_convolution_channel_source(1, 1);
    //lenia_simulator.set_weights(0, &[0.55, 0.45]);
    //lenia_simulator.set_weights(1, &[0.3, 0.7]);
    //lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.185, 0.025], 1);
    //let second_conv_kernel = kernels::multi_gaussian_donut_2d(
    //    kernel_diameter, 
    //    &vec![0.25, 0.75], 
    //    &vec![0.97, 0.45], 
    //    &vec![0.15, 0.2]
    //);
    //lenia_simulator.set_kernel(second_conv_kernel, 1);

    /* 
    // Extended lenia test 2
    lenia_simulator.set_channels(2);
    lenia_simulator.set_convolution_channels(3);
    lenia_simulator.set_convolution_channel_source(1, 1);
    lenia_simulator.set_convolution_channel_source(2, 1);
    lenia_simulator.set_weights(0, &[0.7, 0.0, 0.3]);
    lenia_simulator.set_weights(1, &[0.25, 0.7, 0.05]);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.13, 0.013], 0);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.18, 0.023], 1);
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.112, 0.05], 2);
    lenia_simulator.set_kernel(kernels::gaussian_donut_2d(48, 1.0/6.7), 2);
    let first_conv_kernel = kernels::multi_gaussian_donut_2d(
        kernel_diameter + (kernel_diameter as f64 * 0.15) as usize, 
        &vec![0.2, 0.65], 
        &vec![0.95, 0.4], 
        &vec![0.07, 0.255]
    );
    let second_conv_kernel = kernels::multi_gaussian_donut_2d(
        kernel_diameter, 
        &vec![0.25, 0.75], 
        &vec![0.97, 0.45], 
        &vec![0.15, 0.2]
    );
    lenia_simulator.set_kernel(first_conv_kernel, 0);
    lenia_simulator.set_kernel(second_conv_kernel, 1);
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

    /*for i in 0..STEPS {
        lenia_simulator.iterate();
    }*/

    
    let canvas = Canvas::new(SIDE_LEN * SCALE, SIDE_LEN * SCALE)
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
                        6, 
                        0.45, 
                        false
                    ), 
                    0
                );
                /*lenia_simulator.fill_channel(
                    &lenia_ca::seeders::random_hypercubic_patches(
                        lenia_simulator.shape(), 
                        kernel_diameter, 
                        6, 
                        0.45, 
                        false
                    ), 
                    1
                );
                lenia_simulator.fill_channel(
                    &lenia_ca::seeders::random_hypercubic_patches(
                        &[SIDE_LEN, SIDE_LEN], 
                        SIDE_LEN / 3, 
                        6, 
                        0.45, 
                        false
                    ), 
                    2
                );*/
            }
            's' => { simulating = true; }
            'k' => { simulating = false; }
            '+' => { lenia_simulator.set_dt(lenia_simulator.dt() * 1.5); }
            '-' => { lenia_simulator.set_dt(lenia_simulator.dt() * 0.75); }
            'u' => { z_depth += 1; kernel_z_depth += 1; }
            'd' => { z_depth -= 1; kernel_z_depth -= 1; }
            't' => { checking_transformed = !checking_transformed; }
            _ => {}
        }
        keyboardstate.character = '\0';

        //let r_frame = lenia_simulator.get_data_as_ref(0);
        //let g_frame = lenia_simulator.get_data_as_ref(1);
        //let b_frame = lenia_simulator.get_data_as_ref(2);

        //let r_frame = lenia_simulator.get_data_as_f64(0);
        //let g_frame = lenia_simulator.get_data_as_f64(1);
        //let b_frame = lenia_simulator.get_data_as_f64(2);

        let r_frame = lenia_simulator.get_frame(0, &[0, 1], &[0, 0, z_depth]);
        //let g_frame = lenia_simulator.get_frame(1, &[0, 1], &[0, 0]);
        //let b_frame = lenia_simulator.get_frame(2, &[0, 1], &[0, 0]);

        let width = image.width() as usize;
        if !simulating {
            if checking_transformed {
                for (y, row) in image.chunks_mut(width).enumerate() {
                    for (x, pixel) in row.iter_mut().enumerate() {
                        if kernel_3d_clone.transformed.shape().len() == 2 {
                            let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                            pixel.r = 0;//(127 + ((kernel_3d_clone.transformed[[coords.0, coords.1]].re * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
                            pixel.g = (kernel_3d_clone.shifted[[coords.0, coords.1]] * 255.0) as u8;//127;
                            pixel.b = 0;//(127 + ((kernel_3d_clone.transformed[[coords.0, coords.1]].im * 127.0 * GAIN).clamp(-127.0, 127.0) as i32)) as u8;
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
                        let x_index = ((x as f64 / (SCALE * SIDE_LEN) as f64) * kernel_diameter as f64) as usize;
                        let y_index = ((y as f64 / (SCALE * SIDE_LEN) as f64) * kernel_diameter as f64) as usize;
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
            for (y, row) in image.chunks_mut(width).enumerate() {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let coords = ((x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize);
                    unsafe {
                        pixel.r = (r_frame.uget([coords.0, coords.1]) * 255.0) as u8;
                        pixel.g = pixel.r;
                        pixel.b = pixel.r;
                        //pixel.g = (g_frame.uget([coords.0, coords.1]) * 255.0) as u8;
                        //pixel.b = (b_frame.uget([coords.0, coords.1]).re * 255.0) as u8;
                        //pixel.b = ((pixel.r as u32 + pixel.g as u32) / 2) as u8;
                    }
                    //pixel.r = (r_frame[[coords.0, coords.1]] * 255.0) as u8;
                    //pixel.g = (g_frame[[coords.0, coords.1]] * 255.0) as u8;
                    //pixel.b = (b_frame[[coords.0, coords.1]] * 255.0) as u8;
                }
            }
            lenia_simulator.iterate();
        }
    });

}
