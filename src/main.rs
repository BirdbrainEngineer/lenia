mod lenia;
mod fft;
mod keyboardhandler;
use ndarray::{self, IxDyn};
use lenia::utils::{growth_functions, kernels};
use pixel_canvas::{Canvas, Color, input};

const SIDE_LEN: usize = 150;
const SCALE: usize = 4;
const STEPS: usize = 100;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let inv_scale = 1.0 / SCALE as f64;
    let mut simulating = false;

    let mut lenia_simulator = lenia::Simulator::new(lenia::StandardLenia::new(&[SIDE_LEN, SIDE_LEN]));
    lenia_simulator.fill_channel(
        &lenia::utils::initializations::random_hypercubic(&[SIDE_LEN, SIDE_LEN], 33, 0.4, false), 
        0
    );
    lenia_simulator.set_growth_function(growth_functions::standard_lenia, vec![0.25, 0.03], 0);
    //lenia_simulator.set_kernel(kernels::gaussian_donut_2d(48, 1.0/3.35), 0);
    let kernel_diameter = 80;
    let kernel_for_render = kernels::multi_gaussian_donut_2d(
        kernel_diameter, 
        &vec![0.25, 0.75], 
        &vec![0.95, 0.45], 
        &vec![0.07, 0.07]
    );
    
    let kernel_into_sim = kernel_for_render.clone();
    lenia_simulator.set_kernel(kernel_into_sim, 0);

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
                    &lenia::utils::initializations::random_hypercubic_patches(&[SIDE_LEN, SIDE_LEN], SIDE_LEN / 4, 6, 0.45, false), 
                    0
                );
            }
            's' => { simulating = true; }
            _ => {}
        }
        keyboardstate.character = '\0';

        let frame = lenia_simulator.get_frame(0, &[0, 1], &[0, 0]);

        let width = image.width() as usize;
        if !simulating {
            for (y, row) in image.chunks_mut(width).enumerate() {
                for (x, pixel) in row.iter_mut().enumerate() {
                    let x_index = ((x as f64 / (SCALE * SIDE_LEN) as f64) * kernel_diameter as f64) as usize;
                    let y_index = ((y as f64 / (SCALE * SIDE_LEN) as f64) * kernel_diameter as f64) as usize;
                    pixel.r = (kernel_for_render[[x_index, y_index]] * 255.0) as u8;
                    pixel.g = pixel.r;
                    pixel.b = pixel.r;
                }
            }
        }
        else {
            for (y, row) in image.chunks_mut(width).enumerate() {
                for (x, pixel) in row.iter_mut().enumerate() {
                    pixel.r = (frame[[(x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize]] * 255.0) as u8;
                    pixel.g = pixel.r;
                    pixel.b = pixel.r;
                }
            }
            lenia_simulator.iterate();
        }
    });

}
