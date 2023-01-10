mod lenia;
mod fft;
mod keyboardhandler;
use ndarray::{self, IxDyn};
use num_complex::Complex;
use pixel_canvas::{Canvas, Color, input};

const SIDE_LEN: usize = 150;
const SCALE: usize = 4;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");
    let inv_scale = 1.0 / SCALE as f64;

    let mut lenia_simulator = lenia::Simulator::new(lenia::StandardLenia::new(&[SIDE_LEN, SIDE_LEN]));
    lenia_simulator.fill_channel(
        &lenia::utils::initializations::random_hypercubic(&[SIDE_LEN, SIDE_LEN], 33, 0.4, false), 
        0
    );
    //lenia_simulator.set_dt(0.05);
    /*let mut lenia_simulator = lenia::Simulator::new(lenia::StandardLenia::new(&[SIDE_LEN, SIDE_LEN]));
    lenia_simulator.set_growth_function(lenia::utils::growth_functions::game_of_life, 0);
    lenia_simulator.set_kernel(lenia::utils::kernels::game_of_life(), 0);
    lenia_simulator.set_dt(1.0);
    lenia_simulator.fill_channel(
        &lenia::utils::initializations::random_cube(&[SIDE_LEN, SIDE_LEN], 8, true), 
        0,
    );*/

    let canvas = Canvas::new(SIDE_LEN * SCALE, SIDE_LEN * SCALE)
        .title("Lenia")
        .state(keyboardhandler::KeyboardState::new())
        .input(keyboardhandler::KeyboardState::handle_input);

    canvas.render(move |keyboardstate, image| {

        match keyboardstate.character {
            'r' => {
                /*lenia_simulator.fill_channel(
                    &lenia::utils::initializations::random_gaussian_2d(&[SIDE_LEN, SIDE_LEN], 30.0, true), 
                    0
                );*/
                lenia_simulator.fill_channel(
                    &lenia::utils::initializations::random_hypercubic_patches(&[SIDE_LEN, SIDE_LEN], SIDE_LEN / 6, 15, 0.4, false), 
                    0
                );
            }
            _ => {}
        }
        keyboardstate.character = '\0';

        let frame = lenia_simulator.get_frame(0, &[0, 1], &[0, 0]);

        let width = image.width() as usize;
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                pixel.r = (frame[[(x as f64 * inv_scale) as usize, (y as f64 * inv_scale) as usize]] * 255.0) as u8;
                pixel.g = pixel.r;
                pixel.b = pixel.r;
            }
        }
        lenia_simulator.iterate();
    });
}
