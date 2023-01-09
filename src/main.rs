mod lenia;
mod fft;
mod keyboardhandler;
use ndarray::{self, IxDyn};
use num_complex::Complex;
use pixel_canvas::{Canvas, Color, input};

const SIDE_LEN: usize = 200;
const SCALE: usize = 4;


fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let lenia_simulator = lenia::Simulator::new(lenia::StandardLenia::new());

    let canvas = Canvas::new(SIDE_LEN * SCALE, SIDE_LEN * SCALE)
        .title("Lenia")
        .state(keyboardhandler::KeyboardState::new())
        .input(keyboardhandler::KeyboardState::handle_input);

    canvas.render(move |keyboardstate, image| {

        match keyboardstate.character {
            'r' => {

            }
            _ => {}
        }
        keyboardstate.character = '\0';

        let width = image.width() as usize;
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                pixel.r = ((lenia::utils::growth_functions::standard_lenia(x as f64 / (SIDE_LEN * SCALE) as f64) + 1.0) * 120.0) as u8;
                pixel.g = pixel.r;
                pixel.b = pixel.r;
                
            }
        }
    });
}
