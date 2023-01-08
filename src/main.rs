mod lenia;
mod fft;
mod keyboardhandler;
use ndarray::{self, IxDyn};
use num_complex::Complex;
use pixel_canvas::{Canvas, Color, input};

const SIDE_LEN: usize = 100;
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

        let index = SIDE_LEN / 4;

        let width = image.width() as usize;
        for (y, row) in image.chunks_mut(width).enumerate() {
            for (x, pixel) in row.iter_mut().enumerate() {
                pixel.r = (lenia_simulator.kernel.shifted[[x / SCALE, y / SCALE]] * 200000.0) as u8;
                pixel.g = pixel.r;
                pixel.b = pixel.r;
            }
        }
    });

    // let mut test_data = ndarray::ArrayD::from_shape_vec(IxDyn(&[5, 5]), 
    // vec![
    //     Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
    //     Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
    //     Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
    //     Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
    //     Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)
    // ]
    // ).unwrap();

    // let mut kernel = ndarray::ArrayD::from_shape_vec(IxDyn(&[5, 5]), 
    // vec![
    //     Complex::new(1.0/9.0, 0.0), Complex::new(1.0/9.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0/9.0, 0.0),
    //     Complex::new(1.0/9.0, 0.0), Complex::new(1.0/9.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0/9.0, 0.0),
    //     Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
    //     Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
    //     Complex::new(1.0/9.0, 0.0), Complex::new(1.0/9.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0/9.0, 0.0)
    // ]
    // ).unwrap();

    // let mut data_transformed = ndarray::ArrayD::from_elem(IxDyn(&[5, 5]), Complex::new(0.0, 0.0));
    // let mut kernel_transformed = ndarray::ArrayD::from_elem(IxDyn(&[5, 5]), Complex::new(0.0, 0.0));
    // let mut output = ndarray::ArrayD::from_elem(IxDyn(&[5, 5]), Complex::new(0.0, 0.0));
    // let mut convolution = ndarray::ArrayD::from_elem(IxDyn(&[5, 5]), Complex::new(0.0, 0.0));

    // fft::fftnd(&mut test_data, &mut data_transformed, &[0, 1]);
    // println!("data_transformed");
    // println!("{:?}", data_transformed);
    // println!("data_initial");
    // println!("{:?}", test_data);
    // fft::fftnd(&mut kernel, &mut kernel_transformed, &[0, 1]);
    // println!("kernel_transformed");
    // println!("{:?}", kernel_transformed);
    // convolution = data_transformed * kernel_transformed;
    // println!("convolution");
    // println!("{:?}", convolution);
    // fft::ifftnd(&mut convolution, &mut test_data, &[1, 0]);
    // println!("output");
    // println!("{:?}", test_data);
}
