use pixel_canvas::canvas::CanvasInfo;
use glium::glutin::event::{Event, WindowEvent};

// Implementation of glutin's WindowEvent::ReceivedCharacter event for pixel-canvas 0.2.3

pub struct KeyboardState {
    pub character: char,
}

impl KeyboardState {
    pub fn new() -> Self {
        Self {
            character: '\0',
        }
    }

    pub fn handle_input(_info: &CanvasInfo, keyboard: &mut KeyboardState, event: &Event<()>) -> bool {
        match event {
            Event::WindowEvent {
                event: WindowEvent::ReceivedCharacter(c), .. 
            } => {
                keyboard.character = *c;
                true
            }
            _ => false,
        }
    }
}
