mod shape;
use shape::TensorShape;

pub struct Tensor {
    buffer: Vec<f32>,
    shape: TensorShape,
}

impl Tensor {
    pub fn new<S: Into<TensorShape>>(shape: S) -> Self {
        Self {
            buffer: Vec::new(),
            shape: shape.into(),
        }
    }
}
