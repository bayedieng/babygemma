pub enum TensorShape {
    D0(()),
    D1((usize)),
    D2((usize, usize)),
    D3((usize, usize, usize)),
    D4((usize, usize, usize, usize)),
}

impl Into<TensorShape> for () {
    fn into(self) -> TensorShape {
        TensorShape::D0(())
    }
}

impl Into<TensorShape> for (usize) {
    fn into(self) -> TensorShape {
        TensorShape::D1(self)
    }
}

impl Into<TensorShape> for (usize, usize) {
    fn into(self) -> TensorShape {
        TensorShape::D2(self)
    }
}

impl Into<TensorShape> for (usize, usize, usize) {
    fn into(self) -> TensorShape {
        TensorShape::D3(self)
    }
}

impl Into<TensorShape> for (usize, usize, usize, usize) {
    fn into(self) -> TensorShape {
        TensorShape::D4(self)
    }
}
