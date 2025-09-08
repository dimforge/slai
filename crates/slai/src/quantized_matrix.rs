use crate::ops::{
    GpuBlockQ4K, GpuBlockQ4_0x2, GpuBlockQ4_1x2, GpuBlockQ5K, GpuBlockQ5_0x2, GpuBlockQ5_1x2,
    GpuBlockQ6Kx2, GpuBlockQ8K, GpuBlockQ8_0x2,
};
use slang_hal::backend::{Backend, ShaderBinding};
use slang_hal::shader::ShaderArgsError;
use slang_hal::ShaderArgs;
use stensor::shapes::ViewShape;
use stensor::tensor::GpuMatrix;

pub enum GpuQuantMatrix<B: Backend> {
    F32(GpuMatrix<f32, B>),
    Q8_0(GpuMatrix<GpuBlockQ8_0x2, B>),
    Q5_0(GpuMatrix<GpuBlockQ5_0x2, B>),
    Q5_1(GpuMatrix<GpuBlockQ5_1x2, B>),
    Q4_0(GpuMatrix<GpuBlockQ4_0x2, B>),
    Q4_1(GpuMatrix<GpuBlockQ4_1x2, B>),
    Q8K(GpuMatrix<GpuBlockQ8K, B>),
    Q6K(GpuMatrix<GpuBlockQ6Kx2, B>),
    Q5K(GpuMatrix<GpuBlockQ5K, B>),
    Q4K(GpuMatrix<GpuBlockQ4K, B>),
}

impl<'b, B: Backend> ShaderArgs<'b, B> for GpuQuantMatrix<B> {
    fn write_arg<'a>(
        &'b self,
        binding: ShaderBinding,
        name: &str,
        dispatch: &mut B::Dispatch<'a>,
    ) -> Result<(), ShaderArgsError>
    where
        'b: 'a,
    {
        match self {
            GpuQuantMatrix::F32(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q8_0(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q5_0(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q5_1(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q4_0(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q4_1(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q8K(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q6K(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q5K(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
            GpuQuantMatrix::Q4K(matrix) => matrix.buffer().write_arg(binding, name, dispatch),
        }
    }
}

macro_rules! impl_from(
    ($($variant: ident, $scalar: ident);*) => {$(
        impl<B: Backend> From<GpuMatrix<$scalar, B>> for GpuQuantMatrix<B> {
            fn from(value: GpuMatrix<$scalar, B>) -> Self {
                Self::$variant(value)
            }
        }
    )*}
);

impl_from!(
    F32, f32;
    Q8_0, GpuBlockQ8_0x2;
    Q5_0, GpuBlockQ5_0x2;
    Q5_1, GpuBlockQ5_1x2;
    Q4_0, GpuBlockQ4_0x2;
    Q4_1, GpuBlockQ4_1x2;
    Q8K, GpuBlockQ8K;
    Q6K, GpuBlockQ6Kx2;
    Q5K, GpuBlockQ5K;
    Q4K, GpuBlockQ4K
);

impl<B: Backend> GpuQuantMatrix<B> {
    pub fn shape(&self) -> ViewShape {
        match self {
            Self::F32(m) => m.as_view().shape(),
            Self::Q8_0(m) => m.as_view().shape(),
            Self::Q5_0(m) => m.as_view().shape(),
            Self::Q5_1(m) => m.as_view().shape(),
            Self::Q4_0(m) => m.as_view().shape(),
            Self::Q4_1(m) => m.as_view().shape(),
            Self::Q8K(m) => m.as_view().shape(),
            Self::Q6K(m) => m.as_view().shape(),
            Self::Q5K(m) => m.as_view().shape(),
            Self::Q4K(m) => m.as_view().shape(),
        }
    }

    // pub fn buffer(&self) -> &B::Buffer {
    //     match self {
    //         Self::F32(m) => m.buffer(),
    //         Self::Q8_0(m) => m.buffer(),
    //         Self::Q5_0(m) => m.buffer(),
    //         Self::Q5_1(m) => m.buffer(),
    //         Self::Q4_0(m) => m.buffer(),
    //         Self::Q4_1(m) => m.buffer(),
    //         Self::Q8K(m) => m.buffer(),
    //         Self::Q6K(m) => m.buffer(),
    //         Self::Q5K(m) => m.buffer(),
    //         Self::Q4K(m) => m.buffer(),
    //     }
    // }
}
