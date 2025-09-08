use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::{GpuTensor, GpuTensorView};

#[derive(Shader)]
#[shader(module = "slai::win_part")]
pub struct WinPart<B: Backend> {
    pub win_part: GpuFunction<B>,
    pub win_unpart: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct WinPartArgs<'a, B: Backend> {
    shape_source: &'a B::Buffer<ViewShape>,
    shape_result: &'a B::Buffer<ViewShape>,
    source: B::BufferSlice<'a, f32>,
    result: B::BufferSlice<'a, f32>,
}

#[derive(ShaderArgs)]
struct WinUnpartArgs<'a, B: Backend> {
    w: &'a B::Buffer<u32>,
    shape_source: &'a B::Buffer<ViewShape>,
    shape_result: &'a B::Buffer<ViewShape>,
    source: B::BufferSlice<'a, f32>,
    result: B::BufferSlice<'a, f32>,
}

impl<B: Backend> WinPart<B> {
    pub fn launch<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        result: impl Into<GpuTensorView<'b, f32, B>>,
        source: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let result = result.into();
        let source = source.into();
        shapes.insert(backend, result.shape())?;
        shapes.insert(backend, source.shape())?;
        let shape_result = shapes.get(result.shape()).unwrap();
        let shape_source = shapes.get(source.shape()).unwrap();

        let args = WinPartArgs {
            shape_source,
            shape_result,
            result: result.buffer(),
            source: source.buffer(),
        };

        self.win_part
            .launch(backend, pass, &args, [result.len() as u32, 1, 1])
    }

    pub fn launch_unpart<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        window_size: &GpuTensor<u32, B>,
        result: impl Into<GpuTensorView<'b, f32, B>>,
        source: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let result = result.into();
        let source = source.into();
        shapes.insert(backend, result.shape())?;
        shapes.insert(backend, source.shape())?;
        let shape_result = shapes.get(result.shape()).unwrap();
        let shape_source = shapes.get(source.shape()).unwrap();

        let args = WinUnpartArgs {
            w: window_size.buffer(),
            shape_source,
            shape_result,
            result: result.buffer(),
            source: source.buffer(),
        };

        self.win_unpart
            .launch(backend, pass, &args, [result.len() as u32, 1, 1])
    }
}
