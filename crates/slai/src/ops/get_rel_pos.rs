use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{MatrixOrdering, ViewShape, ViewShapeBuffers};
use stensor::tensor::GpuTensorView;

#[derive(Shader)]
#[shader(module = "slai::get_rel_pos")]
pub struct GetRelPos<B: Backend> {
    pub get_rel_pos: GpuFunction<B>,
    pub add_rel_pos_phase1: GpuFunction<B>,
    pub add_rel_pos_phase2: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct GetRelPosArgs<'a, B: Backend> {
    shape_source: &'a B::Buffer<ViewShape>,
    shape_result: &'a B::Buffer<ViewShape>,
    source: B::BufferSlice<'a, f32>,
    result: B::BufferSlice<'a, f32>,
}

#[derive(ShaderArgs)]
struct AddRelPosArgs<'a, B: Backend> {
    shape_src1: &'a B::Buffer<ViewShape>,
    src1: B::BufferSlice<'a, f32>,
    src2: B::BufferSlice<'a, f32>,
    dst: B::BufferSlice<'a, f32>,
}

impl<B: Backend> GetRelPos<B> {
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

        let args = GetRelPosArgs {
            shape_source,
            shape_result,
            result: result.buffer(),
            source: source.buffer(),
        };

        self.get_rel_pos
            .launch(backend, pass, &args, [result.len() as u32, 1, 1])
    }

    pub fn launch_add_rel_pos<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        dst: impl Into<GpuTensorView<'b, f32, B>>,
        src1: impl Into<GpuTensorView<'b, f32, B>>,
        src2: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let dst = dst.into();
        let src1 = src1.into();
        let src2 = src2.into();

        assert_eq!(dst.is_contiguous(), Some(MatrixOrdering::RowMajor));
        assert_eq!(src1.is_contiguous(), Some(MatrixOrdering::RowMajor));
        assert_eq!(src2.is_contiguous(), Some(MatrixOrdering::RowMajor));
        assert_eq!(src1.shape().size, src2.shape().size);
        assert_eq!(src1.size(3), dst.size(2));
        assert_eq!(src1.size(1) * src1.size(1), dst.size(1));
        assert_eq!(src1.size(0) * src1.size(1), dst.size(0));

        shapes.insert(backend, src1.shape())?;
        let shape_src1 = shapes.get(src1.shape()).unwrap();

        let args = AddRelPosArgs {
            shape_src1,
            dst: dst.buffer(),
            src1: src1.buffer(),
            src2: src2.buffer(),
        };

        self.add_rel_pos_phase1
            .launch(backend, pass, &args, [src1.len() as u32, 1, 1])?;
        self.add_rel_pos_phase2
            .launch(backend, pass, &args, [src1.len() as u32, 1, 1])
    }
}
