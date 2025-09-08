use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::{GpuScalar, GpuTensorView};

#[derive(Copy, Clone, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
#[repr(C)]
pub struct ConvTranspose2dConfig {
    pub stride: u32,
}

#[derive(Shader)]
#[shader(module = "slai::conv_transpose_2d")]
pub struct ConvTranspose2d<B: Backend> {
    pub init_dest: GpuFunction<B>,
    pub init_wdata: GpuFunction<B>,
    pub init_src0: GpuFunction<B>,
    pub init_src1: GpuFunction<B>,
    pub conv_transpose_2d_ref: GpuFunction<B>,
    pub conv_transpose_2d: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct ConvTranspose2dRefArgs<'a, B: Backend> {
    stride: &'a B::Buffer<ConvTranspose2dConfig>,
    shape_src0: &'a B::Buffer<ViewShape>,
    shape_src1: &'a B::Buffer<ViewShape>,
    shape_dest: &'a B::Buffer<ViewShape>,
    shape_wdata: &'a B::Buffer<ViewShape>,
    src0: B::BufferSlice<'a, f32>,
    src1: B::BufferSlice<'a, f32>,
    dest: B::BufferSlice<'a, f32>,
    wdata: B::BufferSlice<'a, f32>,
}

#[derive(ShaderArgs)]
struct ConvTranspose2dArgs<'a, B: Backend> {
    stride: &'a B::Buffer<ConvTranspose2dConfig>,
    shape_src0: &'a B::Buffer<ViewShape>,
    shape_src1: &'a B::Buffer<ViewShape>,
    shape_dest: &'a B::Buffer<ViewShape>,
    src0: B::BufferSlice<'a, f32>,
    src1: B::BufferSlice<'a, f32>,
    dest: B::BufferSlice<'a, f32>,
}

impl<B: Backend> ConvTranspose2d<B> {
    pub fn launch_ref<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        shapes: &mut ViewShapeBuffers<B>,
        params: &GpuScalar<ConvTranspose2dConfig, B>,
        dest: impl Into<GpuTensorView<'b, f32, B>>,
        src0: impl Into<GpuTensorView<'b, f32, B>>,
        src1: impl Into<GpuTensorView<'b, f32, B>>,
        wdata: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let dest = dest.into();
        let src0 = src0.into();
        let src1 = src1.into();
        let wdata = wdata.into();

        assert_eq!(wdata.len(), src0.len() + src1.len());

        shapes.insert(backend, dest.shape())?;
        shapes.insert(backend, src0.shape())?;
        shapes.insert(backend, src1.shape())?;
        shapes.insert(backend, wdata.shape())?;
        let shape_dest = shapes.get(dest.shape()).unwrap();
        let shape_src0 = shapes.get(src0.shape()).unwrap();
        let shape_src1 = shapes.get(src1.shape()).unwrap();
        let shape_wdata = shapes.get(wdata.shape()).unwrap();

        let args = ConvTranspose2dRefArgs {
            stride: params.buffer(),
            shape_dest,
            shape_src0,
            shape_src1,
            shape_wdata,
            dest: dest.buffer(),
            src0: src0.buffer(),
            src1: src1.buffer(),
            wdata: wdata.buffer(),
        };

        self.init_dest
            .launch(backend, pass, &args, [dest.len() as u32, 1, 1])?;
        self.init_wdata
            .launch(backend, pass, &args, [wdata.len() as u32, 1, 1])?;
        self.init_src0
            .launch(backend, pass, &args, [src0.len() as u32, 1, 1])?;
        self.init_src1
            .launch(backend, pass, &args, [src1.len() as u32, 1, 1])?;
        self.conv_transpose_2d_ref
            .launch(backend, pass, &args, [dest.size(2), 1, 1])?;

        Ok(())
    }

    pub fn launch<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        shapes: &mut ViewShapeBuffers<B>,
        params: &mut GpuScalar<ConvTranspose2dConfig, B>,
        dest: impl Into<GpuTensorView<'b, f32, B>>,
        src0: impl Into<GpuTensorView<'b, f32, B>>,
        src1: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let dest = dest.into();
        let src0 = src0.into();
        let src1 = src1.into();

        let src0 = src0.permute([3, 0, 1, 2]);
        let src1 = src1.permute([2, 0, 1, 3]);

        shapes.insert(backend, dest.shape())?;
        shapes.insert(backend, src0.shape())?;
        shapes.insert(backend, src1.shape())?;
        let shape_dest = shapes.get(dest.shape()).unwrap();
        let shape_src0 = shapes.get(src0.shape()).unwrap();
        let shape_src1 = shapes.get(src1.shape()).unwrap();

        let args = ConvTranspose2dArgs {
            stride: params.buffer(),
            shape_dest,
            shape_src0,
            shape_src1,
            dest: dest.buffer(),
            src0: src0.buffer(),
            src1: src1.buffer(),
        };

        self.conv_transpose_2d
            .launch(backend, pass, &args, [dest.size(2), 1, 1])?;

        Ok(())
    }
}
