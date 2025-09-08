use async_std::sync::RwLock;
use image::{ImageFormat, ImageReader, RgbImage};
use nalgebra::{DMatrix, Point2};
use slai::context::{LlmContext, LlmOps};
use slai::gguf::Gguf;
use slai::models::segment_anything::{
    sam_decode_mask, sam_encode_image, sam_encode_prompt, sam_fill_dense_pe, sam_image_preprocess,
    sam_postprocess_masks, SamImage, SamImageU8, SamModel, SamState,
};
use slai::re_exports::stensor::shapes::ViewShapeBuffers;
use slai::tensor_cache::TensorCache;
use slang_hal::backend::Backend;
use slang_hal::re_exports::minislang::SlangCompiler;
use std::path::PathBuf;

pub struct SegmentAnything<B: Backend> {
    pub ops: LlmOps<B>,
    pub model: SamModel<B>,
    pub state: SamState<B>,
    original_image: Option<RgbImage>,
    shapes: RwLock<ViewShapeBuffers<B>>,
    tensor_cache: RwLock<TensorCache<B>>,
}

impl<B: Backend> SegmentAnything<B> {
    pub const MASK_ON_VALS: u8 = 255;
    pub const MASK_OFF_VALS: u8 = 0;

    pub fn from_gguf(
        backend: &B,
        compiler: &SlangCompiler,
        gguf: &Gguf,
    ) -> anyhow::Result<SegmentAnything<B>> {
        let ops = LlmOps::new(backend, compiler)?;
        let model = SamModel::from_gguf(backend, gguf)?;
        let state = SamState::new(backend)?;
        let result = Self {
            ops,
            model,
            state,
            shapes: RwLock::new(ViewShapeBuffers::new(backend)),
            tensor_cache: RwLock::new(TensorCache::default()),
            original_image: None,
        };
        Ok(result)
    }

    pub fn original_image(&self) -> Option<&RgbImage> {
        self.original_image.as_ref()
    }

    pub async fn load_image(&mut self, backend: &B, img_path: &PathBuf) -> anyhow::Result<()> {
        let img = ImageReader::open(img_path)?.decode()?.to_rgb8();
        let (nrows, ncols) = img.dimensions();
        let img_matrix = DMatrix::from_fn(nrows as usize, ncols as usize, |i, j| {
            img.get_pixel(i as u32, j as u32).0
        });
        let preprocessed_img = sam_image_preprocess(&img_matrix);
        let sam_img = SamImage {
            pixels: preprocessed_img,
        };

        let mut shapes = self.shapes.write_blocking();
        let mut tensor_cache = self.tensor_cache.write().await;
        shapes.clear_tmp();
        tensor_cache.clear();

        {
            let t0 = std::time::Instant::now();
            let mut ctx = LlmContext {
                backend,
                shapes: &mut *shapes,
                cache: &mut *tensor_cache,
                pass: None,
                encoder: None,
                ops: &self.ops,
            };

            ctx.begin_submission();
            sam_encode_image(&mut ctx, &self.model, &mut self.state, &sam_img)?;
            self.state.pe_img = sam_fill_dense_pe(&mut ctx, &self.model)?.into_inner();
            ctx.submit();
            backend.synchronize()?;
            println!("#### IMAGE ENCODING: {}", t0.elapsed().as_secs_f32());
        }

        self.original_image = Some(img);

        Ok(())
    }

    pub async fn apply_prompt(
        &mut self,
        backend: &B,
        pt: Point2<f32>,
    ) -> anyhow::Result<Vec<SamImageU8>> {
        let Some(img) = &self.original_image else {
            anyhow::bail!("No image loaded.");
        };

        let (nrows, ncols) = img.dimensions();
        let mut shapes = self.shapes.write().await;
        let mut tensor_cache = self.tensor_cache.write().await;

        let t0 = std::time::Instant::now();
        let mut ctx = LlmContext {
            backend,
            shapes: &mut *shapes,
            cache: &mut *tensor_cache,
            pass: None,
            encoder: None,
            ops: &self.ops,
        };
        ctx.begin_submission();

        let encoded_prompt = sam_encode_prompt(&mut ctx, &self.model, nrows, ncols, pt)?;
        sam_decode_mask(&mut ctx, &self.model, &encoded_prompt, &mut self.state)?;

        ctx.submit();
        backend.synchronize()?;
        println!("#### MASKS CALCULATION: {}", t0.elapsed().as_secs_f32());

        let masks = sam_postprocess_masks(
            backend,
            &self.model.hparams,
            nrows as usize,
            ncols as usize,
            &self.state,
            Self::MASK_ON_VALS,
            Self::MASK_OFF_VALS,
        )
        .await?;
        Ok(masks)
    }

    pub fn save_masked_image<W>(&mut self, mask: &SamImageU8, mut writer: W) -> anyhow::Result<()>
    where
        W: std::io::Write + std::io::Seek,
    {
        let Some(img) = &self.original_image else {
            anyhow::bail!("No image loaded.");
        };

        let mut rgb = RgbImage::new(mask.nx as u32, mask.ny as u32);

        for i in 0..mask.nx {
            for j in 0..mask.ny {
                let val = mask.data[j * mask.nx + i];
                let mut pixel = img[(i as u32, j as u32)].0;
                if val == Self::MASK_ON_VALS {
                    pixel[2] = 200;
                }
                rgb.put_pixel(i as u32, j as u32, pixel.into());
            }
        }

        rgb.write_to(&mut writer, ImageFormat::Png)?;

        Ok(())
    }
}
