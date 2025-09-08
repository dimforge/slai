//! Loading ggml files.

use crate::gguf::{Gguf, GgufTensor, GgufTensorData};
use bytemuck::{Pod, PodCastError};
use std::collections::HashMap;

const GGML_FILE_MAGIC: u32 = 0x67676d6c; // "ggml"
const GGML_QNT_VERSION_FACTOR: u32 = 1000; // do not change this

#[derive(thiserror::Error, Debug, Copy, Clone)]
pub enum SamGgmlParseError {
    #[error(
        "the input file isn’t a ggml binary file. Got a magic number of {0} instead of 0x67676d6c"
    )]
    IncorrectMagicNumber(u32),
    #[error("invalid vocabulary size (expected {expected}, found {found})")]
    VocabSizeMismatch { expected: u32, found: u32 },
    #[error(transparent)]
    Cast(#[from] PodCastError),
}

/// Legacy GGML format used by sam.cpp.
/// TODO: load pth directly
pub struct SamGgmlFile {
    pub params: HParams,
    pub weights: HashMap<String, ([u32; 4], Vec<f32>)>,
}

#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Eq, Debug)]
#[repr(C)]
pub struct HParams {
    pub n_enc_state: u32,
    pub n_enc_layers: u32,
    pub n_enc_heads: u32,
    pub n_enc_out_chans: u32,
    pub n_pt_embd: u32,
    pub ftype: u32,
}

impl SamGgmlFile {
    pub fn into_gguf(self) -> Gguf {
        let mut metadata = HashMap::new();
        let mut tensors = HashMap::new();
        metadata.insert("general.architecture".to_owned(), "sam".to_owned().into());
        metadata.insert("n_enc_state".to_string(), self.params.n_enc_state.into());
        metadata.insert("n_enc_layers".to_string(), self.params.n_enc_layers.into());
        metadata.insert("n_enc_heads".to_string(), self.params.n_enc_heads.into());
        metadata.insert(
            "n_enc_out_chans".to_string(),
            self.params.n_enc_out_chans.into(),
        );
        metadata.insert("n_pt_embd".to_string(), self.params.n_pt_embd.into());
        metadata.insert("ftype".to_string(), self.params.ftype.into());

        for (name, (mut ne, data)) in self.weights {
            ne.swap(0, 1); // Account for ggml’s swaped row/cols indices.
            tensors.insert(
                name,
                GgufTensor {
                    dimensions: ne.map(|e| e as u64),
                    offset: u64::MAX, // Unknown
                    data: GgufTensorData::F32(data),
                },
            );
        }

        Gguf {
            version: 3,
            metadata,
            tensors,
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SamGgmlParseError> {
        let mut offset = 0;
        let params = Self::load_header(bytes, &mut offset)?;
        // println!("Found params: {:?}", params);
        let weights = Self::load_weights(bytes, &mut offset)?;

        Ok(Self { params, weights })
    }

    fn load_header(bytes: &[u8], offset: &mut usize) -> Result<HParams, SamGgmlParseError> {
        let magic: u32 = Self::read_pod(bytes, offset)?;

        if magic != GGML_FILE_MAGIC {
            return Err(SamGgmlParseError::IncorrectMagicNumber(magic));
        }
        let mut hparams: HParams = Self::read_pod(bytes, offset)?;
        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
        Ok(hparams)
    }

    #[allow(clippy::type_complexity)]
    fn load_weights(
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<HashMap<String, ([u32; 4], Vec<f32>)>, SamGgmlParseError> {
        let mut weights = HashMap::new();

        while *offset < bytes.len() {
            // Number of tensor dimensions.
            let n_dims: u32 = Self::read_pod(bytes, offset)?;
            // The tensor name length.
            let length: u32 = Self::read_pod(bytes, offset)?;
            // Type of the elements in the tensor.
            let ttype: u32 = Self::read_pod(bytes, offset)?;

            // println!(
            //     "Found n_dims: {}, length: {}, ttype: {}",
            //     n_dims, length, ttype
            // );

            if *offset >= bytes.len() {
                break; // Reached eof
            }

            let mut nelements = 1u32;
            let mut ne = [1u32; 4];

            for i in 0..n_dims as usize {
                ne[i] = Self::read_pod(bytes, offset)?;
                nelements *= ne[i];
            }

            let name = String::from_utf8_lossy(Self::read_array(length as usize, bytes, offset)?);
            // println!(
            //     "Found tensor with name: {}, elements: {}, dimensions: {:?}",
            //     name, nelements, ne
            // );

            // TODO: support other tensor data types.
            let data: Vec<f32> = match ttype {
                0 => Self::read_array_unaligned(nelements as usize, bytes, offset),
                1 => Self::read_f32_array_from_f16_unaligned(nelements as usize, bytes, offset),
                _ => panic!("Unsupported tensor element type: {ttype}."),
            }?;
            // println!("fetched {} elements", data.len());
            weights.insert(name.into_owned(), (ne, data));
        }

        Ok(weights)
    }

    fn read_pod<T: Pod>(bytes: &[u8], offset: &mut usize) -> Result<T, PodCastError> {
        let sz = std::mem::size_of::<T>();
        let bytes = &bytes[*offset..(*offset + sz)];
        *offset += sz;
        bytemuck::try_pod_read_unaligned(bytes)
    }

    fn read_array_unaligned<T: Pod>(
        len: usize,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<Vec<T>, PodCastError> {
        let mut result = Vec::with_capacity(len);
        for _ in 0..len {
            result.push(Self::read_pod(bytes, offset)?);
        }
        Ok(result)
    }

    fn read_array<'a, T: Pod>(
        len: usize,
        bytes: &'a [u8],
        offset: &mut usize,
    ) -> Result<&'a [T], PodCastError> {
        let sz = std::mem::size_of::<T>() * len;
        let bytes = &bytes[*offset..(*offset + len)];
        *offset += sz;
        bytemuck::try_cast_slice(bytes)
    }

    fn read_f32_array_from_f16_unaligned(
        len: usize,
        bytes: &[u8],
        offset: &mut usize,
    ) -> Result<Vec<f32>, PodCastError> {
        let mut result = Vec::with_capacity(len);
        for _ in 0..len {
            let elt: u16 = Self::read_pod(bytes, offset)?;
            result.push(decode_f16(elt));
        }
        Ok(result)
    }
}

// From https://stackoverflow.com/questions/36008434/how-can-i-decode-f16-to-f32-using-only-the-stable-standard-library
fn decode_f16(half: u16) -> f32 {
    let exp: u16 = half >> 10 & 0x1f;
    let mant: u16 = half & 0x3ff;
    let val: f32 = if exp == 0 {
        (mant as f32) * (2.0f32).powi(-24)
    } else if exp != 31 {
        (mant as f32 + 1024f32) * (2.0f32).powi(exp as i32 - 25)
    } else if mant == 0 {
        f32::INFINITY
    } else {
        f32::NAN
    };
    if half & 0x8000 != 0 {
        -val
    } else {
        val
    }
}
