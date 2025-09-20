use dashmap::DashMap;
use slang_hal::backend::{Backend, DeviceValue};
use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use stensor::shapes::MatrixOrdering;
use stensor::tensor::{GpuTensor, GpuTensorView, GpuTensorViewMut};
use wgpu::BufferUsages;
// HACK: this is a last-minute workaround to keep tensors alive so they don’t get freed before
//       the pipeline runs when using the `LlmContext`. Need to revisit this after the conference.

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct TensorKey {
    ty: TypeId,
    shape: [u32; 4],
    ordering: MatrixOrdering,
    usage: BufferUsages,
}

impl TensorKey {
    pub fn with_type<T: Any>(
        shape: [u32; 4],
        ordering: MatrixOrdering,
        usage: BufferUsages,
    ) -> TensorKey {
        Self {
            ty: TypeId::of::<T>(),
            shape,
            ordering,
            usage,
        }
    }
    pub fn new<T: DeviceValue, B: Backend>(
        tensor: &GpuTensor<T, B>,
        usage: BufferUsages,
    ) -> TensorKey {
        TensorKey {
            ty: TypeId::of::<T>(),
            shape: tensor.shape(),
            ordering: tensor.ordering(),
            usage,
        }
    }
}

pub struct CachedTensor<T: DeviceValue, B: Backend> {
    tensor: Option<Box<GpuTensor<T, B>>>, // Use an Option to move-out the tensor on drop.
    cache: TensorCache<B>,
    usage: BufferUsages,
}

impl<T: DeviceValue, B: Backend> CachedTensor<T, B> {
    pub fn tensor(&self) -> &GpuTensor<T, B> {
        self.tensor
            .as_ref()
            .expect("internal error: tensor was already dropped")
    }

    pub fn tensor_mut(&mut self) -> &mut GpuTensor<T, B> {
        self.tensor
            .as_mut()
            .expect("internal error: tensor was already dropped")
    }

    // TODO: return the naked tensor once `Box::into_inner` is stabilized.
    pub fn into_inner(mut self) -> Box<GpuTensor<T, B>> {
        self.tensor
            .take()
            .expect("internal error: tensor was already dropped")
    }
}

impl<T: DeviceValue, B: Backend> AsRef<GpuTensor<T, B>> for CachedTensor<T, B> {
    fn as_ref(&self) -> &GpuTensor<T, B> {
        self.tensor()
    }
}

impl<T: DeviceValue, B: Backend> AsMut<GpuTensor<T, B>> for CachedTensor<T, B> {
    fn as_mut(&mut self) -> &mut GpuTensor<T, B> {
        self.tensor_mut()
    }
}

impl<T: DeviceValue, B: Backend> Deref for CachedTensor<T, B> {
    type Target = GpuTensor<T, B>;

    fn deref(&self) -> &Self::Target {
        self.tensor()
    }
}

impl<T: DeviceValue, B: Backend> DerefMut for CachedTensor<T, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.tensor_mut()
    }
}

impl<'a, T: DeviceValue, B: Backend> From<&'a CachedTensor<T, B>> for GpuTensorView<'a, T, B> {
    fn from(val: &'a CachedTensor<T, B>) -> Self {
        val.tensor().as_view()
    }
}

impl<'a, T: DeviceValue, B: Backend> From<&'a mut CachedTensor<T, B>>
    for GpuTensorViewMut<'a, T, B>
{
    fn from(val: &'a mut CachedTensor<T, B>) -> Self {
        val.tensor_mut().as_view_mut()
    }
}

impl<T: DeviceValue, B: Backend> Drop for CachedTensor<T, B> {
    fn drop(&mut self) {
        if let Some(t) = self.tensor.take() {
            self.cache.reclaim(t, self.usage);
        }
    }
}

pub struct TensorCache<B: Backend> {
    // TODO: would be great to have a type-erased tensor type `GpuTensor<_, B>` that
    //       we can store in the hashmap. That would avoid the need for `Box<Any>` in
    //       the dashmap.
    tensors: Arc<DashMap<TensorKey, Vec<Box<dyn Any + Send + Sync>>>>,
    _phantom: PhantomData<B>,
}

impl<B: Backend> Clone for TensorCache<B> {
    fn clone(&self) -> Self {
        Self {
            tensors: self.tensors.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> Default for TensorCache<B> {
    fn default() -> Self {
        Self {
            tensors: Arc::new(DashMap::new()),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> TensorCache<B> {
    pub fn get<T: DeviceValue>(&self, key: TensorKey) -> Option<CachedTensor<T, B>> {
        let t = self.tensors.get_mut(&key)?.pop()?;
        let tensor = t
            .downcast()
            .expect("internal error: invalid cached tensor downcast");

        Some(CachedTensor {
            tensor: Some(tensor),
            cache: self.clone(),
            usage: key.usage,
        })
    }

    pub fn get_or_insert<T: DeviceValue>(
        &self,
        key: TensorKey,
        insert: impl FnOnce() -> Result<GpuTensor<T, B>, B::Error>,
    ) -> Result<CachedTensor<T, B>, B::Error> {
        let mut tensors = self.tensors.entry(key).or_default();

        if let Some(t) = tensors.pop() {
            let tensor = t
                .downcast()
                .expect("internal error: invalid cached tensor downcast");
            Ok(CachedTensor {
                tensor: Some(tensor),
                cache: self.clone(),
                usage: key.usage,
            })
        } else {
            let tensor = insert()?;
            Ok(CachedTensor {
                tensor: Some(Box::new(tensor)),
                cache: self.clone(),
                usage: key.usage,
            })
        }
    }

    pub fn enroll<T: DeviceValue>(
        &self,
        tensor: GpuTensor<T, B>,
        usage: BufferUsages,
    ) -> CachedTensor<T, B> {
        CachedTensor {
            tensor: Some(Box::new(tensor)),
            cache: self.clone(),
            usage,
        }
    }

    pub fn clear(&mut self) {
        self.tensors.clear();
    }

    fn reclaim<T: DeviceValue>(&self, t: Box<GpuTensor<T, B>>, usage: BufferUsages) {
        let key = TensorKey::new(&t, usage);
        self.tensors.entry(key).or_default().push(t);
    }
}
