// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition and implementation of [`FixedSizeListVectorMut`].

use std::sync::Arc;

use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_ensure};
use vortex_mask::MaskMut;

use crate::{FixedSizeListVector, VectorMut, VectorMutOps, match_vector_pair};

/// TODO docs.
#[derive(Debug, Clone)]
pub struct FixedSizeListVectorMut {
    /// The mutable child vector of elements.
    pub(super) elements: Box<VectorMut>,

    /// The size of every list in the vector.
    pub(super) list_size: u32,

    /// The validity mask (where `true` represents a list is **not** null).
    ///
    /// Note that the `elements` vector will have its own internal validity, denoting if individual
    /// list elements are null.
    pub(super) validity: MaskMut,

    /// The length of the vector (which is the same as the length of the validity mask).
    ///
    /// This is stored here as a convenience, as the validity also tracks this information.
    pub(super) len: usize,
}

impl FixedSizeListVectorMut {
    /// Creates a new [`FixedSizeListVectorMut`] from the given `elements` vector, size of each
    /// list, and validity mask.
    ///
    /// # Panics
    ///
    /// Panics if the length of the `validity` mask multiplied by the `list_size` is not
    /// equal to the length of the `elements` vector.
    ///
    /// Put another way, the length of the `elements` vector divided by the `list_size` must be
    /// equal to the length of the validity, or this function will panic.
    pub fn new(elements: Box<VectorMut>, list_size: u32, validity: MaskMut) -> Self {
        Self::try_new(elements, list_size, validity)
            .vortex_expect("Failed to create `FixedSizeListVectorMut`")
    }

    /// Tries to create a new [`FixedSizeListVectorMut`] from the given `elements` vector, size of
    /// each list, and validity mask.
    ///
    /// # Errors
    ///
    /// Returns and error if the length of the `validity` mask multiplied by the `list_size` is not
    /// equal to the length of the `elements` vector.
    ///
    /// Put another way, the length of the `elements` vector divided by the `list_size` must be
    /// equal to the length of the validity.
    pub fn try_new(
        elements: Box<VectorMut>,
        list_size: u32,
        validity: MaskMut,
    ) -> VortexResult<Self> {
        let len = validity.len();
        let elements_len = elements.len();

        if list_size == 0 {
            vortex_ensure!(
                elements.is_empty(),
                "A degenerate (`list_size == 0`) `FixedSizeListVectorMut` should have no underlying elements",
            );
        } else {
            vortex_ensure!(
                list_size as usize * len == elements_len,
                "Tried to create a `FixedSizeListVectorMut` of length {len} and list_size {list_size} \
                with an child vector of size {elements_len} ({list_size} * {len} != {elements_len})",
            );
        }

        Ok(Self {
            elements,
            list_size,
            validity,
            len,
        })
    }

    /// Tries to create a new [`FixedSizeListVectorMut`] from the given `elements` vector, size of
    /// each list, and validity mask without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the length of the `validity` mask multiplied by the `list_size`
    /// is exactly equal to the length of the `elements` vector.
    pub unsafe fn new_unchecked(
        elements: Box<VectorMut>,
        list_size: u32,
        validity: MaskMut,
    ) -> Self {
        let len = validity.len();

        if cfg!(debug_assertions) {
            Self::new(elements, list_size, validity)
        } else {
            Self {
                elements,
                list_size,
                validity,
                len,
            }
        }
    }

    /// Creates a new [`FixedSizeListVectorMut`] with given element type, list size, and capacity.
    pub fn with_capacity(elem_dtype: &DType, list_size: u32, capacity: usize) -> Self {
        let elements = Box::new(VectorMut::with_capacity(
            elem_dtype,
            capacity * list_size as usize,
        ));

        let validity = MaskMut::with_capacity(capacity);
        let len = validity.len();

        Self {
            elements,
            list_size,
            validity,
            len,
        }
    }

    /// Decomposes the `FixedSizeListVector` into its constituent parts (child elements, list size,
    /// and validity).
    pub fn into_parts(self) -> (Box<VectorMut>, u32, MaskMut) {
        (self.elements, self.list_size, self.validity)
    }

    /// Returns the child vector of elements, which represents the contiguous fixed-size lists of
    /// the `FixedSizeListVector`.
    pub fn elements(&self) -> &VectorMut {
        &self.elements
    }

    /// Returns the size of every list in the vector.
    pub fn list_size(&self) -> u32 {
        self.list_size
    }
}

impl VectorMutOps for FixedSizeListVectorMut {
    type Immutable = FixedSizeListVector;

    fn len(&self) -> usize {
        self.len
    }

    fn capacity(&self) -> usize {
        self.elements
            .capacity()
            .checked_div(self.list_size as usize)
            .unwrap_or(usize::MAX)
    }

    fn reserve(&mut self, additional: usize) {
        self.elements.reserve(additional * self.list_size as usize);
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match_vector_pair!(
            self.elements.as_mut(),
            other.elements.as_ref(),
            |a: VectorMut, b: Vector| {
                // This will panic if `other.elements` is not the correct type of vector.
                a.extend_from_vector(b);
            }
        );

        self.validity.append_mask(&other.validity);
        self.len += other.len;
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn append_nulls(&mut self, n: usize) {
        self.elements.append_nulls(n * self.list_size as usize);
        self.validity.append_n(false, n);
        debug_assert_eq!(self.len, self.validity.len());
    }

    fn freeze(self) -> Self::Immutable {
        FixedSizeListVector {
            elements: Arc::new(self.elements.freeze()),
            list_size: self.list_size,
            validity: self.validity.freeze(),
            len: self.len,
        }
    }

    fn split_off(&mut self, at: usize) -> Self {
        assert!(
            at <= self.capacity(),
            "split_off out of bounds: {} > {}",
            at,
            self.capacity()
        );

        let split_elements = self.elements.split_off(at * self.list_size as usize);

        let split_validity = self.validity.split_off(at);
        let split_len = self.len.saturating_sub(at);
        self.len = at;

        debug_assert_eq!(self.len, self.validity.len());

        Self {
            elements: Box::new(split_elements),
            list_size: self.list_size,
            validity: split_validity,
            len: split_len,
        }
    }

    fn unsplit(&mut self, other: Self) {
        assert_eq!(self.list_size, other.list_size);

        self.elements.unsplit(*other.elements);
        self.validity.unsplit(other.validity);

        self.len += other.len;
        debug_assert_eq!(self.len, self.validity.len());
    }
}
