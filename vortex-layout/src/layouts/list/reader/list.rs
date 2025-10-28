// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reader implementation for ListLayouts containing lists with reified offsets buffers.

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::MaskFuture;
use vortex_dtype::{DType, FieldMask};
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_mask::Mask;

use crate::layouts::list::ListLayout;
use crate::{ArrayFuture, LayoutReader, LayoutReaderRef};

pub struct ListReader {
    name: Arc<str>,
    layout: ListLayout,
    validity: Option<LayoutReaderRef>,
    offsets: LayoutReaderRef,
    elements: LayoutReaderRef,
}

#[async_trait]
impl LayoutReader for ListReader {
    fn name(&self) -> &Arc<str> {
        todo!()
    }

    fn dtype(&self) -> &DType {
        todo!()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        row_range: &Range<u64>,
        splits: &mut BTreeSet<u64>,
    ) -> VortexResult<()> {
        // TODO(aduffy): What am I supposed to do for a FieldMask of vec![Field::ElementType] ?

        // Only register for the elements splits.
        self.elements
            .register_splits(field_mask, row_range, splits)?;

        Ok(())
    }

    fn pruning_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // TODO(aduffy): if the root is an UNNEST, we can splat the mask out into the elements
        //  space and prune that way
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        todo!()
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &ExprRef,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        // If we
        todo!()
    }
}
