{-# LANGUAGE GADTs #-}

module A2_Interpretation.B2_Typological.Categories.TENS where

import Torch (Tensor)

-- | The category TENS
-- Objects are tensor spaces (embedding spaces).
-- A Tensor represents a point in some R^{d1 x ... x dk}.
data TensObj a where
  -- | A tensor space
  TensorSpace :: TensObj Tensor
