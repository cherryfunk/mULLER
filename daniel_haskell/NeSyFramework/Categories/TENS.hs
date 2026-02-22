{-# LANGUAGE GADTs #-}

module NeSyFramework.Categories.TENS where

import Torch (Tensor)

-- | The category TENS
-- Objects are tensor-based structures.
data TensObj a where
  -- | A list of Tensors
  TensorList :: TensObj [Tensor]
