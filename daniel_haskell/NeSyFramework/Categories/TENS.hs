{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}

module NeSyFramework.Categories.TENS where

import Torch
import NeSyFramework.Categories.Utils (In, TENS)

-- | Instances for TENS category (Lists of Tensors) 
instance In TENS [Tensor]
