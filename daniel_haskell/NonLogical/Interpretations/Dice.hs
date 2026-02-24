{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Non-logical interpretation: Dice domain (in DATA, with Dist monad)
module NonLogical.Interpretations.Dice where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Monads.Dist (Dist (..))
import NonLogical.Signatures.DiceSig

instance DiceSig DATA Dist where
  -- Sor (proving sorts are objects in DATA)
  dieResultObj = Finite [1, 2, 3, 4, 5, 6]

  -- Const

  -- Fun

  -- mFun
  die = Dist [(i, 1.0 / 6.0) | i <- [1 .. 6]]

  -- Rel
  eqDie x y = if x == y then v1 else v0
  evenDie x = if even x then v1 else v0

-- mRel
