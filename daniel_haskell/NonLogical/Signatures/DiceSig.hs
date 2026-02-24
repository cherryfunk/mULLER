{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Non-logical signature: Dice domain
module NonLogical.Signatures.DiceSig where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat)

-- | Sor
type DieResult = Int

class (TwoMonBLat Omega) => DiceSig cat t where
  -- Sor (prove sorts are objects in cat)
  dieResultObj :: cat DieResult

  -- Const

  -- Fun

  -- mFun
  die :: t DieResult

  -- Rel
  eqDie :: DieResult -> DieResult -> Omega
  evenDie :: DieResult -> Omega

-- mRel
