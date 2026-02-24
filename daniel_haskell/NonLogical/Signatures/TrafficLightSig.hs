{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}

-- | Non-logical signature: TrafficLight domain
module NonLogical.Signatures.TrafficLightSig where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat)
import NonLogical.Categories.DATA (MonadOver)

-- | Sor
type LightColor = String

class (TwoMonBLat Omega, MonadOver cat t) => TrafficLightSig cat t where
  -- Sor (prove sorts are objects in cat)
  lightColorObj :: cat LightColor

  -- Const

  -- Fun

  -- mFun
  light :: t LightColor
  driveF :: LightColor -> t Bool

  -- Rel
  eqLight :: LightColor -> LightColor -> Omega

-- mRel
