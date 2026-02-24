{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Non-logical interpretation: TrafficLight domain (in DATA, with Giry monad)
module NonLogical.Interpretations.TrafficLight where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Monads.Giry (Giry, categorical)
import NonLogical.Signatures.TrafficLightSig

instance TrafficLightSig DATA Giry where
  -- Sor (proving sorts are objects in DATA)
  lightColorObj = Finite ["Red", "Green", "Yellow"]

  -- Const

  -- Fun

  -- mFun
  light = categorical [("Red", 0.6), ("Green", 0.3), ("Yellow", 0.1)]
  driveF "Red" = categorical [(True, 0.1), (False, 0.9)]
  driveF "Yellow" = categorical [(True, 0.2), (False, 0.8)]
  driveF "Green" = categorical [(True, 0.9), (False, 0.1)]

  -- Rel
  eqLight x y = if x == y then v1 else v0

-- mRel
