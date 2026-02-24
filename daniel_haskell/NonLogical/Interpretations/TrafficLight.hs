{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Non-logical interpretation: TrafficLight domain (in DATA, with Giry monad)
module NonLogical.Interpretations.TrafficLight
  ( light',
    eqLight',
  )
where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Monads.Giry (Giry, categorical)
import NonLogical.Signatures.TrafficLightSig
import TypedSyntax

instance TrafficLightSig DATA Giry where
  lightColorObj = Finite ["Red", "Green", "Yellow"]
  light = categorical [("Red", 0.6), ("Green", 0.3), ("Yellow", 0.1)]
  driveF "Red" = categorical [(True, 0.1), (False, 0.9)]
  driveF "Yellow" = categorical [(True, 0.2), (False, 0.8)]
  driveF "Green" = categorical [(True, 0.9), (False, 0.1)]
  eqLight x y = if x == y then v1 else v0

-- | Formula-ready symbols
light' :: Term (Giry String)
light' = con (light @DATA @Giry)

eqLight' :: Term String -> Term String -> Formula Omega
eqLight' x y = rel (con (eqLight @DATA @Giry) $$ x $$ y)
