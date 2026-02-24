{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

-- | Logical interpretation: Łukasiewicz Logic
module Logical.Interpretations.Lukasiewicz (Omega) where

import Logical.Signatures.TwoMonBLat

-- | Ω := I(τ) = [0,1]
type Omega = Double

instance TwoMonBLat Omega where
  -- Comparison:
  vdash :: Omega -> Omega -> Bool
  vdash = (<=)

  -- Bounded Lattice:
  -- Join Lattice:
  vee :: Omega -> Omega -> Omega
  vee = max
  bot :: Omega
  bot = 0.0

  -- Meet Lattice:
  wedge :: Omega -> Omega -> Omega
  wedge = min
  top :: Omega
  top = 1.0

  -- Monoids:
  -- Monoid 1: Bounded Sum
  oplus :: Omega -> Omega -> Omega
  oplus x y = min 1.0 (x + y)
  v0 :: Omega
  v0 = 0.0

  -- Monoid 2: Bounded Product
  otimes :: Omega -> Omega -> Omega
  otimes x y = max 0.0 (x + y - 1.0)
  v1 :: Omega
  v1 = 1.0
