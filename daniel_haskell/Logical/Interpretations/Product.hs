{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

-- | Logical interpretation: Product Logic (Ω = [0,1] ⊂ ℝ)
module Logical.Interpretations.Product (Omega) where

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
  -- Monoid 1: Probabilistic Sum
  oplus :: Omega -> Omega -> Omega
  oplus x y = x + y - (x * y)
  v0 :: Omega
  v0 = 0.0

  -- Monoid 2: Probabilistic Product
  otimes :: Omega -> Omega -> Omega
  otimes = (*)
  v1 :: Omega
  v1 = 1.0