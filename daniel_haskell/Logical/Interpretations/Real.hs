{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

-- | Logical interpretation: Real-valued Logic
module Logical.Interpretations.Real (Omega) where

import Logical.Signatures.TwoMonBLat

-- | Ω := I(τ) = ℝ (approximated by IEEE 754 Double)
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
  bot = -1.0 / 0.0

  -- Meet Lattice:
  wedge :: Omega -> Omega -> Omega
  wedge = min
  top :: Omega
  top = 1.0 / 0.0

  -- Monoids:
  -- Monoid 1: Additive
  oplus :: Omega -> Omega -> Omega
  oplus = (+)
  v0 :: Omega
  v0 = 0.0

  -- Monoid 2: Multiplicative
  otimes :: Omega -> Omega -> Omega
  otimes = (*)
  v1 :: Omega
  v1 = 1.0
