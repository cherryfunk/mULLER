{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

-- | Logical interpretation: Gödel Logic
module Logical.Interpretations.Goedel (Omega) where

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
  -- Monoid 1: Max
  oplus :: Omega -> Omega -> Omega
  oplus = max
  v0 :: Omega
  v0 = 0.0

  -- Monoid 2: Min
  otimes :: Omega -> Omega -> Omega
  otimes = min
  v1 :: Omega
  v1 = 1.0
