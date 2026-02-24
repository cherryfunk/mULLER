{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

-- | Logical interpretation: Classical Boolean Logic
module Logical.Interpretations.Boolean (Omega) where

import Logical.Signatures.TwoMonBLat

-- | Ω := I(τ) = {True, False}
type Omega = Bool

instance TwoMonBLat Omega where
  -- Comparison: (False <= True)
  vdash :: Omega -> Omega -> Bool
  vdash = (<=)

  -- Bounded Lattice:
  -- Join Lattice:
  vee :: Omega -> Omega -> Omega
  vee = (||)
  bot :: Omega
  bot = False

  -- Meet Lattice:
  wedge :: Omega -> Omega -> Omega
  wedge = (&&)
  top :: Omega
  top = True

  -- Monoids:
  -- Monoid 1: Disjunction
  oplus :: Omega -> Omega -> Omega
  oplus = (||)
  v0 :: Omega
  v0 = False

  -- Monoid 2: Conjunction
  otimes :: Omega -> Omega -> Omega
  otimes = (&&)
  v1 :: Omega
  v1 = True
