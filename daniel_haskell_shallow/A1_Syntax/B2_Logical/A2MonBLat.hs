{-# LANGUAGE RankNTypes #-}

module A1_Syntax.B2_Logical.A2MonBLat where

import A3_Semantics.B3_NonLogical.Categories.DATA (DATA)

-- | Theory of an aggregated 2-monoid bounded lattice (A2Mon-BLat).
--   Extends 2Mon-BLat with four infinitary quantifiers.
class A2MonBLat tau where
  -- Infinitary Lattice:
  bigVee :: forall a. DATA a -> (a -> tau) -> tau
  bigWedge :: forall a. DATA a -> (a -> tau) -> tau

  -- Infinitary Monoids:
  bigOplus :: forall a. DATA a -> (a -> tau) -> tau
  bigOtimes :: forall a. DATA a -> (a -> tau) -> tau
