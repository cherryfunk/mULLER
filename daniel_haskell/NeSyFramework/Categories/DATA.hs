{-# LANGUAGE GADTs #-}
{-# LANGUAGE StandaloneDeriving #-}

module NeSyFramework.Categories.DATA where

import Data.Typeable (Typeable)

-- | The category DATA
-- Objects are specific sets: infinite sets (Reals, Integers) or finite enumerations.
-- This module defines objects ONLY. Integration strategies are the Giry monad's responsibility.
data DataObj a where
  -- | The real line ℝ (approximated by IEEE 754 Double)
  Reals :: DataObj Double
  -- | The integers ℤ
  Integers :: DataObj Int
  -- | The two-element set {True, False}
  Booleans :: DataObj Bool
  -- | A specific finite "set" (list), e.g. Finite [1,2,3,4,5,6] or Finite ["Red","Green","Yellow"]
  Finite :: (Eq a, Show a, Typeable a) => [a] -> DataObj a
  -- | Finite products
  -- | The terminal object (empty product)
  UnitObj :: DataObj ()
  -- | Recursive binary product of two objects
  ProductObj :: DataObj a -> DataObj b -> DataObj (a, b)
