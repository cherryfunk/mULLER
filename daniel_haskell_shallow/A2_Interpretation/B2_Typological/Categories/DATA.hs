{-# LANGUAGE GADTs #-}
{-# LANGUAGE StandaloneDeriving #-}

module A2_Interpretation.B2_Typological.Categories.DATA where

import Data.Typeable (Typeable)
import Numeric.Natural (Natural)

-- | The category DATA
-- Objects are specific sets: infinite sets (Reals, Integers) or finite enumerations.
-- 'DATA a' means "a is an object in the DATA category."
data DATA a where
  -- | Base types (matching DataVocab order):
  Booleans :: DATA Bool
  Naturals :: DATA Natural
  Integers :: DATA Integer -- maybe instead use Int for speed?
  Strings :: DATA String
  Reals :: DATA Double
  -- | A specific finite "set" (list), e.g. Finite [1,2,3,4,5,6]
  Finite :: (Eq a, Show a, Typeable a) => [a] -> DATA a
  -- | Constructed types (matching DataVocab order):
  Unit :: DATA ()
  Prod :: DATA a -> DATA b -> DATA (a, b)
  Lists :: DATA a -> DATA [a]

-- | Generic table lookup (the "WHERE key = k" query).
--   tableLookup keyOf k rows -- find the first row whose key matches k.
tableLookup :: (Eq k, Show k) => (row -> k) -> k -> [row] -> row
tableLookup keyOf k rows = case filter (\r -> keyOf r == k) rows of
  (r : _) -> r
  [] -> error $ "No such key: " ++ show k
