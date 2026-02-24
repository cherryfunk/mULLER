{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE StandaloneDeriving #-}

module NonLogical.Categories.DATA where

import Data.Kind (Type)
import Data.Typeable (Typeable)

-- | MonadOver: declares that monad m lives on category cat.
-- This is a bare association — no methods. Each monad defines its own
-- evaluation/integration logic separately. This just prevents misuse:
-- you can't instantiate a signature with a category-monad pair unless
-- the monad is registered as living on that category.
class (Monad m) => MonadOver (cat :: Type -> Type) (m :: Type -> Type)

-- | The category DATA
-- Objects are specific sets: infinite sets (Reals, Integers) or finite enumerations.
-- 'DATA a' means "a is an object in the DATA category."
data DATA a where
  -- | The real line ℝ (approximated by IEEE 754 Double)
  Reals :: DATA Double
  -- | The integers ℤ
  Integers :: DATA Int
  -- | The two-element set {True, False}
  Booleans :: DATA Bool
  -- | The set of all strings (countably infinite)
  Strings :: DATA String
  -- | A specific finite "set" (list), e.g. Finite [1,2,3,4,5,6] or Finite ["Red","Green","Yellow"]
  Finite :: (Eq a, Show a, Typeable a) => [a] -> DATA a
  -- | Finite products
  -- | The terminal object (empty product)
  UnitObj :: DATA ()
  -- | Recursive binary product of two objects
  ProductObj :: DATA a -> DATA b -> DATA (a, b)

-- | Generic table lookup (the "WHERE key = k" query).
--   tableLookup keyOf k rows — find the first row whose key matches k.
tableLookup :: (Eq k, Show k) => (row -> k) -> k -> [row] -> row
tableLookup keyOf k rows = case filter (\r -> keyOf r == k) rows of
  (r : _) -> r
  [] -> error $ "No such key: " ++ show k
