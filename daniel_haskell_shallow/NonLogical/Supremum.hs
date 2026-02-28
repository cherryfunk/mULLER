{-# LANGUAGE GADTs #-}

-- | Supremum and Infimum over DATA objects.
--   Analogous to Expectation.hs (which provides the generalized sum / integral),
--   this module provides the generalized maximum (supremum) and minimum (infimum):
--
--     sup  ::  DATA a  ->  (a -> Double) -> Double     -- sup_a phi(a)
--     inf  ::  DATA a  ->  (a -> Double) -> Double     -- inf_a phi(a)
--
--   For finite / countably infinite objects, these use lazy bounded convergence.
--   For R (uncountable), these require numerical optimization and are not yet supported.
module NonLogical.Supremum
  ( sup,
    inf,
    enumAll,
  )
where

import NonLogical.Categories.DATA (DATA (..))

------------------------------------------------------
-- Enumeration
------------------------------------------------------

-- | Canonical enumeration of Z: [0, 1, -1, 2, -2, ...]
enumIntegers :: [Int]
enumIntegers = 0 : concatMap (\n -> [n, -n]) [1 ..]

-- | Canonical enumeration of all strings: ["", "a", "b", ..., "aa", ...]
enumStrings :: [String]
enumStrings = "" : [c : s | s <- enumStrings, c <- ['a' .. 'z']]

-- | Enumerate all elements of a DATA object (finite or lazily infinite).
enumAll :: DATA a -> [a]
enumAll (Finite xs) = xs
enumAll Booleans = [True, False]
enumAll Unit = [()]
enumAll Integers = enumIntegers
enumAll Strings = enumStrings
enumAll (Prod da db) = [(a, b) | a <- enumAll da, b <- enumAll db]
enumAll Reals = error "Cannot enumerate R."

------------------------------------------------------
-- Budget for lazy convergence
------------------------------------------------------

maxBudget :: Int
maxBudget = 10000

------------------------------------------------------
-- sup / inf
------------------------------------------------------

-- | Supremum: sup_a phi(a) = max { phi(a) | a in D }
--   For countable sets, uses lazy bounded convergence.
sup :: DATA a -> (a -> Double) -> Double
sup Reals _ = error "sup over R requires numerical optimization."
sup d phi = lazyFold max (-(1.0 / 0.0)) (map phi (enumAll d))

-- | Infimum: inf_a phi(a) = min { phi(a) | a in D }
--   For countable sets, uses lazy bounded convergence.
inf :: DATA a -> (a -> Double) -> Double
inf Reals _ = error "inf over R requires numerical optimization."
inf d phi = lazyFold min (1.0 / 0.0) (map phi (enumAll d))

------------------------------------------------------
-- Internal: lazy fold with budget
------------------------------------------------------

lazyFold :: (b -> a -> b) -> b -> [a] -> b
lazyFold f = go 0
  where
    go _ acc [] = acc
    go n acc _ | n >= maxBudget = acc
    go n acc (x : xs) = go (n + 1) (f acc x) xs
