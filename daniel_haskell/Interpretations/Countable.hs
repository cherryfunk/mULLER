{-# LANGUAGE ScopedTypeVariables #-}

module Interpretations.Countable (countableInterp) where

import Data.List (isPrefixOf)
import qualified Data.Map as Map
import Data.Typeable (cast)
import NeSyFramework.Monads.Giry (Giry, categorical)
import Semantics (DynVal (..), Interpretation (..))

-- | Infinite geometric distribution over Int
-- P(k) = (1-p)^k * p
drawIntMeasure :: Giry DynVal
drawIntMeasure =
  let p = 0.5
      probs = [(DynVal (k :: Int), (1 - p) ^ k * p) | k <- [0 ..]]
   in categorical probs

-- | Infinite geometric distribution over String
-- Simulate flipping a coin until it lands on 'H'
-- "H" (1/2), "TH" (1/4), "TTH" (1/8)...
drawStrMeasure :: Giry DynVal
drawStrMeasure =
  let p = 0.5
      toStr k = replicate k 'T' ++ "H"
      probs = [(DynVal (toStr k), (1 - p) ^ k * p) | k <- [0 ..]]
   in categorical probs

-- | Non-Geometric but fast decaying (Zeta Distribution with s=3)
-- Decays as 1/k^3. This will cause algebraicSimplify to fail (non-constant ratio),
-- but will converge within the Lazy Bounded Loop.
drawLazyMeasure :: Giry DynVal
drawLazyMeasure =
  let zeta3 = 1.202056903159594
      probs = [(DynVal (k :: Int), (1 / fromIntegral (k + 1) ** 3) / zeta3) | k <- [0 ..]]
   in categorical probs

-- | Heavy-tailed distribution (Zeta Distribution with s=1.1)
-- Decays very slowly (1/k^1.1). This will exceed the 10,000 iteration limit
-- before reaching giryQuadPrecision, forcing the Monte Carlo fallback.
drawHeavyMeasure :: Giry DynVal
drawHeavyMeasure =
  let zeta11 = 10.5844484649508
      probs = [(DynVal (k :: Int), (1 / fromIntegral (k + 1) ** 1.1) / zeta11) | k <- [0 ..]]
   in categorical probs

-- | An interpretation showing infinite sets in action
countableInterp :: Interpretation Giry Double
countableInterp =
  Interpretation
    { funcs = Map.empty,
      rels =
        Map.fromList
          [ ( ">3",
              \[DynVal x] -> case cast x of
                Just (a :: Int) -> if a > 3 then 1.0 else 0.0
                _ -> 0.0
            ),
            ( "startsTT",
              \[DynVal s] -> case cast s of
                Just (str :: String) -> if "TT" `isPrefixOf` str then 1.0 else 0.0
                _ -> 0.0
            ),
            ( "isEven",
              \[DynVal x] -> case cast x of
                Just (a :: Int) -> if even a then 1.0 else 0.0
                _ -> 0.0
            ),
            ( "isAnything",
              \[_] -> 1.0
            )
          ],
      mfuncs =
        Map.fromList
          [ ("drawInt", \[] -> drawIntMeasure),
            ("drawStr", \[] -> drawStrMeasure),
            ("drawLazy", \[] -> drawLazyMeasure),
            ("drawHeavy", \[] -> drawHeavyMeasure)
          ],
      mrels = Map.empty
    }
