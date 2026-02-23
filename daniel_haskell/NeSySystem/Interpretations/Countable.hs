{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeSySystem.Interpretations.Countable (countableInterp) where

import Data.List (isPrefixOf)
import Data.Typeable (cast)
import NeSyFramework.Categories.DATA (DataObj (..))
import NeSyFramework.Monads.Giry (Giry, categorical)
import NeSySystem.Signatures.Countable (countableSig)
import Semantics (DynVal (..), Interpretation (..), SomeObj (..))

-- | Infinite geometric distribution over Int
drawIntMeasure :: Giry DynVal
drawIntMeasure =
  let p = 0.5
      probs = [(DynVal (k :: Int), (1 - p) ^ k * p) | k <- [0 ..]]
   in categorical probs

-- | Infinite geometric distribution over String
drawStrMeasure :: Giry DynVal
drawStrMeasure =
  let p = 0.5
      toStr k = replicate k 'T' ++ "H"
      probs = [(DynVal (toStr k), (1 - p) ^ k * p) | k <- [0 ..]]
   in categorical probs

-- | Non-Geometric but fast decaying (Zeta Distribution with s=3)
drawLazyMeasure :: Giry DynVal
drawLazyMeasure =
  let zeta3 = 1.202056903159594
      probs = [(DynVal (k :: Int), (1 / fromIntegral (k + 1) ** 3) / zeta3) | k <- [0 ..]]
   in categorical probs

-- | Heavy-tailed distribution (Zeta Distribution with s=1.1)
drawHeavyMeasure :: Giry DynVal
drawHeavyMeasure =
  let zeta11 = 10.5844484649508
      probs = [(DynVal (k :: Int), (1 / fromIntegral (k + 1) ** 1.1) / zeta11) | k <- [0 ..]]
   in categorical probs

countableInterp :: Interpretation Giry Double
countableInterp =
  Interpretation
    { sig = countableSig,
      interpSort = \case
        "Nat" -> SomeObj Integers
        "CoinSeq" -> SomeObj Integers
        s -> error $ "Countable: unknown sort " ++ s,
      interpFunc = \f _ -> error $ "Countable: unknown func " ++ f,
      interpRel = \case
        ">3" -> \[DynVal x] -> case cast x of
          Just (a :: Int) -> if a > 3 then 1.0 else 0.0
          _ -> 0.0
        "startsTT" -> \[DynVal s] -> case cast s of
          Just (str :: String) -> if "TT" `isPrefixOf` str then 1.0 else 0.0
          _ -> 0.0
        "isEven" -> \[DynVal x] -> case cast x of
          Just (a :: Int) -> if even a then 1.0 else 0.0
          _ -> 0.0
        "isAnything" -> \[_] -> 1.0
        r -> error $ "Countable: unknown rel " ++ r,
      interpMFunc = \case
        "drawInt" -> \[] -> drawIntMeasure
        "drawStr" -> \[] -> drawStrMeasure
        "drawLazy" -> \[] -> drawLazyMeasure
        "drawHeavy" -> \[] -> drawHeavyMeasure
        mf -> error $ "Countable: unknown mfunc " ++ mf,
      interpMRel = \mr _ -> error $ "Countable: unknown mrel " ++ mr
    }
