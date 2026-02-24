{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Non-logical interpretation: Countable sets domain (in DATA, with Giry monad)
module NonLogical.Interpretations.Countable where

import Data.List (isPrefixOf)
import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA)
import NonLogical.Monads.Giry (Giry, categorical)
import NonLogical.Signatures.CountableSig

instance CountableSig DATA Giry where
  -- Sor

  -- Const

  -- Fun

  -- mFun
  drawInt =
    let p = 0.5
        probs = [(k :: Int, (1 - p) ^ k * p) | k <- [0 ..]]
     in categorical probs
  drawStr =
    let p = 0.5
        toStr k = replicate k 'T' ++ "H"
        probs = [(toStr k, (1 - p) ^ k * p) | k <- [0 ..]]
     in categorical probs
  drawLazy =
    let zeta3 = 1.202056903159594
        probs = [(k :: Int, (1 / fromIntegral (k + 1) ** 3) / zeta3) | k <- [0 ..]]
     in categorical probs
  drawHeavy =
    let zeta11 = 10.5844484649508
        probs = [(k :: Int, (1 / fromIntegral (k + 1) ** 1.1) / zeta11) | k <- [0 ..]]
     in categorical probs

  -- Rel
  gt3 a = if a > 3 then v1 else v0
  startsTT s = if isPrefixOf "TT" s then v1 else v0
  isEven x = if even x then v1 else v0
  isAnything _ = v1

-- mRel
