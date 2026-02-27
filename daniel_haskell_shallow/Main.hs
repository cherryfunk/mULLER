{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.List (isPrefixOf)
-- 𝓘_Υ: Logical interpretation (Product logic)
import Logical.Interpretations.Product
import NonLogical.Categories.DATA (DATA (..))
-- 𝓘_Σ: Domain-specific interpretations
import NonLogical.Interpretations.Countable
import NonLogical.Interpretations.Crossing
import NonLogical.Interpretations.Dice
import NonLogical.Interpretations.Weather
import NonLogical.Monads.Dist (Dist, expectDist)
import NonLogical.Monads.Giry (Giry, expectation)
import System.Environment (getArgs)

--------------------------------------------------------------------------------
-- 1. DICE (Dist monad)
--------------------------------------------------------------------------------
dieSen1 :: Dist Omega
dieSen1 = do
  x <- die
  return (x .== 6 `wedge` b2o (even x))

dieSen2 :: Dist Omega
dieSen2 = do
  p <- do x <- die; return (x .== 6)
  q <- do x <- die; return (b2o (even x))
  return (p `wedge` q)

--------------------------------------------------------------------------------
-- 2. CROSSING (Dist monad) — Uller paper
--    "For every crossing, only continue driving if there is a green light."
--    ∀x ∈ X(l := 🚦(x), d := 🚗(x, l)(¬true(d) ∨ l = 🟢))
--    still missing the universal quantifier
--------------------------------------------------------------------------------
crossingSen :: Dist Omega
crossingSen = do
  l <- lightDetector
  d <- drivingDecision l
  return (neg (d .== 1) `vee` l .== "Green")

--------------------------------------------------------------------------------
-- 3. WEATHER (Giry monad) - DeepSeaProbLog paper
--------------------------------------------------------------------------------
weatherSen1 :: Giry Omega
weatherSen1 = do
  h <- bernoulli (humidDetect data1)
  t <- normalDist (tempPredict data1)
  return $
    (h .== 1 `wedge` t .< 0.0)
      `vee` (h .== 0 `wedge` t .> 15.0)

--------------------------------------------------------------------------------
-- 4. COUNTABLE SETS (Giry monad)
--------------------------------------------------------------------------------
countableSen1 :: Giry Omega
countableSen1 = do
  x <- drawInt
  y <- drawStr
  return (x .> 3 `wedge` b2o (isPrefixOf "TT" y))

countableSenLazy :: Giry Omega
countableSenLazy = do
  x <- drawLazy
  return (b2o (even x))

countableSenHeavy :: Giry Omega
countableSenHeavy = do
  x <- drawHeavy
  return top

--------------------------------------------------------------------------------
-- EXECUTION
--------------------------------------------------------------------------------
main :: IO ()
main = do
  args <- getArgs
  case args of
    ["baseline"] -> return ()
    ["benchmark-weather"] -> do
      print (expectation Reals weatherSen1 id)
    ["benchmark-countable"] -> do
      print (expectation Reals countableSen1 id)
    ["benchmark-countable-lazy"] -> do
      print (expectation Reals countableSenLazy id)
    ["benchmark-countable-heavy"] -> do
      print (expectation Reals countableSenHeavy id)
    _ -> do
      putStrLn "--- Testing mULLER Framework (SHALLOW, Product Logic) ---"

      putStrLn "\n[DICE] P(die == 6 ∧ even(die))"
      print (expectDist dieSen1 id)

      putStrLn "\n[DICE] P(die == 6) ∧ P(even(die))"
      print (expectDist dieSen2 id)

      putStrLn "\n[CROSSING] P(¬true(d) ∨ l = Green)"
      print (expectDist crossingSen id)

      putStrLn "\n[WEATHER] P((h=1 ∧ t<0) ∨ (h=0 ∧ t>15))"
      print (expectation Reals weatherSen1 id)

      putStrLn "\n[COUNTABLE] P(x > 3 ∧ isPrefixOf \"TT\" y)"
      print (expectation Reals countableSen1 id)
