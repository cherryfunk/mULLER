{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.List (isPrefixOf)
-- 𝓘_Υ: Logical interpretation (Product logic, includes .==, .<, .>)
import Logical.Interpretations.Product
import NonLogical.Categories.DATA (DATA (..))
-- 𝓘_Σ: Domain-specific interpretations
import NonLogical.Interpretations.Countable
import NonLogical.Interpretations.Dice
import NonLogical.Interpretations.TrafficLight
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
-- 2. TRAFFIC LIGHT (Giry monad)
--------------------------------------------------------------------------------
trafficSen1 :: Giry Omega
trafficSen1 = do
  l <- light
  return (l .== "Green")

--------------------------------------------------------------------------------
-- 3. WEATHER (Giry monad)
--------------------------------------------------------------------------------
weatherSen1 :: Giry Omega
weatherSen1 = do
  h <- bernoulli (humid_detector data1)
  t <- normalDist (temperature_predictor data1)
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

      putStrLn "\n[DICE] Evaluating P(die == 6 AND even(die))"
      print (expectDist dieSen1 id)

      putStrLn "\n[DICE] Evaluating P(die == 6) AND P(even(die))"
      print (expectDist dieSen2 id)

      putStrLn "\n[TRAFFIC] Evaluating P(light == green)"
      print (expectation Reals trafficSen1 id)

      putStrLn "\n[WEATHER] Evaluating P(vee (wedge (h=1) (t<0)) (wedge (h=0) (t>15)))"
      print (expectation Reals weatherSen1 id)

      putStrLn "\n[COUNTABLE] Evaluating P(wedge (int > 3) (string starts with TT))"
      print (expectation Reals countableSen1 id)
