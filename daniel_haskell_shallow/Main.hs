{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Data.List (isPrefixOf)
-- \$\mathcal{I}_\Upsilon$: Logical interpretation (Product logic)

-- \$\mathcal{I}_\Sigma$: Domain-specific interpretations

import qualified Data.Map as Map
import Formulas (loadFormulas)
import Logical.Interpretations.Boolean
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Interpretations.Countable
import NonLogical.Interpretations.Crossing
import NonLogical.Interpretations.Dice
import NonLogical.Interpretations.Weather
import NonLogical.Monads.Dist (Dist)
import NonLogical.Monads.Expectation (HasExpectation (..), probDist, probGiry)
import NonLogical.Monads.Giry (Giry)
import System.Environment (getArgs)

------------------------------------------------------

------------------------------------------------------
-- 1. DICE (Dist monad)
------------------------------------------------------
dieSen1 :: Dist Omega
dieSen1 = do
  x <- die
  return (x .== 6 `wedge` b2o (even x))

dieExp1 = probDist dieSen1

dieSen2 :: Dist Omega
dieSen2 = do
  p <- do x <- die; return (x .== 6)
  q <- do x <- die; return (b2o (even x))
  return (p `wedge` q)

dieExp2 = probDist dieSen2

------------------------------------------------------
-- 2. CROSSING (Dist monad) -- Uller paper
--    "For every crossing, only continue driving if there is a green light."
--    $\forall x \in X(l := \text{traffic\_light}(x),\; d := \text{car}(x, l))\;(\neg\text{true}(d) \vee l = \text{green})$
--    still missing the universal quantifier
------------------------------------------------------
crossingSen :: Dist Omega
crossingSen = do
  l <- lightDetector
  d <- drivingDecision l
  return (neg (d .== 1) `vee` l .== "Green")

crossingExp :: Double
crossingExp = probDist crossingSen

------------------------------------------------------
-- 3. WEATHER (Giry monad) - DeepSeaProbLog paper
------------------------------------------------------

-- | Weather scenario 1: "it is humid AND hot (t > 30)"
weatherSen1 :: Giry Omega
weatherSen1 = do
  h <- bernoulli (humidDetect data1)
  t <- normalDist (tempPredict data1)
  return (h .== 1 `wedge` t .> 30.0)

weatherExp1 :: Double
weatherExp1 = probGiry weatherSen1

-- | Weather scenario 2: "it is humid AND warm (t > 25)"
--   Since (t > 30) implies (t > 25), we expect P(sen1) <= P(sen2).
weatherSen2 :: Giry Omega
weatherSen2 = do
  h <- bernoulli (humidDetect data1)
  t <- normalDist (tempPredict data1)
  return (h .== 1 `wedge` t .> 25.0)

weatherExp2 :: Double
weatherExp2 = probGiry weatherSen2

-- | Natural entailment: "humid and very hot" entails "humid and warm"
--   p(weatherSen1) <= p(weatherSen2)
weatherEntails :: Bool
weatherEntails = probGiry weatherSen1 <= probGiry weatherSen2

------------------------------------------------------
-- 4. COUNTABLE SETS (Giry monad)
------------------------------------------------------
countableSen1 :: Giry Omega
countableSen1 = do
  x <- drawInt
  y <- drawStr
  return (x .> 3 `wedge` b2o (isPrefixOf "TT" y))

countableExp1 :: Double
countableExp1 = probGiry countableSen1

countableSenLazy :: Giry Omega
countableSenLazy = do
  x <- drawLazy
  return (b2o (even x))

countableExpLazy :: Double
countableExpLazy = probGiry countableSenLazy

countableSenHeavy :: Giry Omega
countableSenHeavy = do
  x <- drawHeavy
  return top

countableExpHeavy :: Double
countableExpHeavy = probGiry countableSenHeavy

------------------------------------------------------
-- EXECUTION
------------------------------------------------------
main :: IO ()
main = do
  args <- getArgs
  case args of
    ["baseline"] -> return ()
    ["benchmark-weather"] -> do
      print weatherExp1
    ["benchmark-countable"] -> do
      print countableExp1
    ["benchmark-countable-lazy"] -> do
      print countableExpLazy
    ["benchmark-countable-heavy"] -> do
      print countableExpHeavy
    _ -> do
      putStrLn "-- Testing mULLER Framework (SHALLOW, Product Logic) --"

      forms <- loadFormulas "../ULLER_paper/3. NeSyCat PyTorch/Conference Paper (NeSy26)/nesy2026-paper.tex"
      let getF name = Map.findWithDefault name name forms

      putStrLn $ "\n[DICE] " ++ getF "fDiceOne"
      print dieExp1

      putStrLn $ "\n[DICE] " ++ getF "fDiceTwo"
      print dieExp2

      putStrLn $ "\n[CROSSING] " ++ getF "fCrossing"
      print crossingExp

      putStrLn $ "\n[WEATHER 1 - Berlin] " ++ getF "fWeather"
      print weatherExp1

      putStrLn $ "\n[WEATHER 2 - Hamburg] " ++ getF "fWeather"
      print weatherExp2

      putStrLn "\n[WEATHER] Berlin entails Hamburg?"
      print weatherEntails

      putStrLn $ "\n[COUNTABLE] " ++ getF "fCountable"
      print countableExp1
