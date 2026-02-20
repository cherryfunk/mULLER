{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Map as Map
import Interpretations.Dice (diceInterp)
import Interpretations.TrafficLight (trafficInterp)
import Interpretations.Weather (weatherInterp)
import NeSyFramework.Monads.Giry
import NeSyFramework.TruthSpaces.Real ()
import Semantics
import Syntax
import System.Environment (getArgs)

--------------------------------------------------------------------------------
-- 1. DICE EXAMPLE
--------------------------------------------------------------------------------
-- x := die() (x == 6 ∧ even(x))
dieSen1 :: Formula
dieSen1 =
  Compu
    "x"
    "die"
    []
    (Wedge (Rel "==" [Var "x", Con (6 :: Int)]) (Rel "even" [Var "x"]))

-- (x := die() (x == 6)) ∧ (x := die() even(x))
dieSen2 :: Formula
dieSen2 =
  Wedge
    (Compu "x" "die" [] (Rel "==" [Var "x", Con (6 :: Int)]))
    (Compu "x" "die" [] (Rel "even" [Var "x"]))

--------------------------------------------------------------------------------
-- 2. TRAFFIC LIGHT EXAMPLE
--------------------------------------------------------------------------------
-- l := light(), d := driveF(l) (eval d -> l==green)
-- (Since we don't have Implies, we evaluate if l==green directly for the test)
trafficSen1 :: Formula
trafficSen1 = Compu "l" "light" [] (Rel "==" [Var "l", Con ("Green" :: String)])

--------------------------------------------------------------------------------
-- 3. WEATHER EXAMPLE (from NeSy.hs)
--------------------------------------------------------------------------------
-- h := bernoulli(humid_detector(1))
-- t := normal(temperature_predictor(1))
-- (h == 1 ∧ t < 0.0) ∨ (h == 0 ∧ t > 15.0)
weatherSen1 :: Formula
weatherSen1 =
  Compu "h" "bernoulli" [Fun "humid_detector" [Con (1 :: Int)]] $
    Compu "t" "normal" [Fun "temperature_predictor" [Con (1 :: Int)]] $
      Vee
        (Wedge (Rel "==" [Var "h", Con (1 :: Int)]) (Rel "<" [Var "t", Con (0.0 :: Double)]))
        (Wedge (Rel "==" [Var "h", Con (0 :: Int)]) (Rel ">" [Var "t", Con (15.0 :: Double)]))

--------------------------------------------------------------------------------
-- EXECUTION
--------------------------------------------------------------------------------
main :: IO ()
main = do
  args <- getArgs
  case args of
    ["baseline"] -> return ()
    ["benchmark"] -> do
      let w1 = evalFormula weatherInterp Map.empty weatherSen1
      print (expectation w1 id)
    _ -> do
      putStrLn "--- Testing mULLER Framework Domain Interpretations ---"

  putStrLn "\n[DICE] Evaluatin P(die == 6 AND even(die))"
  let d1 = evalFormula diceInterp Map.empty dieSen1
  print (expectation d1 id)

  putStrLn "\n[DICE] Evaluatin P(die == 6) AND P(even(die))"
  let d2 = evalFormula diceInterp Map.empty dieSen2
  print (expectation d2 id)

  putStrLn "\n[TRAFFIC] Evaluating P(light == green)"
  let t1 = evalFormula trafficInterp Map.empty trafficSen1
  print (expectation t1 id)

  putStrLn "\n[WEATHER] Evaluating P((h=1 ∧ t<0) ∨ (h=0 ∧ t>15))"
  let w1 = evalFormula weatherInterp Map.empty weatherSen1
  print (expectation w1 id)
