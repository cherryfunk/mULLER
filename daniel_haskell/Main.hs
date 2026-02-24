{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Map as Map
import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Interpretations.Countable (drawHeavy', drawInt', drawLazy', drawStr', gt3', isAnything', isEven', startsTT')
import NonLogical.Interpretations.Dice (die', eqDie', evenDie')
import NonLogical.Interpretations.TrafficLight (eqLight', light')
import NonLogical.Interpretations.Weather (eqI', gtD', humid', ltD', temp')
import NonLogical.Monads.Dist (Dist, expectDist)
import NonLogical.Monads.Giry (Giry, expectation)
import System.Environment (getArgs)
import TypedSemantics
import TypedSyntax

--------------------------------------------------------------------------------
-- 1. DICE
--------------------------------------------------------------------------------
dieSen1 :: Formula Omega
dieSen1 =
  bind "x" die' $
    wedge (eqDie' (var "x") 6) (evenDie' (var "x"))

dieSen2 :: Formula Omega
dieSen2 =
  wedge
    (bind "x" die' (eqDie' (var "x") 6))
    (bind "x" die' (evenDie' (var "x")))

--------------------------------------------------------------------------------
-- 2. TRAFFIC LIGHT
--------------------------------------------------------------------------------
trafficSen1 :: Formula Omega
trafficSen1 =
  bind "l" light' $
    eqLight' (var "l") (con "Green")

--------------------------------------------------------------------------------
-- 3. WEATHER
--------------------------------------------------------------------------------
weatherSen1 :: Formula Omega
weatherSen1 =
  bind "h" humid' $
    bind "t" temp' $
      vee
        (wedge (eqI' (var "h") 1) (ltD' (var "t") (con 0.0)))
        (wedge (eqI' (var "h") 0) (gtD' (var "t") (con 15.0)))

--------------------------------------------------------------------------------
-- 4. COUNTABLE SETS
--------------------------------------------------------------------------------
countableSen1 :: Formula Omega
countableSen1 =
  bind "x" drawInt' $
    bind "y" drawStr' $
      wedge (gt3' (var "x")) (startsTT' (var "y"))

countableSenLazy :: Formula Omega
countableSenLazy =
  bind "x" drawLazy' $
    isEven' (var "x")

countableSenHeavy :: Formula Omega
countableSenHeavy =
  bind "x" drawHeavy' $
    isAnything' (var "x")

--------------------------------------------------------------------------------
-- EXECUTION
--------------------------------------------------------------------------------
main :: IO ()
main = do
  args <- getArgs
  case args of
    ["baseline"] -> return ()
    ["benchmark-weather"] -> do
      let w1 = evalFormula @Giry @Omega @Omega Map.empty weatherSen1
      print (expectation Reals w1 id)
    ["benchmark-countable"] -> do
      let c1 = evalFormula @Giry @Omega @Omega Map.empty countableSen1
      print (expectation Reals c1 id)
    ["benchmark-countable-lazy"] -> do
      let c1 = evalFormula @Giry @Omega @Omega Map.empty countableSenLazy
      print (expectation Reals c1 id)
    ["benchmark-countable-heavy"] -> do
      let c1 = evalFormula @Giry @Omega @Omega Map.empty countableSenHeavy
      print (expectation Reals c1 id)
    _ -> do
      putStrLn "--- Testing mULLER Framework Domain Interpretations ---"

      putStrLn "\n[DICE] Evaluating P(die == 6 AND even(die))"
      let d1 = evalFormula @Dist @Omega @Omega Map.empty dieSen1
      print (expectDist d1 id)

      putStrLn "\n[DICE] Evaluating P(die == 6) AND P(even(die))"
      let d2 = evalFormula @Dist @Omega @Omega Map.empty dieSen2
      print (expectDist d2 id)

      putStrLn "\n[TRAFFIC] Evaluating P(light == green)"
      let t1 = evalFormula @Giry @Omega @Omega Map.empty trafficSen1
      print (expectation Reals t1 id)

      putStrLn "\n[WEATHER] Evaluating P(vee (wedge (h=1) (t<0)) (wedge (h=0) (t>15)))"
      let w1 = evalFormula @Giry @Omega @Omega Map.empty weatherSen1
      print (expectation Reals w1 id)

      putStrLn "\n[COUNTABLE] Evaluating P(wedge (int > 3) (string starts with TT))"
      let c1 = evalFormula @Giry @Omega @Omega Map.empty countableSen1
      print (expectation Reals c1 id)
