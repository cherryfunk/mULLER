{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main where

import qualified Data.Map as Map
import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Interpretations.Countable ()
import NonLogical.Interpretations.Dice ()
import NonLogical.Interpretations.TrafficLight ()
import NonLogical.Interpretations.Weather ()
import NonLogical.Monads.Dist (Dist, expectDist)
import NonLogical.Monads.Giry (Giry, expectation)
import NonLogical.Signatures.CountableSig
import NonLogical.Signatures.DiceSig
import NonLogical.Signatures.TrafficLightSig
import NonLogical.Signatures.WeatherSig
import System.Environment (getArgs)
import TypedSemantics
import TypedSyntax

--------------------------------------------------------------------------------
-- 1. DICE EXAMPLE (in DATA, with Dist)
--------------------------------------------------------------------------------
dieSen1 :: Formula Omega
dieSen1 =
  Compu "x" (Con (die @DATA @Dist)) $
    BinConn
      wedge
      (Rel (Con (eqDie @DATA @Dist) `Fun` Var @Int "x" `Fun` Con (6 :: Int)))
      (Rel (Con (evenDie @DATA @Dist) `Fun` Var @Int "x"))

dieSen2 :: Formula Omega
dieSen2 =
  BinConn
    wedge
    (Compu "x" (Con (die @DATA @Dist)) (Rel (Con (eqDie @DATA @Dist) `Fun` Var @Int "x" `Fun` Con (6 :: Int))))
    (Compu "x" (Con (die @DATA @Dist)) (Rel (Con (evenDie @DATA @Dist) `Fun` Var @Int "x")))

--------------------------------------------------------------------------------
-- 2. TRAFFIC LIGHT EXAMPLE (in DATA, with Giry)
--------------------------------------------------------------------------------
trafficSen1 :: Formula Omega
trafficSen1 =
  Compu "l" (Con (light @DATA @Giry)) $
    Rel (Con (eqLight @DATA @Giry) `Fun` Var @String "l" `Fun` Con ("Green" :: String))

--------------------------------------------------------------------------------
-- 3. WEATHER EXAMPLE (in DATA, with Giry)
--------------------------------------------------------------------------------
weatherSen1 :: Formula Omega
weatherSen1 =
  Compu "h" (Con (bernoulli @DATA @Giry) `Fun` (Con (humid_detector @DATA @Giry) `Fun` Con (data1 @DATA @Giry))) $
    Compu "t" (Con (normalDist @DATA @Giry) `Fun` (Con (temperature_predictor @DATA @Giry) `Fun` Con (data1 @DATA @Giry))) $
      BinConn
        vee
        ( BinConn
            wedge
            (Rel (Con (eqInt @DATA @Giry) `Fun` Var @Int "h" `Fun` Con (1 :: Int)))
            (Rel (Con (ltDouble @DATA @Giry) `Fun` Var @Double "t" `Fun` Con (0.0 :: Double)))
        )
        ( BinConn
            wedge
            (Rel (Con (eqInt @DATA @Giry) `Fun` Var @Int "h" `Fun` Con (0 :: Int)))
            (Rel (Con (gtDouble @DATA @Giry) `Fun` Var @Double "t" `Fun` Con (15.0 :: Double)))
        )

--------------------------------------------------------------------------------
-- 4. COUNTABLE SETS EXAMPLE (in DATA, with Giry)
--------------------------------------------------------------------------------
countableSen1 :: Formula Omega
countableSen1 =
  Compu "x" (Con (drawInt @DATA @Giry)) $
    Compu "y" (Con (drawStr @DATA @Giry)) $
      BinConn
        wedge
        (Rel (Con (gt3 @DATA @Giry) `Fun` Var @Int "x"))
        (Rel (Con (startsTT @DATA @Giry) `Fun` Var @String "y"))

countableSenLazy :: Formula Omega
countableSenLazy =
  Compu "x" (Con (drawLazy @DATA @Giry)) $
    Rel (Con (isEven @DATA @Giry) `Fun` Var @Int "x")

countableSenHeavy :: Formula Omega
countableSenHeavy =
  Compu "x" (Con (drawHeavy @DATA @Giry)) $
    Rel (Con (isAnything @DATA @Giry) `Fun` Var @Int "x")

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
