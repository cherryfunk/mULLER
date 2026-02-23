{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeSySystem.Interpretations.Weather (weatherInterp) where

import Data.Typeable (cast)
import NeSyFramework.Categories.DATA (DataObj (..))
import NeSyFramework.Monads.Giry (Giry, categorical, normal)
import NeSySystem.Signatures.Weather (weatherSig)
import Semantics (DynVal (..), Interpretation (..), SomeObj (..))

-- Dummy Worlds type
data World = World deriving (Show, Eq)

humid_detector :: World -> Double
humid_detector _ = 0.5

temperature_predictor :: World -> (Double, Double)
temperature_predictor _ = (0.0, 2.0)

weatherInterp :: Interpretation Giry Double
weatherInterp =
  Interpretation
    { sig = weatherSig,
      interpSort = \case
        "Worlds" -> SomeObj (Finite [World])
        "Unit_Interval" -> SomeObj Reals
        "Reals2" -> SomeObj Reals -- Using Reals as a proxy for the product space
        "Reals" -> SomeObj Reals
        "Humidity" -> SomeObj (Finite [0 :: Int, 1])
        "Temperature" -> SomeObj Reals
        s -> error $ "Weather: unknown sort " ++ s,
      interpFunc = \case
        "data1" -> \[] -> DynVal World
        "humid_detector" -> \[DynVal d] -> case cast d of
          Just (w :: World) -> DynVal (humid_detector w)
          _ -> error "Expected World"
        "temperature_predictor" -> \[DynVal d] -> case cast d of
          Just (w :: World) -> DynVal (temperature_predictor w)
          _ -> error "Expected World"
        f -> error $ "Weather: unknown func " ++ f,
      interpRel = \case
        "==" -> \[DynVal x, DynVal y] -> case (cast x, cast y) of
          (Just (a :: Int), Just (b :: Int)) -> if a == b then 1.0 else 0.0
          _ -> 0.0
        "<" -> \[DynVal x, DynVal y] -> case (cast x, cast y) of
          (Just (a :: Double), Just (b :: Double)) ->
            1.0 / (1.0 + exp (100.0 * (a - b)))
          _ -> 0.0
        ">" -> \[DynVal x, DynVal y] -> case (cast x, cast y) of
          (Just (a :: Double), Just (b :: Double)) ->
            1.0 / (1.0 + exp (-100.0 * (a - b)))
          _ -> 0.0
        r -> error $ "Weather: unknown rel " ++ r,
      interpMFunc = \case
        "bernoulli" -> \[DynVal p] -> case cast p of
          Just (prob :: Double) -> categorical [(DynVal (1 :: Int), prob), (DynVal (0 :: Int), 1.0 - prob)]
          _ -> error "Expected Double for bernoulli prob"
        "normal" -> \[DynVal pair] -> case cast pair of
          Just (mu :: Double, sigma :: Double) -> fmap DynVal (normal mu sigma)
          _ -> error "Expected (Double, Double) pair"
        mf -> error $ "Weather: unknown mfunc " ++ mf,
      interpMRel = \mr _ -> error $ "Weather: unknown mrel " ++ mr
    }
