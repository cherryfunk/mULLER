{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeSySystem.Interpretations.TrafficLight (trafficInterp) where

import Data.Typeable (cast)
import NeSyFramework.Categories.DATA (DataObj (..))
import NeSyFramework.Monads.Giry (Giry, categorical)
import NeSySystem.Signatures.TrafficLight (trafficSig)
import Semantics (DynVal (..), Interpretation (..), SomeObj (..))

trafficInterp :: Interpretation Giry Double
trafficInterp =
  Interpretation
    { sig = trafficSig,
      interpSort = \case
        "Light" -> SomeObj (Finite ["Red" :: String, "Green", "Yellow"])
        "Drive" -> SomeObj Booleans
        s -> error $ "TrafficLight: unknown sort " ++ s,
      interpFunc = \f _ -> error $ "TrafficLight: unknown func " ++ f,
      interpRel = \case
        "==" -> \[DynVal x, DynVal y] -> case (cast x, cast y) of
          (Just (a :: String), Just (b :: String)) -> if a == b then 1.0 else 0.0
          _ -> 0.0
        "eval" -> \[DynVal b] -> case cast b of
          Just True -> 1.0
          Just False -> 0.0
          _ -> 0.0
        r -> error $ "TrafficLight: unknown rel " ++ r,
      interpMFunc = \case
        "light" -> \[] ->
          categorical
            [ (DynVal ("Red" :: String), 0.6),
              (DynVal ("Green" :: String), 0.3),
              (DynVal ("Yellow" :: String), 0.1)
            ]
        "driveF" -> \[DynVal l] -> case cast l of
          Just "Red" -> categorical [(DynVal True, 0.1), (DynVal False, 0.9)]
          Just "Yellow" -> categorical [(DynVal True, 0.2), (DynVal False, 0.8)]
          Just "Green" -> categorical [(DynVal True, 0.9), (DynVal False, 0.1)]
          _ -> error "Expected Traffic Light String"
        mf -> error $ "TrafficLight: unknown mfunc " ++ mf,
      interpMRel = \case
        "driveP" -> \[DynVal l] -> case cast l of
          Just "Red" -> categorical [(1.0, 0.1), (0.0, 0.9)]
          Just "Yellow" -> categorical [(1.0, 0.2), (0.0, 0.8)]
          Just "Green" -> categorical [(1.0, 0.9), (0.0, 0.1)]
          _ -> error "Expected Traffic Light String"
        mr -> error $ "TrafficLight: unknown mrel " ++ mr
    }
