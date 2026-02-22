{-# LANGUAGE ScopedTypeVariables #-}

module NeSySystem.Interpretations.TrafficLight (trafficInterp) where

import qualified Data.Map as Map
import Data.Typeable (cast)
import NeSyFramework.Categories.DATA (DataObj (..))
import NeSyFramework.Monads.Giry (Giry, categorical)
import Semantics (DynVal (..), Interpretation (..), SomeObj (..))

-- | Interpretation of the TrafficLight signature on DATA with Giry monad.
-- sort Light -> Finite {"Red", "Green", "Yellow"}
-- sort Drive -> Booleans
trafficInterp :: Interpretation Giry Double
trafficInterp =
  Interpretation
    { sorts =
        Map.fromList
          [ ("Light", SomeObj (Finite ["Red" :: String, "Green", "Yellow"])),
            ("Drive", SomeObj Booleans)
          ],
      funcs = Map.empty,
      rels =
        Map.fromList
          [ ( "==",
              \[DynVal x, DynVal y] -> case (cast x, cast y) of
                (Just (a :: String), Just (b :: String)) -> if a == b then 1.0 else 0.0
                _ -> 0.0
            ),
            ( "eval",
              \[DynVal b] -> case cast b of
                Just True -> 1.0
                Just False -> 0.0
                _ -> 0.0
            )
          ],
      mfuncs =
        Map.fromList
          [ ( "light",
              \[] ->
                categorical
                  [ (DynVal ("Red" :: String), 0.6),
                    (DynVal ("Green" :: String), 0.3),
                    (DynVal ("Yellow" :: String), 0.1)
                  ]
            ),
            ( "driveF",
              \[DynVal l] -> case cast l of
                Just "Red" -> categorical [(DynVal True, 0.1), (DynVal False, 0.9)]
                Just "Yellow" -> categorical [(DynVal True, 0.2), (DynVal False, 0.8)]
                Just "Green" -> categorical [(DynVal True, 0.9), (DynVal False, 0.1)]
                _ -> error "Expected Traffic Light String"
            )
          ],
      mrels =
        Map.fromList
          [ ( "driveP",
              \[DynVal l] -> case cast l of
                Just "Red" -> categorical [(1.0, 0.1), (0.0, 0.9)]
                Just "Yellow" -> categorical [(1.0, 0.2), (0.0, 0.8)]
                Just "Green" -> categorical [(1.0, 0.9), (0.0, 0.1)]
                _ -> error "Expected Traffic Light String"
            )
          ]
    }
