{-# LANGUAGE ScopedTypeVariables #-}

module Interpretations.Dice (diceInterp) where

import qualified Data.Map as Map
import Data.Typeable (cast)
import NeSyFramework.Monads.Giry (Giry, categorical)
import Semantics (DynVal (..), Interpretation (..))

diceInterp :: Interpretation Giry Double
diceInterp =
  Interpretation
    { funcs = Map.empty,
      rels =
        Map.fromList
          [ ( "==",
              \[DynVal x, DynVal y] -> case (cast x, cast y) of
                (Just (a :: Int), Just (b :: Int)) -> if a == b then 1.0 else 0.0
                _ -> 0.0
            ),
            ( "even",
              \[DynVal x] -> case cast x of
                Just (a :: Int) -> if even a then 1.0 else 0.0
                _ -> 0.0
            )
          ],
      mfuncs =
        Map.fromList
          [ ("die", \[] -> categorical [(DynVal (i :: Int), 1.0 / 6.0) | i <- [1 .. 6]])
          ],
      mrels = Map.empty
    }
