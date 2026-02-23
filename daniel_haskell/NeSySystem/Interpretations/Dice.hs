{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeSySystem.Interpretations.Dice (diceInterp) where

import Data.Typeable (cast)
import NeSyFramework.Categories.DATA (DataObj (..))
import NeSyFramework.Monads.Giry (Giry, categorical)
import NeSySystem.Signatures.Dice (diceSig)
import Semantics (DynVal (..), Interpretation (..), SomeObj (..))

diceInterp :: Interpretation Giry Double
diceInterp =
  Interpretation
    { sig = diceSig,
      interpSort = \case
        "DieResult" -> SomeObj (Finite [1 :: Int, 2, 3, 4, 5, 6])
        s -> error $ "Dice: unknown sort " ++ s,
      interpFunc = \f _ -> error $ "Dice: unknown func " ++ f,
      interpRel = \case
        "==" -> \[DynVal x, DynVal y] -> case (cast x, cast y) of
          (Just (a :: Int), Just (b :: Int)) -> if a == b then 1.0 else 0.0
          _ -> 0.0
        "even" -> \[DynVal x] -> case cast x of
          Just (a :: Int) -> if even a then 1.0 else 0.0
          _ -> 0.0
        r -> error $ "Dice: unknown rel " ++ r,
      interpMFunc = \case
        "die" -> \[] -> categorical [(DynVal (i :: Int), 1.0 / 6.0) | i <- [1 .. 6]]
        mf -> error $ "Dice: unknown mfunc " ++ mf,
      interpMRel = \mr _ -> error $ "Dice: unknown mrel " ++ mr
    }
