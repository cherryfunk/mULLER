{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

-- | Non-logical interpretation: Dice domain (in DATA, with Dist monad)
module NonLogical.Interpretations.Dice
  ( -- * Formula-ready symbols (use these in formulas)
    die',
    eqDie',
    evenDie',
  )
where

import Logical.Interpretations.Real (Omega)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Monads.Dist (Dist (..))
import NonLogical.Signatures.DiceSig
import TypedSyntax

instance DiceSig DATA Dist where
  -- Sor (proving sorts are objects in DATA)
  dieResultObj :: DATA DieResult
  dieResultObj = Finite [1, 2, 3, 4, 5, 6]

  -- mFun
  die :: Dist DieResult
  die = Dist [(i, 1.0 / 6.0) | i <- [1 .. 6]]

  -- Rel
  eqDie :: DieResult -> DieResult -> Omega
  eqDie x y = if x == y then v1 else v0
  evenDie :: DieResult -> Omega
  evenDie x = if even x then v1 else v0

-- | Formula-ready symbols: all plumbing hidden
die' :: Term (Dist Int)
die' = con (die @DATA @Dist)

eqDie' :: Term Int -> Term Int -> Formula Omega
eqDie' x y = rel (con (eqDie @DATA @Dist) $$ x $$ y)

evenDie' :: Term Int -> Formula Omega
evenDie' x = rel (con (evenDie @DATA @Dist) $$ x)
