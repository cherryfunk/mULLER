{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE StandaloneDeriving #-}

module Syntax where

import Data.Typeable (Typeable)

-- | Symbolsets
type SortSym = String

type FunSym = String

type RelSym = String

type VarSym = String

-- | Terms
data Term where
  -- | Variable Term
  Var :: VarSym -> Term
  -- | Constant Term
  Con :: (Typeable a) => a -> Term
  -- | Functional Term
  Fun :: FunSym -> [Term] -> Term

-- | Formulas
data Formula where
  -- | Atomic Formula
  Rel :: RelSym -> [Term] -> Formula
  MRel :: RelSym -> [Term] -> Formula
  -- | Compound Formulas
  Bot :: Formula
  Top :: Formula
  V0 :: Formula
  V1 :: Formula
  Wedge :: Formula -> Formula -> Formula
  Vee :: Formula -> Formula -> Formula
  Oplus :: Formula -> Formula -> Formula
  Otimes :: Formula -> Formula -> Formula
  -- | Substitution Formulas and Monadic Computation (a special kind of substitution)
  Subst :: [(VarSym, Term)] -> Formula -> Formula
  Compu :: VarSym -> FunSym -> [Term] -> Formula -> Formula

-- | Comparison of Formulas
data Comparison where
  Comparison :: Formula -> Formula -> Comparison
