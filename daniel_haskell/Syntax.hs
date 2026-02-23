{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE StandaloneDeriving #-}

module Syntax where

import Data.Typeable (Typeable)

-- | Symbol sets (disjoint namespaces, documented by type alias)
type SortSym = String -- sort symbols

type FunSym = String -- pure function symbols

type RelSym = String -- pure relation symbols

type MFunSym = String -- monadic function symbols

type MRelSym = String -- monadic relation symbols

type VarSym = String -- variable symbols

-- | A Signature (Schema) declares sorts, function symbols, relation symbols, monadic function symbols, and monadic relation symbols.
data Signature = Signature
  { sortDecls :: [SortSym],
    funDecls :: [(FunSym, [SortSym], SortSym)], -- f : S_1 x ... x S_n -> S
    relDecls :: [(RelSym, [SortSym])], -- R : S_1 x ... x S_n -> tau
    mFunDecls :: [(MFunSym, [SortSym], SortSym)], -- mf : S_1 x ... x S_n -> T(S)
    mRelDecls :: [(MRelSym, [SortSym])] -- mR : S_1 x ... x S_n -> T(tau)
  }

-- | Terms
data Term where
  -- | Variable Term
  Var :: VarSym -> Term
  -- | Constant Term
  Con :: (Typeable a) => a -> Term
  -- | Functional Term (f ∈ Fun)
  Fun :: FunSym -> [Term] -> Term

-- | Formulas
data Formula where
  -- | Atomic Formula (R ∈ Rel — pure relation)
  Rel :: RelSym -> [Term] -> Formula
  -- | Monadic Relation (mR ∈ MRel — monadic relation)
  MRel :: MRelSym -> [Term] -> Formula
  -- | Compound Formulas (logical connectives from the logical vocabulary)
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
  Compu :: VarSym -> MFunSym -> [Term] -> Formula -> Formula

-- | Comparison of Formulas
data Comparison where
  Comparison :: Formula -> Formula -> Comparison
