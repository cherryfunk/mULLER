{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TypedSyntax
  ( -- * Types
    VarSym,
    SomeTerm (..),
    Term (..),
    Formula (..),
    Comparison (..),

    -- * Comparison builder
    comparison,
  )
where

import Data.Kind (Type)
import Data.Typeable (Typeable)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))

-- | Variable names (only for Compu binding)
type VarSym = String

-- | Wrapper for heterogeneous terms (needed for Subst)
data SomeTerm where
  SomeTerm :: (Typeable a) => Term a -> SomeTerm

-- | Typed Term AST
data Term a where
  Var :: (Typeable a) => VarSym -> Term a
  Con :: (Typeable a) => a -> Term a
  Fun :: Term (a -> b) -> Term a -> Term b

-- | Typed Formula AST
-- Logical connectives use TwoMonBLat symbols: wedge, vee, oplus, otimes, etc.
data Formula tau where
  Rel :: (Typeable tau) => Term tau -> Formula tau
  Bot_ :: (Typeable tau) => Formula tau
  Top_ :: (Typeable tau) => Formula tau
  V0_ :: (Typeable tau) => Formula tau
  V1_ :: (Typeable tau) => Formula tau
  Wedge_ :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Vee_ :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Oplus_ :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Otimes_ :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Subst :: (Typeable tau) => [(VarSym, SomeTerm)] -> Formula tau -> Formula tau
  Compu :: forall (m :: Type -> Type) (a :: Type) tau. (Typeable a, Typeable m, Typeable tau) => VarSym -> Term (m a) -> Formula tau -> Formula tau

-- | Formulas are built using TwoMonBLat symbols (wedge, vee, etc.).
instance (Typeable tau) => TwoMonBLat (Formula tau) where
  vdash _ _ = error "vdash on Formula: use 'comparison' to build a Comparison"
  wedge = Wedge_
  vee = Vee_
  oplus = Oplus_
  otimes = Otimes_
  bot = Bot_
  top = Top_
  v0 = V0_
  v1 = V1_

-- | Comparison of two formulas (syntax-level vdash)
data Comparison tau where
  Comparison_ :: (Typeable tau) => Formula tau -> Formula tau -> Comparison tau

-- | Build a comparison: phi1 ⊢ phi2
comparison :: (Typeable tau) => Formula tau -> Formula tau -> Comparison tau
comparison = Comparison_
