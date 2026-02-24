{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TypedSyntax where

import Data.Kind (Type)
import Data.Typeable (Typeable)

-- | Variable names (only for Compu binding)
type VarSym = String

-- | Wrapper for heterogeneous terms (needed for Subst)
data SomeTerm where
  SomeTerm :: (Typeable a) => Term a -> SomeTerm

-- | Typed Term AST
data Term a where
  -- | Variable (bound by Compu)
  Var :: (Typeable a) => VarSym -> Term a
  -- | Embed any Haskell value (constant, function, monadic action)
  Con :: (Typeable a) => a -> Term a
  -- | Curried function application
  Fun :: Term (a -> b) -> Term a -> Term b

-- | Typed Formula AST
data Formula tau where
  -- | Lift a term returning tau into a formula
  Rel :: (Typeable tau) => Term tau -> Formula tau
  -- | Logical constants
  Bot :: (Typeable tau) => Formula tau
  Top :: (Typeable tau) => Formula tau
  V0 :: (Typeable tau) => Formula tau
  V1 :: (Typeable tau) => Formula tau
  -- | Logical connectives
  Wedge :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Vee :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Oplus :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  Otimes :: (Typeable tau) => Formula tau -> Formula tau -> Formula tau
  -- | Substitution
  Subst :: (Typeable tau) => [(VarSym, SomeTerm)] -> Formula tau -> Formula tau
  -- | Monadic computation (bind)
  Compu :: forall (m :: Type -> Type) (a :: Type) tau. (Typeable a, Typeable m, Typeable tau) => VarSym -> Term (m a) -> Formula tau -> Formula tau

-- | Comparison of two formulas
data Comparison tau where
  Comparison :: (Typeable tau) => Formula tau -> Formula tau -> Comparison tau
