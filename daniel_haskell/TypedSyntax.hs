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
-- Paper: xi ::= x | f(xi)
data Term a where
  -- | xi ::= x -- Variable term
  Var :: (Typeable a) => VarSym -> Term a
  -- | Embed any Haskell value (constant, function symbol, monadic action)
  Con :: (Typeable a) => a -> Term a
  -- | xi ::= f(xi) -- Function application (curried)
  Fun :: Term (a -> b) -> Term a -> Term b

-- | Typed Formula AST
-- Paper: phi ::= R(xi) | *(phi) | Qx(phi) | phi[xi/x]
data Formula tau where
  -- | phi ::= R(xi) -- Atomic formula (relation symbol applied to terms)
  Rel :: (Typeable tau) => Term tau -> Formula tau
  -- | phi ::= c -- Nullary connective (bot, top, v0, v1, ...)
  NulConn :: (Typeable tau) => tau -> Formula tau
  -- | phi ::= *(phi) -- Unary connective (negation, exponentials, ...)
  UnConn :: (Typeable tau) => (tau -> tau) -> Formula tau -> Formula tau
  -- | phi ::= *(phi, phi) -- Binary connective, where * in Conn
  BinConn :: (Typeable tau) => (tau -> tau -> tau) -> Formula tau -> Formula tau -> Formula tau
  -- | phi ::= phi[xi/x] -- Substitution
  Subst :: (Typeable tau) => [(VarSym, SomeTerm)] -> Formula tau -> Formula tau
  -- | phi ::= Qx(phi) -- Quantified formula (monadic computation)
  Compu :: forall (m :: Type -> Type) (a :: Type) tau. (Typeable a, Typeable m, Typeable tau) => VarSym -> Term (m a) -> Formula tau -> Formula tau

-- | Comparison of formulas
-- Paper: psi ::= prec(phi, phi) -- where prec in Comp
data Comparison tau where
  Comp :: (Typeable tau) => (tau -> tau -> Bool) -> Formula tau -> Formula tau -> Comparison tau
