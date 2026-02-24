{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE UndecidableInstances #-}

module TypedSyntax
  ( -- * AST types (for pattern matching in evaluator)
    VarSym,
    SomeTerm (..),
    Term (..),
    Formula (..),
    Comparison (..),

    -- * Sugar: write formulas as natural Haskell
    var,
    con,
    ($$),
    rel,
    bind,
    comparison,
  )
where

import Data.Kind (Type)
import Data.Typeable (Typeable)
import qualified Logical.Signatures.TwoMonBLat as L (TwoMonBLat (..))

-- | Variable names (only for Compu binding)
type VarSym = String

-- | Wrapper for heterogeneous terms (needed for Subst)
data SomeTerm where
  SomeTerm :: (Typeable a) => Term a -> SomeTerm

--------------------------------------------------------------------------------
-- Term AST: xi ::= x | f(xi)
--------------------------------------------------------------------------------
data Term a where
  Var :: (Typeable a) => VarSym -> Term a
  Con :: (Typeable a) => a -> Term a
  Fun :: Term (a -> b) -> Term a -> Term b

--------------------------------------------------------------------------------
-- Formula AST: phi ::= R(xi) | c | *(phi) | *(phi,phi) | Qx(phi) | phi[xi/x]
--------------------------------------------------------------------------------
data Formula tau where
  Rel :: (Typeable tau) => Term tau -> Formula tau
  NulConn :: (Typeable tau) => tau -> Formula tau
  UnConn :: (Typeable tau) => (tau -> tau) -> Formula tau -> Formula tau
  BinConn :: (Typeable tau) => (tau -> tau -> tau) -> Formula tau -> Formula tau -> Formula tau
  Subst :: (Typeable tau) => [(VarSym, SomeTerm)] -> Formula tau -> Formula tau
  Compu :: forall (m :: Type -> Type) (a :: Type) tau. (Typeable a, Typeable m, Typeable tau) => VarSym -> Term (m a) -> Formula tau -> Formula tau

--------------------------------------------------------------------------------
-- Comparison: psi ::= prec(phi, phi)
--------------------------------------------------------------------------------
data Comparison tau where
  Comp :: (Typeable tau) => (tau -> tau -> Bool) -> Formula tau -> Formula tau -> Comparison tau

--------------------------------------------------------------------------------
-- TwoMonBLat instance: use wedge/vee/bot/top directly on formulas
--------------------------------------------------------------------------------
instance (Typeable tau, L.TwoMonBLat tau) => L.TwoMonBLat (Formula tau) where
  vdash _ _ = error "vdash on Formula: use 'comparison'"
  wedge = BinConn L.wedge
  vee = BinConn L.vee
  oplus = BinConn L.oplus
  otimes = BinConn L.otimes
  bot = NulConn L.bot
  top = NulConn L.top
  v0 = NulConn L.v0
  v1 = NulConn L.v1

--------------------------------------------------------------------------------
-- Sugar: write formulas as natural Haskell
--------------------------------------------------------------------------------

-- | Variable term: var "x"
var :: (Typeable a) => String -> Term a
var = Var

-- | Embed a Haskell value as a term: con f
con :: (Typeable a) => a -> Term a
con = Con

-- | Function application (infix): con f $$ var "x"
infixl 9 $$

($$) :: Term (a -> b) -> Term a -> Term b
($$) = Fun

-- | Atomic formula from a fully-applied term
rel :: (Typeable tau) => Term tau -> Formula tau
rel = Rel

-- | Quantified formula (monadic bind): bind "x" (con die) phi
bind :: (Typeable a, Typeable m, Typeable tau) => VarSym -> Term (m a) -> Formula tau -> Formula tau
bind = Compu

-- | Comparison of formulas
comparison :: (Typeable tau) => (tau -> tau -> Bool) -> Formula tau -> Formula tau -> Comparison tau
comparison = Comp

-- | Num instance for Term: write 6 instead of con (6 :: Int)
instance (Typeable a, Num a) => Num (Term a) where
  fromInteger = Con . fromInteger
  (+) = error "Term is syntax, not arithmetic"
  (*) = error "Term is syntax, not arithmetic"
  abs = error "Term is syntax, not arithmetic"
  signum = error "Term is syntax, not arithmetic"
  negate = error "Term is syntax, not arithmetic"
