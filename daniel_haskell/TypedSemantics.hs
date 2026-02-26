{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module TypedSemantics where

import Data.Dynamic (Dynamic, fromDynamic, toDyn)
import qualified Data.Map as Map
import Data.Type.Equality ((:~:) (Refl))
import Data.Typeable (Typeable, eqT)
import Logical.Signatures.TwoMonBLat (TwoMonBLat (..))
import TypedSyntax

-- | Valuation: maps variable names to dynamic values (for Compu binding)
type Valuation = Map.Map String Dynamic

-- | Evaluate a term. No interpretation needed — functions are embedded directly.
evalTerm :: Valuation -> Term a -> a
evalTerm _ (Con x) = x
evalTerm val (Var name) =
  case Map.lookup name val of
    Just dyn -> case fromDynamic dyn of
      Just v -> v
      Nothing -> error $ "Type mismatch for var: " ++ name
    Nothing -> error $ "Unbound variable: " ++ name
evalTerm val (Fun f arg) =
  let fVal = evalTerm val f
      argVal = evalTerm val arg
   in fVal argVal

-- | Evaluate a formula. Logical connectives come from the TwoMonBLat constraint.
evalFormula :: forall m r tau. (TwoMonBLat tau, Typeable m, Monad m, Typeable r, Typeable tau) => Valuation -> Formula r -> m r
evalFormula val (Rel term) = return (evalTerm val term)
evalFormula val (BinConn op p q) = case eqT @r @tau of
  Just Refl -> do
    p' <- evalFormula @m @r @tau val p
    q' <- evalFormula @m @r @tau val q
    return (op p' q')
  Nothing -> error "Type mismatch in BinConn"
evalFormula val (UnConn op p) = case eqT @r @tau of
  Just Refl -> do
    p' <- evalFormula @m @r @tau val p
    return (op p')
  Nothing -> error "Type mismatch in UnConn"
evalFormula _ (NulConn c) = case eqT @r @tau of
  Just Refl -> return c
  Nothing -> error "Type mismatch in NulConn"
evalFormula val (Compu x (mTerm :: Term (m1 a)) phi) = do
  let mVal = evalTerm val mTerm
  case eqT @m1 @m of
    Just Refl -> do
      a <- mVal
      evalFormula @m @r @tau (Map.insert x (toDyn a) val) phi
    Nothing -> error "Monad mismatch in Compu!"

-- | Evaluate a comparison
evalComparison :: forall tau m. (TwoMonBLat tau, Typeable m, Monad m, Typeable tau) => Valuation -> Comparison tau -> m Bool
evalComparison val (Comp cmp p q) = do
  p' <- evalFormula @m @tau @tau val p
  q' <- evalFormula @m @tau @tau val q
  return (cmp p' q')
