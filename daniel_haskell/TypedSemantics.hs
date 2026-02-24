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
evalFormula _ Top = case eqT @r @tau of
  Just Refl -> return top
  Nothing -> error "Type mismatch: Top"
evalFormula _ Bot = case eqT @r @tau of
  Just Refl -> return bot
  Nothing -> error "Type mismatch: Bot"
evalFormula _ V0 = case eqT @r @tau of
  Just Refl -> return v0
  Nothing -> error "Type mismatch: V0"
evalFormula _ V1 = case eqT @r @tau of
  Just Refl -> return v1
  Nothing -> error "Type mismatch: V1"
evalFormula val (Rel term) = return (evalTerm val term)
evalFormula val (Wedge p q) = case eqT @r @tau of
  Just Refl -> do
    p' <- evalFormula @m @r @tau val p
    q' <- evalFormula @m @r @tau val q
    return (wedge p' q')
  Nothing -> error "Type mismatch in Wedge"
evalFormula val (Vee p q) = case eqT @r @tau of
  Just Refl -> do
    p' <- evalFormula @m @r @tau val p
    q' <- evalFormula @m @r @tau val q
    return (vee p' q')
  Nothing -> error "Type mismatch in Vee"
evalFormula val (Oplus p q) = case eqT @r @tau of
  Just Refl -> do
    p' <- evalFormula @m @r @tau val p
    q' <- evalFormula @m @r @tau val q
    return (oplus p' q')
  Nothing -> error "Type mismatch in Oplus"
evalFormula val (Otimes p q) = case eqT @r @tau of
  Just Refl -> do
    p' <- evalFormula @m @r @tau val p
    q' <- evalFormula @m @r @tau val q
    return (otimes p' q')
  Nothing -> error "Type mismatch in Otimes"
evalFormula val (Subst ss phi) =
  let val' = foldl (\v (x, SomeTerm t) -> Map.insert x (toDyn (evalTerm val t)) v) val ss
   in evalFormula @m @r @tau val' phi
evalFormula val (Compu x (mTerm :: Term (m1 a)) phi) = do
  let mVal = evalTerm val mTerm
  case eqT @m1 @m of
    Just Refl -> do
      a <- mVal
      evalFormula @m @r @tau (Map.insert x (toDyn a) val) phi
    Nothing -> error "Monad mismatch in Compu!"

-- | Evaluate a comparison
evalComparison :: forall tau m. (TwoMonBLat tau, Typeable m, Monad m, Typeable tau) => Valuation -> Comparison tau -> m Bool
evalComparison val (Comparison p q) = do
  p' <- evalFormula @m @tau @tau val p
  q' <- evalFormula @m @tau @tau val q
  return (vdash p' q')
