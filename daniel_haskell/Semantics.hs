{-# LANGUAGE GADTs #-}

module Semantics where

import qualified Data.Map as Map
import Data.Typeable (Typeable)
import NeSyFramework.Categories.DATA (DataObj)
import NeSyFramework.TruthSpaces.TwoMonBLat (TwoMonBLat)
import qualified NeSyFramework.TruthSpaces.TwoMonBLat as Truth
import Syntax

data DynVal where
  DynVal :: (Typeable a) => a -> DynVal

-- | Existential wrapper for heterogeneous collections (e.g., mapping sort symbols to objects).
-- This is interpretation plumbing, not part of the category definition itself.
data SomeObj where
  SomeObj :: (Typeable a) => DataObj a -> SomeObj

type Valuation = Map.Map VarSym DynVal

-- | An Interpretation of a Signature in a category with monad m and truth type tau.
-- Each field is a function that maps a declared symbol to its semantic meaning.
-- The Signature provides the symbols; the functions provide the mappings.
data Interpretation m tau = Interpretation
  { sig :: Signature,
    interpSort :: SortSym -> SomeObj,
    interpFunc :: FunSym -> [DynVal] -> DynVal,
    interpRel :: RelSym -> [DynVal] -> tau,
    interpMFunc :: MFunSym -> [DynVal] -> m DynVal,
    interpMRel :: MRelSym -> [DynVal] -> m tau
  }

evalTerm :: Interpretation m tau -> Valuation -> Term -> DynVal
evalTerm _ val (Var x) = Map.findWithDefault (error $ "Missing var: " ++ x) x val
evalTerm _ _ (Con c) = DynVal c
evalTerm i val (Fun f args) = interpFunc i f (map (evalTerm i val) args)

evalFormula :: (Monad m, TwoMonBLat tau) => Interpretation m tau -> Valuation -> Formula -> m tau
evalFormula i val f = case f of
  Rel r args -> return $ interpRel i r (map (evalTerm i val) args)
  MRel r args -> interpMRel i r (map (evalTerm i val) args)
  Bot -> return Truth.bot
  Top -> return Truth.top
  V0 -> return Truth.v0
  V1 -> return Truth.v1
  Wedge p q -> liftB Truth.wedge p q
  Vee p q -> liftB Truth.vee p q
  Oplus p q -> liftB Truth.oplus p q
  Otimes p q -> liftB Truth.otimes p q
  Subst ss phi -> evalFormula i (foldl (\v (x, t) -> Map.insert x (evalTerm i val t) v) val ss) phi
  Compu x m_sym as phi -> do
    valM <- interpMFunc i m_sym (map (evalTerm i val) as)
    evalFormula i (Map.insert x valM val) phi
  where
    liftB op p q = do
      p' <- evalFormula i val p
      q' <- evalFormula i val q
      return (op p' q')

evalComparison :: (Monad m, TwoMonBLat tau) => Interpretation m tau -> Valuation -> Comparison -> m Bool
evalComparison i val (Comparison p q) = do
  p' <- evalFormula i val p
  q' <- evalFormula i val q
  return (Truth.vdash p' q')
