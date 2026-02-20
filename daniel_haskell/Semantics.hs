{-# LANGUAGE GADTs #-}

module Semantics where

import qualified Data.Map as Map
import Data.Typeable (Typeable)
import NeSyFramework.TruthSpaces.TwoMonBLat (TwoMonBLat)
import qualified NeSyFramework.TruthSpaces.TwoMonBLat as Truth
import Syntax

data DynVal where
  DynVal :: (Typeable a) => a -> DynVal

type Valuation = Map.Map VarSym DynVal

data Interpretation m tau = Interpretation
  { funcs :: Map.Map FunSym ([DynVal] -> DynVal),
    rels :: Map.Map RelSym ([DynVal] -> tau),
    mfuncs :: Map.Map FunSym ([DynVal] -> m DynVal),
    mrels :: Map.Map RelSym ([DynVal] -> m tau)
  }

look :: String -> Map.Map String v -> v
look k m = Map.findWithDefault (error $ "Missing: " ++ k) k m

evalTerm :: Interpretation m tau -> Valuation -> Term -> DynVal
evalTerm _ val (Var x) = look x val
evalTerm _ _ (Con c) = DynVal c
evalTerm i val (Fun f args) = look f (funcs i) (map (evalTerm i val) args)

evalFormula :: (Monad m, TwoMonBLat tau) => Interpretation m tau -> Valuation -> Formula -> m tau
evalFormula i val f = case f of
  Rel r args -> return $ look r (rels i) (map (evalTerm i val) args)
  MRel r args -> look r (mrels i) (map (evalTerm i val) args)
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
    valM <- look m_sym (mfuncs i) (map (evalTerm i val) as)
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
