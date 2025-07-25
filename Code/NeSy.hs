{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}

import qualified Data.Set as Set
import qualified Data.Map as Map
import Data.Maybe

type Ident = String -- identifiers

-- algebra of truth values
class Ord a => Aggr2SGrpBLat a where
  top, bot :: a
  neg :: a -> a
  conj, disj, implies :: a -> a -> a
  aggrE, aggrA :: Set.Set a -> a
  -- default implementations
  neg a = a `implies` bot
  a `implies` b = (neg a) `disj` b
  aggrE = Set.fold disj bot
  aggrA = Set.fold conj top

-- NeSy frameworks provide an algebra on T Omega
-- Beware that for a given t and omega, there can be only one instance
-- If needed, use several isomorphic copies of omega to
--   distinguish several instances
-- Another solution (which do not follow here) are identity types as in Hets
class (Monad t, Aggr2SGrpBLat (t omega)) => NeSyFramework t omega

-- Non-empty powerset instance
-- there is no standard non-empty set monad in Haskell
-- so we use the list monad instead. Omega is Bool
instance  Aggr2SGrpBLat [Bool] where
  top = [True]
  bot = [False]
  neg a = [not x | x<-a]
  conj a b = [x && y | x<-a, y<-b]
  disj a b = [x || y | x<-a, y<-b]

instance NeSyFramework [] Bool 
  
-- for simplicity, we use untyped FOL
data Term = Var Ident
          | Appl Ident [Term]
data Formula = T | F
             | Pred Ident [Term]
             | MPred Ident [Term]
             | Not Formula  
             | And Formula Formula 
             | Or Formula Formula
             | Implies Formula Formula
             | Forall Ident Formula
             | Exists Ident Formula
             | Comp Ident Ident [Term] Formula -- x:=m(T1,...,Tn)(F)
               
data Interpretation t omega a =
     Interpretation { universe :: (Set.Set a),
                      funcs :: Map.Map Ident ([a] -> a),
                      mfuncs :: Map.Map Ident ([a] -> t a),
                      preds :: Map.Map Ident ([a] -> omega),
                      mpreds :: Map.Map Ident ([a] -> t omega) }

type Valuation a = Map.Map Ident a

-- we risk a runtime error if some identifier is not declared
forcedLookup :: Ord k => k -> Map.Map k v -> v
forcedLookup k m = fromJust $ Map.lookup k m 

evalT :: NeSyFramework t omega =>
         Interpretation t omega a -> Valuation a -> Term -> a
evalT _ val (Var var) = forcedLookup var val
evalT i val (Appl f ts) = f_sem $ map (evalT i val) ts
      where f_sem = forcedLookup f (funcs i)
           
evalF :: NeSyFramework t omega =>
         Interpretation t omega a -> Valuation a -> Formula -> t omega
evalF _ _ T = top
evalF _ _ F = bot
evalF i val (Pred p ts) = return $ p_sem $ map (evalT i val) ts
      where p_sem = forcedLookup p (preds i)
evalF i val (MPred p ts) = p_sem $ map (evalT i val) ts
      where p_sem = forcedLookup p (mpreds i)
evalF i val (Not f) = neg $ evalF i val f
evalF i val (And f1 f2) = conj (evalF i val f1) (evalF i val f2)
evalF i val (Or f1 f2) = disj (evalF i val f1) (evalF i val f2)
evalF i val (Implies f1 f2) = implies (evalF i val f1) (evalF i val f2)
evalF i val (Forall var f) = aggrA $ Set.map evalAux $ universe i
      where evalAux a = evalF i (Map.insert var a val) f
evalF i val (Exists var f) = aggrE $ Set.map evalAux $ universe i
      where evalAux a = evalF i (Map.insert var a val) f
evalF i val (Comp var m ts f) = -- var:=m(ts)(f)
      do a <- m_sem $ map (evalT i val) ts
         evalF i (Map.insert var a val) f
      where m_sem = forcedLookup m (mfuncs i)
                

main :: IO ()
main = putStrLn "NeSy framework loaded successfully"
