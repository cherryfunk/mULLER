{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances #-}

import qualified Data.Set as Set
import qualified Data.Map as Map
import Data.Maybe

-- algebra of truth values
class Ord a => Aggr2SGrpBLat a where
  top, bot :: a
  neg :: a -> a
  conj, disj, implies :: a -> a -> a
  aggrE, aggrA :: [a] -> a
  -- default implementations
  neg a = a `implies` bot
  a `implies` b = (neg a) `disj` b
  aggrE = foldr disj bot
  aggrA = foldr conj top

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
data Term = Var String
          | Appl String [Term]
data Formula = T | F
             | Pred String [Term]
             | Not Formula  
             | And Formula Formula 
             | Or Formula Formula
             | Implies Formula Formula
             | Forall String Formula
             | Exist String Formula
             | Comp String String [Term] Formula -- x:=m(T1,...,Tn)(F)
               
data Interpretation t omega a =
     Interpretation { universe :: (Set.Set a),
                      funcs :: Map.Map String ([a] -> a),
                      mfuncs :: Map.Map String ([a] -> t a),
                      preds :: Map.Map String ([a] -> omega),
                      mpreds :: Map.Map String ([a] -> t omega) }

type Valuation a = Map.Map String a

forcedLookup :: Ord k => k -> Map.Map k v -> v
forcedLookup k m = fromJust $ Map.lookup k m 

evalT :: NeSyFramework t omega =>
         Interpretation t omega a -> Valuation a -> Term -> a
evalT i v (Var var) = forcedLookup var v
evalT i v (Appl f ts) = undefined
  
evalF :: NeSyFramework t omega =>
         Interpretation t omega a -> Valuation a -> Formula -> t omega
evalF = undefined 

main :: IO ()
main = putStrLn "NeSy framework loaded successfully"
