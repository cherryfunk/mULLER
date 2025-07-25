{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, MonadComprehensions #-}

import qualified Data.Set as Set
import qualified Data.Map as Map
import Data.Maybe
-- for specific NeSy frameworks
import qualified Data.Set.Monad as SM
import qualified Numeric.Probability.Distribution as Dist

type Ident = String -- identifiers

-- algebra of truth values
class Aggr2SGrpBLat a where
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

-- Distribution instance, Omega is Bool
newtype DBool = DBool { getDBool :: Bool } deriving (Eq, Ord, Show)
instance Num prob => Aggr2SGrpBLat (Dist.T prob DBool) where
  top = return $ DBool True
  bot = return $ DBool False
  neg a = [DBool $ not x | DBool x<-a]
  conj a b = [DBool (x && y) | DBool x<-a, DBool y<-b]
  disj a b = [DBool (x || y) | DBool x<-a, DBool y<-b]
instance Num prob => NeSyFramework (Dist.T prob) DBool 
  
-- Non-empty powerset instance
-- there is no standard non-empty set monad in Haskell
-- so we use the set monad instead. Omega is Bool
newtype SBool = SBool { getSBool :: Bool } deriving (Eq, Ord, Show)
instance  Aggr2SGrpBLat (SM.Set SBool) where
  top = return $ SBool True
  bot = return $ SBool False
  neg a = [SBool $ not x | SBool x<-a]
  conj a b = [SBool (x && y) | SBool x<-a, SBool y<-b]
  disj a b = [SBool (x || y) | SBool x<-a, SBool y<-b]
instance NeSyFramework SM.Set SBool 
  
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
     Interpretation { universe :: [a],
                      funcs :: Map.Map Ident ([a] -> a),
                      mfuncs :: Map.Map Ident ([a] -> t a),
                      preds :: Map.Map Ident ([a] -> omega),
                      mpreds :: Map.Map Ident ([a] -> t omega) }

type Valuation a = Map.Map Ident a

-- throw a useful runtime error if some identifier is not declared
lookupId :: (Show k, Ord k)=> k -> Map.Map k v -> v
lookupId k m = case Map.lookup k m of
   Just x -> x
   Nothing -> error (show k++" has not been declared")

evalT :: NeSyFramework t omega =>
         Interpretation t omega a -> Valuation a -> Term -> a
evalT _ val (Var var) = lookupId var val
evalT i val (Appl f ts) = f_sem $ map (evalT i val) ts
      where f_sem = lookupId f (funcs i)
           
evalF :: NeSyFramework t omega =>
         Interpretation t omega a -> Valuation a -> Formula -> t omega
evalF _ _ T = top
evalF _ _ F = bot
evalF i val (Pred p ts) = return $ p_sem $ map (evalT i val) ts
      where p_sem = lookupId p (preds i)
evalF i val (MPred p ts) = p_sem $ map (evalT i val) ts
      where p_sem = lookupId p (mpreds i)
evalF i val (Not f) = neg $ evalF i val f
evalF i val (And f1 f2) = conj (evalF i val f1) (evalF i val f2)
evalF i val (Or f1 f2) = disj (evalF i val f1) (evalF i val f2)
evalF i val (Implies f1 f2) = implies (evalF i val f1) (evalF i val f2)
evalF i val (Forall var f) = aggrA $ map evalAux $ universe i
      where evalAux a = evalF i (Map.insert var a val) f
evalF i val (Exists var f) = aggrE $ map evalAux $ universe i
      where evalAux a = evalF i (Map.insert var a val) f
evalF i val (Comp var m ts f) = -- var:=m(ts)(f)
      do a <- m_sem $ map (evalT i val) ts
         evalF i (Map.insert var a val) f
      where m_sem = lookupId m (mfuncs i)
                
main :: IO ()
main = putStrLn "NeSy framework loaded successfully"

-- dice example
dieModel :: Interpretation (Dist.T Double) DBool Integer
dieModel =  Interpretation { universe = [1..6],
               funcs = Map.fromList $ map (\x -> (show x,\_ -> x)) [1..6],
               mfuncs = Map.fromList [("die",\_ -> Dist.uniform [1..6])],
               preds = Map.fromList[("==",\[x,y] -> DBool $ x==y),
                                    ("even",\[x] -> DBool $ even x)],
               mpreds = Map.empty }
dieSen1 :: Formula  
dieSen1 = Comp "x" "die" [] (And (Pred "==" [Var "x",Appl "6" []])
                                 (Pred "even" [Var "x"]) ) 
d1 :: Dist.T Double DBool
d1 = evalF dieModel Map.empty dieSen1  
dieSen2 :: Formula  
dieSen2 = And (Comp "x" "die" [] (Pred "==" [Var "x",Appl "6" []]))
              (Comp "x" "die" [] (Pred "even" [Var "x"]))
d2 :: Dist.T Double DBool
d2 = evalF dieModel Map.empty dieSen2  

