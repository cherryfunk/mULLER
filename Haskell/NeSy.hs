{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, RankNTypes #-}

-- cabal install probability set-monad

import qualified Data.Set as Set
import qualified Data.Map as Map
import Data.Maybe
-- for specific NeSy frameworks
import Control.Monad.Identity
import qualified Data.Set.Monad as SM
import qualified Numeric.Probability.Distribution as Dist
import Control.Monad.Bayes.Class (MonadSample(..))
import Control.Monad.Bayes.Sampler (SamplerIO)
-- for sampling
import System.Random

-------------------------- NeSy Frameworks ------------------------------
-- algebra of truth values
class TwoSGrpBLat a where
  top, bot :: a
  neg :: a -> a
  conj, disj, implies :: a -> a -> a
  -- default implementations
  neg a = a `implies` bot
  a `implies` b = (neg a) `disj` b

-- aggregation functions for quantifiers
class (TwoSGrpBLat a) => Aggr2SGrpBLat s a where
  -- for a structure on b and a predicate on b, aggregate truth values a 
  aggrE, aggrA :: s b -> (b -> a) -> a

-- NeSy frameworks provide an algebra on T Omega
class (Monad t, Aggr2SGrpBLat s (t omega)) => NeSyFramework t s omega 

-- generic Aggr2SGrpBLat instance for any monad
instance Monad t => TwoSGrpBLat (t Bool) where
  top = return True
  bot = return False
  neg a = do x<-a; return $ not x
  conj a b = do x<-a; y<-b; return $ x && y 
  disj a b = do x<-a; y<-b; return $ x || y 
  --implies a b = do x<-a; y<-b; return $ ((not x) || y)

-- the mainly used Aggr2SGrpBLat: no additional stucture + Booleans
instance Monad t => Aggr2SGrpBLat [] (t Bool) where
  aggrE s f = foldr disj bot $ map f s
  aggrA s f = foldr conj top $ map f s
  
-- Classical instance using identity monad, Omega is Bool
instance NeSyFramework Identity [] Bool 
  
-- Distribution instance, Omega is Bool
instance Num prob => NeSyFramework (Dist.T prob) [] Bool 

-- Non-empty powerset instance (non-determinism)
-- there is no standard non-empty set monad in Haskell
-- so we use the set monad instead. Omega is Bool
instance NeSyFramework SM.Set [] Bool 

-- Giry monad instance, using monad-bayes for both aggregation and the monad
no_samples = 1000
instance Aggr2SGrpBLat SamplerIO (SamplerIO Bool) where
  -- Expectation-style aggregation over a distribution
  -- Here we approximate via Monte Carlo with n samples
  -- aggrE :: SamplerIO a -> (a -> SamplerIO Bool) -> SamplerIO Bool
  aggrE dist f = do
    samples <- sequence (replicate no_samples dist)
    vals    <- mapM f samples
    return (or vals)
  -- aggrA :: SamplerIO a -> (a -> SamplerIO Bool) -> SamplerIO Bool
  aggrA dist f = do
    samples <- sequence (replicate no_samples dist)
    vals    <- mapM f samples
    return (and vals)
instance NeSyFramework SamplerIO SamplerIO Bool

-------------------------- Syntax ------------------------------
         
type Ident = String -- identifiers

-- For simplicity, we use untyped FOL
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

-------------------------- Semantics ------------------------------
               
data Interpretation t s omega a =
     Interpretation { universe :: s a,
                      funcs :: Map.Map Ident ([a] -> a),
                      mfuncs :: Map.Map Ident ([a] -> t a),
                      preds :: Map.Map Ident ([a] -> omega),
                      mpreds :: Map.Map Ident ([a] -> t omega) }

type NeSyFrameworkTransformation t1 s1 omega1 t2 s2 omega2 =
     forall a . Ord a => Interpretation t1 s1 omega1 a ->
                         Interpretation t2 s2 omega2 a

-- argmax NeSy transformation
maximalValues :: (Num prob, Ord prob, Eq prob, Ord a) =>
                   Dist.T prob a -> SM.Set a
maximalValues dist = SM.fromList maxVals
  where
    probMap = Map.fromListWith (+) $  Dist.decons dist
    maxProb = maximum $ Map.elems probMap
    maxVals = map fst $ filter ((== maxProb) . snd) $ Map.toList probMap

argmax ::  (Num prob, Ord prob, Eq prob) =>
           NeSyFrameworkTransformation (Dist.T prob) [] Bool
                                       SM.Set [] Bool
argmax i = Interpretation { universe = universe i,
             funcs = funcs i,
             mfuncs = Map.map (maximalValues .) $ mfuncs i,
             preds = preds i,
             mpreds = Map.map (maximalValues .) $ mpreds i }

-- Tarskian semantics  
type Valuation a = Map.Map Ident a

-- throw a useful runtime error if some identifier is not declared
lookupId :: (Show k, Ord k) => k -> Map.Map k v -> v
lookupId k m = case Map.lookup k m of
   Just x -> x
   Nothing -> error (show k++" has not been declared")

evalT :: NeSyFramework t s omega =>
         Interpretation t s omega a -> Valuation a -> Term -> a
evalT _ val (Var var) = lookupId var val
evalT i val (Appl f ts) = f_sem $ map (evalT i val) ts
      where f_sem = lookupId f (funcs i)
           
evalF :: NeSyFramework t s omega =>
         Interpretation t s omega a -> Valuation a -> Formula -> t omega
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
evalF i val (Forall var f) = aggrA (universe i) predicate 
      where predicate a = evalF i (Map.insert var a val) f
evalF i val (Exists var f) = aggrE (universe i) predicate 
      where predicate a = evalF i (Map.insert var a val) f
evalF i val (Comp var m ts f) = -- var:=m(ts)(f)
      do a <- m_sem $ map (evalT i val) ts
         evalF i (Map.insert var a val) f
      where m_sem = lookupId m (mfuncs i)

-------------------------- Dice example ------------------------------
              
-- interpretation for dice example
dieModel :: Interpretation (Dist.T Double) [] Bool Integer
dieModel =  Interpretation { universe = [1..6],              
               funcs = Map.fromList $ map (\x -> (show x,\_ -> x)) [1..6],
               mfuncs = Map.fromList [("die",\_ -> Dist.uniform [1..6])],
               preds = Map.fromList[("==",\[x,y] -> x==y),
                                    ("even",\[x] -> even x)],
               mpreds = Map.empty }

-- x:=dice() (x==6 ∧ even(x))  
dieSen1 :: Formula  
dieSen1 = Comp "x" "die" [] (And (Pred "==" [Var "x",Appl "6" []])
                                 (Pred "even" [Var "x"]) ) 
d1 :: Dist.T Double Bool
d1 = evalF dieModel Map.empty dieSen1  

-- (x:=dice() (x==6)) ∧ (x:=dice() even(x))
dieSen2 :: Formula  
dieSen2 = And (Comp "x" "die" [] (Pred "==" [Var "x",Appl "6" []]))
              (Comp "x" "die" [] (Pred "even" [Var "x"]))
d2 :: Dist.T Double Bool
d2 = evalF dieModel Map.empty dieSen2  

-- after transformation to non-deterministic NeSy framework
dieModelC = argmax dieModel
d1C = evalF dieModelC Map.empty dieSen1  
d2C = evalF dieModelC Map.empty dieSen2

-------------------------- Traffic light example ------------------------------
              
-- interpretation for traffic light example
data Universe = Red | Yellow | Green | B Bool deriving (Eq, Ord, Show)
trafficModel :: Interpretation (Dist.T Double) [] Bool Universe
trafficModel =  Interpretation {
  universe = [Red, Yellow, Green, B False, B True],
  funcs = Map.fromList [("green",\_ -> Green)],
  mfuncs = Map.fromList [("light",\_ ->
              Dist.fromFreqs [(Red, 0.6),(Green, 0.3),(Yellow, 0.1)]),
              ("driveF",\[l] -> case l of
                  Red -> Dist.fromFreqs [(B True, 0.1),(B False, 0.9)]
                  Yellow -> Dist.fromFreqs [(B True, 0.2),(B False, 0.8)]
                  Green -> Dist.fromFreqs [(B True, 0.9),(B False, 0.1)])
                    ],
  preds = Map.fromList[
             ("==",\[x,y] -> x==y),
             ("eval",\[b] -> case b of
                               B False -> False
                               B True -> True)],
  mpreds = Map.fromList[("driveP",\[l] -> case l of
                  Red -> Dist.fromFreqs [(True, 0.1),(False, 0.9)]
                  Yellow -> Dist.fromFreqs [(True, 0.2),(False, 0.8)]
                  Green -> Dist.fromFreqs [(True, 0.9),(False, 0.1)])]}

-- l:=light(), d:=driveF(l) (eval d -> l==green)
trafficSen1 :: Formula  
trafficSen1 = Comp "l" "light" []
                (Comp "d" "driveF" [Var "l"]
                   (Implies (Pred "eval" [Var "d"])
                            (Pred "==" [Var "l",Appl "green" []])))
t1 :: Dist.T Double Bool
t1 = evalF trafficModel Map.empty trafficSen1

-- l:=light() (driveP(l) -> l==green)
trafficSen2 :: Formula  
trafficSen2 = Comp "l" "light" []
                (Implies (MPred "driveP" [Var "l"])
                         (Pred "==" [Var "l",Appl "green" []]))
t2 :: Dist.T Double Bool
t2 = evalF trafficModel Map.empty trafficSen2

-- after transformation to non-deterministic NeSy framework
trafficModelC = argmax trafficModel
t1C = evalF trafficModelC Map.empty trafficSen1
t2C = evalF trafficModelC Map.empty trafficSen2

-------------------------- Random sampling -------------------------

-- Sample one element uniformly from a list
sampleUniform :: [a] -> IO a
sampleUniform xs = do
    index <- randomRIO (0, length xs - 1)
    return (xs !! index)

-- Sample from non-deterministic values and print result
sample :: (Ord a, Show a) => (String, SM.Set a) -> IO ()
sample (label,set) =
  do sample <- sampleUniform $ SM.toList set
     putStrLn $ label ++ ": "++ show sample

main :: IO()
main = do
  let values = [("d1C",d1C),("d2C",d2C),("t1C",t1C),("t2C",t2C)]
  mapM_ sample values


