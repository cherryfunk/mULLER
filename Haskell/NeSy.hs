{-# LANGUAGE MultiParamTypeClasses, FlexibleInstances, RankNTypes #-}

-- cabal install probability set-monad

import qualified Data.Set as Set
import qualified Data.Map as Map
import Data.Maybe
-- for specific NeSy frameworks
import Control.Monad.Identity
import qualified Data.Set.Monad as SM
import qualified Numeric.Probability.Distribution as Dist
import Control.Monad.Bayes.Class
import Control.Monad.Bayes.Sampler.Strict (SamplerIO, sampleIO)
import Control.Monad.Bayes.Integrator (Integrator, runIntegrator, integrator, expectation)
import Math.GaussianQuadratureIntegration (nIntegrate256)
import qualified Data.Vector as V
import Numeric.Tools.Integration (quadTrapezoid, quadSimpson, quadRomberg, defQuad, quadRes, quadPrecEst, quadNIter)
-- for sampling
import System.Random (randomRIO)
import Debug.Trace

-------------------------- NeSy Frameworks ------------------------------
-- algebra of truth values
class TwoMonBLat a where
  top, bot :: a
  neg :: a -> a
  conj, disj, implies :: a -> a -> a
  -- default implementations
  neg a = a `implies` bot
  a `implies` b = (neg a) `disj` b

-- aggregation functions for quantifiers
class (TwoMonBLat a) => Aggr2MonBLat s a where
  -- for a structure on b and a predicate on b, aggregate truth values a 
  aggrE, aggrA :: s b -> (b -> a) -> a

-- NeSy frameworks provide an algebra on T Omega
class (Monad t, Aggr2MonBLat s (t omega)) => NeSyFramework t s omega 

-- generic Aggr2MonBLat instance for any monad
instance Monad t => TwoMonBLat (t Bool) where
  top = return True
  bot = return False
  neg a = do x<-a; return $ not x
  conj a b = do x<-a; y<-b; return $ x && y 
  disj a b = do x<-a; y<-b; return $ x || y 
  --implies a b = do x<-a; y<-b; return $ ((not x) || y)

-- the mainly used Aggr2MonBLat: no additional stucture  (just lists) + Booleans
instance Monad t => Aggr2MonBLat [] (t Bool) where
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

-- Expectation-style aggregation over a distribution
-- Here we approximate via Monte Carlo with no_samples samples
no_samples = 1000
aggregation :: Monad m => ([a] -> a) -> m b -> (b -> m a) -> m a
aggregation connective dist f = do
  samples <- sequence (replicate no_samples dist)
  vals    <- mapM f samples
  return (connective vals)
instance Aggr2MonBLat SamplerIO (SamplerIO Bool) where
  aggrE = aggregation or
  aggrA = aggregation and
-- Giry monad instance, using SamplerIO for both aggregation and the monad
instance NeSyFramework SamplerIO SamplerIO Bool

instance Aggr2MonBLat Integrator (Integrator Bool) where
  aggrA meas f =
    integrator $ \meas_fun ->
        exp $ runIntegrator (runIntegrator (log . meas_fun) . f) meas
  aggrE meas f =
    neg (aggrA meas (neg . f))
-- Giry monad instance, using Integrator for both aggregation and the monad
instance NeSyFramework Integrator Integrator Bool

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

mainM :: IO()
mainM = do
  let values = [("d1C",d1C),("d2C",d2C),("t1C",t1C),("t2C",t2C)]
  mapM_ sample values
  integrationComparison
  testInfiniteLists

-------------------------- Wheather example ------------------------------

-- Toy stubs for predictors:
humid_detector :: Int -> Double
humid_detector d = 0.5
-- "probability it's humid"

temperature_predictor :: Int -> (Double,Double)
temperature_predictor d = (0,2)
-- mean temperature and variance for data poin

-- interpretation 
data UniverseW = Int Int | Double Double | Pair (Double,Double)
                 deriving (Eq, Ord, Show)
b2UW :: Bool -> UniverseW
b2UW b = Int (if b then 1 else 0)
d2UW :: Double -> UniverseW
d2UW d = Double d

-- continuous distribution over UniverseW
uniformUniverseW :: MonadDistribution m => m UniverseW
uniformUniverseW = do
  choice <- categorical $ V.fromList [1,1,1]
  case choice of
    0 -> do
      -- sample an integer uniformly from [-10..10]
      i <- categorical $ V.fromList $ replicate 21 1
      return (Int (i-10))
    1 -> do
      x <- uniform 0 1       -- x ~ U(0,1)
      return (Double x)
    2 -> do
      x <- uniform (-5) 20      -- Pair (x,y), each U(-1,1)
      y <- uniform 0 5
      return (Pair (x, y))

weatherModel :: Interpretation SamplerIO SamplerIO Bool UniverseW
weatherModel =  Interpretation {
  universe = uniformUniverseW,
  funcs = Map.fromList [("humid_detector",\[Int d] ->
                            Double (humid_detector d)),
                        ("temperature_predictor",\[Int d] ->
                            Pair (temperature_predictor d)),
                        ("data1",\_ -> Int 1),
                        ("0",\_ -> Int 0),
                        ("1",\_ -> Int 1),
                        ("0.0",\_ -> Double 0.0),
                        ("15.0",\_ -> Double 15.0)
                        ],
  mfuncs = Map.fromList [("bernoulli",\[Double d] -> fmap b2UW $ bernoulli d),
                         ("normal",\[Pair(d1,d2)] -> fmap d2UW $ normal d1 d2)
                        ],
  preds = Map.fromList[
             ("==",\[x,y] -> x==y),
             ("<",\[x,y] -> x<y),
             (">",\[x,y] -> x>y)],
  mpreds = Map.fromList[]}

-- [h := bernoulli(humid_detector(data1))]
--   [t := normal(temperature_predictor(data1))]
--     (h = 1 ∧ t < 0) ∨ (h = 0 ∧ t > 15)
weatherSen1 :: Formula  
weatherSen1 =
  Comp "h" "bernoulli" [Appl "humid_detector" [Appl "data1" []]]
       (Comp "t" "normal" [Appl "temperature_predictor" [Appl "data1" []]]
                   (Or (And (Pred "==" [Var "h", Appl "1" []])
                            (Pred "<" [Var "t",Appl "0.0" []]))
                       (And (Pred "==" [Var "h", Appl "0" []])
                            (Pred ">" [Var "t",Appl "15.0" []]))
                   ))
weatherBody :: Formula
weatherBody = Comp "h" "bernoulli" [Appl "humid_detector" [Var "d"]]
                (Comp "t" "normal" [Appl "temperature_predictor" [Var "d"]]
                   (Or (And (Pred "==" [Var "h", Appl "1" []])
                            (Pred "<" [Var "t",Appl "0.0" []]))
                       (And (Pred "==" [Var "h", Appl "0" []])
                            (Pred ">" [Var "t",Appl "15.0" []]))
                   ))
-- forall d : [h := bernoulli(humid_detector(d))]
--               [t := normal(temperature_predictor(d))]
--                  (h = 1 ∧ t < 0) ∨ (h = 0 ∧ t > 15)
weatherSen2 :: Formula  
weatherSen2 = Forall "d" weatherBody
-- exists d : [h := bernoulli(humid_detector(d))]
--               [t := normal(temperature_predictor(d))]
--                  (h = 1 ∧ t < 0) ∨ (h = 0 ∧ t > 15)
weatherSen3 :: Formula  
weatherSen3 = Exists "d" weatherBody

w1 :: SamplerIO Bool
w1 = evalF weatherModel Map.empty weatherSen1
w2 :: SamplerIO Bool
w2 = evalF weatherModel Map.empty weatherSen2
w3 :: SamplerIO Bool
w3 = evalF weatherModel Map.empty weatherSen3

-- compute the probability of True by sampling
no_samples2 = 1000000
evaluate :: SamplerIO Bool -> IO Double
evaluate sampler = do
  results <- sampleIO $ sequence $ replicate no_samples2 sampler
  let (trues, total) =
        foldr (\b (t, n) -> (if b then t + 1 else t, n + 1)) (0, 0) results
  return $ if total == 0 then 0 else fromIntegral trues / fromIntegral total

main :: IO ()
main = do
  putStrLn "frequency of True for each weather example:"
  (evaluate w1) >>= print
  (evaluate w2) >>= print
  (evaluate w3) >>= print

-- Different approaches to integration in Haskell
integrationComparison :: IO ()
integrationComparison = do
  putStrLn "\n=== Integration Methods Comparison ==="

  -- Method 1: Using monad-bayes Integrator (what we showed earlier)
  putStrLn "\n1. Monad-Bayes Integrator (more points?):"
  let points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  let betterUniform = integrator (\f -> sum (map f points) / fromIntegral (length points))
  let integral1b = runIntegrator (\x -> x*x) betterUniform
  putStrLn $ "   Using " ++ show (length points) ++ " points: ∫ x² dx from 0 to 1 ≈ " ++ show integral1b

  -- Method 2: Trapezoidal rule via numeric-tools
  putStrLn "\n2. Trapezoidal Rule (numeric-tools):"
  let resTrap = quadTrapezoid defQuad (0.0, 1.0) (\x -> x*x)
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show (quadRes resTrap)
           ++ ", prec≈ " ++ show (quadPrecEst resTrap)
           ++ ", iters= " ++ show (quadNIter resTrap)

  -- Method 3: Simpson's rule via numeric-tools
  putStrLn "\n3. Simpson's Rule (numeric-tools):"
  let resSimp = quadSimpson defQuad (0.0, 1.0) (\x -> x*x)
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show (quadRes resSimp)
           ++ ", prec≈ " ++ show (quadPrecEst resSimp)
           ++ ", iters= " ++ show (quadNIter resSimp)

  -- Method 3b: Romberg via numeric-tools
  putStrLn "\n3b. Romberg (numeric-tools):"
  let resRomb = quadRomberg defQuad (0.0, 1.0) (\x -> x*x)
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show (quadRes resRomb)
           ++ ", prec≈ " ++ show (quadPrecEst resRomb)
           ++ ", iters= " ++ show (quadNIter resRomb)

  -- Method 4: Analytical (when possible)
  putStrLn "\n4. Analytical Solution (Exact):"
  let analytical = 1/3 :: Double
  putStrLn $ "   ∫ x² dx from 0 to 1 = " ++ show analytical

  -- Method 5: Monte Carlo (simple implementation)
  putStrLn "\n5. Monte Carlo Integration:"
  let monteCarlo :: (Double -> Double) -> Double -> Double -> Int -> IO Double
      monteCarlo f a b n = do
        points <- sequence (replicate n (randomRIO (a, b)))
        let values = map f points
        return $ ((b - a) / fromIntegral n) * sum values
  integral5 <- monteCarlo (\x -> x*x) 0.0 1.0 10000
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show integral5

  -- Method 6: GaussQuadIntegration (professional numerical library)
  putStrLn "\n6. GaussQuadIntegration (Professional Library):"
  let integral6 = nIntegrate256 (\x -> x*x) 0.0 1.0
  putStrLn $ "   ∫ x² dx from 0 to 1 ≈ " ++ show integral6

  -- Demonstrate different integration domains
  putStrLn "\n=== Integration Domain Capabilities ==="

  putStrLn "\nGaussQuadIntegration Capabilities:"
  putStrLn "✅ 1D intervals: [a,b]"
  putStrLn "✅ High precision (up to 1024 points)"
  putStrLn "✅ Arbitrary precision types (Fixed, etc.)"
  putStrLn "❌ Multi-dimensional domains"
  putStrLn "❌ Non-rectangular regions"
  putStrLn "❌ Complex geometries"

  -- Example: Different 1D intervals
  putStrLn $ "\nDifferent 1D examples with GaussQuadIntegration:"
  let int1 = nIntegrate256 (\x -> sin x) 0.0 pi
  putStrLn $ "∫ sin(x) dx from 0 to π = " ++ show int1 ++ " (exact: 2.0)"

  let int2 = nIntegrate256 (\x -> exp (-x*x)) (-1.0) 1.0
  putStrLn $ "∫ e^(-x²) dx from -1 to 1 ≈ " ++ show int2

  let int3 = nIntegrate256 (\x -> 1/sqrt(2*pi) * exp(-x*x/2)) (-3.0) 3.0
  putStrLn $ "∫ φ(x) dx from -3 to 3 (normal CDF) ≈ " ++ show int3 ++ " (should be ~1.0)"

  putStrLn "\n=== How Monad-Bayes Integrator Works ==="
  putStrLn "• integrator :: ((a -> Double) -> Double) -> Integrator a"
  putStrLn "• runIntegrator :: (a -> Double) -> Integrator a -> Double"
  putStrLn "• The 'integrator' function creates a measure from an integration kernel"
  putStrLn "• The kernel (a->Double)->Double defines how to integrate any function"
  putStrLn "• Our example: uniform on [0,1] using discrete point evaluation"

  putStrLn "\n=== Alternatives for Complex Domains ==="
  putStrLn "For multi-dimensional and complex geometries:"
  putStrLn "• adaptive-cubature: Multi-dimensional adaptive integration"
  putStrLn "• scubature: Integration over simplices (triangles/tetrahedrons)"
  putStrLn "• Monte Carlo methods: For very high-dimensional problems"
  putStrLn "• Domain decomposition: Break complex regions into simple pieces"

  putStrLn "\nExample complex domain approaches:"
  putStrLn "• 2D circle: Use polar coordinates + 1D integration"
  putStrLn "• 3D sphere: Use spherical coordinates + 2D integration"
  putStrLn "• Irregular regions: Use adaptive quadrature or triangulation"
  putStrLn "• Complex boundaries: Use Green's theorem or change of variables"

  putStrLn "\n=== Summary ==="
  putStrLn "• For exact results: Use analytical methods when possible"
  putStrLn "• For 1D intervals: GaussQuadIntegration ⭐ (fast, accurate)"
  putStrLn "• For multi-D: adaptive-cubature or scubature"
  putStrLn "• For complex geometries: Monte Carlo or domain decomposition"
  putStrLn "• For probabilistic programming: Use monad-bayes Integrator"
  putStrLn "• For general scientific computing: Use hmatrix ecosystem"
  putStrLn "• For simple cases: Implement trapezoidal/Simpson's rule directly"
  putStrLn "• For custom measures: Build your own integration kernel with Integrator"

-- Testing infinite lists in NeSy. Is it not supposed to handle infinite lists?
testInfiniteLists :: IO ()
testInfiniteLists = do
  putStrLn "\n=== Testing Infinite Lists in NeSy ==="

  -- Create an infinite list
  let infiniteDomain = [0.0, 0.1 ..] :: [Double]  -- Infinite: [0.0, 0.1, 0.2, ...]
  putStrLn $ "Infinite domain created: [0.0, 0.1, 0.2, ...] (lazy)"

  -- Test with finite prefix (safe)
  let finitePrefix = take 5 infiniteDomain
  putStrLn $ "Finite prefix: " ++ show finitePrefix

  -- Test quantifiers with finite prefix
  let testPred x = x > 0.5
  putStrLn $ "∃x ∈ finitePrefix: x > 0.5 = " ++ show (any testPred finitePrefix)
  putStrLn $ "∀x ∈ finitePrefix: x > 0.5 = " ++ show (all testPred finitePrefix)

  -- Now try with larger finite prefix
  let largerPrefix = take 20 infiniteDomain
  putStrLn $ "\nWith 20 elements:"
  putStrLn $ "∃x ∈ largerPrefix: x > 0.5 = " ++ show (any testPred largerPrefix)
  putStrLn $ "∀x ∈ largerPrefix: x > 0.5 = " ++ show (all testPred largerPrefix)

  -- Show what happens with very large prefix (approaches infinite behavior)
  let bigPrefix = take 100 infiniteDomain
  putStrLn $ "\nWith 100 elements (approximating infinite):"
  putStrLn $ "∃x ∈ bigPrefix: x > 0.5 = " ++ show (any testPred bigPrefix)
  putStrLn $ "∀x ∈ bigPrefix: x > 0.5 = " ++ show (all testPred bigPrefix)

  -- Demonstrate the problem: what happens if we try to use infinite directly?
  putStrLn $ "\n=== The Problem with True Infinite Lists ==="
  putStrLn "If we tried: foldr disj bot $ map testPred infiniteDomain"
  putStrLn "This would: 1) Never terminate, 2) Use infinite memory, 3) Loop forever"
  putStrLn "Because foldr would keep evaluating elements forever!"

  -- Show lazy evaluation to the rescue
  putStrLn $ "\n=== Lazy Evaluation Solution ==="
  let lazyExists pred list = any pred (take 1000 list)  -- Limit to 1000 elements
  let lazyForall pred list = all pred (take 1000 list)  -- Limit to 1000 elements

  putStrLn $ "Lazy ∃x: x > 0.5 (first 1000 elements) = " ++ show (lazyExists testPred infiniteDomain)
  putStrLn $ "Lazy ∀x: x > 0.5 (first 1000 elements) = " ++ show (lazyForall testPred infiniteDomain)
