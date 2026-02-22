{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

module NeSyFramework.Monads.Giry where

import Control.Monad (ap)
import NeSyFramework.Categories.DATA (DataObj (..))
import Numeric.Tools.Integration (QuadParam (..), defQuad, quadBestEst, quadRes, quadRomberg)
import Statistics.Distribution (ContDistr (density, quantile), Mean (mean), Variance (stdDev))
import qualified Statistics.Distribution.Beta as B
import qualified Statistics.Distribution.Exponential as E
import qualified Statistics.Distribution.Gamma as G
import qualified Statistics.Distribution.Laplace as L
import qualified Statistics.Distribution.Normal as N
import qualified Statistics.Distribution.StudentT as T

-- | The Giry Monad represented as an Abstract Syntax Tree (Free Monad).
-- This separates the monadic structure from the concrete evaluation, allowing
-- standard unconstrained Monad compliance while enabling highly optimized
-- mathematical execution (sums for discrete, Lebesgue quadrature for continuous).
data Giry a where
  -- | Standard Monad Operations
  Pure :: a -> Giry a
  Bind :: Giry x -> (x -> Giry a) -> Giry a
  -- | Concrete efficient primitives
  -- Discrete
  Categorical :: [(a, Double)] -> Giry a
  -- Continuous (over Reals)
  Normal :: Double -> Double -> Giry Double
  Uniform :: Double -> Double -> Giry Double
  Exponential :: Double -> Giry Double
  Beta :: Double -> Double -> Giry Double
  Gamma :: Double -> Double -> Giry Double
  Laplace :: Double -> Double -> Giry Double
  StudentT :: Double -> Giry Double
  -- | Escape Hatch: ANY Continuous distribution from the `statistics` package!
  GenericCont :: (ContDistr d, Mean d, Variance d) => d -> Giry Double
  -- | Universal Escape Hatch: Arbitrary Continuous Distribution (pdf function and lower/upper bounds)
  ContinuousPdf :: (Double -> Double) -> (Double, Double) -> Giry Double

instance Functor Giry where
  fmap :: (a -> b) -> Giry a -> Giry b
  fmap f m = Bind m (Pure . f)

instance Applicative Giry where
  pure :: a -> Giry a
  pure = Pure

  (<*>) :: Giry (a -> b) -> Giry a -> Giry b
  (<*>) = ap

instance Monad Giry where
  return :: a -> Giry a
  return = pure

  (>>=) :: Giry a -> (a -> Giry b) -> Giry b
  (>>=) = Bind

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-- EXPECTATION EVALUATION (DRIVEN BY CATEGORY DATA)
--------------------------------------------------------------------------------

-- | The public API for evaluating expectations.
-- Takes a DataObj to specify which object of DATA we are computing the expectation on.
-- This tells the Giry monad how to integrate at the top level.
--
-- Information flow: DATA defines objects -> Giry reads them and decides how to integrate.
--
-- For compound expressions (Bind chains), the inner measures self-identify via their
-- constructors (Categorical, Normal, etc.), so the DataObj drives the top-level strategy
-- while inner layers are handled by the constructor-driven 'eval' function.
expectation :: DataObj a -> Giry a -> (a -> Double) -> Double
expectation obj giry f = eval giry f
  where
    -- For Categorical at the top level, use the DataObj to choose strategy
    eval (Categorical xs) g = integrateDiscrete obj xs g
    -- All other cases delegate to the constructor-driven evaluator
    eval other g = evalGiry other g

-- | Constructor-driven evaluator (handles Bind chains and all Giry constructors).
-- This is the internal workhorse that evaluates the AST recursively.
-- For Categorical, it defaults to finite summation (safe for Bind-internal categoricals).
-- For continuous constructors, it uses their built-in distribution information.
evalGiry :: Giry a -> (a -> Double) -> Double
evalGiry (Pure x) f = f x
evalGiry (Bind m k) f =
  -- Law of Total Expectation: E[E[f(Y) | X]] = E[f(Y)]
  evalGiry m (\x -> evalGiry (k x) f)
evalGiry (Categorical xs) f =
  -- Must use chained strategy since Categorical can hold infinite lazy lists.
  -- chainedDiscreteStrategy handles both finite and infinite cases correctly.
  chainedDiscreteStrategy xs f
evalGiry (Normal mu sigma) f =
  evalGiry (GenericCont (N.normalDistr mu sigma)) f
evalGiry (Uniform a b) f =
  if a == b
    then 0.0
    else integrateNT (\x -> (1 / (b - a)) * f x) (a, b)
evalGiry (Exponential lambda) f =
  evalGiry (GenericCont (E.exponential lambda)) f
evalGiry (Beta alpha beta) f =
  evalGiry (GenericCont (B.betaDistr alpha beta)) f
evalGiry (Gamma shape scale) f =
  evalGiry (GenericCont (G.gammaDistr shape scale)) f
evalGiry (Laplace loc scale) f =
  evalGiry (GenericCont (L.laplace loc scale)) f
evalGiry (StudentT ndf) f =
  -- StudentT often lacks a finite Variance or Mean (e.g., Cauchy when ndf=1),
  -- so it cannot route through GenericCont's O(1) typeclass bounds.
  let dist = T.studentT ndf
      lower = quantile dist 1e-15
      upper = quantile dist (1 - 1e-15)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
evalGiry (GenericCont dist) f =
  -- Universal Fallback: Inspect the distribution for support limits.
  let trueMin = quantile dist 0.0
      trueMax = quantile dist 1.0
      -- We use O(1) Mean and Variance lookups
      -- We span 12 standard deviations, strictly bounded by the mathematical reality of the distribution.
      lower = max trueMin (mean dist - 12 * stdDev dist)
      upper = min trueMax (mean dist + 12 * stdDev dist)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
evalGiry (ContinuousPdf pdf (a, b)) f =
  integrateNT (\x -> pdf x * f x) (a, b)

-- | Discrete integration strategy, driven by the DataObj.
-- This is where the category DATA tells the Giry monad HOW to sum.
integrateDiscrete :: DataObj a -> [(a, Double)] -> (a -> Double) -> Double
-- Finite objects: direct finite summation (always terminates)
integrateDiscrete (Finite _) xs f = sum [p * f x | (x, p) <- xs]
integrateDiscrete Booleans xs f = sum [p * f x | (x, p) <- xs]
integrateDiscrete UnitObj xs f = sum [p * f x | (x, p) <- xs]
integrateDiscrete (ProductObj _ _) xs f = sum [p * f x | (x, p) <- xs]
-- Countably infinite objects: chained fallback strategy
integrateDiscrete Integers xs f = chainedDiscreteStrategy xs f
-- Reals with Categorical: likely an error
integrateDiscrete Reals _ _ =
  error "Categorical on Reals: use continuous Giry constructors (Normal, Uniform, etc.) instead."

--------------------------------------------------------------------------------
-- COUNTABLY INFINITE OBJECTS (CHAIN OF FALLBACKS)
--------------------------------------------------------------------------------

-- | Maximum discrete iterations for infinite support like Countable Sets
giryMaxDiscreteIter :: Int
giryMaxDiscreteIter = 10000

-- | Chained Fallback Strategy for Countably Infinite Types
-- 1. Try Algebraic Simplify (Heuristic check)
chainedDiscreteStrategy :: [(a, Double)] -> (a -> Double) -> Double
chainedDiscreteStrategy xs f =
  case algebraicSimplify xs f of
    Just exactVal -> exactVal
    Nothing ->
      -- Fallback 2: Lazy Evaluated Bounded Convergence
      let go _ _ acc [] = acc
          go iter pSum acc ((x, p) : rest)
            | pSum >= 1.0 - giryQuadPrecision = acc + p * f x
            | iter >= giryMaxDiscreteIter =
                -- Fallback 3: Monte Carlo Quasi-Sampling of the infinite heavy tail
                acc + p * f x + stridedMonteCarlo rest (pSum + p) f
            | otherwise = go (iter + 1) (pSum + p) (acc + p * f x) rest
       in go 0 0.0 0.0 xs

-- | Fallback 1: Algebraic Simplify (Heuristic Runtime CAS check)
algebraicSimplify :: [(a, Double)] -> (a -> Double) -> Maybe Double
algebraicSimplify xs f =
  let numTerms = 10
      terms = take numTerms [p * f x | (x, p) <- xs]
      checkGeometric ts
        | length ts < numTerms = Nothing
        | otherwise =
            let (a : b : rest) = ts
             in if abs a < 1e-14
                  then Nothing
                  else
                    let r = b / a
                        isGeometric _ [] = True
                        isGeometric prev (curr : rs)
                          | abs prev < 1e-14 = False
                          | abs ((curr / prev) - r) < 1e-9 = isGeometric curr rs
                          | otherwise = False
                     in if abs r < (1.0 - 1e-9) && isGeometric b rest
                          then Just (a / (1.0 - r))
                          else Nothing
   in checkGeometric terms

-- | Fallback 3: Strided Monte Carlo Approximation (Quasi-Monte Carlo)
stridedMonteCarlo :: [(a, Double)] -> Double -> (a -> Double) -> Double
stridedMonteCarlo tailXs pSumStart f =
  let stride = 50
      maxSamples = 100
      go [] acc _ _ = acc
      go xs acc pSum samples
        | pSum >= 1.0 - giryQuadPrecision = acc
        | samples >= maxSamples =
            let remainingMass = 1.0 - pSum
                avgValue = if pSum > pSumStart then acc / (pSum - pSumStart) else 0.0
             in acc + (remainingMass * avgValue)
        | otherwise =
            case splitAt stride xs of
              ([], _) -> acc
              (block, rest) ->
                let (sampleX, _) = head block
                    blockMass = sum (map snd block)
                 in go rest (acc + blockMass * f sampleX) (pSum + blockMass) (samples + 1)
   in go tailXs 0.0 pSumStart 0

--------------------------------------------------------------------------------
-- CONTINUOUS OBJECTS (LEBESGUE INTEGRATION)
--------------------------------------------------------------------------------

giryQuadMaxIter :: Int
giryQuadMaxIter = 16

giryQuadPrecision :: Double
giryQuadPrecision = 1e-12

giryTailSigmas :: Double
giryTailSigmas = 8.0

-- | Robust integration using numeric-tools Romberg rule.
integrateNT :: (Double -> Double) -> (Double, Double) -> Double
integrateNT f (a, b) =
  let customQuad = defQuad {quadMaxIter = giryQuadMaxIter, quadPrecision = giryQuadPrecision}
      res = quadRomberg customQuad (a, b) f
   in case quadRes res of
        Just v -> v
        Nothing -> quadBestEst res

--------------------------------------------------------------------------------
-- HELPER CONSTRUCTORS
--------------------------------------------------------------------------------

categorical :: [(a, Double)] -> Giry a
categorical = Categorical

normal :: Double -> Double -> Giry Double
normal = Normal

uniform :: Double -> Double -> Giry Double
uniform = Uniform

exponential :: Double -> Giry Double
exponential = Exponential

beta :: Double -> Double -> Giry Double
beta = Beta

gamma :: Double -> Double -> Giry Double
gamma = Gamma

laplace :: Double -> Double -> Giry Double
laplace = Laplace

studentT :: Double -> Giry Double
studentT = StudentT

genericCont :: (ContDistr d, Mean d, Variance d) => d -> Giry Double
genericCont = GenericCont

continuousPdf :: (Double -> Double) -> (Double, Double) -> Giry Double
continuousPdf = ContinuousPdf
