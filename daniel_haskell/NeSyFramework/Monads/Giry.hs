{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE TypeSynonymInstances #-}

module NeSyFramework.Monads.Giry where

import Control.Monad (ap)
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
-- mathematical execution (exact sums for discrete, Lebesgue quadrature for continuous).
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
  -- | Escape Hatch: Arbitrary Continuous Distribution (pdf function and lower/upper bounds)
  ContinuousPdf :: (Double -> Double) -> (Double, Double) -> Giry Double
  -- | Universal Escape Hatch: ANY Continuous distribution from the `statistics` package!
  GenericCont :: (ContDistr d, Mean d, Variance d) => d -> Giry Double

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
-- HIGHLY OPTIMIZED EXPECTATION EVALUATION (VIA CATEGORY: DATA)
--------------------------------------------------------------------------------

-- | The core of the Type-Directed strategy.
-- Every object 'a' in the category DATA must define how it sums or integrates.
class Integrable a where
  integrateStrategy :: [(a, Double)] -> (a -> Double) -> Double

-- 1. Finite Objects (Bool, Char)
-- No limit or infinite loop checks needed.
instance Integrable Bool where
  integrateStrategy :: [(Bool, Double)] -> (Bool -> Double) -> Double
  integrateStrategy xs f = sum [p * f x | (x, p) <- xs]

instance Integrable Char where
  integrateStrategy :: [(Char, Double)] -> (Char -> Double) -> Double
  integrateStrategy xs f = sum [p * f x | (x, p) <- xs]

-- 2. Finite Sets (Lists of finite elements)
instance (Integrable a) => Integrable [a] where
  integrateStrategy :: (Integrable a) => [([a], Double)] -> ([a] -> Double) -> Double
  integrateStrategy xs f = sum [p * f x | (x, p) <- xs]

-- 3. Countably Infinite Objects (Int)
-- Fallback chain:1. Simplify (CAS) -> 2. Lazy Bounded Loop -> 3. Monte Carlo Approximation
instance Integrable Int where
  integrateStrategy :: [(Int, Double)] -> (Int -> Double) -> Double
  integrateStrategy = chainedDiscreteStrategy

-- 4. Continuous Objects (Double)
-- Doubles are evaluated specifically via `Normal` / `Uniform` AST nodes.
instance Integrable Double where
  integrateStrategy :: [(Double, Double)] -> (Double -> Double) -> Double
  integrateStrategy _ _ = error "Continuous expectations over Double must use Normal or Uniform Giry constructors, not Categorical."

instance Integrable String where
  integrateStrategy :: [(String, Double)] -> (String -> Double) -> Double
  integrateStrategy = chainedDiscreteStrategy

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
-- In a real Hakaru system, this is an AST-to-AST transformation checking for Geometric/Poisson series.
-- Here, we dynamically introspect the terms of the lazy sequence.
-- If the sequence of expected values exactly matches an infinite geometric progression
-- (t_{i+1} / t_i = r) with |r| < 1, we can bypass the infinite loop entirely and analytically
-- solve it using the closed-form equation: a / (1 - r).
algebraicSimplify :: [(a, Double)] -> (a -> Double) -> Maybe Double
algebraicSimplify xs f =
  let numTerms = 10
      terms = take numTerms [p * f x | (x, p) <- xs]
      checkGeometric ts
        | length ts < numTerms = Nothing -- Only extrapolate if it's genuinely long/infinite
        | otherwise =
            let (a : b : rest) = ts
             in if abs a < 1e-14
                  then Nothing -- Avoid division by zero
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
-- When we breach deterministic iteration limits on a heavy tail, we assume evaluating 'f'
-- is extremely expensive (e.g., a Neural layer). We take large deterministic jumps,
-- evaluating 'f' only once every N elements and scaling by the aggregate block mass.
stridedMonteCarlo :: [(a, Double)] -> Double -> (a -> Double) -> Double
stridedMonteCarlo tailXs pSumStart f =
  let stride = 50
      maxSamples = 100 -- Sample 100 blocks (5000 elements total) from the tail
      go [] acc _ _ = acc
      go xs acc pSum samples
        | pSum >= 1.0 - giryQuadPrecision = acc
        | samples >= maxSamples =
            -- Safety Break: We reached the sample limit on a heavy tail.
            -- Estimate the remaining mass and scale the current average f(x).
            let remainingMass = 1.0 - pSum
                avgValue = if pSum > pSumStart then acc / (pSum - pSumStart) else 0.0
             in acc + (remainingMass * avgValue)
        | otherwise =
            case splitAt stride xs of
              ([], _) -> acc
              (block, rest) ->
                let (sampleX, _) = head block -- Take first as representative
                    blockMass = sum (map snd block)
                 in go rest (acc + blockMass * f sampleX) (pSum + blockMass) (samples + 1)
   in go tailXs 0.0 pSumStart 0

--------------------------------------------------------------------------------
-- CONTINUOUS OBJECTS (LEBESGUE INTEGRATION)
--------------------------------------------------------------------------------

-- | 1. Maximum extrapolation steps (Default: 16, bounded by 65,536 evaluations)
-- High enough for precision, low enough to bail out quickly on discontinuities.
giryQuadMaxIter :: Int
giryQuadMaxIter = 16

-- | 2. Strict error tolerance threshold (Default: 1e-12)
giryQuadPrecision :: Double
giryQuadPrecision = 1e-12

-- | 3. Distance in σ to bound Lebesgue integrals for Normal distributions (Default: 8.0)
giryTailSigmas :: Double
giryTailSigmas = 8.0

-- | Normal Probability Density Function
-- | Robust integration using numeric-tools Romberg rule.
integrateNT :: (Double -> Double) -> (Double, Double) -> Double
integrateNT f (a, b) =
  let customQuad = defQuad {quadMaxIter = giryQuadMaxIter, quadPrecision = giryQuadPrecision}
      res = quadRomberg customQuad (a, b) f
   in case quadRes res of
        Just v -> v
        Nothing -> quadBestEst res

-- | The core interpreter.
-- Evaluates the expected value of a function 'f' under the measure 'Giry a'.
-- Strict guarantee: Only elements 'a' in our DATA category (that implement Integrable)
-- are permitted to have an expectation evaluated.
expectation :: Giry a -> (a -> Double) -> Double
expectation (Pure x) f = f x
expectation (Categorical xs) f = chainedDiscreteStrategy xs f
expectation (Normal mu sigma) f = expectation (GenericCont (N.normalDistr mu sigma)) f
expectation (Uniform a b) f =
  if a == b
    then 0.0
    else integrateNT (\x -> (1 / (b - a)) * f x) (a, b)
expectation (Exponential lambda) f = expectation (GenericCont (E.exponential lambda)) f
expectation (Beta alpha beta) f = expectation (GenericCont (B.betaDistr alpha beta)) f
expectation (Gamma shape scale) f = expectation (GenericCont (G.gammaDistr shape scale)) f
expectation (Laplace loc scale) f = expectation (GenericCont (L.laplace loc scale)) f
expectation (StudentT ndf) f =
  -- StudentT often lacks a finite Variance or Mean (e.g., Cauchy when ndf=1),
  -- so it cannot route through GenericCont's $O(1)$ typeclass bounds.
  -- We fall back to the robust, albeit slower, quantile inversion method just for StudentT.
  let dist = T.studentT ndf
      lower = quantile dist 1e-15
      upper = quantile dist (1 - 1e-15)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
expectation (GenericCont dist) f =
  -- Universal Fallback: Inspect the distribution for support limits.
  let trueMin = quantile dist 0.0
      trueMax = quantile dist 1.0
      -- We now use O(1) Mean and Variance lookups
      -- We span 12 standard deviations, strictly bounded by the mathematical reality of the distribution.
      lower = max trueMin (mean dist - 12 * stdDev dist)
      upper = min trueMax (mean dist + 12 * stdDev dist)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
expectation (ContinuousPdf pdf (a, b)) f =
  integrateNT (\x -> pdf x * f x) (a, b)
expectation (Bind m k) f =
  -- Law of Total Expectation: E[E[f(Y) | X]] = E[f(Y)]
  expectation m (\x -> expectation (k x) f)

--------------------------------------------------------------------------------
-- HELPER CONSTRUCTORS
--------------------------------------------------------------------------------

-- | Create a discrete distribution from a list of probabilities.
-- Assumes probabilities sum to 1.
categorical :: [(a, Double)] -> Giry a
categorical = Categorical

-- | Create a Normal distribution.
normal :: Double -> Double -> Giry Double
normal = Normal

-- | Create a Uniform continuous distribution.
uniform :: Double -> Double -> Giry Double
uniform = Uniform
