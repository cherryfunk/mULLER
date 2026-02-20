{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}

module NeSyFramework.Monads.Giry where

import Control.Monad (ap)
import Numeric.Tools.Integration (QuadParam (..), defQuad, quadBestEst, quadRes, quadRomberg)

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
-- HIGHLY OPTIMIZED EXPECTATION EVALUATION
--------------------------------------------------------------------------------

-- | Normal Probability Density Function
normalPdf :: Double -> Double -> Double -> Double
normalPdf mu sigma x =
  let s2 = sigma * sigma
   in (1 / (sqrt (2 * pi) * sigma)) * exp (-(x - mu) * (x - mu) / (2 * s2))

-- | Robust integration using numeric-tools Romberg rule.
--
-- TWEAKING PRECISION VS PERFORMANCE:
-- 1. `quadMaxIter`: (Default: 1000) The maximum number of extrapolation steps.
--    - Increase (e.g., 2000+) if integrating severely steep cliffs (like high-scale sigmoids) to prevent `NaN` crashes.
--    - Decrease (e.g., 100) for faster evaluation if formulas are mostly smooth.
-- 2. `quadPrecision`: (Default: 1e-12) The strict error tolerance threshold.
--    - Decrease (e.g., 1e-14) for insane mathematical perfection.
--    - Increase (e.g., 1e-6) for much faster machine learning / neural network level speeds.
-- 3. Lebesgue Support Boundaries:
-- We compute the expectation by bounding the infinite tails of the Normal distribution.
-- - 8*sigma captures 99.9999999999999% of the mathematical area (Extreme Precision).
-- - Drop to 4*sigma (99.99%) or 5*sigma if you want significantly faster integral evaluations.
--
-- | 1. Maximum extrapolation steps (Default: 1000)
giryQuadMaxIter :: Int
giryQuadMaxIter = 1000

-- | 2. Strict error tolerance threshold (Default: 1e-12)
giryQuadPrecision :: Double
giryQuadPrecision = 1e-12

-- | 3. Distance in σ to bound Lebesgue integrals for Normal distributions (Default: 8.0)
giryTailSigmas :: Double
giryTailSigmas = 8.0

-- | Robust integration using numeric-tools Romberg rule.
-- Falls back to the best estimate (`quadBestEst`) if strict convergence fails,
-- mathematically preventing the entire tree from violently crashing to `NaN`.
integrateNT :: (Double -> Double) -> (Double, Double) -> Double
integrateNT f (a, b) =
  let customQuad = defQuad {quadMaxIter = giryQuadMaxIter, quadPrecision = giryQuadPrecision}
      res = quadRomberg customQuad (a, b) f
   in case quadRes res of
        Just v -> v
        Nothing -> quadBestEst res

-- | The core interpreter.
-- Evaluates the expected value of a function 'f' under the measure 'Giry a'.
-- E[f(X)] = ∫ f(x) dX
expectation :: Giry a -> (a -> Double) -> Double
expectation (Pure x) f = f x
expectation (Categorical xs) f =
  sum [p * f x | (x, p) <- xs]
expectation (Normal mu sigma) f =
  integrateNT (\x -> normalPdf mu sigma x * f x) (mu - giryTailSigmas * sigma, mu + giryTailSigmas * sigma)
expectation (Uniform a b) f =
  if a == b
    then 0.0
    else integrateNT (\x -> (1 / (b - a)) * f x) (a, b)
expectation (Bind m k) f =
  -- Law of Total Expectation: E[E[f(Y) | X]] = E[f(Y)]
  -- We compute the expectation of 'k' under the measure 'm'.
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
