{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeSynonymInstances #-}

module NonLogical.Monads.Giry where

import Control.Monad (ap)
import NonLogical.Categories.DATA (DATA (..), MonadOver (..))
import Numeric.Tools.Integration (QuadParam (..), defQuad, quadBestEst, quadRes, quadRomberg)
import Statistics.Distribution (ContDistr (density, quantile), Mean (mean), Variance (stdDev))
import qualified Statistics.Distribution.Beta as B
import qualified Statistics.Distribution.Exponential as E
import qualified Statistics.Distribution.Gamma as G
import qualified Statistics.Distribution.Laplace as L
import qualified Statistics.Distribution.Normal as N
import qualified Statistics.Distribution.StudentT as T
import qualified Statistics.Distribution.Uniform as U

-- | Giry is a monad ON the DATA category.
instance MonadOver DATA Giry

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
-- EXPECTATION EVALUATION
-- Giry is a monad ON DATA. For EACH OBJECT of DATA, we define how to integrate.
-- The DATA object drives the strategy for Categorical (discrete) distributions.
-- Continuous distributions (Normal, GenericCont, etc.) self-evaluate via their
-- built-in distribution information.
-- Bind chains use the Law of Total Expectation: E[f(Y)] = E[E[f(Y)|X]].
--------------------------------------------------------------------------------

-- | The public API: compute E_μ[f] where μ : Giry a, for object (DATA a).
-- This is THE contract: for every constructor of DATA, we specify the strategy.
expectation :: DATA a -> Giry a -> (a -> Double) -> Double
-- Pure: structural, same for all objects
expectation _ (Pure x) f = f x
-- Bind: Law of Total Expectation E[f(Y)] = E_x[E[f(Y)|X=x]]
-- The intermediate type x is existentially hidden, so we recurse generically.
-- The inner computation (k x) will hit the correct object-specific branch when
-- it reaches a Categorical leaf.
expectation obj (Bind m k) f = evalBind m (\x -> expectation obj (k x) f)
-- 1. Reals (ℝ) — Categorical on Reals is an error. Use continuous constructors.
expectation Reals (Categorical _) _ = error "Giry: Categorical on Reals — use Normal, Uniform, GenericCont, etc."
expectation Reals giry f = evalContinuous giry f
-- 2. Integers (ℤ) — countably infinite: chained convergence strategy.
expectation Integers (Categorical xs) f = chainedDiscreteStrategy xs f
-- 3. Strings (countably infinite, like Integers).
expectation Strings (Categorical xs) f = chainedDiscreteStrategy xs f
-- 4. Booleans ({True, False}) — finite set with exactly 2 elements.
expectation Booleans (Categorical xs) f = sum [p * f x | (x, p) <- xs]
-- 5. Finite sets ({x₁, ..., xₙ}) — finite direct summation.
expectation (Finite _) (Categorical xs) f = sum [p * f x | (x, p) <- xs]
-- 6. Unit ({()}) — trivial: only one element.
expectation UnitObj _ f = f ()
-- 6. Products (A × B) — Categorical on products: finite summation.
--    For continuous products (ℝ × ℝ), one would need iterated quadrature (Fubini).
--    Currently, products are constructed via Bind chains which decompose them.
expectation (ProductObj _ _) (Categorical xs) f = sum [p * f x | (x, p) <- xs]

--------------------------------------------------------------------------------
-- INTERNAL: Generic Bind evaluator
-- Handles the existential type in Bind :: Giry x -> (x -> Giry a) -> Giry a
-- The intermediate type x is unknown, so we evaluate m generically and pass
-- the result to the continuation.
--------------------------------------------------------------------------------

evalBind :: Giry x -> (x -> Double) -> Double
evalBind (Pure x) f = f x
evalBind (Bind m k) f = evalBind m (\x -> evalBind (k x) f)
evalBind (Categorical xs) f = chainedDiscreteStrategy xs f
evalBind (Normal mu sigma) f =
  evalBind (GenericCont (N.normalDistr mu sigma)) f
evalBind (Uniform a b) f =
  evalBind (GenericCont (U.uniformDistr a b)) f
evalBind (Exponential lambda) f =
  evalBind (GenericCont (E.exponential lambda)) f
evalBind (Beta alpha beta) f =
  evalBind (GenericCont (B.betaDistr alpha beta)) f
evalBind (Gamma shape scale) f =
  evalBind (GenericCont (G.gammaDistr shape scale)) f
evalBind (Laplace loc scale) f =
  evalBind (GenericCont (L.laplace loc scale)) f
evalBind (StudentT ndf) f =
  let dist = T.studentT ndf
      lower = quantile dist 1e-15
      upper = quantile dist (1 - 1e-15)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
evalBind (GenericCont dist) f =
  let trueMin = quantile dist 0.0
      trueMax = quantile dist 1.0
      lower = max trueMin (mean dist - 12 * stdDev dist)
      upper = min trueMax (mean dist + 12 * stdDev dist)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
evalBind (ContinuousPdf pdf (a, b)) f =
  integrateNT (\x -> pdf x * f x) (a, b)

--------------------------------------------------------------------------------
-- INTERNAL: Continuous distribution evaluator (only for Reals)
--------------------------------------------------------------------------------

evalContinuous :: Giry Double -> (Double -> Double) -> Double
evalContinuous (Normal mu sigma) f =
  evalContinuous (GenericCont (N.normalDistr mu sigma)) f
evalContinuous (Uniform a b) f =
  evalContinuous (GenericCont (U.uniformDistr a b)) f
evalContinuous (Exponential lambda) f =
  evalContinuous (GenericCont (E.exponential lambda)) f
evalContinuous (Beta alpha beta) f =
  evalContinuous (GenericCont (B.betaDistr alpha beta)) f
evalContinuous (Gamma shape scale) f =
  evalContinuous (GenericCont (G.gammaDistr shape scale)) f
evalContinuous (Laplace loc scale) f =
  evalContinuous (GenericCont (L.laplace loc scale)) f
evalContinuous (StudentT ndf) f =
  let dist = T.studentT ndf
      lower = quantile dist 1e-15
      upper = quantile dist (1 - 1e-15)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
evalContinuous (GenericCont dist) f =
  let trueMin = quantile dist 0.0
      trueMax = quantile dist 1.0
      lower = max trueMin (mean dist - 12 * stdDev dist)
      upper = min trueMax (mean dist + 12 * stdDev dist)
   in integrateNT (\x -> density dist x * f x) (lower, upper)
evalContinuous (ContinuousPdf pdf (a, b)) f =
  integrateNT (\x -> pdf x * f x) (a, b)
evalContinuous giry f = evalBind giry f -- Pure, Bind, Categorical fallback

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
      -- Fallback 1: Lazy Evaluated Bounded Convergence
      let go _ _ acc [] = acc
          go iter pSum acc ((x, p) : rest)
            | pSum >= 1.0 - giryQuadPrecision = acc + p * f x
            | iter >= giryMaxDiscreteIter =
                -- Fallback 2: Monte Carlo Quasi-Sampling of the infinite heavy tail
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

-- | Fallback 2: Strided Monte Carlo Approximation (Quasi-Monte Carlo)
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
