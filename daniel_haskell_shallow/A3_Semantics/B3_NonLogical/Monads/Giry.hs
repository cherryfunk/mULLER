{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeSynonymInstances #-}

module NonLogical.Monads.Giry where

import Control.Monad (ap)
import NonLogical.Categories.DATA (DATA (..), MonadOver (..))
import Statistics.Distribution (ContDistr, Mean, Variance)

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
  -- | Discrete uniform on a finite set
  DisUniform :: [a] -> Giry a
  -- | Poisson distribution (canonical measure on $\mathbb{N}$)
  Poisson :: Double -> Giry Int
  -- | Geometric distribution (canonical measure on $\mathbb{N}_0$)
  Geometric :: Double -> Giry Int
  -- Continuous (over Reals)
  Normal :: Double -> Double -> Giry Double
  Uniform :: Double -> Double -> Giry Double
  Exponential :: Double -> Giry Double
  Beta :: Double -> Double -> Giry Double
  Gamma :: Double -> Double -> Giry Double
  Laplace :: Double -> Double -> Giry Double
  StudentT :: Double -> Giry Double
  -- | Escape Hatch: ANY Continuous distribution from `statistics`
  GenericCont :: (ContDistr d, Mean d, Variance d) => d -> Giry Double
  -- | Escape Hatch: Arbitrary pdf function and lower/upper bounds
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
