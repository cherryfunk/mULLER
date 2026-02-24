{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module NonLogical.Monads.Dist where

import Control.Monad (ap, liftM)
import NonLogical.Categories.DATA (DATA, MonadOver (..))

-- | The Distribution Monad.
-- We use newtype to define our custom probability logic for >>=
newtype Dist a = Dist {runDist :: [(a, Double)]}
  deriving (Show)

-- Standard Haskell Monad Hierarchy
instance Functor Dist where
  fmap :: (a -> b) -> Dist a -> Dist b
  fmap = liftM

instance Applicative Dist where
  pure :: a -> Dist a
  pure x = Dist [(x, 1.0)]

  (<*>) :: Dist (a -> b) -> Dist a -> Dist b
  (<*>) = ap

instance Monad Dist where
  return :: a -> Dist a
  return = pure

  (>>=) :: Dist a -> (a -> Dist b) -> Dist b
  (Dist xs) >>= f =
    Dist $
      concat
        [[(y, p * q) | (y, q) <- runDist (f x)] | (x, p) <- xs]

-- | Dist is a monad ON the DATA category.
instance MonadOver DATA Dist

-- | Compute expectation of a function over a distribution
expectDist :: Dist a -> (a -> Double) -> Double
expectDist (Dist xs) f = sum [p * f x | (x, p) <- xs]
