{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

-- | Logical interpretation: Logic Tensor Networks with p-mean approximation (LTNp, $\$\Omega = [0,1]$ \subset \mathbb{R}$)
module Logical.Interpretations.LTNp where

import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Monads.Expectation (HasExpectation (..))
import NonLogical.Monads.Giry (Giry (..))
import NonLogical.Supremum (enumAll, inf, sup)

infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | Omega := I(tau) = [0,1]
type Omega = Double

-- | Hyperparameter p for LTN approximations (default p=2)
p_LTN :: Double
p_LTN = 2.0

-- | $\mathcal{I}(\vdash)$ : Comparison
vdash :: Omega -> Omega -> Bool
vdash = (<=)

-- | $\mathcal{I}(\wedge)$ : Meet
wedge :: Omega -> Omega -> Omega
wedge = min

-- | $\mathcal{I}(\vee)$ : Join
vee :: Omega -> Omega -> Omega
vee = max

-- | $\mathcal{I}(\bot)$ : Bottom
bot :: Omega
bot = 0.0

-- | $\mathcal{I}(\top)$ : Top
top :: Omega
top = 1.0

-- | $\mathcal{I}(\oplus)$ : Probabilistic sum (S_P)
oplus :: Omega -> Omega -> Omega
oplus x y = x + y - (x * y)

-- | $\mathcal{I}(\otimes)$ : Product (T_P)
otimes :: Omega -> Omega -> Omega
otimes = (*)

-- | $\mathcal{I}(\vec{0})$ : Additive unit
v0 :: Omega
v0 = 0.0

-- | $\mathcal{I}(\vec{1})$ : Multiplicative unit
v1 :: Omega
v1 = 1.0

-- | $\mathcal{I}(\neg)$ : Classical Negation (neg_C: 1 - x)
neg :: Omega -> Omega
neg x = 1.0 - x

-- | I(->) : S-Product implication (I_{SP}: 1 - x + xy)
impl :: Omega -> Omega -> Omega
impl x y = 1.0 - x + (x * y)

------------------------------------------------------
-- LTN-specific aggregators
------------------------------------------------------

-- | The generalized p-mean: (1/n * Sigma x_i^p)^(1/p)
pMean :: Double -> [Double] -> Double
pMean p xs =
  let n = fromIntegral (length xs)
      sum_p = sum (map (** p) xs)
   in (sum_p / n) ** (1.0 / p)

-- | The error p-mean: 1 - pMean(1 - x)
errPMean :: Double -> [Double] -> Double
errPMean p xs = 1.0 - pMean p (map (\x -> 1.0 - x) xs)

------------------------------------------------------
-- Quantifiers ($Q_a :: (a \to \Omega) \to \Omega$)
-- LTNp uses p-mean for bigvee and error p-mean for bigwedge (smooth approximations
-- of sup/inf), while bigoplus and bigotimes use standard Product logic operators.
------------------------------------------------------

-- | $\mathcal{I}(\bigvee)$ : p-mean (smooth approximation of supremum)
bigVee :: forall a. DATA a -> (a -> Omega) -> Omega
bigVee Reals _ = error "LTNp bigVee over R requires numerical optimization."
bigVee (Prod da db) phi = bigVee da (\a -> bigVee db (\b -> phi (a, b)))
bigVee d phi = pMean p_LTN (map phi (enumAll d))

-- | $\mathcal{I}(\bigwedge)$ : Error p-mean (smooth approximation of infimum)
bigWedge :: forall a. DATA a -> (a -> Omega) -> Omega
bigWedge Reals phi = error "LTNp bigWedge over R requires numerical optimization."
bigWedge (Prod da db) phi = bigWedge da (\a -> bigWedge db (\b -> phi (a, b)))
bigWedge d phi = errPMean p_LTN (map phi (enumAll d))

-- | $\mathcal{I}(\bigoplus)$ : Infinitary Probabilistic Sum: $1 - \prod (1 - \varphi(x))$
bigOplus :: forall a. DATA a -> (a -> Omega) -> Omega
bigOplus Reals phi = 1.0 - exp (expect Reals (Uniform 0.0 1.0) (\x -> log (1.0 - phi x)))
bigOplus (Prod da db) phi = bigOplus da (\a -> bigOplus db (\b -> phi (a, b)))
bigOplus d phi = 1.0 - product (map (\x -> 1.0 - phi x) (enumAll d))

-- | $\mathcal{I}(\bigotimes)$ : Infinitary Product: $\prod \varphi(x)$
bigOtimes :: forall a. DATA a -> (a -> Omega) -> Omega
bigOtimes Reals phi = exp (expect Reals (Uniform 0.0 1.0) (log . phi))
bigOtimes (Prod da db) phi = bigOtimes da (\a -> bigOtimes db (\b -> phi (a, b)))
bigOtimes d phi = product (map phi (enumAll d))

------------------------------------------------------
-- General predicates
------------------------------------------------------

(.==) :: (Eq a) => a -> a -> Omega
x .== y = if x == y then top else bot

(.<) :: (Ord a) => a -> a -> Omega
x .< y = if x < y then top else bot

(.>) :: (Ord a) => a -> a -> Omega
x .> y = if x > y then top else bot

(.<=) :: (Ord a) => a -> a -> Omega
x .<= y = if x <= y then top else bot

(.>=) :: (Ord a) => a -> a -> Omega
x .>= y = if x >= y then top else bot

(./=) :: (Eq a) => a -> a -> Omega
x ./= y = if x /= y then top else bot

b2o :: Bool -> Omega
b2o b = if b then top else bot
