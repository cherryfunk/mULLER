{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

-- | Logical interpretation: Product Logic ($\$\Omega = [0,1]$ \subset \mathbb{R}$)
module Logical.Interpretations.Product where

import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Monads.Expectation (HasExpectation (..))
import NonLogical.Monads.Giry (Giry (..))
import NonLogical.Supremum (enumAll, inf, sup)

-- Fixity: comparisons (.==, .<, .>) bind tighter than connectives (wedge, vee)
infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | Omega := I(tau) = [0,1]
type Omega = Double

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

-- | $\mathcal{I}(\oplus)$ : Probabilistic sum
oplus :: Omega -> Omega -> Omega
oplus x y = x + y - (x * y)

-- | $\mathcal{I}(\otimes)$ : Product
otimes :: Omega -> Omega -> Omega
otimes = (*)

-- | $\mathcal{I}(\vec{0})$ : Additive unit
v0 :: Omega
v0 = 0.0

-- | $\mathcal{I}(\vec{1})$ : Multiplicative unit
v1 :: Omega
v1 = 1.0

-- | $\mathcal{I}(\neg)$ : Negation (G\"odel negation: neg0 = 1, negx = 0 for x > 0)
neg :: Omega -> Omega
neg x = if x == bot then top else bot

------------------------------------------------------
-- Quantifiers ($Q_a :: (a \to \Omega) \to \Omega$)
------------------------------------------------------

-- | $\mathcal{I}(\bigvee)$ : Supremum
bigVee :: forall a. DATA a -> (a -> Omega) -> Omega
bigVee = sup

-- | $\mathcal{I}(\bigwedge)$ : Infimum
bigWedge :: forall a. DATA a -> (a -> Omega) -> Omega
bigWedge = inf

-- | $\mathcal{I}(\bigoplus)$ : Infinitary Probabilistic Sum: $1 - \prod (1 - \varphi(x))$
--   = 1 - exp(E_mu[log(1 - phi)]) for continuous domains.
bigOplus :: forall a. DATA a -> (a -> Omega) -> Omega
bigOplus Reals phi = 1.0 - exp (expect Reals (Uniform 0.0 1.0) (\x -> log (1.0 - phi x)))
bigOplus (Prod da db) phi = bigOplus da (\a -> bigOplus db (\b -> phi (a, b)))
bigOplus d phi = 1.0 - product (map (\x -> 1.0 - phi x) (enumAll d))

-- | $\mathcal{I}(\bigotimes)$ : Infinitary Product: $\prod \varphi(x)$
--   = $\exp(\mathbb{E}_\mu[\log \circ \varphi])$ for continuous domains.
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