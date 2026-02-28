{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

-- | Logical interpretation: G\"odel Logic ($\Omega = [0,1]$)
module Logical.Interpretations.Goedel where

import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Supremum (inf, sup)

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

-- | $\mathcal{I}(\oplus)$ : Max
oplus :: Omega -> Omega -> Omega
oplus = max

-- | $\mathcal{I}(\otimes)$ : Min
otimes :: Omega -> Omega -> Omega
otimes = min

-- | $\mathcal{I}(\vec{0})$ : Additive unit
v0 :: Omega
v0 = 0.0

-- | $\mathcal{I}(\vec{1})$ : Multiplicative unit
v1 :: Omega
v1 = 1.0

-- | $\mathcal{I}(\neg)$ : Negation (intuitionistic: neg0 = 1, negx = 0 for x > 0)
neg :: Omega -> Omega
neg x = if x == bot then top else bot

------------------------------------------------------
-- Quantifiers ($Q_a :: (a \to \Omega) \to \Omega$)
-- G\"odel: idempotent, so bigoplus = bigvee and bigotimes = bigwedge
------------------------------------------------------

-- | $\mathcal{I}(\bigvee)$ : Supremum
bigVee :: forall a. DATA a -> (a -> Omega) -> Omega
bigVee = sup

-- | $\mathcal{I}(\bigwedge)$ : Infimum
bigWedge :: forall a. DATA a -> (a -> Omega) -> Omega
bigWedge = inf

-- | $\mathcal{I}(\bigoplus)$ : Infinitary Strong Disjunction
bigOplus :: forall a. DATA a -> (a -> Omega) -> Omega
bigOplus = sup

-- | $\mathcal{I}(\bigotimes)$ : Infinitary Strong Conjunction
bigOtimes :: forall a. DATA a -> (a -> Omega) -> Omega
bigOtimes = inf

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
