{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

-- | Logical interpretation: Classical Boolean Logic ($\Omega = \{\text{True}, \text{False}\}$)
module Logical.Interpretations.Boolean where

import NonLogical.Categories.DATA (DATA (..))
import NonLogical.Supremum (enumAll)

infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | Omega := I(tau) = {True, False}
type Omega = Bool

-- | $\mathcal{I}(\vdash)$ : Comparison ($\text{False} \leq \text{True}$)
vdash :: Omega -> Omega -> Bool
vdash = (<=)

-- | $\mathcal{I}(\wedge)$ : Conjunction
wedge :: Omega -> Omega -> Omega
wedge = (&&)

-- | $\mathcal{I}(\vee)$ : Disjunction
vee :: Omega -> Omega -> Omega
vee = (||)

-- | $\mathcal{I}(\bot)$ : Bottom
bot :: Omega
bot = False

-- | $\mathcal{I}(\top)$ : Top
top :: Omega
top = True

-- | $\mathcal{I}(\oplus)$ : Disjunction
oplus :: Omega -> Omega -> Omega
oplus = (||)

-- | $\mathcal{I}(\otimes)$ : Conjunction
otimes :: Omega -> Omega -> Omega
otimes = (&&)

-- | $\mathcal{I}(\vec{0})$ : Additive unit
v0 :: Omega
v0 = False

-- | $\mathcal{I}(\vec{1})$ : Multiplicative unit
v1 :: Omega
v1 = True

-- | $\mathcal{I}(\neg)$ : Negation
neg :: Omega -> Omega
neg = not

------------------------------------------------------
-- Quantifiers ($Q_a :: (a \to \Omega) \to \Omega$)
-- `any`/`all` are lazy and short-circuit on infinite lists.
------------------------------------------------------

-- | $\mathcal{I}(\bigvee)$ : Infinitary Join
bigVee :: forall a. DATA a -> (a -> Omega) -> Omega
bigVee Reals _ = error "Boolean bigVee over R is uncomputable."
bigVee d phi = any phi (enumAll d)

-- | $\mathcal{I}(\bigwedge)$ : Infinitary Meet
bigWedge :: forall a. DATA a -> (a -> Omega) -> Omega
bigWedge Reals _ = error "Boolean bigWedge over R is uncomputable."
bigWedge d phi = all phi (enumAll d)

-- | $\mathcal{I}(\bigoplus)$ : Infinitary Strong Disjunction
bigOplus :: forall a. DATA a -> (a -> Omega) -> Omega
bigOplus = bigVee

-- | $\mathcal{I}(\bigotimes)$ : Infinitary Strong Conjunction
bigOtimes :: forall a. DATA a -> (a -> Omega) -> Omega
bigOtimes = bigWedge

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
