{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

-- | Logical interpretation: \L ukasiewicz Logic ($\Omega = [0,1]$)
module A2_Interpretation.B3_Logical.Lukasiewicz where

import A2_Interpretation.B2_Typological.Categories.DATA (DATA (..))
import A3_Semantics.B4_NonLogical.Monads.Expectation (HasExpectation (..))
import A2_Interpretation.B1_Categorical.Monads.Giry (Giry (..))
import A2_Interpretation.B4_NonLogical.Supremum (enumAll, inf, sup)

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

-- | $\mathcal{I}(\oplus)$ : Bounded sum
oplus :: Omega -> Omega -> Omega
oplus x y = min 1.0 (x + y)

-- | $\mathcal{I}(\otimes)$ : Bounded product
otimes :: Omega -> Omega -> Omega
otimes x y = max 0.0 (x + y - 1.0)

-- | $\mathcal{I}(\vec{0})$ : Additive unit
v0 :: Omega
v0 = 0.0

-- | $\mathcal{I}(\vec{1})$ : Multiplicative unit
v1 :: Omega
v1 = 1.0

-- | $\mathcal{I}(\neg)$ : Negation
neg :: Omega -> Omega
neg x = 1.0 - x

------------------------------------------------------
-- Quantifiers ($Q_a :: (a \to \Omega) \to \Omega$)
------------------------------------------------------

-- | $\mathcal{I}(\bigvee)$ : Supremum
bigVee :: forall a. DATA a -> (a -> Omega) -> Omega
bigVee = sup

-- | $\mathcal{I}(\bigwedge)$ : Infimum
bigWedge :: forall a. DATA a -> (a -> Omega) -> Omega
bigWedge = inf

-- | $\mathcal{I}(\bigoplus)$ : Infinitary Bounded Sum: min(1, Sigma phi(x))
--   = min(1, $\mathbb{E}_\mu[\varphi]$) for continuous domains.
bigOplus :: forall a. DATA a -> (a -> Omega) -> Omega
bigOplus Reals phi = min 1.0 (expect Reals (Uniform 0.0 1.0) phi)
bigOplus (Prod da db) phi = bigOplus da (\a -> bigOplus db (\b -> phi (a, b)))
bigOplus d phi = min 1.0 (sum (map phi (enumAll d)))

-- | $\mathcal{I}(\bigotimes)$ : Infinitary Bounded Product: $\max(0, \sum \varphi(x) - (n-1))$
bigOtimes :: forall a. DATA a -> (a -> Omega) -> Omega
bigOtimes Reals phi = error "Lukasiewicz bigOtimes over R requires integration."
bigOtimes (Prod da db) phi = bigOtimes da (\a -> bigOtimes db (\b -> phi (a, b)))
bigOtimes d phi =
  let xs = map phi (enumAll d)
   in max 0.0 (sum xs - fromIntegral (length xs - 1))

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
