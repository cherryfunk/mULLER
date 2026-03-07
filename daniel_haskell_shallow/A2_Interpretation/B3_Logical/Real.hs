{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

-- | Logical interpretation: Real-valued Logic ($\Omega = \mathbb{R}$)
module A2_Interpretation.B3_Logical.Real where

import A2_Interpretation.B4_NonLogical.Supremum (enumAll, inf, sup)
import A2_Interpretation.B2_Typological.Categories.DATA (DATA (..))
import A3_Semantics.B4_NonLogical.Monads.Expectation (HasExpectation (..))
import A2_Interpretation.B1_Categorical.Monads.Giry (Giry (..))
import Numeric.Natural (Natural)

infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | \$\Omega := \mathcal{I}(\tau) = \mathbb{R}$ (approximated by IEEE 754 Double)
type Omega = Double

-- | \$\mathcal{I}(\vdash)$ : Comparison
vdash :: Omega -> Omega -> Bool
vdash = (<=)

-- | \$\mathcal{I}(\wedge)$ : Meet
wedge :: Omega -> Omega -> Omega
wedge = min

-- | \$\mathcal{I}(\vee)$ : Join
vee :: Omega -> Omega -> Omega
vee = max

-- | \$\mathcal{I}(\bot)$ : Bottom
bot :: Omega
bot = -1.0 / 0.0

-- | \$\mathcal{I}(\top)$ : Top
top :: Omega
top = 1.0 / 0.0

-- | \I(+) : Additive monoid
oplus :: Omega -> Omega -> Omega
oplus = (+)

-- | \I(*) : Multiplicative monoid
otimes :: Omega -> Omega -> Omega
otimes = (*)

-- | \$\mathcal{I}(\vec{0})$ : Additive unit
v0 :: Omega
v0 = 0.0

-- | \$\mathcal{I}(\vec{1})$ : Multiplicative unit
v1 :: Omega
v1 = 1.0

-- | \$\mathcal{I}(\neg)$ : Negation (additive inverse)
neg :: Omega -> Omega
neg x = -x

------------------------------------------------------
-- Quantifiers ($Q_a :: (a \to \Omega) \to \Omega$)
------------------------------------------------------

-- | \$\mathcal{I}(\bigvee)$ : Supremum
bigVee :: forall a. DATA a -> (a -> Omega) -> Omega
bigVee = sup

-- | \$\mathcal{I}(\bigwedge)$ : Infimum
bigWedge :: forall a. DATA a -> (a -> Omega) -> Omega
bigWedge = inf

-- | Probability measure on Strings (discrete uniform over first 1000 strings)
stringsDist :: Giry String
stringsDist = DisUniform (take 1000 strs)
  where
    strs = "" : [c : s | s <- strs, c <- ['a' .. 'z']]

-- | \$\mathcal{I}(\bigoplus)$ : Infinitary Sum = $\mathbb{E}_\mu[\varphi]$ (integral w.r.t.\ chosen measure)
--   The choice of $\mu \in \mathcal{G}(a)$ for each DATA object IS (the whole content of) the interpretation.
bigOplus :: forall a. DATA a -> (a -> Omega) -> Omega
bigOplus Reals phi = expect Reals (Normal 0.0 1.0) phi
bigOplus Naturals phi = expect Naturals (fmap fromIntegral (Poisson 1.0)) phi
bigOplus Integers phi = expect Integers (fmap fromIntegral (Poisson 1.0)) phi
bigOplus Strings phi = expect Strings stringsDist phi
bigOplus Booleans phi = expect Booleans (DisUniform [True, False]) phi
bigOplus Unit phi = expect Unit (Pure ()) phi
bigOplus (Finite xs) phi = expect (Finite xs) (DisUniform xs) phi
bigOplus (Prod da db) phi = bigOplus da (\a -> bigOplus db (\b -> phi (a, b)))
bigOplus (Lists _da) _phi = error "bigOplus over Lists not yet supported"

-- | \$\mathcal{I}(\bigotimes)$ : Infinitary Product = $\exp(\mathbb{E}_\mu[\log \circ \varphi])$  (product integral)
bigOtimes :: forall a. DATA a -> (a -> Omega) -> Omega
bigOtimes Reals phi = exp (expect Reals (Normal 0.0 1.0) (log . phi))
bigOtimes Naturals phi = exp (expect Naturals (fmap fromIntegral (Poisson 1.0)) (log . phi))
bigOtimes Integers phi = exp (expect Integers (fmap fromIntegral (Poisson 1.0)) (log . phi))
bigOtimes Strings phi = exp (expect Strings stringsDist (log . phi))
bigOtimes Booleans phi = exp (expect Booleans (DisUniform [True, False]) (log . phi))
bigOtimes Unit phi = exp (expect Unit (Pure ()) (log . phi))
bigOtimes (Finite xs) phi = exp (expect (Finite xs) (DisUniform xs) (log . phi))
bigOtimes (Prod da db) phi = bigOtimes da (\a -> bigOtimes db (\b -> phi (a, b)))
bigOtimes (Lists _da) _phi = error "bigOtimes over Lists not yet supported"

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
