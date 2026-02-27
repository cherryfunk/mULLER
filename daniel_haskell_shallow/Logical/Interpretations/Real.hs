-- | Logical interpretation: Real-valued Logic (𝓘_Υ)
module Logical.Interpretations.Real where

infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | Ω := 𝓘(τ) = ℝ (approximated by IEEE 754 Double)
type Omega = Double

-- | 𝓘(⊢) : Comparison
vdash :: Omega -> Omega -> Bool
vdash = (<=)

-- | 𝓘(∧) : Meet
wedge :: Omega -> Omega -> Omega
wedge = min

-- | 𝓘(∨) : Join
vee :: Omega -> Omega -> Omega
vee = max

-- | 𝓘(⊥) : Bottom
bot :: Omega
bot = -1.0 / 0.0

-- | 𝓘(⊤) : Top
top :: Omega
top = 1.0 / 0.0

-- | 𝓘(⊕) : Additive monoid
oplus :: Omega -> Omega -> Omega
oplus = (+)

-- | 𝓘(⊗) : Multiplicative monoid
otimes :: Omega -> Omega -> Omega
otimes = (*)

-- | 𝓘(0⃗) : Additive unit
v0 :: Omega
v0 = 0.0

-- | 𝓘(1⃗) : Multiplicative unit
v1 :: Omega
v1 = 1.0

-- | 𝓘(¬) : Negation (additive inverse)
neg :: Omega -> Omega
neg x = -x

--------------------------------------------------------------------------------
-- General predicates (implicit in every signature using this logic)
-- These are NOT part of the logical interpretation itself.
-- They lift Haskell's native comparisons to Omega-valued predicates.
--------------------------------------------------------------------------------

-- | Omega-valued equality
(.==) :: (Eq a) => a -> a -> Omega
x .== y = if x == y then top else bot

-- | Omega-valued less-than
(.<) :: (Ord a) => a -> a -> Omega
x .< y = if x < y then top else bot

-- | Omega-valued greater-than
(.>) :: (Ord a) => a -> a -> Omega
x .> y = if x > y then top else bot

-- | Omega-valued less-than-or-equal
(.<=) :: (Ord a) => a -> a -> Omega
x .<= y = if x <= y then top else bot

-- | Omega-valued greater-than-or-equal
(.>=) :: (Ord a) => a -> a -> Omega
x .>= y = if x >= y then top else bot

(./=) :: (Eq a) => a -> a -> Omega
x ./= y = if x /= y then top else bot

b2o :: Bool -> Omega
b2o b = if b then top else bot
