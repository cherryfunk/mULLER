-- | Logical interpretation: Classical Boolean Logic (Ω = {True, False})
module Logical.Interpretations.Boolean where

infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | Ω := 𝓘(τ) = {True, False}
type Omega = Bool

-- | 𝓘(⊢) : Comparison (False ≤ True)
vdash :: Omega -> Omega -> Bool
vdash = (<=)

-- | 𝓘(∧) : Conjunction
wedge :: Omega -> Omega -> Omega
wedge = (&&)

-- | 𝓘(∨) : Disjunction
vee :: Omega -> Omega -> Omega
vee = (||)

-- | 𝓘(⊥) : Bottom
bot :: Omega
bot = False

-- | 𝓘(⊤) : Top
top :: Omega
top = True

-- | 𝓘(⊕) : Disjunction
oplus :: Omega -> Omega -> Omega
oplus = (||)

-- | 𝓘(⊗) : Conjunction
otimes :: Omega -> Omega -> Omega
otimes = (&&)

-- | 𝓘(0⃗) : Additive unit
v0 :: Omega
v0 = False

-- | 𝓘(1⃗) : Multiplicative unit
v1 :: Omega
v1 = True

-- | 𝓘(¬) : Negation
neg :: Omega -> Omega
neg = not

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

(.<=) :: (Ord a) => a -> a -> Omega
x .<= y = if x <= y then top else bot

(.>=) :: (Ord a) => a -> a -> Omega
x .>= y = if x >= y then top else bot

(./=) :: (Eq a) => a -> a -> Omega
x ./= y = if x /= y then top else bot

b2o :: Bool -> Omega
b2o b = if b then top else bot
