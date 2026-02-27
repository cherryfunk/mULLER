-- | Logical interpretation: Product Logic (Ω = [0,1] ⊂ ℝ)
module Logical.Interpretations.Product where

-- Fixity: comparisons (.==, .<, .>) bind tighter than connectives (wedge, vee)
infix 4 .==, ./=, .<, .>, .<=, .>=

infixr 3 `wedge`

infixr 2 `vee`

-- | Ω := 𝓘(τ) = [0,1]
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
bot = 0.0

-- | 𝓘(⊤) : Top
top :: Omega
top = 1.0

-- | 𝓘(⊕) : Probabilistic sum
oplus :: Omega -> Omega -> Omega
oplus x y = x + y - (x * y)

-- | 𝓘(⊗) : Probabilistic product
otimes :: Omega -> Omega -> Omega
otimes = (*)

-- | 𝓘(0⃗) : Additive unit
v0 :: Omega
v0 = 0.0

-- | 𝓘(1⃗) : Multiplicative unit
v1 :: Omega
v1 = 1.0

--------------------------------------------------------------------------------
-- General predicates (implicit in every signature using this logic)
-- These are NOT part of the logical interpretation itself.
-- They lift Haskell's native comparisons to Omega-valued predicates.
--------------------------------------------------------------------------------

-- | Omega-valued equality: a .== b returns top if a == b, bot otherwise
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

-- | Omega-valued not-equal
(./=) :: (Eq a) => a -> a -> Omega
x ./= y = if x /= y then top else bot

-- | General lifter: converts any Bool to Omega.
--   Use for predicates without a dot-operator, e.g.:
--   b2o (even x), b2o (elem x [1,2,3]), b2o (null xs)
b2o :: Bool -> Omega
b2o b = if b then top else bot