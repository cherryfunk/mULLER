-- | Crossing domain — Signature + Interpretation (Ola paper, Fig. 1)
--
-- "For every crossing, only continue driving if there is a green light."
--  ∀x ∈ X(l := 🚦(x), d := 🚗(x, l)(¬true(d) ∨ l = 🟢))
module NonLogical.Interpretations.Crossing where

import NonLogical.Monads.Dist (Dist (..))

--------------------------------------------------------------------------------
-- Σ: Non-Logical Vocabulary (sorts)
--------------------------------------------------------------------------------

-- | Sor
type LightColor = String -- "Red", "Orange", "Green"

type Decision = Int -- 0 = stop, 1 = go

-- | mFun: lightDetector   :: Dist LightColor
-- |       drivingDecision :: LightColor -> Dist Decision

--------------------------------------------------------------------------------
-- 𝓘: Interpretation and Syntctic Type Declarations
--------------------------------------------------------------------------------

-- | 𝓘(🚦) : mFun — light detector (conditional distribution over colors)
--   P(Red) = 0.6, P(Orange) = 0.1, P(Green) = 0.3
lightDetector :: Dist LightColor
lightDetector = Dist [("Red", 0.6), ("Orange", 0.1), ("Green", 0.3)]

-- | 𝓘(🚗) : mFun — driving decision (conditional on light color)
--   P(go | Red) = 0.1, P(go | Orange) = 0.2, P(go | Green) = 0.9
drivingDecision :: LightColor -> Dist Decision
drivingDecision "Red" = Dist [(0, 0.9), (1, 0.1)]
drivingDecision "Orange" = Dist [(0, 0.8), (1, 0.2)]
drivingDecision "Green" = Dist [(0, 0.1), (1, 0.9)]
drivingDecision _ = Dist [(0, 1.0)]
