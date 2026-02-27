-- | Interpretation 𝓘_Σ of the Crossing scenario (from the Ola paper, Fig. 1)
--
-- "For every crossing, only continue driving if there is a green light."
--  ∀x ∈ X(l := 🚦(x), d := 🚗(x, l)(¬true(d) ∨ l = 🟢))
--
--  🚦 : X → ({Red, Orange, Green} → [0,1])
--    — detects the light color from the crossing image
--  🚗 : (X × {Red, Orange, Green}) → ({0,1} → [0,1])
--    — decides whether to continue driving given the light
module NonLogical.Interpretations.Crossing where

import NonLogical.Monads.Dist (Dist (..))

-- | The type of light color concepts
type LightColor = String -- "Red", "Orange", "Green"

-- | The type of driving decisions (0 = stop, 1 = go)
type Decision = Int

-- | 𝓘(🚦) : mFun — light detector (conditional distribution over colors)
--   For crossing x_i, the detector outputs:
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
