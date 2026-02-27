-- | Interpretation 𝓘_Σ of TrafficLightSig in (DATA, Giry)
module NonLogical.Interpretations.TrafficLight (light, driveF) where

import NonLogical.Monads.Giry (Giry, categorical)
import NonLogical.Signatures.TrafficLightSig (LightColor)

-- | 𝓘(light) : mFun
light :: Giry LightColor
light = categorical [("Red", 0.6), ("Green", 0.3), ("Yellow", 0.1)]

-- | 𝓘(driveF) : mFun
driveF :: LightColor -> Giry Bool
driveF "Red" = categorical [(True, 0.1), (False, 0.9)]
driveF "Yellow" = categorical [(True, 0.2), (False, 0.8)]
driveF "Green" = categorical [(True, 0.9), (False, 0.1)]
