-- | Interpretation 𝓘_Σ of CountableSig in (DATA, Giry)
module NonLogical.Interpretations.Countable where

import NonLogical.Monads.Giry (Giry, categorical)

-- | 𝓘(drawInt) : mFun — geometric distribution
drawInt :: Giry Int
drawInt =
  let p = 0.5
      probs = [(k, (1 - p) ^ k * p) | k <- [0 ..]]
   in categorical probs

-- | 𝓘(drawStr) : mFun — geometric over coin-flip strings
drawStr :: Giry String
drawStr =
  let p = 0.5
      toStr k = replicate k 'T' ++ "H"
      probs = [(toStr k, (1 - p) ^ k * p) | k <- [0 ..]]
   in categorical probs

-- | 𝓘(drawLazy) : mFun — zeta(3) distribution (light tail)
drawLazy :: Giry Int
drawLazy =
  let zeta3 = 1.202056903159594
      probs = [(k, (1 / fromIntegral (k + 1) ** 3) / zeta3) | k <- [0 ..]]
   in categorical probs

-- | 𝓘(drawHeavy) : mFun — zeta(1.1) distribution (heavy tail)
drawHeavy :: Giry Int
drawHeavy =
  let zeta11 = 10.5844484649508
      probs = [(k, (1 / fromIntegral (k + 1) ** 1.1) / zeta11) | k <- [0 ..]]
   in categorical probs
