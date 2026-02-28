-- | Countable sets domain -- Signature + Interpretation
module NonLogical.Interpretations.Countable where

import NonLogical.Monads.Giry (Giry (..))

------------------------------------------------------
-- Sigma: Non-Logical Vocabulary (sorts)
------------------------------------------------------

-- (Uses Haskell built-in sorts: Int, String)

-- | mFun: drawInt   :: Giry Int
-- |       drawStr   :: Giry String
-- |       drawLazy  :: Giry Int
-- |       drawHeavy :: Giry Int

------------------------------------------------------
-- I: Interpretation and Syntctic Type Declarations
------------------------------------------------------

-- | I(drawInt) : mFun -- geometric distribution
drawInt :: Giry Int
drawInt =
  let p = 0.5
      probs = [(k, (1 - p) ^ k * p) | k <- [0 ..]]
   in Categorical probs

-- | I(drawStr) : mFun -- geometric over coin-flip strings
drawStr :: Giry String
drawStr =
  let p = 0.5
      toStr k = replicate k 'T' ++ "H"
      probs = [(toStr k, (1 - p) ^ k * p) | k <- [0 ..]]
   in Categorical probs

-- | I(drawLazy) : mFun -- zeta(3) distribution (light tail)
drawLazy :: Giry Int
drawLazy =
  let zeta3 = 1.202056903159594
      probs = [(k, (1 / fromIntegral (k + 1) ** 3) / zeta3) | k <- [0 ..]]
   in Categorical probs

-- | I(drawHeavy) : mFun -- zeta(1.1) distribution (heavy tail)
drawHeavy :: Giry Int
drawHeavy =
  let zeta11 = 10.5844484649508
      probs = [(k, (1 / fromIntegral (k + 1) ** 1.1) / zeta11) | k <- [0 ..]]
   in Categorical probs
