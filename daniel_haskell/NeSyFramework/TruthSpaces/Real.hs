{-# LANGUAGE InstanceSigs #-}
module NeSyFramework.TruthSpaces.Real where

import NeSyFramework.TruthSpaces.TwoMonBLat

-- | Real Numbers Truth Space
-- Domain: R U {-inf, +inf} (implemented as Double)
-- Typical operations of real numbers.
instance TwoMonBLat Double where
    -- Comparison: x <= y
    vdash :: Double -> Double -> Bool
    vdash x y = x <= y

    -- Monoid 1 (Additive): +
    -- Neutral element would be 0, absorbing element would be top/bot depending on direction
    oplus :: Double -> Double -> Double
    oplus = (+)

    -- Monoid 2 (Multiplicative): *
    -- Neutral element is 1
    otimes :: Double -> Double -> Double
    otimes = (*)

    -- Lattice
    vee :: Double -> Double -> Double
    vee = max

    wedge :: Double -> Double -> Double
    wedge = min

    -- Constants
    bot :: Double
    bot = -1.0 / 0.0 -- Negative Infinity

    top :: Double
    top = 1.0 / 0.0  -- Positive Infinity

    v0 :: Double
    v0 = 0.0

    v1 :: Double
    v1 = 1.0
