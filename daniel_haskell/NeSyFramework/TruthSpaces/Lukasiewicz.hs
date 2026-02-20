{-# LANGUAGE InstanceSigs #-}
module NeSyFramework.TruthSpaces.Lukasiewicz where

import NeSyFramework.TruthSpaces.TwoMonBLat

-- | Lukasiewicz Logic instance for 2Mon-BLat
instance TwoMonBLat Double where
    -- Entailment: x <= y
    vdash :: Double -> Double -> Bool
    vdash x y = x <= y
    
    -- Monoid oplus: min(1, x + y)
    oplus :: Double -> Double -> Double
    oplus x y = min 1.0 (x + y)
    
    -- Monoid otimes: max(0, x + y - 1)
    otimes :: Double -> Double -> Double
    otimes x y = max 0.0 (x + y - 1.0)
    
    -- Lattice symbols
    vee :: Double -> Double -> Double
    vee x y   = max x y
    wedge :: Double -> Double -> Double
    wedge x y = min x y
    
    -- Constants
    bot :: Double
    bot = 0.0
    top :: Double
    top = 1.0
    v0 :: Double
    v0  = 0.0
    v1 :: Double
    v1  = 1.0
