{-# LANGUAGE InstanceSigs #-}
module NeSyFramework.TruthSpaces.Product where

import NeSyFramework.TruthSpaces.TwoMonBLat

-- | Product Logic instance for 2Mon-BLat
instance TwoMonBLat Double where
    -- Entailment: x <= y
    vdash :: Double -> Double -> Bool
    vdash x y = x <= y
    
    -- Monoids: Probabilistic Sum and Product
    oplus :: Double -> Double -> Double
    oplus x y  = x + y - (x * y)
    otimes :: Double -> Double -> Double
    otimes x y = x * y
    
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
