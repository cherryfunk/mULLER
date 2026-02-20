{-# LANGUAGE InstanceSigs #-}
module NeSyFramework.TruthSpaces.Boolean where

import NeSyFramework.TruthSpaces.TwoMonBLat

-- | Classical Boolean Logic instance for 2Mon-BLat
instance TwoMonBLat Bool where
    vdash :: Bool -> Bool -> Bool
    vdash = (<=)
    oplus :: Bool -> Bool -> Bool
    oplus     = (||)
    otimes :: Bool -> Bool -> Bool
    otimes    = (&&)
    vee :: Bool -> Bool -> Bool
    vee       = (||)
    wedge :: Bool -> Bool -> Bool
    wedge     = (&&)
    bot :: Bool
    bot       = False
    top :: Bool
    top       = True
    v0 :: Bool
    v0        = False
    v1 :: Bool
    v1        = True
