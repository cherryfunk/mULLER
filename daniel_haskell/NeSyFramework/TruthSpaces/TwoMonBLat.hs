module NeSyFramework.TruthSpaces.TwoMonBLat where

-- | Theory of a double monoid bounded lattice (2Mon-BLat), still without axioms.
class TwoMonBLat tau where
    -- Arities tau^2 -> tau
    vdash  :: tau -> tau -> Bool
    oplus  :: tau -> tau -> tau
    otimes :: tau -> tau -> tau
    vee    :: tau -> tau -> tau
    wedge  :: tau -> tau -> tau
    
    -- Arities 1 -> tau
    bot, top :: tau
    v0, v1   :: tau
