-- | Weather domain -- Signature + Interpretation
module A2_Interpretation.B4_NonLogical.Weather where

import A1_Syntax.B4_NonLogical.Weather_Vocab (Worlds, WorldsRow (..))
import A2_Interpretation.B2_Typological.Categories.DATA (tableLookup)
import A2_Interpretation.B1_Categorical.Monads.Giry (Giry (..))

------------------------------------------------------
-- I: Interpretation (Schema Instance + Function Definitions)
------------------------------------------------------

-- | Data instance (rows of the table)
worldsTable :: [WorldsRow]
worldsTable =
  [ WorldsRow "Berlin" 0.6 22.0 5.0,
    WorldsRow "Munich" 0.8 5.0 3.0,
    WorldsRow "Hamburg" 0.7 (-2.0) 1.5,
    WorldsRow "Bremen" 0.5 0.0 2.0
  ]

-- | Row lookup (the "WHERE id = w" query)
lookupRow :: Worlds -> WorldsRow
lookupRow w = tableLookup worldId w worldsTable

-- | I(data1) : Con
data1 :: Worlds
data1 = "Berlin"

-- | I(data2) : Con
data2 :: Worlds
data2 = "Hamburg"

-- | I(data3) : Con
data3 :: Worlds
data3 = "Bremen"

-- | I(humidDetect) : Fun
humidDetect :: Worlds -> Double
humidDetect = humidityPval . lookupRow

-- | I(tempPredict) : Fun
tempPredict :: Worlds -> (Double, Double)
tempPredict w = let r = lookupRow w in (tempMean r, tempStd r)

-- | I(bernoulli) : mFun
bernoulli :: Double -> Giry Int
bernoulli p = Categorical [(1 :: Int, p), (0 :: Int, 1.0 - p)]

-- | I(normalDist) : mFun
normalDist :: (Double, Double) -> Giry Double
normalDist (mu, sigma) = Normal mu sigma
