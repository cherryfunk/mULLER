-- | Weather domain -- Signature + Interpretation
module NonLogical.Interpretations.Weather where

import NonLogical.Categories.DATA (tableLookup)
import NonLogical.Monads.Giry (Giry (..))

------------------------------------------------------
-- Sigma: Non-Logical Vocabulary (sorts, data schema)
------------------------------------------------------

-- | Sor
type Worlds = String

-- | Data layout (analogous to a database schema)
data WorldsRow = WorldsRow
  { worldId :: Worlds,
    humidityPval :: Double,
    tempMean :: Double,
    tempStd :: Double
  }

-- | Con:  data1       :: Worlds
-- | Fun:  humidDetect :: Worlds -> Double
-- |       tempPredict :: Worlds -> (Double, Double)
-- | mFun: bernoulli   :: Double -> Giry Int
-- |       normalDist  :: (Double, Double) -> Giry Double

------------------------------------------------------
-- I: Interpretation and their Syntctic Type Declarations
------------------------------------------------------

-- | Data instance (rows of the table)
worldsTable :: [WorldsRow]
worldsTable =
  [ WorldsRow "Berlin" 0.6 22.0 5.0,
    WorldsRow "Munich" 0.8 5.0 3.0,
    WorldsRow "Hamburg" 0.7 (-2.0) 1.5
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
